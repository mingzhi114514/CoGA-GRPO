# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import textwrap
from collections import defaultdict, deque
from typing import Any, Callable, Optional, Union, Sized

import torch
import torch.utils.data
import transformers
import torch.nn as nn
from torch.utils.data import DataLoader, Sampler
import datasets
from datasets import Dataset, IterableDataset
from packaging import version
from transformers import (
    AriaForConditionalGeneration,
    AriaProcessor,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoProcessor,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    Trainer,
    TrainerCallback,
    is_wandb_available,
)
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.trainer import SCHEDULER_NAME
from transformers.utils import is_peft_available, is_rich_available,  is_datasets_available
from transformers.trainer_utils import has_length, seed_worker
from trl.data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
from trl.models import create_reference_model, prepare_deepspeed, unwrap_model_for_generation
from trl.trainer.utils import generate_model_card, get_comet_experiment_url, print_prompt_completions_sample

from .grpo_config import GRPOConfig
from accelerate.utils import (
    DeepSpeedSchedulerWrapper,
    broadcast_object_list,
    gather,
    gather_object,
    is_peft_model,
    set_seed,
)
import PIL.Image

import copy
from torch.utils.data import Sampler
import warnings
import torch.distributed as dist


if is_peft_available():
    from peft import PeftConfig, get_peft_model

if is_wandb_available():
    import wandb

from open_r1.vlm_modules.vlm_module import VLMBaseModule

logger = logging.getLogger(__name__)


def _append_full_completions_log(
    output_dir: str,
    step: int,
    prompts: list[str],
    completions: list[str],
    rewards: list,
    gen_seq: Optional[int] = None,
    per_sample_meta: Optional[list[str]] = None,
) -> Optional[str]:
    """Append full prompt/completion to a log file (Rich table is display-only). Call only on main process.

    Set ``GRPO_SKIP_FULL_COMPLETION_LOG=1`` to disable. Override path with ``GRPO_COMPLETIONS_LOG_PATH``.
    """
    if os.environ.get("GRPO_SKIP_FULL_COMPLETION_LOG", "").lower() in ("1", "true", "yes"):
        return None
    path = os.environ.get("GRPO_COMPLETIONS_LOG_PATH")
    if not path:
        path = os.path.join(output_dir, "completions_full.log")
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    header = f"step={step}"
    if gen_seq is not None:
        header += f" gen_seq={gen_seq}"
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"\n{'=' * 80}\n{header}\n")
        for i, (p, c, r) in enumerate(zip(prompts, completions, rewards)):
            f.write(f"\n--- sample {i} reward={r} ---\n")
            if per_sample_meta is not None and i < len(per_sample_meta):
                f.write(f"[decode_meta] {per_sample_meta[i]}\n")
            f.write(f"[PROMPT]\n{p}\n\n[COMPLETION]\n{c}\n")
            f.write(f"[len_chars prompt={len(p)} completion={len(c)}]\n")
    return path


# What we call a reward function is a callable that takes a list of prompts and completions and returns a list of
# rewards. When it's a string, it's a model ID, so it's loaded as a pretrained model.
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]


def find_mask_between_patterns_1d(input_tensor: torch.Tensor,
                                  start_pattern_list: list,
                                  end_pattern_list: list) -> torch.Tensor:
    """
    (Helper function - same as before)
    Finds the mask for a single 1D tensor.
    """
    assert input_tensor.ndim == 1, "Input tensor must be 1-dimensional"

    device = input_tensor.device
    dtype = input_tensor.dtype # Use input tensor's dtype

    # Ensure patterns are tensors on the correct device and dtype
    start_pattern = torch.tensor(start_pattern_list, dtype=dtype, device=device)
    end_pattern = torch.tensor(end_pattern_list, dtype=dtype, device=device)

    n = input_tensor.shape[0]
    len_start = len(start_pattern)
    len_end = len(end_pattern)

    start_idx = -1
    end_idx = -1

    # --- Find start_pattern index ---
    if n >= len_start:
        start_windows = input_tensor.unfold(0, len_start, 1)
        start_matches = (start_windows == start_pattern).all(dim=1)
        start_indices = start_matches.nonzero(as_tuple=True)[0]
        if start_indices.numel() > 0:
            start_idx = start_indices[0].item() # Assume first match
        else:
            # Indicate pattern not found for this row
            return torch.zeros_like(input_tensor, dtype=torch.long, device=device)
            # raise ValueError("Start pattern not found in the tensor.") # Original behavior
    else:
        return torch.zeros_like(input_tensor, dtype=torch.long, device=device) # Too short

    # --- Find end_pattern index ---
    if n >= len_end:
        # Search *after* the start pattern to ensure correct order if multiple end patterns exist
        # Although problem states "only one region", this adds robustness
        search_area_end = input_tensor[start_idx + len_start:]
        if search_area_end.numel() >= len_end:
            end_windows = search_area_end.unfold(0, len_end, 1)
            end_matches = (end_windows == end_pattern).all(dim=1)
            end_indices = end_matches.nonzero(as_tuple=True)[0]
            if end_indices.numel() > 0:
                 # Index relative to the start of search_area_end, need to add offset
                relative_end_idx = end_indices[0].item()
                end_idx = start_idx + len_start + relative_end_idx
            else:
                 # End pattern not found *after* start pattern
                return torch.zeros_like(input_tensor, dtype=torch.long, device=device)
        else:
            # Not enough elements after start pattern to contain end pattern
             return torch.zeros_like(input_tensor, dtype=torch.long, device=device)
    else:
       return torch.zeros_like(input_tensor, dtype=torch.long, device=device) # Too short

    # --- Calculate mask region ---
    mask_start = start_idx + len_start
    mask_end = end_idx # end_idx is the *start* of the end pattern

    # mask_start = start_idx
    # mask_end = end_idx + len_end

    # --- Create and fill mask ---
    mask = torch.zeros_like(input_tensor, dtype=torch.long, device=device)

    # if mask_start < mask_end:
    #     mask[mask_start:-1] = 1
    if mask_start < mask_end:
        mask[:mask_end] = 1

    # if mask_start < mask_end:
    #     mask[mask_start:mask_end] = 1
    # else: patterns adjacent or end before start, mask remains zero, no warning needed here

    return mask



def generate_2d_mask(input_tensor_2d: torch.Tensor,
                       start_pattern_list: list,
                       end_pattern_list: list) -> torch.Tensor:
    """
    Generates a 2D mask by applying the 1D pattern finding logic to each row.

    Args:
        input_tensor_2d: The input 2D PyTorch Tensor (Batch x SequenceLength).
        start_pattern_list: The start pattern list.
        end_pattern_list: The end pattern list.

    Returns:
        A 2D mask tensor of the same shape as input_tensor_2d (dtype=torch.long),
        where each row's mask is generated based on the patterns found in that row.
        Rows where patterns are not found (or order is wrong) will have a mask of all zeros.
    """
    assert input_tensor_2d.ndim == 2, "Input tensor must be 2-dimensional"

    num_rows = input_tensor_2d.shape[0]
    if num_rows == 0:
        return torch.empty_like(input_tensor_2d, dtype=torch.long) # Handle empty input

    row_masks = []
    for i in range(num_rows):
        current_row = input_tensor_2d[i]
        # Call the 1D function for the current row
        # Modify 1D function to return zeros instead of raising error if pattern not found
        mask_1d = find_mask_between_patterns_1d(current_row, start_pattern_list, end_pattern_list)
        row_masks.append(mask_1d)

    # Stack the generated 1D masks along the batch dimension (dim=0)
    mask_2d = torch.stack(row_masks, dim=0)

    return mask_2d

class RepeatRandomSampler(Sampler):
    """
    Sampler that repeats the indices of a dataset in a structured manner.

    Args:
        data_source (`Sized`):
            Dataset to sample from.
        mini_repeat_count (`int`):
            Number of times to repeat each index per batch.
        batch_size (`int`, *optional*, defaults to `1`):
            Number of unique indices per batch.
        repeat_count (`int`, *optional*, defaults to `1`):
            Number of times to repeat the full sampling process.
        seed (`int` or `None`, *optional*, defaults to `None`):
            Random seed for reproducibility.
    """

    def __init__(
        self,
        data_source: Sized,
        mini_repeat_count: int,
        batch_size: int = 1,
        repeat_count: int = 1,
        seed: Optional[int] = None,
    ):
        self.data_source = data_source
        self.mini_repeat_count = mini_repeat_count
        self.batch_size = batch_size
        self.repeat_count = repeat_count
        self.num_samples = len(data_source)
        self.seed = seed
        self.generator = torch.Generator()
        if seed is not None:
            self.generator.manual_seed(seed)

    def __iter__(self):
        indexes = torch.randperm(self.num_samples, generator=self.generator).tolist()
        indexes = [indexes[i : i + self.batch_size] for i in range(0, len(indexes), self.batch_size)]
        indexes = [chunk for chunk in indexes if len(chunk) == self.batch_size]

        for chunk in indexes:
            for _ in range(self.repeat_count):
                for index in chunk:
                    for _ in range(self.mini_repeat_count):
                        yield index

    def __len__(self) -> int:
        return self.num_samples * self.mini_repeat_count * self.repeat_count


class VLMGRPOTrainer(Trainer):
    """
    Trainer for the Group Relative Policy Optimization (GRPO) method. This algorithm was initially proposed in the
    paper [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://huggingface.co/papers/2402.03300).

    Example:

    ```python
    from datasets import load_dataset
    from trl import GRPOTrainer

    dataset = load_dataset("trl-lib/tldr", split="train")

    trainer = GRPOTrainer(
        model="Qwen/Qwen2-0.5B-Instruct",
        reward_funcs="weqweasdas/RM-Gemma-2B",
        train_dataset=dataset,
    )

    trainer.train()
    ```

    Args:
        model (`Union[str, PreTrainedModel]`):
            Model to be trained. Can be either:

            - A string, being the *model id* of a pretrained model hosted inside a model repo on huggingface.co, or
              a path to a *directory* containing model weights saved using
              [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is
              loaded using [`~transformers.AutoModelForCausalLM.from_pretrained`] with the keywork arguments
              in `args.model_init_kwargs`.
            - A [`~transformers.PreTrainedModel`] object. Only causal language models are supported.
        reward_funcs (`Union[RewardFunc, list[RewardFunc]]`):
            Reward functions to be used for computing the rewards. To compute the rewards, we call all the reward
            functions with the prompts and completions and sum the rewards. Can be either:

            - A single reward function, such as:
                - A string: The *model ID* of a pretrained model hosted inside a model repo on huggingface.co, or a
                path to a *directory* containing model weights saved using
                [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is loaded
                using [`~transformers.AutoModelForSequenceClassification.from_pretrained`] with `num_labels=1` and the
                keyword arguments in `args.model_init_kwargs`.
                - A [`~transformers.PreTrainedModel`] object: Only sequence classification models are supported.
                - A custom reward function: The function is provided with the prompts and the generated completions,
                  plus any additional columns in the dataset. It should return a list of rewards. For more details, see
                  [Using a custom reward function](#using-a-custom-reward-function).
            - A list of reward functions, where each item can independently be any of the above types. Mixing different
            types within the list (e.g., a string model ID and a custom reward function) is allowed.
        args ([`GRPOConfig`], *optional*, defaults to `None`):
            Configuration for this trainer. If `None`, a default configuration is used.
        train_dataset ([`~datasets.Dataset`] or [`~datasets.IterableDataset`]):
            Dataset to use for training. It must include a column `"prompt"`. Any additional columns in the dataset is
            ignored. The format of the samples can be either:

            - [Standard](dataset_formats#standard): Each sample contains plain text.
            - [Conversational](dataset_formats#conversational): Each sample contains structured messages (e.g., role
              and content).
        eval_dataset ([`~datasets.Dataset`], [`~datasets.IterableDataset`] or `dict[str, Union[Dataset, IterableDataset]]`):
            Dataset to use for evaluation. It must meet the same requirements as `train_dataset`.
        processing_class ([`~transformers.PreTrainedTokenizerBase`], *optional*, defaults to `None`):
            Processing class used to process the data. The padding side must be set to "left". If `None`, the
            processing class is loaded from the model's name with [`~transformers.AutoTokenizer.from_pretrained`].
        reward_processing_classes (`Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]`, *optional*, defaults to `None`):
            Processing classes corresponding to the reward functions specified in `reward_funcs`. Can be either:

            - A single processing class: Used when `reward_funcs` contains only one reward function.
            - A list of processing classes: Must match the order and length of the reward functions in `reward_funcs`.
            If set to `None`, or if an element of the list corresponding to a [`~transformers.PreTrainedModel`] is
            `None`, the tokenizer for the model is automatically loaded using [`~transformers.AutoTokenizer.from_pretrained`].
            For elements in `reward_funcs` that are custom reward functions (not [`~transformers.PreTrainedModel`]),
            the corresponding entries in `reward_processing_classes` are ignored.
        callbacks (list of [`~transformers.TrainerCallback`], *optional*, defaults to `None`):
            List of callbacks to customize the training loop. Will add those to the list of default callbacks
            detailed in [here](https://huggingface.co/docs/transformers/main_classes/callback).

            If you want to remove one of the default callbacks used, use the [`~transformers.Trainer.remove_callback`]
            method.
        optimizers (`tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`, *optional*, defaults to `(None, None)`):
            A tuple containing the optimizer and the scheduler to use. Will default to an instance of [`AdamW`] on your
            model and a scheduler given by [`get_linear_schedule_with_warmup`] controlled by `args`.
        peft_config ([`~peft.PeftConfig`], *optional*, defaults to `None`):
            PEFT configuration used to wrap the model. If `None`, the model is not wrapped.
    """

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: GRPOConfig = None,
        vlm_module: VLMBaseModule = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        peft_config: Optional["PeftConfig"] = None,
        adapter_name_or_path: Optional[str] = None,
        freeze_vision_modules: Optional[bool] = True,
        attn_implementation: str = "flash_attention_2",
        torch_dtype: str = "bfloat16",
        use_audio_in_video: bool = False,
        **kwargs,
    ):
        # Args
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = GRPOConfig(f"{model_name}-GRPO")

        self.vlm_module = vlm_module
        self.use_audio_in_video = use_audio_in_video
        self.adapter_name_or_path = adapter_name_or_path

        # Models
        # Trained model
        model_init_kwargs = args.model_init_kwargs or {}
        # FIXME
        # Remember to modify it in the invernvl
        model_init_kwargs["attn_implementation"] = attn_implementation
        if model_init_kwargs.get("torch_dtype") is None:
            model_init_kwargs["torch_dtype"] = torch_dtype

        assert isinstance(model, str), "model must be a string in the current implementation"
        model_id = model
        torch_dtype = model_init_kwargs.get("torch_dtype")
        if isinstance(torch_dtype, torch.dtype) or torch_dtype == "auto" or torch_dtype is None:
            pass  # torch_dtype is already a torch.dtype or "auto" or None
        elif isinstance(torch_dtype, str):  # it's a str, but not "auto"
            torch_dtype = getattr(torch, torch_dtype)
        else:
            raise ValueError(
                "Invalid `torch_dtype` passed to `GRPOConfig`. Expected either 'auto' or a string representing "
                f"a `torch.dtype` (e.g., 'float32'), but got {torch_dtype}."
            )
        # model_init_kwargs["enable_audio_output"] = False
        # model_init_kwargs["use_cache"] = (
        #     False if args.gradient_checkpointing else model_init_kwargs.get("use_cache")
        # )
        #     # Disable caching if gradient checkpointing is enabled (not supported)
        # model_init_kwargs["use_cache"] = (
        #     False if args.gradient_checkpointing else model_init_kwargs.get("use_cache")
        # )
        model_cls = self.vlm_module.get_model_class(model_id, model_init_kwargs)
        model = model_cls.from_pretrained(model_id, **model_init_kwargs)
        # model = model.thinker # for qwen-omni

        # LoRA: resume from disk (SFT adapter) or fresh LoRA via PEFT config
        self.vision_modules_keywords = self.vlm_module.get_vision_modules_keywords()
        if self.adapter_name_or_path:
            if not is_peft_available():
                raise ValueError("PEFT is required to load adapter_name_or_path.")
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, self.adapter_name_or_path, is_trainable=True)
        elif peft_config is not None:
            def find_all_linear_names(model, multimodal_keywords):
                cls = torch.nn.Linear
                lora_module_names = set()
                for name, module in model.named_modules():
                    # LoRA is not applied to the vision modules
                    if any(mm_keyword in name for mm_keyword in multimodal_keywords):
                        continue
                    if isinstance(module, cls):
                        lora_module_names.add(name)
                for m in lora_module_names:  # needed for 16-bit
                    if "embed_tokens" in m:
                        lora_module_names.remove(m)
                return list(lora_module_names)
            target_modules = find_all_linear_names(model, self.vision_modules_keywords)
            peft_config.target_modules = target_modules
            model = get_peft_model(model, peft_config)

        # Freeze vision modules
        if freeze_vision_modules:
            print("Freezing vision modules...")
            for n, p in model.named_parameters():
                if any(keyword in n for keyword in self.vision_modules_keywords):
                    p.requires_grad = False

        # Enable gradient checkpointing if requested
        if args.gradient_checkpointing:
            model = self._enable_gradient_checkpointing(model, args)

        # Reference model (KL / beta > 0). Never load a second full base when using PEFT: use disable_adapter()
        # for ref logps (see _generate_and_score_completions). The old branch used is_deepspeed_available(), which is
        # True whenever the DeepSpeed package is installed — not "this run uses ZeRO-3" — and forced a duplicate
        # from_pretrained, causing OOM with ZeRO-2 + 7B multimodal + batch 2.
        beta = float(getattr(args, "beta", 0.04))
        using_peft = bool(self.adapter_name_or_path) or (peft_config is not None)
        if beta == 0.0:
            self.ref_model = None
        elif using_peft:
            self.ref_model = None
        elif is_deepspeed_zero3_enabled():
            self.ref_model = model_cls.from_pretrained(model_id, **model_init_kwargs)
        else:
            self.ref_model = create_reference_model(model)

        # Processing class
        if processing_class is None:
            processing_cls = self.vlm_module.get_processing_class()
            processing_class = processing_cls.from_pretrained(model_id, trust_remote_code=model_init_kwargs.get("trust_remote_code", None))
            for processing_keyword in self.vlm_module.get_custom_processing_keywords():
                if processing_keyword in kwargs:
                    setattr(processing_class, processing_keyword, kwargs[processing_keyword])
            if getattr(processing_class, "tokenizer",  None) is not None:
                pad_token_id = processing_class.tokenizer.pad_token_id
                processing_class.pad_token_id = pad_token_id
                processing_class.eos_token_id = processing_class.tokenizer.eos_token_id
            else:
                assert isinstance(processing_class, PreTrainedTokenizerBase), "processing_class must be an instance of PreTrainedTokenizerBase if it has no tokenizer attribute"
                pad_token_id = processing_class.pad_token_id
        # print(processing_class.tokenizer)
        self.vlm_module.post_model_init(model, processing_class)
        self.vlm_module.post_model_init(self.ref_model, processing_class)


        total_trainable_params = 0
        for name, p in model.named_parameters():
            if p.requires_grad:
                # print(f'train param: {name}')
                total_trainable_params += p.numel()


        # Reward functions
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        for i, reward_func in enumerate(reward_funcs):
            if isinstance(reward_func, str):
                reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(
                    reward_func, num_labels=1, **model_init_kwargs
                )
        self.reward_funcs = reward_funcs

        # Reward weights
        if args.reward_weights is not None:
            if len(args.reward_weights) != len(reward_funcs):
                raise ValueError(
                    f"Number of reward weights ({len(args.reward_weights)}) must match number of reward "
                    f"functions ({len(reward_funcs)})"
                )
            self.reward_weights = torch.tensor(args.reward_weights, dtype=torch.float32)
        else:
            self.reward_weights = torch.ones(len(reward_funcs), dtype=torch.float32)


        # Reward processing class
        if reward_processing_classes is None:
            reward_processing_classes = [None] * len(reward_funcs)
        elif not isinstance(reward_processing_classes, list):
            reward_processing_classes = [reward_processing_classes]
        else:
            if len(reward_processing_classes) != len(reward_funcs):
                raise ValueError("The number of reward processing classes must match the number of reward functions.")

        for i, (reward_processing_class, reward_func) in enumerate(zip(reward_processing_classes, reward_funcs)):
            if isinstance(reward_func, PreTrainedModel):
                if reward_processing_class is None:
                    reward_processing_class = AutoTokenizer.from_pretrained(reward_func.config._name_or_path)
                if reward_processing_class.pad_token_id is None:
                    reward_processing_class.pad_token = reward_processing_class.eos_token
                # The reward model computes the reward for the latest non-padded token in the input sequence.
                # So it's important to set the pad token ID to the padding token ID of the processing class.
                reward_func.config.pad_token_id = reward_processing_class.pad_token_id
                reward_processing_classes[i] = reward_processing_class
        self.reward_processing_classes = reward_processing_classes

        # Data collator
        def data_collator(features):  # No data collation is needed in GRPO
            return features

        # Training arguments
        self.max_prompt_length = args.max_prompt_length
        self.max_prompt_length = None  # TODO
        self.num_generations = args.num_generations  # = G in the GRPO paper
        self.temperature = args.temperature
        self.top_p = args.top_p
        self.top_k = args.top_k
        self.min_p = args.min_p
        self.repetition_penalty = args.repetition_penalty
        self.markov_reward = args.markov_reward



        if args.max_prompt_length is not None:
            warnings.warn("Setting max_prompt_length is currently not supported, it has been set to None")

        self.max_completion_length = args.max_completion_length  # = |o_i| in the GRPO paper
        self.num_generations = args.num_generations  # = G in the GRPO paper
        self.generation_config = GenerationConfig(
            max_new_tokens=self.max_completion_length,
            do_sample=True,
            pad_token_id=processing_class.tokenizer.pad_token_id,
            bos_token_id=processing_class.tokenizer.bos_token_id,
            eos_token_id=processing_class.tokenizer.eos_token_id,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            min_p=self.min_p,
            repetition_penalty=self.repetition_penalty,
            cache_implementation=args.cache_implementation,
        )
        if hasattr(self.vlm_module, "get_eos_token_id"):  # For InternVL
            self.generation_config.eos_token_id = self.vlm_module.get_eos_token_id(processing_class)

        # Qwen2.5-Omni uses image/video/audio placeholder tokens only for encoded
        # prompt inputs.  If sampling emits one of them in a completion, the next
        # full-sequence forward treats it as real multimodal input.  There is no
        # corresponding feature tensor for a generated placeholder (for example,
        # audio_seqlens is None), so get_rope_index crashes.  Suppress these tokens
        # during generation and retain their ids for a defensive post-generation
        # sanitization below.
        model_config = getattr(model, "config", None)
        input_only_multimodal_token_ids = []
        # Qwen checkpoints use ``*_token_index`` in config.json, while some
        # Transformers model configs expose the same values as ``*_token_id``.
        # Support both spellings instead of assuming one config representation.
        for modality in ("image", "video", "audio"):
            for suffix in ("token_id", "token_index"):
                token_id = getattr(model_config, f"{modality}_{suffix}", None)
                if isinstance(token_id, int) and token_id >= 0:
                    input_only_multimodal_token_ids.append(token_id)

        # The tokenizer is the authoritative fallback. Qwen2.5-Omni uses
        # uppercase placeholders (<|IMAGE|>, <|VIDEO|>, <|AUDIO|>), whereas other
        # Qwen variants use *_pad spellings.  convert_tokens_to_ids returns unk for
        # unknown strings, so only retain ids whose round trip matches exactly.
        tokenizer = getattr(processing_class, "tokenizer", processing_class)
        candidate_tokens = (
            "<|IMAGE|>", "<|VIDEO|>", "<|AUDIO|>",
            "<|image_pad|>", "<|video_pad|>", "<|audio_pad|>",
        )
        for token in candidate_tokens:
            token_id = tokenizer.convert_tokens_to_ids(token)
            if (
                isinstance(token_id, int)
                and token_id >= 0
                and tokenizer.convert_ids_to_tokens(token_id) == token
            ):
                input_only_multimodal_token_ids.append(token_id)
        self._input_only_multimodal_token_ids = sorted(set(input_only_multimodal_token_ids))
        if self._input_only_multimodal_token_ids:
            existing_suppressed = list(self.generation_config.suppress_tokens or [])
            self.generation_config.suppress_tokens = sorted(
                set(existing_suppressed + self._input_only_multimodal_token_ids)
            )
            logger.warning(
                "[GRPO] Suppressing input-only multimodal token ids during generation: %s",
                self._input_only_multimodal_token_ids,
            )
        else:
            logger.warning(
                "[GRPO] No input-only multimodal token ids were discovered; multimodal placeholder generation "
                "cannot be suppressed for this model."
            )
        self.beta = args.beta
        self.epsilon_low = args.epsilon
        self.epsilon_high = args.epsilon_high if args.epsilon_high is not None else args.epsilon


        # Multi-step
        self.num_iterations = args.num_iterations  # = 𝜇 in the GRPO paper
        # Tracks the number of iterations (forward + backward passes), including those within a gradient accumulation cycle
        self._step = 0
        # Buffer the batch to reuse generated outputs across multiple updates
        self._buffered_inputs = [None] * args.gradient_accumulation_steps

        # The trainer estimates the number of FLOPs (floating-point operations) using the number of elements in the
        # input tensor associated with the key "input_ids". However, in GRPO, the sampled data does not include the
        # "input_ids" key. Instead, the available keys is "prompt". As a result, the trainer issues the warning:
        # "Could not estimate the number of tokens of the input, floating-point operations will not be computed." To
        # suppress this warning, we set the "estimate_tokens" key in the model's "warnings_issued" dictionary to True.
        # This acts as a flag to indicate that the warning has already been issued.
        model.warnings_issued["estimate_tokens"] = True

        # Initialize the metrics
        self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
        self._total_train_tokens = 0
        self.log_completions = args.log_completions
        self._passk_ema = {}
        self._replay_ids = deque(maxlen=max(1, int(getattr(args, "replay_capacity", 4096))))
        w = int(getattr(args, "difficulty_window_size", 32) or 0)
        self._difficulty_window = deque(maxlen=max(0, w))
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )

        # Check if the per_device_train/eval_batch_size * num processes can be divided by the number of generations
        num_processes = self.accelerator.num_processes
        global_batch_size = args.per_device_train_batch_size * num_processes
        possible_values = [n_gen for n_gen in range(2, global_batch_size + 1) if (global_batch_size) % n_gen == 0]
        # if self.num_generations not in possible_values:
        #     raise ValueError(
        #         f"The global train batch size ({num_processes} x {args.per_device_train_batch_size}) must be evenly "
        #         f"divisible by the number of generations per prompt ({self.num_generations}). Given the current train "
        #         f"batch size, the valid values for the number of generations are: {possible_values}."
        #     )
        if self.args.eval_strategy != "no":
            global_batch_size = args.per_device_eval_batch_size * num_processes
            possible_values = [n_gen for n_gen in range(2, global_batch_size + 1) if (global_batch_size) % n_gen == 0]
            if self.num_generations not in possible_values:
                raise ValueError(
                    f"The global eval batch size ({num_processes} x {args.per_device_eval_batch_size}) must be evenly "
                    f"divisible by the number of generations per prompt ({self.num_generations}). Given the current "
                    f"eval batch size, the valid values for the number of generations are: {possible_values}."
                )

        # Ensure each process receives a unique seed to prevent duplicate completions when generating with
        # transformers if num_generations exceeds per_device_train_batch_size. We could skip it if we use vLLM, but
        # it's safer to set it in all cases.
        set_seed(args.seed, device_specific=True)

        # Gradient accumulation requires scaled loss. Normally, loss scaling in the parent class depends on whether the
        # model accepts loss-related kwargs. Since we compute our own loss, this check is irrelevant. We set
        # self.model_accepts_loss_kwargs to False to enable scaling.
        self.model_accepts_loss_kwargs = False

        if self.ref_model is not None:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                self.reward_funcs[i] = self.accelerator.prepare_model(reward_func, evaluation_mode=True)

    def _enable_gradient_checkpointing(self, model: PreTrainedModel, args: GRPOConfig) -> PreTrainedModel:
        """Enables gradient checkpointing for the model."""
        # Ensure use_cache is disabled
        model.config.use_cache = False

        # Enable gradient checkpointing on the base model for PEFT
        if is_peft_model(model):
            model.base_model.gradient_checkpointing_enable()
        # Enable gradient checkpointing for non-PEFT models
        else:
            try:
                model.gradient_checkpointing_enable()
            except:
                # For InternVL; these operations are copied from the original training script of InternVL
                model.language_model.config.use_cache = False
                model.vision_model.gradient_checkpointing = True
                model.vision_model.encoder.gradient_checkpointing = True
                model.language_model._set_gradient_checkpointing()
                # This line is necessary, otherwise the `model.gradient_checkpointing_enable()` will be executed during the training process, leading to an error since InternVL does not support this operation.
                args.gradient_checkpointing = False

        gradient_checkpointing_kwargs = args.gradient_checkpointing_kwargs or {}
        use_reentrant = (
            "use_reentrant" not in gradient_checkpointing_kwargs or gradient_checkpointing_kwargs["use_reentrant"]
        )

        if use_reentrant:
            model.enable_input_require_grads()

        return model

    def _set_signature_columns_if_needed(self):
        # If `self.args.remove_unused_columns` is True, non-signature columns are removed.
        # By default, this method sets `self._signature_columns` to the model's expected inputs.
        # In GRPOTrainer, we preprocess data, so using the model's signature columns doesn't work.
        # Instead, we set them to the columns expected by the `training_step` method, hence the override.
        if self._signature_columns is None:
            self._signature_columns = ["prompt"]

    def _load_optimizer_and_scheduler(self, checkpoint):
        # ZeRO checkpoints store optimizer under global_step*; HF may omit scheduler.pt while still
        # taking this branch (non-DeepSpeedSchedulerWrapper). Base Trainer then raises FileNotFoundError.
        if checkpoint is None:
            return
        if self.is_deepspeed_enabled and not isinstance(
            self.lr_scheduler, DeepSpeedSchedulerWrapper
        ):
            sched_path = os.path.join(checkpoint, SCHEDULER_NAME)
            if not os.path.isfile(sched_path):
                logger.warning(
                    "Checkpoint %s has no %s; skipping LR scheduler restore (optimizer loaded via DeepSpeed). "
                    "Global step still resumes from trainer_state.json.",
                    checkpoint,
                    SCHEDULER_NAME,
                )
                return
        super()._load_optimizer_and_scheduler(checkpoint)

    # Get the per-token log probabilities for the completions for the model and the reference model
    def _get_per_token_logps(self, model, input_ids, attention_mask, **custom_multimodal_inputs):
        logits = model(input_ids=input_ids, attention_mask=attention_mask, **custom_multimodal_inputs).logits  # (B, L, V)
        logits = logits[:, :-1, :]  # (B, L-1, V)
        shifted_ids = input_ids[:, 1:]  # (B, L-1)
        log_probs = torch.log_softmax(logits, dim=-1)
        per_token_logps = torch.gather(log_probs, dim=2, index=shifted_ids.unsqueeze(-1)).squeeze(-1)
        return per_token_logps

    def _decode_completions_masked(self, completion_ids: torch.Tensor, completion_mask: torch.Tensor) -> tuple[list[str], list[str]]:
        """Decode only tokens kept by ``completion_mask`` (avoids ``batch_decode`` quirks on padded rows)."""
        tok = getattr(self.processing_class, "tokenizer", None)
        if tok is None:
            tok = self.processing_class
        texts: list[str] = []
        meta: list[str] = []
        for i in range(completion_ids.size(0)):
            sel = completion_mask[i].bool()
            ids = completion_ids[i][sel]
            nt = int(ids.numel())
            if nt == 0:
                texts.append("")
                meta.append("n_tokens=0")
                continue
            head = ids[: min(12, nt)].tolist()
            text = tok.decode(ids.cpu(), skip_special_tokens=True, clean_up_tokenization_spaces=False)
            texts.append(text)
            meta.append(f"n_tokens={nt} first_token_ids={head!s}")
        return texts, meta

    def _get_assistant_prefill(self) -> str:
        """Optional text appended to prompt before tokenization; also prepended to decoded completion for rewards."""
        s = getattr(self.args, "assistant_prefill", None)
        if s is None or (isinstance(s, str) and not str(s).strip()):
            s = os.environ.get("GRPO_ASSISTANT_PREFILL", "")
        if s is None:
            return ""
        s = str(s).strip()
        if not s:
            return ""
        return s.replace("\\n", "\n").replace("\\t", "\t")

    def _prepare_inputs(self, inputs):
        mode = "eval" if self.control.should_evaluate else "train"
        if mode == "train":
            if self.state.global_step % self.num_iterations == 0:
                inputs = self._generate_and_score_completions(inputs)
                self._buffered_inputs[self._step % self.args.gradient_accumulation_steps] = inputs
            else:
                inputs = self._buffered_inputs[self._step % self.args.gradient_accumulation_steps]
                # print(inputs)
            self._step += 1
        else:
            # In evaluation, we don't reuse completions across multiple updates, so we don't need to buffer inputs.
            inputs = self._generate_and_score_completions(inputs)
        return inputs

        # # Simple pass-through, just like original
        # return inputs

    def _get_key_from_inputs(self, x, key):
        ele = x.get(key, None)
        assert ele is not None, f"The key {key} is not found in the input"
        if isinstance(ele, list):
            return [e for e in ele]
        else:
            return [ele]

    def _get_dynamic_filter_values(self):
        if not getattr(self.args, "dynamic_filter_schedule", None):
            return float(self.args.filter_low), float(self.args.filter_high)

        progress = 0.0
        if self.state.max_steps and self.state.max_steps > 0:
            progress = float(self.state.global_step) / float(self.state.max_steps)

        schedule = sorted(self.args.dynamic_filter_schedule, key=lambda x: x["epoch_ratio"])
        low, high = float(self.args.filter_low), float(self.args.filter_high)
        for p in schedule:
            if progress <= p["epoch_ratio"]:
                low = float(p.get("filter_low", low))
                high = float(p.get("filter_high", high))
                break
        else:
            if schedule:
                low = float(schedule[-1].get("filter_low", low))
                high = float(schedule[-1].get("filter_high", high))
        return low, high

    def _replay_scores_for_buffer(
        self,
        replay_score_key: str,
        group_difficulty: torch.Tensor,
        group_correct_rate: torch.Tensor,
        rewards_1d: torch.Tensor,
        G: int,
    ) -> torch.Tensor:
        """Per-group scalar for replay admission; higher means harder (more worth replaying)."""
        key = (replay_score_key or "difficulty").lower()
        if key == "accuracy":
            return 1.0 - group_correct_rate
        if key == "reward":
            gmean = rewards_1d.view(-1, G).mean(dim=1)
            return -gmean
        return group_difficulty

    def _generate_and_score_completions(self, inputs: dict[str, Union[torch.Tensor, Any]]) -> dict[str, Union[torch.Tensor, Any]]:
        device = self.accelerator.device
        self._grpo_gen_seq = getattr(self, "_grpo_gen_seq", 0) + 1
        # print(inputs)
        prompts = [x["prompt"] for x in inputs]
        # Full batch of prompt strings (do not use [0] — that would repeat one prompt for every row).
        prompts_text = self.vlm_module.prepare_prompt(self.processing_class, inputs)
        use_audio_in_video = self.use_audio_in_video
        # Prefill: put fixed tags in the prompt so the model continues inside <context> (avoids random first tokens).
        prefill = self._get_assistant_prefill()
        self._last_assistant_prefill = prefill
        if prefill:
            prompts_text = [p + prefill for p in prompts_text]
        # print(prompts_text)
        images, videos, audios = [], [], []


        for each in inputs:
            if each["images"] is not None:
                images.extend(each["images"])
            if each["audios"] is not None:
                audios.extend(each["audios"])
            if each["videos"] is not None:
                videos.extend(each["videos"])
        if len(images) == 0: images = None
        if len(audios) == 0: audios = None
        if len(videos) == 0: videos = None


        prompt_inputs = self.vlm_module.prepare_model_inputs(
            self.processing_class,
            prompts_text,
            images,
            audios,
            videos,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
            use_audio_in_video=use_audio_in_video,
        )
        prompt_inputs = super()._prepare_inputs(prompt_inputs)
        prompt_inputs["use_audio_in_video"] = use_audio_in_video

        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]
        prompt_ids_before_gen = prompt_ids.clone()

        # max_prompt_length is not supported yet
        # if self.max_prompt_length is not None:
        #     prompt_ids = prompt_ids[:, -self.max_prompt_length :]
        #     prompt_inputs["input_ids"] = prompt_ids
        #     prompt_mask = prompt_mask[:, -self.max_prompt_length :]
        #     prompt_inputs["attention_mask"] = prompt_mask

        # Generate completions
        gen_kw = {k: v for k, v in prompt_inputs.items() if k not in self.vlm_module.get_non_generate_params()}
        ds_plugin = getattr(self.accelerator.state, "deepspeed_plugin", None)
        # ZeRO-3: TRL's unwrap_model_for_generation(..., gather_deepspeed3_params=True) uses GatheredParameters +
        # remove_hooks around the inner HF module. After the context exits, the next DeepSpeedEngine forward
        # (e.g. compute_loss → _get_per_token_logps) can hit IndexError: pop from an empty deque in
        # partitioned_param_coordinator.reset_step(). So we must call generate on the engine itself here.
        # Tradeoff: engine autoregressive generate is much slower under Z3+offload than a one-shot gather path.
        # Mitigations: ZeRO-2 / optimizer-only offload, more GPUs, smaller model, or separate inference worker.
        if ds_plugin is not None and getattr(ds_plugin, "zero_stage", 0) == 3:
            generate_returned_result = self.model_wrapped.generate(
                **gen_kw,
                generation_config=self.generation_config,
            )
        else:
            gather_gen = bool(getattr(self.args, "ds3_gather_for_generation", True))
            with unwrap_model_for_generation(
                self.model_wrapped,
                self.accelerator,
                gather_deepspeed3_params=gather_gen,
            ) as unwrapped_model:
                generate_returned_result = unwrapped_model.generate(
                    **gen_kw,
                    generation_config=self.generation_config,
                )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        prompt_length = int(prompt_ids_before_gen.size(1))
        if not self.vlm_module.is_embeds_input():
            seq = generate_returned_result
            # Standard HF generate returns full sequence [prompt || new_tokens]. If the prefix != input_ids,
            # slicing at prompt_length mis-assigns tokens and the decoded "completion" can look like a mid-string start.
            if seq.size(1) < prompt_length:
                logger.error(
                    "[GRPO] generate() output width %s < prompt length %s — cannot split prompt/completion.",
                    seq.size(1),
                    prompt_length,
                )
                raise RuntimeError("generate() returned a sequence shorter than the prompt; check model.generate().")
            prefix = seq[:, :prompt_length]
            if not torch.equal(prefix, prompt_ids_before_gen):
                nwarn = int(getattr(self, "_grpo_prompt_align_mismatch_count", 0))
                if nwarn < 3:
                    mism = (prefix != prompt_ids_before_gen)
                    bad_rows = mism.any(dim=1).nonzero(as_tuple=True)[0]
                    for r in bad_rows[:4].tolist():
                        row_m = mism[r].nonzero(as_tuple=True)[0]
                        j = int(row_m[0]) if row_m.numel() else -1
                        logger.warning(
                            "[GRPO] generate() prefix != input_ids row=%s first_mismatch_col=%s in_tok=%s out_tok=%s "
                            "(slicing here would garble completion; fix model.generate vs input_ids).",
                            r,
                            j,
                            prompt_ids_before_gen[r, j].item() if j >= 0 else None,
                            prefix[r, j].item() if j >= 0 else None,
                        )
                    self._grpo_prompt_align_mismatch_count = nwarn + 1
                    if nwarn == 2:
                        logger.warning("[GRPO] Further prefix mismatches suppressed (max 3 logs).")
                if os.environ.get("GRPO_STRICT_PROMPT_ALIGNMENT", "").lower() in ("1", "true", "yes"):
                    raise RuntimeError(
                        "generate() output prefix does not match input_ids; set GRPO_STRICT_PROMPT_ALIGNMENT=0 to continue."
                    )
            prompt_completion_ids = seq
            prompt_ids = prompt_completion_ids[:, :prompt_length]
            completion_ids = prompt_completion_ids[:, prompt_length:]
        else:
            # In this case, the input of the LLM backbone is the embedding of the combination of the image and text prompt
            # So the returned result of the `generate` method only contains the completion ids
            completion_ids = generate_returned_result
            prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)

        # Defense in depth for model/custom-generation implementations that ignore
        # GenerationConfig.suppress_tokens.  A generated multimodal placeholder has
        # no matching pixel/audio feature tensor.  End that completion at the first
        # invalid placeholder instead of allowing the reference/policy forward to
        # enter the multimodal branch with missing metadata.
        invalid_token_ids = getattr(self, "_input_only_multimodal_token_ids", [])
        if invalid_token_ids and completion_ids.numel() > 0:
            invalid_ids = torch.tensor(invalid_token_ids, device=completion_ids.device, dtype=completion_ids.dtype)
            invalid_mask = torch.isin(completion_ids, invalid_ids)
            if invalid_mask.any():
                bad_rows = invalid_mask.any(dim=1).nonzero(as_tuple=True)[0]
                logger.warning(
                    "[GRPO] Suppressed generated input-only multimodal token(s) in %d completion row(s); "
                    "terminating those completions before log-prob computation.",
                    int(bad_rows.numel()),
                )
                eos_token_id = self.processing_class.eos_token_id
                if eos_token_id is None:
                    eos_token_id = self.processing_class.pad_token_id
                completion_ids = completion_ids.clone()
                for row in bad_rows.tolist():
                    first_bad = int(invalid_mask[row].nonzero(as_tuple=True)[0][0])
                    completion_ids[row, first_bad:] = eos_token_id
                prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)


        # Mask everything after the first EOS token
        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        # Concatenate prompt_mask with completion_mask for logit computation
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B, P+C)

        # Get the multimodal inputs
        multimodal_keywords = self.vlm_module.get_custom_multimodal_keywords()
        # IMPORTANT: only pass keys that are present AND not None.
        # Passing e.g. `audio_feature_lengths=None` can crash some multimodal models
        # (Qwen2.5-Omni expects audio_seqlens to be a tensor when audio branch is triggered).
        multimodal_inputs = {}
        for k in multimodal_keywords:
            if k not in prompt_inputs:
                continue
            v = prompt_inputs[k]
            if v is None:
                continue
            multimodal_inputs[k] = v
        with torch.no_grad():
            # When using num_iterations == 1, old_per_token_logps == per_token_logps, so we can skip its
            # computation here, and use per_token_logps.detach() instead.
            if self.num_iterations > 1:
                old_per_token_logps = self._get_per_token_logps(
                    self.model, prompt_completion_ids, attention_mask, **multimodal_inputs
                )
                old_per_token_logps = old_per_token_logps[:, prompt_length - 1:]
            else:
                old_per_token_logps = None

            if self.beta == 0.0:
                ref_per_token_logps = None
            elif self.ref_model is not None:
                ref_per_token_logps = self._get_per_token_logps(
                    self.ref_model, prompt_completion_ids, attention_mask, **multimodal_inputs
                )
                ref_per_token_logps = ref_per_token_logps[:, prompt_length - 1:]
            else:
                with self.accelerator.unwrap_model(self.model).disable_adapter():
                    ref_per_token_logps = self._get_per_token_logps(
                        self.model, prompt_completion_ids, attention_mask, **multimodal_inputs
                    )
                ref_per_token_logps = ref_per_token_logps[:, prompt_length - 1:]

        # Decode only tokens covered by completion_mask (per-row decode; avoids batch_decode edge cases on padding).
        completions_text, completions_decode_meta = self._decode_completions_masked(completion_ids, completion_mask)
        prefill = getattr(self, "_last_assistant_prefill", "") or ""
        completions_text_full = [prefill + c for c in completions_text] if prefill else list(completions_text)
        if is_conversational(inputs[0]):
            completions = [[{"role": "assistant", "content": completion}] for completion in completions_text_full]

        # Compute the rewards
        # No need to duplicate prompts as we're not generating multiple completions per prompt

        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
        for i, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes)
        ):
            if isinstance(reward_func, PreTrainedModel):
                if is_conversational(inputs[0]):
                    messages = [{"messages": p + c} for p, c in zip(prompts, completions)]
                    texts = [apply_chat_template(x, reward_processing_class)["text"] for x in messages]
                else:
                    texts = [p + c for p, c in zip(prompts, completions)]
                reward_inputs = reward_processing_class(
                    texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
                )
                reward_inputs = super()._prepare_inputs(reward_inputs)
                with torch.inference_mode():
                    rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]  # Shape (B*G,)
            else:
                # Repeat all input columns (but "prompt" and "completion") to match the number of generations
                reward_kwargs = {key: [] for key in inputs[0].keys() if key not in ["prompt", "completion"]}
                for key in reward_kwargs:
                    for example in inputs:
                        # No need to duplicate prompts as we're not generating multiple completions per prompt
                        # reward_kwargs[key].extend([example[key]] * self.num_generations)
                        reward_kwargs[key].extend([example[key]])
                output_reward_func = reward_func(prompts=prompts, completions=completions, **reward_kwargs)
                output_reward_func = [reward if reward is not None else torch.nan for reward in output_reward_func]
                rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        # markov
        if rewards_per_func.size(1) ==4 and self.markov_reward: # format, acc, reason, evi
            print("using markov")
            not_valid_evidence_index = rewards_per_func[:, -1]<=0.4
            not_valid_reason_index = rewards_per_func[:, -2]<=0.4
            rewards_per_func[not_valid_evidence_index, 1] = 0
            rewards_per_func[not_valid_evidence_index, 2] = 0
            rewards_per_func[not_valid_reason_index, 1] = 0

            # not_valid_format_index = rewards_per_func[:, 1]<=0.2
            # rewards_per_func[not_valid_format_index, 0] = 0

        # if rewards_per_func.size(1) ==2: # format, acc, reason, evi
        #     print("using markov")
        # not_valid_format_index = rewards_per_func[:, 1]<=0.2

        # rewards_per_func[not_valid_format_index, :] = 0

        # If all reward functions return None for a given row, issue a detailed warning
        if torch.isnan(rewards_per_func).all(dim=1).any():
            nan_row_idx = torch.isnan(rewards_per_func).all(dim=1).nonzero(as_tuple=True)[0][0]
            row_reward_kwargs = {key: value[nan_row_idx] for key, value in reward_kwargs.items()}
            row_reward_kwargs["prompt"] = prompts[nan_row_idx]
            row_reward_kwargs["completion"] = completions[nan_row_idx]
            warnings.warn(
                f"All reward functions returned None for the following kwargs: {row_reward_kwargs}. "
                "Please ensure that at least one reward function returns a valid reward."
            )
        # Gather rewards across processes
        rewards_per_func = self.accelerator.gather(rewards_per_func)

        # Gather auxiliary fields for enhanced FAST-style mechanisms
        def _flatten_gathered_objects(obj):
            if isinstance(obj, list) and len(obj) > 0 and isinstance(obj[0], list):
                flat = []
                for sub in obj:
                    flat.extend(sub)
                return flat
            return obj

        sample_ids_local = [str(x.get("sample_id", str(i))) for i, x in enumerate(inputs)]
        sample_ids = _flatten_gathered_objects(gather_object(sample_ids_local))

        h_mm_local = []
        mc_img_local, mc_aud_local, mc_vid_local, mc_txt_local = [], [], [], []
        complexity_weights = {
            "image": float(self.args.complexity_weight_image),
            "audio": float(self.args.complexity_weight_audio),
            "video": float(self.args.complexity_weight_video),
            "text": float(getattr(self.args, "complexity_weight_text", 0.0)),
        }
        weight_sum = sum(complexity_weights.values())
        if abs(weight_sum - 1.0) > 1e-6:
            raise ValueError(f"Modality complexity weights must sum to 1.0, got {weight_sum}: {complexity_weights}")
        for x in inputs:
            mc = x.get("modality_complexity")
            required_components = ("image", "audio", "video", "text")
            valid_mc = isinstance(mc, dict) and all(component in mc for component in required_components)
            if not valid_mc:
                if bool(getattr(self.args, "require_modality_complexity", True)):
                    raise ValueError(
                        "Missing/invalid modality_complexity. The dataset loader must provide image/audio/video/text "
                        f"for every sample; sample_id={x.get('sample_id', '<unknown>')!r}, value={mc!r}"
                    )
                mc = {"image": 0.5, "audio": 0.5, "video": 0.5, "text": 0.5}
            for component in required_components:
                value = float(mc[component])
                if not 0.0 <= value <= 1.0:
                    raise ValueError(
                        f"modality_complexity[{component!r}] must be in [0,1], got {value}; "
                        f"sample_id={x.get('sample_id', '<unknown>')!r}"
                    )
            mc_img_local.append(float(mc.get("image", 0.5)))
            mc_aud_local.append(float(mc.get("audio", 0.5)))
            mc_vid_local.append(float(mc.get("video", 0.5)))
            mc_txt_local.append(float(mc.get("text", 0.5)))
            h_mm_val = (
                float(self.args.complexity_weight_image) * float(mc.get("image", 0.5))
                + float(self.args.complexity_weight_audio) * float(mc.get("audio", 0.5))
                + float(self.args.complexity_weight_video) * float(mc.get("video", 0.5))
                + float(getattr(self.args, "complexity_weight_text", 0.0)) * float(mc.get("text", 0.5))
            )
            h_mm_local.append(h_mm_val)
        h_mm = self.accelerator.gather(torch.tensor(h_mm_local, dtype=torch.float32, device=device))
        mc_img = self.accelerator.gather(torch.tensor(mc_img_local, dtype=torch.float32, device=device))
        mc_aud = self.accelerator.gather(torch.tensor(mc_aud_local, dtype=torch.float32, device=device))
        mc_vid = self.accelerator.gather(torch.tensor(mc_vid_local, dtype=torch.float32, device=device))
        mc_txt = self.accelerator.gather(torch.tensor(mc_txt_local, dtype=torch.float32, device=device))

        # Sum the rewards from all configured reward functions.
        full_rewards: list[torch.Tensor] = []
        patial_rewards: list[tuple[str, torch.Tensor]] = []
        for i, reward_func in enumerate(self.reward_funcs):
            func_name = reward_func.__name__ if hasattr(reward_func, "__name__") else reward_func.__class__.__name__
            weighted = rewards_per_func[:, i] * self.reward_weights[i].to(device)
            if "patial" not in func_name:
                full_rewards.append(weighted)
            else:
                patial_rewards.append((func_name, weighted))

        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )

        log_reward_breakdown = bool(getattr(self.args, "log_reward_breakdown", True))
        log_reward_to_console = bool(getattr(self.args, "log_reward_to_console", False))
        should_log_now = (
            self.accelerator.is_main_process
            and self.args.logging_steps > 0
            and (self.state.global_step % self.args.logging_steps == 0)
        )

        if len(full_rewards) == 0:
            rewards = torch.zeros(rewards_per_func.size(0), device=device, dtype=torch.float32)
        else:
            rewards = torch.stack(full_rewards, dim=-1).nansum(dim=1)

        if log_reward_breakdown:
            rb_base_mean = float(torch.nanmean(rewards).item())
            # Record modality complexity stats so they show up in the logged metrics dict.
            # (These are batch-level stats after gather across processes.)
            self._metrics["train"]["mm/image_mean"].append(float(mc_img.mean().item()) if mc_img.numel() else 0.0)
            self._metrics["train"]["mm/audio_mean"].append(float(mc_aud.mean().item()) if mc_aud.numel() else 0.0)
            self._metrics["train"]["mm/video_mean"].append(float(mc_vid.mean().item()) if mc_vid.numel() else 0.0)
            self._metrics["train"]["mm/text_mean"].append(float(mc_txt.mean().item()) if mc_txt.numel() else 0.0)
            self._metrics["train"]["mm/h_mm_mean"].append(float(h_mm.mean().item()) if h_mm.numel() else 0.0)
            self._metrics["train"]["mm/image_min"].append(float(mc_img.min().item()) if mc_img.numel() else 0.0)
            self._metrics["train"]["mm/image_max"].append(float(mc_img.max().item()) if mc_img.numel() else 0.0)
            self._metrics["train"]["mm/video_min"].append(float(mc_vid.min().item()) if mc_vid.numel() else 0.0)
            self._metrics["train"]["mm/video_max"].append(float(mc_vid.max().item()) if mc_vid.numel() else 0.0)
            self._metrics["train"]["mm/audio_min"].append(float(mc_aud.min().item()) if mc_aud.numel() else 0.0)
            self._metrics["train"]["mm/audio_max"].append(float(mc_aud.max().item()) if mc_aud.numel() else 0.0)
            self._metrics["train"]["mm/text_min"].append(float(mc_txt.min().item()) if mc_txt.numel() else 0.0)
            self._metrics["train"]["mm/text_max"].append(float(mc_txt.max().item()) if mc_txt.numel() else 0.0)
            self._metrics["train"]["mm/h_mm_min"].append(float(h_mm.min().item()) if h_mm.numel() else 0.0)
            self._metrics["train"]["mm/h_mm_max"].append(float(h_mm.max().item()) if h_mm.numel() else 0.0)

        if log_reward_to_console and should_log_now:
            logger.warning(
                "[mm_complexity] step=%s image=%.3f audio=%.3f video=%.3f text=%.3f h_mm=%.3f "
                "(img[min/max]=%.3f/%.3f vid[min/max]=%.3f/%.3f)",
                self.state.global_step,
                float(mc_img.mean().item()) if mc_img.numel() else 0.0,
                float(mc_aud.mean().item()) if mc_aud.numel() else 0.0,
                float(mc_vid.mean().item()) if mc_vid.numel() else 0.0,
                float(mc_txt.mean().item()) if mc_txt.numel() else 0.0,
                float(h_mm.mean().item()) if h_mm.numel() else 0.0,
                float(mc_img.min().item()) if mc_img.numel() else 0.0,
                float(mc_img.max().item()) if mc_img.numel() else 0.0,
                float(mc_vid.min().item()) if mc_vid.numel() else 0.0,
                float(mc_vid.max().item()) if mc_vid.numel() else 0.0,
            )

        G_cfg = int(self.num_generations)
        n_flat = int(rewards.numel())
        if G_cfg <= 0 or n_flat % G_cfg != 0:
            if bool(getattr(self.args, "strict_group_generation", True)):
                raise RuntimeError(
                    f"Invalid GRPO grouping: gathered rows={n_flat}, num_generations={G_cfg}. "
                    "Refusing to silently degrade to G=1; check distributed sampler/batch settings."
                )
            G_eff = 1
        else:
            G_eff = G_cfg

        def compute_advantage(rewards_flat: torch.Tensor):
            mean_grouped_rewards = rewards_flat.view(-1, G_eff).mean(dim=1)
            if G_eff <= 1:
                std_grouped_rewards = torch.zeros_like(mean_grouped_rewards)
            else:
                std_grouped_rewards = rewards_flat.view(-1, G_eff).std(dim=1)
            mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(G_eff, dim=0)
            std_grouped_rewards = std_grouped_rewards.repeat_interleave(G_eff, dim=0)
            if G_eff <= 1:
                advantages = rewards_flat
            else:
                advantages = rewards_flat - mean_grouped_rewards
            if self.args.scale_rewards and G_eff > 1:
                advantages = advantages / (std_grouped_rewards + 1e-4)
            return advantages, std_grouped_rewards

        # 1) Find the index of the accuracy reward function
        accuracy_col = None
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, nn.Module):
                name = getattr(reward_func.config, "_name_or_path", "").lower()
            else:
                name = getattr(reward_func, "__name__", "").lower()
            if "accuracy" in name:
                accuracy_col = i
                break

        # 2) Extract accuracy values and generate correct indicators
        if accuracy_col is None:
            raw_acc = torch.zeros_like(rewards)
            correct_indicator = torch.zeros_like(rewards)
        else:
            raw_acc = rewards_per_func[:, accuracy_col]
            correct_indicator = (raw_acc >= float(self.args.kl_acc_threshold)).float()

        # Honour the documented correctness gate.  Incorrect completions must
        # not obtain a positive scalar advantage from format/style rewards.
        if accuracy_col is not None and bool(getattr(self.args, "gate_rewards_on_accuracy", True)):
            weighted_accuracy = rewards_per_func[:, accuracy_col] * self.reward_weights[accuracy_col].to(device)
            rewards = torch.where(correct_indicator > 0.5, rewards, weighted_accuracy)
            if log_reward_breakdown:
                rb_base_mean = float(torch.nanmean(rewards).item())

        # 3) Compute group-level correctness
        group_correct_rate = correct_indicator.view(-1, G_eff).mean(dim=1)

        # Rollout-derived model signal. Keep it distinct from input complexity.
        group_var = correct_indicator.view(-1, G_eff).var(dim=1, unbiased=False)
        group_uncertainty = (group_var / 0.25).clamp(0, 1)
        group_rollout_hardness = (1.0 - group_correct_rate).clamp(0, 1)
        if bool(getattr(self.args, "use_uncertainty_in_beta", True)):
            lam = float(getattr(self.args, "beta_mix_lambda", 0.8))
            if not 0.0 <= lam <= 1.0:
                raise ValueError(f"beta_mix_lambda must be in [0,1], got {lam}")
            group_model_signal = (
                lam * group_rollout_hardness + (1.0 - lam) * group_uncertainty
            ).clamp(0, 1)
        else:
            group_model_signal = group_rollout_hardness

        # Calibrated input complexity and historical model-relative diagnostic.
        group_h_mm = h_mm.view(-1, G_eff).mean(dim=1)
        group_sample_ids = []
        for gi in range(group_correct_rate.shape[0]):
            idx = gi * G_eff
            group_sample_ids.append(sample_ids[idx] if idx < len(sample_ids) else str(gi))

        group_s_ext_ema = []
        alpha = float(getattr(self.args, "passk_ema_alpha", 0.3))
        for gi, sid in enumerate(group_sample_ids):
            passk_now = float(group_correct_rate[gi].item())
            prev = self._passk_ema.get(sid, passk_now)
            passk_ema = alpha * passk_now + (1.0 - alpha) * prev
            self._passk_ema[sid] = passk_ema
            group_s_ext_ema.append(1.0 - passk_ema)
        group_s_ext_ema = torch.tensor(group_s_ext_ema, dtype=torch.float32, device=device)

        input_mix = float(getattr(self.args, "input_complexity_mix", 0.25))
        if not 0.0 <= input_mix <= 1.0:
            raise ValueError(f"input_complexity_mix must be in [0,1], got {input_mix}")
        if bool(getattr(self.args, "use_intrinsic_difficulty", True)):
            group_allocation_score = (
                (1.0 - input_mix) * group_model_signal + input_mix * group_h_mm
            ).clamp(0, 1)
        else:
            group_allocation_score = group_model_signal
        # Backward-compatible alias for filtering/replay and existing metrics.
        group_difficulty = group_allocation_score
        self._metrics["train"]["allocation/rollout_hardness_mean"].append(
            float(group_rollout_hardness.mean().item())
        )
        self._metrics["train"]["allocation/uncertainty_mean"].append(float(group_uncertainty.mean().item()))
        self._metrics["train"]["allocation/model_signal_mean"].append(float(group_model_signal.mean().item()))
        self._metrics["train"]["allocation/input_complexity_mean"].append(float(group_h_mm.mean().item()))
        self._metrics["train"]["allocation/score_mean"].append(float(group_allocation_score.mean().item()))

        # Use the same declared allocation score for KL and length control.
        beta_direction = str(getattr(self.args, "beta_difficulty_direction", "lower_for_hard"))
        if beta_direction == "higher_for_hard":
            beta_position = group_allocation_score
        elif beta_direction == "lower_for_hard":
            beta_position = 1.0 - group_allocation_score
        else:
            raise ValueError(
                "beta_difficulty_direction must be 'higher_for_hard' or 'lower_for_hard', "
                f"got {beta_direction!r}"
            )
        group_beta = float(self.args.beta_min) + (
            float(self.args.beta_max) - float(self.args.beta_min)
        ) * beta_position
        dynamic_beta = group_beta.repeat_interleave(G_eff, dim=0)

        # 4) Length reward with confidence gate + tanh smoothing
        completion_token_len = self.accelerator.gather(completion_mask.sum(dim=1).float())
        group_avg_len = completion_token_len.view(-1, G_eff).mean(dim=1).clamp_min(1.0)
        avg_len_per_completion = group_avg_len.repeat_interleave(G_eff)
        avg_len = completion_token_len.mean().clamp_min(1.0)
        q = float(self.args.difficulty_percentile) / 100.0
        w = int(getattr(self.args, "difficulty_window_size", 32) or 0)
        if w > 0:
            self._difficulty_window.extend(group_difficulty.detach().float().cpu().tolist())
            if len(self._difficulty_window) >= 2:
                win = torch.tensor(list(self._difficulty_window), dtype=torch.float32, device=device)
                thr = torch.quantile(win, q=q)
            else:
                thr = torch.quantile(group_difficulty, q=q)
        else:
            thr = torch.quantile(group_difficulty, q=q)
        # Use strict `>` so when thr==0, groups with difficulty exactly 0 are NOT all marked hard.
        # With `>=`, (gd >= 0) is true for every non-negative difficulty, which collapses easy_correct to ~0
        # and removes length reward for hard+correct (the dominant case when pass@k is high).
        is_hard_sample = (group_difficulty > thr).repeat_interleave(G_eff)
        len_reward = torch.zeros_like(correct_indicator)
        easy_correct = (~is_hard_sample) & (correct_indicator > 0.5)
        hard_correct = is_hard_sample & (correct_indicator > 0.5)
        hard_incorrect = is_hard_sample & (correct_indicator < 0.5)
        # Length shaping: reward shorter completions whenever the answer is correct (easy or hard).
        # Previously only easy_correct received this; hard_correct was always 0 — bad when most groups are "hard".
        len_reward[easy_correct] = (
            1.0 - completion_token_len[easy_correct] / avg_len_per_completion[easy_correct]
        )
        if bool(getattr(self.args, "reward_long_incorrect", True)):
            len_reward[hard_incorrect] = (
                completion_token_len[hard_incorrect] / avg_len_per_completion[hard_incorrect] - 1.0
            )
        confident = (raw_acc - float(self.args.kl_acc_threshold)).abs() > float(self.args.length_reward_margin)
        len_reward = len_reward * confident.float()
        len_reward = torch.tanh(float(self.args.length_reward_alpha) * len_reward)
        length_term = torch.zeros_like(rewards)
        if bool(getattr(self.args, "enable_length_reward", True)):
            length_term = float(self.args.length_reward_weight) * len_reward
            rewards = rewards + length_term
        if log_reward_breakdown:
            rb_length_mean = float(torch.nanmean(length_term).item())
            rb_pre_filter_mean = float(torch.nanmean(rewards).item())
        if (
            bool(getattr(self.args, "log_length_reward_details", False))
            and self.accelerator.is_main_process
            and self.args.logging_steps > 0
            and self.state.global_step % self.args.logging_steps == 0
        ):
            current_mode_for_lenlog = "eval" if self.control.should_evaluate else "train"
            # Group-level summary
            gcr = group_correct_rate.detach()
            gd = group_difficulty.detach()
            hard_frac = float((gd > thr).float().mean().item()) if gd.numel() else 0.0
            # Sample-level summary
            conf_frac = float(confident.float().mean().item()) if confident.numel() else 0.0
            easy_correct_frac = float(easy_correct.float().mean().item()) if easy_correct.numel() else 0.0
            hard_correct_frac = float(hard_correct.float().mean().item()) if hard_correct.numel() else 0.0
            hard_incorrect_frac = float(hard_incorrect.float().mean().item()) if hard_incorrect.numel() else 0.0
            # Length term stats (after tanh, before weight is len_reward; after weight is length_term)
            lr_mean = float(len_reward.mean().item()) if len_reward.numel() else 0.0
            lr_abs_mean = float(len_reward.abs().mean().item()) if len_reward.numel() else 0.0
            lt_mean = float(length_term.mean().item()) if length_term.numel() else 0.0
            lt_abs_mean = float(length_term.abs().mean().item()) if length_term.numel() else 0.0
            # Completion length distribution
            clen = completion_token_len.detach()
            clen_mean = float(clen.mean().item()) if clen.numel() else 0.0
            clen_min = float(clen.min().item()) if clen.numel() else 0.0
            clen_max = float(clen.max().item()) if clen.numel() else 0.0
            # Also push into metrics dict so it always shows up.
            self._metrics[current_mode_for_lenlog]["length/avg_len"].append(float(avg_len.item()))
            self._metrics[current_mode_for_lenlog]["length/thr"].append(float(thr.item()) if thr.numel() else 0.0)
            self._metrics[current_mode_for_lenlog]["length/hard_frac"].append(hard_frac)
            self._metrics[current_mode_for_lenlog]["length/conf_frac"].append(conf_frac)
            self._metrics[current_mode_for_lenlog]["length/easy_correct_frac"].append(easy_correct_frac)
            self._metrics[current_mode_for_lenlog]["length/hard_correct_frac"].append(hard_correct_frac)
            self._metrics[current_mode_for_lenlog]["length/hard_incorrect_frac"].append(hard_incorrect_frac)
            self._metrics[current_mode_for_lenlog]["length/len_reward_abs_mean"].append(lr_abs_mean)
            self._metrics[current_mode_for_lenlog]["length/length_term_abs_mean"].append(lt_abs_mean)
            self._metrics[current_mode_for_lenlog]["length/len_min"].append(clen_min)
            self._metrics[current_mode_for_lenlog]["length/len_max"].append(clen_max)
            self._metrics[current_mode_for_lenlog]["length/len_mean"].append(clen_mean)

            logger.warning(
                "[length_calc] step=%s G=%s avg_len=%.2f len[min/mean/max]=%.0f/%.2f/%.0f "
                "thr=%.6f hard_frac=%.3f gcr_mean=%.3f gcr_min=%.3f gcr_max=%.3f "
                "conf_frac=%.3f easy_correct=%.3f hard_correct=%.3f hard_incorrect=%.3f "
                "len_reward_mean=%.6f len_reward_abs_mean=%.6f length_term_mean=%.6f length_term_abs_mean=%.6f "
                "(weight=%.4f alpha=%.3f margin=%.3f diff_pct=%.1f)",
                self.state.global_step,
                G_eff,
                float(avg_len.item()),
                clen_min,
                clen_mean,
                clen_max,
                float(thr.item()) if thr.numel() else 0.0,
                hard_frac,
                float(gcr.mean().item()) if gcr.numel() else 0.0,
                float(gcr.min().item()) if gcr.numel() else 0.0,
                float(gcr.max().item()) if gcr.numel() else 0.0,
                conf_frac,
                easy_correct_frac,
                hard_correct_frac,
                hard_incorrect_frac,
                lr_mean,
                lr_abs_mean,
                lt_mean,
                lt_abs_mean,
                float(getattr(self.args, "length_reward_weight", 0.0)),
                float(getattr(self.args, "length_reward_alpha", 0.0)),
                float(getattr(self.args, "length_reward_margin", 0.0)),
                float(getattr(self.args, "difficulty_percentile", 0.0)),
            )
            nshow = int(getattr(self.args, "log_length_reward_groups", 0) or 0)
            if nshow > 0 and gcr.numel() and gd.numel():
                nshow = min(nshow, int(gcr.numel()), int(gd.numel()))
                thr_v = float(thr.item()) if thr.numel() else 0.0
                for gi in range(nshow):
                    sid = group_sample_ids[gi] if gi < len(group_sample_ids) else str(gi)
                    gcr_i = float(gcr[gi].item())
                    gd_i = float(gd[gi].item())
                    is_hard = bool(gd_i > thr_v)
                    passk_ema_i = (
                        float((1.0 - group_s_ext_ema[gi]).item())
                        if gi < int(group_s_ext_ema.numel())
                        else float("nan")
                    )
                    self._metrics[current_mode_for_lenlog][f"length/group{gi}_passk_ema"].append(passk_ema_i)
                    self._metrics[current_mode_for_lenlog][f"length/group{gi}_gcr"].append(gcr_i)
                    self._metrics[current_mode_for_lenlog][f"length/group{gi}_difficulty"].append(gd_i)
                    self._metrics[current_mode_for_lenlog][f"length/group{gi}_hard"].append(1.0 if is_hard else 0.0)
                    logger.warning(
                        "[length_group] step=%s group=%s sample_id=%s passk_ema=%.3f gcr=%.3f difficulty=%.6f thr=%.6f hard=%s",
                        self.state.global_step,
                        gi,
                        sid,
                        passk_ema_i,
                        gcr_i,
                        gd_i,
                        thr_v,
                        is_hard,
                    )

        # 5) Online filtering + replay (soft skip via zeroed rewards/advantages)
        sample_keep_mask = torch.ones_like(rewards, dtype=torch.float32)
        if bool(getattr(self.args, "online_filtering", False)):
            low, high = self._get_dynamic_filter_values()
            filter_key = getattr(self.args, "filter_key", "accuracy")
            if filter_key == "difficulty":
                group_metric = group_difficulty
            elif filter_key == "reward":
                group_metric = rewards.view(-1, G_eff).mean(dim=1)
            else:
                group_metric = group_correct_rate

            base_keep = (group_metric > low) & (group_metric < high)
            replay_id_set = set(self._replay_ids)
            replay_keep = torch.tensor([sid in replay_id_set for sid in group_sample_ids], device=device, dtype=torch.bool)
            keep_group = base_keep | replay_keep

            # Update replay buffer: top `replay_ratio` fraction by `replay_score_key` (hardest groups).
            rr = float(getattr(self.args, "replay_ratio", 0.2))
            rr = min(max(rr, 1e-6), 1.0 - 1e-6)
            rkey = getattr(self.args, "replay_score_key", "difficulty")
            replay_scores = self._replay_scores_for_buffer(
                rkey, group_difficulty, group_correct_rate, rewards, G_eff
            )
            if replay_scores.numel() > 0:
                hard_thr = torch.quantile(replay_scores, 1.0 - rr)
                for sid, score in zip(group_sample_ids, replay_scores.tolist()):
                    if score >= float(hard_thr.item()):
                        self._replay_ids.append(sid)

            sample_keep_mask = keep_group.repeat_interleave(G_eff).float()
            # soft skip filtered groups while preserving tensor shapes
            rewards = rewards * sample_keep_mask
            dynamic_beta = dynamic_beta * sample_keep_mask

        if log_reward_breakdown:
            rb_post_filter_mean = float(torch.nanmean(rewards).item())

        current_mode = "eval" if self.control.should_evaluate else "train"
        self._metrics[current_mode]["keep_ratio"].append(float(sample_keep_mask.mean().item()))
        self._metrics[current_mode]["dynamic_beta_mean"].append(float(dynamic_beta.mean().item()))
        # -------- End enhanced additions --------

        advantages, std_grouped_rewards = compute_advantage(rewards)
        if log_reward_breakdown:
            rb_adv_mean = float(torch.nanmean(advantages).item())
            nanstd_fn = getattr(torch, "nanstd", None)
            if nanstd_fn is not None:
                rb_adv_std = float(nanstd_fn(advantages).item())
            else:
                flat = advantages.detach().flatten()
                ok = ~torch.isnan(flat)
                rb_adv_std = (
                    float(flat[ok].std(unbiased=False).item())
                    if ok.any() and flat[ok].numel() > 1
                    else 0.0
                )
        local_keep_mask = sample_keep_mask[process_slice]
        advantages = advantages[process_slice] * local_keep_mask

        patial_advantages = []
        if len(patial_rewards) > 0:
            # Same scalar advantage as main GRPO signal (reference: compute_advantage(rewards)[0][process_slice] * mask).
            partial_reward = advantages
            for func_name, _ in patial_rewards:
                if "context" in func_name:
                    mask = generate_2d_mask(completion_ids, [34528], [522, 2147])
                else:
                    mask = generate_2d_mask(completion_ids, [13708, 766], [522, 26865])
                patial_advantages.append({"name": func_name, "reward": partial_reward, "mask": mask})



        mode = "eval" if self.control.should_evaluate else "train"

        if mode == "train":
            self._total_train_tokens += self.accelerator.gather_for_metrics(attention_mask.sum()).sum().item()
        self._metrics[mode]["num_tokens"] = [self._total_train_tokens]


        # Log the metrics
        completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        self._metrics[mode]["completion_length"].append(completion_length)
        # No EOS in the completion span usually means generation stopped at max_new_tokens (hard truncate).
        # That breaks format_reward regex (missing closing tags), not terminal Rich column width.
        completion_no_eos_rate = (
            self.accelerator.gather_for_metrics((~is_eos.any(dim=1)).float()).mean().item()
        )
        self._metrics[mode]["completion_no_eos_rate"].append(completion_no_eos_rate)

        # reward_per_func = self.accelerator.gather_for_metrics(rewards_per_func).mean(0)
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, nn.Module):  # Module instead of PretrainedModel for compat with compiled models
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = reward_func.__name__
            # Only calculate mean for samples where this reward function was applied (non-NaN values)
            mean_rewards = torch.nanmean(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}"].append(mean_rewards)
        self._metrics[mode]["reward"].append(rewards.mean().item())
        self._metrics[mode]["reward_std"].append(std_grouped_rewards.mean().item())

        if log_reward_breakdown:
            self._metrics[mode]["rewards/base_weighted_mean"].append(rb_base_mean)
            self._metrics[mode]["rewards/length_term_mean"].append(rb_length_mean)
            self._metrics[mode]["rewards/pre_filter_mean"].append(rb_pre_filter_mean)
            self._metrics[mode]["rewards/post_filter_mean"].append(rb_post_filter_mean)
            self._metrics[mode]["rewards/advantage_mean"].append(rb_adv_mean)
            self._metrics[mode]["rewards/advantage_std"].append(rb_adv_std)

        if (
            bool(getattr(self.args, "log_reward_to_console", False))
            and log_reward_breakdown
            and self.accelerator.is_main_process
            and self.args.logging_steps > 0
            and self.state.global_step % self.args.logging_steps == 0
        ):
            logger.info(
                "[reward_calc] step=%s base_weighted_mean=%.6f length_term_mean=%.6f pre_filter=%.6f "
                "post_filter=%.6f adv_mean=%.6f adv_std=%.6f | "
                "base=sum_i(w_i*full_reward_i); +length; *keep_mask if online_filter; "
                "adv=reward-mean_group[/std_group if scale_rewards]; partial rewards add masked adv in loss",
                self.state.global_step,
                rb_base_mean,
                rb_length_mean,
                rb_pre_filter_mean,
                rb_post_filter_mean,
                rb_adv_mean,
                rb_adv_std,
            )
        elif (
            bool(getattr(self.args, "log_reward_to_console", False))
            and not log_reward_breakdown
            and self.accelerator.is_main_process
            and self.args.logging_steps > 0
            and self.state.global_step % self.args.logging_steps == 0
        ):
            parts = [
                f"step={self.state.global_step}",
                f"reward_mean={rewards.mean().item():.6f}",
                f"reward_std_mean={std_grouped_rewards.mean().item():.6f}",
                f"completion_no_eos_rate={self._metrics[mode]['completion_no_eos_rate'][-1]:.4f}",
            ]
            for i, reward_func in enumerate(self.reward_funcs):
                name = (
                    reward_func.config._name_or_path.split("/")[-1]
                    if isinstance(reward_func, nn.Module)
                    else getattr(reward_func, "__name__", type(reward_func).__name__)
                )
                parts.append(
                    f"rewards/{name}={torch.nanmean(rewards_per_func[:, i]).item():.6f}"
                )
            logger.info("[reward] %s", " ".join(parts))

        if self.log_completions and self.state.global_step % self.args.logging_steps == 0:
            prompts_to_log = gather_object(prompts_text)
            completions_to_log = gather_object(completions_text_full)
            meta_to_log = gather_object(completions_decode_meta)
            rewards_to_log = rewards.tolist()

            if self.accelerator.is_main_process:
                if is_rich_available():
                    print_prompt_completions_sample(
                        prompts_to_log,
                        completions_to_log,
                        rewards_to_log,
                        self.state.global_step,
                    )
                full_log = _append_full_completions_log(
                    self.args.output_dir,
                    self.state.global_step,
                    prompts_to_log,
                    completions_to_log,
                    rewards_to_log,
                    gen_seq=self._grpo_gen_seq,
                    per_sample_meta=meta_to_log,
                )
                if full_log:
                    logger.info(
                        "[GRPO] Full prompts/completions (%d samples) -> %s",
                        len(completions_to_log),
                        full_log,
                    )
                if completions_to_log:
                    r0 = self._metrics[mode]["completion_no_eos_rate"][-1]
                    if r0 > 0.05:
                        c0 = completions_to_log[0]
                        tail = c0[-400:] if len(c0) > 400 else c0
                        logger.info(
                            "[GRPO] Rewards use full decoded text (Rich table is display-only). "
                            "completion_no_eos_rate=%.2f (high => often hitting max_new_tokens=%s). "
                            "completion[0] len=%s tail=%r",
                            r0,
                            self.max_completion_length,
                            len(c0),
                            tail,
                        )
                # if self.args.report_to and "wandb" in self.args.report_to and wandb.run is not None:
                #     import pandas as pd

                #     # For logging
                #     table = {
                #         "step": [str(self.state.global_step)] * len(rewards),
                #         "prompt": prompts_to_log,
                #         "completion": completions_to_log,
                #         "reward": rewards.tolist(),
                #     }
                #     df = pd.DataFrame(table)
                #     wandb.log({"completions": wandb.Table(dataframe=df)})

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "old_per_token_logps": old_per_token_logps,
            "ref_per_token_logps": ref_per_token_logps,
            "advantages": advantages,
            "multimodal_inputs": multimodal_inputs,
            "patial_advantages": patial_advantages,
            "dynamic_beta": dynamic_beta[process_slice],
            "sample_keep_mask": local_keep_mask,
        }

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")



        # Check if we need to generate new completions or use buffered ones
        # if self.state.global_step % self.num_iterations == 0:
        #     inputs = self._generate_and_score_completions(inputs)
        #     self._buffered_inputs[self._step % self.args.gradient_accumulation_steps] = inputs
        # else:
        #     inputs = self._buffered_inputs[self._step % self.args.gradient_accumulation_steps]
        # self._step += 1

        # Get the prepared inputs
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        multimodal_inputs = inputs["multimodal_inputs"]

        # Concatenate for full sequence
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)

        # Get the current policy's log probabilities
        per_token_logps = self._get_per_token_logps(model, input_ids, attention_mask, **multimodal_inputs)
        # Get rid of the prompt (-1 because of the shift done in get_per_token_logps)
        per_token_logps = per_token_logps[:, prompt_ids.size(1) - 1:]

        # Get the advantages from inputs. Baseline is scalar per sample; partial terms use (B,1)*(B,T) masks.
        advantages = inputs["advantages"]
        patial_advantages = inputs["patial_advantages"]

        adv_per_token = advantages.unsqueeze(-1).to(dtype=torch.float32)
        for patial_advantage in patial_advantages:
            pr = patial_advantage["reward"].unsqueeze(-1).to(dtype=torch.float32)
            m = patial_advantage["mask"].to(dtype=torch.float32, device=adv_per_token.device)
            adv_per_token = adv_per_token + pr * m

        mode = "eval" if self.control.should_evaluate else "train"
        # When using num_iterations == 1, old_per_token_logps == per_token_logps, so we can skip its computation
        # and use per_token_logps.detach() instead
        old_per_token_logps = inputs["old_per_token_logps"] if self.num_iterations > 1 else per_token_logps.detach()

        # Compute the policy ratio and clipped version
        coef_1 = torch.exp(per_token_logps - old_per_token_logps)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)
        per_token_loss1 = coef_1 * adv_per_token
        per_token_loss2 = coef_2 * adv_per_token
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)

        # Add KL penalty if beta > 0
        if self.beta > 0:
            ref_per_token_logps = inputs["ref_per_token_logps"]
            per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1

            if bool(getattr(self.args, "use_dynamic_beta", True)) and "dynamic_beta" in inputs:
                beta_vec = inputs["dynamic_beta"].unsqueeze(1)
                if "sample_keep_mask" in inputs:
                    beta_vec = beta_vec * inputs["sample_keep_mask"].unsqueeze(1)
            else:
                beta_vec = torch.full((per_token_kl.size(0), 1), float(self.beta), device=per_token_kl.device)

            per_token_loss = per_token_loss + beta_vec * per_token_kl

            # Log KL divergence

            mean_kl = (per_token_kl * completion_mask).sum() / completion_mask.sum()
            self._metrics[mode]["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())

        # Compute final loss
        loss = (per_token_loss * completion_mask).sum() / completion_mask.sum()

        # Log clip ratio
        is_clipped = (per_token_loss1 < per_token_loss2).float()
        clip_ratio = (is_clipped * completion_mask).sum() / completion_mask.sum()
        self._metrics[mode]["clip_ratio"].append(self.accelerator.gather_for_metrics(clip_ratio).mean().item())

        return loss

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        mode = "eval" if self.control.should_evaluate else "train"
        metrics = {key: sum(val) / len(val) for key, val in self._metrics[mode].items()}  # average the metrics

        # This method can be called both in training and evaluation. When called in evaluation, the keys in `logs`
        # start with "eval_". We need to add the prefix "eval_" to the keys in `metrics` to match the format.
        if mode == "eval":
            metrics = {f"eval_{key}": val for key, val in metrics.items()}

        logs = {**logs, **metrics}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:  # transformers<=4.46
            super().log(logs)
        self._metrics[mode].clear()

    def create_model_card(
        self,
        model_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        tags: Union[str, list[str], None] = None,
    ):
        """
        Creates a draft of a model card using the information available to the `Trainer`.

        Args:
            model_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the model.
            dataset_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the dataset used for training.
            tags (`str`, `list[str]` or `None`, *optional*, defaults to `None`):
                Tags to be associated with the model card.
        """
        if not self.is_world_process_zero():
            return

        if hasattr(self.model.config, "_name_or_path") and not os.path.isdir(self.model.config._name_or_path):
            base_model = self.model.config._name_or_path
        else:
            base_model = None

        tags = tags or []
        if isinstance(tags, str):
            tags = [tags]

        if hasattr(self.model.config, "unsloth_version"):
            tags.append("unsloth")

        citation = textwrap.dedent(
            """\
            @article{zhihong2024deepseekmath,
                title        = {{DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models}},
                author       = {Zhihong Shao and Peiyi Wang and Qihao Zhu and Runxin Xu and Junxiao Song and Mingchuan Zhang and Y. K. Li and Y. Wu and Daya Guo},
                year         = 2024,
                eprint       = {arXiv:2402.03300},
            """
        )

        model_card = generate_model_card(
            base_model=base_model,
            model_name=model_name,
            hub_model_id=self.hub_model_id,
            dataset_name=dataset_name,
            tags=tags,
            wandb_url=wandb.run.get_url() if is_wandb_available() and wandb.run is not None else None,
            comet_url=get_comet_experiment_url(),
            trainer_name="GRPO",
            trainer_citation=citation,
            paper_title="DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models",
            paper_id="2402.03300",
        )

        model_card.save(os.path.join(self.args.output_dir, "README.md"))

    def _get_train_sampler(self, train_dataset: Optional[Any] = None) -> Optional[Sampler]:
        """Returns a sampler that ensures proper data sampling for GRPO training."""
        if train_dataset is None:
            train_dataset = self.train_dataset
        if train_dataset is None or not has_length(train_dataset):
            return None

        effective_batch_size = (
            self.args.per_device_train_batch_size
            * self.accelerator.num_processes
            * self.args.gradient_accumulation_steps
        )
        # RepeatRandomSampler uses range(0, n, batch_size) — batch_size must be >= 1.
        # Need per_device * world_size * grad_acc >= num_generations (e.g. 1*1*1 with G=2 -> 0).
        sampler_batch_size = effective_batch_size // self.num_generations
        if sampler_batch_size < 1:
            logger.warning(
                "GRPO train sampler: per_device_train_batch_size * num_processes * "
                "gradient_accumulation_steps (%s) < num_generations (%s); using sampler batch_size=1. "
                "Prefer gradient_accumulation_steps >= ceil(num_generations / (per_device * world_size)) "
                "for correct group sizing.",
                effective_batch_size,
                self.num_generations,
            )
            sampler_batch_size = 1

        return RepeatRandomSampler(
            data_source=train_dataset,
            mini_repeat_count=self.num_generations,
            batch_size=sampler_batch_size,
            repeat_count=self.num_iterations,
            seed=self.args.seed,
        )


    def _get_eval_sampler(self, eval_dataset) -> Sampler:
        """Returns a sampler for evaluation."""
        return RepeatRandomSampler(
            data_source=eval_dataset,
            mini_repeat_count=self.num_generations,
            seed=self.args.seed,
        )
