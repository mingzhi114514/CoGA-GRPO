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

from dataclasses import dataclass, field
from typing import Optional

from transformers import TrainingArguments


@dataclass
class GRPOConfig(TrainingArguments):
    r"""
    Configuration class for the [`GRPOTrainer`].

    Only the parameters specific to GRPO training are listed here. For details on other parameters, refer to the
    [`~transformers.TrainingArguments`] documentation.

    Using [`~transformers.HfArgumentParser`] we can turn this class into
    [argparse](https://docs.python.org/3/library/argparse#module-argparse) arguments that can be specified on the
    command line.

    Parameters:
        > Parameters that control the model and reference model

        model_init_kwargs (`dict[str, Any]` or `None`, *optional*, defaults to `None`):
            Keyword arguments for [`~transformers.AutoModelForCausalLM.from_pretrained`], used when the `model`
            argument of the [`GRPOTrainer`] is provided as a string.

        > Parameters that control the data preprocessing

        remove_unused_columns (`bool`, *optional*, defaults to `False`):
            Whether to only keep the column `"prompt"` in the dataset. If you use a custom reward function that
            requires any column other than `"prompts"` and `"completions"`, you should keep this to `False`.
        max_prompt_length (`int` or `None`, *optional*, defaults to `512`):
            Maximum length of the prompt. If the prompt is longer than this value, it will be truncated left.
        num_generations (`int` or `None`, *optional*, defaults to `8`):
            Number of generations per prompt to sample. The global batch size (num_processes * per_device_batch_size)
            must be divisible by this value.
        max_completion_length (`int` or `None`, *optional*, defaults to `256`):
            Maximum length of the generated completion.
        ds3_gather_for_generation (`bool`, *optional*, defaults to `True`):
            This setting applies to DeepSpeed ZeRO-3. If enabled, the policy model weights are gathered for generation,
            improving generation speed. However, disabling this option allows training models that exceed the VRAM
            capacity of a single GPU, albeit at the cost of slower generation. Disabling this option is not compatible
            with vLLM generation.

        > Parameters that control generation

        temperature (`float`, defaults to `0.9`):
            Temperature for sampling. The higher the temperature, the more random the completions.
        top_p (`float`, *optional*, defaults to `1.0`):
            Float that controls the cumulative probability of the top tokens to consider. Must be in (0, 1]. Set to
            `1.0` to consider all tokens.
        top_k (`int` or `None`, *optional*, defaults to `50`):
            Number of highest probability vocabulary tokens to keep for top-k-filtering. If `None`, top-k-filtering is
            disabled.
        min_p (`float` or `None`, *optional*, defaults to `None`):
            Minimum token probability, which will be scaled by the probability of the most likely token. It must be a
            value between `0.0` and `1.0`. Typical values are in the `0.01-0.2` range.
        repetition_penalty (`float`, *optional*, defaults to `1.0`):
            Float that penalizes new tokens based on whether they appear in the prompt and the generated text so far.
            Values > `1.0` encourage the model to use new tokens, while values < `1.0` encourage the model to repeat
            tokens.
        cache_implementation (`str` or `None`, *optional*, defaults to `None`):
            Implementation of the cache method for faster generation when use_vllm is set to False.

        > Parameters that control generation acceleration powered by vLLM

        use_vllm (`bool`, *optional*, defaults to `False`):
            Whether to use vLLM for generating completions. If set to `True`, ensure that a GPU is kept unused for
            training, as vLLM will require one for generation. vLLM must be installed (`pip install vllm`).
        vllm_device (`str`, *optional*, defaults to `"auto"`):
            Device where vLLM generation will run, e.g. `"cuda:1"`. If set to `"auto"` (default), the system will
            automatically select the next available GPU after the last one used for training. This assumes that
            training has not already occupied all available GPUs. If only one device is available, the device will be
            shared between both training and vLLM.
        vllm_gpu_memory_utilization (`float`, *optional*, defaults to `0.9`):
            Ratio (between 0 and 1) of GPU memory to reserve for the model weights, activations, and KV cache on the
            device dedicated to generation powered by vLLM. Higher values will increase the KV cache size and thus
            improve the model's throughput. However, if the value is too high, it may cause out-of-memory (OOM) errors
            during initialization.
        vllm_dtype (`str`, *optional*, defaults to `"auto"`):
            Data type to use for vLLM generation. If set to `"auto"`, the data type will be automatically determined
            based on the model configuration. Find the supported values in the vLLM documentation.
        vllm_max_model_len (`int` or `None`, *optional*, defaults to `None`):
            If set, the `max_model_len` to use for vLLM. This could be useful when running with reduced
            `vllm_gpu_memory_utilization`, leading to a reduced KV cache size. If not set, vLLM will use the model
            context size, which might be much larger than the KV cache, leading to inefficiencies.
        vllm_enable_prefix_caching (`bool`, *optional*, defaults to `True`):
            Whether to enable prefix caching in vLLM. If set to `True` (default), ensure that the model and the hardware
            support this feature.
        vllm_guided_decoding_regex (`str` or `None`, *optional*, defaults to `None`):
            Regex for vLLM guided decoding. If `None` (default), guided decoding is disabled.

        > Parameters that control the training

        learning_rate (`float`, *optional*, defaults to `1e-6`):
            Initial learning rate for [`AdamW`] optimizer. The default value replaces that of
            [`~transformers.TrainingArguments`].
        beta (`float`, *optional*, defaults to `0.04`):
            KL coefficient. If `0.0`, the reference model is not loaded, reducing memory usage and improving training
            speed, but may be numerically unstable for long training runs.
        num_iterations (`int`, *optional*, defaults to `1`):
            Number of iterations per batch (denoted as μ in the algorithm).
        epsilon (`float`, *optional*, defaults to `0.2`):
            Epsilon value for clipping.
        epsilon_high (`float` or `None`, *optional*, defaults to `None`):
            Upper-bound epsilon value for clipping. If not specified, it defaults to the same value as the lower-bound
            specified in argument `epsilon`. Paper [DAPO](https://huggingface.co/papers/2503.14476) recommends `0.28`.
        reward_weights (`list[float]` or `None`, *optional*, defaults to `None`):
            Weights for each reward function. Must match the number of reward functions. If `None`, all rewards are
            weighted equally with weight `1.0`.
        sync_ref_model (`bool`, *optional*, defaults to `False`):
            Whether to synchronize the reference model with the active model every `ref_model_sync_steps` steps, using
            the `ref_model_mixup_alpha` parameter. This synchronization originites from the
            [TR-DPO](https://huggingface.co/papers/2404.09656) paper.
        ref_model_mixup_alpha (`float`, *optional*, defaults to `0.6`):
            α parameter from the [TR-DPO](https://huggingface.co/papers/2404.09656) paper, which controls the mix
            between the current policy and the previous reference policy during updates. The reference policy is
            updated according to the equation: `π_ref = α * π_θ + (1 - α) * π_ref_prev`. To use this parameter, you
            must set `sync_ref_model=True`.
        ref_model_sync_steps (`int`, *optional*, defaults to `512`):
            τ parameter from the [TR-DPO](https://huggingface.co/papers/2404.09656) paper, which determines how
            frequently the current policy is synchronized with the reference policy. To use this parameter, you must
            set `sync_ref_model=True`.

        > Parameters that control the logging

        log_completions (`bool`, *optional*, defaults to `False`):
            Whether to log a sample of (prompt, completion) pairs every `logging_steps` steps. If `rich` is
            installed, it prints the sample. If `wandb` logging is enabled, it logs it to `wandb`.
    """

    # Parameters that control the model and reference model
    model_init_kwargs: Optional[dict] = field(
        default=None,
        metadata={
            "help": "Keyword arguments for `transformers.AutoModelForCausalLM.from_pretrained`, used when the `model` "
            "argument of the `GRPOTrainer` is provided as a string."
        },
    )

    # Parameters that control the data preprocessing
    # The default value remove_unused_columns is overwritten from the parent class, because in GRPO we usually rely on
    # additional columns to compute the reward
    remove_unused_columns: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Whether to only keep the column 'prompt' in the dataset. If you use a custom reward function "
            "that requires any column other than 'prompts' and 'completions', you should keep this to `False`."
        },
    )
    max_prompt_length: Optional[int] = field(
        default=512,
        metadata={
            "help": "Maximum length of the prompt. If the prompt is longer than this value, it will be truncated left."
        },
    )
    num_generations: Optional[int] = field(
        default=8,
        metadata={
            "help": "Number of generations to sample. The global batch size (num_processes * per_device_batch_size) "
            "must be divisible by this value."
        },
    )
    max_completion_length: Optional[int] = field(
        default=256,
        metadata={"help": "Maximum length of the generated completion."},
    )
    ds3_gather_for_generation: bool = field(
        default=True,
        metadata={
            "help": "This setting applies to DeepSpeed ZeRO-3. If enabled, the policy model weights are gathered for "
            "generation, improving generation speed. However, disabling this option allows training models that "
            "exceed the VRAM capacity of a single GPU, albeit at the cost of slower generation. Disabling this option "
            "is not compatible with vLLM generation."
        },
    )

    # Parameters that control generation
    temperature: float = field(
        default=0.9,
        metadata={"help": "Temperature for sampling. The higher the temperature, the more random the completions."},
    )
    top_p: float = field(
        default=1.0,
        metadata={
            "help": "Float that controls the cumulative probability of the top tokens to consider. Must be in (0, 1]. "
            "Set to 1.0 to consider all tokens."
        },
    )
    top_k: Optional[int] = field(
        default=50,
        metadata={
            "help": "Number of highest probability vocabulary tokens to keep for top-k-filtering. If `None`, "
            "top-k-filtering is disabled."
        },
    )
    min_p: Optional[float] = field(
        default=None,
        metadata={
            "help": "Minimum token probability, which will be scaled by the probability of the most likely token. It "
            "must be a value between 0.0 and 1.0. Typical values are in the 0.01-0.2 range."
        },
    )
    repetition_penalty: float = field(
        default=1.0,
        metadata={
            "help": "Float that penalizes new tokens based on whether they appear in the prompt and the generated "
            "text so far. Values > 1.0 encourage the model to use new tokens, while values < 1.0 encourage the model "
            "to repeat tokens."
        },
    )
    cache_implementation: Optional[str] = field(
        default=None,
        metadata={"help": "Implementation of the cache method for faster generation when use_vllm is set to False."},
    )

    # Parameters that control generation acceleration powered by vLLM
    use_vllm: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Whether to use vLLM for generating completions. If set to `True`, ensure that a GPU is kept "
            "unused for training, as vLLM will require one for generation. vLLM must be installed "
            "(`pip install vllm`)."
        },
    )
    vllm_device: Optional[str] = field(
        default="auto",
        metadata={
            "help": "Device where vLLM generation will run, e.g. 'cuda:1'. If set to 'auto' (default), the system "
            "will automatically select the next available GPU after the last one used for training. This assumes "
            "that training has not already occupied all available GPUs."
        },
    )
    vllm_gpu_memory_utilization: float = field(
        default=0.9,
        metadata={
            "help": "Ratio (between 0 and 1) of GPU memory to reserve for the model weights, activations, and KV "
            "cache on the device dedicated to generation powered by vLLM. Higher values will increase the KV cache "
            "size and thus improve the model's throughput. However, if the value is too high, it may cause "
            "out-of-memory (OOM) errors during initialization."
        },
    )
    vllm_dtype: Optional[str] = field(
        default="auto",
        metadata={
            "help": "Data type to use for vLLM generation. If set to 'auto', the data type will be automatically "
            "determined based on the model configuration. Find the supported values in the vLLM documentation."
        },
    )
    vllm_max_model_len: Optional[int] = field(
        default=None,
        metadata={
            "help": "If set, the `max_model_len` to use for vLLM. This could be useful when running with reduced "
            "`vllm_gpu_memory_utilization`, leading to a reduced KV cache size. If not set, vLLM will use the model "
            "context size, which might be much larger than the KV cache, leading to inefficiencies."
        },
    )
    vllm_enable_prefix_caching: Optional[bool] = field(
        default=True,
        metadata={
            "help": "Whether to enable prefix caching in vLLM. If set to `True` (default), ensure that the model and "
            "the hardware support this feature."
        },
    )
    vllm_guided_decoding_regex: Optional[str] = field(
        default=None,
        metadata={"help": "Regex for vLLM guided decoding. If `None` (default), guided decoding is disabled."},
    )

    # Parameters that control the training
    learning_rate: float = field(
        default=1e-6,
        metadata={
            "help": "Initial learning rate for `AdamW` optimizer. The default value replaces that of "
            "`transformers.TrainingArguments`."
        },
    )
    beta: float = field(
        default=0.04,
        metadata={
            "help": "KL coefficient. If `0.0`, the reference model is not loaded, reducing memory usage and improving "
            "training speed, but may be numerically unstable for long training runs."
        },
    )
    num_iterations: int = field(
        default=1,
        metadata={"help": "Number of iterations per batch (denoted as μ in the algorithm)."},
    )
    epsilon: float = field(
        default=0.2,
        metadata={"help": "Epsilon value for clipping."},
    )
    epsilon_high: Optional[float] = field(
        default=None,
        metadata={
            "help": "Upper-bound epsilon value for clipping. If not specified, it defaults to the same value as the "
            "lower-bound specified in argument `epsilon`. Paper DAPO recommends `0.28`."
        },
    )
    reward_weights: Optional[list[float]] = field(
        default=None,
        metadata={
            "help": "Weights for each reward function. Must match the number of reward functions. If `None`, all "
            "rewards are weighted equally with weight `1.0`."
        },
    )
    sync_ref_model: bool = field(
        default=False,
        metadata={
            "help": "Whether to synchronize the reference model with the active model every `ref_model_sync_steps` "
            "steps, using the `ref_model_mixup_alpha` parameter."
        },
    )
    ref_model_mixup_alpha: float = field(
        default=0.6,
        metadata={
            "help": "α parameter from the TR-DPO paper, which controls the mix between the current policy and the "
            "previous reference policy during updates. The reference policy is updated according to the equation: "
            "`π_ref = α * π_θ + (1 - α) * π_ref_prev`. To use this parameter, you must set `sync_ref_model=True`."
        },
    )
    ref_model_sync_steps: int = field(
        default=512,
        metadata={
            "help": "τ parameter from the TR-DPO paper, which determines how frequently the current policy is "
            "synchronized with the reference policy. To use this parameter, you must set `sync_ref_model=True`."
        },
    )
    scale_rewards: bool = field(
        default=True,
        metadata={
            "help": "Whether to synchronize the reference model with the active model every `ref_model_sync_steps` "
            "steps, using the `ref_model_mixup_alpha` parameter."
        },
    )
    markov_reward: bool = field(
        default=False,
        metadata={
            "help": "Whether to synchronize the reference model with the active model every `ref_model_sync_steps` "
            "steps, using the `ref_model_mixup_alpha` parameter."
        },
    )
    assistant_prefill: Optional[str] = field(
        default=None,
        metadata={
            "help": "Appended to each prompt string before tokenization so generation continues after this literal "
            "prefix (e.g. '\\n<context>\\n' for Qwen three-block format). Reward functions and logs see "
            "prefill + generated text; policy gradients still apply only to generated tokens. "
            "Can also be set via env GRPO_ASSISTANT_PREFILL."
        },
    )

    # Enhanced FAST-style controls
    use_dynamic_beta: bool = field(
        default=True,
        metadata={"help": "Enable dynamic KL beta."},
    )
    beta_min: float = field(
        default=1e-4,
        metadata={"help": "Minimum dynamic KL coefficient."},
    )
    beta_max: float = field(
        default=0.04,
        metadata={"help": "Maximum dynamic KL coefficient."},
    )
    kl_acc_threshold: float = field(
        default=0.5,
        metadata={"help": "Threshold to binarize continuous accuracy when computing pass@k."},
    )
    gate_rewards_on_accuracy: bool = field(
        default=True,
        metadata={
            "help": "If True and an accuracy reward is registered: when raw_acc < kl_acc_threshold, the scalar "
            "reward uses only the weighted accuracy column (format/patial/etc. do not contribute); length reward is "
            "not added in that case. If False, all reward terms always contribute as weighted sum."
        },
    )
    use_uncertainty_in_beta: bool = field(
        default=True,
        metadata={"help": "Mix uncertainty into dynamic beta for stability."},
    )
    beta_mix_lambda: float = field(
        default=0.8,
        metadata={"help": "Mix ratio lambda*(1-pass@k)+(1-lambda)*uncertainty."},
    )
    input_complexity_mix: float = field(
        default=0.25,
        metadata={
            "help": "Convex weight assigned to calibrated input complexity when forming the allocation score."
        },
    )
    beta_difficulty_direction: str = field(
        default="lower_for_hard",
        metadata={
            "help": "Dynamic KL mapping: lower_for_hard (method) or higher_for_hard (direction stress test)."
        },
    )

    enable_length_reward: bool = field(
        default=True,
        metadata={"help": "Enable difficulty-aware length reward."},
    )
    length_reward_weight: float = field(
        default=0.25,
        metadata={"help": "Weight of length reward."},
    )
    difficulty_percentile: float = field(
        default=85.0,
        metadata={"help": "Hard-group percentile threshold (quantile over group_difficulty)."},
    )
    difficulty_window_size: int = field(
        default=32,
        metadata={
            "help": "If >0, compute the hard/easy threshold `thr` from a sliding window of the most recent "
            "K group_difficulty values (across steps) instead of only the current step. This avoids quantile "
            "degeneracy when a step contains too few groups or many ties."
        },
    )
    length_reward_margin: float = field(
        default=0.15,
        metadata={"help": "Confidence margin for enabling length reward."},
    )
    length_reward_alpha: float = field(
        default=1.0,
        metadata={"help": "Smoothing factor for tanh length reward."},
    )
    reward_long_incorrect: bool = field(
        default=False,
        metadata={
            "help": "Unsafe stress-test switch: reward longer hard+incorrect completions. Keep False for the method."
        },
    )

    use_intrinsic_difficulty: bool = field(
        default=True,
        metadata={"help": "Use intrinsic multimodal difficulty in difficulty estimation."},
    )
    passk_ema_alpha: float = field(
        default=0.3,
        metadata={"help": "EMA alpha for pass@k smoothing."},
    )
    # Modality complexity weights (must sum to ~1.0 for interpretability).
    # Default: video=0.55, image=0.25, text=0.15, audio=0.05
    complexity_weight_image: float = field(
        default=0.25,
        metadata={"help": "Weight for image complexity in H_mm."},
    )
    complexity_weight_audio: float = field(
        default=0.05,
        metadata={"help": "Weight for audio complexity in H_mm."},
    )
    complexity_weight_video: float = field(
        default=0.55,
        metadata={"help": "Weight for video complexity in H_mm."},
    )
    complexity_weight_text: float = field(
        default=0.15,
        metadata={"help": "Weight for text complexity in H_mm."},
    )
    require_modality_complexity: bool = field(
        default=True,
        metadata={
            "help": "Fail when a batch lacks a valid four-component modality_complexity mapping instead of silently using 0.5."
        },
    )
    strict_group_generation: bool = field(
        default=True,
        metadata={
            "help": "Fail if gathered generations cannot be partitioned into complete num_generations groups."
        },
    )

    online_filtering: bool = field(
        default=False,
        metadata={
            "help": "Enable online filtering (soft skip). Off by default to preserve the controlled GRPO protocol; "
            "enable explicitly for FAST-style curriculum filtering."
        },
    )
    filter_key: str = field(
        default="accuracy",
        metadata={"help": "Metric key used for filtering: accuracy|difficulty|reward."},
    )
    filter_low: float = field(
        default=0.0,
        metadata={"help": "Lower bound of filtering interval."},
    )
    filter_high: float = field(
        default=1.0,
        metadata={"help": "Upper bound of filtering interval."},
    )
    dynamic_filter_schedule: Optional[list[dict]] = field(
        default=None,
        metadata={"help": "Progressive filter schedule list with epoch_ratio/filter_low/filter_high."},
    )
    replay_ratio: float = field(
        default=0.2,
        metadata={
            "help": "Fraction of hardest groups (per step) to enqueue into the replay buffer, e.g. 0.2 = top 20 percent by "
            "`replay_score_key`."
        },
    )
    replay_capacity: int = field(
        default=4096,
        metadata={"help": "Replay buffer capacity (number of sample_ids)."},
    )
    replay_score_key: str = field(
        default="difficulty",
        metadata={
            "help": "Per-group score for replay admission: difficulty (H_mm * S_ext), accuracy (1-pass@k), or "
            "reward (-mean group reward; lower reward => higher score)."
        },
    )

    # Parameters that control the logging
    log_completions: bool = field(
        default=False,
        metadata={
            "help": "Whether to log a sample of (prompt, completion) pairs every `logging_steps` steps. If `rich` is "
            "installed, it prints the sample. If `wandb` logging is enabled, it logs it to `wandb`."
        },
    )
    log_reward_to_console: bool = field(
        default=False,
        metadata={
            "help": "If True, log aggregate reward-related scalars to the logger at `logging_steps` (main process only)."
        },
    )
    log_reward_breakdown: bool = field(
        default=True,
        metadata={
            "help": "If True, record base/length/pre/post-filter reward means and advantage stats in metrics; "
            "pairs with log_reward_to_console for [reward_calc] lines."
        },
    )
    log_length_reward_details: bool = field(
        default=False,
        metadata={
            "help": "If True, log detailed intermediate stats for length reward computation at logging_steps "
            "(main process only). Useful for debugging hard/easy split, confidence gate, and length_term scale."
        },
    )
    log_length_reward_groups: int = field(
        default=0,
        metadata={
            "help": "If >0 and log_length_reward_details is enabled, additionally log the first N groups' "
            "sample_id, group_correct_rate, group_difficulty, thr, and hard/easy decision at logging_steps. "
            "Set small (e.g., 1-3) to avoid verbose logs."
        },
    )
    save_only_model: bool = field(
        default=False,
        metadata={
            "help": (
                "When checkpointing, whether to only save the model, or also the optimizer, scheduler & rng state."
                "Note that when this is true, you won't be able to resume training from checkpoint."
                "This enables you to save storage by not storing the optimizer, scheduler & rng state."
                "You can only load the model using from_pretrained with this option set to True."
            )
        },
    )
