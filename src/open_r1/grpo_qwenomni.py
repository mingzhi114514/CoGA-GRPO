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

# import debugpy
# try:
#     # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
#     debugpy.listen(("localhost", 9501))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# except Exception as e:
#     pass
import logging
import os
import re
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional
import pathlib


from PIL import Image
from torch.utils.data import Dataset
from transformers import Qwen2VLForConditionalGeneration

from math_verify import parse, verify
from open_r1.trainer import VLMGRPOTrainer, GRPOConfig
from open_r1.vlm_modules import *
from trl import ModelConfig, ScriptArguments, TrlParser, get_peft_config
from transformers import TrainingArguments
import yaml
import json
import random
import math

import whisper
import librosa
from decord import VideoReader, cpu, AudioReader
import numpy as np

# ----------------------- Fix the flash attention bug in the current version of transformers -----------------------
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLVisionFlashAttention2, apply_rotary_pos_emb_flashatt, flash_attn_varlen_func
import torch
from typing import Tuple
import copy
from qwen_omni_utils import process_mm_info
import av

def check_if_video_has_audio(video_path):
    try:
        container = av.open(video_path)
        audio_streams = [stream for stream in container.streams if stream.type == "audio"]
        if not audio_streams:
            return False
        return True
    except:
        return False



logger = logging.getLogger(__name__)



def custom_forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        q, k, v = self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        # print(111, 222, 333, 444, 555, 666, 777, 888, 999)
        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `rotary_pos_emb` (2D tensor of RoPE theta values), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.54 `rotary_pos_emb` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
            cos = emb.cos().float()
            sin = emb.sin().float()
        else:
            cos, sin = position_embeddings
            # Add this
            cos = cos.to(torch.float)
            sin = sin.to(torch.float)
        q, k = apply_rotary_pos_emb_flashatt(q.unsqueeze(0), k.unsqueeze(0), cos, sin)
        q = q.squeeze(0)
        k = k.squeeze(0)

        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
        attn_output = flash_attn_varlen_func(q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen).reshape(
            seq_length, -1
        )
        attn_output = self.proj(attn_output)
        return attn_output

Qwen2_5_VLVisionFlashAttention2.forward = custom_forward


# ----------------------- Main Script -----------------------
@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format'.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["format", "accuracy", "context", "reasoning"],
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format'"},
    )
    max_pixels: Optional[int] = field(
        default=128 * 28 * 28,
        metadata={
            "help": "Max pixels per image/video frame (must be passed into processor __call__; HF default video cap is 768*28*28 without this)."
        },
    )
    min_pixels: Optional[int] = field(
        default=64 * 28 * 28,
        metadata={"help": "Min pixels for Qwen2.5-Omni vision preprocessing."},
    )
    video_max_frames: int = field(
        default=4,
        metadata={"help": "Max frames sampled per video (passed into qwen_omni_utils fetch_video)."},
    )
    video_reader_fps: float = field(
        default=1.0,
        metadata={"help": "Target FPS for video frame sampling (with max_frames caps total frames)."},
    )
    max_anyres_num: Optional[int] = field(
        default=12,
        metadata={"help": "Maximum number of anyres blocks for the image (for InternVL)"},
    )
    image_root: Optional[str] = field(
        default=None,
        metadata={"help": "Root directory of the image"},
    )
    use_audio_in_video: Optional[bool] = field(
        default=False,
        metadata={"help": "Maximum number of anyres blocks for the image (for InternVL)"},
    )

@dataclass
class GRPOModelConfig(ModelConfig):
    freeze_vision_modules: bool = False
    adapter_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Optional PEFT adapter dir (e.g. SFT checkpoint with adapter_model.safetensors). "
            "Set model_name_or_path to the base model. Implies LoRA weights are loaded from disk instead of random init."
        },
    )




SYSTEM_PROMPT = """You are a helpful assistant. Your primary goal is to deeply analyze and interpret information from available various modalities (image, video, audio, text context) to answer questions with human-like depth and a clear, traceable thought process.

Begin by thoroughly understanding the image, video, audio or other available context information, and then proceed with an in-depth analysis related to the question.

In reasoning, It is encouraged to incorporate self-reflection and verification into your reasoning process. You are encouraged to review the image, video, audio, or other context information to ensure the answer accuracy.

Provide your understanding of the image, video, and audio between the <context> </context> tags, detail the reasoning between the <think> </think> tags, and then give your final answer between the <answer> </answer> tags.
"""
class LazySupervisedDataset(Dataset):

    TYPE_TEMPLATE = {
        "multiple choice": " Please provide only the single option letter (e.g., A, B, C, D, etc.) within the <answer> </answer> tags.",
        "numerical": " Please provide the numerical value (e.g., 42 or 3.14) within the <answer> </answer> tags.",
        "OCR": " Please transcribe text from the image/video clearly and provide your text answer within the <answer> </answer> tags.",
        "free-form": " Please provide your text answer within the <answer> </answer> tags.",
        "regression": " Please provide the numerical value (e.g., 42 or 3.14) within the <answer> </answer> tags.",
        "emer_ov": " Please provide the words to describe emotions within the  <answer> </answer> tags.",
        "emer_ov_mc": " Please provide only the single or multiple option letter (e.g., A for single option or A,E for multi option, etc.) within the <answer> </answer> tags.",
        "judge": " Please answer Yes or No within the <answer> </answer> tags.",


    }

    def __init__(self, data_path: str, script_args: GRPOScriptArguments, question_template: str):
        super(LazySupervisedDataset, self).__init__()
        self.script_args = script_args
        self.list_data_dict = []
        self.question_template = question_template
        self.use_audio_in_video = script_args.use_audio_in_video

        if data_path.endswith(".yaml"):
            with open(data_path, "r") as file:
                yaml_data = yaml.safe_load(file)
                datasets = yaml_data.get("datasets")
                # file should be in the format of:
                # datasets:
                #   - json_path: xxxx1.json
                #     sampling_strategy: first:1000
                #   - json_path: xxxx2.json
                #     sampling_strategy: end:3000
                #   - json_path: xxxx3.json
                #     sampling_strategy: random:999
                #     data_root: xxxx/xx

                for data in datasets:
                    json_path = data.get("json_path")
                    sampling_strategy = data.get("sampling_strategy", "all")
                    sampling_number = None

                    if json_path.endswith(".jsonl"):
                        cur_data_dict = []
                        with open(json_path, "r") as json_file:
                            for line in json_file:
                                cur_data_dict.append(json.loads(line.strip()))
                    elif json_path.endswith(".json"):
                        with open(json_path, "r") as json_file:
                            cur_data_dict = json.load(json_file)
                    else:
                        raise ValueError(f"Unsupported file type: {json_path}")

                    if ":" in sampling_strategy:
                        sampling_strategy, sampling_number = sampling_strategy.split(":")
                        if "%" in sampling_number:
                            sampling_number = math.ceil(int(sampling_number.split("%")[0]) * len(cur_data_dict) / 100)
                        else:
                            sampling_number = int(sampling_number)



                    # Apply the sampling strategy
                    if sampling_strategy == "first" and sampling_number is not None:
                        cur_data_dict = cur_data_dict[:sampling_number]
                    elif sampling_strategy == "end" and sampling_number is not None:
                        cur_data_dict = cur_data_dict[-sampling_number:]
                    elif sampling_strategy == "random" and sampling_number is not None:
                        random.shuffle(cur_data_dict)
                        cur_data_dict = cur_data_dict[:sampling_number]

                    if data.get("data_root", None):
                        for each in cur_data_dict:
                            if "path" in each:
                                if isinstance(each["path"], str):
                                    each["path"] = os.path.join(data["data_root"], each["path"])
                                elif isinstance(each["path"], dict):
                                    for k in each["path"].keys():
                                        each["path"][k] = os.path.join(data["data_root"], each["path"][k])
                    print(f"Loaded {len(cur_data_dict)} samples from {json_path}")
                    self.list_data_dict.extend(cur_data_dict)
        else:
            if data_path.endswith(".jsonl"):
                cur_data_dict = []
                with open(data_path, "r") as json_file:
                    for line in json_file:
                        cur_data_dict.append(json.loads(line.strip()))
            elif data_path.endswith(".json"):
                with open(data_path, "r") as json_file:
                    cur_data_dict = json.load(json_file)
            self.list_data_dict = cur_data_dict

        self.mel_size = 128
        # Online multimodal complexity references (for log-normalization).
        # You can override these via env vars if you want to reshape the distribution.
        self._mm_complexity_refs = {
            "text_chars_ref": int(os.environ.get("MM_TEXT_CHARS_REF", "2000")),
            "audio_seconds_ref": float(os.environ.get("MM_AUDIO_SECONDS_REF", "10")),
            "video_seconds_ref": float(os.environ.get("MM_VIDEO_SECONDS_REF", "30")),
        }

    @staticmethod
    def _clip01(x: float) -> float:
        return float(max(0.0, min(1.0, x)))

    @staticmethod
    def _log_norm(x: float, ref: float) -> float:
        """Map x>=0 smoothly to [0,1), with ``ref`` as the half-saturation point.

        The previous clipped log normalization mapped every value at or above
        ``ref`` to exactly 1.0.  With fixed frame/pixel caps this made almost
        every Stage-2 video indistinguishable.  The rational form preserves
        ordering above the reference and therefore remains calibratable.
        """
        x = max(0.0, float(x))
        ref = max(1e-6, float(ref))
        return float(x / (x + ref))

    @staticmethod
    def _sum_text_chars_from_conversation(conversation) -> int:
        """Best-effort count of text characters in a multimodal chat message list."""
        try:
            total = 0
            for msg in conversation or []:
                if not isinstance(msg, dict):
                    continue
                parts = msg.get("content", [])
                if not isinstance(parts, list):
                    continue
                for part in parts:
                    if isinstance(part, dict) and part.get("type") == "text":
                        t = part.get("text", "")
                        if isinstance(t, str):
                            total += len(t)
            return int(total)
        except Exception:
            return 0

    @staticmethod
    def _safe_image_pixels(image_obj) -> Optional[int]:
        """Return pixels for a path or PIL image; None if unavailable."""
        try:
            if image_obj is None:
                return None
            if isinstance(image_obj, Image.Image):
                w, h = image_obj.size
                return int(w) * int(h)
            if isinstance(image_obj, str) and os.path.isfile(image_obj):
                with Image.open(image_obj) as im:
                    w, h = im.size
                return int(w) * int(h)
        except Exception:
            return None
        return None

    @staticmethod
    def _safe_av_durations_and_pixels(path: str) -> tuple[Optional[float], Optional[int], Optional[int]]:
        """
        For a video/audio container path, return:
        - duration_seconds (best-effort)
        - video_frame_pixels (best-effort from stream metadata)
        - n_frames (best-effort from stream metadata)
        """
        try:
            if not path or not isinstance(path, str) or not os.path.isfile(path):
                return None, None, None
            container = av.open(path)
            vstreams = [s for s in container.streams if s.type == "video"]
            astreams = [s for s in container.streams if s.type == "audio"]

            dur = None
            if getattr(container, "duration", None):
                dur = float(container.duration) / 1e6  # microseconds
            if dur is None and astreams:
                s = astreams[0]
                if getattr(s, "duration", None) and getattr(s, "time_base", None):
                    dur = float(s.duration * s.time_base)
            if dur is None and vstreams:
                s = vstreams[0]
                if getattr(s, "duration", None) and getattr(s, "time_base", None):
                    dur = float(s.duration * s.time_base)

            v_pixels = None
            n_frames = None
            if vstreams:
                vs = vstreams[0]
                w = getattr(vs, "width", None)
                h = getattr(vs, "height", None)
                if w and h:
                    v_pixels = int(w) * int(h)
                n_frames = getattr(vs, "frames", None)
                if isinstance(n_frames, int) and n_frames <= 0:
                    n_frames = None
            container.close()
            return dur, v_pixels, int(n_frames) if n_frames is not None else None
        except Exception:
            return None, None, None

    def _compute_modality_complexity(
        self,
        source: dict,
        conversation,
        images,
        audios,
        videos,
    ) -> dict:
        """
        Online complexity estimation for (image/audio/video/text) in [0,1].
        If dataset already provides `modality_complexity`, we keep it.
        """
        if isinstance(source, dict):
            mc = source.get("modality_complexity")
            if isinstance(mc, dict):
                return mc

        max_pixels = int(getattr(self.script_args, "max_pixels", 12845056) or 12845056)
        video_max_frames = int(getattr(self.script_args, "video_max_frames", 12) or 12)
        text_ref = int(self._mm_complexity_refs["text_chars_ref"])
        aud_ref = float(self._mm_complexity_refs["audio_seconds_ref"])
        video_seconds_ref = float(self._mm_complexity_refs["video_seconds_ref"])

        def _collect_media_paths_from_source() -> tuple[list[str], list[str], list[str]]:
            """
            Prefer raw dataset `source` paths over `process_mm_info()` outputs, because `process_mm_info()`
            may return processed objects (tensors, dicts) rather than filesystem paths.
            Returns (image_paths, audio_paths, video_paths).
            """
            ipaths: list[str] = []
            apaths: list[str] = []
            vpaths: list[str] = []
            try:
                if not isinstance(source, dict):
                    return ipaths, apaths, vpaths
                dt = source.get("data_type", None)
                p = source.get("path", None)
                # Common formats:
                # 1) {"data_type":"video","path":"/abs/xxx.mp4"}
                # 2) {"path":{"image": "...jpg", "audio":"...wav"}} or similar
                if isinstance(p, str):
                    if dt == "video":
                        vpaths.append(p)
                        # If we use audio-in-video, audio may also come from the same container.
                        if bool(getattr(self.script_args, "use_audio_in_video", False)):
                            apaths.append(p)
                    elif dt == "audio":
                        apaths.append(p)
                    elif dt == "image":
                        ipaths.append(p)
                elif isinstance(p, dict):
                    for k, v in p.items():
                        if not isinstance(v, str):
                            continue
                        if k == "image":
                            ipaths.append(v)
                        elif k == "audio":
                            apaths.append(v)
                        elif k == "video":
                            vpaths.append(v)
            except Exception:
                pass
            return ipaths, apaths, vpaths

        src_img_paths, src_audio_paths, src_video_paths = _collect_media_paths_from_source()

        # Text
        text_chars = self._sum_text_chars_from_conversation(conversation)
        c_text = self._log_norm(text_chars, text_ref) if text_chars > 0 else 0.0

        # Image: max pixels among images
        img_pixels = None
        try:
            if isinstance(images, list) and images:
                pix_list = [self._safe_image_pixels(x) for x in images]
                pix_list = [p for p in pix_list if isinstance(p, int) and p > 0]
                img_pixels = max(pix_list) if pix_list else None
            # Fallback to source image paths if `images` are not materialized as PIL/path
            if not img_pixels and src_img_paths:
                pix_list = [self._safe_image_pixels(p) for p in src_img_paths]
                pix_list = [p for p in pix_list if isinstance(p, int) and p > 0]
                img_pixels = max(pix_list) if pix_list else img_pixels
        except Exception:
            img_pixels = None
        c_image = self._log_norm(img_pixels, max_pixels) if img_pixels else 0.0

        # Video: combine source duration and spatial density.  Do not use the
        # fixed processing cap itself as the numerator; doing so collapses all
        # videos to the same score.
        c_video = 0.0
        try:
            v_paths: list[str] = []
            if isinstance(videos, list):
                v_paths = [v for v in videos if isinstance(v, str)]
            elif isinstance(videos, str):
                v_paths = [videos]
            if not v_paths and src_video_paths:
                v_paths = list(src_video_paths)
            if v_paths:
                best = 0.0
                for vp in v_paths:
                    dur, v_pix, n_frames = self._safe_av_durations_and_pixels(vp)
                    if dur is None and n_frames and n_frames > 0:
                        dur = float(n_frames) / max(1e-6, float(getattr(self.script_args, "video_reader_fps", 1.0)))
                    temporal = self._log_norm(dur, video_seconds_ref) if dur and dur > 0 else 0.0
                    spatial = self._log_norm(v_pix, max_pixels) if v_pix and v_pix > 0 else 0.0
                    best = max(best, math.sqrt(max(0.0, temporal * spatial)))
                c_video = best
        except Exception:
            c_video = 0.0

        # Audio: seconds from audio path(s) or video container duration
        c_audio = 0.0
        try:
            secs = None
            a_paths: list[str] = []
            if isinstance(audios, list):
                a_paths = [a for a in audios if isinstance(a, str)]
            elif isinstance(audios, str):
                a_paths = [audios]
            if not a_paths and src_audio_paths:
                a_paths = list(src_audio_paths)
            for ap in a_paths:
                dur, _, _ = self._safe_av_durations_and_pixels(ap)
                if dur:
                    secs = max(secs or 0.0, float(dur))
            # A video's duration is an audio feature only when audio from the
            # video is actually presented to the model.
            if secs is None and bool(getattr(self.script_args, "use_audio_in_video", False)):
                v_paths: list[str] = []
                if isinstance(videos, list):
                    v_paths = [v for v in videos if isinstance(v, str)]
                elif isinstance(videos, str):
                    v_paths = [videos]
                if not v_paths and src_video_paths:
                    v_paths = list(src_video_paths)
                for vp in v_paths:
                    dur, _, _ = self._safe_av_durations_and_pixels(vp)
                    if dur:
                        secs = max(secs or 0.0, float(dur))
            if secs is not None and secs > 0:
                c_audio = self._log_norm(secs, aud_ref)
        except Exception:
            c_audio = 0.0

        return {
            "image": self._clip01(c_image),
            "audio": self._clip01(c_audio),
            "video": self._clip01(c_video),
            "text": self._clip01(c_text),
        }

    def __len__(self):
        return len(self.list_data_dict)




    def _make_conversation_image_and_video(self, example, use_audio_in_video=False):
        if example["problem_type"] == 'multiple choice' or  example["problem_type"] == 'emer_ov_mc':
            question = example['problem'] + " Options:\n"
            for op in example["options"]:
                question += op + "\n"
        else:
            question = example['problem']

        text_prompt =  f"{question}\n" + self.TYPE_TEMPLATE[example['problem_type']]

        def _user_vision_item(data_type: str, path: str):
            item = {"type": data_type, data_type: path}
            if data_type == "video":
                item["max_frames"] = self.script_args.video_max_frames
                item["fps"] = self.script_args.video_reader_fps
            return item

        if use_audio_in_video:
            if isinstance(example['path'], str):
                video_audio_avaliable = check_if_video_has_audio(example['path']) and example['data_type'] == "video"
                if video_audio_avaliable:
                    msg =[{
                            "role": "user",
                            "content": [
                                _user_vision_item(example['data_type'], example['path']),
                                {
                                "type": "audio",
                                "audio": example['path']
                                },
                                {
                                    "type": "text",
                                    "text": f"Here is a {example['data_type']}, with the audio from the video.\n" + text_prompt
                                }
                                ]
                        }]

                else:
                    msg =[{
                            "role": "user",
                            "content": [
                                _user_vision_item(example['data_type'], example['path']),
                                {
                                    "type": "text",
                                    "text": f"Here is the {example['data_type']}, and there is no audio information, you don't need to process the audio.\n" + text_prompt
                                }
                                ]
                        }]
            else:
                msg =[{
                            "role": "user",
                            "content": [
                                {
                                    "type": "image",
                                    "image": example['path']["image"]
                                },
                                {
                                    "type": "audio",
                                    "audio": example['path']["audio"]
                                },
                                {
                                    "type": "text",
                                    "text": f"Here is the image, with the coresponding audio.\n" + text_prompt
                                }
                                ]
                        }]
        else:
            msg =[{
                        "role": "user",
                        "content": [
                            _user_vision_item(example['data_type'], example['path']),
                            {
                                "type": "text",
                                "text": text_prompt
                            }
                            ]
                    }]



        msg.insert(0, {
                        "role": "system",
                        "content": [
                            {
                                "type": "text",
                                "text": SYSTEM_PROMPT
                            }
                            ]
                    })


        return msg

    def __getitem__(self, i):
        # Format into conversation
        num_base_retries = 3
        import traceback

        try:
            return self._get_item(i)
        except Exception as e:
            print(i)
            traceback.print_exc()


        for attempt_idx in range(num_base_retries):
            try:
                sample_idx = random.choice(range(len(self)))
                sample = self._get_item(sample_idx)
                return sample
            except Exception as e:
                # no need to sleep
                traceback.print_exc()
                print(f'[try other #{attempt_idx}] Failed to fetch sample {sample_idx}. Exception:', e)
                pass




    def _get_item(self, i):
        source = self.list_data_dict[i]


        # has_speech = ('audio' in source or 'audio_q' in source)
        # has_image = ('image' in source) or ('video' in source) or ('video_long' in source)
        # print(self.use_audio_in_video)
        if "path" in source:
            conversation  = self._make_conversation_image_and_video(source, use_audio_in_video=self.use_audio_in_video)
            problem_type = source["problem_type"]
            audios, images, videos = process_mm_info(
                conversation, use_audio_in_video=self.use_audio_in_video
            )

        solution = source["solution"]
        # print(conversation, solution)
        # delay tokenizer
        modality_complexity = self._compute_modality_complexity(
            source=source,
            conversation=conversation,
            images=images,
            audios=audios,
            videos=videos,
        )

        return {
            'images': images,
            'audios': audios,
            'videos': videos,
            'conversation': conversation,
            'prompt': conversation,
            'solution': solution,
            "problem_type": problem_type,
            "sample_id": source.get("id", str(i)),
            "modality_complexity": modality_complexity,
        #
        }


def _infer_vlm_module_from_config(model_name_or_path: str):
    """When the path has no model name in it (e.g. .../models), read config.json."""
    cfg_path = os.path.join(model_name_or_path, "config.json")
    if not os.path.isfile(cfg_path):
        return None
    with open(cfg_path, encoding="utf-8") as f:
        cfg = json.load(f)
    mt = (cfg.get("model_type") or "").lower()
    archs = cfg.get("architectures") or []
    arch_blob = " ".join(archs).lower()
    if "qwen2_5_omni" in mt or "qwen2_5_omni" in arch_blob or "qwen2_5omni" in arch_blob or "omni" in mt:
        return QwenOmniModule
    return None


def get_vlm_module(model_name_or_path):
    lower = model_name_or_path.lower()
    if "qwen" in lower and "omni" in lower:
        return QwenOmniModule
    inferred = _infer_vlm_module_from_config(model_name_or_path)
    if inferred is not None:
        return inferred
    raise ValueError(
        f"Unsupported model: {model_name_or_path}. "
        "This release supports Qwen2.5-Omni checkpoints only."
    )

def main(script_args, training_args, model_args):
    # Load the VLM module
    vlm_module_cls = get_vlm_module(model_args.model_name_or_path)
    print("using vlm module:", vlm_module_cls.__name__)

    # Load the reward functions
    reward_funcs_registry = {
        "accuracy": vlm_module_cls.accuracy_reward,
        "format": vlm_module_cls.format_reward,
        "reasoning": vlm_module_cls.patial_reasoning_reward,
        "context": vlm_module_cls.patial_context_reward
    }
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]
    print(script_args.reward_funcs)

    print("reward_funcs:", reward_funcs)
    # import ipdb;ipdb.set_trace()

    # Load the dataset
    dataset = LazySupervisedDataset(script_args.dataset_name, script_args, question_template=vlm_module_cls.get_question_template(task_type="rec"))


    adapter_path = getattr(model_args, "adapter_name_or_path", None)
    if adapter_path and not model_args.use_peft:
        raise ValueError("adapter_name_or_path requires --use_peft true")
    peft_cfg = None if adapter_path else get_peft_config(model_args)

    # Initialize the GRPO trainer
    trainer = VLMGRPOTrainer(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        vlm_module=vlm_module_cls(),
        train_dataset=dataset,
        eval_dataset=None,
        peft_config=peft_cfg,
        adapter_name_or_path=adapter_path,
        freeze_vision_modules=model_args.freeze_vision_modules,
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
        max_anyres_num=script_args.max_anyres_num,
        torch_dtype=model_args.torch_dtype,
        use_audio_in_video=bool(script_args.use_audio_in_video),
    )

    # Train and push the model to the Hub
    # Resume:
    # - Default: if any checkpoint-* exists, resume from the latest (HF/DeepSpeed chooses).
    # - Explicit: export RESUME_FROM_CHECKPOINT=/abs/path/to/checkpoint-300
    #   (or relative to cwd, e.g. output/stage2_human/checkpoint-300) to continue from that step.
    # DeepSpeed ZeRO: resume must use the same GPU/world size as when that checkpoint was saved.
    explicit_resume = (os.environ.get("RESUME_FROM_CHECKPOINT") or "").strip()
    has_ckpt = bool(list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")))
    if explicit_resume:
        trainer.train(resume_from_checkpoint=explicit_resume)
    elif has_ckpt:
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    # if training_args.push_to_hub:
    #     trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, GRPOModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
