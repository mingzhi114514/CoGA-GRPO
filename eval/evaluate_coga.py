import os
import json
import re
import hashlib
import warnings
import logging
from datetime import timedelta
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import torch
import torch.nn.functional as F
import itertools
import random
import gc

import argparse
from typing import Optional
from transformers import Qwen2_5OmniThinkerForConditionalGeneration, Qwen2_5OmniProcessor
from transformers.generation.stopping_criteria import StoppingCriteria, StoppingCriteriaList
from peft import PeftModel
from qwen_omni_utils import process_mm_info
from torch.utils.data import DataLoader, Dataset

import av

logger = logging.getLogger(__name__)


def _is_oom_error(error: BaseException) -> bool:
    if isinstance(error, torch.cuda.OutOfMemoryError):
        return True
    message = str(error).lower()
    return "out of memory" in message or "cuda error: memory allocation" in message


def _recover_from_oom() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _sanitize_openmp_env() -> None:
    """
    Empty or invalid OMP_NUM_THREADS triggers libgomp warnings and can contribute to
    native crashes (SIGSEGV) when decord/ffmpeg run inside DataLoader workers.
    """
    for key in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        raw = os.environ.get(key)
        if raw is None:
            continue
        s = str(raw).strip()
        if not s:
            del os.environ[key]
            continue
        try:
            int(s)
        except ValueError:
            del os.environ[key]


def get_assistant_prefill(cli_value: Optional[str] = None) -> str:
    """
    Match GRPO training (`grpo_trainer._get_assistant_prefill`): optional literal prefix appended to the
    prompt string before the processor runs, so generation continues in-format (e.g. after `<context>`).
    After `generate`, decoded new tokens do not include this prefix; training prepends it back for
    rewards/logging — eval does the same so outputs are not missing the leading blocks.
    """
    s = cli_value
    if s is None or (isinstance(s, str) and not str(s).strip()):
        s = os.environ.get("GRPO_ASSISTANT_PREFILL", "")
    if s is None:
        return ""
    s = str(s).strip()
    if not s:
        return ""
    return s.replace("\\n", "\n").replace("\\t", "\t")


def truncate_at_first_answer_end(text: str, end_tag: str = "</answer>") -> str:
    """Keep text through the first closing </answer> to avoid spurious multi-turn continuations."""
    if not text:
        return text
    i = text.find(end_tag)
    if i == -1:
        return text
    return text[: i + len(end_tag)]


class StopOnSubstrInNewTokens(StoppingCriteria):
    """Stop once ``</answer>`` appears in newly generated tokens (batch size 1 only)."""

    def __init__(self, tokenizer, prompt_token_len: int, stop_substr: str = "</answer>"):
        self.tokenizer = tokenizer
        self.prompt_token_len = int(prompt_token_len)
        self.stop_substr = stop_substr

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if input_ids.shape[0] != 1:
            return False
        new_tokens = input_ids[0, self.prompt_token_len :]
        if new_tokens.numel() == 0:
            return False
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return self.stop_substr in text


def format_question_for_display(raw: dict, dataset_name: str) -> str:
    """Match MyDataset user text (stem only, no system / media preamble)."""
    pt = raw.get("problem_type") or ""
    if pt in ("multiple choice", "emer_ov_mc") and raw.get("options"):
        question = raw["problem"] + " Options:\n"
        for op in raw["options"]:
            question += op + "\n"
    else:
        question = raw.get("problem") or ""
    if dataset_name == "socialiq_trans":
        question = question.split("The question is:\n")[-1]
    return question.strip()

def _log_line_is_tee_noise(line: str) -> bool:
    """Lines tee'd from tqdm / loader between [Prompt] and [Token Usage] must not pollute resume keys."""
    s = line.replace("\r", "").strip()
    if not s:
        return False
    if s.startswith("Processing batches:") or "Loading checkpoint shards:" in s:
        return True
    if s.startswith("qwen-vl-utils"):
        return True
    if "[rank" in s and "]: " in s:
        return True
    return False


def load_done_questions_from_log(log_path: Optional[str]) -> set[str]:
    """
    Best-effort resume: parse a previous tee'd console log and recover which questions were already processed.
    We key by the exact question text printed under the `[Prompt]` block.
    """
    done: set[str] = set()
    if not log_path:
        return done
    p = os.path.abspath(str(log_path))
    if not os.path.isfile(p):
        return done
    try:
        in_prompt = False
        buf: list[str] = []
        with open(p, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if "[Prompt]" in line:
                    in_prompt = True
                    buf = []
                    continue
                if in_prompt:
                    # Prompt block ends at token usage or answer markers
                    if "[Token Usage]" in line or "[Answer]" in line or "[Solution]" in line:
                        q = "".join(buf).strip()
                        if q:
                            done.add(q)
                        in_prompt = False
                        buf = []
                        continue
                    if _log_line_is_tee_noise(line):
                        continue
                    buf.append(line)
        return done
    except Exception:
        return done


def _silence_noisy_warnings_and_logs():
    # HuggingFace/transformers logging can be very chatty in eval.
    try:
        from transformers.utils import logging as hf_logging

        hf_logging.set_verbosity_error()
    except Exception:
        pass

    # Python warnings (FutureWarning / UserWarning) from optional audio backends, etc.
    # If you still see a warning you consider harmless, add a targeted filter below.
    warnings.filterwarnings(
        "ignore",
        message=r".*PySoundFile failed\. Trying audioread instead\..*",
        category=UserWarning,
    )
    # librosa deprecations are common and do not affect evaluation results.
    warnings.filterwarnings("ignore", category=FutureWarning, module=r"librosa(\..*)?")
    # Model/config deprecation / compatibility warnings that are not actionable during eval.
    warnings.filterwarnings(
        "ignore",
        message=r".*Unrecognized keys in `rope_scaling`.*",
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*video processor config saved in `preprocessor\.json` file which is deprecated.*",
    )

    # Logging warnings: keep real errors, drop known noisy lines.
    class _DropKnownNoisyLogs(logging.Filter):
        _DROP_SUBSTRINGS = (
            "System prompt modified, audio output may not work as expected.",
            "qwen-vl-utils using decord to read video.",
            "Unused or unrecognized kwargs: images, return_tensors.",
            "Token indices sequence length is longer than the specified maximum sequence length",
        )

        def filter(self, record: logging.LogRecord) -> bool:
            msg = record.getMessage()
            return not any(s in msg for s in self._DROP_SUBSTRINGS)

    logging.getLogger().addFilter(_DropKnownNoisyLogs())


def check_if_video_has_audio(video_path):
    try:
        container = av.open(video_path)
        audio_streams = [stream for stream in container.streams if stream.type == "audio"]
        if not audio_streams:
            return False
        return True
    except Exception:
        return False


def extract_think(output_str):
    pattern = r'<think>\s*(.*?)\s*</think>'
    match = re.search(pattern, output_str, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""

def extract_answer(text):
    pattern = r'<answer>\s*(.*?)\s*</answer>'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


def build_stable_sample_id(dataset_name: str, raw: dict, index: int) -> str:
    """Return a stable ID that distinguishes repeated question text across videos."""
    for key in ("sample_id", "id", "qid", "question_id", "uid"):
        value = raw.get(key)
        if value is not None and str(value).strip():
            return f"{dataset_name}:{value}"
    media = raw.get("video", raw.get("path", ""))
    payload = json.dumps(
        {
            "dataset": dataset_name,
            "index": int(index),
            "media": media,
            "problem": raw.get("problem", ""),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return f"{dataset_name}:{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:20]}"


def binary_correct_fn(prediction: str, target: str, question_type: str) -> int:
    """Strict 0/1 correctness, kept separate from partial-credit reward."""
    pred = str(prediction or "").strip()
    tgt = str(target or "").strip()
    if question_type == "emer_ov_mc":
        pred_set = {x.strip().upper() for x in pred.split(",") if x.strip()}
        tgt_set = {x.strip().upper() for x in tgt.split(",") if x.strip()}
        return int(bool(tgt_set) and pred_set == tgt_set)
    if question_type in ("multiple choice", "judge"):
        return int(pred.upper() == tgt.upper())
    if question_type in ("numerical", "regression"):
        pnum, tnum = normalize_number(pred), normalize_number(tgt)
        if pnum is None or tnum is None:
            return 0
        return int(abs(pnum - tnum) <= 1e-6 * max(1.0, abs(tnum)))
    return int(pred.casefold() == tgt.casefold())

def normalize_number(num_str):
    try:
        num_str = num_str.replace(',', '')
        return float(num_str)
    except Exception as e:
        return None

def mean_relative_accuracy(pred, target, start=0.5, end=0.95, interval=0.05):

    if not torch.is_tensor(pred):
        pred = torch.tensor(pred, dtype=torch.float32)
    if not torch.is_tensor(target):
        target = torch.tensor(target, dtype=torch.float32)

    epsilon = 1e-8
    rel_error = torch.abs(pred - target) / (torch.abs(target) + epsilon)

    thresholds = torch.arange(start, end + interval/2, interval, dtype=torch.float32)

    conditions = rel_error < (1 - thresholds)
    mra = conditions.float().mean()
    return mra.item()


def emer_ov_mc(reference, hypothesis):
    list_a = [x.strip() for x in reference.split(",") if x.strip()]
    list_b = [x.strip() for x in hypothesis.split(",") if x.strip()]
    true_positive = len(set(list_a) & set(list_b))
    precision = true_positive / len(list_a) if list_a else 0.0
    recall = true_positive / len(list_b) if list_b else 0.0
    if precision + recall > 0:
        f1_score = 2 * (precision * recall) / (precision + recall)
    else:
        f1_score = 0

    return f1_score

def judge(reference, hypothesis):
    if "yes" in reference.lower()  and "yes" in hypothesis.lower():
        return 1
    elif "no" in reference.lower()  and "no" in hypothesis.lower():
        return 1
    else:
        return 0

def reward_fn(output_ans, gt_ans, question_type):
    try:

        if question_type == "multiple choice":
            return 1.0 if output_ans.strip() == gt_ans.strip() else 0.0
        elif question_type == "numerical":
            gt_has_decimal = ("." in gt_ans) or ("," in gt_ans)
            out_has_decimal = ("." in output_ans) or ("," in output_ans)
            if gt_has_decimal != out_has_decimal:
                return 0.0
            gt_number = normalize_number(gt_ans)
            out_number = normalize_number(output_ans)
            if gt_number is None or out_number is None:
                return 0.0
            return 1.0 if round(gt_number, 2) == round(out_number, 2) else 0.0
        elif question_type == "regression":
            gt_number = normalize_number(gt_ans)
            out_number = normalize_number(output_ans)
            if gt_number is None or out_number is None:
                return 0.0
            mra = mean_relative_accuracy(out_number, gt_number)
            return mra
        elif question_type == "emer_ov_mc":
            return emer_ov_mc(output_ans, gt_ans)
        elif  question_type == "judge":
            return judge(output_ans, gt_ans)

        else:
            return 0.0
    except Exception as e:
        return 0.0




SYSTEM_PROMPT = """You are a helpful assistant. Your primary goal is to deeply analyze and interpret information from available various modalities (image, video, audio, text context) to answer questions with human-like depth and a clear, traceable thought process.

Begin by thoroughly understanding the image, video, audio or other available context information, and then proceed with an in-depth analysis related to the question.

In reasoning, It is encouraged to incorporate self-reflection and verification into your reasoning process. You are encouraged to review the image, video, audio, or other context information to ensure the answer accuracy.

Provide your understanding of the image, video, and audio between the <context> </context> tags, detail the reasoning between the <think> </think> tags, and then give your final answer between the <answer> </answer> tags.
"""


def flatten_worldsense_qa(raw: dict) -> list:
    """
    Convert HuggingFace WorldSense `worldsense_qa.json` (nested dict keyed by video_id)
    into the flat list format expected by `MyDataset` (same shape as `qa_think.json`).
    Video path defaults to ``{video_id}.mp4`` under `video_root` (adjust filenames if your rip differs).
    """
    rows: list = []
    for vid in sorted(raw.keys()):
        obj = raw[vid]
        if not isinstance(obj, dict):
            continue
        video_rel = f"{vid}.mp4"
        for key in sorted(obj.keys()):
            if not str(key).startswith("task"):
                continue
            t = obj[key]
            if not isinstance(t, dict):
                continue
            q = t.get("question")
            candidates = t.get("candidates")
            ans = t.get("answer")
            if not q or not candidates or ans is None:
                continue
            rows.append(
                {
                    "problem": q,
                    "problem_type": "multiple choice",
                    "options": list(candidates),
                    "solution": f"<answer>{str(ans).strip()}</answer>",
                    "data_type": "video",
                    "video": video_rel,
                }
            )
    return rows


def normalize_daily_omni_qa_think(rows: list) -> list:
    """
    Daily-Omni HuggingFace `qa_think.json` uses Question / Choice / Answer / video_id;
    `MyDataset` expects problem / options / problem_type / solution / data_type / video.
    """
    out: list = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        if row.get("problem") and row.get("problem_type"):
            out.append(row)
            continue
        q = row.get("Question") or row.get("question")
        choices = row.get("Choice") or row.get("choice")
        ans = row.get("Answer") or row.get("answer")
        vid = row.get("video_id") or row.get("video")
        if not q or choices is None or ans is None or not vid:
            continue
        vs = str(vid).strip()
        video_rel = vs if vs.lower().endswith(".mp4") else f"{vs}.mp4"
        merged = dict(row)
        merged["problem"] = q
        merged["problem_type"] = "multiple choice"
        merged["options"] = list(choices)
        merged["solution"] = f"<answer>{str(ans).strip()}</answer>"
        merged["data_type"] = "video"
        merged["video"] = video_rel
        out.append(merged)
    if rows and not out:
        raise ValueError(
            "Daily-Omni qa_think.json: expected list of objects with Question/Choice/Answer/video_id "
            "(or already-normalized problem/problem_type), but normalization produced 0 samples."
        )
    return out


_DAILY_NESTED_VIDEO_EXTS = (".mp4", ".mkv", ".webm", ".avi", ".mov")


def resolve_daily_omni_video_paths(rows: list, video_root: str) -> list:
    """
    Many Daily-Omni mirrors use ``videos/<video_id>/`` folders instead of flat ``<video_id>.mp4``.
    After ``normalize_daily_omni_qa_think`` (default ``{video_id}.mp4``), rewrite ``row["video"]`` to the
    first relative path under ``video_root`` that exists: nested common names, else first video file in the folder.
    """
    root = os.path.abspath(video_root)
    for row in rows:
        if not isinstance(row, dict):
            continue
        rel = row.get("video") or row.get("path")
        if not rel:
            continue
        rel = str(rel).strip().replace("\\", "/")
        if os.path.isfile(os.path.join(root, rel)):
            row["video"] = rel
            continue
        vid = row.get("video_id")
        if vid is not None and str(vid).strip():
            vid = str(vid).strip()
        else:
            base = os.path.basename(rel)
            vid = os.path.splitext(base)[0]
        if not vid:
            continue
        found: Optional[str] = None
        for c in (
            f"{vid}.mp4",
            f"{vid}/{vid}.mp4",
            f"{vid}/video.mp4",
        ):
            if os.path.isfile(os.path.join(root, c.replace("/", os.sep))):
                found = c
                break
        if found is None:
            sub = os.path.join(root, vid)
            if os.path.isdir(sub):
                try:
                    entries = sorted(os.scandir(sub), key=lambda e: e.name.lower())
                except OSError:
                    entries = []
                for ent in entries:
                    if ent.is_file() and ent.name.lower().endswith(_DAILY_NESTED_VIDEO_EXTS):
                        found = f"{vid}/{ent.name}"
                        break
        if found:
            row["video"] = found
    return rows


def _row_video_abs_path(row: dict, video_root: str) -> str:
    rel = row.get("video") or row.get("path")
    if not rel:
        return ""
    return os.path.join(video_root, str(rel))


def filter_or_fail_on_missing_videos(
    rows: list,
    video_root: str,
    *,
    skip_missing: bool,
) -> tuple[list, int]:
    """
    Video benchmarks expect files under ``video_root``. Missing files cause decord to fail and
    qwen_omni_utils' torchvision fallback to raise KeyError: 'video_fps' on empty reads — fail early
    with a clear error, or optionally drop those rows.
    Returns ``(rows_out, n_dropped)`` where ``n_dropped`` is the number of rows removed when skipping.
    """
    vr = os.path.abspath(video_root)
    missing_examples: list[str] = []
    n_bad = 0
    for row in rows:
        full = _row_video_abs_path(row, video_root)
        ok = bool(full) and os.path.isfile(full)
        if ok:
            continue
        n_bad += 1
        if len(missing_examples) < 5:
            missing_examples.append(full if full else "<row without video/path>")

    if n_bad == 0:
        return rows, 0
    if not skip_missing:
        raise FileNotFoundError(
            f"{n_bad} / {len(rows)} samples reference missing videos under {vr!r}.\n"
            f"Example paths:\n  " + "\n  ".join(missing_examples)
            + "\nDownload the benchmark videos or use `--skip-missing-video-samples` "
            "to evaluate only samples with available videos."
        )
    kept = [row for row in rows if _row_video_abs_path(row, video_root) and os.path.isfile(_row_video_abs_path(row, video_root))]
    n_drop = len(rows) - len(kept)
    if not kept:
        raise ValueError(
            f"No samples remain under {vr!r} after skipping {n_bad} missing or invalid video paths."
        )
    return kept, n_drop


class MyDataset(Dataset):
    def __init__(
        self,
        dataset_name,
        data_path,
        processor,
        video_root,
        skip_missing_video_samples: bool = False,
        max_samples: int = 0,
        eval_seed: int = 42,
    ):
        super(MyDataset, self).__init__()
        self.dataset_name = dataset_name
        self.video_root = video_root

        if data_path.endswith(".jsonl"):
            data = []
            with open(data_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        data.append(json.loads(line))
        elif data_path.endswith(".json"):
            with open(data_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            raise ValueError("Input file must be .json or .jsonl")

        if isinstance(data, dict):
            if dataset_name == "world":
                data = flatten_worldsense_qa(data)
                if not data:
                    raise ValueError(
                        "WorldSense JSON loaded as dict but flattened to 0 samples. "
                        "Expected worldsense_qa.json-style nested tasks or a flat qa_think.json list."
                    )
            else:
                raise ValueError(
                    f"Dataset {dataset_name!r}: JSON root must be a list, got dict "
                    f"(unless world + worldsense_qa nested format)."
                )

        if dataset_name == "daily":
            data = normalize_daily_omni_qa_think(data)
            data = resolve_daily_omni_video_paths(data, video_root)

        self._full_dataset_size = len(data)
        self._sampled_indices = list(range(len(data)))
        max_samples = int(max_samples or 0)
        if max_samples < 0:
            raise ValueError(f"--max-samples must be >= 0, got {max_samples}")
        if max_samples and max_samples < len(data):
            rng = random.Random(int(eval_seed))
            self._sampled_indices = sorted(rng.sample(range(len(data)), max_samples))
            data = [data[i] for i in self._sampled_indices]
        self._sampling_seed = int(eval_seed)

        self.processor = processor



        self.TYPE_TEMPLATE = {
            "multiple choice": " Please provide only the single option letter (e.g., A, B, C, D, etc.) within the <answer> </answer> tags.",
            "numerical": " Please provide the numerical value (e.g., 42 or 3.14) within the <answer> </answer> tags.",
            "OCR": " Please transcribe text from the image/video clearly and provide your text answer within the <answer> </answer> tags.",
            "free-form": " Please provide your text answer within the <answer> </answer> tags.",
            "regression": " Please provide the numerical value (e.g., 42 or 3.14) within the <answer> </answer> tags.",
            "emer_ov_mc": " Please provide only the single or multiple option letter (e.g., A for single option or A,E for multi option, etc.) within the <answer> </answer> tags.",
            "judge": " Please answer Yes or No within the <answer> </answer> tags.",


        }



        if dataset_name in DEFAULT_EVAL_DATASETS:
            data, dropped = filter_or_fail_on_missing_videos(
                data,
                video_root,
                skip_missing=bool(skip_missing_video_samples),
            )
            self._video_rows_dropped_missing = dropped
        else:
            self._video_rows_dropped_missing = 0
        self.data = data

    def __getitem__(self, index):
            data = self.data[index]

            if data["problem_type"] == 'multiple choice' or  data["problem_type"] == 'emer_ov_mc' :
                question = data['problem'] + " Options:\n"
                for op in data["options"]:
                    question += op + "\n"
            else:
                question = data['problem']

            if "socialiq_trans" == self.dataset_name:
                question = question.split("The question is:\n")[-1]

            video_path = data['video'] if "video" in data else data["path"]
            video_path = os.path.join(self.video_root, video_path)
            video_audio_avaliable = check_if_video_has_audio(video_path)


            # text_prompt = self.QUESTION_TEMPLATE.format(Question=question) + self.TYPE_TEMPLATE[data['problem_type']]
            text_prompt =  question + self.TYPE_TEMPLATE[data['problem_type']]
            if video_audio_avaliable:
                message = [{
                    "role": "user",
                    "content": [
                        {
                            "type": data['data_type'],
                            data['data_type']: video_path
                        },
                        {
                            "type": "audio",
                            "audio": video_path
                        },

                        {
                            "type": "text",
                            "text": f"Here is a {data['data_type']}, with the audio from the video.\n" + text_prompt
                        }
                    ]
                }]
            else:
                 message = [{
                    "role": "user",
                    "content": [
                        {
                            "type": data['data_type'],
                            data['data_type']: video_path
                        },

                        {
                            "type": "text",
                            "text": text_prompt
                        }
                    ]
                }]

            message.insert(0, {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": SYSTEM_PROMPT
                    }
                    ]
            })


            audios, images, videos = process_mm_info(message, use_audio_in_video=False)
            # inputs = self.processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=True)
            data_dict = {
                'images': images,
                'audios': audios,
                'videos': videos,
                'prompt': message,
                'solution': data["solution"],
                "problem_type": data["problem_type"],
                "raw": data,
                "dataset_index": int(index),
                "sample_id": build_stable_sample_id(self.dataset_name, data, index),
                "media_path": video_path,
            }
            # import ipdb; ipdb.set_trace()
            return data_dict

    def __len__(self):
        return len(self.data)

def collate_fn(examples):


    return examples

class InferenceSampler(torch.utils.data.sampler.Sampler):

    def __init__(self, size):
        self._size = int(size)
        assert size > 0
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            self._rank = torch.distributed.get_rank()
            self._world_size = torch.distributed.get_world_size()
        else:
            self._rank = 0
            self._world_size = 1
        self._local_indices = self._get_local_indices(size, self._world_size,
                                                      self._rank)

    @staticmethod
    def _get_local_indices(total_size, world_size, rank):
        # shard_size = total_size // world_size
        # left = total_size % world_size
        # shard_sizes = [shard_size + int(r < left) for r in range(world_size)]

        # begin = sum(shard_sizes[:rank])
        # end = min(sum(shard_sizes[:rank + 1]), total_size)
        # return range(begin, end)
        indices = []
        for i in range(total_size):
            if i % world_size == rank:
                indices.append(i)
        return indices

    def __iter__(self):
        yield from self._local_indices

    def __len__(self):
        return len(self._local_indices)


# Paths are relative to --data-root (default: current working directory).
DEFAULT_EVAL_DATASETS = {
    "daily": {"gt_path": "Daily-Omni/qa_think.json", "video_root": "Daily-Omni/videos"},
    "world": {"gt_path": "WorldSense/qa_think.json", "video_root": "WorldSense/videos"},
    "ib": {"gt_path": "IntentBench/qa.json", "video_root": "IntentBench/videos"},
}



def main(args):
    _sanitize_openmp_env()
    _silence_noisy_warnings_and_logs()

    file_name = args.file_name

    if not torch.distributed.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29502")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("LOCAL_RANK", "0")
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        # NCCL barrier/all_gather use the store; default ~10min is too short when ranks finish
        # the eval loop at very different times (straggler still in generate()).
        _pg_timeout_min = int(os.environ.get("TORCH_DIST_TIMEOUT_MIN", "120"))
        torch.distributed.init_process_group(
            backend=backend,
            init_method="env://",
            timeout=timedelta(minutes=max(10, _pg_timeout_min)),
        )

    if torch.cuda.is_available():
        torch.cuda.set_device(int(os.getenv("LOCAL_RANK", "0")))




    # Vision tower: `sdpa` uses a dense bool mask [1,S,S] in Qwen2.5-Omni vision blocks → O(S^2) VRAM and
    # often OOMs on long videos even when pixel budgets look modest. Prefer FlashAttention2 (varlen) on CUDA.
    attn_impl = getattr(args, "attn_implementation", None) or "auto"
    if attn_impl == "auto":
        attn_impl = "flash_attention_2" if torch.cuda.is_available() else "sdpa"
    device_map = getattr(args, "device_map", None) or "auto"
    # Work around a transformers tensor-parallel init path that can crash when device_type=="auto".
    if device_map == "auto":
        device_map = "cuda" if torch.cuda.is_available() else "cpu"

    def _load_thinker(attn: str):
        return Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
            attn_implementation=attn,
        )

    try:
        base = _load_thinker(attn_impl)
    except Exception as e:
        if attn_impl == "flash_attention_2":
            logger.warning("flash_attention_2 load failed (%s); falling back to sdpa (may OOM on long videos).", e)
            base = _load_thinker("sdpa")
        else:
            raise
    if args.adapter_path:
        model = PeftModel.from_pretrained(base, args.adapter_path)
    else:
        model = base
    # Ensure we have a concrete device for moving inputs.
    if torch.cuda.is_available():
        try:
            model = model.to("cuda")
        except Exception:
            pass
    processor = Qwen2_5OmniProcessor.from_pretrained(args.model_path)
    tokenizer = getattr(processor, "tokenizer", None) or processor

    def _processor_call_with_limits(*, text, audios, images, videos):
        """
        Qwen2.5-Omni merges kwargs via `_merge_kwargs`; top-level `max_pixels` may NOT reach
        `Qwen2VLVideoProcessor`. Always pass `videos_kwargs` / `images_kwargs` explicitly.
        """
        videos_kwargs = {}
        if getattr(args, "max_pixels", None) is not None:
            videos_kwargs["max_pixels"] = int(args.max_pixels)
        if getattr(args, "min_pixels", None) is not None:
            videos_kwargs["min_pixels"] = int(args.min_pixels)
        # HF path: `fps` is passed to the video reader. Smaller fps => fewer decoded frames for a fixed clip.
        # `--video-max-frames` is only a coarse cap: fps_eff = min(user_fps, 8/max(1, video_max_frames)).
        base_fps = float(getattr(args, "video_reader_fps", 1.0) or 1.0)
        vmf = int(getattr(args, "video_max_frames", 0) or 0)
        if vmf > 0:
            cap_fps = 8.0 / max(1, vmf)
            videos_kwargs["fps"] = min(base_fps, max(0.125, cap_fps))
        else:
            videos_kwargs["fps"] = base_fps

        images_kwargs = {}
        if getattr(args, "max_pixels", None) is not None:
            images_kwargs["max_pixels"] = int(args.max_pixels)
        if getattr(args, "min_pixels", None) is not None:
            images_kwargs["min_pixels"] = int(args.min_pixels)

        try:
            return processor(
                text=text,
                audio=audios,
                images=images,
                videos=videos,
                return_tensors="pt",
                padding=True,
                use_audio_in_video=False,
                videos_kwargs=videos_kwargs,
                images_kwargs=images_kwargs,
            )
        except TypeError:
            return processor(
                text=text,
                audio=audios,
                images=images,
                videos=videos,
                return_tensors="pt",
                padding=True,
                use_audio_in_video=False,
            )



    data_root = getattr(args, "data_root", None) or os.getcwd()
    data_root = os.path.abspath(data_root)

    for dataset_name in args.dataset:
        # model_name = args.model_path.split("/")[-1]
        print(file_name)
        output_dir = getattr(args, "output_dir", None) or os.path.join(data_root, "eval_results")
        os.makedirs(output_dir, exist_ok=True)
        OUTPUT_PATH = os.path.join(output_dir, f"{dataset_name}_{file_name}.json")
        cfg = DEFAULT_EVAL_DATASETS[dataset_name]
        gt_path = os.path.join(data_root, cfg["gt_path"])
        video_root = os.path.join(data_root, cfg["video_root"])
        if not os.path.isfile(gt_path):
            # Official layout uses WorldSense/qa_think.json; many HF downloads only ship worldsense_qa.json.
            if dataset_name == "world":
                alt = os.path.join(data_root, "WorldSense", "worldsense_qa.json")
                if os.path.isfile(alt):
                    gt_path = alt
                else:
                    raise FileNotFoundError(
                        f"Annotation file not found: {gt_path}\n"
                        f"Also tried fallback: {alt}\n"
                        f"Download WorldSense from README links or place qa_think.json / worldsense_qa.json under WorldSense/."
                    )
            elif dataset_name == "daily":
                # Some mirrors / local copies use the misspelled folder name `Daliy-Omni/`.
                alt_gt = os.path.join(data_root, "Daliy-Omni", "qa_think.json")
                alt_video = os.path.join(data_root, "Daliy-Omni", "videos")
                if os.path.isfile(alt_gt):
                    gt_path = alt_gt
                    video_root = alt_video
                else:
                    raise FileNotFoundError(
                        f"Annotation file not found: {gt_path}\n"
                        f"Also tried fallback: {alt_gt}\n"
                        f"Set --data-root to the folder that contains Daily-Omni/qa_think.json (or Daliy-Omni/qa_think.json)."
                    )
            else:
                raise FileNotFoundError(
                    f"Annotation file not found: {gt_path}\n"
                    f"Set --data-root to the folder that contains {cfg['gt_path']} (see README / benchmark layout)."
                )
        dataset = MyDataset(
            dataset_name,
            gt_path,
            processor,
            video_root,
            skip_missing_video_samples=bool(getattr(args, "skip_missing_video_samples", False)),
            max_samples=int(getattr(args, "max_samples", 0) or 0),
            eval_seed=int(getattr(args, "eval_seed", 42)),
        )
        if torch.distributed.get_rank() == 0:
            print(
                f"[sampling] dataset={dataset_name} full={dataset._full_dataset_size} "
                f"selected={len(dataset)} max_samples={int(getattr(args, 'max_samples', 0) or 0)} "
                f"seed={dataset._sampling_seed}"
            )
        if torch.distributed.get_rank() == 0 and getattr(dataset, "_video_rows_dropped_missing", 0) > 0:
            print(
                f"[videos] skipped {dataset._video_rows_dropped_missing} samples with missing videos "
                f"under {video_root}; {len(dataset)} samples remain."
            )
        # problem_type = dataset[0]["problem_type"]
        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            sampler=InferenceSampler(len(dataset)),
            batch_size=1,
            num_workers=int(getattr(args, "num_workers", 2)),
            pin_memory=True,
            drop_last=False,
            collate_fn=collate_fn,
        )

        # import pdb; pdb.set_trace()

        final_output = []
        mean_acc = []
        mean_mra = []

        gts = []
        sources = []
        rets = []
        prompt_token_lens = []
        gen_token_lens = []
        oom_skips = []
        done_questions = load_done_questions_from_log(getattr(args, "resume_log", None))
        if done_questions and torch.distributed.get_rank() == 0:
            print(f"[resume] loaded {len(done_questions)} done prompts from log: {args.resume_log}")

        # if os.path.exists(OUTPUT_PATH):
        #     continue


        for inputs in tqdm(dataloader, desc="Processing batches"):
            # Resume: must skip before processor/encode/generate (batch_size is 1 for this eval).
            if inputs:
                try:
                    q_skip = format_question_for_display(inputs[0]["raw"], dataset_name)
                except Exception:
                    q_skip = ""
                if q_skip and q_skip in done_questions:
                    continue

            # import ipdb; ipdb.set_trace()

            images, videos, audios, prompts = [], [], [], []
            for each in inputs:
                prompts.append(each["prompt"])
                if each["images"] is not None:
                    images.extend(each["images"])
                if each["audios"] is not None:
                    audios.extend(each["audios"])
                if each["videos"] is not None:
                    videos.extend(each["videos"])
            if len(images) == 0: images = None
            if len(audios) == 0: audios = None
            if len(videos) == 0: videos = None

            text = processor.apply_chat_template(
                prompts,
                tokenize=False,
                add_generation_prompt=True,
            )
            prefill = get_assistant_prefill(getattr(args, "assistant_prefill", None))
            if prefill:
                # `apply_chat_template` may return str or list[str] depending on processor / batch shape.
                if isinstance(text, list):
                    text = [t + prefill for t in text]
                else:
                    text = text + prefill
            # print(text)
            try:
                model_inputs = _processor_call_with_limits(
                    text=text,
                    audios=audios,
                    images=images,
                    videos=videos,
                )
            except Exception as e:
                if bool(getattr(args, "skip_oom", True)) and _is_oom_error(e):
                    try:
                        question = format_question_for_display(inputs[0]["raw"], dataset_name)
                    except Exception:
                        question = "<unknown>"
                    oom_skips.append({"phase": "preprocess", "question": question})
                    _recover_from_oom()
                    print(f"[skip_oom] OOM during preprocessing; skipped. question={question!r}")
                    continue
                raise

            # VRAM control must happen inside `processor(...)` via `videos_kwargs` (so video_grid_thw,
            # pixel_values_videos, and input_ids stay aligned). Do not post-hoc edit video tensors alone.

            # Move inputs to device; optionally skip OOM samples instead of aborting eval.
            try:
                model_inputs = model_inputs.to(model.device).to(model.dtype)
            except Exception as e:
                if bool(getattr(args, "skip_oom", True)) and _is_oom_error(e):
                    if torch.distributed.get_rank() == 0:
                        try:
                            q0 = format_question_for_display(inputs[0]["raw"], dataset_name)
                        except Exception:
                            q0 = "<unknown>"
                    else:
                        q0 = "<hidden-on-worker>"
                    oom_skips.append({"phase": "device_transfer", "question": q0})
                    _recover_from_oom()
                    if torch.distributed.get_rank() == 0:
                        print(f"[skip_oom] OOM while moving inputs; skipped. question={q0!r}")
                    continue
                raise
            # Guard against overlong sequences that exceed the model context window.
            # Keep the most recent tokens (right side) to preserve the question.
            max_ctx = int(getattr(getattr(model, "config", None), "max_position_embeddings", 0) or 0)
            max_input_tokens = int(getattr(args, "max_input_tokens", 0) or 0)
            if hasattr(model_inputs, "input_ids"):
                seq_len = int(model_inputs.input_ids.shape[1])
                # First, respect the model's context window if known.
                if max_ctx > 0 and seq_len > max_ctx:
                    overflow = seq_len - max_ctx
                    model_inputs.input_ids = model_inputs.input_ids[:, overflow:]
                    if hasattr(model_inputs, "attention_mask") and model_inputs.attention_mask is not None:
                        model_inputs.attention_mask = model_inputs.attention_mask[:, overflow:]
                    seq_len = int(model_inputs.input_ids.shape[1])
                # Second, optional hard cap (training often uses a large max_prompt_length; truncating
                # here drops the *start* of the prompt and can make outputs look "cut from the front").
                if max_input_tokens > 0 and seq_len > max_input_tokens:
                    overflow = seq_len - max_input_tokens
                    model_inputs.input_ids = model_inputs.input_ids[:, overflow:]
                    if hasattr(model_inputs, "attention_mask") and model_inputs.attention_mask is not None:
                        model_inputs.attention_mask = model_inputs.attention_mask[:, overflow:]
            stopping_criteria = None
            if getattr(args, "stop_after_first_answer", True) and hasattr(model_inputs, "input_ids"):
                bs = int(model_inputs.input_ids.shape[0])
                if bs == 1:
                    plen = int(model_inputs.input_ids.shape[1])
                    tok = getattr(processor, "tokenizer", None) or processor
                    stopping_criteria = StoppingCriteriaList(
                        [StopOnSubstrInNewTokens(tok, plen)]
                    )
            try:
                with torch.inference_mode():
                    gen_kw = dict(
                        **model_inputs,
                        use_audio_in_video=False,
                        max_new_tokens=int(getattr(args, "max_new_tokens", 4096)),
                        stopping_criteria=stopping_criteria,
                    )
                    rp = float(getattr(args, "repetition_penalty", 1.0))
                    if rp != 1.0:
                        gen_kw["repetition_penalty"] = rp
                    if getattr(args, "do_sample", False):
                        gen_kw["do_sample"] = True
                        gen_kw["temperature"] = float(getattr(args, "temperature", 0.7))
                        gen_kw["top_p"] = float(getattr(args, "top_p", 0.92))
                    else:
                        gen_kw["do_sample"] = False
                    text_ids = model.generate(**gen_kw)
            except Exception as e:
                if bool(getattr(args, "skip_oom", True)) and _is_oom_error(e):
                    if torch.distributed.get_rank() == 0:
                        try:
                            q0 = format_question_for_display(inputs[0]["raw"], dataset_name)
                        except Exception:
                            q0 = "<unknown>"
                    else:
                        q0 = "<hidden-on-worker>"
                    oom_skips.append({"phase": "generation", "question": q0})
                    del model_inputs
                    _recover_from_oom()
                    if torch.distributed.get_rank() == 0:
                        print(f"[skip_oom] OOM during generate(); skipped. question={q0!r}")
                    continue
                raise


            for j, (sample, model_output, input_text) in enumerate(zip(inputs, text_ids, model_inputs.input_ids)):
                # Resume support: skip prompts already present in previous log.
                try:
                    qtext = format_question_for_display(sample["raw"], dataset_name)
                except Exception:
                    qtext = ""
                if qtext and qtext in done_questions:
                    continue
                # Token lengths (local; gathered across ranks later).
                try:
                    prompt_len = int(input_text.size(0))
                except Exception:
                    prompt_len = 0
                try:
                    gen_len = int(model_output.size(0) - input_text.size(0))
                    gen_len = max(0, gen_len)
                except Exception:
                    gen_len = 0
                prompt_token_lens.append(prompt_len)
                gen_token_lens.append(gen_len)
                # Per-sample token usage (new tokens only = model generation length).
                sample["output_token_count"] = int(gen_len)
                sample["prompt_token_count"] = int(prompt_len)

                raw_completion = processor.decode(
                    model_output[input_text.size(0):],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                # Same as training: prefill is in the prompt, not in generated token ids — prepend for full text.
                completion_text = (prefill + raw_completion) if prefill else raw_completion
                if getattr(args, "truncate_at_first_answer", True):
                    completion_text = truncate_at_first_answer_end(completion_text)
                if getattr(args, "qa_output", True):
                    sep = "-" * 52
                    print(
                        f"\n{sep}\n[Prompt]\n{qtext}\n\n"
                        f"[Token Usage] Generated={gen_len}, Prompt={prompt_len}\n\n"
                        f"[Answer]\n{completion_text}\n\n[Solution]\n{sample['solution']}\n{sep}"
                    )
                else:
                    print(completion_text, sample["solution"])
                sample.pop("images")
                sample.pop("videos")
                sample.pop("audios")

                # print(completion_text)
                # print(sample)
                rets.append(completion_text)
                gts.append(sample["solution"])
                sources.append(sample)





        if torch.distributed.is_available() and torch.distributed.get_backend() == "nccl" and torch.cuda.is_available():
            try:
                torch.distributed.barrier(
                    device_ids=[int(os.environ.get("LOCAL_RANK", "0"))],
                )
            except TypeError:
                torch.distributed.barrier()
        else:
            torch.distributed.barrier()

        world_size = torch.distributed.get_world_size()
        merged_gts = [None for _ in range(world_size)]
        merged_sources = [None for _ in range(world_size)]
        merged_responses = [None for _ in range(world_size)]
        merged_prompt_token_lens = [None for _ in range(world_size)]
        merged_gen_token_lens = [None for _ in range(world_size)]
        merged_oom_skips = [None for _ in range(world_size)]
        torch.distributed.all_gather_object(merged_gts, gts)
        torch.distributed.all_gather_object(merged_sources, sources)
        torch.distributed.all_gather_object(merged_responses, rets)
        torch.distributed.all_gather_object(merged_prompt_token_lens, prompt_token_lens)
        torch.distributed.all_gather_object(merged_gen_token_lens, gen_token_lens)
        torch.distributed.all_gather_object(merged_oom_skips, oom_skips)

        merged_gts = [_ for _ in itertools.chain.from_iterable(merged_gts)]
        merged_sources = [_ for _ in itertools.chain.from_iterable(merged_sources)]
        merged_responses = [
            _ for _ in itertools.chain.from_iterable(merged_responses)
        ]
        merged_prompt_token_lens = [
            _ for _ in itertools.chain.from_iterable(merged_prompt_token_lens)
        ]
        merged_gen_token_lens = [
            _ for _ in itertools.chain.from_iterable(merged_gen_token_lens)
        ]
        merged_oom_skips = [
            _ for _ in itertools.chain.from_iterable(merged_oom_skips)
        ]


        final_output = []
        reward_sum = 0
        binary_correct_sum = 0
        if torch.distributed.get_rank() == 0:
            print(f"Evaluating {dataset_name} ...")
            if len(merged_gts) == 0:
                print(
                    "[eval] merged results are empty (no samples inferred on this run). "
                    "Common cause: --resume-log already contains every question, so all batches were skipped."
                )
                final_acc = {
                    "mean_acc": 0.0,
                    "num_samples": 0,
                    "avg_prompt_tokens": 0.0,
                    "avg_gen_tokens": 0.0,
                }
            else:
                if not (
                    len(merged_gts) == len(merged_responses) == len(merged_sources)
                ):
                    print(
                        "[eval] warning: merged list length mismatch: "
                        f"gts={len(merged_gts)} responses={len(merged_responses)} sources={len(merged_sources)}"
                    )
                for gt, response, sample in zip(merged_gts, merged_responses, merged_sources):

                    # think_chain = extract_think(response)
                    final_ans = extract_answer(response)# response #extract_answer(response)
                    gt_ans = extract_answer(gt)
                    if final_ans == "":
                        final_ans = response

                    sample["output"] = response
                    sample["question"] = format_question_for_display(sample["raw"], dataset_name)
                    sample["prediction"] = final_ans
                    # Token counts (from generate): already set on sample in worker loop; keep if missing.
                    if "output_token_count" not in sample:
                        sample["output_token_count"] = 0
                    if "prompt_token_count" not in sample:
                        sample["prompt_token_count"] = 0
                    problem_type = sample['raw']["problem_type"]
                    reward = reward_fn(final_ans, gt_ans, problem_type)
                    binary_correct = binary_correct_fn(final_ans, gt_ans, problem_type)


                    sample['reward'] = reward
                    sample['binary_correct'] = int(binary_correct)
                    reward_sum += reward
                    binary_correct_sum += int(binary_correct)

                    final_output.append(sample)


                final_acc={'mean_acc': 0.0}
                n_out = len(final_output)
                final_acc['mean_acc'] = float(reward_sum) / n_out if n_out else 0.0
                final_acc['mean_reward'] = final_acc['mean_acc']
                final_acc['binary_accuracy'] = float(binary_correct_sum) / n_out if n_out else 0.0
                final_acc["num_samples"] = int(n_out)
                final_acc["expected_samples"] = int(len(dataset))
                final_acc["missing_samples"] = int(max(0, len(dataset) - n_out))
                final_acc["complete"] = bool(n_out == len(dataset))
            final_acc["num_samples"] = int(len(final_output))
            final_acc["expected_samples"] = int(len(dataset))
            final_acc["missing_samples"] = int(max(0, len(dataset) - len(final_output)))
            final_acc["oom_skipped"] = int(len(merged_oom_skips))
            final_acc["complete"] = bool(len(final_output) == len(dataset))
            # Average token stats (prompt and generated new tokens).
            try:
                n = max(1, len(merged_gen_token_lens))
                final_acc["avg_prompt_tokens"] = float(sum(merged_prompt_token_lens)) / n
                final_acc["avg_gen_tokens"] = float(sum(merged_gen_token_lens)) / n
            except Exception:
                final_acc["avg_prompt_tokens"] = 0.0
                final_acc["avg_gen_tokens"] = 0.0

            print(final_acc)
            try:
                os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
                eval_config = {
                    key: value
                    for key, value in vars(args).items()
                    if isinstance(value, (str, int, float, bool, type(None)))
                }
                config_blob = json.dumps(eval_config, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                eval_config_hash = hashlib.sha256(config_blob.encode("utf-8")).hexdigest()
                with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "results": final_output,
                            "final_acc": [final_acc],
                            "eval_config": eval_config,
                            "eval_config_sha256": eval_config_hash,
                            "oom_skips": merged_oom_skips,
                        },
                        f,
                        indent=2,
                        ensure_ascii=False,
                    )
                print(f"Final accuracy saved to {OUTPUT_PATH}")
            except Exception as e:
                print(f"Error writing final accuracy to output file: {e}")

            print(f"Results saved to {OUTPUT_PATH}")
            if bool(getattr(args, "strict_completeness", False)) and len(final_output) != len(dataset):
                raise RuntimeError(
                    f"Incomplete evaluation: wrote {len(final_output)} of {len(dataset)} samples. "
                    "Inspect missing videos/OOM/parse failures before reporting results."
                )
        # torch.distributed.barrier()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation benchmark")
    parser.add_argument(
        "--strict-completeness",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Fail after writing diagnostics when output count differs from the dataset size.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Base model directory (local or Hub id). With --adapter-path, only base weights + processor are loaded from here.",
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        default=None,
        help="Optional PEFT adapter dir (e.g. GRPO output/stage2 with adapter_model.safetensors). Omit if --model-path is already merged full weights.",
    )
    parser.add_argument("--file-name", type=str, default="debug", help="Suffix for result JSON filenames")
    parser.add_argument(
        "--resume-log",
        type=str,
        default=None,
        help="Optional tee log path to resume from by skipping already-printed [Prompt] blocks (best-effort).",
    )
    parser.add_argument(
        "--skip-oom",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If enabled (default), skip samples that raise CUDA OOM during input move or generate(), instead of aborting eval.",
    )
    parser.add_argument(
        "--skip-missing-video-samples",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Skip annotations whose videos are missing under data-root. By default, fail with a clear error.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Deterministically evaluate at most this many examples per dataset. 0 evaluates the full dataset.",
    )
    parser.add_argument(
        "--eval-seed",
        type=int,
        default=42,
        help="Seed used for deterministic --max-samples subset selection.",
    )
    parser.add_argument("--dataset", default=["ib", "daily", "world"], nargs="+", type=str, choices=list(DEFAULT_EVAL_DATASETS.keys()))
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Root directory containing benchmark folders (IntentBench/, Daily-Omni/ or Daliy-Omni/, WorldSense/). Defaults to cwd.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Where to write eval_results JSON files. Default: <data-root>/eval_results",
    )
    parser.add_argument(
        "--attn-implementation",
        type=str,
        default="auto",
        choices=["auto", "sdpa", "flash_attention_2"],
        help=(
            "Attention backend. On CUDA, `auto` prefers flash_attention_2 (much lower vision VRAM than sdpa). "
            "`sdpa` can allocate an O(S^2) dense mask in the vision tower and OOM on long videos."
        ),
    )
    parser.add_argument(
        "--device-map",
        type=str,
        default="auto",
        help='Device map for from_pretrained (e.g. "auto"). Avoid forcing "cuda" here; use auto for safety.',
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=4096,
        help="Max new tokens to generate (align with training max_completion_length; too low truncates long CoT).",
    )
    parser.add_argument(
        "--do-sample",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Enable stochastic decoding (uses --temperature / --top-p). Default off (greedy). "
            "Benchmark evals often keep greedy for reproducibility."
        ),
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature when --do-sample is set (ignored when greedy).",
    )
    parser.add_argument(
        "--top-p",
        dest="top_p",
        type=float,
        default=0.92,
        help="Nucleus sampling top_p when --do-sample is set.",
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.0,
        help=(
            "HF repetition_penalty on logits; values >1.0 suppress repeats (also applied when greedy). "
            "1.0 leaves generation unchanged."
        ),
    )
    parser.add_argument(
        "--max-input-tokens",
        type=int,
        default=0,
        help=(
            "If >0, right-truncate input_ids to this many tokens (drops prompt *start*). "
            "0 = disable this cap (only model max_position_embeddings applies). "
            "Non-zero values can cause apparent missing leading context in outputs."
        ),
    )
    parser.add_argument(
        "--assistant-prefill",
        type=str,
        default=None,
        help=(
            "Appended to the chat string before processing (same role as GRPO `assistant_prefill` / "
            "env GRPO_ASSISTANT_PREFILL, e.g. newline + <context> block). Omit to use env only."
        ),
    )
    parser.add_argument(
        "--truncate-at-first-answer",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="After decode, keep text only through the first </answer> (default: on). Use --no-truncate-at-first-answer for full raw completion.",
    )
    parser.add_argument(
        "--stop-after-first-answer",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "During generation, stop once </answer> appears in new tokens (batch size 1 only; default: on). "
            "Use --no-stop-after-first-answer to rely only on post-decode truncation."
        ),
    )
    parser.add_argument(
        "--qa-output",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print each sample as question, prediction, and reference blocks. Use --no-qa-output for one-line output.",
    )
    parser.add_argument(
        "--num-workers",
        dest="num_workers",
        type=int,
        default=0,
        help=(
            "DataLoader workers for video/audio decode. "
            "Use 0 (default) to avoid fork + decord SIGSEGV issues; increase only if stable."
        ),
    )
    parser.add_argument(
        "--video-max-frames",
        type=int,
        default=8,
        help=(
            "Coarse temporal cap: effective fps is min(--video-reader-fps, 8/max(1, value)). "
            "Set 0 to ignore and use --video-reader-fps only."
        ),
    )
    parser.add_argument(
        "--video-reader-fps",
        type=float,
        default=1.0,
        help="Video sampling FPS for eval preprocessing (if supported by processor).",
    )
    parser.add_argument(
        "--max-pixels",
        type=int,
        default=96 * 28 * 28,
        help="Max pixels (passed via videos_kwargs/images_kwargs). Lower aggressively to reduce vision VRAM.",
    )
    parser.add_argument(
        "--min-pixels",
        type=int,
        default=56 * 56,
        help="Min pixels for smart_resize lower bound (videos_kwargs/images_kwargs).",
    )
    args = parser.parse_args()

    main(args)

"""
Example (CoGA-GRPO, with benchmarks under ./benchmarks):

export PYTHONPATH=src
torchrun --nproc_per_node=8 --master-port=29502 eval/evaluate_coga.py \\
    --model-path output/coga-grpo \\
    --file-name coga_grpo \\
    --data-root /path/to/benchmarks

Single GPU:

export PYTHONPATH=src
python eval/evaluate_coga.py --model-path output/coga-grpo --dataset ib --data-root /path/to/benchmarks

LoRA adapter (base + PEFT dir):

python eval/evaluate_coga.py \\
    --model-path /path/to/Qwen2.5-Omni-base \\
    --adapter-path output/stage2 \\
    --dataset ib --data-root /path/to/benchmarks
"""
