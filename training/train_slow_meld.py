import argparse
import csv
import json
import math
import os
import random
import shutil
import traceback
import warnings
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
Qwen2_5OmniForConditionalGeneration,
Qwen2_5OmniProcessor,
Trainer,
TrainingArguments,
)

from dataset_io import load_jsonl as load_jsonl_rows
from dataset_io import resolve_repo_path
from train_slow_track import (
SYSTEM_PROMPT,
canonicalize_prediction,
load_label_map,
parse_csv,
parse_json_response,
resolve_audio_evidence,
resolve_conflict_explanation,
resolve_fast_confidence,
resolve_gt_label,
resolve_new_summary,
resolve_reason_text,
resolve_video_paths,
resolve_visual_evidence,
str2bool,
)

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=FutureWarning, module="librosa")
warnings.filterwarnings("ignore", message="PySoundFile failed. Trying audioread instead.")

try:
    from qwen_omni_utils import process_mm_info
except ImportError:
    raise ImportError("qwen_omni_utils not found. Please ensure it is in PYTHONPATH.")


_MEDIA_VALIDATION_CACHE: Dict[Tuple[str, str, str, str], bool] = {}


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    return load_jsonl_rows(path)


def set_random_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_steps_from_epoch_fraction(
    dataloader_len: int,
    gradient_accumulation_steps: int,
    epoch_fraction: float,
) -> int:
    if dataloader_len <= 0:
        raise ValueError("dataloader_len must be > 0 when converting epoch fraction to save/eval steps.")
    if epoch_fraction <= 0:
        raise ValueError("epoch_fraction must be > 0.")

    updates_per_epoch = max(1, math.ceil(dataloader_len / max(1, gradient_accumulation_steps)))
    return max(1, math.ceil(updates_per_epoch * epoch_fraction))


def format_probs(prob_dict: Dict[str, float]) -> str:
    if not isinstance(prob_dict, dict):
        return ""
    items = sorted(prob_dict.items(), key=lambda kv: kv[1], reverse=True)
    return ", ".join([f"{k}:{v:.4f}" for k, v in items])


def select_recent_local_summary(local_summary_text: str, keep_last_n: int) -> str:
    text = str(local_summary_text or "").strip()
    if not text or keep_last_n <= 0:
        return ""
    parts = [seg.strip() for seg in text.split("|") if seg.strip()]
    if not parts:
        return ""
    return " | ".join(parts[-keep_last_n:])


def _iter_audio_waveforms(audio_item: Any):
    if audio_item is None:
        return
    if isinstance(audio_item, dict):
        for key in ["array", "audio", "waveform", "samples", "speech"]:
            if key in audio_item:
                yield from _iter_audio_waveforms(audio_item[key])
                return
        raise ValueError(f"Unsupported audio dict keys: {list(audio_item.keys())}")
    if isinstance(audio_item, (list, tuple)):
        if len(audio_item) == 0:
            return
        if len(audio_item) == 2 and isinstance(audio_item[1], (int, float, np.integer, np.floating)):
            yield from _iter_audio_waveforms(audio_item[0])
            return
        if len(audio_item) == 1:
            yield from _iter_audio_waveforms(audio_item[0])
            return
        if all(isinstance(x, (int, float, np.integer, np.floating)) for x in audio_item):
            yield np.asarray(audio_item, dtype=np.float32)
            return
        for child in audio_item:
            yield from _iter_audio_waveforms(child)
        return

    if isinstance(audio_item, torch.Tensor):
        array = audio_item.detach().cpu().float().numpy()
    else:
        array = np.asarray(audio_item, dtype=np.float32)

    if array.ndim == 0:
        raise ValueError("Audio item should be array-like, got scalar.")
    yield np.asarray(array.reshape(-1), dtype=np.float32)


def normalize_audio_inputs(audios: Any) -> List[np.ndarray]:
    if not audios:
        return []
    normalized: List[np.ndarray] = []
    for idx, item in enumerate(audios):
        try:
            normalized.extend(list(_iter_audio_waveforms(item)))
        except Exception as e:
            raise ValueError(f"Failed to normalize audio item at index={idx}, type={type(item)}") from e
    return normalized


def _get_original(sample: Dict[str, Any]) -> Dict[str, Any]:
    original = sample.get("original", None)
    if isinstance(original, dict):
        return original
    return {}


def _pick_text(candidates: List[Any]) -> str:
    for item in candidates:
        if item is None:
            continue
        text = str(item).strip()
        if text:
            return text
    return ""


def resolve_stream_id(sample: Dict[str, Any]) -> str:
    original = _get_original(sample)
    return _pick_text([sample.get("stream_id", None), original.get("stream_id", None)])


def resolve_step(sample: Dict[str, Any]) -> Optional[int]:
    for source in [sample, _get_original(sample)]:
        if not isinstance(source, dict):
            continue
        value = source.get("step", source.get("speaker_turn_index", None))
        if value is None:
            continue
        try:
            return int(value)
        except Exception:
            continue
    return None


def resolve_current_video_path(sample: Dict[str, Any], video_key: str) -> Optional[str]:
    paths = resolve_video_paths(sample, video_key)
    if not paths:
        return None
    return paths[0]


def _extract_numeric_suffix(text: str) -> Optional[int]:
    value = str(text).strip()
    if not value:
        return None
    tail = value.rsplit("::", 1)[-1]
    if tail.isdigit():
        try:
            return int(tail)
        except Exception:
            return None
    return None


def resolve_dialogue_group_id(sample: Dict[str, Any]) -> str:
    original = _get_original(sample)
    return _pick_text(
        [
            sample.get("dialogue_id", None),
            sample.get("Dialogue_ID", None),
            sample.get("group_id", None),
            sample.get("clip_id", None),
            original.get("dialogue_id", None),
            original.get("Dialogue_ID", None),
            original.get("group_id", None),
            original.get("clip_id", None),
        ]
    )


def resolve_dialogue_turn_index(sample: Dict[str, Any], video_key: str) -> Optional[int]:
    original = _get_original(sample)
    for source in [sample, original]:
        if not isinstance(source, dict):
            continue
        for key in ["utterance_id", "Utterance_ID"]:
            value = source.get(key, None)
            if value is None:
                continue
            try:
                return int(value)
            except Exception:
                continue
        for key in ["global_turn_index", "turn_index"]:
            value = source.get(key, None)
            if value is None:
                continue
            try:
                return int(value)
            except Exception:
                continue

        for key in ["sample_id", "segment_id", "id"]:
            parsed = _extract_numeric_suffix(source.get(key, ""))
            if parsed is not None:
                return parsed

    current_video_path = resolve_current_video_path(sample, video_key)
    if current_video_path:
        stem = Path(current_video_path).stem
        if "_utt" in stem:
            tail = stem.rsplit("_utt", 1)[-1]
            if tail.isdigit():
                try:
                    return int(tail)
                except Exception:
                    return None
        if stem.isdigit():
            try:
                return int(stem)
            except Exception:
                return None
    return None


def build_dialogue_previous_video_map(
    samples: List[Dict[str, Any]],
    video_key: str,
) -> Dict[Tuple[str, int], str]:
    grouped: Dict[str, List[Tuple[int, str]]] = {}
    for sample in samples:
        group_id = resolve_dialogue_group_id(sample)
        turn_index = resolve_dialogue_turn_index(sample, video_key)
        current_video_path = resolve_current_video_path(sample, video_key) or ""
        if not group_id or turn_index is None or not current_video_path:
            continue
        grouped.setdefault(group_id, []).append((turn_index, current_video_path))

    out: Dict[Tuple[str, int], str] = {}
    for group_id, items in grouped.items():
        items.sort(key=lambda x: x[0])
        previous_video_path = ""
        for turn_index, current_video_path in items:
            out[(group_id, turn_index)] = previous_video_path
            previous_video_path = current_video_path
    return out


def build_dialogue_prev_next_video_maps(
    samples: List[Dict[str, Any]],
    video_key: str,
) -> Tuple[Dict[Tuple[str, int], str], Dict[Tuple[str, int], str]]:
    grouped: Dict[str, List[Tuple[int, str]]] = {}
    for sample in samples:
        group_id = resolve_dialogue_group_id(sample)
        turn_index = resolve_dialogue_turn_index(sample, video_key)
        current_video_path = resolve_current_video_path(sample, video_key) or ""
        if not group_id or turn_index is None or not current_video_path:
            continue
        grouped.setdefault(group_id, []).append((turn_index, current_video_path))

    prev_map: Dict[Tuple[str, int], str] = {}
    next_map: Dict[Tuple[str, int], str] = {}
    for group_id, items in grouped.items():
        items.sort(key=lambda x: x[0])
        for idx, (turn_index, _) in enumerate(items):
            prev_map[(group_id, turn_index)] = items[idx - 1][1] if idx > 0 else ""
            next_map[(group_id, turn_index)] = items[idx + 1][1] if idx + 1 < len(items) else ""
    return prev_map, next_map


def resolve_previous_video_path_from_filename(current_video_path: str) -> str:
    current_path = Path(current_video_path)
    stem = current_path.stem
    if not stem.isdigit():
        return ""
    current_index = int(stem)
    if current_index <= 0:
        return ""
    prev_name = f"{current_index - 1:0{len(stem)}d}{current_path.suffix}"
    prev_path = current_path.with_name(prev_name)
    resolved = resolve_repo_path(str(prev_path))
    if not resolved.exists():
        return ""
    return str(prev_path)


def resolve_next_video_path_from_filename(current_video_path: str) -> str:
    current_path = Path(current_video_path)
    stem = current_path.stem
    if "_utt" in stem:
        prefix, tail = stem.rsplit("_utt", 1)
        if tail.isdigit():
            next_path = current_path.with_name(f"{prefix}_utt{int(tail) + 1}{current_path.suffix}")
            resolved = resolve_repo_path(str(next_path))
            if resolved.exists():
                return str(next_path)
            return ""
    if not stem.isdigit():
        return ""
    current_index = int(stem)
    next_name = f"{current_index + 1:0{len(stem)}d}{current_path.suffix}"
    next_path = current_path.with_name(next_name)
    resolved = resolve_repo_path(str(next_path))
    if not resolved.exists():
        return ""
    return str(next_path)


def build_dual_video_prompt_direct(
    sample: Dict[str, Any],
    valid_labels: List[str],
    output_fields: List[str],
    local_summary_last_n: int = 1,
) -> str:
    labels_str = ", ".join(valid_labels)
    original = _get_original(sample)
    basic_info = original.get("basic_info", {})
    if not isinstance(basic_info, dict):
        basic_info = {}

    speaker = _pick_text([sample.get("speaker", None), original.get("speaker", None), basic_info.get("speaker", None)])
    dialogue = _pick_text([sample.get("dialogue", None), original.get("dialogue", None), basic_info.get("dialogue", None)])
    running_summary = _pick_text(
        [
            sample.get("latest_running_summary", None),
            sample.get("running_memory_summary", None),
            sample.get("history_running_summary_before", None),
            sample.get("prev_summary", None),
        ]
    )
    local_summary_raw = _pick_text([sample.get("local_summary", None), original.get("local_summary", None)])
    local_summary = select_recent_local_summary(local_summary_raw, local_summary_last_n)
    reason_text = resolve_reason_text(sample)
    audio_evidence = resolve_audio_evidence(sample)
    visual_evidence = resolve_visual_evidence(sample)
    conflict_explanation = resolve_conflict_explanation(sample)
    fast_confidence = resolve_fast_confidence(sample)
    probs_text = format_probs(sample.get("fast_probs", {}))
    previous_video_source = str(sample.get("previous_video_source", "step")).strip().lower()
    previous_video_path = str(sample.get("previous_video_path", "")).strip()
    next_video_path = str(sample.get("next_video_path", "")).strip()
    context_mode = str(sample.get("context_mode", "prev_current_next")).strip().lower()

    if previous_video_source == "file_prev":
        previous_video_desc = (
            "Input 1 is audio from the temporally nearest previous clip in the same dialogue timeline. "
            "It may belong to a different speaker, so use it only as short-range temporal context.\n"
            "Input 4 is audio from the next clip on the same speaker track as the target clip."
            "Use this only for future content, as a supplementary reference to your judgment of the target clip."
        )
    else:
        previous_video_desc = (
            "Input 1 is audio from the previous clip on the same speaker track as the target clip. "
            "Use it only as speaker-history context.\n"
            "Input 4 is audio from the next clip on the same speaker track as the target clip."
            "Use this only for future content, as a supplementary reference to your judgment of the target clip."
        )

    mode_desc_map = {
        "current": [
            "You will receive only the current target audio and current target video.",
            "Do not assume previous or next context exists for this run.",
        ],
        "prev_current": [
            "You may receive one previous-context audio before the current target audio and video.",
            "Use the previous audio only as auxiliary dialogue context for the current clip.",
        ],
        "prev_current_next": [
            "You may receive both previous-context audio and next-context audio around the current target audio and video.",
            "Use the previous and next audios only as auxiliary context for the current clip.",
        ],
    }

    availability_desc = ""
    if context_mode in {"prev_current", "prev_current_next"} and not previous_video_path:
        availability_desc += (
            "There is no valid previous-context audio for this sample. "
            "Do not infer missing context that is not provided.\n"
        )
    if context_mode == "prev_current_next" and not next_video_path:
        availability_desc += (
            "There is no valid next-context audio for this sample. "
            "Do not assume future evidence beyond the target clip.\n"
        )

    prompt_lines = [
        "Task: predict the target speaker's emotion in the current clip.",
        "You may receive up to four multimodal inputs in this order:",
        "1. optional previous-context audio",
        "2. the current target audio clip",
        "3. the current target video clip",
        "4. optional next-context audio",
        "The current target audio and current target video describe the same clip and are the primary evidence.",
        "Prioritize the current target audio when judging subtle emotions, especially prosody, speaking rate, pauses, intensity, breathiness, pitch movement, trembling, laughter, crying, sighs, and voice quality changes.",
        "Use the current target video to verify or refine the emotion from facial expression, gaze, posture, and visible interaction context.",
        *mode_desc_map.get(context_mode, mode_desc_map["prev_current_next"]),
        f"Valid emotion labels: {labels_str}.",
        "Return the final emotion label for the current clip.",
        "Do not output whether correction is needed.",
    ]
    if context_mode in {"prev_current", "prev_current_next"}:
        prompt_lines.extend(
            [
                "The previous and next audios are auxiliary context only.",
                "Use previous or next audio only to resolve ambiguity in the current clip, never as the main basis for prediction.",
                "Never predict the emotion of the previous or next clip.",
                previous_video_desc.strip(),
            ]
        )
    if availability_desc:
        prompt_lines.append(availability_desc.strip())

    # prompt_lines.extend(
    #     [
    #         f"Reference prediction for the current clip: {sample.get('fast_pred', '')}.",
    #         f"Reference confidence: {fast_confidence:.4f}.",
    #         f"Reference label distribution: {probs_text}.",
    #         f"Reference entropy: {float(sample.get('fast_entropy', 0.0)):.4f}.",
    #         f"Reference margin: {float(sample.get('fast_margin', 0.0)):.4f}.",
    #     ]
    # )

    prompt = "\n".join(prompt_lines) + "\n"

    if speaker:
        prompt += f"Target speaker in the current clip: {speaker}.\n"
    if dialogue:
        prompt += f'Current dialogue: "{dialogue}".\n'
    # if running_summary:
    #     running_summary=running_summary.split("|")[-1]
    #     prompt += (
    #         f'History summary before the current clip: "{running_summary}".\n'
    #         "Use it as background state, not as a replacement for current evidence.\n"
    #     )
    # if local_summary:
    #     prompt += (
    #         f'Current-clip summary: "{local_summary}".\n'
    #         "Treat this as a compressed description of the current clip and verify it against the multimodal evidence.\n"
    #     )
    # if audio_evidence:
    #     prompt += f'Observed audio evidence for the current clip: "{audio_evidence}".\n'
    # if visual_evidence:
    #     prompt += f'Observed visual evidence for the current clip: "{visual_evidence}".\n'
    # if conflict_explanation:
    #     prompt += f'Cross-modal conflict note for the current clip: "{conflict_explanation}".\n'
    # if reason_text:
    #     prompt += f'Possible disambiguation note: "{reason_text}".\n'
    prompt += (
        "When audio and video disagree, trust the current target audio more for affective state unless the audio is clearly off-screen, corrupted, silent, or belongs to another speaker.\n"
        "Focus on the target speaker's own voice rather than background music, sound effects, or other speakers.\n"
    )
    prompt += (
        "Output only one emotion label from the valid set.\n"
        "Do not output JSON, explanations, punctuation, or any extra words.\n"
    )
    return prompt


def build_explicit_av_content(
    sample: Dict[str, Any],
    prompt_text: str,
    use_video: bool,
    fps: int,
    max_pixels: int,
    check_exists: bool,
) -> List[Dict[str, Any]]:
    content: List[Dict[str, Any]] = []
    if use_video:
        prev_video_path = str(sample.get("previous_video_path", "")).strip()
        current_video_path = str(sample.get("current_video_path", "")).strip()
        next_video_path = str(sample.get("next_video_path", "")).strip()
        context_mode = str(sample.get("context_mode", "prev_current_next")).strip().lower()

        def _exists(path: str) -> bool:
            return (not check_exists) or os.path.exists(path)

        if context_mode in {"prev_current", "prev_current_next"} and prev_video_path and _exists(prev_video_path):
            content.append({"type": "text", "text": "Input 1. Previous-context audio."})
            content.append({"type": "audio", "audio": prev_video_path})

        if current_video_path and _exists(current_video_path):
            audio_idx = 2 if context_mode in {"prev_current", "prev_current_next"} else 1
            video_idx = 3 if context_mode in {"prev_current", "prev_current_next"} else 2
            content.append({"type": "text", "text": f"Input {audio_idx}. Current target audio."})
            content.append({"type": "audio", "audio": current_video_path})
            content.append({"type": "text", "text": f"Input {video_idx}. Current target video."})
            content.append(
                {
                    "type": "video",
                    "video": current_video_path,
                    "fps": fps,
                    "max_pixels": max_pixels,
                }
            )

        if context_mode == "prev_current_next" and next_video_path and _exists(next_video_path):
            content.append({"type": "text", "text": "Input 4. Next-context audio."})
            content.append({"type": "audio", "audio": next_video_path})

    content.append({"type": "text", "text": prompt_text})
    return content


def validate_sample_media_readable(sample: Dict[str, Any]) -> bool:
    context_mode = str(sample.get("context_mode", "prev_current_next")).strip().lower()
    prev_path = str(sample.get("previous_video_path", "")).strip()
    current_path = str(sample.get("current_video_path", "")).strip()
    next_path = str(sample.get("next_video_path", "")).strip()
    cache_key = (context_mode, prev_path, current_path, next_path)
    cached = _MEDIA_VALIDATION_CACHE.get(cache_key, None)
    if cached is not None:
        return cached

    try:
        content = build_explicit_av_content(
            sample=sample,
            prompt_text="Validate media loading.",
            use_video=True,
            fps=1,
            max_pixels=4096,
            check_exists=True,
        )
        conv = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": content},
        ]
        # process_mm_info(conv, use_audio_in_video=False)
        _MEDIA_VALIDATION_CACHE[cache_key] = True
        return True
    except Exception:
        _MEDIA_VALIDATION_CACHE[cache_key] = False
        return False



def build_target_text_direct(
    sample: Dict[str, Any],
    gt_label: str,
    valid_labels: List[str],
    output_fields: List[str],
) -> str:
    del sample, output_fields
    return gt_label if gt_label in valid_labels else valid_labels[0]


def parse_final_emotion_response(text: str, valid_labels: List[str]) -> str:
    parsed = parse_json_response(text)
    for key in ["final_emotion", "corrected_emotion", "emotion", "label"]:
        if key in parsed and str(parsed[key]).strip():
            return canonicalize_prediction(str(parsed[key]), valid_labels)
    return canonicalize_prediction(text, valid_labels)


class DualVideoStateDataset(Dataset):
    def __init__(
        self,
        video_key: str,
        source_path: str = "",
        previous_video_source: str = "step",
        require_previous: bool = True,
        require_next: bool = True,
        min_fast_confidence: Optional[float] = None,
        max_fast_confidence: Optional[float] = None,
        context_mode: str = "prev_current_next",
        media_validation_workers: int = 8,
        raw_samples: Optional[List[Dict[str, Any]]] = None,
        source_name: str = "",
    ):
        if raw_samples is None:
            if not source_path:
                raise ValueError("Either source_path or raw_samples must be provided.")
            raw_samples = load_jsonl(source_path)
        self.jsonl_path = source_name or source_path or "<memory>"
        self.original_size = len(raw_samples)
        self.video_key = video_key
        self.previous_video_source = previous_video_source
        self.require_previous = require_previous
        self.require_next = require_next
        self.min_fast_confidence = min_fast_confidence
        self.max_fast_confidence = max_fast_confidence
        self.context_mode = context_mode
        self.media_validation_workers = max(1, int(media_validation_workers))
        self.skipped_missing_prev = 0
        self.skipped_missing_current_video = 0
        self.skipped_missing_prev_video = 0
        self.skipped_missing_next = 0
        self.skipped_missing_next_video = 0
        self.skipped_unreadable_media = 0
        self.skipped_confidence = 0
        self.samples = self._build_samples(raw_samples)
        # self.samples=raw_samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx]

    def _video_exists(self, video_path: Optional[str]) -> bool:
        if not video_path:
            return False
        return resolve_repo_path(video_path).exists()

    def _build_samples(self, raw_samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        disable_tqdm = local_rank != 0

        by_stream_step: Dict[Tuple[str, int], Dict[str, Any]] = {}
        for sample in tqdm(
            raw_samples,
            desc=f"Index videos: {Path(self.jsonl_path).name}",
            disable=disable_tqdm,
        ):
            stream_id = resolve_stream_id(sample)
            step = resolve_step(sample)
            if stream_id and step is not None:
                by_stream_step[(stream_id, step)] = sample
        dialogue_prev_map, dialogue_next_map = build_dialogue_prev_next_video_maps(raw_samples, self.video_key)

        prelim: List[Dict[str, Any]] = []
        for sample in tqdm(
            raw_samples,
            desc=f"Filter samples: {Path(self.jsonl_path).name}",
            disable=disable_tqdm,
        ):
            conf = resolve_fast_confidence(sample)
            # if self.min_fast_confidence is not None and conf < self.min_fast_confidence:
            #     self.skipped_confidence += 1
            #     continue
            # if self.max_fast_confidence is not None and conf > self.max_fast_confidence:
            #     self.skipped_confidence += 1
            #     continue

            current_video_path = resolve_current_video_path(sample, self.video_key)
            if not self._video_exists(current_video_path):
                self.skipped_missing_current_video += 1
                continue

            step = resolve_step(sample)
            previous_video_path = ""
            next_video_path = ""
            group_id = resolve_dialogue_group_id(sample)
            turn_index = resolve_dialogue_turn_index(sample, self.video_key)

            if self.previous_video_source == "step":
                stream_id = resolve_stream_id(sample)
                prev_sample = None
                if stream_id and step is not None:
                    prev_sample = by_stream_step.get((stream_id, step - 1))
                if prev_sample is not None:
                    previous_video_path = resolve_current_video_path(prev_sample, self.video_key) or ""
            elif self.previous_video_source == "file_prev":
                if group_id and turn_index is not None:
                    previous_video_path = dialogue_prev_map.get((group_id, turn_index), "")
                if not previous_video_path:
                    previous_video_path = resolve_previous_video_path_from_filename(current_video_path)

            if group_id and turn_index is not None:
                next_video_path = dialogue_next_map.get((group_id, turn_index), "")
            if not next_video_path:
                next_video_path = resolve_next_video_path_from_filename(current_video_path)

            effective_require_previous = self.require_previous and self.context_mode in {"prev_current", "prev_current_next"}
            effective_require_next = self.require_next and self.context_mode == "prev_current_next"

            if not previous_video_path:
                self.skipped_missing_prev += 1
                if effective_require_previous:
                    continue

            if not self._video_exists(previous_video_path):
                self.skipped_missing_prev_video += 1
                if effective_require_previous:
                    continue

            if not next_video_path:
                self.skipped_missing_next += 1
                if effective_require_next:
                    continue
            elif not self._video_exists(next_video_path):
                self.skipped_missing_next_video += 1
                if effective_require_next:
                    continue

            enriched = dict(sample)
            enriched["current_video_path"] = str(resolve_repo_path(current_video_path))
            enriched["previous_video_path"] = (
                str(resolve_repo_path(previous_video_path)) if previous_video_path else ""
            )
            enriched["next_video_path"] = str(resolve_repo_path(next_video_path)) if next_video_path else ""
            enriched["previous_video_source"] = self.previous_video_source
            enriched["context_mode"] = self.context_mode
            enriched["current_step"] = step
            enriched["previous_step"] = None if step is None else step - 1
            enriched["next_step"] = None if step is None else step + 1
            prelim.append(enriched)

        out: List[Dict[str, Any]] = []
        if self.media_validation_workers <= 1:
            for sample in tqdm(
                prelim,
                desc=f"Validate media: {Path(self.jsonl_path).name}",
                disable=disable_tqdm,
            ):
                if not validate_sample_media_readable(sample):
                    self.skipped_unreadable_media += 1
                    continue
                out.append(sample)
            return out

        future_to_index = {}
        with ThreadPoolExecutor(max_workers=self.media_validation_workers) as executor:
            for idx, sample in enumerate(prelim):
                future_to_index[executor.submit(validate_sample_media_readable, sample)] = idx

            kept_flags = [False] * len(prelim)
            for future in tqdm(
                as_completed(future_to_index),
                total=len(future_to_index),
                desc=f"Validate media: {Path(self.jsonl_path).name}",
                disable=disable_tqdm,
            ):
                idx = future_to_index[future]
                try:
                    kept_flags[idx] = bool(future.result())
                except Exception:
                    kept_flags[idx] = False

        for idx, keep in enumerate(kept_flags):
            if not keep:
                self.skipped_unreadable_media += 1
                continue
            out.append(prelim[idx])
        return out


MELD_DEFAULT_LABELS = [
    "neutral",
    "joy",
    "surprise",
    "sadness",
    "anger",
    "disgust",
    "fear",
]


def _normalize_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _canonicalize_text(value: Any) -> str:
    return "".join(ch for ch in _normalize_text(value).lower() if ch.isalnum())


def _as_int(value: Any, default: int = 0) -> int:
    text = _normalize_text(value)
    if not text:
        return default
    try:
        return int(text)
    except Exception:
        return default


def _build_meld_label_lookup(valid_labels: List[str]) -> Dict[str, str]:
    lookup = {_canonicalize_text(label): str(label).lower() for label in valid_labels}
    alias_pairs = {
        "happy": "joy",
        "happiness": "joy",
        "sad": "sadness",
        "angry": "anger",
        "disgust": "anger" if "anger" in valid_labels and "disgust" not in valid_labels else "disgust",
        "fear": "anxiety" if "anxiety" in valid_labels else "fear",
    }
    for raw_key, target in alias_pairs.items():
        if target in valid_labels:
            lookup[raw_key] = target
    return lookup


def load_meld_csv_samples(
    csv_path: str,
    video_dir: str,
    valid_labels: List[str],
    video_pattern: str,
    dialogue_id_column: str,
    utterance_id_column: str,
    utterance_column: str,
    speaker_column: str,
    label_column: str,
    skip_missing_videos: bool,
    split_name: str,
) -> Tuple[List[Dict[str, Any]], Counter, int]:
    csv_file = Path(csv_path).expanduser().resolve()
    if not csv_file.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_file}")

    base_dir = Path(video_dir).expanduser().resolve()
    with csv_file.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {csv_file}")
        raw_rows = list(reader)

    label_lookup = _build_meld_label_lookup(valid_labels)
    grouped_rows: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in raw_rows:
        grouped_rows[_normalize_text(row.get(dialogue_id_column, ""))].append(row)

    samples: List[Dict[str, Any]] = []
    label_counter: Counter = Counter()
    missing_video_count = 0
    unknown_labels: Counter = Counter()

    for dialogue_id, dialogue_rows in grouped_rows.items():
        ordered_rows = sorted(
            dialogue_rows,
            key=lambda item: (
                _as_int(item.get(utterance_id_column, 0), 0),
                _normalize_text(item.get("Sr No.", "")),
            ),
        )
        for row in ordered_rows:
            raw_label = _normalize_text(row.get(label_column, "")).lower()
            normalized_label = label_lookup.get(_canonicalize_text(raw_label), "")
            if not normalized_label:
                unknown_labels.update([raw_label or "<empty>"])
                continue

            utterance_id = _normalize_text(row.get(utterance_id_column, ""))
            filename = video_pattern.format_map(defaultdict(str, row))
            video_path = (base_dir / filename).resolve()
            if not video_path.exists():
                missing_video_count += 1
                if skip_missing_videos:
                    continue
                raise FileNotFoundError(f"Missing video for {split_name}: {video_path}")

            label_counter.update([normalized_label])
            samples.append(
                {
                    "sample_id": f"{split_name}:dia{dialogue_id}_utt{utterance_id}",
                    "id": f"{split_name}:dia{dialogue_id}_utt{utterance_id}",
                    "dialogue_id": dialogue_id,
                    "utterance_id": utterance_id,
                    "speaker": _normalize_text(row.get(speaker_column, "")),
                    "dialogue": _normalize_text(row.get(utterance_column, "")),
                    "gt_emotion": normalized_label,
                    "emotion": normalized_label,
                    "video_path": str(video_path),
                    "split_name": split_name,
                    "original": dict(row),
                }
            )

    if unknown_labels:
        items = ", ".join(f"{key}={count}" for key, count in unknown_labels.most_common())
        raise ValueError(f"Unknown labels in {csv_file}: {items}")
    return samples, label_counter, missing_video_count


class MeldDualVideoDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        video_dir: str,
        valid_labels: List[str],
        video_key: str,
        previous_video_source: str,
        require_previous: bool,
        require_next: bool,
        context_mode: str,
        media_validation_workers: int,
        video_pattern: str,
        dialogue_id_column: str,
        utterance_id_column: str,
        utterance_column: str,
        speaker_column: str,
        label_column: str,
        skip_missing_videos: bool,
        split_name: str,
    ):
        self.csv_path = csv_path
        self.original_size = 0
        self.skipped_missing_prev = 0
        self.skipped_missing_current_video = 0
        self.skipped_missing_prev_video = 0
        self.skipped_missing_next = 0
        self.skipped_missing_next_video = 0
        self.skipped_unreadable_media = 0
        self.skipped_confidence = 0
        self.label_counter = Counter()
        self.missing_video_count = 0

        raw_samples, label_counter, missing_video_count = load_meld_csv_samples(
            csv_path=csv_path,
            video_dir=video_dir,
            valid_labels=valid_labels,
            video_pattern=video_pattern,
            dialogue_id_column=dialogue_id_column,
            utterance_id_column=utterance_id_column,
            utterance_column=utterance_column,
            speaker_column=speaker_column,
            label_column=label_column,
            skip_missing_videos=skip_missing_videos,
            split_name=split_name,
        )
        self.original_size = len(raw_samples)
        self.label_counter = label_counter
        self.missing_video_count = missing_video_count
        helper = DualVideoStateDataset(
            video_key=video_key,
            previous_video_source=previous_video_source,
            require_previous=require_previous,
            require_next=require_next,
            context_mode=context_mode,
            media_validation_workers=media_validation_workers,
            raw_samples=raw_samples,
            source_name=f"{split_name}:{Path(csv_path).name}",
        )
        self.samples = helper.samples
        self.skipped_missing_prev = helper.skipped_missing_prev
        self.skipped_missing_current_video = helper.skipped_missing_current_video
        self.skipped_missing_prev_video = helper.skipped_missing_prev_video
        self.skipped_missing_next = helper.skipped_missing_next
        self.skipped_missing_next_video = helper.skipped_missing_next_video
        self.skipped_unreadable_media = helper.skipped_unreadable_media
        self.skipped_confidence = helper.skipped_confidence

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx]


class DualVideoDirectLabelCollator:
    def __init__(
        self,
        processor: Qwen2_5OmniProcessor,
        valid_labels: List[str],
        label_map: Dict[str, str],
        label_keys: List[str],
        default_label: str,
        use_video: bool,
        use_audio_in_video: bool,
        fps: int,
        prev_fps: int,
        max_pixels: int,
        prev_max_pixels: int,
        output_fields: List[str],
        local_summary_last_n: int,
    ):
        self.processor = processor
        self.valid_labels = valid_labels
        self.label_map = label_map
        self.label_keys = label_keys
        self.default_label = default_label
        self.use_video = use_video
        self.use_audio_in_video = use_audio_in_video
        self.fps = fps
        self.prev_fps = prev_fps
        self.max_pixels = max_pixels
        self.prev_max_pixels = prev_max_pixels
        self.output_fields = output_fields
        self.local_summary_last_n = local_summary_last_n
        self.ignore_index = -100

    def _build_content(self, sample: Dict[str, Any], prompt_text: str) -> List[Dict[str, Any]]:
        return build_explicit_av_content(
            sample=sample,
            prompt_text=prompt_text,
            use_video=self.use_video,
            fps=self.fps,
            max_pixels=self.max_pixels,
            check_exists=True,
        )

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        full_texts, prompt_texts = [], []
        audios_list, videos_list = [], []
        sample_meta: List[Dict[str, Any]] = []

        for sample in examples:
            sample_meta.append(
                {
                    "id": sample.get("id", sample.get("sample_id", "unknown")),
                    "prev_video": str(sample.get("previous_video_path", "")).strip(),
                    "current_video": str(sample.get("current_video_path", "")).strip(),
                    "next_video": str(sample.get("next_video_path", "")).strip(),
                }
            )
            
            gt_label = resolve_gt_label(
                sample,
                self.label_keys,
                self.valid_labels,
                self.label_map,
                self.default_label,
            )
            prompt_text = build_dual_video_prompt_direct(
                sample,
                self.valid_labels,
                self.output_fields,
                local_summary_last_n=self.local_summary_last_n,
            )
            target_text = build_target_text_direct(
                sample,
                gt_label,
                self.valid_labels,
                self.output_fields,
            )
            content = self._build_content(sample, prompt_text)

            full_conv = [
                {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                {"role": "user", "content": content},
                {"role": "assistant", "content": [{"type": "text", "text": target_text}]},
            ]
            prompt_conv = [
                {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                {"role": "user", "content": content},
            ]
            try:
                audios, _, videos = process_mm_info(full_conv, use_audio_in_video=self.use_audio_in_video)
            except Exception as e:
                sample_id = sample.get("id", sample.get("sample_id", "unknown"))
                prev_video_path = str(sample.get("previous_video_path", "")).strip()
                current_video_path = str(sample.get("current_video_path", "")).strip()
                next_video_path = str(sample.get("next_video_path", "")).strip()
                raise ValueError(
                    "process_mm_info failed during training collation "
                    "for sample "
                    f"{sample_id}, prev_video={prev_video_path}, current_video={current_video_path}, "
                    f"next_video={next_video_path}"
                ) from e
            if audios:
                audios_list.extend(normalize_audio_inputs(audios))
            if videos:
                videos_list.extend(videos)

            full_text = self.processor.apply_chat_template(full_conv, tokenize=False)
            prompt_only = self.processor.apply_chat_template(
                prompt_conv,
                add_generation_prompt=True,
                tokenize=False,
            )
            full_texts.append(full_text)
            prompt_texts.append(prompt_only)

        batch = self.processor(
            text=full_texts,
            audio=audios_list if audios_list else None,
            videos=videos_list if videos_list else None,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=self.use_audio_in_video,
        )

        batch_prompt = self.processor(
            text=prompt_texts,
            audio=audios_list if audios_list else None,
            videos=videos_list if videos_list else None,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=self.use_audio_in_video,
        )

        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        prompt_input_ids = batch_prompt["input_ids"]
        prompt_attention_mask = batch_prompt["attention_mask"]

        bsz, seq_len = input_ids.shape
        labels = input_ids.clone()
        labels[attention_mask == 0] = self.ignore_index

        full_lens = attention_mask.sum(dim=1)
        prompt_lens = prompt_attention_mask.sum(dim=1)

        for i in range(bsz):
            full_len = int(full_lens[i].item())
            prompt_len = int(prompt_lens[i].item())

            if prompt_len > full_len:
                raise ValueError(f"Sample {i}: prompt_len ({prompt_len}) > full_len ({full_len})")

            if prompt_len == full_len:
                raise ValueError(
                    f"Sample {i}: prompt_len == full_len, no supervised answer tokens found."
                )

            pad_len = seq_len - full_len
            prompt_start = pad_len
            prompt_end = pad_len + prompt_len

            full_prefix = input_ids[i, prompt_start:prompt_end]
            prompt_valid = prompt_input_ids[i, -prompt_len:]

            if not torch.equal(full_prefix, prompt_valid):
                raise ValueError("Prompt valid tokens are not equal to full prefix tokens.")

            labels[i, prompt_start:prompt_end] = self.ignore_index

        batch["labels"] = labels
        batch["_sample_meta"] = sample_meta
        return batch


class EvalIndexDataset(Dataset):
    def __init__(self, dataset: Dataset, indices: List[int]):
        self.dataset = dataset
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample_idx = int(self.indices[idx])
        sample = dict(self.dataset[sample_idx])
        sample["__eval_index__"] = sample_idx
        return sample


def identity_collate(item: Any) -> Any:
    return item


class DualVideoEvalCollator:
    def __init__(
        self,
        *,
        processor: Qwen2_5OmniProcessor,
        valid_labels: List[str],
        label_map: Dict[str, str],
        label_keys: List[str],
        default_label: str,
        use_video: bool,
        use_audio_in_video: bool,
        fps: int,
        max_pixels: int,
        output_fields: List[str],
        local_summary_last_n: int,
    ):
        self.processor = processor
        self.valid_labels = valid_labels
        self.label_map = label_map
        self.label_keys = label_keys
        self.default_label = default_label
        self.use_video = use_video
        self.use_audio_in_video = use_audio_in_video
        self.fps = fps
        self.max_pixels = max_pixels
        self.output_fields = output_fields
        self.local_summary_last_n = local_summary_last_n

    def _resolve_sample_id(self, sample: Dict[str, Any], fallback_idx: int) -> str:
        for key in ["id", "sample_id", "uid", "segment_id", "instance_id"]:
            value = sample.get(key, None)
            if value is not None and str(value).strip():
                return str(value).strip()

        original = _get_original(sample)
        clip_id = _pick_text([sample.get("clip_id", None), original.get("clip_id", None)])
        speaker = _pick_text([sample.get("speaker", None), original.get("speaker", None)])
        step = sample.get("current_step", resolve_step(sample))
        if clip_id and speaker and step is not None and str(step).strip():
            return f"{clip_id}::{speaker}::{step}"
        if clip_id:
            return clip_id
        return str(fallback_idx)

    def _prepare_one(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        sample_idx = int(sample.get("__eval_index__", -1))
        gt_label = resolve_gt_label(
            sample,
            self.label_keys,
            self.valid_labels,
            self.label_map,
            self.default_label,
        )
        prompt_text = build_dual_video_prompt_direct(
            sample,
            self.valid_labels,
            self.output_fields,
            local_summary_last_n=self.local_summary_last_n,
        )
        content = build_explicit_av_content(
            sample=sample,
            prompt_text=prompt_text,
            use_video=self.use_video,
            fps=self.fps,
            max_pixels=self.max_pixels,
            check_exists=False,
        )
        conv = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": content},
        ]
        audios, _, videos = process_mm_info(conv, use_audio_in_video=self.use_audio_in_video)
        text = self.processor.apply_chat_template(
            conv,
            add_generation_prompt=True,
            tokenize=False,
        )
        return {
            "eval_index": sample_idx,
            "gt_label": gt_label,
            "prompt_text": prompt_text,
            "batch_id": self._resolve_sample_id(sample, sample_idx),
            "text": text,
            "audios": normalize_audio_inputs(audios) if audios else [],
            "videos": videos if videos else [],
        }

    def __call__(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        prepared = [self._prepare_one(sample) for sample in samples]
        batch_audios, batch_videos = [], []
        for item in prepared:
            if item["audios"]:
                batch_audios.extend(item["audios"])
            if item["videos"]:
                batch_videos.extend(item["videos"])

        return {
            "eval_indices": [int(item["eval_index"]) for item in prepared],
            "gt_labels": [item["gt_label"] for item in prepared],
            "prompts": [item["prompt_text"] for item in prepared],
            "ids": [item["batch_id"] for item in prepared],
            "texts": [item["text"] for item in prepared],
            "audios": batch_audios,
            "videos": batch_videos,
        }


class EvalPreparedDataset(Dataset):
    def __init__(
        self,
        dataset: Dataset,
        indices: List[int],
        preparer: DualVideoEvalCollator,
    ):
        self.indexed_dataset = EvalIndexDataset(dataset, indices)
        self.preparer = preparer

    def __len__(self) -> int:
        return len(self.indexed_dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.preparer._prepare_one(self.indexed_dataset[idx])


class DualVideoDirectLabelTrainer(Trainer):

    def __init__(
        self,
        *args,
        processor: Qwen2_5OmniProcessor,
        valid_labels: List[str],
        label_map: Dict[str, str],
        label_keys: List[str],
        default_label: str,
        use_video: bool,
        use_audio_in_video: bool,
        fps: int,
        prev_fps: int,
        max_pixels: int,
        prev_max_pixels: int,
        output_fields: List[str],
        local_summary_last_n: int,
        eval_records_dir: str = "",
        eval_records_max_samples: int = 0,
        extra_eval_datasets: Optional[Dict[str, Dataset]] = None,
        eval_preprocess_workers: int = 8,
        eval_use_dataloader: bool = True,
        eval_dataloader_num_workers: int = 0,
        eval_prefetch_factor: int = 2,
        primary_eval_prefix: str = "eval",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.processor = processor
        self.valid_labels = valid_labels
        self.label_map = label_map
        self.label_keys = label_keys
        self.default_label = default_label
        self.use_video = use_video
        self.use_audio_in_video = use_audio_in_video
        self.fps = fps
        self.prev_fps = prev_fps
        self.max_pixels = max_pixels
        self.prev_max_pixels = prev_max_pixels
        self.output_fields = output_fields
        self.local_summary_last_n = local_summary_last_n
        self.eval_records_dir = str(eval_records_dir or "").strip()
        self.eval_records_max_samples = max(0, int(eval_records_max_samples))
        self.extra_eval_datasets = dict(extra_eval_datasets or {})
        self.eval_preprocess_workers = max(1, int(eval_preprocess_workers))
        self.eval_use_dataloader = bool(eval_use_dataloader)
        self.eval_dataloader_num_workers = max(0, int(eval_dataloader_num_workers))
        self.eval_prefetch_factor = max(1, int(eval_prefetch_factor))
        self.primary_eval_prefix = str(primary_eval_prefix or "eval").strip()

    def _checkpoint_dir(self) -> Path:
        return Path(self.args.output_dir) / f"checkpoint-{int(self.state.global_step)}"

    def _sorted_checkpoints(self) -> List[Path]:
        def _extract_step(path: Path) -> int:
            suffix = path.name.rsplit("-", 1)[-1]
            return int(suffix) if suffix.isdigit() else -1

        return sorted(
            [path for path in Path(self.args.output_dir).glob("checkpoint-*") if path.is_dir()],
            key=_extract_step,
        )

    def _rotate_lora_checkpoints(self) -> None:
        save_total_limit = getattr(self.args, "save_total_limit", None)
        if save_total_limit is None or int(save_total_limit) <= 0:
            return
        checkpoints = self._sorted_checkpoints()
        stale = checkpoints[:-int(save_total_limit)]
        for path in stale:
            shutil.rmtree(path, ignore_errors=True)

    def _save_lora_adapter(self, output_dir: Path, model=None) -> None:
        if not self.args.should_save:
            return

        output_dir.mkdir(parents=True, exist_ok=True)
        model_to_save = model if model is not None else self.model
        unwrapped_model = self.accelerator.unwrap_model(model_to_save, keep_torch_compile=False)
        save_kwargs = {"safe_serialization": self.args.save_safetensors}

        if self.is_deepspeed_enabled:
            save_kwargs["state_dict"] = self.accelerator.get_state_dict(self.deepspeed)
        elif self.is_fsdp_enabled:
            save_kwargs["state_dict"] = self.accelerator.get_state_dict(model_to_save)

        unwrapped_model.save_pretrained(str(output_dir), **save_kwargs)

        meta = {
            "global_step": int(self.state.global_step),
            "epoch": None if self.state.epoch is None else float(self.state.epoch),
            "seed": int(self.args.seed),
        }
        with open(output_dir / "training_state.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    def _save_checkpoint(self, model, trial) -> None:
        output_dir = self._checkpoint_dir()
        self._save_lora_adapter(output_dir, model=model)
        self._rotate_lora_checkpoints()

        if self.args.should_save and self.args.local_rank <= 0:
            print(f"[LoRACheckpoint] Saved adapter checkpoint to {output_dir}")

    def _dist_rank_world_size(self) -> tuple[int, int]:
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank(), dist.get_world_size()
        return 0, 1

    def _gather_eval_rows(self, local_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        rank, world_size = self._dist_rank_world_size()
        if world_size == 1:
            return local_rows

        gathered: List[Optional[List[Dict[str, Any]]]] = [None for _ in range(world_size)]
        dist.all_gather_object(gathered, local_rows)

        merged: List[Dict[str, Any]] = []
        for rows in gathered:
            if rows:
                merged.extend(rows)
        if rank == 0:
            merged.sort(key=lambda row: int(row.get("eval_index", -1)))
        return merged

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        sample_meta = inputs.pop("_sample_meta", None)
        try:
            return super().compute_loss(
                model,
                inputs,
                return_outputs=return_outputs,
                num_items_in_batch=num_items_in_batch,
            )
        except Exception:
            rank, world_size = self._dist_rank_world_size()
            print(f"[ComputeLossError] rank={rank}, world_size={world_size}")
            if sample_meta is not None:
                print(f"[ComputeLossError] sample_meta={sample_meta}")
            for key, value in inputs.items():
                if hasattr(value, "shape"):
                    print(f"[ComputeLossError] {key}.shape={tuple(value.shape)}")
            raise

    def _resolve_sample_id(self, sample: Dict[str, Any], fallback_idx: int) -> str:
        for key in ["id", "sample_id", "uid", "segment_id", "instance_id"]:
            value = sample.get(key, None)
            if value is not None and str(value).strip():
                return str(value).strip()

        original = _get_original(sample)
        clip_id = _pick_text([sample.get("clip_id", None), original.get("clip_id", None)])
        speaker = _pick_text([sample.get("speaker", None), original.get("speaker", None)])
        step = sample.get("current_step", resolve_step(sample))
        if clip_id and speaker and step is not None and str(step).strip():
            return f"{clip_id}::{speaker}::{step}"
        if clip_id:
            return clip_id
        return str(fallback_idx)

    def _build_content(self, sample: Dict[str, Any], prompt_text: str) -> List[Dict[str, Any]]:
        return build_explicit_av_content(
            sample=sample,
            prompt_text=prompt_text,
            use_video=self.use_video,
            fps=self.fps,
            max_pixels=self.max_pixels,
            check_exists=False,
        )

    def _prepare_eval_sample(
        self,
        sample_idx: int,
        sample: Dict[str, Any],
    ) -> Dict[str, Any]:
        gt_label = resolve_gt_label(
            sample,
            self.label_keys,
            self.valid_labels,
            self.label_map,
            self.default_label,
        )
        prompt_text = build_dual_video_prompt_direct(
            sample,
            self.valid_labels,
            self.output_fields,
            local_summary_last_n=self.local_summary_last_n,
        )
        content = self._build_content(sample, prompt_text)
        conv = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": content},
        ]
        audios, _, videos = process_mm_info(conv, use_audio_in_video=self.use_audio_in_video)
        text = self.processor.apply_chat_template(
            conv,
            add_generation_prompt=True,
            tokenize=False,
        )
        return {
            "gt_label": gt_label,
            "prompt_text": prompt_text,
            "batch_id": self._resolve_sample_id(sample, sample_idx),
            "text": text,
            "audios": normalize_audio_inputs(audios) if audios else [],
            "videos": videos if videos else [],
        }

    def _make_eval_dataloader(self, eval_dataset: Dataset, eval_indices: List[int]) -> DataLoader:
        num_workers = self.eval_dataloader_num_workers
        preparer = DualVideoEvalCollator(
            processor=self.processor,
            valid_labels=self.valid_labels,
            label_map=self.label_map,
            label_keys=self.label_keys,
            default_label=self.default_label,
            use_video=self.use_video,
            use_audio_in_video=self.use_audio_in_video,
            fps=self.fps,
            max_pixels=self.max_pixels,
            output_fields=self.output_fields,
            local_summary_last_n=self.local_summary_last_n,
        )
        kwargs = {
            "batch_size": None,
            "shuffle": False,
            "num_workers": num_workers,
            "collate_fn": identity_collate,
            "pin_memory": False,
            "persistent_workers": num_workers > 0,
        }
        if num_workers > 0:
            kwargs["prefetch_factor"] = self.eval_prefetch_factor
        return DataLoader(EvalPreparedDataset(eval_dataset, eval_indices, preparer), **kwargs)

    @staticmethod
    def _pack_prepared_eval_items(prepared_items: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch_audios, batch_videos = [], []
        for item in prepared_items:
            if item["audios"]:
                batch_audios.extend(item["audios"])
            if item["videos"]:
                batch_videos.extend(item["videos"])

        return {
            "eval_indices": [int(item["eval_index"]) for item in prepared_items],
            "gt_labels": [item["gt_label"] for item in prepared_items],
            "prompts": [item["prompt_text"] for item in prepared_items],
            "ids": [item["batch_id"] for item in prepared_items],
            "texts": [item["text"] for item in prepared_items],
            "audios": batch_audios,
            "videos": batch_videos,
        }

    def _iter_eval_dataloader_batches(self, eval_loader: DataLoader, batch_size: int):
        buffer: List[Dict[str, Any]] = []
        for item in eval_loader:
            buffer.append(item)
            if len(buffer) >= batch_size:
                yield self._pack_prepared_eval_items(buffer)
                buffer = []
        if buffer:
            yield self._pack_prepared_eval_items(buffer)

    def _maybe_save_eval_records(
        self,
        rows: List[Dict[str, Any]],
        metric_key_prefix: str,
    ) -> None:
        if not self.eval_records_dir:
            return

        rank, _ = self._dist_rank_world_size()
        if rank != 0:
            return

        out_dir = Path(self.eval_records_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        epoch = self.state.epoch
        epoch_tag = "na" if epoch is None else f"{float(epoch):.2f}".replace(".", "p")
        out_path = out_dir / f"{metric_key_prefix}_step{int(self.state.global_step)}_epoch{epoch_tag}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"[EvalDump] Saved {len(rows)} records to {out_path}")

    def _maybe_save_eval_summary(
        self,
        metrics: Dict[str, float],
        metric_key_prefix: str,
    ) -> None:
        if not self.eval_records_dir:
            return

        rank, _ = self._dist_rank_world_size()
        if rank != 0:
            return

        out_dir = Path(self.eval_records_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        epoch = self.state.epoch
        epoch_tag = "na" if epoch is None else f"{float(epoch):.2f}".replace(".", "p")
        out_path = out_dir / f"{metric_key_prefix}_step{int(self.state.global_step)}_epoch{epoch_tag}_summary.json"
        payload = {
            "metric_key_prefix": metric_key_prefix,
            "global_step": int(self.state.global_step),
            "epoch": None if epoch is None else float(epoch),
            "metrics": metrics,
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"[EvalSummaryDump] Saved metrics to {out_path}")

    def _evaluate_single_dataset(self, eval_dataset, metric_key_prefix: str) -> Dict[str, float]:
        if eval_dataset is None:
            return {}

        batch_size = self.args.per_device_eval_batch_size
        device = self.args.device
        rank, world_size = self._dist_rank_world_size()

        self.model.eval()
        old_padding_side = self.processor.tokenizer.padding_side
        self.processor.tokenizer.padding_side = "left"

        total_count = len(eval_dataset)
        eval_indices = list(range(rank, total_count, world_size))
        local_count = len(eval_indices)
        final_ok = 0
        eval_rows: List[Dict[str, Any]] = []
        model_to_gen = self.accelerator.unwrap_model(self.model)
        if rank == 0:
            print(
                f"[EvalStart] split={metric_key_prefix}, "
                f"samples={total_count}, local_samples={local_count}, "
                f"use_dataloader={self.eval_use_dataloader}, "
                f"dataloader_workers={self.eval_dataloader_num_workers}"
            )

        try:
            with torch.no_grad():
                disable_tqdm = rank > 0
                if self.eval_use_dataloader:
                    eval_iter = self._make_eval_dataloader(eval_dataset, eval_indices)
                    batch_iter = self._iter_eval_dataloader_batches(eval_iter, batch_size)
                    progress_iter = tqdm(
                        batch_iter,
                        total=math.ceil(local_count / batch_size) if batch_size > 0 else 0,
                        desc="Dual-video direct-label eval",
                        disable=disable_tqdm,
                    )
                else:
                    progress_iter = tqdm(
                        range(0, local_count, batch_size),
                        desc="Dual-video direct-label eval",
                        disable=disable_tqdm,
                    )

                for batch in progress_iter:
                    if self.eval_use_dataloader:
                        batch_indices = batch["eval_indices"]
                        batch_texts = batch["texts"]
                        batch_audios = batch["audios"]
                        batch_videos = batch["videos"]
                        gt_labels = batch["gt_labels"]
                        batch_prompts = batch["prompts"]
                        batch_ids = batch["ids"]
                    else:
                        start = int(batch)
                        batch_indices = eval_indices[start:start + batch_size]
                        batch_samples = [eval_dataset[idx] for idx in batch_indices]

                        batch_texts, batch_audios, batch_videos = [], [], []
                        gt_labels, batch_prompts, batch_ids = [], [], []
                        prepared_items: List[Optional[Dict[str, Any]]] = [None] * len(batch_samples)
                        worker_count = min(self.eval_preprocess_workers, len(batch_samples))
                        if worker_count <= 1:
                            for local_i, (sample_idx, sample) in enumerate(zip(batch_indices, batch_samples)):
                                prepared_items[local_i] = self._prepare_eval_sample(sample_idx, sample)
                        else:
                            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                                future_to_local_idx = {
                                    executor.submit(self._prepare_eval_sample, sample_idx, sample): local_i
                                    for local_i, (sample_idx, sample) in enumerate(zip(batch_indices, batch_samples))
                                }
                                for future in as_completed(future_to_local_idx):
                                    local_i = future_to_local_idx[future]
                                    prepared_items[local_i] = future.result()

                        for item in prepared_items:
                            if item is None:
                                raise RuntimeError("Prepared eval item is None.")
                            gt_labels.append(item["gt_label"])
                            batch_prompts.append(item["prompt_text"])
                            batch_ids.append(item["batch_id"])
                            batch_texts.append(item["text"])
                            if item["audios"]:
                                batch_audios.extend(item["audios"])
                            if item["videos"]:
                                batch_videos.extend(item["videos"])

                    inputs = self.processor(
                        text=batch_texts,
                        audio=batch_audios if batch_audios else None,
                        videos=batch_videos if batch_videos else None,
                        return_tensors="pt",
                        padding=True,
                        use_audio_in_video=self.use_audio_in_video,
                    ).to(device)

                    generated_ids = model_to_gen.generate(
                        **inputs,
                        max_new_tokens=96,
                        temperature=0.1,
                        pad_token_id=self.processor.tokenizer.pad_token_id,
                        eos_token_id=self.processor.tokenizer.eos_token_id,
                        use_audio_in_video=self.use_audio_in_video,
                    )

                    generated_trim = [
                        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                    ]
                    pred_texts = self.processor.batch_decode(
                        generated_trim,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )

                    for pred_text, gt_label in zip(pred_texts, gt_labels):
                        pred_label = parse_final_emotion_response(pred_text, self.valid_labels)
                        final_ok += int(pred_label == gt_label)

                    if self.eval_records_dir:
                        for k, (pred_text, gt_label) in enumerate(zip(pred_texts, gt_labels)):
                            pred_label = parse_final_emotion_response(pred_text, self.valid_labels)
                            eval_rows.append(
                                {
                                    "id": batch_ids[k],
                                    "eval_index": int(batch_indices[k]),
                                    "prompt": batch_prompts[k],
                                    "output": pred_text,
                                    "output_final_emotion": pred_label,
                                    "target_final_emotion": gt_label,
                                    "final_match": int(pred_label == gt_label),
                                }
                            )
        finally:
            self.processor.tokenizer.padding_side = old_padding_side
            self.model.train()

        if world_size > 1:
            stats = torch.tensor(
                [final_ok, local_count],
                device=device,
                dtype=torch.long,
            )
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
            final_total = int(stats[0].item())
            total_eval = int(stats[1].item())
        else:
            final_total = final_ok
            total_eval = local_count

        metrics = {
            f"{metric_key_prefix}_final_acc": final_total / total_eval if total_eval else 0.0,
        }
        self.log(metrics)
        if rank == 0:
            metric_str = ", ".join(f"{k}={v:.6f}" for k, v in metrics.items())
            print(f"[EvalSummary] {metric_str}")
        if self.eval_records_dir:
            eval_rows = self._gather_eval_rows(eval_rows)
            if self.eval_records_max_samples > 0:
                eval_rows = eval_rows[:self.eval_records_max_samples]
            self._maybe_save_eval_records(eval_rows, metric_key_prefix)
            self._maybe_save_eval_summary(metrics, metric_key_prefix)
        return metrics

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        del ignore_keys
        primary_metric_prefix = (
            self.primary_eval_prefix
            if metric_key_prefix == "eval" and self.primary_eval_prefix
            else metric_key_prefix
        )
        metrics = self._evaluate_single_dataset(eval_dataset, primary_metric_prefix)

        should_run_extra = metric_key_prefix == "eval"
        if should_run_extra:
            for extra_prefix, extra_dataset in self.extra_eval_datasets.items():
                if extra_dataset is None:
                    continue
                if eval_dataset is extra_dataset and metric_key_prefix == extra_prefix:
                    continue
                extra_metrics = self._evaluate_single_dataset(extra_dataset, extra_prefix)
                metrics.update(extra_metrics)
        return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a dual-video slow-track direct-label model using "
            "previous-audio + current-audio + current-video + next-audio inputs."
        )
    )
    parser.add_argument("--local_rank", "--local-rank", type=int, default=-1)
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--train-states-jsonl", type=str, default="")
    parser.add_argument("--val-states-jsonl", type=str, default="")
    parser.add_argument("--test-states-jsonl", type=str, default="")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--train-csv", type=str, default="")
    parser.add_argument("--val-csv", type=str, default="")
    parser.add_argument("--test-csv", type=str, default="")
    parser.add_argument("--train-video-dir", type=str, default="")
    parser.add_argument("--val-video-dir", type=str, default="")
    parser.add_argument("--test-video-dir", type=str, default="")
    parser.add_argument("--label-column", type=str, default="Emotion")
    parser.add_argument("--dialogue-id-column", type=str, default="Dialogue_ID")
    parser.add_argument("--utterance-id-column", type=str, default="Utterance_ID")
    parser.add_argument("--utterance-column", type=str, default="Utterance")
    parser.add_argument("--speaker-column", type=str, default="Speaker")
    parser.add_argument("--video-pattern", type=str, default="dia{Dialogue_ID}_utt{Utterance_ID}.mp4")
    parser.add_argument("--skip-missing-videos", type=str2bool, default=False)

    parser.add_argument(
        "--labels",
        type=str,
        default="neutral,anger,anxiety,sadness,joy,surprise,embarrassment",
    )
    parser.add_argument("--default-label", type=str, default="neutral")
    parser.add_argument("--label-map-json", type=str, default="")
    parser.add_argument("--label-keys", type=str, default="mapped_emotion,gt_emotion,emotion,target")

    parser.add_argument("--video-key", type=str, default="video_path")
    parser.add_argument("--use-video", type=str2bool, default=True)
    parser.add_argument("--use-audio-in-video", type=str2bool, default=False)
    parser.add_argument("--fps", type=int, default=2)
    parser.add_argument("--prev-fps", type=int, default=2)
    parser.add_argument("--max-pixels", type=int, default=100352)
    parser.add_argument("--prev-max-pixels", type=int, default=100352)
    parser.add_argument(
        "--previous-video-source",
        type=str,
        default="step",
        help="How to find the previous video: 'step' or 'file_prev'.",
    )
    parser.add_argument(
        "--context-mode",
        type=str,
        default="prev_current_next",
        help="One of: current, prev_current, prev_current_next.",
    )
    parser.add_argument("--require-previous-video", type=str2bool, default=True)
    parser.add_argument(
        "--require-next-video",
        type=str2bool,
        default=True,
        help="Require next clip (resolved as current filename +1) for the optional next-context audio input.",
    )
    parser.add_argument(
        "--slow-output-fields",
        type=str,
        default="",
        help="Extra JSON output keys besides final_emotion, e.g. 'reason,new_summary'.",
    )
    parser.add_argument(
        "--local-summary-last-n",
        type=int,
        default=1,
        help="Keep the latest N segments from local_summary split by '|'; <=0 disables local_summary in prompt.",
    )

    parser.add_argument("--num-train-epochs", type=int, default=5)
    parser.add_argument("--per-device-train-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--dataloader-num-workers", type=int, default=4)
    parser.add_argument(
        "--media-validation-workers",
        type=int,
        default=8,
        help="Number of threads used to validate whether media files are readable before training.",
    )
    parser.add_argument(
        "--eval-preprocess-workers",
        type=int,
        default=8,
        help="Fallback thread count for CPU-side media preprocessing when eval DataLoader is disabled.",
    )
    parser.add_argument(
        "--eval-use-dataloader",
        type=str2bool,
        default=False,
        help="Use a PyTorch DataLoader for eval media decoding so workers can prefetch prepared batches.",
    )
    parser.add_argument(
        "--eval-dataloader-num-workers",
        type=int,
        default=-1,
        help="Number of DataLoader workers for eval. If <0, reuse --dataloader-num-workers.",
    )
    parser.add_argument(
        "--eval-prefetch-factor",
        type=int,
        default=2,
        help="Eval DataLoader prefetch_factor when eval workers > 0.",
    )
    parser.add_argument(
        "--eval-splits",
        type=str,
        default="test",
        help=(
            "Comma-separated eval splits to run with generation accuracy. "
            "Options: val, test, val,test, none. Default: test."
        ),
    )
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--save-steps", type=int, default=10000)
    parser.add_argument(
        "--save-epoch-fraction",
        type=float,
        default=0.0,
        help="If >0, save and eval every N epochs worth of update steps. Example: 0.5 means half an epoch, 1 means one epoch.",
    )
    parser.add_argument(
        "--save-total-limit",
        type=int,
        default=0,
        help="Max number of LoRA step checkpoints to keep. <=0 keeps all.",
    )

    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--deepspeed-config", type=str, default="")
    parser.add_argument(
        "--run-name",
        type=str,
        default="qwen-omni-slow-track-dual-video-direct-label-prev-next-audio-labeled-slots",
    )

    parser.add_argument("--wandb-project", type=str, default="")
    parser.add_argument("--wandb-api-key", type=str, default="")
    parser.add_argument("--eval-records-dir", type=str, default="")
    parser.add_argument("--eval-records-max-samples", type=int, default=0)
    parser.add_argument("--train-min-fast-confidence", type=float, default=-1.0)
    parser.add_argument("--train-max-fast-confidence", type=float, default=-1.0)
    parser.add_argument("--val-min-fast-confidence", type=float, default=-1.0)
    parser.add_argument("--val-max-fast-confidence", type=float, default=-1.0)
    return parser.parse_args()


def main():
    args = parse_args()
    set_random_seed(args.seed)
    if args.use_audio_in_video:
        print(
            "[Config] This script now feeds the current clip audio as an explicit standalone input. "
            "Forcing --use-audio-in-video=false to avoid duplicating the current audio stream."
        )
    args.use_audio_in_video = False
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.wandb_project:
        os.environ["WANDB_PROJECT"] = args.wandb_project
    if args.wandb_api_key:
        os.environ["WANDB_API_KEY"] = args.wandb_api_key

    valid_labels = [x.lower() for x in parse_csv(args.labels)]
    context_mode = str(args.context_mode).strip().lower()
    if context_mode not in {"current", "prev_current", "prev_current_next"}:
        raise ValueError("--context-mode must be one of: current, prev_current, prev_current_next")
    if args.default_label.lower() not in valid_labels:
        raise ValueError("default-label must be in --labels")
    requested_eval_splits = [
        item.strip().lower()
        for item in str(args.eval_splits or "test").split(",")
        if item.strip()
    ]
    if requested_eval_splits == ["none"]:
        requested_eval_splits = []
    valid_eval_split_names = {"val", "test"}
    unknown_eval_splits = sorted(set(requested_eval_splits) - valid_eval_split_names)
    if unknown_eval_splits:
        raise ValueError(
            f"Unsupported --eval-splits values: {unknown_eval_splits}. "
            "Use val, test, val,test, or none."
        )
    need_val_dataset = "val" in requested_eval_splits
    need_test_dataset = "test" in requested_eval_splits
    use_meld_csv = bool(args.train_csv or args.val_csv or args.test_csv)
    use_jsonl = bool(args.train_states_jsonl or args.val_states_jsonl or args.test_states_jsonl)
    if use_meld_csv and use_jsonl:
        raise ValueError("Use either jsonl inputs or csv inputs, not both.")
    if not use_meld_csv and not use_jsonl:
        raise ValueError("Provide either --train-states-jsonl or --train-csv.")
    if use_meld_csv:
        if not args.train_csv:
            raise ValueError("CSV mode requires --train-csv.")
        if not args.train_video_dir:
            raise ValueError("CSV mode requires --train-video-dir.")
        if need_val_dataset and (not args.val_csv or not args.val_video_dir):
            raise ValueError("--eval-splits includes val, so CSV mode requires --val-csv and --val-video-dir.")
        if need_test_dataset and (not args.test_csv or not args.test_video_dir):
            raise ValueError("--eval-splits includes test, so CSV mode requires --test-csv and --test-video-dir.")
        if context_mode in {"prev_current", "prev_current_next"} and args.previous_video_source == "step":
            args.previous_video_source = "file_prev"
    effective_require_previous = bool(args.require_previous_video and context_mode in {"prev_current", "prev_current_next"})
    effective_require_next = bool(args.require_next_video and context_mode == "prev_current_next")

    label_map = load_label_map(args.label_map_json)
    label_keys = parse_csv(args.label_keys)
    output_fields = parse_csv(args.slow_output_fields)
    train_min_conf = args.train_min_fast_confidence if args.train_min_fast_confidence >= 0 else None
    train_max_conf = args.train_max_fast_confidence if args.train_max_fast_confidence >= 0 else None
    val_min_conf = args.val_min_fast_confidence if args.val_min_fast_confidence >= 0 else None
    val_max_conf = args.val_max_fast_confidence if args.val_max_fast_confidence >= 0 else None

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if local_rank == 0:
        print("Loading processor/model for dual-video direct-label prev+next-audio track...")

    processor = Qwen2_5OmniProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    processor.tokenizer.padding_side = "left"
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )
    model = model.thinker
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if local_rank == 0:
        print("Loading dual-video state datasets...")

    val_dataset = None
    test_dataset = None
    if use_jsonl:
        if not args.train_states_jsonl:
            raise ValueError("JSONL mode requires --train-states-jsonl.")
        if need_val_dataset and not args.val_states_jsonl:
            raise ValueError("--eval-splits includes val, so JSONL mode requires --val-states-jsonl.")
        if need_test_dataset and not args.test_states_jsonl:
            raise ValueError("--eval-splits includes test, so JSONL mode requires --test-states-jsonl.")
        train_dataset = DualVideoStateDataset(
            source_path=args.train_states_jsonl,
            video_key=args.video_key,
            previous_video_source=args.previous_video_source,
            require_previous=effective_require_previous,
            require_next=effective_require_next,
            min_fast_confidence=train_min_conf,
            max_fast_confidence=train_max_conf,
            context_mode=context_mode,
            media_validation_workers=args.media_validation_workers,
        )
        if need_val_dataset:
            val_dataset = DualVideoStateDataset(
                source_path=args.val_states_jsonl,
                video_key=args.video_key,
                previous_video_source=args.previous_video_source,
                require_previous=effective_require_previous,
                require_next=effective_require_next,
                min_fast_confidence=val_min_conf,
                max_fast_confidence=val_max_conf,
                context_mode=context_mode,
                media_validation_workers=args.media_validation_workers,
            )
        if need_test_dataset:
            test_dataset = DualVideoStateDataset(
                source_path=args.test_states_jsonl,
                video_key=args.video_key,
                previous_video_source=args.previous_video_source,
                require_previous=effective_require_previous,
                require_next=effective_require_next,
                context_mode=context_mode,
                media_validation_workers=args.media_validation_workers,
            )
    else:
        train_dataset = MeldDualVideoDataset(
            csv_path=args.train_csv,
            video_dir=args.train_video_dir,
            valid_labels=valid_labels,
            video_key=args.video_key,
            previous_video_source=args.previous_video_source,
            require_previous=effective_require_previous,
            require_next=effective_require_next,
            context_mode=context_mode,
            media_validation_workers=args.media_validation_workers,
            video_pattern=args.video_pattern,
            dialogue_id_column=args.dialogue_id_column,
            utterance_id_column=args.utterance_id_column,
            utterance_column=args.utterance_column,
            speaker_column=args.speaker_column,
            label_column=args.label_column,
            skip_missing_videos=args.skip_missing_videos,
            split_name="train",
        )
        if need_val_dataset:
            val_dataset = MeldDualVideoDataset(
                csv_path=args.val_csv,
                video_dir=args.val_video_dir,
                valid_labels=valid_labels,
                video_key=args.video_key,
                previous_video_source=args.previous_video_source,
                require_previous=effective_require_previous,
                require_next=effective_require_next,
                context_mode=context_mode,
                media_validation_workers=args.media_validation_workers,
                video_pattern=args.video_pattern,
                dialogue_id_column=args.dialogue_id_column,
                utterance_id_column=args.utterance_id_column,
                utterance_column=args.utterance_column,
                speaker_column=args.speaker_column,
                label_column=args.label_column,
                skip_missing_videos=args.skip_missing_videos,
                split_name="dev",
            )
        if need_test_dataset:
            test_dataset = MeldDualVideoDataset(
                csv_path=args.test_csv,
                video_dir=args.test_video_dir,
                valid_labels=valid_labels,
                video_key=args.video_key,
                previous_video_source=args.previous_video_source,
                require_previous=effective_require_previous,
                require_next=effective_require_next,
                context_mode=context_mode,
                media_validation_workers=args.media_validation_workers,
                video_pattern=args.video_pattern,
                dialogue_id_column=args.dialogue_id_column,
                utterance_id_column=args.utterance_id_column,
                utterance_column=args.utterance_column,
                speaker_column=args.speaker_column,
                label_column=args.label_column,
                skip_missing_videos=args.skip_missing_videos,
                split_name="test",
            )

    if local_rank == 0:
        print(
            "Train dataset filtered: "
            f"{len(train_dataset)}/{train_dataset.original_size} kept "
            f"(missing_prev={train_dataset.skipped_missing_prev}, "
            f"missing_prev_video={train_dataset.skipped_missing_prev_video}, "
            f"missing_next={train_dataset.skipped_missing_next}, "
            f"missing_next_video={train_dataset.skipped_missing_next_video}, "
            f"missing_current_video={train_dataset.skipped_missing_current_video}, "
            f"unreadable_media={train_dataset.skipped_unreadable_media}, "
            f"conf_filtered={train_dataset.skipped_confidence})"
        )
        if val_dataset is not None:
            print(
                "Val dataset filtered: "
                f"{len(val_dataset)}/{val_dataset.original_size} kept "
                f"(missing_prev={val_dataset.skipped_missing_prev}, "
                f"missing_prev_video={val_dataset.skipped_missing_prev_video}, "
                f"missing_next={val_dataset.skipped_missing_next}, "
                f"missing_next_video={val_dataset.skipped_missing_next_video}, "
                f"missing_current_video={val_dataset.skipped_missing_current_video}, "
                f"unreadable_media={val_dataset.skipped_unreadable_media}, "
                f"conf_filtered={val_dataset.skipped_confidence})"
            )
        if test_dataset is not None:
            print(
                "Test dataset filtered: "
                f"{len(test_dataset)}/{test_dataset.original_size} kept "
                f"(missing_prev={test_dataset.skipped_missing_prev}, "
                f"missing_prev_video={test_dataset.skipped_missing_prev_video}, "
                f"missing_next={test_dataset.skipped_missing_next}, "
                f"missing_next_video={test_dataset.skipped_missing_next_video}, "
                f"missing_current_video={test_dataset.skipped_missing_current_video}, "
                f"unreadable_media={test_dataset.skipped_unreadable_media}, "
                f"conf_filtered={test_dataset.skipped_confidence})"
            )

    if len(train_dataset) == 0:
        raise ValueError("Training dataset is empty after dual-video filtering.")
    if val_dataset is not None and len(val_dataset) == 0:
        raise ValueError("Validation dataset is empty after dual-video filtering.")
    eval_datasets_by_name = {
        "val": val_dataset,
        "test": test_dataset,
    }
    selected_eval_items: List[Tuple[str, Dataset]] = []
    for split_name in requested_eval_splits:
        split_dataset = eval_datasets_by_name.get(split_name)
        if split_dataset is None:
            raise ValueError(f"--eval-splits includes {split_name}, but that dataset was not provided.")
        selected_eval_items.append((split_name, split_dataset))
    primary_eval_name = selected_eval_items[0][0] if selected_eval_items else "eval"
    primary_eval_dataset = selected_eval_items[0][1] if selected_eval_items else None
    extra_eval_datasets = {
        split_name: split_dataset
        for split_name, split_dataset in selected_eval_items[1:]
    }

    collator = DualVideoDirectLabelCollator(
        processor=processor,
        valid_labels=valid_labels,
        label_map=label_map,
        label_keys=label_keys,
        default_label=args.default_label.lower(),
        use_video=args.use_video,
        use_audio_in_video=args.use_audio_in_video,
        fps=args.fps,
        prev_fps=args.prev_fps,
        max_pixels=args.max_pixels,
        prev_max_pixels=args.prev_max_pixels,
        output_fields=output_fields,
        local_summary_last_n=args.local_summary_last_n,
    )

    training_kwargs = dict(
        output_dir=str(output_dir),
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        max_grad_norm=args.max_grad_norm,
        logging_steps=args.logging_steps,
        seed=args.seed,
        data_seed=args.seed,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=False,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        remove_unused_columns=False,
        report_to="wandb" if args.wandb_project else "none",
        run_name=args.run_name,
        dataloader_num_workers=args.dataloader_num_workers,
        dataloader_pin_memory=True,
        ddp_find_unused_parameters=False,
    )

    init_vars = TrainingArguments.__init__.__code__.co_varnames
    eval_strategy_value = "epoch" if selected_eval_items else "no"
    if "eval_strategy" in init_vars:
        training_kwargs["eval_strategy"] = eval_strategy_value
    else:
        training_kwargs["evaluation_strategy"] = eval_strategy_value

    if args.deepspeed_config:
        training_kwargs["deepspeed"] = args.deepspeed_config

    training_args = TrainingArguments(**training_kwargs)
    eval_dataloader_num_workers = (
        args.dataloader_num_workers
        if args.eval_dataloader_num_workers < 0
        else args.eval_dataloader_num_workers
    )

    pre_eval_trainer = DualVideoDirectLabelTrainer(
        processor=processor,
        valid_labels=valid_labels,
        label_map=label_map,
        label_keys=label_keys,
        default_label=args.default_label.lower(),
        use_video=args.use_video,
        use_audio_in_video=args.use_audio_in_video,
        fps=args.fps,
        prev_fps=args.prev_fps,
        max_pixels=args.max_pixels,
        prev_max_pixels=args.prev_max_pixels,
        output_fields=output_fields,
        local_summary_last_n=args.local_summary_last_n,
        eval_records_dir=args.eval_records_dir,
        eval_records_max_samples=args.eval_records_max_samples,
        eval_preprocess_workers=args.eval_preprocess_workers,
        eval_use_dataloader=args.eval_use_dataloader,
        eval_dataloader_num_workers=eval_dataloader_num_workers,
        eval_prefetch_factor=args.eval_prefetch_factor,
        primary_eval_prefix=primary_eval_name,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=primary_eval_dataset,
        data_collator=collator,
    )

    if local_rank == 0:
        rank = int(os.environ.get("RANK", "0"))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        print(
            f"Dist env: rank={rank}, local_rank={local_rank}, world_size={world_size}"
        )

        debug_loader = DataLoader(
            train_dataset,
            batch_size=args.per_device_train_batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collator,
        )
        debug_samples = [train_dataset[idx] for idx in range(min(args.per_device_train_batch_size, len(train_dataset)))]
        debug_sample_meta = [
            {
                "id": sample.get("id", sample.get("sample_id", "unknown")),
                "prev_video": str(sample.get("previous_video_path", "")),
                "current_video": str(sample.get("current_video_path", "")),
                "next_video": str(sample.get("next_video_path", "")),
            }
            for sample in debug_samples
        ]
        print(f"Local debug samples: {debug_sample_meta}")
        try:
            debug_batch = next(iter(debug_loader))
            debug_shapes = {
                key: tuple(value.shape)
                for key, value in debug_batch.items()
                if hasattr(value, "shape")
            }
            print(f"Local debug batch keys: {list(debug_batch.keys())}")
            print(f"Local debug batch shapes: {debug_shapes}")
        except Exception as e:
            print("[Local debug] fetching the first plain DataLoader batch failed.")
            traceback.print_exc()
            raise RuntimeError("Local plain DataLoader failed before trainer.train().") from e

    # for split_name, split_dataset in selected_eval_items:
    #     if local_rank == 0:
    #         print(f"Running pre-train evaluation on {split_name} set with the base model...")
    #     # pre_eval_trainer.evaluate(eval_dataset=split_dataset, metric_key_prefix=split_name)

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )
    model = get_peft_model(model, lora_config)

    if local_rank == 0:
        model.print_trainable_parameters()

    trainer = DualVideoDirectLabelTrainer(
        processor=processor,
        valid_labels=valid_labels,
        label_map=label_map,
        label_keys=label_keys,
        default_label=args.default_label.lower(),
        use_video=args.use_video,
        use_audio_in_video=args.use_audio_in_video,
        fps=args.fps,
        prev_fps=args.prev_fps,
        max_pixels=args.max_pixels,
        prev_max_pixels=args.prev_max_pixels,
        output_fields=output_fields,
        local_summary_last_n=args.local_summary_last_n,
        eval_records_dir=args.eval_records_dir,
        eval_records_max_samples=args.eval_records_max_samples,
        extra_eval_datasets=extra_eval_datasets,
        eval_preprocess_workers=args.eval_preprocess_workers,
        eval_use_dataloader=args.eval_use_dataloader,
        eval_dataloader_num_workers=eval_dataloader_num_workers,
        eval_prefetch_factor=args.eval_prefetch_factor,
        primary_eval_prefix=primary_eval_name,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=primary_eval_dataset,
        data_collator=collator,
    )

    train_dataloader = trainer.get_train_dataloader()
    try:
        train_dataloader_len = len(train_dataloader)
    except TypeError:
        train_dataloader_len = -1

    effective_save_steps = args.save_steps
    if args.save_epoch_fraction > 0:
        if train_dataloader_len <= 0:
            raise ValueError("Cannot infer save/eval interval from epoch fraction because train dataloader length is unavailable.")
        effective_save_steps = resolve_steps_from_epoch_fraction(
            train_dataloader_len,
            args.gradient_accumulation_steps,
            args.save_epoch_fraction,
        )
        trainer.args.save_steps = effective_save_steps
        if "eval_strategy" in init_vars:
            trainer.args.eval_strategy = "steps"
            trainer.args.eval_steps = effective_save_steps
        else:
            trainer.args.evaluation_strategy = "steps"
            trainer.args.eval_steps = effective_save_steps

    if local_rank == 0:
        print(
            "Train dataloader ready: "
            f"dataset_len={len(train_dataset)}, "
            f"dataloader_len={train_dataloader_len}, "
            f"per_device_bs={args.per_device_train_batch_size}, "
            f"grad_acc={args.gradient_accumulation_steps}, "
            f"save_steps={effective_save_steps}, "
            f"save_epoch_fraction={args.save_epoch_fraction}"
        )
        print(
            "Eval loader config: "
            f"splits={','.join(requested_eval_splits) if requested_eval_splits else 'none'}, "
            f"primary={primary_eval_name}, "
            f"use_dataloader={args.eval_use_dataloader}, "
            f"num_workers={eval_dataloader_num_workers}, "
            f"prefetch_factor={args.eval_prefetch_factor}, "
            f"fallback_preprocess_workers={args.eval_preprocess_workers}"
        )

    if local_rank == 0:
        print("Start dual-video direct-label prev+next-audio LoRA training...")

    trainer.train()

    if local_rank == 0:
        print("Saving dual-video direct-label prev+next-audio LoRA model...")

    final_dir = output_dir / "slow_track_dual_video_direct_label_prev_next_audio_labeled_slots_lora"
    if trainer.is_world_process_zero():
        trainer._save_lora_adapter(final_dir)

    if local_rank == 0:
        processor.save_pretrained(str(final_dir))
        config_dump = {
            "labels": valid_labels,
            "default_label": args.default_label.lower(),
            "label_keys": label_keys,
            "video_key": args.video_key,
            "train_states_jsonl": args.train_states_jsonl,
            "val_states_jsonl": args.val_states_jsonl,
            "train_csv": args.train_csv,
            "val_csv": args.val_csv,
            "test_csv": args.test_csv,
            "train_video_dir": args.train_video_dir,
            "val_video_dir": args.val_video_dir,
            "test_video_dir": args.test_video_dir,
            "label_column": args.label_column,
            "dialogue_id_column": args.dialogue_id_column,
            "utterance_id_column": args.utterance_id_column,
            "utterance_column": args.utterance_column,
            "speaker_column": args.speaker_column,
            "video_pattern": args.video_pattern,
            "skip_missing_videos": args.skip_missing_videos,
            "media_validation_workers": args.media_validation_workers,
            "eval_preprocess_workers": args.eval_preprocess_workers,
            "eval_use_dataloader": args.eval_use_dataloader,
            "eval_dataloader_num_workers": eval_dataloader_num_workers,
            "eval_prefetch_factor": args.eval_prefetch_factor,
            "eval_splits": requested_eval_splits,
            "primary_eval_split": primary_eval_name if selected_eval_items else "",
            "use_video": args.use_video,
            "use_audio_in_video": args.use_audio_in_video,
            "fps": args.fps,
            "prev_fps": args.prev_fps,
            "max_pixels": args.max_pixels,
            "prev_max_pixels": args.prev_max_pixels,
            "slow_output_fields": output_fields,
            "local_summary_last_n": args.local_summary_last_n,
            "label_map": label_map,
            "context_mode": context_mode,
            "previous_video_source": args.previous_video_source,
            "require_previous_video": effective_require_previous,
            "require_next_video": effective_require_next,
            "train_min_fast_confidence": train_min_conf,
            "train_max_fast_confidence": train_max_conf,
            "val_min_fast_confidence": val_min_conf,
            "val_max_fast_confidence": val_max_conf,
            "seed": args.seed,
            "save_steps": args.save_steps,
            "effective_save_steps": effective_save_steps,
            "save_epoch_fraction": args.save_epoch_fraction,
            "save_total_limit": args.save_total_limit,
        }
        with open(
            output_dir / "slow_track_dual_video_direct_label_prev_next_audio_labeled_slots_config.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(config_dump, f, ensure_ascii=False, indent=2)

        print(f"Dual-video direct-label prev+next-audio training complete. Saved to {final_dir}")


if __name__ == "__main__":
    main()
