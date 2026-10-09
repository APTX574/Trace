import argparse
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.distributed as dist
from peft import PeftModel
from tqdm import tqdm
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "training"))
import av
import hashlib
import librosa
import warnings
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
from torch.utils.data import Dataset

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=FutureWarning, module="librosa")
warnings.filterwarnings("ignore", message="PySoundFile failed. Trying audioread instead.")

try:
    import decord
except ImportError:
    decord = None


def format_prefix_tag(prefix_sec: float, clip_duration_sec: float, is_full_prefix: bool) -> str:
    if is_full_prefix:
        rounded = round(float(clip_duration_sec), 3)
        if abs(rounded - round(rounded)) < 1e-6:
            return f"{int(round(rounded))}s"
        safe = str(rounded).replace(".", "p")
        return f"full_{safe}s"

    rounded = round(float(prefix_sec), 3)
    if abs(rounded - round(rounded)) < 1e-6:
        return f"{int(round(rounded))}s"
    return f"{str(rounded).replace('.', 'p')}s"


def build_prefix_audio_frame_prompt(sample: Dict[str, Any], valid_labels: List[str], include_dialogue: bool) -> str:
    labels_str = ", ".join(valid_labels)
    dialogue = str(sample.get("dialogue", "") or "").strip()
    prefix_sec = float(sample.get("prefix_sec", 0.0) or 0.0)
    clip_duration_sec = float(sample.get("clip_duration_sec", 0.0) or 0.0)
    observed_ratio = 0.0
    if clip_duration_sec > 1e-6:
        observed_ratio = min(1.0, prefix_sec / clip_duration_sec)
    lines = [
        "You are the fast belief tracker in streaming emotion understanding.",
        "Infer the speaker's current emotion from the observed prefix only.",
        "The evidence is partial and arrives incrementally.",
        "Prioritize vocal cues such as prosody, speaking rate, pauses, tremble, breathiness, laughter, crying, and intensity.",
        "Use the single visual frame only as supporting evidence for facial expression or posture.",
        "Do not assume evidence from the unobserved future part of the clip.",
        f"Observed prefix duration: {prefix_sec:.2f}s out of total clip duration {clip_duration_sec:.2f}s.",
        f"Observed fraction of the clip: {observed_ratio:.2%}.",
        f"Reply with exactly one emotion label from: {labels_str}.",
    ]
    if include_dialogue and dialogue:
        lines.append(f'Current dialogue transcript: "{dialogue}".')
    lines.append("The speaker's emotion in the observed prefix is:")
    return "\n".join(lines)


class PrefixAudioFrameDataset(Dataset):
    def __init__(self, jsonl_path: str, video_key: str, label_keys: List[str], default_label: str, valid_labels: List[str], label_map: Dict[str, str], prefix_step_sec: float = 1.0, min_prefix_sec: float = 1.0, include_full_prefix: bool = True, max_prefix_sec: float = 0.0, dataset_build_workers: int = 0, min_clip_duration_sec: float = 0.0):
        from dataset_io import load_jsonl as load_jsonl_rows
        from dataset_io import resolve_repo_path
        self._resolve_repo_path = resolve_repo_path
        self.jsonl_path = jsonl_path
        self.video_key = video_key
        self.label_keys = label_keys
        self.default_label = default_label
        self.valid_labels = valid_labels
        self.label_map = label_map
        self.prefix_step_sec = max(0.1, float(prefix_step_sec))
        self.min_prefix_sec = max(0.1, float(min_prefix_sec))
        self.include_full_prefix = bool(include_full_prefix)
        self.max_prefix_sec = max(0.0, float(max_prefix_sec))
        self.dataset_build_workers = max(0, int(dataset_build_workers))
        self.min_clip_duration_sec = max(0.0, float(min_clip_duration_sec))
        self._duration_cache: Dict[str, float] = {}
        raw_samples = load_jsonl_rows(jsonl_path)
        self.samples = self._build_prefix_samples(raw_samples)

    def _resolve_video_path(self, sample: Dict[str, Any]) -> str:
        value = sample.get(self.video_key, None)
        if value is None and self.video_key != "video_path":
            value = sample.get("video_path", None)
        if value is None or str(value).strip() == "":
            raise KeyError(f"Video path missing. Expected key '{self.video_key}' or 'video_path'.")
        return str(self._resolve_repo_path(str(value))).strip()

    def _normalize_target(self, sample: Dict[str, Any]) -> str:
        from train_fast_track import extract_raw_label, normalize_label
        raw = extract_raw_label(sample, self.label_keys, self.default_label)
        return normalize_label(raw, self.valid_labels, self.label_map, self.default_label)

    def _probe_duration_sec(self, video_path: str) -> float:
        cached = self._duration_cache.get(video_path)
        if cached is not None:
            return cached
        duration = 0.0
        try:
            with av.open(video_path) as container:
                if container.duration is not None:
                    duration = float(container.duration / av.time_base)
                else:
                    stream = container.streams.video[0]
                    if stream.duration is not None and stream.time_base is not None:
                        duration = float(stream.duration * stream.time_base)
        except Exception:
            duration = 0.0
        self._duration_cache[video_path] = max(0.0, duration)
        return self._duration_cache[video_path]

    def _resolve_clip_duration_sec(self, sample: Dict[str, Any], video_path: str) -> float:
        del sample
        return self._probe_duration_sec(video_path)

    def _build_prefix_list(self, clip_duration_sec: float) -> List[Tuple[float, bool]]:
        if clip_duration_sec <= 1e-6:
            return []
        capped_duration = clip_duration_sec
        if self.max_prefix_sec > 0:
            capped_duration = min(capped_duration, self.max_prefix_sec)
        prefixes: List[Tuple[float, bool]] = []
        current = self.min_prefix_sec
        while current + 1e-6 < capped_duration:
            rounded = float(round(current, 6))
            prefixes.append((rounded, False))
            current += self.prefix_step_sec
        if not prefixes:
            prefixes.append((float(round(capped_duration, 6)), True))
            return prefixes
        last_prefix = prefixes[-1][0]
        if self.include_full_prefix and abs(last_prefix - capped_duration) > 1e-6:
            prefixes.append((float(round(capped_duration, 6)), True))
        elif abs(last_prefix - capped_duration) <= 1e-6:
            prefixes[-1] = (prefixes[-1][0], True)
        return prefixes

    def _build_prefix_items_for_sample(self, sample_index: int, sample: Dict[str, Any]) -> List[Dict[str, Any]]:
        try:
            video_path = self._resolve_video_path(sample)
        except Exception:
            return []
        if not Path(video_path).exists():
            return []
        clip_duration_sec = self._resolve_clip_duration_sec(sample, video_path)
        if clip_duration_sec <= 1e-6:
            return []
        target = self._normalize_target(sample)
        prefix_list = self._build_prefix_list(clip_duration_sec)
        if not prefix_list:
            return []
        if self.min_clip_duration_sec > 0:
            filtered_prefix_list = [(prefix_sec, is_full_prefix) for prefix_sec, is_full_prefix in prefix_list if is_full_prefix or prefix_sec + 1e-6 >= self.min_clip_duration_sec]
            if filtered_prefix_list:
                prefix_list = filtered_prefix_list
        items: List[Dict[str, Any]] = []
        base_id = str(sample.get("sample_id", sample.get("id", sample_index))).strip()
        for prefix_index, (prefix_sec, is_full_prefix) in enumerate(prefix_list):
            item = dict(sample)
            item["resolved_video_path"] = video_path
            item["clip_duration_sec"] = float(clip_duration_sec)
            item["prefix_sec"] = float(prefix_sec)
            item["prefix_index"] = int(prefix_index)
            item["is_full_prefix"] = int(is_full_prefix)
            item["prefix_tag"] = format_prefix_tag(prefix_sec, clip_duration_sec, is_full_prefix)
            item["prefix_sample_id"] = f"{base_id}::prefix_{item['prefix_tag']}"
            item["prefix_target_label"] = target
            item["dataset_tag"] = Path(self.jsonl_path).stem
            items.append(item)
        return items

    def _build_prefix_samples(self, raw_samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        indexed_samples = list(enumerate(raw_samples))
        if self.dataset_build_workers <= 1:
            for sample_index, sample in indexed_samples:
                out.extend(self._build_prefix_items_for_sample(sample_index, sample))
            return out
        with ThreadPoolExecutor(max_workers=self.dataset_build_workers) as executor:
            for items in executor.map(lambda pair: self._build_prefix_items_for_sample(pair[0], pair[1]), indexed_samples):
                out.extend(items)
        return out

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx]


class PrefixAudioFrameCollator:
    def __init__(self, processor: Qwen2_5OmniProcessor, valid_labels: List[str], include_dialogue: bool, frame_resize_max: int = 448, use_audio_in_video: bool = False, audio_sample_rate: int = 16000, prefix_cache_dir: str = "", audio_only: bool = False):
        self.processor = processor
        self.valid_labels = valid_labels
        self.include_dialogue = include_dialogue
        self.frame_resize_max = max(64, int(frame_resize_max))
        self.use_audio_in_video = bool(use_audio_in_video)
        self.audio_sample_rate = int(audio_sample_rate)
        self.prefix_cache_dir = Path(prefix_cache_dir).expanduser() if str(prefix_cache_dir).strip() else None
        self.audio_only = bool(audio_only)
        self.ignore_index = -100
        self._audio_cache: Dict[str, np.ndarray] = {}
        self._frame_cache: Dict[Tuple[str, int], Image.Image] = {}
        self._video_reader_cache: Dict[str, Any] = {}
        self._audio_warning_cache = set()
        self._frame_warning_cache = set()

    def _cache_root_for_sample(self, sample: Dict[str, Any]) -> Optional[Path]:
        if self.prefix_cache_dir is None:
            return None
        dataset_tag = str(sample.get("dataset_tag", "dataset")).strip() or "dataset"
        prefix_id = str(sample.get("prefix_sample_id", "")).strip()
        if not prefix_id:
            video_path = str(sample.get("resolved_video_path", "")).strip()
            prefix_sec = float(sample.get("prefix_sec", 0.0) or 0.0)
            prefix_id = f"{video_path}::{prefix_sec:.3f}"
        digest = hashlib.sha1(f"{dataset_tag}::{prefix_id}".encode("utf-8")).hexdigest()
        return self.prefix_cache_dir / dataset_tag / digest[:2] / digest

    def _cache_paths_for_sample(self, sample: Dict[str, Any]) -> Tuple[Optional[Path], Optional[Path]]:
        root = self._cache_root_for_sample(sample)
        if root is None:
            return None, None
        return root / "audio.npy", root / "frame.png"

    def _load_cached_assets(self, sample: Dict[str, Any]) -> Tuple[Optional[np.ndarray], Optional[Image.Image]]:
        audio_path, frame_path = self._cache_paths_for_sample(sample)
        if audio_path is None or frame_path is None or not audio_path.exists() or not frame_path.exists():
            return None, None
        audio_prefix = np.load(audio_path).astype(np.float32, copy=False)
        with Image.open(frame_path) as image:
            frame = image.convert("RGB").copy()
        return audio_prefix, frame

    def _save_cached_assets(self, sample: Dict[str, Any], audio_prefix: np.ndarray, frame: Image.Image) -> None:
        audio_path, frame_path = self._cache_paths_for_sample(sample)
        if audio_path is None or frame_path is None:
            return
        frame_path.parent.mkdir(parents=True, exist_ok=True)
        if not audio_path.exists():
            tmp_audio_path = audio_path.parent / f"{audio_path.name}.tmp.{os.getpid()}"
            with open(tmp_audio_path, "wb") as f:
                np.save(f, np.asarray(audio_prefix, dtype=np.float32))
            os.replace(tmp_audio_path, audio_path)
        if not frame_path.exists():
            tmp_frame_path = frame_path.parent / f"{frame_path.name}.tmp.{os.getpid()}"
            frame.save(tmp_frame_path, format="PNG")
            os.replace(tmp_frame_path, frame_path)

    def prepare_prefix_assets(self, sample: Dict[str, Any]) -> Tuple[np.ndarray, Optional[Image.Image]]:
        if self.audio_only:
            video_path = str(sample["resolved_video_path"])
            prefix_sec = float(sample["prefix_sec"])
            audio_prefix = self._slice_audio_prefix(video_path, prefix_sec)
            return audio_prefix, None
        cached_audio, cached_frame = self._load_cached_assets(sample)
        if cached_audio is not None and cached_frame is not None:
            return cached_audio, cached_frame
        video_path = str(sample["resolved_video_path"])
        prefix_sec = float(sample["prefix_sec"])
        clip_duration_sec = float(sample["clip_duration_sec"])
        frame = self._extract_prefix_frame(video_path, prefix_sec, clip_duration_sec)
        audio_prefix = self._slice_audio_prefix(video_path, prefix_sec)
        self._save_cached_assets(sample, audio_prefix, frame)
        return audio_prefix, frame

    def _decode_audio_with_av(self, video_path: str) -> np.ndarray:
        chunks: List[np.ndarray] = []
        with av.open(video_path) as container:
            if not container.streams.audio:
                raise ValueError(f"No audio stream found in {video_path}")
            resampler = av.audio.resampler.AudioResampler(format="fltp", layout="mono", rate=self.audio_sample_rate)
            for frame in container.decode(audio=0):
                resampled = resampler.resample(frame)
                if resampled is None:
                    continue
                frames = resampled if isinstance(resampled, list) else [resampled]
                for item in frames:
                    chunk = item.to_ndarray()
                    chunk = np.asarray(chunk, dtype=np.float32)
                    if chunk.ndim == 2:
                        chunk = chunk[0]
                    chunks.append(chunk.reshape(-1))
        if not chunks:
            raise ValueError(f"Decoded zero audio samples from {video_path}")
        return np.concatenate(chunks, axis=0)

    def _load_audio_waveform(self, video_path: str) -> np.ndarray:
        cached = self._audio_cache.get(video_path)
        if cached is not None:
            return cached
        try:
            waveform = self._decode_audio_with_av(video_path)
        except Exception:
            waveform, _ = librosa.load(video_path, sr=self.audio_sample_rate, mono=True)
            waveform = np.asarray(waveform, dtype=np.float32)
        self._audio_cache[video_path] = waveform
        return waveform

    def _slice_audio_prefix(self, video_path: str, prefix_sec: float) -> np.ndarray:
        end_idx = max(1, int(round(prefix_sec * self.audio_sample_rate)))
        try:
            waveform = self._load_audio_waveform(video_path)
        except Exception:
            return np.zeros((end_idx,), dtype=np.float32)
        end_idx = min(end_idx, waveform.shape[0])
        return np.asarray(waveform[:end_idx], dtype=np.float32)

    def _get_video_reader(self, video_path: str):
        if decord is None:
            raise ImportError("decord is required for prefix frame extraction but is not installed.")
        reader = self._video_reader_cache.get(video_path)
        if reader is not None:
            return reader
        last_err = None
        for _ in range(2):
            try:
                reader = decord.VideoReader(video_path)
                self._video_reader_cache[video_path] = reader
                return reader
            except Exception as e:
                last_err = e
                try:
                    stale = self._video_reader_cache.pop(video_path, None)
                    del stale
                except Exception:
                    pass
        raise last_err

    def _extract_frame_with_av(self, video_path: str, prefix_sec: float, clip_duration_sec: float) -> Image.Image:
        with av.open(video_path) as container:
            if not container.streams.video:
                raise ValueError(f"No video stream found in {video_path}")
            stream = container.streams.video[0]
            fps = float(stream.average_rate) if stream.average_rate is not None else 0.0
            if fps <= 0 and stream.base_rate is not None:
                fps = float(stream.base_rate)
            if fps <= 0:
                raise ValueError(f"Unable to determine fps for {video_path}")
            safe_prefix = max(0.0, min(prefix_sec, clip_duration_sec))
            frame_time = max(0.0, safe_prefix - 1e-3)
            target_idx = max(0, int(round(frame_time * fps)))
            last_frame = None
            for idx, frame in enumerate(container.decode(video=0)):
                last_frame = frame
                if idx >= target_idx:
                    break
            if last_frame is None:
                raise ValueError(f"Decoded zero video frames from {video_path}")
            frame_np = last_frame.to_ndarray(format="rgb24")
            return Image.fromarray(frame_np).convert("RGB")

    def _resize_frame(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        long_side = max(width, height)
        if long_side <= self.frame_resize_max:
            return image
        scale = self.frame_resize_max / float(long_side)
        return image.resize((max(1, int(round(width * scale))), max(1, int(round(height * scale)))))

    def _extract_prefix_frame(self, video_path: str, prefix_sec: float, clip_duration_sec: float) -> Image.Image:
        cache_key = (video_path, int(round(prefix_sec * 1000.0)))
        cached = self._frame_cache.get(cache_key)
        if cached is not None:
            return cached
        image = None
        try:
            reader = self._get_video_reader(video_path)
            total_frames = len(reader)
            fps = float(reader.get_avg_fps())
            if total_frames <= 0 or fps <= 0:
                raise ValueError(f"Invalid video reader state for {video_path}: total_frames={total_frames}, fps={fps}")
            safe_prefix = max(0.0, min(prefix_sec, clip_duration_sec))
            frame_time = max(0.0, safe_prefix - 1e-3)
            frame_idx = max(0, min(int(round(frame_time * fps)), total_frames - 1))
            frame_np = reader[frame_idx].asnumpy()
            image = Image.fromarray(frame_np).convert("RGB")
        except Exception as e:
            try:
                stale = self._video_reader_cache.pop(video_path, None)
                del stale
            except Exception:
                pass
            image = self._extract_frame_with_av(video_path, prefix_sec, clip_duration_sec)
        image = self._resize_frame(image)
        self._frame_cache[cache_key] = image
        return image
from train_fast_track import (
    SYSTEM_PROMPT,
    canonicalize_prediction,
    load_label_map,
    parse_csv,
    str2bool,
)
from qwen_omni_utils import process_mm_info


DEFAULT_LABELS = "neutral,anger,anxiety,sadness,joy,surprise,embarrassment"


def dist_rank_world_size() -> Tuple[int, int]:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


def gather_object_rows(local_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rank, world_size = dist_rank_world_size()
    if world_size == 1:
        return local_rows
    gathered: List[Optional[List[Dict[str, Any]]]] = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, local_rows)
    if rank != 0:
        return []
    merged: List[Dict[str, Any]] = []
    for rows in gathered:
        if rows:
            merged.extend(rows)
    merged.sort(key=lambda row: (str(row.get("sample_id", "")), float(row.get("prefix_sec", 0.0))))
    return merged


def stable_softmax(log_values: Dict[str, float]) -> Dict[str, float]:
    if not log_values:
        return {}
    max_value = max(float(v) for v in log_values.values())
    exps = {k: math.exp(float(v) - max_value) for k, v in log_values.items()}
    denom = sum(exps.values())
    if denom <= 0:
        uniform = 1.0 / max(1, len(exps))
        return {k: uniform for k in exps}
    return {k: float(v / denom) for k, v in exps.items()}


def entropy_from_probs(probabilities: Dict[str, float]) -> float:
    total = 0.0
    for prob in probabilities.values():
        p = max(float(prob), 1e-12)
        total -= p * math.log(p)
    return float(total)


def margin_from_probs(probabilities: Dict[str, float]) -> float:
    values = sorted((float(v) for v in probabilities.values()), reverse=True)
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    return float(values[0] - values[1])


def compute_macro_f1(rows: Sequence[Dict[str, Any]], labels: Sequence[str]) -> float:
    if not rows:
        return 0.0
    f1_values: List[float] = []
    for label in labels:
        tp = fp = fn = 0
        for row in rows:
            pred = str(row.get("pred_label", ""))
            gold = str(row.get("gold_label", ""))
            if pred == label and gold == label:
                tp += 1
            elif pred == label and gold != label:
                fp += 1
            elif pred != label and gold == label:
                fn += 1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 0.0 if (precision + recall) <= 0 else 2.0 * precision * recall / (precision + recall)
        f1_values.append(f1)
    return float(sum(f1_values) / len(f1_values)) if f1_values else 0.0


def format_float_tag(value: float) -> str:
    rounded = round(float(value), 3)
    if abs(rounded - round(rounded)) < 1e-6:
        return f"{int(round(rounded))}s"
    return f"{str(rounded).replace('.', 'p')}s"


def _extract_sample_id(sample: Dict[str, Any]) -> str:
    """Extract sample_id, falling back to prefix_sample_id base or 'id' field."""
    sid = str(sample.get("sample_id", "")).strip()
    if sid:
        return sid
    psid = str(sample.get("prefix_sample_id", "")).strip()
    if "::" in psid:
        return psid.split("::")[0]
    return str(sample.get("id", "")).strip()


class LikelihoodPrefixAnalyzer:
    def __init__(
        self,
        model: Qwen2_5OmniForConditionalGeneration,
        processor: Qwen2_5OmniProcessor,
        collator: PrefixAudioFrameCollator,
        valid_labels: List[str],
        include_dialogue: bool,
        use_audio_in_video: bool,
        device: torch.device,
        dtype: torch.dtype,
        audio_only: bool = False,
    ):
        self.model = model
        self.processor = processor
        self.collator = collator
        self.valid_labels = valid_labels
        self.include_dialogue = include_dialogue
        self.use_audio_in_video = use_audio_in_video
        self.audio_only = audio_only
        self.device = device
        self.dtype = dtype
        # Pre-compute label token IDs once (avoid repeated tokenizer.encode per sample)
        self._label_token_ids: Dict[str, List[int]] = {}
        self._label_first_token_ids: Dict[str, int] = {}
        for _label in valid_labels:
            _ids = processor.tokenizer.encode(_label, add_special_tokens=False)
            self._label_token_ids[_label] = _ids
            self._label_first_token_ids[_label] = int(_ids[0]) if _ids else -1

    def _build_prompt_conv(self, sample: Dict[str, Any], audio_prefix: np.ndarray, frame) -> Tuple[List[Dict[str, Any]], str]:
        prompt_text = build_prefix_audio_frame_prompt(
            sample=sample,
            valid_labels=self.valid_labels,
            include_dialogue=self.include_dialogue,
        )
        content = [
            {"type": "audio", "audio": audio_prefix},
        ]
        if frame is not None:
            content.append({"type": "image", "image": frame})
        content.append({"type": "text", "text": prompt_text})
        conv = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": content},
        ]
        return conv, prompt_text

    def _prepare_model_inputs(self, conv: List[Dict[str, Any]], audio_prefix: np.ndarray, frame):
        """Build processor inputs with add_generation_prompt=True (the only call needed)."""
        prompt_text_only = self.processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
        audios, images, _ = process_mm_info(conv, use_audio_in_video=self.use_audio_in_video)
        audio_item = audios[0] if audios else audio_prefix
        image_item = images[0] if images else frame
        processor_kwargs = {
            "text": [prompt_text_only],
            "audio": [audio_item],
            "return_tensors": "pt",
            "padding": True,
            "use_audio_in_video": self.use_audio_in_video,
        }
        if image_item is not None:
            processor_kwargs["images"] = [image_item]
        batch = self.processor(**processor_kwargs)
        return {k: v.to(self.device) for k, v in batch.items()}

    def _decode_logits(self, next_token_logits: torch.Tensor) -> Tuple[Dict[str, float], Dict[str, int], str, float, float, float]:
        """Extract label probabilities from last-token logits using pre-computed token IDs."""
        next_token_log_probs = torch.log_softmax(next_token_logits, dim=-1)
        label_log_likelihoods: Dict[str, float] = {}
        label_first_token_ids: Dict[str, int] = {}
        for label in self.valid_labels:
            first_token_id = self._label_first_token_ids[label]
            if first_token_id < 0:
                label_log_likelihoods[label] = float("-inf")
                label_first_token_ids[label] = -1
                continue
            label_first_token_ids[label] = first_token_id
            label_log_likelihoods[label] = float(next_token_log_probs[first_token_id].item())
        probabilities = stable_softmax(label_log_likelihoods)
        pred_label = max(probabilities.items(), key=lambda item: item[1])[0]
        confidence = float(probabilities[pred_label])
        entropy = entropy_from_probs(probabilities)
        margin = margin_from_probs(probabilities)
        return probabilities, label_first_token_ids, pred_label, confidence, entropy, margin

    def score_candidates(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        # Load assets + build conv (NO redundant _encode_conv call)
        audio_prefix, frame = self.collator.prepare_prefix_assets(sample)
        prompt_conv, prompt_text = self._build_prompt_conv(sample, audio_prefix, frame)

        # Single processor call (was previously called TWICE)
        model_inputs = self._prepare_model_inputs(prompt_conv, audio_prefix, frame)

        with torch.no_grad():
            outputs = self.model(**model_inputs, use_audio_in_video=self.use_audio_in_video)
            next_token_logits = outputs.logits[0, -1, :]

        probabilities, label_first_token_ids, pred_label, confidence, entropy, margin = self._decode_logits(next_token_logits)
        label_log_likelihoods = {k: float(v) for k, v in probabilities.items()}
        gold_label = canonicalize_prediction(str(sample.get("prefix_target_label", "")), self.valid_labels)
        gold_prob = float(probabilities.get(gold_label, 0.0))

        return {
            "sample_id": _extract_sample_id(sample),
            "prefix_sample_id": str(sample.get("prefix_sample_id", "")),
            "clip_id": str(sample.get("clip_id", "")),
            "stream_id": str(sample.get("stream_id", "")),
            "speaker": str(sample.get("speaker", "")),
            "video_path": str(sample.get("resolved_video_path", sample.get("video_path", ""))),
            "dialogue": str(sample.get("dialogue", "") or ""),
            "prefix_sec": float(sample.get("prefix_sec", 0.0) or 0.0),
            "prefix_tag": str(sample.get("prefix_tag", "")),
            "prefix_index": int(sample.get("prefix_index", 0) or 0),
            "clip_duration_sec": float(sample.get("clip_duration_sec", 0.0) or 0.0),
            "is_full_prefix": int(sample.get("is_full_prefix", 0) or 0),
            "gold_label": gold_label,
            "pred_label": pred_label,
            "is_correct": int(pred_label == gold_label),
            "confidence": confidence,
            "entropy": entropy,
            "margin": margin,
            "gold_prob": gold_prob,
            "probabilities": probabilities,
            "label_log_likelihoods": label_log_likelihoods,
            "label_first_token_ids": label_first_token_ids,
            "prompt": prompt_text,
        }

    def score_batch(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Score a batch of samples together for better GPU utilization.

        Each sample is processed through the processor individually (due to variable
        audio lengths), then batched for a single GPU forward pass.
        """
        if len(samples) == 1:
            return [self.score_candidates(samples[0])]

        batch_inputs = []
        batch_meta = []
        for sample in samples:
            audio_prefix, frame = self.collator.prepare_prefix_assets(sample)
            prompt_conv, prompt_text = self._build_prompt_conv(sample, audio_prefix, frame)
            model_inputs = self._prepare_model_inputs(prompt_conv, audio_prefix, frame)
            batch_inputs.append(model_inputs)
            batch_meta.append((sample, prompt_text))

        # Pad and stack all inputs into a single batch
        keys = batch_inputs[0].keys()
        padded = {}
        for key in keys:
            tensors = [bi[key] for bi in batch_inputs]
            if key == "attention_mask":
                padded[key] = torch.nn.utils.rnn.pad_sequence(
                    [t.squeeze(0) for t in tensors], batch_first=True, padding_value=0
                )
            elif key == "input_ids":
                padded[key] = torch.nn.utils.rnn.pad_sequence(
                    [t.squeeze(0) for t in tensors], batch_first=True, padding_value=0
                )
            elif key == "labels":
                padded[key] = torch.nn.utils.rnn.pad_sequence(
                    [t.squeeze(0) for t in tensors], batch_first=True, padding_value=-100
                )
            else:
                try:
                    padded[key] = torch.cat(tensors, dim=0)
                except Exception:
                    # For keys that can't be simply concatenated, fall back to first item
                    padded[key] = tensors[0]

        with torch.no_grad():
            outputs = self.model(**padded, use_audio_in_video=self.use_audio_in_video)
            all_logits = outputs.logits  # (batch, seq, vocab)

        results = []
        for i, (sample, prompt_text) in enumerate(batch_meta):
            next_token_logits = all_logits[i, -1, :]
            probabilities, label_first_token_ids, pred_label, confidence, entropy, margin = self._decode_logits(next_token_logits)
            label_log_likelihoods = {k: float(v) for k, v in probabilities.items()}
            gold_label = canonicalize_prediction(str(sample.get("prefix_target_label", "")), self.valid_labels)
            gold_prob = float(probabilities.get(gold_label, 0.0))
            results.append({
                "sample_id": _extract_sample_id(sample),
                "prefix_sample_id": str(sample.get("prefix_sample_id", "")),
                "clip_id": str(sample.get("clip_id", "")),
                "stream_id": str(sample.get("stream_id", "")),
                "speaker": str(sample.get("speaker", "")),
                "video_path": str(sample.get("resolved_video_path", sample.get("video_path", ""))),
                "dialogue": str(sample.get("dialogue", "") or ""),
                "prefix_sec": float(sample.get("prefix_sec", 0.0) or 0.0),
                "prefix_tag": str(sample.get("prefix_tag", "")),
                "prefix_index": int(sample.get("prefix_index", 0) or 0),
                "clip_duration_sec": float(sample.get("clip_duration_sec", 0.0) or 0.0),
                "is_full_prefix": int(sample.get("is_full_prefix", 0) or 0),
                "gold_label": gold_label,
                "pred_label": pred_label,
                "is_correct": int(pred_label == gold_label),
                "confidence": confidence,
                "entropy": entropy,
                "margin": margin,
                "gold_prob": gold_prob,
                "probabilities": probabilities,
                "label_log_likelihoods": label_log_likelihoods,
                "label_first_token_ids": label_first_token_ids,
                "prompt": prompt_text,
            })
        return results


def write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_jsonl_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_completed_prefix_ids(partial_dir: Path) -> set:
    completed = set()
    if not partial_dir.exists():
        return completed
    for path in sorted(partial_dir.glob("prefix_predictions.rank*.jsonl")):
        for row in load_jsonl_rows(path):
            prefix_id = str(row.get("prefix_sample_id", "")).strip()
            if prefix_id:
                completed.add(prefix_id)
    return completed


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def summarize_prefix_metrics(rows: Sequence[Dict[str, Any]], labels: Sequence[str]) -> List[Dict[str, Any]]:
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["prefix_tag"])].append(row)

    summary_rows: List[Dict[str, Any]] = []
    for prefix_tag, group_rows in sorted(groups.items(), key=lambda item: (float(item[1][0]["prefix_sec"]), item[0])):
        num_samples = len(group_rows)
        accuracy = sum(int(r["is_correct"]) for r in group_rows) / num_samples if num_samples else 0.0
        macro_f1 = compute_macro_f1(group_rows, labels)
        avg_confidence = sum(float(r["confidence"]) for r in group_rows) / num_samples if num_samples else 0.0
        avg_entropy = sum(float(r["entropy"]) for r in group_rows) / num_samples if num_samples else 0.0
        avg_margin = sum(float(r["margin"]) for r in group_rows) / num_samples if num_samples else 0.0
        avg_gold_prob = sum(float(r["gold_prob"]) for r in group_rows) / num_samples if num_samples else 0.0
        summary_rows.append(
            {
                "prefix_tag": prefix_tag,
                "prefix_sec": float(group_rows[0]["prefix_sec"]),
                "num_samples": num_samples,
                "accuracy": accuracy,
                "macro_f1": macro_f1,
                "avg_confidence": avg_confidence,
                "avg_entropy": avg_entropy,
                "avg_margin": avg_margin,
                "avg_gold_prob": avg_gold_prob,
            }
        )
    return summary_rows


def build_belief_changes(
    rows: Sequence[Dict[str, Any]],
    entropy_threshold: float,
    margin_threshold: float,
    stable_conf_threshold: float,
) -> List[Dict[str, Any]]:
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        sample_id = str(row.get("sample_id") or row.get("clip_id") or row.get("prefix_sample_id"))
        groups[sample_id].append(row)

    changes: List[Dict[str, Any]] = []
    for sample_id, group in groups.items():
        sorted_group = sorted(group, key=lambda x: (float(x["prefix_sec"]), int(x.get("prefix_index", 0))))
        for prev_row, curr_row in zip(sorted_group[:-1], sorted_group[1:]):
            prev_correct = int(prev_row["is_correct"])
            curr_correct = int(curr_row["is_correct"])
            prev_pred = str(prev_row["pred_label"])
            curr_pred = str(curr_row["pred_label"])
            delta_conf = float(curr_row["confidence"]) - float(prev_row["confidence"])
            delta_entropy = float(curr_row["entropy"]) - float(prev_row["entropy"])
            delta_margin = float(curr_row["margin"]) - float(prev_row["margin"])
            delta_gold_prob = float(curr_row["gold_prob"]) - float(prev_row["gold_prob"])

            event_flags = {
                "strengthen": int(prev_correct == 1 and delta_gold_prob > 0.0),
                "corrective_update": int(prev_correct == 0 and curr_correct == 1),
                "harmful_update": int(prev_correct == 1 and curr_correct == 0),
                "uncertain_wait": int(
                    float(curr_row["entropy"]) >= entropy_threshold or float(curr_row["margin"]) <= margin_threshold
                ),
                "stable_commit": int(
                    prev_pred == curr_pred
                    and float(prev_row["confidence"]) >= stable_conf_threshold
                    and float(curr_row["confidence"]) >= stable_conf_threshold
                ),
            }

            row = {
                "sample_id": sample_id,
                "prefix_sample_id_prev": str(prev_row.get("prefix_sample_id", "")),
                "prefix_sample_id_curr": str(curr_row.get("prefix_sample_id", "")),
                "dialogue": str(curr_row.get("dialogue", "") or prev_row.get("dialogue", "")),
                "gold_label": str(curr_row["gold_label"]),
                "prev_prefix_sec": float(prev_row["prefix_sec"]),
                "curr_prefix_sec": float(curr_row["prefix_sec"]),
                "transition": f"{format_float_tag(prev_row['prefix_sec'])}->{format_float_tag(curr_row['prefix_sec'])}",
                "prev_pred_label": prev_pred,
                "curr_pred_label": curr_pred,
                "label_changed": int(prev_pred != curr_pred),
                "was_correct_prev": prev_correct,
                "is_correct_curr": curr_correct,
                "prev_confidence": float(prev_row["confidence"]),
                "curr_confidence": float(curr_row["confidence"]),
                "prev_entropy": float(prev_row["entropy"]),
                "curr_entropy": float(curr_row["entropy"]),
                "prev_margin": float(prev_row["margin"]),
                "curr_margin": float(curr_row["margin"]),
                "prev_gold_prob": float(prev_row["gold_prob"]),
                "curr_gold_prob": float(curr_row["gold_prob"]),
                "delta_confidence": delta_conf,
                "delta_entropy": delta_entropy,
                "delta_margin": delta_margin,
                "delta_gold_prob": delta_gold_prob,
                "prev_probabilities": prev_row["probabilities"],
                "curr_probabilities": curr_row["probabilities"],
            }
            row.update(event_flags)
            changes.append(row)
    return changes


def summarize_belief_changes(changes: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    event_names = ["strengthen", "corrective_update", "harmful_update", "uncertain_wait", "stable_commit"]
    summary: List[Dict[str, Any]] = []
    total = len(changes)
    for event_name in event_names:
        matched = [row for row in changes if int(row.get(event_name, 0)) == 1]
        count = len(matched)
        transition_counter = Counter(str(row["transition"]) for row in matched)
        summary.append(
            {
                "event": event_name,
                "count": count,
                "ratio": (count / total) if total > 0 else 0.0,
                "avg_delta_gold_prob": (sum(float(r["delta_gold_prob"]) for r in matched) / count) if count else 0.0,
                "avg_delta_entropy": (sum(float(r["delta_entropy"]) for r in matched) / count) if count else 0.0,
                "avg_delta_margin": (sum(float(r["delta_margin"]) for r in matched) / count) if count else 0.0,
                "transition_breakdown": json.dumps(dict(sorted(transition_counter.items())), ensure_ascii=False),
            }
        )
    return summary


def export_cases(case_dir: Path, changes: Sequence[Dict[str, Any]], max_cases: int) -> None:
    case_dir.mkdir(parents=True, exist_ok=True)
    case_specs = {
        "corrective_update_cases.jsonl": (
            [row for row in changes if int(row.get("corrective_update", 0)) == 1],
            lambda row: (-float(row["delta_gold_prob"]), -float(row["curr_confidence"])),
        ),
        "harmful_update_cases.jsonl": (
            [row for row in changes if int(row.get("harmful_update", 0)) == 1],
            lambda row: (float(row["delta_gold_prob"]), -float(row["curr_entropy"])),
        ),
        "high_entropy_wait_cases.jsonl": (
            [row for row in changes if int(row.get("uncertain_wait", 0)) == 1],
            lambda row: (-float(row["curr_entropy"]), float(row["curr_margin"])),
        ),
        "stable_commit_cases.jsonl": (
            [row for row in changes if int(row.get("stable_commit", 0)) == 1],
            lambda row: (-float(row["curr_confidence"]), -float(row["curr_margin"])),
        ),
    }
    for filename, (rows, sort_key) in case_specs.items():
        selected = sorted(rows, key=sort_key)[:max_cases]
        write_jsonl(case_dir / filename, selected)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze prefix-level fast belief with label token likelihood.")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--adapter-path", type=str, required=True)
    parser.add_argument("--input-jsonl", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--labels", type=str, default=DEFAULT_LABELS)
    parser.add_argument("--default-label", type=str, default="neutral")
    parser.add_argument("--label-map-json", type=str, default="")
    parser.add_argument("--label-keys", type=str, default="gt_emotion,emotion,target")
    parser.add_argument("--video-key", type=str, default="video_path")
    parser.add_argument("--include-dialogue", type=str2bool, default=False)
    parser.add_argument("--prefix-step-sec", type=float, default=1.0)
    parser.add_argument("--min-prefix-sec", type=float, default=1.5)
    parser.add_argument("--include-full-prefix", type=str2bool, default=True)
    parser.add_argument("--max-prefix-sec", type=float, default=0.0)
    parser.add_argument("--dataset-build-workers", type=int, default=8)
    parser.add_argument("--frame-resize-max", type=int, default=448)
    parser.add_argument("--audio-sample-rate", type=int, default=16000)
    parser.add_argument("--use-audio-in-video", type=str2bool, default=False)
    parser.add_argument("--audio-only", type=str2bool, default=False, help="Skip video frame extraction and use audio only. Default False: the released fast axis consumes the audio prefix plus the single prefix frame.")
    parser.add_argument("--prefix-cache-dir", type=str, default="", help="Optional pre-extracted audio prefix cache directory.")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for GPU inference. Increase to 2-4 for better GPU utilization on large-VRAM GPUs.")
    parser.add_argument("--prefetch-workers", type=int, default=0, help="Number of CPU prefetch threads for asset loading. 0=disabled, 2-4 recommended.")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--entropy-threshold", type=float, default=1.5)
    parser.add_argument("--margin-threshold", type=float, default=0.15)
    parser.add_argument("--stable-confidence-threshold", type=float, default=0.60)
    parser.add_argument("--export-case-count", type=int, default=50)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--resume", type=str2bool, default=True)
    parser.add_argument("--local_rank", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if "LOCAL_RANK" in os.environ and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    case_dir = output_dir / "cases"

    valid_labels = [x.lower() for x in parse_csv(args.labels)]
    label_map = load_label_map(args.label_map_json)
    label_keys = parse_csv(args.label_keys)

    dataset = PrefixAudioFrameDataset(
        jsonl_path=args.input_jsonl,
        video_key=args.video_key,
        label_keys=label_keys,
        default_label=args.default_label.lower(),
        valid_labels=valid_labels,
        label_map=label_map,
        prefix_step_sec=args.prefix_step_sec,
        min_prefix_sec=args.min_prefix_sec,
        include_full_prefix=args.include_full_prefix,
        max_prefix_sec=args.max_prefix_sec,
        dataset_build_workers=args.dataset_build_workers,
        min_clip_duration_sec=0.0,
    )
    if args.max_samples > 0:
        dataset.samples = dataset.samples[: args.max_samples]

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    if device.type == "cpu":
        dtype = torch.float32

    print("Loading processor/model...")
    processor = Qwen2_5OmniProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    base_model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        trust_remote_code=True,
        attn_implementation="flash_attention_2" if device.type == "cuda" else "eager",
    )
    model = PeftModel.from_pretrained(base_model.thinker, args.adapter_path)
    model = model.to(device)
    model.eval()

    collator = PrefixAudioFrameCollator(
        processor=processor,
        valid_labels=valid_labels,
        include_dialogue=args.include_dialogue,
        frame_resize_max=args.frame_resize_max,
        use_audio_in_video=args.use_audio_in_video,
        audio_sample_rate=args.audio_sample_rate,
        prefix_cache_dir=args.prefix_cache_dir,
    )

    analyzer = LikelihoodPrefixAnalyzer(
        model=model,
        processor=processor,
        collator=collator,
        valid_labels=valid_labels,
        include_dialogue=args.include_dialogue,
        use_audio_in_video=args.use_audio_in_video,
        device=device,
        dtype=dtype,
    )

    rank, world_size = dist_rank_world_size()
    partial_dir = output_dir / "partial"
    partial_dir.mkdir(parents=True, exist_ok=True)
    partial_path = partial_dir / f"prefix_predictions.rank{rank}.jsonl"
    completed_prefix_ids = load_completed_prefix_ids(partial_dir) if args.resume else set()

    local_samples_all = dataset.samples[rank::world_size]
    local_samples = []
    for sample in local_samples_all:
        prefix_id = str(sample.get("prefix_sample_id", "")).strip()
        if args.resume and prefix_id and prefix_id in completed_prefix_ids:
            continue
        local_samples.append(sample)

    disable_tqdm = rank != 0
    local_prediction_rows: List[Dict[str, Any]] = []
    flush_every = max(1, int(args.save_every))
    batch_size = max(1, int(args.batch_size))
    prefetch_workers = max(0, int(args.prefetch_workers))

    if batch_size > 1:
        # Batched scoring mode
        from tqdm import tqdm as _tqdm
        for start_idx in _tqdm(range(0, len(local_samples), batch_size), desc="Scoring prefixes (batched)", disable=disable_tqdm):
            batch_samples = local_samples[start_idx:start_idx + batch_size]
            try:
                batch_rows = analyzer.score_batch(batch_samples)
                local_prediction_rows.extend(batch_rows)
            except Exception as batch_err:
                # Fall back to single-sample scoring on batch failure
                import logging
                logging.warning(f"Batch scoring failed ({batch_err}), falling back to single-sample for batch starting at {start_idx}")
                for s in batch_samples:
                    try:
                        row = analyzer.score_candidates(s)
                        local_prediction_rows.append(row)
                    except Exception:
                        continue
            if len(local_prediction_rows) >= flush_every:
                append_jsonl(partial_path, local_prediction_rows)
                local_prediction_rows = []
    elif prefetch_workers > 0:
        # Prefetch mode: overlap CPU asset loading with GPU inference
        from concurrent.futures import ThreadPoolExecutor

        def _prefetch_assets(sample):
            """Load audio + frame in background thread."""
            try:
                audio_prefix, frame = collator.prepare_prefix_assets(sample)
                return (audio_prefix, frame), sample, None
            except Exception as e:
                return None, sample, str(e)

        pool = ThreadPoolExecutor(max_workers=prefetch_workers)
        prefetch_queue = []
        idx = 0
        # Submit initial prefetches
        while len(prefetch_queue) < prefetch_workers and idx < len(local_samples):
            prefetch_queue.append(pool.submit(_prefetch_assets, local_samples[idx]))
            idx += 1

        for sample_idx in tqdm(range(len(local_samples)), desc="Scoring prefixes (prefetch)", disable=disable_tqdm):
            # Get prefetched result
            if prefetch_queue:
                future = prefetch_queue.pop(0)
                assets_result, sample, error = future.result()
            else:
                assets_result, sample, error = _prefetch_assets(local_samples[sample_idx])

            # Submit next prefetch
            if idx < len(local_samples):
                prefetch_queue.append(pool.submit(_prefetch_assets, local_samples[idx]))
                idx += 1

            if error is not None:
                continue

            audio_prefix, frame = assets_result
            # Build prompt and score (main thread only)
            prompt_conv, prompt_text = analyzer._build_prompt_conv(sample, audio_prefix, frame)
            model_inputs = analyzer._prepare_model_inputs(prompt_conv, audio_prefix, frame)
            with torch.no_grad():
                outputs = analyzer.model(**model_inputs, use_audio_in_video=analyzer.use_audio_in_video)
                next_token_logits = outputs.logits[0, -1, :]
            probabilities, label_first_token_ids, pred_label, confidence, entropy, margin = analyzer._decode_logits(next_token_logits)
            label_log_likelihoods = {k: float(v) for k, v in probabilities.items()}
            gold_label = canonicalize_prediction(str(sample.get("prefix_target_label", "")), valid_labels)
            gold_prob = float(probabilities.get(gold_label, 0.0))
            row = {
                "sample_id": _extract_sample_id(sample),
                "prefix_sample_id": str(sample.get("prefix_sample_id", "")),
                "clip_id": str(sample.get("clip_id", "")),
                "stream_id": str(sample.get("stream_id", "")),
                "speaker": str(sample.get("speaker", "")),
                "video_path": str(sample.get("resolved_video_path", sample.get("video_path", ""))),
                "dialogue": str(sample.get("dialogue", "") or ""),
                "prefix_sec": float(sample.get("prefix_sec", 0.0) or 0.0),
                "prefix_tag": str(sample.get("prefix_tag", "")),
                "prefix_index": int(sample.get("prefix_index", 0) or 0),
                "clip_duration_sec": float(sample.get("clip_duration_sec", 0.0) or 0.0),
                "is_full_prefix": int(sample.get("is_full_prefix", 0) or 0),
                "gold_label": gold_label,
                "pred_label": pred_label,
                "is_correct": int(pred_label == gold_label),
                "confidence": confidence,
                "entropy": entropy,
                "margin": margin,
                "gold_prob": gold_prob,
                "probabilities": probabilities,
                "label_log_likelihoods": label_log_likelihoods,
                "label_first_token_ids": label_first_token_ids,
                "prompt": prompt_text,
            }
            local_prediction_rows.append(row)
            if len(local_prediction_rows) >= flush_every:
                append_jsonl(partial_path, local_prediction_rows)
                local_prediction_rows = []
        pool.shutdown(wait=True)
    else:
        # Original sequential mode (optimized: no redundant processor call)
        for sample in tqdm(local_samples, desc="Scoring prefixes", disable=disable_tqdm):
            row = analyzer.score_candidates(sample)
            local_prediction_rows.append(row)
            if len(local_prediction_rows) >= flush_every:
                append_jsonl(partial_path, local_prediction_rows)
                local_prediction_rows = []
    if local_prediction_rows:
        append_jsonl(partial_path, local_prediction_rows)
        local_prediction_rows = []

    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    if rank != 0:
        return

    prediction_rows = []
    for path in sorted(partial_dir.glob("prefix_predictions.rank*.jsonl")):
        prediction_rows.extend(load_jsonl_rows(path))
    deduped = {}
    for row in prediction_rows:
        prefix_id = str(row.get("prefix_sample_id", "")).strip()
        if prefix_id:
            deduped[prefix_id] = row
    prediction_rows = sorted(deduped.values(), key=lambda row: (str(row.get("sample_id", "")), float(row.get("prefix_sec", 0.0))))

    prefix_metrics = summarize_prefix_metrics(prediction_rows, valid_labels)
    belief_changes = build_belief_changes(
        prediction_rows,
        entropy_threshold=args.entropy_threshold,
        margin_threshold=args.margin_threshold,
        stable_conf_threshold=args.stable_confidence_threshold,
    )
    belief_change_summary = summarize_belief_changes(belief_changes)

    write_jsonl(output_dir / "prefix_predictions.jsonl", prediction_rows)
    write_json(output_dir / "prefix_metrics.json", prefix_metrics)
    write_jsonl(output_dir / "belief_changes.jsonl", belief_changes)
    write_json(output_dir / "belief_change_summary.json", belief_change_summary)
    export_cases(case_dir, belief_changes, args.export_case_count)

    summary = {
        "num_prefix_samples": len(prediction_rows),
        "num_prefix_buckets": len(prefix_metrics),
        "num_belief_changes": len(belief_changes),
        "adapter_path": args.adapter_path,
        "input_jsonl": args.input_jsonl,
        "labels": valid_labels,
        "resume": bool(args.resume),
        "save_every": int(args.save_every),
        "partial_dir": str(partial_dir),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
