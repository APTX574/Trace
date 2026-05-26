import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
import shutil
import time
import warnings
from collections import defaultdict
from pathlib import Path
import threading
from typing import Any, Dict, List, Optional, Set, Tuple

import av
import librosa
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import (
    Qwen2_5OmniForConditionalGeneration,
    Qwen2_5OmniProcessor,
    Trainer,
    TrainingArguments,
)

from dataset_io import load_jsonl as load_jsonl_rows
from dataset_io import resolve_repo_path
from qwen_omni_utils import process_mm_info
from train_fast_track import (
    BestEvalLoraCallback,
    SYSTEM_PROMPT,
    canonicalize_prediction,
    extract_raw_label,
    load_label_map,
    normalize_label,
    parse_csv,
    save_lora_adapter_bundle,
    str2bool,
)

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


def build_prefix_audio_frame_prompt(
    sample: Dict[str, Any],
    valid_labels: List[str],
    include_dialogue: bool,
) -> str:
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
    def __init__(
        self,
        jsonl_path: str,
        video_key: str,
        label_keys: List[str],
        default_label: str,
        valid_labels: List[str],
        label_map: Dict[str, str],
        prefix_step_sec: float = 1.0,
        min_prefix_sec: float = 1.0,
        include_full_prefix: bool = True,
        max_prefix_sec: float = 0.0,
        dataset_build_workers: int = 0,
        min_clip_duration_sec: float = 0.0,
    ):
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
        return str(resolve_repo_path(str(value))).strip()

    def _normalize_target(self, sample: Dict[str, Any]) -> str:
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
            filtered_prefix_list = [
                (prefix_sec, is_full_prefix)
                for prefix_sec, is_full_prefix in prefix_list
                if is_full_prefix or prefix_sec + 1e-6 >= self.min_clip_duration_sec
            ]
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
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        disable_tqdm = local_rank != 0
        indexed_samples = list(enumerate(raw_samples))
        desc = f"Build prefixes: {Path(self.jsonl_path).name}"

        if self.dataset_build_workers <= 1:
            iterator = indexed_samples
            for sample_index, sample in tqdm(iterator, desc=desc, disable=disable_tqdm):
                out.extend(self._build_prefix_items_for_sample(sample_index, sample))
            return out

        with ThreadPoolExecutor(max_workers=self.dataset_build_workers) as executor:
            iterator = executor.map(lambda pair: self._build_prefix_items_for_sample(pair[0], pair[1]), indexed_samples)
            for items in tqdm(iterator, total=len(indexed_samples), desc=desc, disable=disable_tqdm):
                out.extend(items)
        return out

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx]


class PrefixAudioFrameCollator:
    def __init__(
        self,
        processor: Qwen2_5OmniProcessor,
        valid_labels: List[str],
        include_dialogue: bool,
        frame_resize_max: int = 448,
        use_audio_in_video: bool = False,
        audio_sample_rate: int = 16000,
        prefix_cache_dir: str = "",
        prefix_loss_weighting: str = "observation_ratio",
    ):
        self.processor = processor
        self.valid_labels = valid_labels
        self.include_dialogue = include_dialogue
        self.frame_resize_max = max(64, int(frame_resize_max))
        self.use_audio_in_video = bool(use_audio_in_video)
        self.audio_sample_rate = int(audio_sample_rate)
        self.prefix_cache_dir = Path(prefix_cache_dir).expanduser() if str(prefix_cache_dir).strip() else None
        self.prefix_loss_weighting = str(prefix_loss_weighting or "observation_ratio").strip().lower()
        self.ignore_index = -100
        self._audio_cache: Dict[str, np.ndarray] = {}
        self._frame_cache: Dict[Tuple[str, int], Image.Image] = {}
        self._video_reader_cache: Dict[str, Any] = {}
        self._audio_warning_cache: Set[str] = set()
        self._frame_warning_cache: Set[str] = set()

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
        if audio_path is None or frame_path is None:
            return None, None
        if not audio_path.exists() or not frame_path.exists():
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

    def prepare_prefix_assets(self, sample: Dict[str, Any]) -> Tuple[np.ndarray, Image.Image]:
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

    def clear_runtime_caches(self) -> None:
        self._audio_cache.clear()
        self._frame_cache.clear()
        self._video_reader_cache.clear()

    def _decode_audio_with_av(self, video_path: str) -> np.ndarray:
        chunks: List[np.ndarray] = []
        with av.open(video_path) as container:
            if not container.streams.audio:
                raise ValueError(f"No audio stream found in {video_path}")

            resampler = av.audio.resampler.AudioResampler(
                format="fltp",
                layout="mono",
                rate=self.audio_sample_rate,
            )
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

    def _load_audio_waveform_librosa(self, video_path: str) -> np.ndarray:
        waveform, _ = librosa.load(video_path, sr=self.audio_sample_rate, mono=True)
        return np.asarray(waveform, dtype=np.float32)

    def _warn_audio_issue(self, video_path: str, message: str) -> None:
        if video_path in self._audio_warning_cache:
            return
        self._audio_warning_cache.add(video_path)
        print(f"[AudioFallback] {video_path}: {message}")

    def _warn_frame_issue(self, video_path: str, message: str) -> None:
        if video_path in self._frame_warning_cache:
            return
        self._frame_warning_cache.add(video_path)
        print(f"[FrameFallback] {video_path}: {message}")

    def _load_audio_waveform(self, video_path: str) -> np.ndarray:
        cached = self._audio_cache.get(video_path)
        if cached is not None:
            return cached

        av_error: Optional[Exception] = None
        librosa_error: Optional[Exception] = None

        try:
            waveform = self._decode_audio_with_av(video_path)
        except Exception as exc:
            av_error = exc
            try:
                waveform = self._load_audio_waveform_librosa(video_path)
            except Exception as fallback_exc:
                librosa_error = fallback_exc
                raise RuntimeError(
                    f"Failed to decode audio from {video_path} with av ({av_error}) "
                    f"and librosa ({librosa_error})"
                ) from fallback_exc

        self._audio_cache[video_path] = waveform
        return waveform

    def _slice_audio_prefix(self, video_path: str, prefix_sec: float) -> np.ndarray:
        end_idx = int(round(prefix_sec * self.audio_sample_rate))
        end_idx = max(1, end_idx)

        try:
            waveform = self._load_audio_waveform(video_path)
        except Exception as exc:
            self._warn_audio_issue(video_path, f"{exc}. Using silence for this sample.")
            return np.zeros((end_idx,), dtype=np.float32)

        end_idx = min(end_idx, waveform.shape[0])
        return np.asarray(waveform[:end_idx], dtype=np.float32)

    def _get_video_reader(self, video_path: str):
        if decord is None:
            raise ImportError("decord is required for prefix frame extraction but is not installed.")
        reader = self._video_reader_cache.get(video_path)
        if reader is None:
            reader = decord.VideoReader(video_path)
            self._video_reader_cache[video_path] = reader
        return reader

    def _extract_frame_with_av(self, video_path: str, prefix_sec: float, clip_duration_sec: float) -> Image.Image:
        with av.open(video_path) as container:
            if not container.streams.video:
                raise ValueError(f"No video stream found in {video_path}")

            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
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
        new_width = max(1, int(round(width * scale)))
        new_height = max(1, int(round(height * scale)))
        return image.resize((new_width, new_height))

    def _extract_prefix_frame(self, video_path: str, prefix_sec: float, clip_duration_sec: float) -> Image.Image:
        cache_key = (video_path, int(round(prefix_sec * 1000.0)))
        cached = self._frame_cache.get(cache_key)
        if cached is not None:
            return cached

        try:
            reader = self._get_video_reader(video_path)
            total_frames = len(reader)
            fps = float(reader.get_avg_fps())
            if total_frames <= 0 or fps <= 0:
                raise ValueError(
                    f"Invalid video reader state for {video_path}: total_frames={total_frames}, fps={fps}"
                )

            safe_prefix = max(0.0, min(prefix_sec, clip_duration_sec))
            frame_time = max(0.0, safe_prefix - 1e-3)
            frame_idx = int(round(frame_time * fps))
            frame_idx = max(0, min(frame_idx, total_frames - 1))
            frame_np = reader[frame_idx].asnumpy()
            image = Image.fromarray(frame_np).convert("RGB")
        except Exception as exc:
            self._warn_frame_issue(video_path, f"{exc}. Falling back to av frame decode.")
            image = self._extract_frame_with_av(video_path, prefix_sec, clip_duration_sec)

        image = self._resize_frame(image)
        self._frame_cache[cache_key] = image
        return image

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        full_texts: List[str] = []
        prompt_texts: List[str] = []
        audios_list: List[np.ndarray] = []
        images_list: List[Image.Image] = []
        loss_weights: List[float] = []

        for sample in examples:
            target = str(sample["prefix_target_label"])
            prefix_sec = float(sample.get("prefix_sec", 0.0) or 0.0)
            clip_duration_sec = float(sample.get("clip_duration_sec", 0.0) or 0.0)
            if self.prefix_loss_weighting == "none" or clip_duration_sec <= 1e-6:
                loss_weight = 1.0
            else:
                loss_weight = max(1e-6, min(1.0, prefix_sec / clip_duration_sec))
            loss_weights.append(float(loss_weight))
            audio_prefix, frame = self.prepare_prefix_assets(sample)

            prompt_text = build_prefix_audio_frame_prompt(
                sample=sample,
                valid_labels=self.valid_labels,
                include_dialogue=self.include_dialogue,
            )
            content = [
                {"type": "image", "image": frame},
                {"type": "text", "text": prompt_text},
                {"type": "audio", "audio": audio_prefix},
            ]

            full_conv = [
                {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                {"role": "user", "content": content},
                {"role": "assistant", "content": [{"type": "text", "text": target}]},
            ]
            prompt_conv = [
                {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                {"role": "user", "content": content},
            ]

            audios, images, _ = process_mm_info(full_conv, use_audio_in_video=self.use_audio_in_video)
            audios_list.append(audios[0] if audios else audio_prefix)
            images_list.append(images[0] if images else frame)
            full_texts.append(self.processor.apply_chat_template(full_conv, tokenize=False))
            prompt_texts.append(
                self.processor.apply_chat_template(
                    prompt_conv,
                    add_generation_prompt=True,
                    tokenize=False,
                )
            )

        batch = self.processor(
            text=full_texts,
            audio=audios_list,
            images=images_list,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=self.use_audio_in_video,
        )
        batch_prompt = self.processor(
            text=prompt_texts,
            audio=audios_list,
            images=images_list,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=self.use_audio_in_video,
        )

        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        prompt_input_ids = batch_prompt["input_ids"]
        prompt_attention_mask = batch_prompt["attention_mask"]

        labels = input_ids.clone()
        labels[attention_mask == 0] = self.ignore_index

        bsz, seq_len = input_ids.shape
        full_lens = attention_mask.sum(dim=1)
        prompt_lens = prompt_attention_mask.sum(dim=1)

        for i in range(bsz):
            full_len = int(full_lens[i].item())
            prompt_len = int(prompt_lens[i].item())
            if prompt_len > full_len:
                raise ValueError(f"Sample {i}: prompt_len ({prompt_len}) > full_len ({full_len})")
            if prompt_len == full_len:
                raise ValueError(f"Sample {i}: prompt_len == full_len, no supervised answer tokens found.")

            pad_len = seq_len - full_len
            prompt_start = pad_len
            prompt_end = pad_len + prompt_len
            full_prefix = input_ids[i, prompt_start:prompt_end]
            prompt_valid = prompt_input_ids[i, -prompt_len:]
            if not torch.equal(full_prefix, prompt_valid):
                raise ValueError("Prompt valid tokens are not equal to full prefix tokens.")
            labels[i, prompt_start:prompt_end] = self.ignore_index

        batch["labels"] = labels
        batch["prefix_loss_weight"] = torch.tensor(loss_weights, dtype=torch.float32)
        return batch


class PrefixAudioFrameTrainer(Trainer):
    def __init__(
        self,
        *args,
        processor: Qwen2_5OmniProcessor,
        valid_labels: List[str],
        include_dialogue: bool,
        frame_resize_max: int,
        use_audio_in_video: bool,
        prefix_loss_weighting: str = "observation_ratio",
        prefix_cache_dir: str = "",
        eval_records_dir: str = "",
        eval_records_max_samples: int = 0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.processor = processor
        self.valid_labels = valid_labels
        self.include_dialogue = include_dialogue
        self.frame_resize_max = frame_resize_max
        self.use_audio_in_video = use_audio_in_video
        self.prefix_loss_weighting = str(prefix_loss_weighting or "observation_ratio").strip().lower()
        self.prefix_cache_dir = str(prefix_cache_dir or "").strip()
        self.eval_records_dir = str(eval_records_dir or "").strip()
        self.eval_records_max_samples = max(0, int(eval_records_max_samples))
        self.eval_collator = PrefixAudioFrameCollator(
            processor=processor,
            valid_labels=valid_labels,
            include_dialogue=include_dialogue,
            frame_resize_max=frame_resize_max,
            use_audio_in_video=use_audio_in_video,
            prefix_cache_dir=self.prefix_cache_dir,
            prefix_loss_weighting=self.prefix_loss_weighting,
        )

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        prefix_loss_weight = inputs.pop("prefix_loss_weight", None)
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        if labels is None or prefix_loss_weight is None or self.prefix_loss_weighting == "none":
            loss = getattr(outputs, "loss", None)
            if loss is None:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100,
                )
            return (loss, outputs) if return_outputs else loss

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        token_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="none",
            ignore_index=-100,
        ).view(shift_labels.shape)
        valid_tokens = (shift_labels != -100).to(token_loss.dtype)
        per_sample_loss = (token_loss * valid_tokens).sum(dim=1) / valid_tokens.sum(dim=1).clamp_min(1.0)
        weights = prefix_loss_weight.to(per_sample_loss.device, dtype=per_sample_loss.dtype).clamp_min(1e-6)
        loss = (per_sample_loss * weights).sum() / weights.sum().clamp_min(1e-6)
        return (loss, outputs) if return_outputs else loss

    def _dist_rank_world_size(self) -> Tuple[int, int]:
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

    def _maybe_save_eval_records(self, rows: List[Dict[str, Any]], metric_key_prefix: str) -> None:
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

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        batch_size = self.args.per_device_eval_batch_size
        device = self.args.device
        rank, world_size = self._dist_rank_world_size()

        self.model.eval()
        old_padding_side = self.processor.tokenizer.padding_side
        self.processor.tokenizer.padding_side = "left"

        requested_prefix_secs = [1.5 + float(idx) for idx in range(10)]
        total_count = len(eval_dataset)
        eval_indices = list(range(rank, total_count, world_size))
        local_count = len(eval_indices)
        correct_count = 0
        requested_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: {"correct": 0, "total": 0})
        eval_rows: List[Dict[str, Any]] = []
        model_to_gen = self.accelerator.unwrap_model(self.model)

        try:
            with torch.no_grad():
                disable_tqdm = rank > 0
                for start in tqdm(
                    range(0, local_count, batch_size),
                    desc="Prefix fast-track generative eval",
                    disable=disable_tqdm,
                ):
                    batch_indices = eval_indices[start:start + batch_size]
                    batch_samples = [eval_dataset[idx] for idx in batch_indices]

                    batch_texts: List[str] = []
                    batch_audios: List[np.ndarray] = []
                    batch_images: List[Image.Image] = []
                    batch_targets: List[str] = []
                    batch_meta: List[Dict[str, Any]] = []

                    for sample_idx, sample in zip(batch_indices, batch_samples):
                        video_path = str(sample["resolved_video_path"])
                        prefix_sec = float(sample["prefix_sec"])
                        clip_duration_sec = float(sample["clip_duration_sec"])
                        prefix_tag = str(sample["prefix_tag"])
                        target = str(sample["prefix_target_label"])
                        prompt_text = build_prefix_audio_frame_prompt(
                            sample=sample,
                            valid_labels=self.valid_labels,
                            include_dialogue=self.include_dialogue,
                        )
                        audio_prefix, frame = self.eval_collator.prepare_prefix_assets(sample)
                        content = [
                            {"type": "image", "image": frame},
                            {"type": "text", "text": prompt_text},
                            {"type": "audio", "audio": audio_prefix},
                        ]
                        conv = [
                            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                            {"role": "user", "content": content},
                        ]

                        audios, images, _ = process_mm_info(conv, use_audio_in_video=self.use_audio_in_video)
                        batch_audios.append(audios[0] if audios else audio_prefix)
                        batch_images.append(images[0] if images else frame)
                        batch_targets.append(target)
                        batch_meta.append(
                            {
                                "eval_index": int(sample_idx),
                                "id": str(sample.get("prefix_sample_id", sample_idx)),
                                "video_path": video_path,
                                "prefix_sec": prefix_sec,
                                "clip_duration_sec": clip_duration_sec,
                                "prefix_tag": prefix_tag,
                                "is_full_prefix": int(sample.get("is_full_prefix", 0) or 0),
                                "prompt": prompt_text,
                            }
                        )
                        batch_texts.append(
                            self.processor.apply_chat_template(
                                conv,
                                add_generation_prompt=True,
                                tokenize=False,
                            )
                        )

                    inputs = self.processor(
                        text=batch_texts,
                        audio=batch_audios,
                        images=batch_images,
                        return_tensors="pt",
                        padding=True,
                        use_audio_in_video=self.use_audio_in_video,
                    ).to(device)

                    generated_ids = model_to_gen.generate(
                        **inputs,
                        max_new_tokens=8,
                        temperature=0.1,
                        pad_token_id=self.processor.tokenizer.pad_token_id,
                        eos_token_id=self.processor.tokenizer.eos_token_id,
                        use_audio_in_video=self.use_audio_in_video,
                    )
                    generated_trim = [
                        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                    ]
                    preds = self.processor.batch_decode(
                        generated_trim,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )

                    for idx, (pred_text, true_label) in enumerate(zip(preds, batch_targets)):
                        pred_clean = canonicalize_prediction(pred_text, self.valid_labels)
                        true_clean = canonicalize_prediction(true_label, self.valid_labels)
                        match = int(pred_clean == true_clean)
                        correct_count += match
                        prefix_sec = float(batch_meta[idx]["prefix_sec"])
                        if int(batch_meta[idx]["is_full_prefix"]) == 1:
                            requested_stats["full_prefix"]["correct"] += match
                            requested_stats["full_prefix"]["total"] += 1
                        for target_sec in requested_prefix_secs:
                            if abs(prefix_sec - target_sec) <= 1e-6:
                                stat_key = f"prefix_{_format_metric_suffix_from_sec(target_sec)}"
                                requested_stats[stat_key]["correct"] += match
                                requested_stats[stat_key]["total"] += 1
                                break
                        eval_rows.append(
                            {
                                **batch_meta[idx],
                                "output": pred_text,
                                "output_label": pred_clean,
                                "target": true_label,
                                "target_label": true_clean,
                                "match": match,
                            }
                        )
        finally:
            self.processor.tokenizer.padding_side = old_padding_side
            self.model.train()

        if world_size > 1:
            stats = torch.tensor([correct_count, local_count], device=device, dtype=torch.long)
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
            correct_total = int(stats[0].item())
            total_eval = int(stats[1].item())

            stats_payload = {k: {"correct": v["correct"], "total": v["total"]} for k, v in requested_stats.items()}
            gathered_stats: List[Optional[Dict[str, Dict[str, int]]]] = [None for _ in range(world_size)]
            dist.all_gather_object(gathered_stats, stats_payload)
            merged_requested_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: {"correct": 0, "total": 0})
            for payload in gathered_stats:
                if not payload:
                    continue
                for key, value in payload.items():
                    merged_requested_stats[key]["correct"] += int(value.get("correct", 0))
                    merged_requested_stats[key]["total"] += int(value.get("total", 0))
            requested_stats = merged_requested_stats
        else:
            correct_total = correct_count
            total_eval = local_count

        accuracy = correct_total / total_eval if total_eval > 0 else 0.0
        metrics: Dict[str, float] = {f"{metric_key_prefix}_accuracy": accuracy}
        for target_sec in requested_prefix_secs:
            stat_key = f"prefix_{_format_metric_suffix_from_sec(target_sec)}"
            bucket_total = int(requested_stats[stat_key]["total"])
            bucket_correct = int(requested_stats[stat_key]["correct"])
            metrics[f"{metric_key_prefix}_accuracy_{stat_key}"] = (
                bucket_correct / bucket_total if bucket_total > 0 else 0.0
            )
        full_total = int(requested_stats["full_prefix"]["total"])
        full_correct = int(requested_stats["full_prefix"]["correct"])
        metrics[f"{metric_key_prefix}_accuracy_full_prefix"] = (
            full_correct / full_total if full_total > 0 else 0.0
        )

        if rank == 0:
            summary_parts = [f"all={accuracy:.4f}"]
            for target_sec in requested_prefix_secs:
                stat_key = f"prefix_{_format_metric_suffix_from_sec(target_sec)}"
                metric_name = f"{metric_key_prefix}_accuracy_{stat_key}"
                summary_parts.append(f"{_format_metric_suffix_from_sec(target_sec)}={metrics[metric_name]:.4f}")
            summary_parts.append(f"full={metrics[f'{metric_key_prefix}_accuracy_full_prefix']:.4f}")
            print(f"[EvalSummary] {' | '.join(summary_parts)}")

        self.log(metrics)
        self.control = self.callback_handler.on_evaluate(
            self.args,
            self.state,
            self.control,
            metrics,
        )

        if self.eval_records_dir:
            eval_rows = self._gather_eval_rows(eval_rows)
            if self.eval_records_max_samples > 0:
                eval_rows = eval_rows[:self.eval_records_max_samples]
            self._maybe_save_eval_records(eval_rows, metric_key_prefix)

        return metrics


def _process_rank_world_size() -> Tuple[int, int]:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    return rank, world_size


def _format_float(value: float) -> str:
    return f"{float(value):.3f}"


def _format_metric_suffix_from_sec(value: float) -> str:
    rounded = round(float(value), 3)
    if abs(rounded - round(rounded)) < 1e-6:
        return f"{int(round(rounded))}s"
    return f"{str(rounded).replace('.', 'p')}s"


def _percentile(sorted_values: List[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    pos = max(0.0, min(1.0, q)) * (len(sorted_values) - 1)
    low = int(math.floor(pos))
    high = int(math.ceil(pos))
    if low == high:
        return float(sorted_values[low])
    frac = pos - low
    return float(sorted_values[low] * (1.0 - frac) + sorted_values[high] * frac)


def summarize_prefix_dataset(split_name: str, dataset: PrefixAudioFrameDataset) -> None:
    prefix_samples = dataset.samples
    prefix_total = len(prefix_samples)
    print(f"[{split_name}] prefix samples: {prefix_total}")
    if prefix_total == 0:
        return

    unique_clips = {
        str(sample.get("resolved_video_path", "")).strip()
        for sample in prefix_samples
        if str(sample.get("resolved_video_path", "")).strip()
    }
    prefix_secs = sorted(float(sample.get("prefix_sec", 0.0) or 0.0) for sample in prefix_samples)
    clip_durations = sorted(float(sample.get("clip_duration_sec", 0.0) or 0.0) for sample in prefix_samples)

    clip_prefix_counts: Dict[str, int] = defaultdict(int)
    prefix_int_buckets: Dict[int, int] = defaultdict(int)
    clip_int_buckets: Dict[int, int] = defaultdict(int)
    label_counts: Dict[str, int] = defaultdict(int)
    full_prefix_count = 0

    for sample in prefix_samples:
        clip_key = str(sample.get("resolved_video_path", "")).strip()
        if clip_key:
            clip_prefix_counts[clip_key] += 1
        prefix_sec = float(sample.get("prefix_sec", 0.0) or 0.0)
        clip_duration_sec = float(sample.get("clip_duration_sec", 0.0) or 0.0)
        prefix_int_buckets[int(math.floor(prefix_sec))] += 1
        clip_int_buckets[int(math.floor(clip_duration_sec))] += 1
        label_counts[str(sample.get("prefix_target_label", ""))] += 1
        if int(sample.get("is_full_prefix", 0) or 0):
            full_prefix_count += 1

    clip_count = len(unique_clips)
    clip_prefix_values = sorted(clip_prefix_counts.values())
    avg_prefix_per_clip = prefix_total / clip_count if clip_count > 0 else 0.0

    def compact_bucket_str(bucket_counts: Dict[int, int], max_items: int = 20) -> str:
        items = sorted(bucket_counts.items())
        if len(items) <= max_items:
            return ", ".join(f"{sec}s:{count}" for sec, count in items)
        head = items[:10]
        tail = items[-10:]
        return (
            ", ".join(f"{sec}s:{count}" for sec, count in head)
            + ", ..., "
            + ", ".join(f"{sec}s:{count}" for sec, count in tail)
        )

    top_labels = sorted(label_counts.items(), key=lambda item: (-item[1], item[0]))
    label_summary = ", ".join(f"{label}:{count}" for label, count in top_labels)

    print(
        f"[{split_name}] clips: {clip_count} | full-prefix samples: {full_prefix_count} | "
        f"avg prefixes/clip: {_format_float(avg_prefix_per_clip)} | "
        f"min/max prefixes per clip: {min(clip_prefix_values)}/{max(clip_prefix_values)}"
    )
    print(
        f"[{split_name}] prefix_sec min/mean/max: {_format_float(prefix_secs[0])}/"
        f"{_format_float(sum(prefix_secs) / len(prefix_secs))}/{_format_float(prefix_secs[-1])} | "
        f"p50/p90/p99: {_format_float(_percentile(prefix_secs, 0.50))}/"
        f"{_format_float(_percentile(prefix_secs, 0.90))}/"
        f"{_format_float(_percentile(prefix_secs, 0.99))}"
    )
    print(
        f"[{split_name}] clip_duration_sec min/mean/max: {_format_float(clip_durations[0])}/"
        f"{_format_float(sum(clip_durations) / len(clip_durations))}/{_format_float(clip_durations[-1])} | "
        f"p50/p90/p99: {_format_float(_percentile(clip_durations, 0.50))}/"
        f"{_format_float(_percentile(clip_durations, 0.90))}/"
        f"{_format_float(_percentile(clip_durations, 0.99))}"
    )
    print(f"[{split_name}] prefix duration buckets (floor sec): {compact_bucket_str(prefix_int_buckets)}")
    print(f"[{split_name}] clip duration buckets (floor sec): {compact_bucket_str(clip_int_buckets)}")
    print(f"[{split_name}] label distribution: {label_summary}")


def _chunk_list(items: List[Dict[str, Any]], chunk_size: int) -> List[List[Dict[str, Any]]]:
    if chunk_size <= 0:
        chunk_size = 1
    return [items[idx:idx + chunk_size] for idx in range(0, len(items), chunk_size)]


def _build_prefix_cache_chunk(
    samples: List[Dict[str, Any]],
    frame_resize_max: int,
    audio_sample_rate: int,
    prefix_cache_dir: str,
    thread_local_state: threading.local,
) -> int:
    collator = getattr(thread_local_state, "collator", None)
    if collator is None:
        collator = PrefixAudioFrameCollator(
            processor=None,
            valid_labels=[],
            include_dialogue=False,
            frame_resize_max=frame_resize_max,
            use_audio_in_video=False,
            audio_sample_rate=audio_sample_rate,
            prefix_cache_dir=prefix_cache_dir,
        )
        thread_local_state.collator = collator

    built = 0
    for sample in samples:
        collator.prepare_prefix_assets(sample)
        built += 1
    collator.clear_runtime_caches()
    return built


def build_prefix_disk_cache(
    datasets: List[Tuple[str, PrefixAudioFrameDataset]],
    collator: PrefixAudioFrameCollator,
    cache_dir: str,
    rebuild_cache: bool,
    cache_workers: int = 0,
) -> None:
    cache_root = Path(cache_dir)
    ready_flag = cache_root / ".cache_ready"
    rank, world_size = _process_rank_world_size()

    if rank == 0:
        if rebuild_cache and cache_root.exists():
            shutil.rmtree(cache_root)
        cache_root.mkdir(parents=True, exist_ok=True)

        if ready_flag.exists() and not rebuild_cache:
            print(f"[PrefixCache] Reusing existing cache: {cache_root}")
            return

        if ready_flag.exists():
            ready_flag.unlink()

        for split_name, dataset in datasets:
            if cache_workers <= 1:
                for sample in tqdm(dataset.samples, desc=f"Build prefix cache: {split_name}"):
                    collator.prepare_prefix_assets(sample)
            else:
                samples = list(dataset.samples)
                chunk_size = max(1, math.ceil(len(samples) / (cache_workers * 8)))
                chunks = _chunk_list(samples, chunk_size)
                thread_local_state = threading.local()
                with ThreadPoolExecutor(max_workers=cache_workers) as executor:
                    iterator = executor.map(
                        lambda chunk: _build_prefix_cache_chunk(
                            chunk,
                            collator.frame_resize_max,
                            collator.audio_sample_rate,
                            str(cache_root),
                            thread_local_state,
                        ),
                        chunks,
                    )
                    for _ in tqdm(iterator, total=len(chunks), desc=f"Build prefix cache: {split_name}"):
                        pass

        collator.clear_runtime_caches()
        ready_flag.write_text("ready\n", encoding="utf-8")
        print(f"[PrefixCache] Ready: {cache_root}")
        return

    if world_size <= 1:
        return

    wait_seconds = 0.0
    while not ready_flag.exists():
        time.sleep(1.0)
        wait_seconds += 1.0
        if wait_seconds > 3600:
            raise TimeoutError(f"Timed out waiting for prefix cache ready flag: {ready_flag}")

    print(f"[PrefixCache] Rank {rank} detected ready cache: {cache_root}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a prefix-based fast track using audio prefixes and a single frame."
    )
    parser.add_argument("--local_rank", "--local-rank", type=int, default=-1)
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--train-jsonl", type=str, required=True)
    parser.add_argument("--val-jsonl", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)

    parser.add_argument(
        "--labels",
        type=str,
        default="neutral,anger,anxiety,sadness,joy,surprise,embarrassment",
        help="Comma-separated valid label set.",
    )
    parser.add_argument("--default-label", type=str, default="neutral")
    parser.add_argument("--label-map-json", type=str, default="")
    parser.add_argument(
        "--label-keys",
        type=str,
        default="mapped_emotion,gt_emotion,emotion,target",
        help="Comma-separated label candidate keys in priority order.",
    )
    parser.add_argument("--video-key", type=str, default="video_path")
    parser.add_argument("--include-dialogue", type=str2bool, default=True)

    parser.add_argument("--prefix-step-sec", type=float, default=1.0)
    parser.add_argument("--min-prefix-sec", type=float, default=1.5)
    parser.add_argument("--include-full-prefix", type=str2bool, default=True)
    parser.add_argument("--max-prefix-sec", type=float, default=0.0)
    parser.add_argument(
        "--prefix-loss-weighting",
        choices=["observation_ratio", "none"],
        default="observation_ratio",
        help="Weight each prefix CE loss by prefix_sec / clip_duration_sec, or use uniform prefix loss.",
    )
    parser.add_argument(
        "--train-min-clip-sec",
        type=float,
        default=0.0,
        help=(
            "Only for training: drop non-full prefix samples shorter than this threshold, "
            "but still keep each clip's full-prefix sample."
        ),
    )
    parser.add_argument("--dataset-build-workers", type=int, default=8)
    parser.add_argument("--frame-resize-max", type=int, default=448)
    parser.add_argument("--audio-sample-rate", type=int, default=16000)
    parser.add_argument("--use-audio-in-video", type=str2bool, default=False)
    parser.add_argument("--prefix-cache-dir", type=str, default="")
    parser.add_argument("--rebuild-prefix-cache", type=str2bool, default=False)
    parser.add_argument("--prefix-cache-workers", type=int, default=8)

    parser.add_argument("--num-train-epochs", type=int, default=10)
    parser.add_argument("--per-device-train-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--dataloader-num-workers", type=int, default=0)

    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--deepspeed-config", type=str, default="")
    parser.add_argument("--run-name", type=str, default="qwen-omni-fast-prefix-audio-frame")

    parser.add_argument("--wandb-project", type=str, default="")
    parser.add_argument("--wandb-api-key", type=str, default="")
    parser.add_argument("--eval-records-dir", type=str, default="")
    parser.add_argument("--eval-records-max-samples", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix_cache_dir = str(args.prefix_cache_dir).strip() or str(output_dir / "prefix_media_cache")

    if args.wandb_project:
        os.environ["WANDB_PROJECT"] = args.wandb_project
    if args.wandb_api_key:
        os.environ["WANDB_API_KEY"] = args.wandb_api_key

    valid_labels = [x.lower() for x in parse_csv(args.labels)]
    if args.default_label.lower() not in valid_labels:
        raise ValueError("default-label must be in --labels")
    label_map = load_label_map(args.label_map_json)
    label_keys = parse_csv(args.label_keys)

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if local_rank == 0:
        print("Loading processor/model...")

    processor = Qwen2_5OmniProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )
    model = model.thinker
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

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
        print("Loading prefix datasets...")

    train_dataset = PrefixAudioFrameDataset(
        jsonl_path=args.train_jsonl,
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
        min_clip_duration_sec=args.train_min_clip_sec,
    )
    val_dataset = PrefixAudioFrameDataset(
        jsonl_path=args.val_jsonl,
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

    collator = PrefixAudioFrameCollator(
        processor=processor,
        valid_labels=valid_labels,
        include_dialogue=args.include_dialogue,
        frame_resize_max=args.frame_resize_max,
        use_audio_in_video=args.use_audio_in_video,
        audio_sample_rate=args.audio_sample_rate,
        prefix_cache_dir=prefix_cache_dir,
        prefix_loss_weighting=args.prefix_loss_weighting,
    )

    if local_rank == 0:
        print(f"Building / reusing prefix cache at: {prefix_cache_dir}")
    build_prefix_disk_cache(
        datasets=[("train", train_dataset), ("val", val_dataset)],
        collator=collator,
        cache_dir=prefix_cache_dir,
        rebuild_cache=args.rebuild_prefix_cache,
        cache_workers=args.prefix_cache_workers,
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
        save_strategy="no",
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
    if "eval_strategy" in init_vars:
        training_kwargs["eval_strategy"] = "epoch"
    else:
        training_kwargs["evaluation_strategy"] = "epoch"
    if args.deepspeed_config:
        training_kwargs["deepspeed"] = args.deepspeed_config
    training_args = TrainingArguments(**training_kwargs)

    best_lora_callback = BestEvalLoraCallback(
        processor=processor,
        save_dir=str(output_dir / "fast_prefix_audio_frame_lora"),
        metric_name="eval_accuracy",
        greater_is_better=True,
    )

    trainer = PrefixAudioFrameTrainer(
        processor=processor,
        valid_labels=valid_labels,
        include_dialogue=args.include_dialogue,
        frame_resize_max=args.frame_resize_max,
        use_audio_in_video=args.use_audio_in_video,
        prefix_loss_weighting=args.prefix_loss_weighting,
        prefix_cache_dir=prefix_cache_dir,
        eval_records_dir=args.eval_records_dir,
        eval_records_max_samples=args.eval_records_max_samples,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
        callbacks=[best_lora_callback],
    )

    if local_rank == 0:
        summarize_prefix_dataset("train", train_dataset)
        summarize_prefix_dataset("val", val_dataset)
        print("Start prefix fast-track LoRA training...")

    trainer.train()

    final_dir = output_dir / "fast_prefix_audio_frame_lora"
    if local_rank == 0:
        config_dump = {
            "labels": valid_labels,
            "default_label": args.default_label.lower(),
            "label_keys": label_keys,
            "video_key": args.video_key,
            "include_dialogue": args.include_dialogue,
            "prefix_step_sec": args.prefix_step_sec,
            "min_prefix_sec": args.min_prefix_sec,
            "include_full_prefix": args.include_full_prefix,
            "max_prefix_sec": args.max_prefix_sec,
            "prefix_loss_weighting": args.prefix_loss_weighting,
            "train_min_clip_sec": args.train_min_clip_sec,
            "dataset_build_workers": args.dataset_build_workers,
            "frame_resize_max": args.frame_resize_max,
            "audio_sample_rate": args.audio_sample_rate,
            "use_audio_in_video": args.use_audio_in_video,
            "prefix_cache_dir": prefix_cache_dir,
            "rebuild_prefix_cache": args.rebuild_prefix_cache,
            "prefix_cache_workers": args.prefix_cache_workers,
            "label_map": label_map,
        }
        with open(output_dir / "fast_prefix_audio_frame_config.json", "w", encoding="utf-8") as f:
            json.dump(config_dump, f, ensure_ascii=False, indent=2)

        if final_dir.exists():
            print(
                f"Prefix fast-track training complete. LoRA adapter is available at {final_dir} "
                f"(save_reason={best_lora_callback.save_reason}, best_metric={best_lora_callback.best_metric})"
            )
        else:
            print(
                "Prefix fast-track training complete, but no LoRA adapter was saved. "
                f"Last skip reason: {best_lora_callback.last_skip_reason}"
            )


if __name__ == "__main__":
    main()
