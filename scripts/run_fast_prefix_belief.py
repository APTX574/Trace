#!/usr/bin/env python
"""Offline reproduction -- step 1: per-prefix fast-axis belief trajectories.

Runs the fast LoRA model on every audio prefix of every test utterance (min
prefix sec, then a stride, then the full clip) and writes per-prefix belief
records (prediction / confidence / entropy / margin / label probabilities).
The label probability for each emotion is the softmax over the model's
next-token log-likelihoods of the label's first token -- the same scoring as
the paper's online prefix belief formation.

This mirrors the original per-prefix analysis script but stays single-sample
(no batching) and audio-only, so a reproducer gets the exact reported numbers.
"""

import argparse
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import torch
import torch.nn.functional as F

from stream_ser_open.io import first_text, load_jsonl, parse_csv  # noqa: E402
from stream_ser_open.labels import ALIASES  # noqa: E402
from stream_ser_open.models import entropy, load_qwen_omni, resolve_device  # noqa: E402

try:
    from qwen_omni_utils import process_mm_info  # noqa: E402
except ImportError as exc:
    raise ImportError("qwen_omni_utils is required (from the Qwen2.5-Omni release)") from exc

LABELS_DEFAULT = "neutral,anger,anxiety,sadness,joy,surprise,embarrassment"
SYSTEM_PROMPT = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
    "capable of perceiving auditory and visual inputs, as well as generating text and speech."
)


def str2bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def format_prefix_tag(prefix_sec: float, clip_duration_sec: float, is_full_prefix: bool) -> str:
    if is_full_prefix:
        rounded = round(float(clip_duration_sec), 3)
        return f"{int(round(rounded))}s" if abs(rounded - round(rounded)) < 1e-6 else f"full_{str(rounded).replace('.', 'p')}s"
    rounded = round(float(prefix_sec), 3)
    return f"{int(round(rounded))}s" if abs(rounded - round(rounded)) < 1e-6 else f"{str(rounded).replace('.', 'p')}s"


def build_prefix_prompt(dialogue: str, labels: List[str], prefix_sec: float, clip_duration_sec: float, include_dialogue: bool) -> str:
    labels_str = ", ".join(labels)
    observed_ratio = min(1.0, prefix_sec / clip_duration_sec) if clip_duration_sec > 1e-6 else 0.0
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
    if include_dialogue and (dialogue or "").strip():
        lines.append(f'Current dialogue transcript: "{dialogue}".')
    lines.append("The speaker's emotion in the observed prefix is:")
    return "\n".join(lines)


class PrefixExtractor:
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self._wave: Dict[str, np.ndarray] = {}

    def audio_prefix(self, video_path: str, prefix_sec: float, cache_path: Optional[Path] = None) -> np.ndarray:
        end_idx = max(1, int(round(prefix_sec * self.sample_rate)))
        if cache_path is not None and cache_path.exists():
            return np.load(cache_path).astype(np.float32)
        wave = self._wave.get(video_path)
        if wave is None:
            wave = self._decode_full_audio(video_path)
            self._wave[video_path] = wave
        return np.asarray(wave[: min(end_idx, wave.shape[0])], dtype=np.float32)

    @staticmethod
    def cache_root(sample: Dict[str, Any], prefix_cache_dir: str) -> Optional[Path]:
        import hashlib

        if not prefix_cache_dir:
            return None
        dataset_tag = str(sample.get("dataset_tag", "dataset")).strip() or "dataset"
        prefix_id = str(sample.get("prefix_sample_id", "")).strip()
        if not prefix_id:
            return None
        digest = hashlib.sha1(f"{dataset_tag}::{prefix_id}".encode("utf-8")).hexdigest()
        return Path(prefix_cache_dir) / dataset_tag / digest[:2] / digest

    def _decode_full_audio(self, video_path: str) -> np.ndarray:
        import av
        import librosa

        chunks: List[np.ndarray] = []
        try:
            with av.open(video_path) as container:
                if not container.streams.audio:
                    raise ValueError("no audio stream")
                resampler = av.audio.resampler.AudioResampler(format="fltp", layout="mono", rate=self.sample_rate)
                for frame in container.decode(audio=0):
                    resampled = resampler.resample(frame)
                    if resampled is None:
                        continue
                    frames = resampled if isinstance(resampled, list) else [resampled]
                    for item in frames:
                        chunk = np.asarray(item.to_ndarray(), dtype=np.float32)
                        if chunk.ndim == 2:
                            chunk = chunk[0]
                        chunks.append(chunk.reshape(-1))
            wave = np.concatenate(chunks)
        except Exception:
            wave, _ = librosa.load(video_path, sr=self.sample_rate, mono=True)
            wave = np.asarray(wave, dtype=np.float32)
        if wave.size == 0:
            return np.zeros((max(1, int(round(self.sample_rate))),), dtype=np.float32)
        return wave


def probe_duration_sec(video_path: str) -> float:
    try:
        import av

        with av.open(video_path) as container:
            if container.duration is not None:
                return float(container.duration / av.time_base)
            stream = container.streams.video[0]
            if stream.duration is not None and stream.time_base is not None:
                return float(stream.duration * stream.time_base)
    except Exception:
        pass
    return 0.0


def build_prefix_list(duration: float, min_prefix_sec: float, prefix_step_sec: float, include_full_prefix: bool) -> List[Tuple[float, bool]]:
    if duration <= 1e-6:
        return []
    prefixes: List[Tuple[float, bool]] = []
    current = min_prefix_sec
    while current + 1e-6 < duration:
        prefixes.append((round(current, 6), False))
        current += prefix_step_sec
    if not prefixes:
        return [(round(duration, 6), True)]
    last = prefixes[-1][0]
    if include_full_prefix and abs(last - duration) > 1e-6:
        prefixes.append((round(duration, 6), True))
    elif abs(last - duration) <= 1e-6:
        prefixes[-1] = (prefixes[-1][0], True)
    return prefixes


def normalize_target(raw: Any, labels: List[str], default: str) -> str:
    text = str(raw or "").strip().lower()
    if text in labels:
        return text
    if text in ALIASES and ALIASES[text] in labels:
        return ALIASES[text]
    return default


def score_prefix(
    model,
    processor,
    audio_prefix: np.ndarray,
    prompt_text: str,
    labels: List[str],
    device: torch.device,
) -> Dict[str, float]:
    """Last-token logits scoring: softmax over per-label first-token log-likelihoods."""
    conv = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": audio_prefix},
                {"type": "text", "text": prompt_text},
            ],
        },
    ]
    audios, _, _ = process_mm_info(conv, use_audio_in_video=False)
    text = processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
    inputs = processor(
        text=[text],
        audio=[audios[0] if audios else audio_prefix],
        return_tensors="pt",
        padding=True,
        use_audio_in_video=False,
    ).to(device)
    tokenizer = processor.tokenizer
    first_ids = {}
    for label in labels:
        ids = tokenizer.encode(label, add_special_tokens=False)
        first_ids[label] = int(ids[0]) if ids else -1

    with torch.no_grad():
        logits = model(**inputs, use_audio_in_video=False).logits[0, -1, :]
        log_probs = F.log_softmax(logits, dim=-1)
    lambdas = [float("-inf") if first_ids[l] < 0 else float(log_probs[first_ids[l]].item()) for l in labels]
    probs_t = torch.softmax(torch.tensor(lambdas), dim=0)
    return {label: float(probs_t[i].item()) for i, label in enumerate(labels)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Per-prefix fast-axis belief trajectory (offline reproduction, step 1).")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--adapter-path", required=True)
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--video-root", default="")
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--labels", default=LABELS_DEFAULT)
    parser.add_argument("--default-label", default="neutral")
    parser.add_argument("--video-key", default="video_path")
    parser.add_argument("--label-key", default="target")
    parser.add_argument("--include-dialogue", type=str2bool, default=False)
    parser.add_argument("--prefix-step-sec", type=float, default=1.0)
    parser.add_argument("--min-prefix-sec", type=float, default=1.5)
    parser.add_argument("--include-full-prefix", type=str2bool, default=True)
    parser.add_argument("--audio-sample-rate", type=int, default=16000)
    parser.add_argument("--prefix-cache-dir", default="", help="Pre-extracted audio prefix cache (data/audio_1p5s) for exact reproduction.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-samples", type=int, default=0)
    args = parser.parse_args()

    labels = [x.lower() for x in parse_csv(args.labels)]
    device = resolve_device(args.device)
    # Load the processor from the base model (as the paper's per-prefix analysis
    # script does); the adapter bundle is applied to the thinker only.
    from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor  # noqa: E402
    from peft import PeftModel  # noqa: E402

    processor = Qwen2_5OmniProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    base = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else (torch.float16 if device.type == "cuda" else torch.float32),
        trust_remote_code=True,
        attn_implementation="flash_attention_2" if device.type == "cuda" else "eager",
    ).to(device)
    model = PeftModel.from_pretrained(base.thinker, args.adapter_path).to(device)
    model.eval()

    rows = load_jsonl(args.input_jsonl)
    if args.max_samples > 0:
        rows = rows[: args.max_samples]

    extractor = PrefixExtractor(args.audio_sample_rate)
    records: List[Dict[str, Any]] = []
    for r_idx, sample in enumerate(rows):
        raw_video = first_text([sample.get(args.video_key), sample.get("video_path")])
        video_path = str(Path(args.video_root) / raw_video) if args.video_root else str(raw_video)
        duration = probe_duration_sec(video_path)
        if duration <= 1e-6:
            print(f"[skip] no duration {video_path}", flush=True)
            continue
        rec_id = str(sample.get("sample_id") or sample.get("id") or f"sample_{r_idx}")
        clip_id = str(sample.get("clip_id", "") or "")
        stream_id = str(sample.get("stream_id", "") or "")
        speaker = str(sample.get("speaker", "") or "")
        dialogue = first_text([sample.get("dialogue"), sample.get("text")])
        gold = normalize_target(sample.get(args.label_key) or sample.get("gt_emotion") or sample.get("emotion"), labels, args.default_label)

        prefixes = build_prefix_list(duration, args.min_prefix_sec, args.prefix_step_sec, args.include_full_prefix)
        dataset_tag = Path(args.input_jsonl).stem
        for p_idx, (prefix_sec, is_full) in enumerate(prefixes):
            tag = format_prefix_tag(prefix_sec, duration, is_full)
            prefix_sample_id = f"{rec_id}::prefix_{tag}"
            cache_root = PrefixExtractor.cache_root({**sample, "dataset_tag": dataset_tag, "prefix_sample_id": prefix_sample_id}, args.prefix_cache_dir)
            cache_path = cache_root / "audio.npy" if cache_root is not None else None
            audio_prefix = extractor.audio_prefix(video_path, prefix_sec, cache_path)
            prompt_text = build_prefix_prompt(dialogue, labels, prefix_sec, duration, args.include_dialogue)
            probs = score_prefix(model, processor, audio_prefix, prompt_text, labels, device)
            ordered = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
            pred, conf = ordered[0]
            margin = conf - (ordered[1][1] if len(ordered) > 1 else 0.0)
            records.append({
                "sample_id": rec_id,
                "prefix_sample_id": f"{rec_id}::prefix_{tag}",
                "clip_id": clip_id,
                "stream_id": stream_id,
                "speaker": speaker,
                "video_path": raw_video,
                "dialogue": dialogue,
                "prefix_sec": float(prefix_sec),
                "prefix_tag": tag,
                "prefix_index": int(p_idx),
                "clip_duration_sec": float(duration),
                "is_full_prefix": int(is_full),
                "gold_label": gold,
                "pred_label": pred,
                "is_correct": int(pred == gold),
                "confidence": float(conf),
                "entropy": float(entropy(probs)),
                "margin": float(margin),
                "gold_prob": float(probs.get(gold, 0.0)),
                "probabilities": probs,
                "prompt": prompt_text,
            })
        if (r_idx + 1) % 10 == 0 or r_idx + 1 == len(rows):
            print(f"[{r_idx + 1}/{len(rows)}] {rec_id}", flush=True)

    out = Path(args.output_jsonl)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    full = [r for r in records if r["is_full_prefix"]]
    acc = sum(r["is_correct"] for r in full) / len(full) if full else 0.0
    print(json.dumps({"prefix_records": len(records), "full_accuracy": round(acc, 4), "output": str(out)}, ensure_ascii=False))


if __name__ == "__main__":
    main()