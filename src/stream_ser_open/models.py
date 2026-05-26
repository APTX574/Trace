import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from peft import PeftModel
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

from .prompts import SYSTEM_PROMPT, build_fast_prompt

try:
    from qwen_omni_utils import process_mm_info
except ImportError as exc:
    raise ImportError("qwen_omni_utils is required by Qwen2.5-Omni multimodal inference.") from exc


def resolve_device(name: str) -> torch.device:
    if name.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(name)


def resolve_dtype(device: torch.device) -> torch.dtype:
    if device.type != "cuda":
        return torch.float32
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def load_qwen_omni(model_path: str, adapter_path: str, device: torch.device) -> Tuple[Qwen2_5OmniProcessor, torch.nn.Module]:
    processor_source = adapter_path if adapter_path else model_path
    processor = Qwen2_5OmniProcessor.from_pretrained(processor_source, trust_remote_code=True)
    base = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=resolve_dtype(device),
        trust_remote_code=True,
        attn_implementation="flash_attention_2" if device.type == "cuda" else "eager",
    ).to(device)
    model = base.thinker
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path).to(device)
    model.eval()
    return processor, model


def _resolve_video_path(sample: Dict[str, Any], video_key: str) -> str:
    candidates: List[Any] = [sample.get(video_key)]
    if video_key != "video_path":
        candidates.append(sample.get("video_path"))
    if isinstance(sample.get("original"), dict):
        original = sample["original"]
        candidates.extend([original.get(video_key), original.get("video_path")])
    for item in candidates:
        if isinstance(item, (list, tuple)) and item:
            return str(item[-1])
        if item is not None and str(item).strip():
            return str(item).strip()
    raise KeyError(f"Missing video path. Expected '{video_key}' or 'video_path'.")


def build_multimodal_inputs(
    processor: Qwen2_5OmniProcessor,
    sample: Dict[str, Any],
    prompt_text: str,
    video_key: str,
    fps: int,
    max_pixels: int,
    use_audio_in_video: bool,
    device: torch.device,
) -> Tuple[Dict[str, torch.Tensor], str]:
    video_path = _resolve_video_path(sample, video_key)
    conversation = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_text},
                {"type": "video", "video": video_path, "fps": fps, "max_pixels": max_pixels},
            ],
        },
    ]
    audios, _, videos = process_mm_info(conversation, use_audio_in_video=use_audio_in_video)
    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    inputs = processor(
        text=[text],
        audio=audios,
        videos=videos,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=use_audio_in_video,
    ).to(device)
    return inputs, video_path


def score_fast_labels(
    model: torch.nn.Module,
    processor: Qwen2_5OmniProcessor,
    sample: Dict[str, Any],
    labels: List[str],
    include_summary: bool,
    video_key: str,
    fps: int,
    max_pixels: int,
    use_audio_in_video: bool,
    device: torch.device,
) -> Tuple[Dict[str, float], str, str]:
    prompt_text = build_fast_prompt(sample, labels, include_summary)
    inputs, video_path = build_multimodal_inputs(
        processor, sample, prompt_text, video_key, fps, max_pixels, use_audio_in_video, device
    )
    tokenizer = processor.tokenizer
    attention = inputs["attention_mask"][0]
    prompt_len = int(attention.sum().item())
    prompt_ids = inputs["input_ids"][:, :prompt_len]
    prompt_mask = inputs["attention_mask"][:, :prompt_len]
    static_inputs = {k: v for k, v in inputs.items() if k not in {"input_ids", "attention_mask"}}

    avg_logprobs: List[float] = []
    with torch.no_grad():
        for label in labels:
            ids = tokenizer(label, add_special_tokens=False)["input_ids"]
            label_ids = torch.tensor(ids, device=device, dtype=prompt_ids.dtype).unsqueeze(0)
            label_mask = torch.ones((1, len(ids)), device=device, dtype=prompt_mask.dtype)
            model_inputs = {
                "input_ids": torch.cat([prompt_ids, label_ids], dim=1),
                "attention_mask": torch.cat([prompt_mask, label_mask], dim=1),
                **static_inputs,
            }
            logits = model(**model_inputs, use_audio_in_video=use_audio_in_video).logits[0]
            logprob = 0.0
            for j, token_id in enumerate(ids):
                pos = prompt_len + j - 1
                logprob += float(F.log_softmax(logits[pos], dim=-1)[token_id].item())
            avg_logprobs.append(logprob / max(len(ids), 1))
    probs_t = torch.softmax(torch.tensor(avg_logprobs), dim=0)
    return {label: float(probs_t[i].item()) for i, label in enumerate(labels)}, video_path, prompt_text


def generate_text(
    model: torch.nn.Module,
    processor: Qwen2_5OmniProcessor,
    sample: Dict[str, Any],
    prompt_text: str,
    video_key: str,
    fps: int,
    max_pixels: int,
    use_audio_in_video: bool,
    device: torch.device,
    max_new_tokens: int,
    temperature: float,
) -> Tuple[str, str]:
    inputs, video_path = build_multimodal_inputs(
        processor, sample, prompt_text, video_key, fps, max_pixels, use_audio_in_video, device
    )
    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
            use_audio_in_video=use_audio_in_video,
        )
    trimmed = generated[:, inputs["input_ids"].shape[1] :]
    text = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return text.strip(), video_path


def entropy(probs: Dict[str, float]) -> float:
    return float(-sum(p * math.log(max(p, 1e-12)) for p in probs.values() if p > 0.0))
