#!/usr/bin/env python
"""Offline reproduction -- step 2: contextual belief reinterpretation (slow axis).

Reads a JSONL of test/dev rows (each with video_path, speaker, dialogue,
prev_summary, target/gold) plus the fast trajectory JSONL produced by
``run_fast_prefix_belief.py``, and writes slow-axis eval records with the
paper's reinterpretation output schema ``final_emotion / reason / new_summary``.

The slow model receives the previous-context audio, the current target audio
and the current target video (use_audio_in_video=True, fps=2, max_pixels=28224),
mirroring the released slow LoRA checkpoint's training/eval configuration.
The previous / next context clip is resolved from the input order inside each
clip (or by the digit suffix of the clip filename, whichever is available).
"""

import argparse
import json
import os
import sys
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch  # noqa: E402

from stream_ser_open.io import first_text, load_jsonl, parse_csv  # noqa: E402
from stream_ser_open.labels import ALIASES  # noqa: E402
from stream_ser_open.models import load_qwen_omni, resolve_device  # noqa: E402

try:
    from qwen_omni_utils import process_mm_info  # noqa: E402
except ImportError as exc:
    raise ImportError("qwen_omni_utils is required (from the Qwen2.5-Omni release)") from exc

LABELS_DEFAULT = "neutral,anger,anxiety,sadness,joy,surprise,embarrassment"
SYSTEM_PROMPT = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
    "capable of perceiving auditory and visual inputs, as well as generating text and speech."
)


def normalize_label(value: Any, labels: List[str], default: str) -> str:
    text = str(value or "").strip().lower()
    if text in labels:
        return text
    if text in ALIASES and ALIASES[text] in labels:
        return ALIASES[text]
    for lab in labels:
        if lab in text:
            return lab
    return default


def build_slow_prompt(sample: Dict[str, Any], labels: List[str]) -> str:
    labels_str = ", ".join(labels)
    speaker = str(sample.get("speaker", "") or "")
    dialogue = first_text([sample.get("dialogue")])
    history = first_text([sample.get("prev_summary")])
    prev = first_text([sample.get("previous_video_path")])
    nxt = first_text([sample.get("next_video_path")])
    lines = [
        "Task: predict the target speaker's emotion in the current clip.",
        "You may receive up to four multimodal inputs in this order:",
        "1. optional previous-context audio",
        "2. the current target audio clip",
        "3. the current target video clip",
        "4. optional next-context audio",
        "The current target audio and current target video describe the same clip and are the primary evidence.",
        "Prioritize the current target audio when judging subtle emotions, especially prosody, speaking rate, pauses, intensity, breathiness, pitch movement, trembling, laughter, crying, sighs, and voice quality changes.",
        "Use the current target video to verify or refine the emotion from facial expression, gaze, posture, and visible interaction context.",
        "The previous and next audios are auxiliary context only.",
        "Use previous or next audio only to resolve ambiguity in the current clip, never as the main basis for prediction.",
        "Never predict the emotion of the previous or next clip.",
        f"Valid emotion labels: {labels_str}.",
        "Return the final emotion label for the current clip.",
        "Do not output whether correction is needed.",
        "Input 1 is audio from the temporally nearest previous clip in the same dialogue timeline. It may belong to a different speaker, so use it only as short-range temporal context.",
    ]
    if not prev:
        lines.append("There is no valid previous-context audio for this sample. Do not infer missing context that is not provided.")
    if not nxt:
        lines.append("There is no valid next-context audio for this sample. Do not assume future evidence beyond the target clip.")
    lines += [
        f"Target speaker in the current clip: {speaker}.",
        f'Current dialogue: "{dialogue}".',
        f'History summary before the current clip: "{history}".',
        "Use it as background state, not as a replacement for current evidence.",
        "When audio and video disagree, trust the current target audio more for affective state unless the audio is clearly off-screen, corrupted, silent, or belongs to another speaker.",
        "Focus on the target speaker's own voice rather than background music, sound effects, or other speakers.",
        "Output strict JSON with keys:",
        "- final_emotion: one label from valid set",
        "- reason",
        "- new_summary",
    ]
    return "\n".join(lines)


def build_inputs(processor, sample: Dict[str, Any], prompt_text: str, fps: int, max_pixels: int, device: torch.device, use_audio_in_video: bool):
    content: List[Dict[str, Any]] = []
    prev = first_text([sample.get("previous_video_path")])
    cur = first_text([sample.get("current_video_path")])
    nxt = first_text([sample.get("next_video_path")])
    if prev and os.path.exists(prev):
        content.append({"type": "audio", "audio": prev})
    if cur and os.path.exists(cur):
        content.append({"type": "audio", "audio": cur})
        content.append({"type": "video", "video": cur, "fps": fps, "max_pixels": max_pixels})
    if nxt and os.path.exists(nxt):
        content.append({"type": "audio", "audio": nxt})
    content.append({"type": "text", "text": prompt_text})
    conversation = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": content},
    ]
    audios, _, videos = process_mm_info(conversation, use_audio_in_video=use_audio_in_video)
    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    return processor(
        text=[text], audio=audios, videos=videos,
        return_tensors="pt", padding=True, use_audio_in_video=use_audio_in_video,
    ).to(device)


def resolve_prev_next(rows: List[Dict[str, Any]], video_root: str) -> Dict[str, Dict[str, str]]:
    """Map sample_id -> {previous_video_path, next_video_path} from clip ordering."""
    video_root = video_root or ""
    by_clip: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_clip[str(r.get("clip_id") or r.get("group_id") or "")].append(r)
    for v in by_clip.values():
        v.sort(key=lambda r: float(r.get("step", r.get("start_time", 0.0))))

    def _path(r: Dict[str, Any]) -> str:
        p = first_text([r.get("resolved_video_path"), r.get("video_path")])
        if video_root and p and not os.path.isabs(p):
            p = os.path.join(video_root, p)
        return p

    out: Dict[str, Dict[str, str]] = {}
    for clip_rows in by_clip.values():
        for i, r in enumerate(clip_rows):
            sid = str(r.get("sample_id") or r.get("id") or r.get("stream_id"))
            out[sid] = {
                "previous_video_path": _path(clip_rows[i - 1]) if i > 0 else "",
                "next_video_path": _path(clip_rows[i + 1]) if i + 1 < len(clip_rows) else "",
            }
    return out


def parse_json_response(text: str) -> Dict[str, Any]:
    clean = str(text).strip()
    if clean.startswith("```json"):
        clean = clean[7:]
    elif clean.startswith("```"):
        clean = clean[3:]
    if clean.endswith("```"):
        clean = clean[:-3]
    clean = clean.strip()
    start, end = clean.find("{"), clean.rfind("}")
    if start < 0 or end <= start:
        return {}
    try:
        return json.loads(clean[start : end + 1])
    except Exception:
        return {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline reproduction -- step 2: slow-axis reinterpretation batch.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--adapter-path", required=True)
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--trajectory-jsonl", default="", help="Optional fast trajectory JSONL (merges fast forecast into input).")
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--video-root", default="")
    parser.add_argument("--labels", default=LABELS_DEFAULT)
    parser.add_argument("--default-label", default="neutral")
    parser.add_argument("--label-key", default="target")
    parser.add_argument("--fps", type=int, default=2)
    parser.add_argument("--max-pixels", type=int, default=28224)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--use-audio-in-video", type=lambda x: (str(x).lower() in {"1", "true", "yes"}), default=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    labels = [x.lower() for x in parse_csv(args.labels)]
    device = resolve_device(args.device)
    processor, model = load_qwen_omni(args.model_path, args.adapter_path, device)
    tokenizer = processor.tokenizer

    rows = load_jsonl(args.input_jsonl)
    if args.trajectory_jsonl:
        traj = {json.loads(l)["sample_id"]: json.loads(l) for l in open(args.trajectory_jsonl, encoding="utf-8")}
        for r in rows:
            sid = str(r.get("sample_id") or r.get("id"))
            t = traj.get(sid)
            if t:
                r["fast_pred"] = t.get("final_pred", "")
                r["fast_probs"] = t.get("trajectory", [{}])[-1].get("probs", {})

    prev_next = resolve_prev_next(rows, args.video_root)
    outputs: List[Dict[str, Any]] = []
    for i, sample in enumerate(rows):
        sid = str(sample.get("sample_id") or sample.get("id") or f"row_{i}")
        gold = normalize_label(sample.get(args.label_key) or sample.get("gt_emotion") or sample.get("emotion"), labels, args.default_label)
        cur = first_text([sample.get("resolved_video_path"), sample.get("video_path")])
        if args.video_root and cur and not os.path.isabs(cur):
            cur = os.path.join(args.video_root, cur)
        sample["current_video_path"] = cur
        pn = prev_next.get(sid, {})
        sample["previous_video_path"] = first_text([sample.get("previous_video_path"), pn.get("previous_video_path", "")])
        sample["next_video_path"] = first_text([sample.get("next_video_path"), pn.get("next_video_path", "")])

        prompt_text = build_slow_prompt(sample, labels)
        inputs = build_inputs(processor, sample, prompt_text, args.fps, args.max_pixels, device, args.use_audio_in_video)
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_audio_in_video=args.use_audio_in_video,
            )
        trimmed = generated[:, inputs["input_ids"].shape[1]:]
        raw = tokenizer.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
        parsed = parse_json_response(raw)
        pred = normalize_label(parsed.get("final_emotion") or parsed.get("corrected_emotion") or "", labels, args.default_label)
        if pred not in labels:
            pred = args.default_label
        outputs.append({
            "id": sid,
            "sample_id": sid,
            "output_final_emotion": pred,
            "target_final_emotion": gold,
            "reason": str(parsed.get("reason", "")),
            "new_summary": str(parsed.get("new_summary", "")),
            "slow_output_raw": raw,
            "slow_output_valid_json": int(bool(parsed)),
        })
        if (i + 1) % 10 == 0 or i + 1 == len(rows):
            print(f"[{i + 1}/{len(rows)}] {sid} -> {pred}", flush=True)

    out_path = Path(args.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in outputs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    n = len(outputs)
    acc = sum(1 for r in outputs if r["output_final_emotion"] == r["target_final_emotion"]) / n if n else 0.0
    print(json.dumps({"samples": n, "accuracy": round(acc, 4), "output": str(out_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()