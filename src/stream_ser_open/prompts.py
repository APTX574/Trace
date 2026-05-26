import json
from typing import Any, Dict, List

from .io import first_text, get_nested_original


SYSTEM_PROMPT = (
    "You are a multimodal streaming emotion recognition assistant. "
    "Use the current audio-visual evidence as the primary signal."
)


def build_fast_prompt(sample: Dict[str, Any], labels: List[str], include_summary: bool) -> str:
    parts = [
        "Infer the character's current observable emotion from this streaming clip.",
        "Focus on visible expression, tone, timing, and the current dialogue.",
        f"Reply with exactly one label from: {', '.join(labels)}.",
    ]
    summary = first_text(
        [
            sample.get("latest_running_summary"),
            sample.get("history_running_summary_before"),
            sample.get("prev_summary"),
            get_nested_original(sample).get("prev_summary"),
        ]
    )
    dialogue = first_text(
        [
            sample.get("dialogue"),
            sample.get("text"),
            get_nested_original(sample).get("dialogue"),
            get_nested_original(sample).get("text"),
        ]
    )
    if include_summary and summary:
        parts.append(f'Previous summary: "{summary}".')
    if dialogue:
        parts.append(f'Current dialogue: "{dialogue}".')
    parts.append("Current emotion:")
    return "\n".join(parts)


def format_probs(probs: Dict[str, float]) -> str:
    return ", ".join(f"{k}:{float(v):.3f}" for k, v in sorted(probs.items(), key=lambda kv: kv[1], reverse=True))


def build_slow_prompt(state: Dict[str, Any], labels: List[str]) -> str:
    original = get_nested_original(state)
    dialogue = first_text([state.get("dialogue"), original.get("dialogue"), original.get("text")])
    summary = first_text([state.get("latest_running_summary"), state.get("prev_summary"), original.get("prev_summary")])
    parts = [
        "You are the slow correction axis for streaming emotion recognition.",
        "Re-check the current clip using full multimodal evidence.",
        "The fast-axis prediction is a reference, not a final answer.",
        f"Valid labels: {', '.join(labels)}.",
        f"Fast prediction: {state.get('fast_pred', '')}.",
        f"Fast probabilities: {format_probs(state.get('fast_probs', {}))}.",
        f"Fast entropy: {float(state.get('fast_entropy', 0.0)):.4f}.",
        f"Fast margin: {float(state.get('fast_margin', 0.0)):.4f}.",
    ]
    if state.get("traj_last_k"):
        parts.append(f"Recent fast labels: {', '.join(map(str, state['traj_last_k']))}.")
    if dialogue:
        parts.append(f'Current dialogue: "{dialogue}".')
    if summary:
        parts.append(f'Previous summary: "{summary}".')
    parts.append(
        "Return strict JSON with keys: "
        "need_correction, corrected_emotion, correction_delta, new_summary. "
        "Use correction_delta='none' when no correction is needed."
    )
    return "\n".join(parts)


def parse_slow_json(text: str) -> Dict[str, Any]:
    clean = str(text).strip()
    if clean.startswith("```json"):
        clean = clean[7:]
    elif clean.startswith("```"):
        clean = clean[3:]
    if clean.endswith("```"):
        clean = clean[:-3]
    start = clean.find("{")
    end = clean.rfind("}")
    if start < 0 or end <= start:
        return {}
    try:
        return json.loads(clean[start : end + 1])
    except json.JSONDecodeError:
        return {}
