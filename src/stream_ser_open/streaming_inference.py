import argparse
import gc
import json
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Deque, Dict, List

import torch

from .io import dump_jsonl, first_text, get_stream_id, load_jsonl, load_label_map, parse_csv, sort_samples
from .labels import canonicalize_prediction, extract_label
from .models import entropy, generate_text, load_qwen_omni, resolve_device, score_fast_labels
from .prompts import build_slow_prompt, parse_slow_json
from .trigger import SlowTrigger


def str2bool(value: str) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="End-to-end streaming inference with fast axis, trigger, and slow axis.")
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--fast-model-path", required=True)
    parser.add_argument("--fast-adapter-path", default="")
    parser.add_argument("--slow-model-path", required=True)
    parser.add_argument("--slow-adapter-path", default="")
    parser.add_argument("--trigger-model-path", default="", help="Optional joblib bundle from train_trigger_router.py.")
    parser.add_argument("--trigger-threshold", type=float, default=0.5)
    parser.add_argument(
        "--trigger-decision-mode",
        choices=["auto", "threshold", "linear"],
        default="auto",
        help="For dual-head trigger bundles, use saved mode, force threshold routing, or force linear decision head.",
    )
    parser.add_argument("--rule-min-confidence", type=float, default=0.55)
    parser.add_argument("--rule-min-margin", type=float, default=0.15)
    parser.add_argument("--rule-max-entropy", type=float, default=1.20)
    parser.add_argument("--labels", default="neutral,anger,anxiety,sadness,joy,surprise,embarrassment")
    parser.add_argument("--default-label", default="neutral")
    parser.add_argument("--label-map-json", default="")
    parser.add_argument("--label-keys", default="gt_emotion,emotion,target")
    parser.add_argument("--video-key", default="video_path")
    parser.add_argument("--stream-key", default="stream_id")
    parser.add_argument("--step-key", default="step")
    parser.add_argument("--start-key", default="start_time")
    parser.add_argument("--summary-key", default="prev_summary")
    parser.add_argument("--history-size", type=int, default=5)
    parser.add_argument("--summary-history-size", type=int, default=4)
    parser.add_argument("--running-summary-max-chars", type=int, default=512)
    parser.add_argument("--include-summary", type=str2bool, default=True)
    parser.add_argument("--include-original", type=str2bool, default=False)
    parser.add_argument("--use-audio-in-video", type=str2bool, default=True)
    parser.add_argument("--fps", type=int, default=2)
    parser.add_argument("--max-pixels", type=int, default=224 * 126)
    parser.add_argument("--fast-device", default="cuda")
    parser.add_argument("--slow-device", default="cuda")
    parser.add_argument("--slow-max-new-tokens", type=int, default=96)
    parser.add_argument("--slow-temperature", type=float, default=0.1)
    parser.add_argument("--max-samples", type=int, default=0)
    return parser.parse_args()


def _running_summary(summaries: List[str], max_chars: int) -> str:
    text = " | ".join(x for x in summaries if x)
    if max_chars > 0 and len(text) > max_chars:
        return text[-max_chars:]
    return text


def _extract_local_summary(sample: Dict[str, Any], summary_key: str) -> str:
    original = sample.get("original") if isinstance(sample.get("original"), dict) else {}
    return first_text(
        [
            sample.get(summary_key),
            sample.get("event_summary_for_memory"),
            sample.get("summary_gt"),
            sample.get("prev_summary"),
            original.get(summary_key),
            original.get("event_summary_for_memory"),
            original.get("summary_gt"),
            original.get("prev_summary"),
        ]
    )


def _normalize_slow_result(parsed: Dict[str, Any], fast_pred: str, labels: List[str], default_label: str) -> Dict[str, Any]:
    corrected = canonicalize_prediction(parsed.get("corrected_emotion", fast_pred), labels, fast_pred or default_label)
    need = int(str(parsed.get("need_correction", "0")).strip() in {"1", "true", "True", "yes"})
    if corrected == fast_pred:
        need = 0
    final = corrected if need else fast_pred
    delta = str(parsed.get("correction_delta", "")).strip()
    if not delta:
        delta = f"{fast_pred}->{corrected}" if need else "none"
    if not need:
        delta = "none"
    return {
        "need_correction": need,
        "corrected_emotion": corrected,
        "correction_delta": delta,
        "final_emotion": final,
    }


def main() -> None:
    args = parse_args()
    labels = [x.lower() for x in parse_csv(args.labels)]
    default_label = args.default_label.lower()
    if default_label not in labels:
        raise ValueError("--default-label must be contained in --labels")
    label_map = load_label_map(args.label_map_json)
    label_keys = parse_csv(args.label_keys)

    fast_device = resolve_device(args.fast_device)
    slow_device = resolve_device(args.slow_device)
    fast_processor, fast_model = load_qwen_omni(args.fast_model_path, args.fast_adapter_path, fast_device)
    slow_processor, slow_model = load_qwen_omni(args.slow_model_path, args.slow_adapter_path, slow_device)
    trigger = SlowTrigger(
        labels=labels,
        model_path=args.trigger_model_path or None,
        threshold=args.trigger_threshold,
        decision_mode=args.trigger_decision_mode,
        min_confidence=args.rule_min_confidence,
        min_margin=args.rule_min_margin,
        max_entropy=args.rule_max_entropy,
    )

    samples = load_jsonl(args.input_jsonl)
    indexed = sort_samples(samples, args.stream_key, args.step_key, args.start_key)
    if args.max_samples > 0:
        indexed = indexed[: args.max_samples]

    histories: Dict[str, Dict[str, Deque[Any]]] = defaultdict(
        lambda: {
            "preds": deque(maxlen=args.history_size),
            "confs": deque(maxlen=args.history_size),
            "entropies": deque(maxlen=args.history_size),
            "margins": deque(maxlen=args.history_size),
            "summaries": deque(maxlen=args.summary_history_size),
            "corrected_labels": deque(maxlen=args.history_size),
        }
    )
    outputs: List[Dict[str, Any]] = []
    fast_correct = 0
    final_correct = 0
    gt_count = 0

    for n, (orig_idx, sample) in enumerate(indexed, 1):
        stream_id = get_stream_id(sample, args.stream_key, orig_idx)
        hist = histories[stream_id]
        running_summary = _running_summary(list(hist["summaries"]), args.running_summary_max_chars)
        latest_corrected = str(hist["corrected_labels"][-1]) if hist["corrected_labels"] else ""
        local_summary = _extract_local_summary(sample, args.summary_key)

        prompt_sample = dict(sample)
        prompt_sample.update(
            {
                "traj_last_k": list(hist["preds"]),
                "conf_last_k": [float(x) for x in hist["confs"]],
                "entropy_last_k": [float(x) for x in hist["entropies"]],
                "margin_last_k": [float(x) for x in hist["margins"]],
                "summary_last_k": list(hist["summaries"]),
                "latest_running_summary": running_summary,
                "history_running_summary_before": running_summary,
                "latest_corrected_label": latest_corrected,
            }
        )

        probs, video_path, fast_prompt = score_fast_labels(
            fast_model,
            fast_processor,
            prompt_sample,
            labels,
            args.include_summary,
            args.video_key,
            args.fps,
            args.max_pixels,
            args.use_audio_in_video,
            fast_device,
        )
        ranked = sorted(probs.items(), key=lambda item: item[1], reverse=True)
        fast_pred, fast_conf = ranked[0]
        second_conf = ranked[1][1] if len(ranked) > 1 else 0.0
        fast_entropy = entropy(probs)
        fast_margin = float(fast_conf - second_conf)

        state: Dict[str, Any] = {
            "sample_index": orig_idx,
            "stream_id": stream_id,
            "video_path": video_path,
            "fast_prompt_text": fast_prompt,
            "fast_pred": fast_pred,
            "fast_probs": probs,
            "fast_confidence": float(fast_conf),
            "fast_entropy": fast_entropy,
            "fast_margin": fast_margin,
            "traj_last_k": list(hist["preds"]),
            "conf_last_k": [float(x) for x in hist["confs"]],
            "entropy_last_k": [float(x) for x in hist["entropies"]],
            "margin_last_k": [float(x) for x in hist["margins"]],
            "summary_last_k": list(hist["summaries"]),
            "latest_running_summary": running_summary,
            "history_running_summary_before": running_summary,
            "local_summary": local_summary,
            "latest_corrected_label": latest_corrected,
        }
        if args.include_original:
            state["original"] = sample

        gt_label = extract_label(sample, label_keys, labels, label_map, default_label)
        if gt_label:
            state["gt_label"] = gt_label
            gt_count += 1
            fast_correct += int(fast_pred == gt_label)

        decision = trigger.decide(state)
        state["trigger_score"] = decision.score
        state["trigger_reason"] = decision.reason
        state["used_slow"] = int(decision.use_slow)

        if decision.use_slow:
            slow_prompt = build_slow_prompt(state, labels)
            slow_text, slow_video_path = generate_text(
                slow_model,
                slow_processor,
                state,
                slow_prompt,
                args.video_key,
                args.fps,
                args.max_pixels,
                args.use_audio_in_video,
                slow_device,
                args.slow_max_new_tokens,
                args.slow_temperature,
            )
            parsed = parse_slow_json(slow_text)
            state["slow_prompt_text"] = slow_prompt
            state["slow_output_raw"] = slow_text
            state["slow_output_valid_json"] = int(bool(parsed))
            state["slow_video_path"] = slow_video_path
            state.update(parsed)
            state.update(_normalize_slow_result(parsed, fast_pred, labels, default_label))
        else:
            state.update(
                {
                    "need_correction": 0,
                    "corrected_emotion": fast_pred,
                    "correction_delta": "none",
                    "final_emotion": fast_pred,
                }
            )

        if gt_label:
            final_correct += int(state["final_emotion"] == gt_label)
            state["fast_match_gt"] = int(fast_pred == gt_label)
            state["final_match_gt"] = int(state["final_emotion"] == gt_label)

        hist["preds"].append(fast_pred)
        hist["confs"].append(float(fast_conf))
        hist["entropies"].append(float(fast_entropy))
        hist["margins"].append(float(fast_margin))
        new_summary = first_text([state.get("new_summary"), local_summary])
        if new_summary:
            hist["summaries"].append(new_summary)
        hist["corrected_labels"].append(state["final_emotion"])
        outputs.append(state)

        if n % 20 == 0 or n == len(indexed):
            print(f"Processed {n}/{len(indexed)}")

    dump_jsonl(args.output_jsonl, outputs)
    meta = {
        "num_samples": len(outputs),
        "labels": labels,
        "trigger_model_path": args.trigger_model_path,
        "trigger_threshold": args.trigger_threshold,
        "trigger_decision_mode": args.trigger_decision_mode,
        "slow_rate": sum(row["used_slow"] for row in outputs) / max(len(outputs), 1),
    }
    if gt_count:
        meta["gt_count"] = gt_count
        meta["fast_accuracy"] = fast_correct / gt_count
        meta["final_accuracy"] = final_correct / gt_count
    with Path(args.output_jsonl + ".meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    del fast_model, slow_model, fast_processor, slow_processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"Saved {len(outputs)} streaming records to {args.output_jsonl}")


if __name__ == "__main__":
    main()
