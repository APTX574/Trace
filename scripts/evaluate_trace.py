#!/usr/bin/env python
"""Offline evaluation of streaming strategies (paper Table 2 style).

Given a trajectory_samples.jsonl (per-utterance prefix belief trajectories from
build_fast_trajectory.py) and a slow eval JSONL (from run_slow_batch.py),
computes Accuracy / Macro-F1 / Weighted-F1 / avg decision time /
normalized decision ratio / slow call rate for the strategies:
    fast_only_first_1p5, always_full_fast, early_commit_wait_full,
    invoke_slow_when_needed, oracle_slow_trigger, slow_only.
Outputs the same style of CSV and summary JSON as the paper artifacts.
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List

LABELS = ["neutral", "anger", "anxiety", "sadness", "joy", "surprise", "embarrassment"]
ALIASES = {
    "angry": "anger", "disgust": "anger", "contempt": "anger",
    "fear": "anxiety", "happiness": "joy", "happy": "joy",
    "awkwardness": "embarrassment", "guilt": "embarrassment",
    "resignation": "sadness",
}


def normalize_label(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in LABELS:
        return text
    if text in ALIASES and ALIASES[text] in LABELS:
        return ALIASES[text]
    for lab in LABELS:
        if lab in text:
            return lab
    return text


def safe_mean(values: List[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def macro_f1(rows: List[Dict[str, Any]]) -> float:
    labels = sorted(set(r["gold"] for r in rows) | set(r["pred"] for r in rows))
    if not labels:
        return 0.0
    scores = []
    for lab in labels:
        tp = sum(1 for r in rows if r["gold"] == lab and r["pred"] == lab)
        fp = sum(1 for r in rows if r["gold"] != lab and r["pred"] == lab)
        fn = sum(1 for r in rows if r["gold"] == lab and r["pred"] != lab)
        p = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * p * rec / (p + rec) if p + rec else 0.0)
    return safe_mean(scores)


def weighted_f1(rows: List[Dict[str, Any]]) -> float:
    labels = sorted(set(r["gold"] for r in rows))
    if not labels:
        return 0.0
    total = len(rows)
    weighted = 0.0
    for lab in labels:
        w = sum(1 for r in rows if r["gold"] == lab) / total
        tp = sum(1 for r in rows if r["gold"] == lab and r["pred"] == lab)
        fp = sum(1 for r in rows if r["gold"] != lab and r["pred"] == lab)
        fn = sum(1 for r in rows if r["gold"] == lab and r["pred"] != lab)
        p = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f = 2 * p * rec / (p + rec) if p + rec else 0.0
        weighted += f * w
    return weighted


def evaluate(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "count": len(rows),
        "accuracy": sum(r["correct"] for r in rows) / len(rows) if rows else 0.0,
        "macro_f1": macro_f1(rows),
        "weighted_f1": weighted_f1(rows),
        "avg_decision_time": safe_mean([float(r.get("decision_time", 0.0)) for r in rows]),
        "avg_video_length": safe_mean([float(r.get("video_length", 0.0)) for r in rows]),
        "normalized_decision_ratio": safe_mean([float(r.get("normalized_decision_ratio", 0.0)) for r in rows]),
        "slow_call_rate": safe_mean([float(r.get("used_slow", 0.0)) for r in rows]),
    }


def should_commit(step: Dict[str, Any], conf: float, margin: float, entropy: float) -> bool:
    return float(step.get("confidence", 0.0)) >= conf and float(step.get("margin", 0.0)) >= margin and float(step.get("entropy", 0.0)) <= entropy


def should_invoke_slow(final_step: Dict[str, Any], switch_count: int, conf: float, margin: float, entropy: float, near_full_ratio: float) -> bool:
    if switch_count > 0:
        return True
    if float(final_step.get("confidence", 0.0)) >= conf and float(final_step.get("margin", 0.0)) >= margin and float(final_step.get("entropy", 0.0)) <= entropy:
        return False
    return True


def build_rows(trajectories: List[Dict[str, Any]], slow_map: Dict[str, Any], args: argparse.Namespace) -> Dict[str, List[Dict[str, Any]]]:
    strategies = [
        "fast_only_first_1p5", "always_full_fast", "early_commit_wait_full",
        "invoke_slow_when_needed", "oracle_slow_trigger", "slow_only",
    ]
    out: Dict[str, List[Dict[str, Any]]] = {name: [] for name in strategies}
    for traj in trajectories:
        steps = traj.get("trajectory", [])
        if not steps:
            continue
        gold = normalize_label(traj.get("gold", ""))
        slow = slow_map.get(str(traj.get("sample_id") or ""))
        video_length = max(float(traj.get("clip_duration_sec", 0.0) or 0.0), 1e-6)

        def add(name: str, pred: str, decision_time: float, used_slow: int):
            pred = normalize_label(pred)
            pred_final = pred if pred in LABELS else LABELS[0]
            out[name].append(
                {
                    "sample_id": traj.get("sample_id"),
                    "gold": gold,
                    "pred": pred_final,
                    "correct": int(pred_final == gold),
                    "decision_time": float(decision_time),
                    "video_length": video_length,
                    "normalized_decision_ratio": float(decision_time) / video_length,
                    "used_slow": used_slow,
                }
            )

        fast_full = steps[-1]
        # 1) early commit at first 1.5s prefix
        first = steps[0]
        add("fast_only_first_1p5", first["pred"], first["prefix_sec"], 0)
        # 2) wait until full, no reinterpretation
        add("always_full_fast", fast_full["pred"], fast_full["prefix_sec"], 0)
        # 3) adaptive commit (wait until reliability, else full)
        chosen = fast_full
        for step in steps:
            if should_commit(step, args.commit_conf, args.commit_margin, args.commit_entropy):
                chosen = step
                break
        add("early_commit_wait_full", chosen["pred"], chosen["prefix_sec"], 0)
        # 4) adaptive commit + invoke slow when unreliable
        if not (chosen is fast_full and chosen == steps[-1]):
            chosen_fast = chosen
            add("invoke_slow_when_needed", chosen_fast["pred"], chosen_fast["prefix_sec"], 0)
        else:
            switch_count = sum(1 for i in range(1, len(steps)) if steps[i]["pred"] != steps[i - 1]["pred"])
            if slow and should_invoke_slow(fast_full, switch_count, args.slow_conf, args.slow_margin, args.slow_entropy, args.near_full_ratio):
                add("invoke_slow_when_needed", slow["pred_label"], fast_full["prefix_sec"], 1)
            else:
                add("invoke_slow_when_needed", fast_full["pred"], fast_full["prefix_sec"], 0)
        # 5) oracle (use slow when it corrects the fast error)
        if slow and fast_full["pred"] != gold and slow["pred_label"] == gold:
            add("oracle_slow_trigger", slow["pred_label"], fast_full["prefix_sec"], 1)
        else:
            add("oracle_slow_trigger", fast_full["pred"], fast_full["prefix_sec"], 0)
        # 6) slow always
        if slow:
            add("slow_only", slow["pred_label"], fast_full["prefix_sec"], 1)
    return out


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for k in row.keys():
            if k not in seen:
                seen.add(k)
                fieldnames.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory-jsonl", required=True)
    parser.add_argument("--slow-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--target-space", choices=["meld_raw", "mapped7"], default="mapped7")
    parser.add_argument("--commit-conf", type=float, default=0.75)
    parser.add_argument("--commit-margin", type=float, default=0.50)
    parser.add_argument("--commit-entropy", type=float, default=0.75)
    parser.add_argument("--slow-conf", type=float, default=0.70)
    parser.add_argument("--slow-margin", type=float, default=0.40)
    parser.add_argument("--slow-entropy", type=float, default=0.90)
    parser.add_argument("--near-full-ratio", type=float, default=0.90)
    args = parser.parse_args()

    trajectories: List[Dict[str, Any]] = [json.loads(l) for l in Path(args.trajectory_jsonl).open(encoding="utf-8")]
    slow_map: Dict[str, Dict[str, Any]] = {}
    for line in Path(args.slow_jsonl).open(encoding="utf-8"):
        row = json.loads(line)
        sid = str(row.get("id") or row.get("sample_id") or "").strip()
        if not sid:
            continue
        pred = normalize_label(row.get("output_final_emotion") or row.get("corrected_emotion") or row.get("pred"))
        gold = normalize_label(row.get("target_final_emotion") or row.get("gt_label") or row.get("target"))
        slow_map[sid] = {"sample_id": sid, "pred_label": pred, "gold_label": gold, "correct": int(pred == gold)}

    strategy_rows = build_rows(trajectories, slow_map, args)
    metric_rows = []
    for name in strategy_rows:
        rows = strategy_rows[name]
        metrics = evaluate(rows)
        metric_rows.append({"strategy": name, **metrics})

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / f"strategy_delay_metrics_{args.target_space}.csv", metric_rows)

    with (out_dir / f"summary_{args.target_space}.json").open("w", encoding="utf-8") as f:
        json.dump({"target_space": args.target_space, "metrics": metric_rows, "trajectory": len(trajectories), "slow_map": len(slow_map)}, f, ensure_ascii=False, indent=2)

    print(json.dumps({"target_space": args.target_space, "metrics": metric_rows, "outputs": {"csv": str(out_dir / f"strategy_delay_metrics_{args.target_space}.csv")}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()