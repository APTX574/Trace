#!/usr/bin/env python
"""Offline reproduction -- step 1.5: build per-utterance trajectory samples.

Converts the per-prefix records from ``run_fast_prefix_belief.py`` into
``trajectory_samples.jsonl``: for each utterance, the belief trajectory
(one entry per audio prefix) plus final-prefix statistics (prediction,
confidence, entropy, margin, correctness).
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

ALIASES = {"happy": "joy", "angry": "anger", "disgust": "anger", "fear": "anxiety", "happiness": "joy"}


def norm(value: Any) -> str:
    return ALIASES.get(str(value or "").strip().lower(), str(value or "").strip().lower())


def main() -> None:
    parser = argparse.ArgumentParser(description="Build trajectory_samples.jsonl from per-prefix fast records.")
    parser.add_argument("--prefix-predictions-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    args = parser.parse_args()

    rows = [json.loads(l) for l in Path(args.prefix_predictions_jsonl).open(encoding="utf-8")]
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        grouped[str(r["sample_id"])].append(r)
    for k in grouped:
        grouped[k].sort(key=lambda r: (float(r["prefix_sec"]), int(r["is_full_prefix"])))

    out: List[Dict[str, Any]] = []
    for sid, items in grouped.items():
        traj = [{
            "prefix_sec": float(t["prefix_sec"]), "prefix_tag": t["prefix_tag"],
            "is_full_prefix": int(t["is_full_prefix"]), "pred": norm(t["pred_label"]),
            "gold": norm(t["gold_label"]), "correct": bool(t["is_correct"]),
            "confidence": float(t["confidence"]), "entropy": float(t["entropy"]),
            "margin": float(t["margin"]), "gold_prob": float(t["gold_prob"]),
            "probs": t["probabilities"],
            "label_log_likelihoods": t.get("label_log_likelihoods", {}) or {},
        } for t in items]
        final = traj[-1]
        preds = [t["pred"] for t in traj]
        switch_count = sum(1 for i in range(1, len(preds)) if preds[i] != preds[i - 1])
        out.append({
            "sample_id": sid,
            "clip_id": str(items[0].get("clip_id", "") or ""),
            "stream_id": str(items[0].get("stream_id", "") or ""),
            "speaker": str(items[0].get("speaker", "") or ""),
            "dialogue": str(items[0].get("dialogue", "") or ""),
            "video_path": str(items[0].get("video_path", "") or ""),
            "clip_duration_sec": float(items[0].get("clip_duration_sec", 0.0) or 0.0),
            "gold": norm(items[0].get("gold_label", "") or ""),
            "trajectory": traj,
            "switch_count": switch_count,
            "first_correct_time": next((t["prefix_sec"] for t in traj if t["correct"]), None),
            "final_correct": bool(final["correct"]),
            "final_pred": final["pred"],
            "final_confidence": final["confidence"],
            "final_entropy": final["entropy"],
            "final_margin": final["margin"],
        })

    out_path = Path(args.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    acc = sum(1 for r in out if r["final_correct"]) / len(out) if out else 0.0
    print(json.dumps({"samples": len(out), "full_accuracy": round(acc, 4), "output": str(out_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()