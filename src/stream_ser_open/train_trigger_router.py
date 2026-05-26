import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .io import load_jsonl
from .labels import ALIASES


LABELS = ["neutral", "anger", "anxiety", "sadness", "joy", "surprise", "embarrassment"]
HIGH_CONTEXT = {"embarrassment", "anxiety", "neutral", "joy"}


def normalize_label(value: Any) -> str:
    label = str(value or "").strip().lower()
    return ALIASES.get(label, label)


def safe_mean(values: List[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def macro_f1(rows: List[Dict[str, Any]]) -> float:
    labels = sorted(set(row["gold"] for row in rows) | set(row["pred"] for row in rows))
    if not labels:
        return 0.0
    f1s = []
    for label in labels:
        tp = sum(1 for row in rows if row["gold"] == label and row["pred"] == label)
        fp = sum(1 for row in rows if row["gold"] != label and row["pred"] == label)
        fn = sum(1 for row in rows if row["gold"] == label and row["pred"] != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return safe_mean(f1s)


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_fast_trajectory(path: str) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for row in load_jsonl(path):
        sample_id = str(row.get("sample_id") or row.get("id") or row.get("stream_id") or "").strip()
        if sample_id:
            out[sample_id] = row
    return out


def load_slow_outputs(path: str) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for row in load_jsonl(path):
        sample_id = str(row.get("id") or row.get("sample_id") or row.get("stream_id") or "").strip()
        if not sample_id:
            continue
        pred = normalize_label(row.get("output_final_emotion") or row.get("final_emotion") or row.get("pred"))
        gold = normalize_label(row.get("target_final_emotion") or row.get("gt_label") or row.get("gold"))
        out[sample_id] = {"pred": pred, "gold": gold, "correct": int(pred == gold), "raw": row}
    return out


def linear_slope(xs: List[float], ys: List[float]) -> float:
    if len(xs) < 2 or len(xs) != len(ys):
        return 0.0
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    denom = float(np.sum((x - x.mean()) ** 2))
    if denom <= 1e-12:
        return 0.0
    return float(np.sum((x - x.mean()) * (y - y.mean())) / denom)


def build_feature_row(fast: Dict[str, Any], slow: Dict[str, Any]) -> Dict[str, Any]:
    trajectory = fast.get("trajectory", []) or []
    if not trajectory:
        trajectory = [fast]
    final = trajectory[-1]
    prefixes = [float(item.get("prefix_sec", i + 1) or i + 1) for i, item in enumerate(trajectory)]
    confs = [float(item.get("confidence", item.get("fast_confidence", 0.0)) or 0.0) for item in trajectory]
    ents = [float(item.get("entropy", item.get("fast_entropy", 0.0)) or 0.0) for item in trajectory]
    margins = [float(item.get("margin", item.get("fast_margin", 0.0)) or 0.0) for item in trajectory]
    preds = [normalize_label(item.get("pred", item.get("fast_pred", ""))) for item in trajectory]
    final_probs = final.get("probs", final.get("fast_probs", {})) or {}
    fast_pred = normalize_label(final.get("pred", final.get("fast_pred", "")))
    gold = normalize_label(fast.get("gold", slow.get("gold", "")))
    switch_count = sum(1 for i in range(1, len(preds)) if preds[i] != preds[i - 1])
    row: Dict[str, Any] = {
        "sample_id": fast.get("sample_id") or fast.get("id") or fast.get("stream_id"),
        "gold": gold,
        "fast_pred": fast_pred,
        "slow_pred": slow["pred"],
        "invoke_label": int(fast_pred != gold and slow["pred"] == gold),
        "final_confidence": confs[-1],
        "final_entropy": ents[-1],
        "final_margin": margins[-1],
        "switch_count": switch_count,
        "switch_rate": switch_count / max(len(preds) - 1, 1),
        "num_prefixes": len(trajectory),
        "first_prefix_confidence": confs[0],
        "max_confidence": max(confs),
        "min_confidence": min(confs),
        "avg_confidence": safe_mean(confs),
        "std_confidence": float(np.std(confs)),
        "min_entropy": min(ents),
        "max_entropy": max(ents),
        "avg_entropy": safe_mean(ents),
        "std_entropy": float(np.std(ents)),
        "max_margin": max(margins),
        "min_margin": min(margins),
        "avg_margin": safe_mean(margins),
        "std_margin": float(np.std(margins)),
        "conf_slope": linear_slope(prefixes, confs),
        "entropy_slope": linear_slope(prefixes, ents),
        "margin_slope": linear_slope(prefixes, margins),
        "conf_gain": confs[-1] - confs[0],
        "entropy_change": ents[-1] - ents[0],
        "margin_gain": margins[-1] - margins[0],
        "last_label_stable": int(len(preds) >= 2 and preds[-1] == preds[-2]),
        "stable_run_length": 1,
        "label_changed_last_step": int(len(preds) >= 2 and preds[-1] != preds[-2]),
        "high_context_fast_pred": int(fast_pred in HIGH_CONTEXT),
    }
    for previous in reversed(preds[:-1]):
        if previous == preds[-1]:
            row["stable_run_length"] += 1
        else:
            break
    for label in LABELS:
        row[f"fast_pred_is_{label}"] = int(fast_pred == label)
        row[f"final_prob_{label}"] = float(final_probs.get(label, 0.0) or 0.0)
    return row


def select_feature_names(rows: List[Dict[str, Any]]) -> List[str]:
    excluded = {"sample_id", "gold", "fast_pred", "slow_pred", "invoke_label"}
    return [key for key, value in rows[0].items() if key not in excluded and isinstance(value, (int, float, np.integer, np.floating))]


def build_matrix(rows: List[Dict[str, Any]], feature_names: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray([[float(row.get(name, 0.0) or 0.0) for name in feature_names] for row in rows], dtype=np.float64)
    y = np.asarray([int(row["invoke_label"]) for row in rows], dtype=np.int64)
    return x, y


def build_model(model_name: str, seed: int) -> Pipeline:
    if model_name == "logreg":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)),
            ]
        )
    if model_name == "rf":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("clf", RandomForestClassifier(n_estimators=400, min_samples_leaf=2, class_weight="balanced_subsample", n_jobs=-1, random_state=seed)),
            ]
        )
    raise ValueError(f"Unsupported model: {model_name}")


def evaluate_threshold(rows: List[Dict[str, Any]], scores: np.ndarray, threshold: float) -> Dict[str, Any]:
    routed = []
    for row, score in zip(rows, scores):
        use_slow = float(score) >= threshold
        pred = row["slow_pred"] if use_slow and row["slow_pred"] else row["fast_pred"]
        routed.append({"gold": row["gold"], "pred": pred, "correct": int(pred == row["gold"]), "used_slow": int(use_slow)})
    return {
        "threshold": threshold,
        "accuracy": safe_mean([row["correct"] for row in routed]),
        "macro_f1": macro_f1(routed),
        "slow_rate": safe_mean([row["used_slow"] for row in routed]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a lightweight trigger router from fast trajectories and slow outputs.")
    parser.add_argument("--train-trajectory-jsonl", required=True)
    parser.add_argument("--train-slow-jsonl", required=True)
    parser.add_argument("--eval-trajectory-jsonl", required=True)
    parser.add_argument("--eval-slow-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-name", choices=["logreg", "rf"], default="rf")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--thresholds", default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9")
    args = parser.parse_args()

    train_fast = load_fast_trajectory(args.train_trajectory_jsonl)
    train_slow = load_slow_outputs(args.train_slow_jsonl)
    eval_fast = load_fast_trajectory(args.eval_trajectory_jsonl)
    eval_slow = load_slow_outputs(args.eval_slow_jsonl)
    train_ids = sorted(set(train_fast) & set(train_slow))
    eval_ids = sorted(set(eval_fast) & set(eval_slow))
    if not train_ids or not eval_ids:
        raise ValueError("No overlapping sample ids between trajectory and slow files.")

    train_rows = [build_feature_row(train_fast[sid], train_slow[sid]) for sid in train_ids]
    eval_rows = [build_feature_row(eval_fast[sid], eval_slow[sid]) for sid in eval_ids]
    feature_names = select_feature_names(train_rows)
    x_train, y_train = build_matrix(train_rows, feature_names)
    x_eval, y_eval = build_matrix(eval_rows, feature_names)
    model = build_model(args.model_name, args.seed)
    model.fit(x_train, y_train)
    eval_scores = model.predict_proba(x_eval)[:, 1] if hasattr(model, "predict_proba") else model.predict(x_eval)
    thresholds = [float(item) for item in args.thresholds.split(",") if item.strip()]
    threshold_rows = [evaluate_threshold(eval_rows, eval_scores, threshold) for threshold in thresholds]
    best = max(threshold_rows, key=lambda row: (row["accuracy"], row["macro_f1"], -abs(row["threshold"] - 0.5)))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "feature_names": feature_names, "labels": LABELS}, out_dir / f"router_{args.model_name}.joblib")
    write_csv(out_dir / "threshold_search.csv", threshold_rows)
    summary = {
        "model_name": args.model_name,
        "feature_count": len(feature_names),
        "train_count": len(train_rows),
        "eval_count": len(eval_rows),
        "positive_rate_train": safe_mean([float(v) for v in y_train]),
        "positive_rate_eval": safe_mean([float(v) for v in y_eval]),
        "router_binary_acc_at_0p5": float(accuracy_score(y_eval, (eval_scores >= 0.5).astype(np.int64))),
        "best": best,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
