import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from .train_trigger_router import (
    build_feature_row,
    build_matrix,
    load_fast_trajectory,
    load_slow_outputs,
    macro_f1,
    safe_mean,
    select_feature_names,
    write_csv,
)


class RiskGainModel(nn.Module):
    """Dual-head model from scripts/47_ctrt_trigger.py: predict risk and gain."""

    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.3) -> None:
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.risk_head = nn.Linear(hidden_dim // 2, 1)  # P(fast wrong)
        self.gain_head = nn.Linear(hidden_dim // 2, 1)  # E[revision gain]

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.shared(x)
        risk = torch.sigmoid(self.risk_head(h)).squeeze(-1)
        gain = torch.tanh(self.gain_head(h)).squeeze(-1)  # gain can be positive or negative
        return risk, gain


class LinearDecisionHead(nn.Module):
    """Linear decision layer over the trigger outputs [risk, gain]."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 1)

    def forward(self, risk: torch.Tensor, gain: torch.Tensor) -> torch.Tensor:
        x = torch.stack([risk, gain], dim=-1)
        return torch.sigmoid(self.linear(x)).squeeze(-1)


def build_rows(fast_jsonl: str, slow_jsonl: str) -> List[Dict[str, Any]]:
    fast = load_fast_trajectory(fast_jsonl)
    slow = load_slow_outputs(slow_jsonl)
    ids = sorted(set(fast) & set(slow))
    if not ids:
        raise ValueError("No overlapping sample ids between trajectory and slow output files.")
    rows = [build_feature_row(fast[sid], slow[sid]) for sid in ids]
    for row in rows:
        fast_correct = int(row["fast_pred"] == row["gold"])
        slow_correct = int(row["slow_pred"] == row["gold"])
        row["risk_label"] = int(not fast_correct)
        row["gain_label"] = float(slow_correct - fast_correct)
        if row["invoke_label"] == 1:
            row["sample_weight"] = 3.0 if row["gain_label"] == 1 else (0.3 if row["gain_label"] == -1 else 1.0)
        else:
            row["sample_weight"] = 2.0 if row["gain_label"] == -1 else (0.5 if row["gain_label"] == 1 else 1.0)
    return rows


def train_risk_gain_model(
    X_train: np.ndarray,
    y_risk_train: np.ndarray,
    y_gain_train: np.ndarray,
    w_train: np.ndarray,
    X_cal: np.ndarray,
    y_risk_cal: np.ndarray,
    y_gain_cal: np.ndarray,
    input_dim: int,
    epochs: int = 300,
    lr: float = 1e-3,
    device: str = "cuda",
) -> Tuple[RiskGainModel, StandardScaler, np.ndarray, np.ndarray]:
    """Train Risk+Gain dual-head model using the scripts/47_ctrt_trigger.py recipe."""
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train).astype(np.float32)
    X_ca_s = scaler.transform(X_cal).astype(np.float32)
    dataset = TensorDataset(
        torch.tensor(X_tr_s),
        torch.tensor(y_risk_train, dtype=torch.float32),
        torch.tensor(y_gain_train, dtype=torch.float32),
        torch.tensor(w_train, dtype=torch.float32),
    )
    loader = DataLoader(dataset, batch_size=128, shuffle=True, drop_last=False)

    model = RiskGainModel(input_dim).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_loss, best_state, no_imp = float("inf"), None, 0
    for epoch in range(epochs):
        model.train()
        for xb, yr, yg, wb in loader:
            xb, yr, yg, wb = xb.to(device), yr.to(device), yg.to(device), wb.to(device)
            pred_risk, pred_gain = model(xb)
            loss_risk = nn.functional.binary_cross_entropy(pred_risk, yr, reduction="none")
            loss_gain = nn.functional.mse_loss(pred_gain, yg, reduction="none")
            loss = ((loss_risk + loss_gain) * wb).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            ca_t = torch.tensor(X_ca_s).to(device)
            pr, pg = model(ca_t)
            yr_t = torch.tensor(y_risk_cal, dtype=torch.float32).to(device)
            yg_t = torch.tensor(y_gain_cal, dtype=torch.float32).to(device)
            val_loss = nn.functional.binary_cross_entropy(pr, yr_t).item() + nn.functional.mse_loss(pg, yg_t).item()
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1
        if no_imp >= 30:
            break

    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        ca_t = torch.tensor(X_ca_s).to(device)
        risk_scores, gain_scores = model(ca_t)
    return model.cpu(), scaler, risk_scores.cpu().numpy(), gain_scores.cpu().numpy()


def score_risk_gain_model(
    model: RiskGainModel,
    scaler: StandardScaler,
    x: np.ndarray,
    device: str = "cuda",
) -> Tuple[np.ndarray, np.ndarray]:
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    x_s = scaler.transform(x).astype(np.float32)
    with torch.no_grad():
        risk, gain = model(torch.tensor(x_s, dtype=torch.float32, device=device))
    model = model.cpu()
    return risk.cpu().numpy(), gain.cpu().numpy()


def train_linear_decision_head(
    risk_scores: np.ndarray,
    gain_scores: np.ndarray,
    targets: np.ndarray,
    epochs: int = 200,
    lr: float = 1e-2,
    device: str = "cuda",
) -> LinearDecisionHead:
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    x_risk = torch.tensor(risk_scores, dtype=torch.float32, device=device)
    x_gain = torch.tensor(gain_scores, dtype=torch.float32, device=device)
    y = torch.tensor(targets, dtype=torch.float32, device=device)
    model = LinearDecisionHead().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    pos = float(y.sum().item())
    neg = float(y.numel() - pos)
    pos_weight = torch.tensor(neg / max(pos, 1.0), dtype=torch.float32, device=device)

    for _ in range(max(1, int(epochs))):
        pred = model(x_risk, x_gain)
        loss = nn.functional.binary_cross_entropy(
            pred.clamp(1e-6, 1.0 - 1e-6),
            y,
            weight=torch.where(y > 0.5, pos_weight, torch.ones_like(y)),
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    return model.cpu()


def conformal_calibrate(risk_scores: np.ndarray, y_risk: np.ndarray, alpha: float = 0.1) -> Tuple[float, float]:
    """Conformal calibration from scripts/47_ctrt_trigger.py."""
    n = len(y_risk)
    nonconformity = np.zeros(n)
    for i in range(n):
        if y_risk[i] == 1:
            nonconformity[i] = 1.0 - risk_scores[i]
        else:
            nonconformity[i] = risk_scores[i]
    q_hat = np.quantile(nonconformity, min(1.0, (1 - alpha) * (1 + 1.0 / n)))
    threshold = 1.0 - q_hat
    return threshold, q_hat


def evaluate(rows: List[Dict[str, Any]], risk: np.ndarray, gain: np.ndarray, risk_thr: float, gain_thr: float) -> Dict[str, Any]:
    routed = []
    for row, r, g in zip(rows, risk, gain):
        use_slow = bool(float(r) >= risk_thr and float(g) >= gain_thr)
        pred = row["slow_pred"] if use_slow and row["slow_pred"] else row["fast_pred"]
        routed.append(
            {
                "gold": row["gold"],
                "pred": pred,
                "correct": int(pred == row["gold"]),
                "used_slow": int(use_slow),
                "risk_score": float(r),
                "gain_score": float(g),
            }
        )
    return {
        "risk_threshold": risk_thr,
        "gain_threshold": gain_thr,
        "accuracy": safe_mean([row["correct"] for row in routed]),
        "macro_f1": macro_f1(routed),
        "slow_rate": safe_mean([row["used_slow"] for row in routed]),
    }


def evaluate_linear(
    rows: List[Dict[str, Any]],
    decision_scores: np.ndarray,
    threshold: float,
) -> Dict[str, Any]:
    routed = []
    for row, score in zip(rows, decision_scores):
        use_slow = bool(float(score) >= threshold)
        pred = row["slow_pred"] if use_slow and row["slow_pred"] else row["fast_pred"]
        routed.append(
            {
                "gold": row["gold"],
                "pred": pred,
                "correct": int(pred == row["gold"]),
                "used_slow": int(use_slow),
                "decision_score": float(score),
            }
        )
    return {
        "decision_mode": "linear",
        "decision_threshold": threshold,
        "accuracy": safe_mean([row["correct"] for row in routed]),
        "macro_f1": macro_f1(routed),
        "slow_rate": safe_mean([row["used_slow"] for row in routed]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the paper-style dual-head risk/gain trigger.")
    parser.add_argument("--train-trajectory-jsonl", required=True)
    parser.add_argument("--train-slow-jsonl", required=True)
    parser.add_argument("--eval-trajectory-jsonl", required=True)
    parser.add_argument("--eval-slow-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--risk-thresholds", default="0.3,0.4,0.5,0.6,0.7")
    parser.add_argument("--gain-thresholds", default="-0.1,0.0,0.1,0.2")
    parser.add_argument("--decision-mode", choices=["threshold", "linear"], default="threshold")
    parser.add_argument("--linear-thresholds", default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9")
    args = parser.parse_args()

    train_rows = build_rows(args.train_trajectory_jsonl, args.train_slow_jsonl)
    eval_rows = build_rows(args.eval_trajectory_jsonl, args.eval_slow_jsonl)
    label_fields = {"risk_label", "gain_label", "sample_weight"}
    feature_names = [name for name in select_feature_names(train_rows) if name not in label_fields]
    x_train, _ = build_matrix(train_rows, feature_names)
    x_eval, _ = build_matrix(eval_rows, feature_names)
    y_risk_train = np.asarray([row["risk_label"] for row in train_rows], dtype=np.float32)
    y_gain_train = np.asarray([row["gain_label"] for row in train_rows], dtype=np.float32)
    y_risk_eval = np.asarray([row["risk_label"] for row in eval_rows], dtype=np.float32)
    y_gain_eval = np.asarray([row["gain_label"] for row in eval_rows], dtype=np.float32)
    weights = np.asarray([row["sample_weight"] for row in train_rows], dtype=np.float32)

    model, scaler, risk_scores, gain_scores = train_risk_gain_model(
        x_train,
        y_risk_train,
        y_gain_train,
        weights,
        x_eval,
        y_risk_eval,
        y_gain_eval,
        input_dim=x_train.shape[1],
        epochs=args.epochs,
        lr=args.lr,
        device=args.device,
    )
    train_risk_scores, train_gain_scores = score_risk_gain_model(model, scaler, x_train, device=args.device)
    linear_head = train_linear_decision_head(
        train_risk_scores,
        train_gain_scores,
        np.asarray([row["invoke_label"] for row in train_rows], dtype=np.float32),
        device=args.device,
    )
    with torch.no_grad():
        eval_decision_scores = linear_head(
            torch.tensor(risk_scores, dtype=torch.float32),
            torch.tensor(gain_scores, dtype=torch.float32),
        ).numpy()

    threshold_rows = []
    for risk_thr in [float(x) for x in args.risk_thresholds.split(",") if x.strip()]:
        for gain_thr in [float(x) for x in args.gain_thresholds.split(",") if x.strip()]:
            threshold_rows.append(evaluate(eval_rows, risk_scores, gain_scores, risk_thr, gain_thr))
    for alpha in [0.05, 0.1, 0.15, 0.2]:
        risk_thr, q_hat = conformal_calibrate(risk_scores, y_risk_eval, alpha=alpha)
        row = evaluate(eval_rows, risk_scores, gain_scores, risk_thr, float("-inf"))
        row["alpha"] = alpha
        row["q_hat"] = q_hat
        row["strategy"] = "conformal_risk"
        threshold_rows.append(row)
    linear_rows = [
        evaluate_linear(eval_rows, eval_decision_scores, float(x))
        for x in args.linear_thresholds.split(",")
        if x.strip()
    ]
    best = max(threshold_rows, key=lambda row: (row["accuracy"], row["macro_f1"], -row["slow_rate"]))
    best_linear = max(linear_rows, key=lambda row: (row["accuracy"], row["macro_f1"], -row["slow_rate"]))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle = {
        "trigger_type": "dual_head_risk_gain",
        "model_state_dict": model.state_dict(),
        "input_dim": int(x_train.shape[1]),
        "hidden_dim": 128,
        "dropout": 0.3,
        "scaler": scaler,
        "feature_names": feature_names,
        "risk_threshold": best["risk_threshold"],
        "gain_threshold": best["gain_threshold"],
        "decision_mode": args.decision_mode,
        "linear_decision_state_dict": linear_head.state_dict(),
        "linear_decision_threshold": best_linear["decision_threshold"],
    }
    joblib.dump(bundle, out_dir / "dual_head_trigger.joblib")
    write_csv(out_dir / "threshold_search.csv", threshold_rows)
    write_csv(out_dir / "linear_decision_search.csv", linear_rows)
    summary = {
        "trigger_type": "dual_head_risk_gain",
        "decision_mode": args.decision_mode,
        "train_count": len(train_rows),
        "eval_count": len(eval_rows),
        "feature_count": len(feature_names),
        "risk_binary_acc_at_0p5": float(accuracy_score(y_risk_eval, (risk_scores >= 0.5).astype(np.int64))),
        "best_threshold": best,
        "best_linear": best_linear,
        "best": best_linear if args.decision_mode == "linear" else best,
        "outputs": {"model": str(out_dir / "dual_head_trigger.joblib")},
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
