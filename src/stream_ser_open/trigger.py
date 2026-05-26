from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import torch
import torch.nn as nn


@dataclass
class TriggerDecision:
    use_slow: bool
    score: float
    reason: str


def _safe_float(value: Any, fallback: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def build_online_features(state: Dict[str, Any], labels: List[str]) -> Dict[str, float]:
    probs = state.get("fast_probs", {}) or {}
    conf_hist = [_safe_float(x) for x in state.get("conf_last_k", [])]
    ent_hist = [_safe_float(x) for x in state.get("entropy_last_k", [])]
    margin_hist = [_safe_float(x) for x in state.get("margin_last_k", [])]
    preds = [str(x) for x in state.get("traj_last_k", [])] + [str(state.get("fast_pred", ""))]
    switch_count = sum(1 for i in range(1, len(preds)) if preds[i] != preds[i - 1])
    final_conf = _safe_float(state.get("fast_confidence"))
    final_entropy = _safe_float(state.get("fast_entropy"))
    final_margin = _safe_float(state.get("fast_margin"))
    xs = list(range(len(conf_hist) + 1))
    conf_series = conf_hist + [final_conf]
    ent_series = ent_hist + [final_entropy]
    margin_series = margin_hist + [final_margin]

    def slope(values: List[float]) -> float:
        if len(values) < 2:
            return 0.0
        x = np.asarray(xs, dtype=np.float64)
        y = np.asarray(values, dtype=np.float64)
        denom = float(np.sum((x - x.mean()) ** 2))
        if denom <= 1e-12:
            return 0.0
        return float(np.sum((x - x.mean()) * (y - y.mean())) / denom)

    features: Dict[str, float] = {
        "final_confidence": final_conf,
        "final_entropy": final_entropy,
        "final_margin": final_margin,
        "switch_count": float(switch_count),
        "switch_rate": switch_count / max(len(preds) - 1, 1),
        "num_prefixes": float(len(preds)),
        "first_prefix_confidence": conf_series[0],
        "max_confidence": max(conf_series),
        "min_confidence": min(conf_series),
        "avg_confidence": float(np.mean(conf_series)),
        "std_confidence": float(np.std(conf_series)),
        "min_entropy": min(ent_series),
        "max_entropy": max(ent_series),
        "avg_entropy": float(np.mean(ent_series)),
        "std_entropy": float(np.std(ent_series)),
        "max_margin": max(margin_series),
        "min_margin": min(margin_series),
        "avg_margin": float(np.mean(margin_series)),
        "std_margin": float(np.std(margin_series)),
        "conf_slope": slope(conf_series),
        "entropy_slope": slope(ent_series),
        "margin_slope": slope(margin_series),
        "conf_gain": final_conf - (conf_hist[0] if conf_hist else final_conf),
        "entropy_change": final_entropy - (ent_hist[0] if ent_hist else final_entropy),
        "margin_gain": final_margin - (margin_hist[0] if margin_hist else final_margin),
        "last_label_stable": float(len(preds) >= 2 and preds[-1] == preds[-2]),
        "label_changed_last_step": float(len(preds) >= 2 and preds[-1] != preds[-2]),
        "high_context_fast_pred": float(state.get("fast_pred") in {"embarrassment", "anxiety", "neutral", "joy"}),
    }
    stable_run = 1
    for prev in reversed(preds[:-1]):
        if prev == preds[-1]:
            stable_run += 1
        else:
            break
    features["stable_run_length"] = float(stable_run)
    for label in labels:
        features[f"fast_pred_is_{label}"] = float(state.get("fast_pred") == label)
        features[f"final_prob_{label}"] = _safe_float(probs.get(label))
    return features


class SlowTrigger:
    def __init__(
        self,
        labels: List[str],
        model_path: Optional[str] = None,
        threshold: float = 0.5,
        decision_mode: str = "auto",
        min_confidence: float = 0.55,
        min_margin: float = 0.15,
        max_entropy: float = 1.20,
    ) -> None:
        self.labels = labels
        self.threshold = threshold
        self.decision_mode_override = str(decision_mode or "auto").strip().lower()
        self.min_confidence = min_confidence
        self.min_margin = min_margin
        self.max_entropy = max_entropy
        self.model = None
        self.scaler = None
        self.trigger_type = "rule"
        self.risk_threshold = threshold
        self.gain_threshold = 0.0
        self.decision_mode = "threshold"
        self.linear_decision_threshold = 0.5
        self.linear_decision_head = None
        self.hidden_dim = 128
        self.dropout = 0.3
        self.feature_names: List[str] = []
        if model_path:
            bundle = joblib.load(Path(model_path))
            if isinstance(bundle, dict) and bundle.get("trigger_type") == "dual_head_risk_gain":
                self.trigger_type = "dual_head_risk_gain"
                self.feature_names = list(bundle.get("feature_names", []))
                self.scaler = bundle.get("scaler")
                self.risk_threshold = float(bundle.get("risk_threshold", threshold))
                self.gain_threshold = float(bundle.get("gain_threshold", 0.0))
                self.decision_mode = str(bundle.get("decision_mode", "threshold") or "threshold").strip().lower()
                if self.decision_mode_override in {"threshold", "linear"}:
                    self.decision_mode = self.decision_mode_override
                self.linear_decision_threshold = float(bundle.get("linear_decision_threshold", 0.5))
                self.hidden_dim = int(bundle.get("hidden_dim", 128))
                self.dropout = float(bundle.get("dropout", 0.3))
                self.model = _RiskGainModel(
                    int(bundle["input_dim"]),
                    hidden_dim=self.hidden_dim,
                    dropout=self.dropout,
                )
                self.model.load_state_dict(bundle["model_state_dict"])
                self.model.eval()
                if "linear_decision_state_dict" in bundle:
                    self.linear_decision_head = _LinearDecisionHead()
                    self.linear_decision_head.load_state_dict(bundle["linear_decision_state_dict"])
                    self.linear_decision_head.eval()
            elif isinstance(bundle, dict) and "model" in bundle:
                self.trigger_type = "single_head"
                self.model = bundle["model"]
                self.feature_names = list(bundle.get("feature_names", []))
            else:
                self.trigger_type = "single_head"
                self.model = bundle

    def decide(self, state: Dict[str, Any]) -> TriggerDecision:
        if self.model is not None and self.trigger_type == "dual_head_risk_gain":
            features = build_online_features(state, self.labels)
            names = self.feature_names or sorted(features.keys())
            x = np.asarray([[features.get(name, 0.0) for name in names]], dtype=np.float64)
            if self.scaler is not None:
                x = self.scaler.transform(x)
            with torch.no_grad():
                risk, gain = self.model(torch.tensor(x, dtype=torch.float32))
            risk_score = float(risk.item())
            gain_score = float(gain.item())
            if self.decision_mode == "linear" and self.linear_decision_head is not None:
                with torch.no_grad():
                    decision_score = float(self.linear_decision_head(risk, gain).item())
                use_slow = decision_score >= self.linear_decision_threshold
                state["trigger_decision_score"] = decision_score
                decision_reason = "dual_head_linear_decision"
                score = decision_score
            else:
                use_slow = risk_score >= self.risk_threshold and gain_score >= self.gain_threshold
                decision_reason = "dual_head_threshold"
                score = risk_score * max(gain_score, 0.0)
            state["trigger_risk_score"] = risk_score
            state["trigger_gain_score"] = gain_score
            state["trigger_decision_mode"] = self.decision_mode
            return TriggerDecision(use_slow, score, decision_reason)

        if self.model is not None:
            features = build_online_features(state, self.labels)
            names = self.feature_names or sorted(features.keys())
            x = np.asarray([[features.get(name, 0.0) for name in names]], dtype=np.float64)
            if hasattr(self.model, "predict_proba"):
                proba = self.model.predict_proba(x)
                score = float(proba[0, 1]) if proba.ndim == 2 and proba.shape[1] > 1 else float(proba[0])
            elif hasattr(self.model, "decision_function"):
                raw = float(self.model.decision_function(x)[0])
                score = 1.0 / (1.0 + np.exp(-raw))
            else:
                score = float(self.model.predict(x)[0])
            return TriggerDecision(score >= self.threshold, score, "learned_trigger")

        confidence = _safe_float(state.get("fast_confidence"))
        margin = _safe_float(state.get("fast_margin"))
        entropy = _safe_float(state.get("fast_entropy"))
        unstable = bool(state.get("traj_last_k")) and str(state["traj_last_k"][-1]) != str(state.get("fast_pred"))
        use_slow = confidence < self.min_confidence or margin < self.min_margin or entropy > self.max_entropy or unstable
        score = max(1.0 - confidence, 1.0 - margin, entropy / max(self.max_entropy, 1e-6))
        return TriggerDecision(use_slow, float(score), "rule_trigger")


class _RiskGainModel(nn.Module):
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

    def forward(self, x: torch.Tensor):
        h = self.shared(x)
        risk = torch.sigmoid(self.risk_head(h)).squeeze(-1)
        gain = torch.tanh(self.gain_head(h)).squeeze(-1)
        return risk, gain


class _LinearDecisionHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 1)

    def forward(self, risk: torch.Tensor, gain: torch.Tensor) -> torch.Tensor:
        x = torch.stack([risk, gain], dim=-1)
        return torch.sigmoid(self.linear(x)).squeeze(-1)
