#!/usr/bin/env python
"""Offline reproduction - route with a released trigger bundle and report the
strategy evaluation near the paper's reinterpretation-rate operating point
(RR ~= 0.556).

Two released trigger variants are provided in assets/:
  * Three-action policy (paper eqs. 9-10): softmax over {commit, wait,
    reinterpret} on 43 trajectory features; slow is invoked when the
    reinterpret probability dominates (margin sweep).
  * Trajectory XGB router: single-head classifier on 96 trajectory features;
    slow is invoked when the score passes a threshold.

Both variants build features from trajectory_samples.jsonl (per-prefix belief
statistics plus label log-likelihoods) produced by
scripts/build_trajectory_samples.py. The feature names stored inside each
bundle are used verbatim, so offline routing stays consistent with the
features used at training time.
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn

LABELS = ["neutral", "anger", "anxiety", "sadness", "joy", "surprise", "embarrassment"]
ALIASES = {"happy": "joy", "angry": "anger", "disgust": "anger", "fear": "anxiety", "happiness": "joy"}


def norm(x):
    return ALIASES.get(str(x).strip().lower(), str(x).strip().lower())


def probs_vec(d):
    return np.asarray([float(d.get(lab, 0.0) or 0.0) for lab in LABELS], dtype=np.float64)


def js_div(p, q):
    p = probs_vec(p)
    q = probs_vec(q)
    m = 0.5 * (p + q)
    kl = lambda a, b: float(np.sum(np.where(a > 0, a * np.log(a / np.maximum(b, 1e-12)), 0.0)))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def linear_slope(xs, ys):
    x = np.asarray(xs, np.float64)
    y = np.asarray(ys, np.float64)
    denom = float(np.sum((x - x.mean()) ** 2))
    if len(x) < 2 or denom <= 1e-12:
        return 0.0
    return float(np.sum((x - x.mean()) * (y - y.mean())) / denom)


def stable_run_len(preds):
    r = 1
    for prev in reversed(preds[:-1]):
        if prev == preds[-1]:
            r += 1
        else:
            break
    return r


def trajectory_features(steps, clip_duration=0.0):
    """Superset covering the 43-feature (policy) and 96-feature (XGB) names."""
    if not steps:
        return {}
    confs = [float(s["confidence"]) for s in steps]
    ents = [float(s["entropy"]) for s in steps]
    mgns = [float(s["margin"]) for s in steps]
    preds = [norm(s["pred"]) for s in steps]
    T = len(steps)
    xs = np.arange(1, T + 1)
    js = [js_div(steps[i]["probs"], steps[i - 1]["probs"]) for i in range(1, T)]
    ll_seq = [{lab: float(s.get("label_log_likelihoods", {}).get(lab, 0.0) or 0.0) for lab in LABELS} for s in steps]
    p_seq = [{lab: float(s.get("probs", {}).get(lab, 0.0) or 0.0) for lab in LABELS} for s in steps]

    f = {}
    final_pred = preds[-1]
    for lab in LABELS:
        f[f"fast_pred_is_{lab}"] = float(norm(final_pred) == lab)
        f[f"final_prob_{lab}"] = p_seq[-1][lab]
        f[f"prob_final_{lab}"] = p_seq[-1][lab]
        f[f"ll_final_{lab}"] = ll_seq[-1][lab]
        p_this = [p_seq[t][lab] for t in range(T)]
        ll_this = [ll_seq[t][lab] for t in range(T)]
        f[f"prob_{lab}_mean"] = float(np.mean(p_this))
        f[f"prob_{lab}_trend"] = p_this[-1] - p_this[0] if T >= 2 else 0.0
        f[f"ll_{lab}_mean"] = float(np.mean(ll_this))
        f[f"ll_{lab}_std"] = float(np.std(ll_this)) if T > 1 else 0.0
        f[f"ll_{lab}_trend"] = ll_this[-1] - ll_this[0] if T >= 2 else 0.0
        f[f"ll_{lab}_last2_diff"] = ll_this[-1] - ll_this[-2] if T >= 2 else 0.0

    # confidence: both the policy and the XGB name conventions
    f["final_confidence"] = confs[-1]
    f["conf_final"] = confs[-1]
    f["avg_confidence"] = float(np.mean(confs))
    f["conf_mean"] = float(np.mean(confs))
    f["std_confidence"] = float(np.std(confs)) if T > 1 else 0.0
    f["conf_std"] = float(np.std(confs)) if T > 1 else 0.0
    f["first_prefix_confidence"] = confs[0]
    f["max_confidence"] = max(confs)
    f["min_confidence"] = min(confs)
    f["conf_last2_diff"] = confs[-1] - confs[-2] if T >= 2 else 0.0

    # entropy conventions
    f["final_entropy"] = ents[-1]
    f["ent_final"] = ents[-1]
    f["avg_entropy"] = float(np.mean(ents))
    f["ent_mean"] = float(np.mean(ents))
    f["std_entropy"] = float(np.std(ents)) if T > 1 else 0.0
    f["ent_std"] = float(np.std(ents)) if T > 1 else 0.0
    f["max_entropy"] = max(ents)
    f["min_entropy"] = min(ents)
    f["ent_last2_diff"] = ents[-1] - ents[-2] if T >= 2 else 0.0

    # margin conventions
    f["final_margin"] = mgns[-1]
    f["margin_final"] = mgns[-1]
    f["avg_margin"] = float(np.mean(mgns))
    f["margin_mean"] = float(np.mean(mgns))
    f["std_margin"] = float(np.std(mgns)) if T > 1 else 0.0
    f["margin_std"] = float(np.std(mgns)) if T > 1 else 0.0
    f["max_margin"] = max(mgns)
    f["min_margin"] = min(mgns)
    f["margin_last2_diff"] = mgns[-1] - mgns[-2] if T >= 2 else 0.0

    # slopes / gains / trend
    f["conf_slope"] = linear_slope(xs, confs)
    f["entropy_slope"] = linear_slope(xs, ents)
    f["margin_slope"] = linear_slope(xs, mgns)
    f["conf_gain"] = confs[-1] - confs[0]
    f["entropy_change"] = ents[-1] - ents[0]
    f["margin_gain"] = mgns[-1] - mgns[0]
    f["ent_trend"] = ents[-1] - ents[0] if T >= 2 else 0.0
    f["top1_changed_from_first"] = float(final_pred != preds[0])

    # acceleration / bounce / switches / stability
    accel = [confs[i] - 2 * confs[i - 1] + confs[i - 2] for i in range(2, T)] if T >= 3 else []
    f["conf_accel_mean"] = float(np.mean(accel)) if accel else 0.0
    f["conf_accel_std"] = float(np.std(accel)) if len(accel) > 1 else 0.0
    cf_diffs = [confs[i] - confs[i - 1] for i in range(1, T)]
    bounces = sum(1 for i in range(1, len(cf_diffs)) if cf_diffs[i] * cf_diffs[i - 1] < 0) if T >= 3 else 0
    f["conf_bounce_count"] = float(bounces)
    f["conf_bounce_ratio"] = bounces / (T - 2) if T > 2 else 0.0
    if T >= 2:
        top1_diffs = [max(p_seq[i].values()) - max(p_seq[i - 1].values()) for i in range(1, T)]
        f["prob_top1_bounce_count"] = float(sum(1 for i in range(1, len(top1_diffs))
                                                      if top1_diffs[i] * top1_diffs[i - 1] < 0)) if len(top1_diffs) >= 2 else 0.0
    else:
        f["prob_top1_bounce_count"] = 0.0
    f["switch_count"] = float(sum(1 for i in range(1, T) if preds[i] != preds[i - 1]))
    f["switch_rate"] = f["switch_count"] / max(T - 1, 1)
    max_run, cur = 1, 1
    for i in range(1, T):
        cur = cur + 1 if preds[i] == preds[i - 1] else 1
        max_run = max(max_run, cur)
    f["stable_run_length"] = float(max_run)
    f["stable_run_ratio"] = max_run / T if T > 0 else 0.0
    f["last_label_stable"] = int(T >= 2 and preds[-1] == preds[-2])
    f["label_changed_last_step"] = int(T >= 2 and preds[-1] != preds[-2])
    f["high_context_fast_pred"] = float(preds[-1] in {"embarrassment", "anxiety", "neutral", "joy"})

    f["js_mean"] = float(np.mean(js)) if js else 0.0
    f["js_max"] = float(np.max(js)) if js else 0.0

    if p_seq:
        sf = sorted(p_seq[-1].values(), reverse=True)
        f["top1_prob"], f["top2_prob"] = (sf[0], sf[1] if len(sf) > 1 else 0.0)
        f["top1_top2_gap"] = (sf[0] - sf[1]) if len(sf) > 1 else sf[0]
    else:
        f["top1_prob"] = f["top2_prob"] = f["top1_top2_gap"] = 0.0
    t1, t2 = [], []
    for t in range(T):
        si = sorted(p_seq[t].items(), key=lambda x: x[1], reverse=True)
        t1.append(si[0][0] if si else "")
        t2.append(si[1][0] if len(si) > 1 else "")
    f["top1_swap_count"] = float(sum(1 for i in range(1, T) if t1[i] != t1[i - 1]))
    f["top1_top2_identity_swap"] = float(sum(1 for i in range(1, T) if t1[i] == t2[i - 1] and t1[i] != t1[i - 1]))

    vol = [sum(abs(p_seq[t][lab] - p_seq[t - 1][lab]) for lab in LABELS) for t in range(1, T)] if T >= 2 else []
    f["prob_volatility_mean"] = float(np.mean(vol)) if vol else 0.0
    f["prob_volatility_max"] = float(np.max(vol)) if vol else 0.0
    ent_diffs = [ents[i] - ents[i - 1] for i in range(1, T)] if T >= 2 else []
    f["ent_increasing_steps"] = float(sum(1 for d in ent_diffs if d > 0))
    f["ent_decreasing_steps"] = float(sum(1 for d in ent_diffs if d < 0))
    f["top1_persistence"] = sum(1 for p in preds if p == preds[-1]) / T
    f["entropy_auc"] = float(np.mean(ents))
    f["clip_duration"] = clip_duration
    f["num_prefixes"] = float(T)
    return f




def route_acc(utts, use_slow):
    ok = 0
    for i, u in enumerate(utts):
        pred = u["slow_pred"] if use_slow[i] and u["slow_pred"] in LABELS else u["fast_pred"]
        ok += int(pred == u["gold"])
    return ok / len(utts) if utts else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trajectory-jsonl", required=True)
    ap.add_argument("--slow-jsonl", required=True)
    ap.add_argument("--trigger-bundle", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--min-rr", type=float, default=0.50)
    ap.add_argument("--max-rr", type=float, default=0.62)
    ap.add_argument("--target-rr", type=float, default=0.556)
    args = ap.parse_args()

    pack = joblib.load(Path(args.trigger_bundle))
    if isinstance(pack, dict):
        feature_names = list(pack.get("feature_names") or [])
        kind = pack.get("type", "single_head")
        weights = pack.get("model_state_dict")
        model_pipe = pack.get("model") if not isinstance(pack.get("model", None), type(None)) else None
    else:
        feature_names = list(getattr(pack, "feature_names_in_", []))
        kind = "single_head"
        weights = None
        model_pipe = pack
    print("bundle kind:", kind, "| feature count:", len(feature_names), flush=True)

    traj = {str(json.loads(l)["sample_id"]): json.loads(l) for l in open(args.trajectory_jsonl, encoding="utf-8")}
    slow = {}
    for l in open(args.slow_jsonl, encoding="utf-8"):
        r = json.loads(l)
        sid = str(r.get("id") or r.get("sample_id") or "")
        if sid:
            slow[sid] = norm(r.get("output_final_emotion") or r.get("corrected_emotion") or "")
    utts = []
    for sid, t in traj.items():
        sp = slow.get(sid)
        if sp is None or not sp:
            continue
        feats = trajectory_features(t["trajectory"], t.get("clip_duration_sec", 0.0))
        last = t["trajectory"][-1]
        utts.append({**feats, "gold": t["gold"], "fast_pred": last["pred"], "slow_pred": sp})
    print("matched utterances:", len(utts), flush=True)

    use_names = [nn for nn in feature_names if nn in utts[0]]
    missing = [nn for nn in feature_names if nn not in utts[0]]
    if missing:
        print("WARN unavailable feature names:", missing, flush=True)
    X = torch.tensor(np.asarray([[u[nn] for nn in use_names] for u in utts], dtype=np.float32))
    n = len(utts)

    if kind == "policy_paper_serial":
        # Paper-consistent serial trigger (eq. 7-10): shared encoder z,
        # risk/gain heads read z, three-action head consumes (z, risk, gain).
        arch = pack.get("arch", {"hidden": 128, "hidden2": 64, "dropout": 0.3, "action_hidden": 32})

        class _PaperSerial(nn.Module):
            def __init__(self, F, h, h2, dropout, ah):
                super().__init__()
                self.shared = nn.Sequential(
                    nn.Linear(F, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout),
                    nn.Linear(h, h2), nn.BatchNorm1d(h2), nn.ReLU(), nn.Dropout(dropout),
                )
                self.risk = nn.Linear(h2, 1)
                self.gain = nn.Linear(h2, 1)
                self.action = nn.Sequential(nn.Linear(h2 + 2, ah), nn.ReLU(), nn.Linear(ah, 3))
            def forward(self, x):
                z = self.shared(x)
                risk = torch.sigmoid(self.risk(z)).squeeze(-1)
                gain = torch.tanh(self.gain(z)).squeeze(-1)
                a = self.action(torch.cat([z, risk.unsqueeze(-1), gain.unsqueeze(-1)], dim=-1))
                return risk, gain, a

        model = _PaperSerial(len(use_names), arch.get("hidden", 128), arch.get("hidden2", 64),
                             arch.get("dropout", 0.3), arch.get("action_hidden", 32))
        sd = {k: torch.tensor(v) for k, v in weights.items()}
        model.load_state_dict(sd)
        model.eval()
        with torch.no_grad():
            _, _, probs_t = model(X)
        probs = nn.functional.softmax(probs_t, dim=-1).numpy()
        p_co, p_re = probs[:, 0], probs[:, 2]
        rows = []
        for tau in np.arange(0.1, 0.91, 0.01):
            with np.errstate(invalid="ignore", divide="ignore"):
                dom = np.where((p_re + p_co) > 0, p_re / (p_re + p_co), 0.0)
            use = dom >= tau
            rr = float(use.mean())
            if rr < args.min_rr or rr > args.max_rr:
                continue
            rows.append((round(float(tau), 2), round(rr, 4), round(route_acc(utts, use), 4)))
    else:
        model = model_pipe
        Xn = X.numpy()
        if hasattr(model, "predict_proba"):
            scores = np.asarray(model.predict_proba(Xn)[:, 1], np.float64)
        elif hasattr(model, "decision_function"):
            raw = np.asarray(model.decision_function(Xn), np.float64)
            scores = 1.0 / (1.0 + np.exp(-raw))
        else:
            scores = np.asarray(model.predict(Xn), np.float64)
        rows = []
        for thr in np.arange(0.05, 0.96, 0.01):
            use = scores >= thr
            rr = float(use.mean())
            if rr < args.min_rr or rr > args.max_rr:
                continue
            rows.append((round(float(thr), 2), round(rr, 4), round(route_acc(utts, use), 4)))

    best = min(rows, key=lambda r: abs(r[1] - args.target_rr)) if rows else None
    print("--- sweep rows (RR in range) ---", flush=True)
    for r in rows[:60]:
        print("  thresh=%.2f RR=%.4f acc=%.4f" % r, flush=True)
    print("best near target RR=%.3f: %s" % (args.target_rr, best), flush=True)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {"kind": kind, "feature_names": use_names, "matched": n,
             "best_near_target": best, "sweep": rows},
            f, ensure_ascii=False, indent=2,
        )


if __name__ == "__main__":
    main()
