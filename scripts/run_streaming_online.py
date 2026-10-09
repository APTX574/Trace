#!/usr/bin/env python
"""Streaming (online) TRACE inference.

For each utterance in a stream it runs the fast axis progressively at every
audio prefix (1.5 s start, 1.0 s stride, plus the full clip), builds the
43-feature trajectory state, and applies the released three-action policy:

  action = softmax(commit, wait, reinterpret)

  * commit      -> final = current fast-prefix prediction.
  * wait        -> keep accumulating the next prefix.
  * reinterpret -> invoke the slow axis once the full clip has arrived,
                   final = slow output.

The script chains the released fast/slow checkpoints and the three-action
policy bundle (assets/), and reports the paper's streaming metrics: accuracy,
slow-call rate.

The fast axis scores each prefix from audio plus the single prefix-frame
(sampled at prefix_time = prefix - 1 ms, long side <= 448), matching the
released fast checkpoint's input. Two decision modes are provided:

  * `incremental` (default)  — a decision is made at every arriving prefix,
    so a confident early commit emits the final answer without waiting for
    the full clip (true streaming latency).
  * `full_trajectory`         — all prefixes are scored first and the policy
    is applied once on the complete trajectory. This reproduces the offline
    apply_trigger protocol (and hence the offline RR/accuracy operating
    point) on the routed set.

When `--prefix-cache-dir` points at the pre-extracted audio/frame cache the
offline fast-belief script produces (audio.npy + frame.png layout), per-prefix
assets are loaded from the cache for bit-exact agreement with the offline
trajectory; otherwise they are decoded live from the clips.
"""

import argparse
import json
import os
import sys
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import av  # noqa: E402
import hashlib  # noqa: E402
import joblib  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402

try:
    import decord  # noqa: E402
except ImportError:
    decord = None

from qwen_omni_utils import process_mm_info  # noqa: E402
from stream_ser_open.models import load_qwen_omni  # noqa: E402
from run_slow_batch import build_inputs as build_slow_inputs, build_slow_prompt, parse_json_response  # noqa: E402
from apply_trigger import LABELS, trajectory_features  # noqa: E402


def format_prefix_tag(prefix_sec: float, clip_duration_sec: float, is_full_prefix: bool) -> str:
    """Same prefix-tag encoding as the offline fast-belief script."""
    if is_full_prefix:
        rounded = round(float(clip_duration_sec), 3)
        if abs(rounded - round(rounded)) < 1e-6:
            return f"{int(round(rounded))}s"
        safe = str(rounded).replace(".", "p")
        return f"full_{safe}s"
    rounded = round(float(prefix_sec), 3)
    if abs(rounded - round(rounded)) < 1e-6:
        return f"{int(round(rounded))}s"
    return f"{str(rounded).replace('.', 'p')}s"


def extract_prefix_frame(path: str, prefix_sec: float, clip_duration_sec: float, max_pixels: int = 448):
    """Extract the single prefix frame (decord, av fallback), long side <= max_pixels.

    Mirrors the offline fast-belief frame marking (prefix_time = prefix - 1 ms).
    """
    image = None
    if decord is not None:
        try:
            reader = decord.VideoReader(path)
            total_frames = len(reader)
            fps = float(reader.get_avg_fps())
            if total_frames > 0 and fps > 0:
                safe_prefix = max(0.0, min(prefix_sec, clip_duration_sec))
                frame_time = max(0.0, safe_prefix - 1e-3)
                frame_idx = max(0, min(int(round(frame_time * fps)), total_frames - 1))
                image = Image.fromarray(reader[frame_idx].asnumpy()).convert("RGB")
        except Exception:
            image = None
    if image is None:
        with av.open(path) as container:
            if not container.streams.video:
                raise ValueError(f"No video stream found in {path}")
            stream = container.streams.video[0]
            fps = float(stream.average_rate) if stream.average_rate is not None else 0.0
            if fps <= 0 and stream.base_rate is not None:
                fps = float(stream.base_rate)
            safe_prefix = max(0.0, min(prefix_sec, clip_duration_sec))
            frame_time = max(0.0, safe_prefix - 1e-3)
            target_idx = max(0, int(round(frame_time * fps)))
            last_frame = None
            for idx, frame in enumerate(container.decode(video=0)):
                last_frame = frame
                if idx >= target_idx:
                    break
            if last_frame is None:
                raise ValueError(f"Decoded zero video frames from {path}")
            image = Image.fromarray(last_frame.to_ndarray(format="rgb24")).convert("RGB")
    width, height = image.size
    long_side = max(width, height)
    if long_side > max_pixels:
        scale = max_pixels / float(long_side)
        image = image.resize((max(1, int(round(width * scale))), max(1, int(round(height * scale)))))
    return image


def probe_duration_sec(path: str) -> float:
    try:
        with av.open(path) as c:
            if c.duration is not None:
                return float(c.duration / av.time_base)
            st = c.streams.video[0]
            return float(st.duration * st.time_base)
    except Exception:
        return 0.0


def audio_prefix(path: str, sec: float, sr: int = 16000, cache: Dict[str, np.ndarray] = None):
    end = max(1, int(round(sec * sr)))
    wave = None
    if cache is not None:
        wave = cache.get(path)
    if wave is None:
        chunks: List[np.ndarray] = []
        with av.open(path) as c:
            res = av.audio.resampler.AudioResampler(format="fltp", layout="mono", rate=sr)
            for fr in c.decode(audio=0):
                rs = res.resample(fr)
                if rs is None:
                    continue
                for it in (rs if isinstance(rs, list) else [rs]):
                    ch = np.asarray(it.to_ndarray(), dtype=np.float32)
                    if ch.ndim == 2:
                        ch = ch[0]
                    chunks.append(ch.reshape(-1))
        wave = np.concatenate(chunks)
        if cache is not None and wave.size:
            cache[path] = wave
    return np.asarray(wave[: min(end, wave.size)], dtype=np.float32)


def prefix_cache_paths(cache_dir, dataset_tag, sample_id, prefix_tag):
    key = f"{dataset_tag}::{sample_id}::prefix_{prefix_tag}"
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()
    root = Path(cache_dir) / dataset_tag / digest[:2] / digest
    return root / "audio.npy", root / "frame.png"


def fast_score(model, processor, prefix_audio, prompt_text, device, frame=None):
    user_content = [{"type": "audio", "audio": prefix_audio}]
    if frame is not None:
        user_content.append({"type": "image", "image": frame})
    user_content.append({"type": "text", "text": prompt_text})
    conv = [
        {"role": "system", "content": [{"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}]},
        {"role": "user", "content": user_content},
    ]
    audios, images, _ = process_mm_info(conv, use_audio_in_video=False)
    text = processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
    kwargs = {"text": [text], "audio": [audios[0]], "return_tensors": "pt",
              "padding": True, "use_audio_in_video": False}
    if images:
        kwargs["images"] = [images[0]]
    inputs = processor(**kwargs).to(device)
    with torch.no_grad():
        logits = model(**inputs, use_audio_in_video=False).logits[0, -1, :]
    lp = F.log_softmax(logits, dim=-1)
    lambdas = []
    for label in LABELS:
        ids = processor.tokenizer.encode(label, add_special_tokens=False)
        lambdas.append(float(lp[ids[0]].item()) if ids else float("-inf"))
    probs_t = torch.softmax(torch.tensor(lambdas), dim=0)
    probs = {label: float(probs_t[i].item()) for i, label in enumerate(LABELS)}
    ordered = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
    pred, conf = ordered[0]
    margin = conf - (ordered[1][1] if len(ordered) > 1 else 0.0)
    ent = -float(sum(p * np.log(p + 1e-12) for p in probs.values()))
    # NOTE: the released trajectory records store the softmax probabilities under
    # the key `label_log_likelihoods` (matching the offline script's field
    # naming), so the trajectory features align between the online and offline
    # routing paths.
    return {"pred": pred, "confidence": float(conf),
            "entropy": float(ent), "margin": float(margin), "probs": probs,
            "label_log_likelihoods": probs}


def fast_prompt(dialogue: str, prefix_sec: float, clip_duration: float, include_dialogue: bool) -> str:
    labels_str = ", ".join(LABELS)
    ratio = min(1.0, prefix_sec / clip_duration) if clip_duration > 1e-6 else 0.0
    lines = [
        "You are the fast belief tracker in streaming emotion understanding.",
        "Infer the speaker's current emotion from the observed prefix only.",
        "The evidence is partial and arrives incrementally.",
        "Prioritize vocal cues such as prosody, speaking rate, pauses, tremble, breathiness, laughter, crying, and intensity.",
        "Use the single visual frame only as supporting evidence for facial expression or posture.",
        "Do not assume evidence from the unobserved future part of the clip.",
        f"Observed prefix duration: {prefix_sec:.2f}s out of total clip duration {clip_duration:.2f}s.",
        f"Observed fraction of the clip: {ratio:.2%}.",
        f"Reply with exactly one emotion label from: {labels_str}.",
    ]
    d = (dialogue or "").strip()
    if include_dialogue and d:
        lines.append(f'Current dialogue transcript: "{d}".')
    lines.append("The speaker's emotion in the observed prefix is:")
    return "\n".join(lines)


def build_prefix_list(duration: float, min_sec: float, step: float, include_full: bool) -> List[Tuple[float, bool]]:
    if duration <= 1e-6:
        return []
    out = []
    cur = min_sec
    while cur + 1e-6 < duration:
        out.append((round(cur, 6), False))
        cur += step
    if not out:
        return [(round(duration, 6), True)]
    last = out[-1][0]
    if include_full and abs(last - duration) > 1e-6:
        out.append((round(duration, 6), True))
    elif abs(last - duration) <= 1e-6:
        out[-1] = (out[-1][0], True)
    return out


def decide_action(proba, tau):
    """Three-action decision from the policy softmax over
    {commit, wait, reinterpret}. Reinterpret dominates when its share of
    (commit + reinterpret) passes tau — the same rule the offline
    apply_trigger sweep uses to select the operating point.
    """
    p_co, p_wa, p_re = float(proba[0]), float(proba[1]), float(proba[2])
    dom = p_re / (p_re + p_co) if (p_re + p_co) > 0 else 0.0
    if dom >= tau:
        return "reinterpret"
    if p_co >= max(p_wa, p_re):
        return "commit"
    return "wait"


def slow_infer(processor, model, sample, fps, max_pixels, device):
    prompt_text = build_slow_prompt(sample, LABELS)
    inputs = build_slow_inputs(processor, sample, prompt_text, fps, max_pixels, device, use_audio_in_video=True)
    with torch.no_grad():
        generated = model.generate(
            **inputs, max_new_tokens=96, temperature=0.1,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
            use_audio_in_video=True,
        )
    trimmed = generated[:, inputs["input_ids"].shape[1]:]
    raw = processor.tokenizer.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
    parsed = parse_json_response(raw)
    pred = str(parsed.get("final_emotion") or parsed.get("corrected_emotion") or "").strip().lower()
    return (pred if pred in LABELS else "neutral"), raw


def main() -> None:
    ap = argparse.ArgumentParser(description="Streaming (online) TRACE inference with the three-action policy.")
    ap.add_argument("--fast-model-path", required=True)
    ap.add_argument("--fast-adapter-path", required=True)
    ap.add_argument("--slow-model-path", required=True)
    ap.add_argument("--slow-adapter-path", required=True)
    ap.add_argument("--policy-bundle", required=True)
    ap.add_argument("--input-jsonl", required=True)
    ap.add_argument("--video-root", default="")
    ap.add_argument("--output-jsonl", required=True)
    ap.add_argument("--min-prefix-sec", type=float, default=1.5)
    ap.add_argument("--prefix-step-sec", type=float, default=1.0)
    ap.add_argument("--include-full-prefix", type=lambda x: (str(x).lower() in {"1", "true", "yes"}), default=True)
    ap.add_argument("--include-dialogue", type=lambda x: (str(x).lower() in {"1", "true", "yes"}), default=False)
    ap.add_argument("--fps", type=int, default=2)
    ap.add_argument("--max-pixels", type=int, default=28224)
    ap.add_argument("--commit-tau", type=float, default=None, help="Reinterpret-dominance margin; default uses the bundle operating point.")
    ap.add_argument("--decision-mode", choices=["incremental", "full_trajectory"], default="incremental",
                    help="incremental: decide at every arriving prefix (true streaming). "
                         "full_trajectory: run all prefixes first, then decide once on the complete "
                         "trajectory (equivalent to the offline apply_trigger protocol; reproduces "
                         "the offline RR/accuracy operating point).")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--prefix-cache-dir", default="",
                    help="Pre-extracted prefix asset cache (audio.npy + frame.png layout, as produced by the "
                         "offline fast-belief script). When provided, per-prefix audio/frame are loaded from the "
                         "cache for bit-exact reproduction of the offline trajectory; otherwise they are decoded "
                         "live from the video clips.")
    ap.add_argument("--cache-dataset-tag", default="",
                    help="Dataset tag used to locate the cache subtree (defaults to the input file stem, matching "
                         "the offline script's Path(jsonl).stem convention).")
    args = ap.parse_args()

    bundle = joblib.load(Path(args.policy_bundle))
    feature_names = list(bundle["feature_names"])
    # Bundle operating points use `dominance_tau` (policy_paper_serial) or the
    # legacy `margin_tau` (policy3_three_action); --commit-tau overrides both.
    operating_point = bundle.get("operating_point", {}) or {}
    default_tau = operating_point.get("dominance_tau", operating_point.get("margin_tau", 0.5))
    tau = args.commit_tau if args.commit_tau is not None else float(default_tau)

    dev = torch.device(args.device)
    fast_proc, fast_model = load_qwen_omni(args.fast_model_path, args.fast_adapter_path, dev)
    slow_proc, slow_model = load_qwen_omni(args.slow_model_path, args.slow_adapter_path, dev)
    fast_model.eval()
    slow_model.eval()

    arch = bundle["arch"]
    # Paper-consistent serial trigger (eq. 7-10): shared encoder z,
    # risk/gain heads read z, three-action head consumes (z, risk, gain).
    class _PaperSerial(torch.nn.Module):
        def __init__(self, F, h, h2, dropout, ah):
            super().__init__()
            self.shared = torch.nn.Sequential(
                torch.nn.Linear(F, h), torch.nn.BatchNorm1d(h), torch.nn.ReLU(), torch.nn.Dropout(dropout),
                torch.nn.Linear(h, h2), torch.nn.BatchNorm1d(h2), torch.nn.ReLU(), torch.nn.Dropout(dropout),
            )
            self.risk = torch.nn.Linear(h2, 1)
            self.gain = torch.nn.Linear(h2, 1)
            self.action = torch.nn.Sequential(torch.nn.Linear(h2 + 2, ah), torch.nn.ReLU(), torch.nn.Linear(ah, 3))
        def forward(self, x):
            z = self.shared(x)
            risk = torch.sigmoid(self.risk(z)).squeeze(-1)
            gain = torch.tanh(self.gain(z)).squeeze(-1)
            a = self.action(torch.cat([z, risk.unsqueeze(-1), gain.unsqueeze(-1)], dim=-1))
            return a

    pol = _PaperSerial(
        len(feature_names),
        arch.get("hidden", 128), arch.get("hidden2", 64),
        arch.get("dropout", 0.3), arch.get("action_hidden", 32),
    )
    pol.load_state_dict({k: torch.tensor(v) for k, v in bundle["model_state_dict"].items()})
    pol.to(dev).eval()

    dataset_tag = args.cache_dataset_tag or Path(args.input_jsonl).stem
    rows = [json.loads(l) for l in open(args.input_jsonl, encoding="utf-8")]
    by_clip = defaultdict(list)
    for r in rows:
        by_clip[str(r.get("clip_id") or r.get("group_id") or "")].append(r)
    for v in by_clip.values():
        v.sort(key=lambda r: float(r.get("step", r.get("start_time", 0.0))))

    def resolve(path):
        p = str(path or "").strip()
        if args.video_root and p and not os.path.isabs(p):
            p = os.path.join(args.video_root, p)
        return p

    streams = defaultdict(list)
    for r in rows:
        streams[str(r.get("stream_id") or "")].append(r)
    for v in streams.values():
        v.sort(key=lambda r: float(r.get("step", r.get("start_time", 0.0))))

    wav_cache: Dict[str, np.ndarray] = {}
    out = []
    n_total = sum(len(v) for v in streams.values())
    done = 0
    for sid, stream_rows in streams.items():
        clip_key = str(stream_rows[0].get("clip_id") or stream_rows[0].get("group_id") or "")
        clip_rows = by_clip.get(clip_key, [])
        summ = {}
        for si, sample in enumerate(stream_rows):
            rec_id = str(sample.get("sample_id") or sample.get("id") or f"{sid}::{si}")
            gold = str(sample.get("target") or sample.get("gt_emotion") or sample.get("emotion") or "").strip().lower()
            dialogue = str(sample.get("dialogue") or sample.get("text") or "").strip()
            cur = resolve(sample.get("resolved_video_path") or sample.get("video_path"))
            prev_summary = " | ".join([str(x) for x in summ.values() if str(x).strip()])

            idx = next((i for i, c in enumerate(clip_rows) if str(c.get("sample_id") or c.get("id")) == rec_id), -1)
            prev_path = resolve(clip_rows[idx - 1].get("resolved_video_path") or clip_rows[idx - 1].get("video_path")) if idx > 0 else ""

            duration = probe_duration_sec(cur)
            if duration <= 1e-6:
                print(f"[skip] no duration {cur}", flush=True)
                continue
            prefixes = build_prefix_list(duration, args.min_prefix_sec, args.prefix_step_sec, args.include_full_prefix)

            traj = []
            final_pred = None
            source = None
            actions = []

            def policy_feature_proba(trajectory):
                feats = trajectory_features(trajectory, duration)
                x = torch.tensor(np.asarray([[feats.get(n, 0.0) for n in feature_names]], dtype=np.float32)).to(dev)
                with torch.no_grad():
                    return F.softmax(pol(x), dim=-1)[0].cpu().numpy()

            def load_prefix_assets(prefix_sec, is_full):
                # Cache asset when available (bit-exact), otherwise decode live:
                # audio from the clip, and the single prefix-frame at the same
                # time point the offline script marks (prefix_time = sec - 1 ms).
                if args.prefix_cache_dir:
                    ap_, fp = prefix_cache_paths(args.prefix_cache_dir, dataset_tag,
                                                 rec_id, format_prefix_tag(prefix_sec, duration, is_full))
                    if ap_.exists() and fp.exists():
                        return np.load(ap_), Image.open(fp).convert("RGB")
                a = audio_prefix(cur, prefix_sec, cache=wav_cache)
                frame = extract_prefix_frame(cur, prefix_sec, duration, 448)
                return a, frame

            def invoke_slow():
                slow_sample = {
                    "current_video_path": cur,
                    "previous_video_path": prev_path,
                    "speaker": str(sample.get("speaker", "") or ""),
                    "dialogue": dialogue,
                    "prev_summary": prev_summary,
                }
                return slow_infer(slow_proc, slow_model, slow_sample, args.fps, args.max_pixels, dev)

            if args.decision_mode == "full_trajectory":
                # Offline-equivalent protocol: run every prefix first, then make
                # one decision on the complete trajectory (as apply_trigger.py
                # does on trajectory_samples.jsonl).
                for prefix_sec, is_full in prefixes:
                    a, frame = load_prefix_assets(prefix_sec, is_full)
                    pmsg = fast_prompt(dialogue, prefix_sec, duration, args.include_dialogue)
                    st = fast_score(fast_model, fast_proc, a, pmsg, dev, frame=frame)
                    st["prefix_sec"] = prefix_sec
                    traj.append(st)
                proba = policy_feature_proba(traj)
                actions.append([round(float(v), 3) for v in proba])
                if decide_action(proba, tau) == "reinterpret":
                    final_pred, _ = invoke_slow()
                    source = "slow"
                else:
                    final_pred = traj[-1]["pred"]
                    source = "full_commit"
            else:
                # True streaming: decide at every arriving prefix, commit early
                # when confident, wait for more evidence otherwise.
                for prefix_sec, is_full in prefixes:
                    a, frame = load_prefix_assets(prefix_sec, is_full)
                    pmsg = fast_prompt(dialogue, prefix_sec, duration, args.include_dialogue)
                    st = fast_score(fast_model, fast_proc, a, pmsg, dev, frame=frame)
                    st["prefix_sec"] = prefix_sec
                    traj.append(st)
                    proba = policy_feature_proba(traj)
                    action = decide_action(proba, tau)
                    actions.append([round(float(v), 3) for v in proba])
                    if action == "reinterpret":
                        final_pred, _ = invoke_slow()
                        source = "slow"
                        break
                    if action == "commit":
                        final_pred = st["pred"]
                        source = "fast"
                        break
                if final_pred is None:
                    final_pred = traj[-1]["pred"]
                    source = "full_commit"

            loc = str(sample.get("local_summary") or "").strip() or str(sample.get("prev_summary") or "").strip()
            summ[si] = loc

            out.append({
                "id": rec_id, "gold": gold, "final_emotion": final_pred, "source": source,
                "video_length_sec": round(float(duration), 4),
                "action_scores": actions,
                "fast_pred": traj[-1]["pred"],
            })
            done += 1
            if done % 10 == 0 or done == n_total:
                print(f"[{done}/{n_total}] {rec_id} -> {final_pred} ({source})", flush=True)

    n = len(out)
    acc = sum(1 for r in out if r["final_emotion"] == r["gold"]) / n if n else 0.0
    rr = sum(1 for r in out if r["source"] == "slow") / n if n else 0.0
    summary = {"utterances": n, "accuracy": round(acc, 4), "slow_call_rate": round(rr, 4),
               "commit_tau": tau, "policy_bundle": str(args.policy_bundle)}

    out_path = Path(args.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with (out_path.parent / (out_path.stem + "_summary.json")).open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
