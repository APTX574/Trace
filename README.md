# TRACE: Emotion Understanding in Streaming Video with Trajectory-Aware Reliability

Official open-source release for the paper

> **Emotion Understanding in Streaming Video with Trajectory-Aware Reliability**
>
> Qingsong Wang, Qigong Lei, Zitong Wang, Bohan Yu, Zhiang Dong, Jian Liu, Weiqiang Wang, Chang Yao, Jingyuan Chen
>
> **[EMNLP 2026 — accepted]**
>
> Paper: <https://arxiv.org/abs/2608.26786>

This repository provides the streaming **TRA**jectory-aware **C**ontrastive-r**E**liability
(TRACE) framework for streaming video emotion understanding: a low-latency audio-prefix
belief tracker, a trajectory-calibrated reliability trigger, and a selectively invoked
contextual reinterpretation (slow axis).

There are two supported lines of use:

- **Offline reproduction** — regenerate the paper's main tables from the released
  checkpoints (fast LoRA, slow LoRA) and the released annotations.
- **Online inference** — run TRACE on a streaming clip in real time.

---

## Structure

```text
Trace/
  configs/                    Example configs with placeholder paths
  examples/                   Tiny JSONL schema example
  scripts/
    run_fast_prefix_belief.py   Offline step 1: per-prefix fast belief trajectory
    run_slow_batch.py           Offline step 2: slow-axis (reinterpretation) batch
    evaluate_trace.py           Offline step 4: streaming-strategy evaluation table
    train_dual_head_trigger.py  Train the trajectory-calibrated trigger
    train_trigger_router.py     Single-head router baseline (optional)
    run_streaming_inference.py  Online streaming inference entry point
    train_fast_axis.sh          Fast-axis LoRA training launcher
    train_slow_axis.sh          Slow-axis LoRA training launcher
  src/stream_ser_open/        Reusable implementation
  training/                   Fast-axis and slow-axis LoRA training code
```

## What you need to reproduce

- **Base model**: `Qwen2.5-Omni` (fast axis uses `Qwen2.5-Omni-3B/thinker`,
  slow axis uses `Qwen2.5-Omni-7B/thinker`), plus the matching `qwen_omni_utils`.
- **Fast checkpoint**: the released `fast_lora_adapter.zip` LoRA bundle.
- **Slow checkpoint**: the released `slow_lora_adapter.zip` LoRA bundle
  (trained with previous/current context, JSON output
  `final_emotion / reason / new_summary`).
- **Data**: StreamMER annotations are shipped in `test.jsonl` (all paths are
  relative to your locally obtained Friends S1–S2 clips). Original videos are
  copyright-protected and must be obtained lawfully (see the paper's dataset
  appendix). For MELD / MER2024, use the official public data.

> Note: the released `test.jsonl` labels are the authoritative test set; any
> discrepancy with the table in the paper draft is a typesetting issue in the
> draft, not in the data.
>
> No audio cache is required. Audio prefixes are extracted directly from the
> clips at 16 kHz (same as the paper pipeline); the pre-extracted audio cache is
> only used when provided via `--prefix-cache-dir` for bit-exact reproduction.
> Without it, borderline-prefix predictions may flip on a few samples due to
> av container-duration rounding, while overall accuracy stays within ~2 pt of
> the reported numbers.

---

## Offline reproduction

The reported streaming-strategy numbers come from running the fast axis per
prefix, the slow axis selectively, and then evaluating routing strategies
offline. The commands below reproduce the main table.

### Step 1 — fast-axis per-prefix belief trajectories

```bash
PYTHONPATH=src python scripts/run_fast_prefix_belief.py \
  --model-path /path/to/qwen-omni-3b \
  --adapter-path /path/to/fast_lora_adapter/fast_prefix_audio_frame_lora \
  --input-jsonl test.jsonl \
  --video-root /path/to/Friends/clips \
  --output-jsonl prefix_predictions.jsonl \
  --include-dialogue false
```

This writes one record per audio prefix (1.5 s start, 1.0 s stride, plus the
full clip) with `pred_label`, `confidence`, `entropy`, `margin`, `gold_prob`,
and the label `probabilities`.

### Step 2 — slow-axis reinterpretation outputs

```bash
PYTHONPATH=src python scripts/run_slow_batch.py \
  --model-path /path/to/qwen-omni-7b \
  --adapter-path /path/to/slow_lora_adapter/slow_track_lora \
  --input-jsonl test.jsonl \
  --trajectory-jsonl prefix_predictions.jsonl \
  --video-root /path/to/Friends/clips \
  --output-jsonl slow_outputs.jsonl
```

The slow model receives the previous-context audio, the current target audio,
and the current target video (2 fps, 448 long side, audio-in-video), and
returns `final_emotion / reason / new_summary`.

### Step 3 — train the trajectory-calibrated trigger (optional)

On the StreamMER train split, generate out-of-fold fast trajectories and slow
outputs first, then

```bash
PYTHONPATH=src python scripts/train_dual_head_trigger.py \
  --train-trajectory-jsonl /path/to/train_trajectory.jsonl \
  --train-slow-jsonl /path/to/train_slow.jsonl \
  --eval-trajectory-jsonl prefix_predictions.jsonl \
  --eval-slow-jsonl slow_outputs.jsonl \
  --output-dir dual_head_trigger_out \
  --decision-mode threshold
```

Use the threshold grid to pick the operating point at reinterpretation rate
**RR ≈ 55.6%** (the paper's TRACE operating point). The training script's
`threshold_search.csv` also reports accuracy per RR.

### Step 4 — evaluation table

```bash
PYTHONPATH=src python scripts/evaluate_trace.py \
  --trajectory-jsonl prefix_predictions.jsonl \
  --slow-jsonl slow_outputs.jsonl \
  --output-dir eval_out \
  --commit-conf 0.75 --commit-margin 0.50 --commit-entropy 0.75 \
  --slow-conf 0.70 --slow-margin 0.40 --slow-entropy 0.90 \
  --near-full-ratio 0.90
```

Outputs `strategy_delay_metrics_mapped7.csv` with Accuracy / Weighted-F1 /
Macro-F1 / avg decision time / normalized decision ratio / slow call rate for
`fast_only_first_1p5`, `always_full_fast`, `early_commit_wait_full`,
`invoke_slow_when_needed`, `oracle_slow_trigger`, `slow_only` — matching the
paper's Table 2 columns.

---

## Online inference

The online path keeps stable samples on the low-latency fast axis and invokes
the slow axis only when the trajectory-calibrated trigger judges the belief
unreliable:

1. Fast axis scores all emotion labels for the current streaming prefix.
2. The trigger predicts fast-axis risk and expected slow-axis gain.
3. Slow axis is invoked only when triggered.
4. Final emotion and optional summary update the per-stream memory.

```bash
PYTHONPATH=src python scripts/run_streaming_inference.py \
  --input-jsonl /path/to/stream.jsonl \
  --output-jsonl /path/to/predictions.jsonl \
  --fast-model-path /path/to/qwen-omni-3b \
  --fast-adapter-path /path/to/fast-lora \
  --slow-model-path /path/to/qwen-omni-7b \
  --slow-adapter-path /path/to/slow-lora \
  --trigger-model-path /path/to/dual_head_trigger.joblib
```

If `--trigger-model-path` is a `dual_head_trigger.joblib`, inference uses the
paper-style two-head trigger:

- risk head estimates P(fast axis is wrong);
- gain head estimates the expected improvement from invoking the slow axis.

By default the slow axis is invoked when both risk and gain pass the saved
thresholds. Select the linear decision layer over `[risk, gain]` with
`--trigger-decision-mode linear`, or keep the bundle default with
`--trigger-decision-mode auto`. If `--trigger-model-path` is omitted, a rule
trigger (low confidence, low margin, high entropy, unstable recent labels) is
used.

---

## Training the released checkpoints

### Fast Axis

The fast axis is trained with prefix audio/frame supervision:

```bash
bash scripts/train_fast_axis.sh \
  --model-path /path/to/qwen-omni-3b \
  --train-jsonl /path/to/train.jsonl \
  --val-jsonl /path/to/val.jsonl \
  --output-dir /path/to/outputs/fast_axis
```

See `configs/train_fast_axis.example.sh` for the full command template.
Note: keep batch size 1 for the per-prefix inference/scoring path; batched
scoring is not supported.

### Slow Axis

The slow axis trains a context-aware multimodal LoRA model. The default open
configuration uses previous/current (and optionally next) context:

```bash
bash scripts/train_slow_axis.sh \
  --model-path /path/to/qwen-omni-7b \
  --train-states-jsonl /path/to/train_states.jsonl \
  --val-states-jsonl /path/to/val_states.jsonl \
  --test-states-jsonl /path/to/test_states.jsonl \
  --output-dir /path/to/outputs/slow_axis
```

See `configs/train_slow_axis.example.sh`.

### Dual-Head Trigger

Train the risk/gain trigger from fast prefix trajectories and slow-axis outputs
as described in Step 3 above. A single-head baseline router is also available
as `scripts/train_trigger_router.py`.

---

## Input JSONL Schema

Required for evaluation:

- `video_path`: relative clip path (when `--video-root` is given, the root is
  prepended).

Required for training / context resolution (stream info):

- `clip_id` (or `group_id`): scene/dialogue id used to group turns.
- `step` (or `start_time`): temporal order of the utterance inside the clip.
- `stream_id`: conversation or speaker stream id.

Recommended:

- `dialogue` or `text`: transcript for the current clip.
- `prev_summary`: previous local or running context (may be several clauses
  joined by ` | ` when accumulated across the conversation).
- `gt_emotion`, `emotion`, or `target`: label for evaluation.

See `examples/sample_input.jsonl`. The released `test.jsonl` contains all of
the required stream fields.

---

## Dependencies

```bash
pip install -e .
```

Qwen2.5-Omni also requires the matching `qwen_omni_utils` package/module from
the official model release environment.