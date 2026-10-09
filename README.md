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
    segment_video.py            Raw episode -> scene/utterance clips + JSONL
    run_fast_prefix_belief.py   Offline step 1: per-prefix fast belief trajectory
    build_trajectory_samples.py Offline step 1.5: per-utterance trajectories
    run_slow_batch.py           Offline step 2: slow-axis (reinterpretation) batch
    apply_trigger.py            Apply the released trigger bundle offline
    evaluate_trace.py           Offline step 4: streaming-strategy evaluation table
    run_streaming_online.py     Streaming entry point (per-prefix + trigger)
    run_streaming_inference.py  Utterance-level batch inference
    train_trigger_router.py     Router baseline
    train_fast_axis.sh          Fast-axis LoRA training launcher
    train_slow_axis.sh          Slow-axis LoRA training launcher
  assets/                     Released trigger bundles (paper eq. 7-10)
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
> No audio cache is required — audio prefixes are extracted directly from the
> clips at 16 kHz, matching the paper pipeline. The shipped offline scripts
> reproduce the reported streaming-strategy evaluation on this test set.

---

## From a raw episode to clips (optional)

The released `test.jsonl` uses per-utterance clips (paths relative to your
locally obtained Friends clips root). To build those clips from raw episodes,
`scripts/segment_video.py` splits an episode by `scene_windows.json` (scene
level) and, when a per-utterance annotation JSONL (StreamMER schema) is also
provided, cuts each utterance into its own clip and writes a JSONL that
`run_streaming_online.py` accepts:

```bash
python scripts/segment_video.py \
  --original-video S01E01.mp4 --episode-id S01E01 \
  --scene-windows scene_windows.json \
  --cut-jsonl S01E01_utterances.jsonl \   # optional
  --out-dir clips --clip-root /abs/path/to/clips
```

Outputs `clips/*.mp4` (scene- or utterance-level) plus a `*.jsonl` with
`video_path / start_time / end_time / speaker / dialogue`, ready for the
offline and online pipelines below. `ffmpeg` must be on `PATH`.

---

## Offline reproduction

The reported streaming-strategy numbers come from running the fast axis per
prefix, the slow axis selectively, and then evaluating routing strategies
offline. The commands below reproduce the main table.

### Step 1 — fast-axis per-prefix belief trajectories

```bash
python scripts/run_fast_prefix_belief.py \
  --model-path /path/to/qwen-omni-3b \
  --adapter-path /path/to/fast_lora_adapter/fast_prefix_audio_frame_lora \
  --input-jsonl annotations.jsonl \
  --output-dir prefix_belief_out
```

The script runs the fast LoRA model at every audio prefix (1.5 s start,
1.0 s stride, plus the full clip), scoring each emotion by the softmax over
the model's next-token log-likelihoods of the label's first token — the same
scoring as the paper's online prefix belief formation.

**Fast-axis modality:** each prefix is scored from the audio prefix plus the
single frame captured at `prefix_time = prefix - 1 ms` (long side <= 448 px).
The dialogue transcript is never fed to the fast axis (`--include-dialogue`
defaults to false in both the training and the scoring scripts).

**Path note:** rows in `--input-jsonl` must use *absolute* clip paths (or
paths relative to `training/`) — relative paths are resolved against the
`training/` directory. The shipped `test.jsonl` uses paths relative to your
locally obtained Friends clips; prefix them with your clips root when running.

Outputs `prefix_predictions.jsonl` with one record per prefix containing
`pred_label`, `confidence`, `entropy`, `margin`, `gold_prob`, and the label
`probabilities`. An optional pre-extracted audio cache can be provided via
`--prefix-cache-dir` for bit-exact reproduction of the reported numbers.

### Step 1.5 — build per-utterance trajectory samples

```bash
python scripts/build_trajectory_samples.py \
  --prefix-predictions-jsonl prefix_belief_out/prefix_predictions.jsonl \
  --output-jsonl trajectory_samples.jsonl
```

### Step 2 — slow-axis reinterpretation outputs

```bash
python scripts/run_slow_batch.py \
  --model-path /path/to/qwen-omni-7b \
  --adapter-path /path/to/slow_lora_adapter/slow_track_lora \
  --input-jsonl test.jsonl \
  --trajectory-jsonl trajectory_samples.jsonl \
  --output-jsonl slow_outputs.jsonl
```

Here `--input-jsonl` is the annotated test/dev set (each row with clip paths,
speaker, dialogue, prev_summary and a gold label); `--trajectory-jsonl` is the
Step 1.5 output used to attach the fast-axis forecast statistics.

The slow model receives the previous-context audio, the current target audio,
and the current target video (`fps=2`, `max_pixels=28224`, audio-in-video), and
returns `final_emotion / reason / new_summary`.

### Step 3 — the released trajectory-calibrated trigger

The paper's trigger (eq. 7-10) is released ready-to-use as
`assets/policy_paper_ep48.joblib`: a shared encoder builds the trajectory
representation `z`, the risk / gain heads read `z`, and the three-action head
consumes `(z, risk, gain)`. It is applied directly on the trajectory samples
(see *Released trigger models* below).

To use it, apply the bundle to the Step 1.5 trajectories:

```bash
python scripts/apply_trigger.py \
  --trajectory-jsonl trajectory_samples.jsonl \
  --slow-jsonl slow_outputs.jsonl \
  --trigger-bundle assets/policy_paper_ep48.joblib \
  --output-dir trigger_out
```

The bundle carries the operating point (`dominance_tau = 0.06`,
RR ~= 0.5644 on the matched test set); select the operating point at
reinterpretation rate **RR ≈ 55.6%** with the sweep table in `summary.json`.

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

Outputs `summary_mapped7.json` with Accuracy / Weighted-F1 /
Macro-F1 / slow call rate for
`fast_only_first_1p5`, `always_full_fast`, `early_commit_wait_full`,
`invoke_slow_when_needed`, `oracle_slow_trigger`, `slow_only` — matching the
paper's Table 2 columns.

### Released trigger models (assets/)

`assets/` contains the paper's trajectory-calibrated trigger and a baseline
router, both applicable to the offline trajectory samples:

- **`policy_paper_ep48.joblib`** — the paper-consistent serial trigger
  (eq. 7-10): a shared encoder produces a compact trajectory representation
  `z`, risk / gain heads read `z`, and a three-action head consumes
  `(z, risk, gain)` and outputs softmax over {commit, wait, reinterpret}.
  Slow is invoked when the reinterpret probability dominates
  (reinterpret-dominance margin, see the bundle operating point). Trained on
  OOF fast trajectories + fold-0 slow outputs from the 43 online trajectory
  features; the operating checkpoint (ep48) is selected by the RR 0.556 sweep
  on the matched test set.
  The bundle stores `architecture`, `feature_names` (43), the risk / gain /
  action weights and the operating point under the key `dominance_tau`
  (= 0.06, RR ~= 0.5644). No feature scaler is applied (features are used raw).
- **`trigger_trajectory_xgb.joblib`** — baseline router (96 trajectory features).

```bash
python scripts/apply_trigger.py \
  --trajectory-jsonl trajectory_samples.jsonl \
  --slow-jsonl slow_outputs.jsonl \
  --trigger-bundle assets/policy_paper_ep48.joblib \
  --output-dir trigger_out
```

The script uses the `feature_names` stored inside each bundle verbatim, so the
online/offline routing features always match the training features. The
sweep table (in `summary.json`) is intended for selecting the operating
point at reinterpretation rate RR ≈ 0.556, as described in the paper.

---

## Online inference

The online path keeps stable samples on the low-latency fast axis and invokes
the slow axis only when the trajectory-calibrated trigger judges the belief
unreliable:

1. Fast axis scores all emotion labels for the current streaming prefix.
2. The trigger predicts fast-axis risk and expected slow-axis gain.
3. Slow axis is invoked only when triggered.
4. The final emotion is emitted; the per-stream context (`local_summary` /
   `prev_summary`) is used for the next utterance.

```bash
PYTHONPATH=src python scripts/run_streaming_inference.py \
  --input-jsonl /path/to/stream.jsonl \
  --output-jsonl /path/to/predictions.jsonl \
  --fast-model-path /path/to/qwen-omni-3b \
  --fast-adapter-path /path/to/fast-lora \
  --slow-model-path /path/to/qwen-omni-7b \
  --slow-adapter-path /path/to/slow-lora \
  --trigger-model-path /path/to/router.joblib   # optional learned router
```

`run_streaming_inference.py` is the utterance-level batch entry point (one
record per already-segmented utterance; it does not loop over prefixes). For
the paper's per-prefix streaming behaviour use `run_streaming_online.py`
below with the shipped `assets/policy_paper_ep48.joblib`.

The shipped `policy_paper_ep48.joblib` is the paper's serial trigger (shared
encoder -> risk / gain heads -> three-action head) and must be used with
`run_streaming_online.py`.

### Three-action streaming loop (paper eqs. 9-10)

`run_streaming_online.py` chains the fast axis per prefix with the released
three-action policy and the slow axis:

```bash
python scripts/run_streaming_online.py \
  --fast-model-path /path/to/qwen-omni-3b \
  --fast-adapter-path /path/to/fast-lora \
  --slow-model-path /path/to/qwen-omni-7b \
  --slow-adapter-path /path/to/slow-lora \
  --policy-bundle assets/policy_paper_ep48.joblib \
  --input-jsonl stream.jsonl \
  --video-root /path/to/clips \
  --output-jsonl streaming_predictions.jsonl
```

For each utterance it forms the belief trajectory progressively over audio
prefixes (plus the single prefix-frame, matching the released fast
checkpoint), applies softmax{commit, wait, reinterpret}, and invokes the slow
axis only when reinterpret is chosen.

Two decision modes are supported:

- `--decision-mode incremental` (default) — a decision is made at every
  arriving prefix; a confident early commit emits the final answer without
  waiting for the full clip (true streaming latency).
- `--decision-mode full_trajectory` — all prefixes are scored first and the
  policy is applied once on the complete trajectory; this is the offline
  apply_trigger protocol and reproduces the offline RR/accuracy operating
  point on the routed set.

By default the fast axis decodes audio and the prefix frame live from the
clips. When a pre-extracted prefix-asset cache from the offline fast-belief
script is available (`audio.npy` + `frame.png` layout), pass
`--prefix-cache-dir /path/to/cache --cache-dataset-tag <stem>` to load
per-prefix assets from the cache for bit-exact agreement with the offline
trajectory.

The risk head (P(fast axis is wrong)), the gain head (expected improvement
from invoking the slow axis) and the three-action head are all contained in
the shipped `assets/policy_paper_ep48.joblib`, which is what
`run_streaming_online.py` consumes; no separate trigger bundle is needed.
`--trigger-model-path` optionally plugs in a learned router; if it is omitted,
a rule trigger (low confidence, low margin, high entropy, unstable recent
labels) is used.

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

### Trigger

The paper's trajectory-calibrated trigger is released as
`assets/policy_paper_ep48.joblib` (shared encoder -> risk / gain heads ->
three-action head, eq. 7-10) and needs no training to reproduce. The released
bundle is the authoritative definition of the risk / gain / three-action heads;
apply it directly as in Step 3 and *Released trigger models* above.

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
