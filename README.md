# TRACE: Emotion Understanding in Streaming Video with Trajectory-Aware Reliability

Official open-source release for the paper

> **Emotion Understanding in Streaming Video with Trajectory-Aware Reliability**
>
> Qingsong Wang, Qigong Lei, Zitong Wang, Bohan Yu, Zhiang Dong, Jian Liu, Weiqiang Wang, Chang Yao, Jingyuan Chen
>
> **[EMNLP 2026 — accepted]**
>
> Paper: <https://arxiv.org/abs/2608.26786>

TRACE runs streaming video emotion understanding with three components working together:

- **Fast axis** — a low-latency belief tracker that scores every emotion label from the audio prefix.
- **Trajectory-calibrated trigger** — reads the prefix belief trajectory and decides whether the fast axis can be committed to.
- **Slow axis** — a contextual reinterpretation model that is invoked only when the trigger asks for it.

The repository supports two ways of running the framework:

- **Offline pipeline** — batch a JSONL of clips through the fast axis, the slow axis and the routing evaluation.
- **Online inference** — drive TRACE as a streaming loop over incoming clips.

## Repository layout

```text
Trace/
  configs/                    Example configs with placeholder paths
  examples/                   Tiny JSONL schema example
  scripts/
    run_fast_prefix_belief.py   Offline step 1: per-prefix fast belief trajectory
    run_slow_batch.py           Offline step 2: slow-axis (reinterpretation) batch
    evaluate_trace.py           Offline step 3: streaming-strategy evaluation table
    run_streaming_inference.py  Online streaming inference entry point
    train_dual_head_trigger.py  Train the trajectory-calibrated trigger
    train_trigger_router.py     Single-head router baseline (optional)
    train_fast_axis.sh          Fast-axis LoRA training launcher
    train_slow_axis.sh          Slow-axis LoRA training launcher
  src/stream_ser_open/        Reusable implementation
  training/                   Fast-axis and slow-axis LoRA training code
```

## Setup

Install the package:

```bash
pip install -e .
```

Then prepare the pieces the commands below expect:

- **Base model** — `Qwen2.5-Omni`: the fast axis loads `Qwen2.5-Omni-3B` (thinker), the slow axis loads `Qwen2.5-Omni-7B` (thinker), together with the matching `qwen_omni_utils`.
- **Checkpoints** — [Trace-checkpoints.zip](https://drive.google.com/file/d/1Onorc5Z-C754C-x6uVyyw2gOLZO3Md5y/view?usp=sharing) bundles the two released LoRA adapters: `fast_prefix_audio_frame_lora` (base `Qwen2.5-Omni-3B/thinker`) and `slow_track_lora` (base `Qwen2.5-Omni-7B/thinker`). Unzip it and point `--adapter-path` at the matching directory — the fast axis takes the first, the slow axis the second. The slow adapter is trained with previous/current context and returns JSON `final_emotion / reason / new_summary`.
- **Trigger bundle** — optional; produced by `scripts/train_dual_head_trigger.py`. Without it the online path falls back to a rule trigger.
- **Clips** — JSONL `video_path` values are resolved against `--video-root`, so clips stay wherever you keep them locally.

`test.jsonl` ships the StreamMER annotations with all stream fields filled in, and works as `--input-jsonl` for every command below.

## Offline pipeline

### Step 1 — fast-axis per-prefix beliefs

```bash
PYTHONPATH=src python scripts/run_fast_prefix_belief.py \
  --model-path /path/to/qwen-omni-3b \
  --adapter-path /path/to/Trace-checkpoints/fast_prefix_audio_frame_lora \
  --input-jsonl test.jsonl \
  --video-root /path/to/Friends/clips \
  --output-jsonl prefix_predictions.jsonl \
  --include-dialogue false
```

Audio prefixes are cut straight from the clips at 16 kHz, starting at 1.5 s with a 1.0 s stride (`--min-prefix-sec`, `--prefix-step-sec`), plus the full clip. Each record carries `pred_label`, `confidence`, `entropy`, `margin`, `gold_prob` and the per-label `probabilities`. Pass `--prefix-cache-dir` to read pre-extracted 1.5 s prefixes instead of cutting them again.

### Step 2 — slow-axis reinterpretation outputs

```bash
PYTHONPATH=src python scripts/run_slow_batch.py \
  --model-path /path/to/qwen-omni-7b \
  --adapter-path /path/to/Trace-checkpoints/slow_track_lora \
  --input-jsonl test.jsonl \
  --trajectory-jsonl prefix_predictions.jsonl \
  --video-root /path/to/Friends/clips \
  --output-jsonl slow_outputs.jsonl
```

The slow model receives the previous-context audio, the current target audio and the current target video (2 fps, resolution capped by `--max-pixels`, audio-in-video) and returns `final_emotion / reason / new_summary`. `--trajectory-jsonl` merges the fast forecast into each slow prompt.

### Step 3 — strategy evaluation

```bash
PYTHONPATH=src python scripts/evaluate_trace.py \
  --trajectory-jsonl prefix_predictions.jsonl \
  --slow-jsonl slow_outputs.jsonl \
  --output-dir eval_out
```

Writes `eval_out/strategy_delay_metrics_mapped7.csv`, one row per routing strategy (`fast_only_first_1p5`, `always_full_fast`, `early_commit_wait_full`, `invoke_slow_when_needed`, `oracle_slow_trigger`, `slow_only`) with Accuracy, Weighted-F1, Macro-F1, average decision time, normalized decision ratio and slow call rate.

The routing thresholds are all flags; the defaults are `--commit-conf 0.75`, `--commit-margin 0.50`, `--commit-entropy 0.75`, `--slow-conf 0.70`, `--slow-margin 0.40`, `--slow-entropy 0.90`, `--near-full-ratio 0.90`. Use `--target-space mapped7` (default) or `--target-space meld_raw` to pick the label space.

## Online inference

The online loop keeps stable clips on the fast axis and calls the slow axis only when the trigger judges the prefix belief unreliable:

1. The fast axis scores all emotion labels for the current streaming prefix.
2. The trigger predicts fast-axis risk and the expected gain from invoking the slow axis.
3. The slow axis runs only when it is triggered.
4. The final emotion and the new summary update the per-stream memory.

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

Trigger behaviour:

- **No `--trigger-model-path`** — a rule trigger fires on low confidence, low margin, high entropy or an unstable recent label history (`--rule-min-confidence`, `--rule-min-margin`, `--rule-max-entropy`, `--history-size`).
- **`dual_head_trigger.joblib`** — the risk head estimates P(fast axis is wrong) and the gain head estimates the expected improvement from the slow axis; both must pass their saved thresholds. `--trigger-decision-mode` chooses between `threshold` routing, the `linear` decision head over `[risk, gain]`, and `auto` (keep the bundle default).

Other knobs: `--stream-key` / `--step-key` / `--start-key` / `--summary-key` for JSONL field names, `--include-summary` and `--summary-history-size` to control the running memory, `--fps` / `--max-pixels` / `--use-audio-in-video` for the video path, and `--fast-device` / `--slow-device` for placement.

## Training (optional)

### Fast axis

The fast axis is trained with prefix audio/frame supervision:

```bash
bash scripts/train_fast_axis.sh \
  --model-path /path/to/qwen-omni-3b \
  --train-jsonl /path/to/train.jsonl \
  --val-jsonl /path/to/val.jsonl \
  --output-dir /path/to/outputs/fast_axis
```

See `configs/train_fast_axis.example.sh` for the full command; the launcher runs `training/train_fast_1s.py`, and `training/train_fast_track.py` is the track-style variant. Keep batch size 1 on the per-prefix scoring path; batched scoring is not supported.

### Slow axis

The slow axis trains a context-aware multimodal LoRA model on previous/current (and optionally next) context:

```bash
bash scripts/train_slow_axis.sh \
  --model-path /path/to/qwen-omni-7b \
  --train-states-jsonl /path/to/train_states.jsonl \
  --val-states-jsonl /path/to/val_states.jsonl \
  --test-states-jsonl /path/to/test_states.jsonl \
  --output-dir /path/to/outputs/slow_axis
```

See `configs/train_slow_axis.example.sh`; the launcher runs `training/train_slow_meld.py`, and `training/train_slow_track.py` is the track-style variant.

### Trigger

Train the risk/gain trigger from fast trajectories and slow outputs collected on the train split (the two offline scripts pointed at `train.jsonl`):

```bash
PYTHONPATH=src python scripts/train_dual_head_trigger.py \
  --train-trajectory-jsonl /path/to/train_trajectory.jsonl \
  --train-slow-jsonl /path/to/train_slow.jsonl \
  --eval-trajectory-jsonl prefix_predictions.jsonl \
  --eval-slow-jsonl slow_outputs.jsonl \
  --output-dir dual_head_trigger_out
```

`threshold_search.csv` in the output directory lists accuracy per reinterpretation rate (RR), so an operating point can be picked at a target RR. `scripts/train_trigger_router.py` is a single-head router baseline.

## Input JSONL schema

Required for evaluation:

- `video_path` — relative clip path; when `--video-root` is given the root is prepended.

Required for training / context resolution (stream info):

- `clip_id` (or `group_id`) — scene/dialogue id used to group turns.
- `step` (or `start_time`) — temporal order of the utterance inside the clip.
- `stream_id` — conversation or speaker stream id.

Recommended:

- `dialogue` or `text` — transcript for the current clip.
- `prev_summary` — previous local or running context, which may be several clauses joined by ` | ` when accumulated across a conversation.
- `gt_emotion`, `emotion` or `target` — label for evaluation.

See `examples/sample_input.jsonl`; `test.jsonl` contains all of the required stream fields.

## Notes

- `Qwen2.5-Omni` also requires the matching `qwen_omni_utils` package/module from the official model release environment.
- Original clips are copyright-protected; obtain them lawfully. For MELD / MER2024, use the official public data.
