# Streaming SER Open Artifact

This directory is an anonymized, self-contained release layout for the streaming
speech/video emotion recognition system. It keeps the paper-facing components
together and removes local absolute paths, private experiment names, cached
checkpoints, and dataset-specific scratch scripts from the original workspace.

## Structure

```text
open/
  configs/                  Example configs with placeholder paths
  examples/                 Tiny JSONL schema example
  scripts/                  CLI entry points
  src/stream_ser_open/      Reusable implementation
  training/                 Fast-axis and slow-axis LoRA training code
```

## Main Pipeline

The online inference path is:

1. Fast axis scores all emotion labels for the current streaming clip.
2. Dual-head trigger predicts fast-axis risk and expected slow-axis gain.
3. Slow axis is invoked only when triggered.
4. Final emotion and optional summary update the per-stream memory.

Run:

```bash
PYTHONPATH=open/src python open/scripts/run_streaming_inference.py \
  --input-jsonl /path/to/test.jsonl \
  --output-jsonl /path/to/predictions.jsonl \
  --fast-model-path /path/to/qwen-omni \
  --fast-adapter-path /path/to/fast-lora \
  --slow-model-path /path/to/qwen-omni \
  --slow-adapter-path /path/to/slow-lora \
  --trigger-model-path /path/to/dual_head_trigger.joblib
```

If `--trigger-model-path` is a `dual_head_trigger.joblib`, inference uses the
paper-style two-head trigger:

- risk head: estimates `P(fast axis is wrong)`.
- gain head: estimates the expected improvement from invoking the slow axis.

By default, the slow axis is invoked when both risk and gain pass the saved
thresholds. Dual-head bundles can also include a linear decision layer over
`[risk, gain]`; select it with `--trigger-decision-mode linear`, or use the
bundle default with `--trigger-decision-mode auto`.
If `--trigger-model-path` is omitted, a rule trigger is used:
low confidence, low margin, high entropy, or unstable recent labels invoke the
slow axis.

## Training

### Fast Axis

The fast axis is trained with prefix audio/frame supervision:

```bash
bash open/scripts/train_fast_axis.sh \
  --model-path /path/to/qwen-omni \
  --train-jsonl /path/to/train.jsonl \
  --val-jsonl /path/to/val.jsonl \
  --output-dir /path/to/outputs/fast_axis
```

See `configs/train_fast_axis.example.sh` for the full command template.

### Slow Axis

The slow axis trains a context-aware multimodal LoRA model. The default open
configuration uses previous/current/next context when available:

```bash
bash open/scripts/train_slow_axis.sh \
  --model-path /path/to/qwen-omni \
  --train-states-jsonl /path/to/train_states.jsonl \
  --val-states-jsonl /path/to/val_states.jsonl \
  --test-states-jsonl /path/to/test_states.jsonl \
  --output-dir /path/to/outputs/slow_axis
```

See `configs/train_slow_axis.example.sh`.

### Dual-Head Trigger

Train the risk/gain trigger from fast prefix trajectories and slow-axis outputs:

```bash
PYTHONPATH=open/src python open/scripts/train_dual_head_trigger.py \
  --train-trajectory-jsonl /path/to/train_trajectory.jsonl \
  --train-slow-jsonl /path/to/train_slow.jsonl \
  --eval-trajectory-jsonl /path/to/dev_trajectory.jsonl \
  --eval-slow-jsonl /path/to/dev_slow.jsonl \
  --output-dir /path/to/dual_head_trigger_out \
  --decision-mode threshold
```

The saved `dual_head_trigger.joblib` can be passed to
`run_streaming_inference.py`. A single-head baseline router is still available
as `open/scripts/train_trigger_router.py`.

## Input JSONL Schema

Required:

- `video_path`: path to the current clip or prefix video.

Recommended:

- `stream_id`: conversation or speaker stream id.
- `step` or `start_time`: temporal order inside a stream.
- `dialogue` or `text`: transcript for the current clip.
- `prev_summary`: previous local or running context.
- `gt_emotion`, `emotion`, or `target`: optional label for evaluation.

See `examples/sample_input.jsonl`.

## Dependencies

Install project dependencies in your own environment:

```bash
pip install -e open
```

Qwen2.5-Omni also requires the matching `qwen_omni_utils` package/module from
the official model release environment.
