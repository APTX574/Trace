import argparse
import json
import os
import shutil
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.distributed as dist
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import (
    Qwen2_5OmniForConditionalGeneration,
    Qwen2_5OmniProcessor,
    TrainerCallback,
    Trainer,
    TrainingArguments,
)

from dataset_io import load_jsonl as load_jsonl_rows

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=FutureWarning, module="librosa")
warnings.filterwarnings("ignore", message="PySoundFile failed. Trying audioread instead.")

try:
    from qwen_omni_utils import process_mm_info
except ImportError:
    raise ImportError("qwen_omni_utils not found. Please ensure it is in PYTHONPATH.")


SYSTEM_PROMPT = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
    "capable of perceiving auditory and visual inputs, as well as generating text and speech."
)


def str2bool(v: str) -> bool:
    if isinstance(v, bool):
        return v
    x = str(v).strip().lower()
    if x in {"1", "true", "yes", "y", "on"}:
        return True
    if x in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {v}")


def parse_csv(v: str) -> List[str]:
    return [x.strip() for x in v.split(",") if x.strip()]


def load_label_map(label_map_path: Optional[str]) -> Dict[str, str]:
    if not label_map_path:
        return {}
    with open(label_map_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    out = {}
    for k, val in raw.items():
        out[str(k).strip()] = str(val).strip().lower()
        out[str(k).strip().lower()] = str(val).strip().lower()
    return out


def extract_raw_label(sample: Dict[str, Any], label_keys: List[str], default_label: str) -> str:
    for key in label_keys:
        value = sample.get(key, None)
        if value is not None and str(value).strip() != "":
            return str(value).strip()
    return default_label


def normalize_label(
    raw_label: str,
    valid_labels: List[str],
    label_map: Dict[str, str],
    default_label: str,
) -> str:
    x = str(raw_label).strip()
    x_low = x.lower()
    valid_set = set(valid_labels)

    if x_low in valid_set:
        return x_low
    if x_low=='happy':
        return 'joy'
    if x in label_map and label_map[x] in valid_set:
        return label_map[x]
    if x_low in label_map and label_map[x_low] in valid_set:
        return label_map[x_low]
    return default_label


def canonicalize_prediction(pred_text: str, valid_labels: List[str]) -> str:
    cleaned = "".join(ch for ch in pred_text.strip().lower() if ch.isalnum())

    # Keep aliases conservative and always map into valid label space.
    alias_map = {
    "joy": "joy",
    "anxiety": "anxiety",
    "anger": "anger",
    "sadness": "sadness",
    "neutral": "neutral",
    "surprise": "surprise",
    "embarrassment": "embarrassment",
    "disgust": "anger",
    "affection":"joy",

    "happiness": "joy",
    "happy": "joy",

    "awkwardness": "embarrassment",
    "defiance": "anger",
    
    }

    if cleaned in valid_labels:
        return cleaned
    if cleaned in alias_map:
        mapped = alias_map[cleaned]
        if mapped in valid_labels:
            return mapped
    return cleaned


def _pick_first_text(candidates: List[Any]) -> str:
    for item in candidates:
        if item is None:
            continue
        text = str(item).strip()
        if text:
            return text
    return ""


def _format_recent_values(values: List[Any], precision: int = 3) -> str:
    out = []
    for value in values:
        if isinstance(value, (int, float)):
            out.append(f"{float(value):.{precision}f}")
        else:
            out.append(str(value))
    return ", ".join(out)


def build_prompt(
    summary_or_sample: Any,
    valid_labels: List[str],
    include_summary: bool,
    summary_key: str = "prev_summary",
) -> str:
    labels_str = ", ".join(valid_labels)
    base = [
        "You are the fast belief tracker in streaming emotion understanding.",
        "Infer the character's CURRENT observable emotion from the local clip.",
        "Use the current clip as the primary evidence. Previous summaries may provide context but may not reflect the current emotion.",
        "Focus on observable cues such as hesitation, awkward pauses, nervous tone, excitement, surprise, or calm speech.",
        f"Reply with exactly one emotion label from: {labels_str}.",
    ]

    # if not isinstance(summary_or_sample, dict):
    #     summary = str(summary_or_sample).strip()
    #     if include_summary and summary:
    #         base.insert(2, f'Latest running summary: "{summary}".')
    #     return " ".join(base)

    sample = summary_or_sample
    running_summary = _pick_first_text(
        [
            sample.get("latest_running_summary", None),
            sample.get("history_running_summary_before", None),
            sample.get("running_memory_summary", None),
            sample.get(summary_key, None),
            sample.get("prev_summary", None),
        ]
    )
    latest_corrected_label = _pick_first_text(
        [
            sample.get("latest_corrected_label", None),
            sample.get("prev_corrected_label", None),
            sample.get("corrected_emotion", None),
        ]
    )
    dialogue = _pick_first_text(
        [
            sample.get("dialogue", None),
            sample.get("text", None),
        ]
    )

    traj_last_k = sample.get("traj_last_k", []) or []
    conf_last_k = sample.get("conf_last_k", []) or []
    entropy_last_k = sample.get("entropy_last_k", []) or []
    margin_last_k = sample.get("margin_last_k", []) or []
    summary_last_k = sample.get("summary_last_k", []) or sample.get("history_summaries_before", []) or []

    # if include_summary and running_summary:
    #     base.append(f'The previous summary was: {running_summary.split("|")[-1]}')
    # if latest_corrected_label:
    #     base.insert(3, f"Latest corrected label from slow track: {latest_corrected_label}.")
    if dialogue:
        base.append(f'Current dialogue transcript: "{dialogue}".')
    # if traj_last_k:
    #     base.append(f"Recent emotion trajectory(last-k): [{_format_recent_values(traj_last_k, precision=0)}].")
    # if conf_last_k or entropy_last_k or margin_last_k:
    #     parts = []
    #     if conf_last_k:
    #         parts.append(f"confidence [{_format_recent_values(conf_last_k)}]")
    #     if entropy_last_k:
    #         parts.append(f"entropy [{_format_recent_values(entropy_last_k)}]")
    #     if margin_last_k:
    #         parts.append(f"margin [{_format_recent_values(margin_last_k)}]")
    #     base.append("Recent certainty trend(last-k): " + "; ".join(parts) + ".")
    # if summary_last_k:
    #     recent_summaries = [str(x).strip() for x in summary_last_k if str(x).strip()]
    #     if recent_summaries:
    #         base.append(f'Recent local summaries(last-k): "{ " | ".join(recent_summaries) }".')
    # print(" ".join(base))
    base.append("The character's emotions in this clip are:")
    prompt= "\n".join(base)
    # print(prompt)
    return prompt

class JsonlDataset(Dataset):
    def __init__(self, jsonl_path: str):
        data = load_jsonl_rows(jsonl_path)
        self.samples=[]
        for i, sample in enumerate(data):
            video_path = sample.get("video_path", None)
            # 如果文件存在
            if video_path and os.path.exists(video_path):
                self.samples.append(sample)
            else:
                print(f"Video not found: {video_path}")
            
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx]


class FastTrackCollator:
    def __init__(
        self,
        processor: Qwen2_5OmniProcessor,
        valid_labels: List[str],
        label_map: Dict[str, str],
        label_keys: List[str],
        video_key: str,
        summary_key: str,
        include_summary: bool,
        default_label: str,
        fps: int,
        max_pixels: int,
        use_audio_in_video: bool,
    ):
        self.processor = processor
        self.valid_labels = valid_labels
        self.label_map = label_map
        self.label_keys = label_keys
        self.video_key = video_key
        self.summary_key = summary_key
        self.include_summary = include_summary
        self.default_label = default_label
        self.fps = fps
        self.max_pixels = max_pixels
        self.use_audio_in_video = use_audio_in_video
        self.ignore_index = -100

    def _resolve_video_path(self, sample: Dict[str, Any]) -> str:
        p = sample.get(self.video_key, None)
        if p is None and self.video_key != "video_path":
            p = sample.get("video_path", None)
        if p is None:
            raise KeyError(f"Video path missing. Expected key '{self.video_key}' or 'video_path'.")
        return str(p)

    def _normalize_target(self, sample: Dict[str, Any]) -> str:
        raw = extract_raw_label(sample, self.label_keys, self.default_label)
        raw=canonicalize_prediction(raw,["neutral","anger","anxiety","sadness","joy","surprise","embarrassment"])
        return normalize_label(raw, self.valid_labels, self.label_map, self.default_label)

    def __call__(self, examples):
        full_texts, prompt_texts = [], []
        audios_list, videos_list = [], []
        
        # -----------------------------
        # 1. 构造 full / prompt 文本
        # -----------------------------
        for sample in examples:
            video_path = self._resolve_video_path(sample)
            target = self._normalize_target(sample)

            prompt_text = build_prompt(
                sample,
                self.valid_labels,
                self.include_summary,
                summary_key=self.summary_key,
            )
            # print(target)
            full_conv = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": SYSTEM_PROMPT}],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {
                            "type": "video",
                            "video": video_path,
                            "fps": self.fps,
                            "max_pixels": self.max_pixels,
                        },
                    ],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": target}],
                },
            ]

            prompt_conv = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": SYSTEM_PROMPT}],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {
                            "type": "video",
                            "video": video_path,
                            "fps": self.fps,
                            "max_pixels": self.max_pixels,
                        },
                    ],
                },
            ]

            # 从 full_conv 提取多模态信息；prompt 和 full 共用同一份 audio/video
            audios, _, videos = process_mm_info(
                full_conv,
                use_audio_in_video=self.use_audio_in_video
            )

            audios_list.append(audios[0] if audios else None)
            videos_list.append(videos[0] if videos else None)

            full_text = self.processor.apply_chat_template(
                full_conv,
                tokenize=False
            )
            prompt_only = self.processor.apply_chat_template(
                prompt_conv,
                add_generation_prompt=True,
                tokenize=False,
            )

            full_texts.append(full_text)
            prompt_texts.append(prompt_only)

        # -----------------------------
        # 2. 编码 full batch
        # -----------------------------
        batch = self.processor(
            text=full_texts,
            audio=audios_list,
            videos=videos_list,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=self.use_audio_in_video,
        )

        # -----------------------------
        # 3. 编码 prompt-only batch
        # -----------------------------
        batch_prompt = self.processor(
            text=prompt_texts,
            audio=audios_list,
            videos=videos_list,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=self.use_audio_in_video,
        )

        input_ids = batch["input_ids"]                     # [B, L]
        attention_mask = batch["attention_mask"]           # [B, L]
        prompt_input_ids = batch_prompt["input_ids"]       # [B, Lp]
        prompt_attention_mask = batch_prompt["attention_mask"]

        bsz, seq_len = input_ids.shape
        labels = input_ids.clone()

        # -----------------------------
        # 4. 先 mask 掉 full batch 的 padding
        # -----------------------------
        labels[attention_mask == 0] = self.ignore_index

        # 每个样本的真实长度
        full_lens = attention_mask.sum(dim=1)             # [B]
        prompt_lens = prompt_attention_mask.sum(dim=1)    # [B]

        # -----------------------------
        # 5. 按左 padding 方式 mask 掉 prompt 部分
        # -----------------------------
        for i in range(bsz):
            full_len = int(full_lens[i].item())
            prompt_len = int(prompt_lens[i].item())

            if prompt_len > full_len:
                raise ValueError(
                    f"Sample {i}: prompt_len ({prompt_len}) > full_len ({full_len})"
                )

            if prompt_len == full_len:
                raise ValueError(
                    f"Sample {i}: prompt_len == full_len, no supervised answer tokens found."
                )

            # 由于 processor 当前看起来是左 padding：
            # [PAD ... PAD][VALID TOKENS ...]
            # full_len 是有效 token 总长度
            # pad_len 是左侧 pad 的长度
            pad_len = seq_len - full_len

            prompt_start = pad_len
            prompt_end = pad_len + prompt_len

            # 安全检查：prompt 的有效部分是否等于 full 的对应前缀
            full_prefix = input_ids[i, prompt_start:prompt_end]
            prompt_valid = prompt_input_ids[i, -prompt_len:]  # prompt batch 也是左 padding，取最后 prompt_len 个有效 token

            if not torch.equal(full_prefix, prompt_valid):
                print(f"\n[Prefix mismatch] sample={i}")
                print(f"full_len={full_len}, prompt_len={prompt_len}, pad_len={pad_len}")

                try:
                    print("---- full prefix decode ----")
                    print(self.processor.tokenizer.decode(full_prefix.tolist()))
                except Exception:
                    print("decode full_prefix failed")

                try:
                    print("---- prompt valid decode ----")
                    print(self.processor.tokenizer.decode(prompt_valid.tolist()))
                except Exception:
                    print("decode prompt_valid failed")

                print("full_prefix ids:", full_prefix.tolist()[:200])
                print("prompt_valid ids:", prompt_valid.tolist()[:200])

                raise ValueError("Prompt valid tokens are not equal to full prefix tokens.")

            # mask 掉 prompt 部分，不参与 loss
            labels[i, prompt_start:prompt_end] = self.ignore_index

        # -----------------------------
        # 6. 可选 debug：看监督区是否只剩答案
        # -----------------------------
        if getattr(self, "debug_labels", False):
            for i in range(min(2, bsz)):
                supervised_ids = labels[i][labels[i] != self.ignore_index].tolist()
                print(f"\n[DEBUG] sample {i} supervised text:")
                try:
                    print(repr(self.processor.tokenizer.decode(supervised_ids)))
                except Exception:
                    print(supervised_ids[:200])

        batch["labels"] = labels
        return batch

class FastTrackTrainer(Trainer):
    def __init__(
        self,
        *args,
        processor: Qwen2_5OmniProcessor,
        valid_labels: List[str],
        label_map: Dict[str, str],
        label_keys: List[str],
        video_key: str,
        summary_key: str,
        include_summary: bool,
        default_label: str,
        fps: int,
        max_pixels: int,
        use_audio_in_video: bool,
        eval_records_dir: str = "runs/fast_full",
        eval_records_max_samples: int = 0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.processor = processor
        self.valid_labels = valid_labels
        self.label_map = label_map
        self.label_keys = label_keys
        self.video_key = video_key
        self.summary_key = summary_key
        self.include_summary = include_summary
        self.default_label = default_label
        self.fps = fps
        self.max_pixels = max_pixels
        self.use_audio_in_video = use_audio_in_video
        self.eval_records_dir = str(eval_records_dir or "runs/fast_full").strip()
        self.eval_records_max_samples = max(0, int(eval_records_max_samples))

    def _dist_rank_world_size(self) -> tuple[int, int]:
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank(), dist.get_world_size()
        return 0, 1

    def _gather_eval_rows(self, local_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        rank, world_size = self._dist_rank_world_size()
        if world_size == 1:
            return local_rows

        gathered: List[Optional[List[Dict[str, Any]]]] = [None for _ in range(world_size)]
        dist.all_gather_object(gathered, local_rows)

        merged: List[Dict[str, Any]] = []
        for rows in gathered:
            if rows:
                merged.extend(rows)
        if rank == 0:
            merged.sort(key=lambda row: int(row.get("eval_index", -1)))
        return merged

    def _resolve_video_path(self, sample: Dict[str, Any]) -> str:
        p = sample.get(self.video_key, None)
        if p is None and self.video_key != "video_path":
            p = sample.get("video_path", None)
        if p is None:
            raise KeyError(f"Video path missing. Expected key '{self.video_key}' or 'video_path'.")
        return str(p)

    def _normalize_target(self, sample: Dict[str, Any]) -> str:
        raw = extract_raw_label(sample, self.label_keys, self.default_label)
        return normalize_label(raw, self.valid_labels, self.label_map, self.default_label)

    def _resolve_sample_id(self, sample: Dict[str, Any], fallback_idx: int) -> str:
        for key in ["id", "sample_id", "uid", "segment_id", "instance_id"]:
            value = sample.get(key, None)
            if value is not None and str(value).strip():
                return str(value).strip()

        clip_id = str(sample.get("clip_id", "")).strip()
        speaker = str(sample.get("speaker", "")).strip()
        step = sample.get("step", sample.get("speaker_turn_index", None))
        if clip_id and speaker and step is not None and str(step).strip():
            return f"{clip_id}::{speaker}::{step}"
        if clip_id:
            return clip_id
        return str(fallback_idx)

    def _maybe_save_eval_records(
        self,
        rows: List[Dict[str, Any]],
        metric_key_prefix: str,
    ) -> None:
        if not self.eval_records_dir:
            return
        rank, _ = self._dist_rank_world_size()
        if rank != 0:
            return

        out_dir = Path(self.eval_records_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        epoch = self.state.epoch
        epoch_tag = "na" if epoch is None else f"{float(epoch):.2f}".replace(".", "p")
        out_path = out_dir / f"{metric_key_prefix}_step{int(self.state.global_step)}_epoch{epoch_tag}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"[EvalDump] Saved {len(rows)} records to {out_path}")

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        batch_size = self.args.per_device_eval_batch_size
        device = self.args.device
        rank, world_size = self._dist_rank_world_size()

        self.model.eval()
        old_padding_side = self.processor.tokenizer.padding_side
        self.processor.tokenizer.padding_side = "left"

        total_count = len(eval_dataset)
        eval_indices = list(range(rank, total_count, world_size))
        local_count = len(eval_indices)
        correct_count = 0
        eval_rows: List[Dict[str, Any]] = []
        model_to_gen = self.accelerator.unwrap_model(self.model)

        try:
            with torch.no_grad():
                disable_tqdm = rank > 0
                for start in tqdm(
                    range(0, local_count, batch_size),
                    desc="Fast-track generative eval",
                    disable=disable_tqdm,
                ):
                    batch_indices = eval_indices[start:start + batch_size]
                    batch_samples = [eval_dataset[idx] for idx in batch_indices]

                    batch_texts, batch_audios, batch_videos, true_labels = [], [], [], []
                    batch_prompts, batch_ids, batch_video_paths = [], [], []
                    for sample_idx, sample in zip(batch_indices, batch_samples):
                        true_labels.append(self._normalize_target(sample))

                        video_path = self._resolve_video_path(sample)
                        prompt_text = build_prompt(
                            sample,
                            self.valid_labels,
                            self.include_summary,
                            summary_key=self.summary_key,
                        )

                        conv = [
                            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": prompt_text},
                                    {
                                        "type": "video",
                                        "video": video_path,
                                        "fps": self.fps,
                                        "max_pixels": self.max_pixels,
                                    },
                                ],
                            },
                        ]

                        audios, _, videos = process_mm_info(conv, use_audio_in_video=self.use_audio_in_video)
                        batch_audios.append(audios[0] if audios else None)
                        batch_videos.append(videos[0] if videos else None)
                        batch_video_paths.append(video_path)
                        batch_ids.append(self._resolve_sample_id(sample, sample_idx))
                        batch_prompts.append(prompt_text)

                        text = self.processor.apply_chat_template(
                            conv,
                            add_generation_prompt=True,
                            tokenize=False,
                        )
                        batch_texts.append(text)

                    inputs = self.processor(
                        text=batch_texts,
                        audio=batch_audios,
                        videos=batch_videos,
                        return_tensors="pt",
                        padding=True,
                        use_audio_in_video=self.use_audio_in_video,
                    ).to(device)

                    generated_ids = model_to_gen.generate(
                        **inputs,
                        max_new_tokens=8,
                        temperature=0.1,
                        pad_token_id=self.processor.tokenizer.pad_token_id,
                        eos_token_id=self.processor.tokenizer.eos_token_id,
                        use_audio_in_video=self.use_audio_in_video,
                    )

                    generated_trim = [
                        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                    ]
                    preds = self.processor.batch_decode(
                        generated_trim,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )

                    for pred_text, true_label in zip(preds, true_labels):
                        pred_clean = canonicalize_prediction(pred_text, self.valid_labels)
                        true_clean = canonicalize_prediction(true_label, self.valid_labels)
                        correct_count += int(pred_clean == true_clean)

                    if self.eval_records_dir:
                        for k, (pred_text, true_label) in enumerate(zip(preds, true_labels)):
                            pred_clean = canonicalize_prediction(pred_text, self.valid_labels)
                            true_clean = canonicalize_prediction(true_label, self.valid_labels)
                            eval_rows.append(
                                {
                                    "id": batch_ids[k],
                                    "eval_index": int(batch_indices[k]),
                                    "video_path": batch_video_paths[k],
                                    "prompt": batch_prompts[k],
                                    "output": pred_text,
                                    "output_label": pred_clean,
                                    "target": true_label,
                                    "target_label": true_clean,
                                    "match": int(pred_clean == true_clean),
                                }
                            )
        finally:
            self.processor.tokenizer.padding_side = old_padding_side
            self.model.train()

        if world_size > 1:
            stats = torch.tensor([correct_count, local_count], device=device, dtype=torch.long)
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
            correct_total = int(stats[0].item())
            total_eval = int(stats[1].item())
        else:
            correct_total = correct_count
            total_eval = local_count

        accuracy = correct_total / total_eval if total_eval > 0 else 0.0
        metrics = {f"{metric_key_prefix}_accuracy": accuracy}
        self.log(metrics)
        self.control = self.callback_handler.on_evaluate(
            self.args,
            self.state,
            self.control,
            metrics,
        )
        if self.eval_records_dir:
            eval_rows = self._gather_eval_rows(eval_rows)
            if self.eval_records_max_samples > 0:
                eval_rows = eval_rows[:self.eval_records_max_samples]
            self._maybe_save_eval_records(eval_rows, metric_key_prefix)
        return metrics


def unwrap_model_for_saving(model):
    unwrapped = model
    while hasattr(unwrapped, "module"):
        unwrapped = unwrapped.module
    return unwrapped


def save_lora_adapter_bundle(
    model,
    processor: Qwen2_5OmniProcessor,
    save_dir: Path,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    save_dir = Path(save_dir)
    if save_dir.exists():
        shutil.rmtree(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    peft_model = unwrap_model_for_saving(model)
    peft_model.save_pretrained(str(save_dir))
    processor.save_pretrained(str(save_dir))
    if metadata is not None:
        with open(save_dir / "best_adapter_meta.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)


class BestEvalLoraCallback(TrainerCallback):
    def __init__(
        self,
        processor: Qwen2_5OmniProcessor,
        save_dir: str,
        metric_name: str = "eval_accuracy",
        greater_is_better: bool = True,
    ):
        self.processor = processor
        self.save_dir = Path(save_dir)
        self.metric_name = metric_name
        self.greater_is_better = greater_is_better
        self.best_metric: Optional[float] = None
        self.best_step: Optional[int] = None
        self.best_epoch: Optional[float] = None
        self.has_saved = False
        self.save_reason = "final"
        self.last_skip_reason: Optional[str] = None

    def _is_better(self, metric_value: float) -> bool:
        if self.best_metric is None:
            return True
        if self.greater_is_better:
            return metric_value > self.best_metric
        return metric_value < self.best_metric

    def _save(
        self,
        model,
        state,
        metric_value: Optional[float],
        save_reason: str,
    ) -> None:
        metadata = {
            "metric_name": self.metric_name,
            "metric_value": metric_value,
            "global_step": int(state.global_step),
            "epoch": None if state.epoch is None else float(state.epoch),
            "save_reason": save_reason,
        }
        save_lora_adapter_bundle(
            model=model,
            processor=self.processor,
            save_dir=self.save_dir,
            metadata=metadata,
        )
        self.has_saved = True
        self.save_reason = save_reason
        self.last_skip_reason = None

    def _mark_skip(self, reason: str) -> None:
        self.last_skip_reason = reason
        print(f"[BestLoRA] Skip save: {reason}")

    def on_train_begin(self, args, state, control, **kwargs):
        if state.is_world_process_zero and self.save_dir.exists():
            shutil.rmtree(self.save_dir)
        self.has_saved = False
        self.last_skip_reason = None
        return control

    def on_evaluate(self, args, state, control, metrics=None, model=None, **kwargs):
        if not state.is_world_process_zero:
            return control
        if metrics is None or self.metric_name not in metrics or model is None:
            missing = []
            if metrics is None:
                missing.append("metrics=None")
            elif self.metric_name not in metrics:
                missing.append(
                    f"metric '{self.metric_name}' missing, available={sorted(metrics.keys())}"
                )
            if model is None:
                missing.append("model=None")
            self._mark_skip(", ".join(missing))
            return control

        metric_value = float(metrics[self.metric_name])
        if not self._is_better(metric_value):
            self._mark_skip(
                f"{self.metric_name}={metric_value:.6f} did not improve best="
                f"{self.best_metric:.6f}"
            )
            return control

        self.best_metric = metric_value
        self.best_step = int(state.global_step)
        self.best_epoch = None if state.epoch is None else float(state.epoch)
        self._save(model=model, state=state, metric_value=metric_value, save_reason="best_eval")
        print(
            f"[BestLoRA] Saved improved adapter to {self.save_dir} "
            f"({self.metric_name}={metric_value:.6f}, step={self.best_step}, epoch={self.best_epoch})"
        )
        return control

    def on_train_end(self, args, state, control, model=None, **kwargs):
        if not state.is_world_process_zero or model is None:
            if state.is_world_process_zero and model is None:
                self._mark_skip("train end reached with model=None")
            return control
        if self.best_metric is not None and self.save_dir.exists():
            print(
                f"[BestLoRA] Keep existing best adapter at {self.save_dir} "
                f"({self.metric_name}={self.best_metric:.6f}, step={self.best_step}, epoch={self.best_epoch})"
            )
            return control
        self._save(model=model, state=state, metric_value=self.best_metric, save_reason="final_no_eval")
        print(f"[BestLoRA] Saved final adapter to {self.save_dir} (no eval-best available)")
        return control


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Qwen Omni fast-track emotion model with LoRA.")
    parser.add_argument("--local_rank", "--local-rank", type=int, default=-1)
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--train-jsonl", type=str, required=True)
    parser.add_argument("--val-jsonl", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)

    parser.add_argument(
        "--labels",
        type=str,
        default="neutral,anger,anxiety,sadness,joy,surprise",
        help="Comma-separated valid label set.",
    )
    parser.add_argument("--default-label", type=str, default="neutral")
    parser.add_argument("--label-map-json", type=str, default="")
    parser.add_argument(
        "--label-keys",
        type=str,
        default="label",
        help="Comma-separated label candidate keys in priority order.",
    )
    parser.add_argument("--video-key", type=str, default="video_path")
    parser.add_argument("--summary-key", type=str, default="prev_summary")
    parser.add_argument("--include-summary", type=str2bool, default=True)

    parser.add_argument("--use-audio-in-video", type=str2bool, default=True)
    parser.add_argument("--fps", type=int, default=2)
    parser.add_argument("--max-pixels", type=int, default=224 * 126)

    parser.add_argument("--num-train-epochs", type=int, default=10)
    parser.add_argument("--per-device-train-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--dataloader-num-workers", type=int, default=4)

    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--deepspeed-config", type=str, default="")
    parser.add_argument("--run-name", type=str, default="qwen-omni-fast-track-lora")

    parser.add_argument("--wandb-project", type=str, default="")
    parser.add_argument("--wandb-api-key", type=str, default="")
    parser.add_argument(
        "--eval-records-dir",
        type=str,
        default="",
        help="If set, dump per-sample validation outputs to JSONL for each eval run.",
    )
    parser.add_argument(
        "--eval-records-max-samples",
        type=int,
        default=0,
        help="Optional cap for dumped eval samples per evaluation (0 = all).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.wandb_project:
        os.environ["WANDB_PROJECT"] = args.wandb_project
    if args.wandb_api_key:
        os.environ["WANDB_API_KEY"] = args.wandb_api_key

    valid_labels = [x.lower() for x in parse_csv(args.labels)]
    if args.default_label.lower() not in valid_labels:
        raise ValueError("default-label must be in --labels")
    label_map = load_label_map(args.label_map_json)
    label_keys = parse_csv(args.label_keys)

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if local_rank == 0:
        print("Loading processor/model...")

    processor = Qwen2_5OmniProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )
    model = model.thinker
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})


    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )
    model = get_peft_model(model, lora_config)

    if local_rank == 0:
        model.print_trainable_parameters()
        print("Loading datasets...")

    train_dataset = JsonlDataset(args.train_jsonl)
    val_dataset = JsonlDataset(args.val_jsonl)

    collator = FastTrackCollator(
        processor=processor,
        valid_labels=valid_labels,
        label_map=label_map,
        label_keys=label_keys,
        video_key=args.video_key,
        summary_key=args.summary_key,
        include_summary=args.include_summary,
        default_label=args.default_label.lower(),
        fps=args.fps,
        max_pixels=args.max_pixels,
        use_audio_in_video=args.use_audio_in_video,
    )

    training_kwargs = dict(
        output_dir=str(output_dir),
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        max_grad_norm=args.max_grad_norm,
        logging_steps=args.logging_steps,
        save_strategy="no",
        load_best_model_at_end=False,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        remove_unused_columns=False,
        report_to="wandb" if args.wandb_project else "none",
        run_name=args.run_name,
        dataloader_num_workers=args.dataloader_num_workers,
        dataloader_pin_memory=True,
        ddp_find_unused_parameters=False,
    )

    init_vars = TrainingArguments.__init__.__code__.co_varnames
    if "eval_strategy" in init_vars:
        training_kwargs["eval_strategy"] = "epoch"
    else:
        training_kwargs["evaluation_strategy"] = "epoch"

    if args.deepspeed_config:
        training_kwargs["deepspeed"] = args.deepspeed_config

    training_args = TrainingArguments(**training_kwargs)

    best_lora_callback = BestEvalLoraCallback(
        processor=processor,
        save_dir=str(output_dir / "fast_track_lora"),
        metric_name="eval_accuracy",
        greater_is_better=True,
    )

    trainer = FastTrackTrainer(
        processor=processor,
        valid_labels=valid_labels,
        label_map=label_map,
        label_keys=label_keys,
        video_key=args.video_key,
        summary_key=args.summary_key,
        include_summary=args.include_summary,
        default_label=args.default_label.lower(),
        fps=args.fps,
        max_pixels=args.max_pixels,
        use_audio_in_video=args.use_audio_in_video,
        eval_records_dir=args.eval_records_dir,
        eval_records_max_samples=args.eval_records_max_samples,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
        callbacks=[best_lora_callback],
    )

    if local_rank == 0:
        print("Start fast-track LoRA training...")

    trainer.train()

    final_dir = output_dir / "fast_track_lora"
    if local_rank == 0:
        config_dump = {
            "labels": valid_labels,
            "default_label": args.default_label.lower(),
            "label_keys": label_keys,
            "video_key": args.video_key,
            "summary_key": args.summary_key,
            "include_summary": args.include_summary,
            "use_audio_in_video": args.use_audio_in_video,
            "fps": args.fps,
            "max_pixels": args.max_pixels,
            "label_map": label_map,
        }
        with open(output_dir / "fast_track_label_config.json", "w", encoding="utf-8") as f:
            json.dump(config_dump, f, ensure_ascii=False, indent=2)

        if final_dir.exists():
            print(
                f"Fast-track training complete. LoRA adapter is available at {final_dir} "
                f"(save_reason={best_lora_callback.save_reason}, best_metric={best_lora_callback.best_metric})"
            )
        else:
            print(
                "Fast-track training complete, but no LoRA adapter was saved. "
                f"Last skip reason: {best_lora_callback.last_skip_reason}"
            )


if __name__ == "__main__":
    print("------------")
    main()
