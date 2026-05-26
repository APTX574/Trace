import argparse
import json
import os
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import (
    Qwen2_5OmniForConditionalGeneration,
    Qwen2_5OmniProcessor,
    Trainer,
    TrainingArguments,
)
import torch.distributed as dist
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
    if x in label_map and label_map[x] in valid_set:
        return label_map[x]
    if x_low in label_map and label_map[x_low] in valid_set:
        return label_map[x_low]
    # print(f"Invalid label: {x}, using default: {default_label}")
    # Alias map for common non-standard emotion names
    _label_alias = {
        'happy': 'joy',
        'angry': 'anger',
        'disgust': 'anger',
        'contempt': 'anger',
        'fear': 'anxiety',
        'affection': 'joy',
        'happiness': 'joy',
        'resignation': 'sadness',
        'guilt': 'embarrassment',
        'awkwardness': 'embarrassment',
        'defiance': 'anger',
    }
    if x_low in _label_alias and _label_alias[x_low] in valid_set:
        return _label_alias[x_low]
    return default_label


def canonicalize_prediction(pred_text: str, valid_labels: List[str]) -> str:
    cleaned = "".join(ch for ch in pred_text.strip().lower() if ch.isalnum())
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
    if cleaned in alias_map and alias_map[cleaned] in valid_labels:
        return alias_map[cleaned]
    return cleaned


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    return load_jsonl_rows(path)


def format_probs(prob_dict: Dict[str, float]) -> str:
    # if not isinstance(prob_dict, dict):
    #     return "
    # print(prob_dict)
    if 'happy' in prob_dict:
        prob_dict['joy'] = prob_dict.pop('happy')+prob_dict['joy']

    
    items = sorted(prob_dict.items(), key=lambda kv: kv[1], reverse=True)
    ss= ", ".join([f"{k}:{v:.3f}" for k, v in items])

    return ss


def resolve_fast_confidence(sample: Dict[str, Any]) -> float:
    value = sample.get("fast_confidence", None)
    if value is not None:
        try:
            return float(value)
        except Exception:
            pass

    prob_dict = sample.get("fast_probs", None)
    if isinstance(prob_dict, dict) and prob_dict:
        try:
            return float(max(float(v) for v in prob_dict.values()))
        except Exception:
            pass
    return 0.0


def resolve_video_path(sample: Dict[str, Any], video_key: str) -> Optional[str]:
    paths = resolve_video_paths(sample, video_key)
    if not paths:
        return None
    return paths[0]


def resolve_video_paths(sample: Dict[str, Any], video_key: str) -> List[str]:
    candidates: List[Any] = [
        sample.get("history_video_paths", None),
        sample.get("video_paths", None),
        sample.get(video_key, None),
    ]
    if video_key != "video_path":
        candidates.append(sample.get("video_path", None))

    original = sample.get("original", None)
    if isinstance(original, dict):
        candidates.extend(
            [
                original.get("history_video_paths", None),
                original.get("video_paths", None),
                original.get(video_key, None),
                original.get("video_path", None),
            ]
        )

    for item in candidates:
        if item is None:
            continue
        if isinstance(item, (list, tuple)):
            out = [str(x).strip() for x in item if str(x).strip()]
            if out:
                return out
        text = str(item).strip()
        if text:
            return [text]
    return []


def resolve_gt_label(
    sample: Dict[str, Any],
    label_keys: List[str],
    valid_labels: List[str],
    label_map: Dict[str, str],
    default_label: str,
) -> str:
    if sample.get("gt_emotion", None) is not None:
        return normalize_label(sample["gt_emotion"], valid_labels, label_map, default_label)

    source = sample
    if "original" in sample and isinstance(sample["original"], dict):
        source = sample["original"]

    raw = extract_raw_label(source, label_keys, default_label)
    raw=canonicalize_prediction(raw, valid_labels)

    return normalize_label(raw, valid_labels, label_map, default_label)


def resolve_need_correction(sample: Dict[str, Any], gt_label: str) -> int:
    if sample.get("need_correction", None) is not None:
        try:
            return int(sample["need_correction"])
        except Exception:
            pass
    fast_pred = str(sample.get("fast_pred", "")).strip().lower()
    return int(fast_pred != gt_label)


def resolve_delta(sample: Dict[str, Any], need_correction: int, gt_label: str) -> str:
    d = sample.get("correction_delta", None)
    if d is not None and str(d).strip() != "":
        return str(d).strip()
    if need_correction:
        return f"{sample.get('fast_pred', 'unknown')}->{gt_label}"
    return "none"


def maybe_reason_code(sample: Dict[str, Any]) -> Optional[str]:
    if sample.get("reason_code", None):
        return str(sample["reason_code"]).strip()

    original = sample.get("original", None)
    if isinstance(original, dict):
        pe = original.get("perception_evolution", None)
        if isinstance(pe, dict):
            gap_type = str(pe.get("gap_type", "")).strip()
            if gap_type and gap_type.lower() != "none":
                return gap_type
    return None


def _get_original(sample: Dict[str, Any]) -> Dict[str, Any]:
    original = sample.get("original", None)
    if isinstance(original, dict):
        return original
    return {}


def _pick_text(candidates: List[Any]) -> str:
    for item in candidates:
        if item is None:
            continue
        text = str(item).strip()
        if text:
            return text
    return ""


def resolve_audio_evidence(sample: Dict[str, Any]) -> str:
    original = _get_original(sample)
    cma = original.get("cross_modal_conflict_analysis", {})
    if not isinstance(cma, dict):
        cma = {}
    return _pick_text([sample.get("audio_evidence", None), cma.get("audio_evidence", None)])


def resolve_visual_evidence(sample: Dict[str, Any]) -> str:
    original = _get_original(sample)
    cma = original.get("cross_modal_conflict_analysis", {})
    if not isinstance(cma, dict):
        cma = {}
    return _pick_text([sample.get("visual_evidence", None), cma.get("visual_evidence", None)])


def resolve_conflict_explanation(sample: Dict[str, Any]) -> str:
    original = _get_original(sample)
    cma = original.get("cross_modal_conflict_analysis", {})
    if not isinstance(cma, dict):
        cma = {}
    return _pick_text([sample.get("conflict_explanation", None), cma.get("conflict_explanation", None)])


def resolve_reason_text(sample: Dict[str, Any]) -> str:
    original = _get_original(sample)
    pe = original.get("perception_evolution", {})
    if not isinstance(pe, dict):
        pe = {}
    cma = original.get("cross_modal_conflict_analysis", {})
    if not isinstance(cma, dict):
        cma = {}

    return _pick_text(
        [
            sample.get("reason", None),
            pe.get("reason_for_revision", None),
            cma.get("conflict_explanation", None),
        ]
    )


def resolve_new_summary(sample: Dict[str, Any]) -> str:
    original = _get_original(sample)
    return _pick_text(
        [
            sample.get("new_summary", None),
            sample.get("local_summary", None),
            sample.get("running_memory_summary", None),
            sample.get("summary_gt", None),
            original.get("event_summary_for_memory", None),
            original.get("summary_gt", None),
        ]
    )


def build_slow_prompt(
    sample: Dict[str, Any],
    valid_labels: List[str],
    input_fields: List[str],
    output_fields: List[str],
) -> str:
    labels_str = ", ".join(valid_labels)
    field_set = set(input_fields)

    original = _get_original(sample)
    basic_info = original.get("basic_info", {})
    if not isinstance(basic_info, dict):
        basic_info = {}
    speaker = _pick_text([sample.get("speaker", None), original.get("speaker", None), basic_info.get("speaker", None)])
    dialogue = _pick_text([sample.get("dialogue", None), original.get("dialogue", None), basic_info.get("dialogue", None)])

    probs_text = format_probs(sample.get("fast_probs", {}))
    fast_confidence = resolve_fast_confidence(sample)
    traj_text = ", ".join([str(x) for x in sample.get("traj_last_k", [])])
    conf_hist_text = ", ".join([f"{float(x):.3f}" for x in sample.get("conf_last_k", [])])
    summary_hist_text = " | ".join(
        [
            str(x).strip()
            for x in sample.get("summary_last_k", []) or sample.get("history_summaries_before", [])
            if str(x).strip()
        ]
    )
    prev_summary = _pick_text(
        [
            # sample.get("latest_running_summary", None),
            # sample.get("running_memory_summary", None),
            # sample.get("history_running_summary_before", None),
            sample.get("prev_summary", None),
        ]
    )
    local_summary = _pick_text([sample.get("local_summary", None)])
    audio_evidence = resolve_audio_evidence(sample)
    visual_evidence = resolve_visual_evidence(sample)
    conflict_explanation = resolve_conflict_explanation(sample)
    reason_text = resolve_reason_text(sample)
    pe = original.get("perception_evolution", {})
    if not isinstance(pe, dict):
        pe = {}

    prompt = [
        "You are the belief tracker in streaming emotion understanding.",
        "Infer the character's CURRENT observable emotion from the local clip.",
        "Use the current clip as the primary evidence. Previous summaries may provide context but may not reflect the current emotion.",
        "Focus on observable cues such as hesitation, awkward pauses, nervous tone, excitement, surprise, or calm speech.",
        f"Reply with exactly one emotion label from: {labels_str}.",
        "The prediction results from existing models are only for reference and are not definitive answers. You can choose to accept or reject them.",
        "When the confidence of the existing model is high (e.g., >= 0.6) and consistent with video evidence, the reference weight can be increased.",
        "When the confidence of the existing model is low (such as < 0.4), or when it is inconsistent with the expressions, tone, and actions in the video, priority should be given to re-evaluating based on the video.",
        "Your goal is to identify the 'most dominant and observable emotion at the current moment', without speculating on the character's long-term psychological state, nor substituting video evidence with plot common sense.",
    ]

    # if "fast_pred" in field_set:
    #     prompt += f"Fast prediction: {sample.get('fast_pred', '')}.\n"
    # if "fast_confidence" in field_set:
    #     prompt += f"Fast confidence: {fast_confidence:.4f}.\n"
    if "fast_probs" in field_set:
        prompt += f"The previous forecast results were: {probs_text}.\n"
    # if "uncertainty" in field_set:
    #     prompt += (
    #         f"Fast entropy: {float(sample.get('fast_entropy', 0.0)):.4f}.\n"
    #         f"Fast margin: {float(sample.get('fast_margin', 0.0)):.4f}.\n"
    #     )
    # if "traj" in field_set:
    #     prompt += f"Trajectory(last-k): [{traj_text}].\n"
    # if "conf_history" in field_set:
    #     prompt += f"Confidence history(last-k): [{conf_hist_text}].\n"
    # if "summary_history" in field_set and summary_hist_text:
    #     prompt += f'Local summary history(last-k): "{summary_hist_text}".\n'
    # if "running_memory" in field_set and running_summary:
    #     prompt += f"Running memory summary: \"{running_summary}\".\n"
    # if "local_summary" in field_set and local_summary:
    #     prompt += f"Current local summary: \"{local_summary}\".\n"
    # if "speaker" in field_set and speaker:
    #     prompt += f"Speaker: {speaker}.\n"
    if "dialogue" in field_set and dialogue:
        prompt += f'The dialogue: "{dialogue}".\n'
    # if "audio_evidence" in field_set and audio_evidence:
    #     prompt += f'Observed audio evidence: "{audio_evidence}".\n'
    # if "visual_evidence" in field_set and visual_evidence:
    #     prompt += f'Observed visual evidence: "{visual_evidence}".\n'
    # if "conflict_explanation" in field_set and conflict_explanation:
    #     prompt += f'Cross-modal conflict note: "{conflict_explanation}".\n'
    # if "perception_evolution" in field_set and pe:
    #     immediate = _pick_text([pe.get("immediate_interpretation", None)])
    #     truth = _pick_text([pe.get("retrospective_truth", None)])
    #     gap = _pick_text([pe.get("gap_type", None)])
    #     if immediate:
    #         prompt += f"Perception evolution immediate: {immediate}.\n"
    #     if truth:
    #         prompt += f"Perception evolution retrospective: {truth}.\n"
    #     if gap:
    #         prompt += f"Perception gap type: {gap}.\n"
    # if "reason" in field_set and reason_text:
    #     prompt += f'Potential revision rationale: "{reason_text}".\n'
    prev_summary=_pick_text([sample.get("prev_summary", None)])
    if prev_summary:
        prompt += f"Previous summary: \"{prev_summary}\".\n"
    prompt += (
        "Output strict JSON with keys: \n"
        "- need_correction: 0 or 1\n"
        "- corrected_emotion: one label from valid set\n"
        "- correction_delta: format old->new or none\n"
    )
    # for key in output_fields:
    #     if key in {"need_correction", "corrected_emotion", "correction_delta"}:
    #         continue
    #     prompt += f"- {key}\n"
    # print(prompt)
    return prompt


def build_target_json(
    sample: Dict[str, Any],
    gt_label: str,
    valid_labels: List[str],
    output_fields: List[str],
) -> str:
    need = resolve_need_correction(sample, gt_label)
    delta = resolve_delta(sample, need, gt_label)

    target = {
        "need_correction": int(need),
        "corrected_emotion": gt_label if gt_label in valid_labels else valid_labels[0],
        "correction_delta": delta,
    }

    out_set = set(output_fields)
    # if "reason_code" in out_set:
    #     reason_code = maybe_reason_code(sample)
    #     if reason_code:
    #         target["reason_code"] = reason_code
    # if "reason" in out_set:
    #     reason_text = resolve_reason_text(sample)
    #     if reason_text:
    #         target["reason"] = reason_text
    # if "audio_evidence" in out_set:
    #     text = resolve_audio_evidence(sample)
    #     if text:
    #         target["audio_evidence"] = text
    # if "visual_evidence" in out_set:
    #     text = resolve_visual_evidence(sample)
    #     if text:
    #         target["visual_evidence"] = text
    # if "conflict_explanation" in out_set:
    #     text = resolve_conflict_explanation(sample)
    #     if text:
    #         target["conflict_explanation"] = text
    # if "new_summary" in out_set:
    #     summary = resolve_new_summary(sample)
    #     if summary:
    #         target["new_summary"] = summary
    # if "updated_running_summary" in out_set:
    #     summary = resolve_new_summary(sample)
    #     if summary:
    #         target["updated_running_summary"] = summary

    return json.dumps(target, ensure_ascii=False)


def parse_json_response(text: str) -> Dict[str, Any]:
    clean = text.strip()
    if clean.startswith("```json"):
        clean = clean[7:]
    elif clean.startswith("```"):
        clean = clean[3:]
    if clean.endswith("```"):
        clean = clean[:-3]
    clean = clean.strip()

    start = clean.find("{")
    end = clean.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return {}

    try:
        return json.loads(clean[start : end + 1])
    except Exception:
        return {}


def to_int01(v: Any) -> int:
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, (int, float)):
        return int(v > 0)
    s = str(v).strip().lower()
    if s in {"1", "true", "yes", "y"}:
        return 1
    return 0


class StateJsonlDataset(Dataset):
    def __init__(
        self,
        jsonl_path: str,
        min_fast_confidence: Optional[float] = None,
        max_fast_confidence: Optional[float] = None,
    ):
        samples = load_jsonl(jsonl_path)
        self.original_size = len(samples)
        self.min_fast_confidence = min_fast_confidence
        self.max_fast_confidence = max_fast_confidence
        self.samples = self._filter_by_confidence(samples)
        print(samples[0])

    def _filter_by_confidence(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if self.min_fast_confidence is None and self.max_fast_confidence is None:
            return samples

        kept: List[Dict[str, Any]] = []
        for sample in samples:
            conf = resolve_fast_confidence(sample)
            if self.min_fast_confidence is not None and conf < self.min_fast_confidence:
                continue
            if self.max_fast_confidence is not None and conf > self.max_fast_confidence:
                continue
            kept.append(sample)
        return kept

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx]


class SlowTrackCollator:
    def __init__(
        self,
        processor: Qwen2_5OmniProcessor,
        valid_labels: List[str],
        label_map: Dict[str, str],
        label_keys: List[str],
        default_label: str,
        video_key: str,
        use_video: bool,
        use_audio_in_video: bool,
        fps: int,
        max_pixels: int,
        input_fields: List[str],
        output_fields: List[str],
    ):
        self.processor = processor
        self.valid_labels = valid_labels
        self.label_map = label_map
        self.label_keys = label_keys
        self.default_label = default_label
        self.video_key = video_key
        self.use_video = use_video
        self.use_audio_in_video = use_audio_in_video
        self.fps = fps
        self.max_pixels = max_pixels
        self.input_fields = input_fields
        self.output_fields = output_fields
        self.ignore_index = -100

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        full_texts, prompt_texts = [], []
        audios_list, videos_list = [], []

        for sample in examples:
            gt_label = resolve_gt_label(
                sample,
                self.label_keys,
                self.valid_labels,
                self.label_map,
                self.default_label,
            )
            print(gt_label)
            prompt_text = build_slow_prompt(
                sample,
                self.valid_labels,
                self.input_fields,
                self.output_fields,
            )
            target_text = build_target_json(
                sample,
                gt_label,
                self.valid_labels,
                self.output_fields,
            )

            content = []
            if self.use_video:
                video_paths = resolve_video_paths(sample, self.video_key)
                if not video_paths:
                    raise KeyError("use-video=True but video path is missing in state sample")
                for vp in video_paths:
                    content.append(
                        {
                            "type": "video",
                            "video": vp,
                            "fps": self.fps,
                            "max_pixels": self.max_pixels,
                        }
                    )
            content.append({"type": "text", "text": prompt_text})

            full_conv = [
                {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                {"role": "user", "content": content},
                {"role": "assistant", "content": [{"type": "text", "text": target_text}]},
            ]

            prompt_conv = [
                {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                {"role": "user", "content": content},
            ]

            audios, _, videos = process_mm_info(full_conv, use_audio_in_video=self.use_audio_in_video)
            audios_list.append(audios[0] if audios else None)
            videos_list.append(videos[0] if videos else None)

            full_text = self.processor.apply_chat_template(full_conv, tokenize=False)
            prompt_only = self.processor.apply_chat_template(
                prompt_conv,
                add_generation_prompt=True,
                tokenize=False,
            )
            full_texts.append(full_text)
            prompt_texts.append(prompt_only)

        batch = self.processor(
            text=full_texts,
            audio=audios_list,
            videos=videos_list,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=self.use_audio_in_video,
        )

        batch_prompt = self.processor(
            text=prompt_texts,
            audio=audios_list,
            videos=videos_list,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=self.use_audio_in_video,
        )

        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        prompt_input_ids = batch_prompt["input_ids"]
        prompt_attention_mask = batch_prompt["attention_mask"]

        bsz, seq_len = input_ids.shape
        labels = input_ids.clone()

        labels[attention_mask == 0] = self.ignore_index

        full_lens = attention_mask.sum(dim=1)
        prompt_lens = prompt_attention_mask.sum(dim=1)

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

            pad_len = seq_len - full_len

            prompt_start = pad_len
            prompt_end = pad_len + prompt_len

            full_prefix = input_ids[i, prompt_start:prompt_end]
            prompt_valid = prompt_input_ids[i, -prompt_len:]

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

            labels[i, prompt_start:prompt_end] = self.ignore_index

        batch["labels"] = labels
        return batch


class SlowTrackTrainer(Trainer):
    def __init__(
        self,
        *args,
        processor: Qwen2_5OmniProcessor,
        valid_labels: List[str],
        label_map: Dict[str, str],
        label_keys: List[str],
        default_label: str,
        video_key: str,
        use_video: bool,
        use_audio_in_video: bool,
        fps: int,
        max_pixels: int,
        input_fields: List[str],
        output_fields: List[str],
        eval_records_dir: str = "",
        eval_records_max_samples: int = 0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.processor = processor
        self.valid_labels = valid_labels
        self.label_map = label_map
        self.label_keys = label_keys
        self.default_label = default_label
        self.video_key = video_key
        self.use_video = use_video
        self.use_audio_in_video = use_audio_in_video
        self.fps = fps
        self.max_pixels = max_pixels
        self.input_fields = input_fields
        self.output_fields = output_fields
        self.eval_records_dir = str(eval_records_dir or "").strip()
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
        if eval_dataset is None:
            return {}

        batch_size = self.args.per_device_eval_batch_size
        device = self.args.device
        rank, world_size = self._dist_rank_world_size()

        self.model.eval()
        old_padding_side = self.processor.tokenizer.padding_side
        self.processor.tokenizer.padding_side = "left"

        total_count = len(eval_dataset)
        eval_indices = list(range(rank, total_count, world_size))
        local_count = len(eval_indices)

        need_ok = 0
        corr_ok = 0
        joint_ok = 0
        eval_rows: List[Dict[str, Any]] = []
        model_to_gen = self.accelerator.unwrap_model(self.model)

        try:
            with torch.no_grad():
                disable_tqdm = rank > 0
                for start in tqdm(
                    range(0, local_count, batch_size),
                    desc="Slow-track generative eval",
                    disable=disable_tqdm,
                ):
                    batch_indices = eval_indices[start:start + batch_size]
                    batch_samples = [eval_dataset[idx] for idx in batch_indices]

                    batch_texts, batch_audios, batch_videos = [], [], []
                    gt_needs, gt_corrs = [], []
                    batch_prompts, batch_ids = [], []

                    for sample_idx, sample in zip(batch_indices, batch_samples):
                        gt_label = resolve_gt_label(
                            sample,
                            self.label_keys,
                            self.valid_labels,
                            self.label_map,
                            self.default_label,
                        )
                        gt_need = resolve_need_correction(sample, gt_label)

                        gt_needs.append(gt_need)
                        gt_corrs.append(gt_label)

                        content = []
                        if self.use_video:
                            video_paths = resolve_video_paths(sample, self.video_key)
                            if not video_paths:
                                raise KeyError("use-video=True but video path is missing in eval sample")
                            for vp in video_paths:
                                content.append(
                                    {
                                        "type": "video",
                                        "video": vp,
                                        "fps": self.fps,
                                        "max_pixels": self.max_pixels,
                                    }
                                )
                        prompt_text = build_slow_prompt(
                            sample,
                            self.valid_labels,
                            self.input_fields,
                            self.output_fields,
                        )
                        content.append({"type": "text", "text": prompt_text})

                        conv = [
                            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                            {"role": "user", "content": content},
                        ]

                        audios, _, videos = process_mm_info(
                            conv,
                            use_audio_in_video=self.use_audio_in_video,
                        )
                        batch_audios.append(audios[0] if audios else None)
                        batch_videos.append(videos[0] if videos else None)

                        text = self.processor.apply_chat_template(
                            conv,
                            add_generation_prompt=True,
                            tokenize=False,
                        )
                        batch_texts.append(text)
                        batch_ids.append(self._resolve_sample_id(sample, sample_idx))
                        batch_prompts.append(prompt_text)

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
                        max_new_tokens=96,
                        temperature=0.1,
                        pad_token_id=self.processor.tokenizer.pad_token_id,
                        eos_token_id=self.processor.tokenizer.eos_token_id,
                        use_audio_in_video=self.use_audio_in_video,
                    )

                    generated_trim = [
                        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                    ]
                    pred_texts = self.processor.batch_decode(
                        generated_trim,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )

                    for pred_text, gt_need, gt_corr in zip(pred_texts, gt_needs, gt_corrs):
                        parsed = parse_json_response(pred_text)
                        pred_need = to_int01(parsed.get("need_correction", 0))
                        pred_corr = canonicalize_prediction(
                            str(parsed.get("corrected_emotion", "")),
                            self.valid_labels,
                        )

                        need_match = int(pred_need == gt_need)
                        corr_match = int(pred_corr == gt_corr)
                        joint_match = int(need_match and corr_match)

                        need_ok += need_match
                        corr_ok += corr_match
                        joint_ok += joint_match

                    if self.eval_records_dir:
                        for k, (pred_text, gt_need, gt_corr) in enumerate(zip(pred_texts, gt_needs, gt_corrs)):
                            eval_rows.append(
                                {
                                    "id": batch_ids[k],
                                    "eval_index": int(batch_indices[k]),
                                    "prompt": batch_prompts[k],
                                    "output": pred_text,
                                    "output_need_correction": to_int01(
                                        parse_json_response(pred_text).get("need_correction", 0)
                                    ),
                                    "output_corrected_emotion": canonicalize_prediction(
                                        str(parse_json_response(pred_text).get("corrected_emotion", "")),
                                        self.valid_labels,
                                    ),
                                    "target_need_correction": int(gt_need),
                                    "target_corrected_emotion": gt_corr,
                                    "need_match": int(
                                        to_int01(parse_json_response(pred_text).get("need_correction", 0)) == gt_need
                                    ),
                                    "corrected_match": int(
                                        canonicalize_prediction(
                                            str(parse_json_response(pred_text).get("corrected_emotion", "")),
                                            self.valid_labels,
                                        ) == gt_corr
                                    ),
                                    "joint_match": int(
                                        (
                                            to_int01(parse_json_response(pred_text).get("need_correction", 0)) == gt_need
                                        )
                                        and (
                                            canonicalize_prediction(
                                                str(parse_json_response(pred_text).get("corrected_emotion", "")),
                                                self.valid_labels,
                                            ) == gt_corr
                                        )
                                    ),
                                }
                            )
        finally:
            self.processor.tokenizer.padding_side = old_padding_side
            self.model.train()

        if world_size > 1:
            stats = torch.tensor(
                [need_ok, corr_ok, joint_ok, local_count],
                device=device,
                dtype=torch.long,
            )
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
            need_total = int(stats[0].item())
            corr_total = int(stats[1].item())
            joint_total = int(stats[2].item())
            total_eval = int(stats[3].item())
        else:
            need_total = need_ok
            corr_total = corr_ok
            joint_total = joint_ok
            total_eval = local_count

        metrics = {
            f"{metric_key_prefix}_need_acc": need_total / total_eval if total_eval else 0.0,
            f"{metric_key_prefix}_corrected_acc": corr_total / total_eval if total_eval else 0.0,
            f"{metric_key_prefix}_joint_acc": joint_total / total_eval if total_eval else 0.0,
        }
        self.log(metrics)

        if self.eval_records_dir:
            eval_rows = self._gather_eval_rows(eval_rows)
            if self.eval_records_max_samples > 0:
                eval_rows = eval_rows[:self.eval_records_max_samples]
            self._maybe_save_eval_records(eval_rows, metric_key_prefix)

        return metrics

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train slow-track correction model from fast-state jsonl.")
    parser.add_argument("--local_rank", "--local-rank", type=int, default=-1)
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--train-states-jsonl", type=str, required=True)
    parser.add_argument("--val-states-jsonl", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)

    parser.add_argument(
        "--labels",
        type=str,
        default="neutral,anger,anxiety,sadness,joy,surprise",
    )
    parser.add_argument("--default-label", type=str, default="neutral")
    parser.add_argument("--label-map-json", type=str, default="")
    parser.add_argument("--label-keys", type=str, default="gt_emotion,emotion,target")

    parser.add_argument("--video-key", type=str, default="video_path")
    parser.add_argument("--use-video", type=str2bool, default=True)
    parser.add_argument("--use-audio-in-video", type=str2bool, default=True)
    parser.add_argument("--fps", type=int, default=2)
    parser.add_argument("--max-pixels", type=int, default=224 * 126)
    parser.add_argument(
        "--slow-input-fields",
        type=str,
        default=(
            "fast_pred,fast_confidence,fast_probs,uncertainty,traj,conf_history,summary_history,"
            "running_memory,local_summary,speaker,dialogue"
        ),
        help=(
            "Comma-separated prompt input blocks. "
            "Choices include: fast_pred,fast_confidence,fast_probs,uncertainty,traj,conf_history,"
            "summary_history,running_memory,local_summary,speaker,dialogue,audio_evidence,"
            "visual_evidence,conflict_explanation,perception_evolution,reason"
        ),
    )
    parser.add_argument(
        "--slow-output-fields",
        type=str,
        default="reason,new_summary",
        help=(
            "Comma-separated extra output fields in target JSON. Base fields "
            "need_correction/corrected_emotion/correction_delta are always included."
        ),
    )

    parser.add_argument("--num-train-epochs", type=int, default=5)
    parser.add_argument("--per-device-train-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
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
    parser.add_argument("--run-name", type=str, default="qwen-omni-slow-track-lora")

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
    parser.add_argument(
        "--train-min-fast-confidence",
        type=float,
        default=-1.0,
        help="Optional lower bound for training samples; negative disables the filter.",
    )
    parser.add_argument(
        "--train-max-fast-confidence",
        type=float,
        default=-1.0,
        help="Optional upper bound for training samples; negative disables the filter.",
    )
    parser.add_argument(
        "--val-min-fast-confidence",
        type=float,
        default=-1.0,
        help="Optional lower bound for validation samples; negative disables the filter.",
    )
    parser.add_argument(
        "--val-max-fast-confidence",
        type=float,
        default=-1.0,
        help="Optional upper bound for validation samples; negative disables the filter.",
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
    input_fields = parse_csv(args.slow_input_fields)
    output_fields = parse_csv(args.slow_output_fields)
    train_min_conf = args.train_min_fast_confidence if args.train_min_fast_confidence >= 0 else None
    train_max_conf = args.train_max_fast_confidence if args.train_max_fast_confidence >= 0 else None
    val_min_conf = args.val_min_fast_confidence if args.val_min_fast_confidence >= 0 else None
    val_max_conf = args.val_max_fast_confidence if args.val_max_fast_confidence >= 0 else None

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if local_rank == 0:
        print("Loading processor/model for slow track...")

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
        print("Loading state datasets...")

    train_dataset = StateJsonlDataset(
        args.train_states_jsonl,
        min_fast_confidence=train_min_conf,
        max_fast_confidence=train_max_conf,
    )
    val_dataset = StateJsonlDataset(
        args.val_states_jsonl,
        min_fast_confidence=val_min_conf,
        max_fast_confidence=val_max_conf,
    )
    if local_rank == 0:
        print(
            "Train dataset filtered: "
            f"{len(train_dataset)}/{train_dataset.original_size} kept "
            f"(min_conf={train_min_conf}, max_conf={train_max_conf})"
        )
        print(
            "Val dataset filtered: "
            f"{len(val_dataset)}/{val_dataset.original_size} kept "
            f"(min_conf={val_min_conf}, max_conf={val_max_conf})"
        )
    if len(train_dataset) == 0:
        raise ValueError("Training dataset is empty after fast-confidence filtering.")
    if len(val_dataset) == 0:
        raise ValueError("Validation dataset is empty after fast-confidence filtering.")

    collator = SlowTrackCollator(
        processor=processor,
        valid_labels=valid_labels,
        label_map=label_map,
        label_keys=label_keys,
        default_label=args.default_label.lower(),
        video_key=args.video_key,
        use_video=args.use_video,
        use_audio_in_video=args.use_audio_in_video,
        fps=args.fps,
        max_pixels=args.max_pixels,
        input_fields=input_fields,
        output_fields=output_fields,
    )

    training_kwargs = dict(
        output_dir=str(output_dir),
        num_train_epochs=5,
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
        # eval_steps=0.5,
    )

    init_vars = TrainingArguments.__init__.__code__.co_varnames
    if "eval_strategy" in init_vars:
        training_kwargs["eval_strategy"] = "epoch"
    else:
        training_kwargs["evaluation_strategy"] = "epoch"

    if args.deepspeed_config:
        training_kwargs["deepspeed"] = args.deepspeed_config

    training_args = TrainingArguments(**training_kwargs)

    trainer = SlowTrackTrainer(
        processor=processor,
        valid_labels=valid_labels,
        label_map=label_map,
        label_keys=label_keys,
        default_label=args.default_label.lower(),
        video_key=args.video_key,
        use_video=args.use_video,
        use_audio_in_video=args.use_audio_in_video,
        fps=args.fps,
        max_pixels=args.max_pixels,
        input_fields=input_fields,
        output_fields=output_fields,
        eval_records_dir=args.eval_records_dir,
        eval_records_max_samples=args.eval_records_max_samples,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
    )

    if local_rank == 0:
        print("Start slow-track LoRA training...")

    trainer.train()

    if local_rank == 0:
        print("Saving slow-track LoRA model...")

    final_dir = output_dir / "slow_track_lora"
    trainer.save_model(str(final_dir))

    if local_rank == 0:
        processor.save_pretrained(str(final_dir))
        config_dump = {
            "labels": valid_labels,
            "default_label": args.default_label.lower(),
            "label_keys": label_keys,
            "video_key": args.video_key,
            "use_video": args.use_video,
            "use_audio_in_video": args.use_audio_in_video,
            "fps": args.fps,
            "max_pixels": args.max_pixels,
            "slow_input_fields": input_fields,
            "slow_output_fields": output_fields,
            "label_map": label_map,
            "train_min_fast_confidence": train_min_conf,
            "train_max_fast_confidence": train_max_conf,
            "val_min_fast_confidence": val_min_conf,
            "val_max_fast_confidence": val_max_conf,
        }
        with open(output_dir / "slow_track_label_config.json", "w", encoding="utf-8") as f:
            json.dump(config_dump, f, ensure_ascii=False, indent=2)

        print(f"Slow-track training complete. Saved to {final_dir}")


if __name__ == "__main__":
    main()