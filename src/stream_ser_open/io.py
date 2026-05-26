import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def resolve_path(path: str) -> Path:
    return Path(path).expanduser().resolve()


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with resolve_path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
    return rows


def dump_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    output = resolve_path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_label_map(path: Optional[str]) -> Dict[str, str]:
    if not path:
        return {}
    with resolve_path(path).open("r", encoding="utf-8") as f:
        raw = json.load(f)
    out: Dict[str, str] = {}
    for key, value in raw.items():
        out[str(key).strip()] = str(value).strip().lower()
        out[str(key).strip().lower()] = str(value).strip().lower()
    return out


def parse_csv(value: str) -> List[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def safe_float(value: Any, fallback: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def get_nested_original(sample: Dict[str, Any]) -> Dict[str, Any]:
    original = sample.get("original")
    return original if isinstance(original, dict) else {}


def first_text(candidates: Iterable[Any]) -> str:
    for item in candidates:
        if item is None:
            continue
        text = str(item).strip()
        if text:
            return text
    return ""


def get_stream_id(sample: Dict[str, Any], stream_key: str, idx: int) -> str:
    value = first_text([sample.get(stream_key), sample.get("stream_id")])
    if value:
        return value
    clip_id = first_text([sample.get("clip_id"), get_nested_original(sample).get("clip_id")])
    speaker = first_text([sample.get("speaker"), get_nested_original(sample).get("speaker")])
    if clip_id and speaker:
        return f"{clip_id}::{speaker}"
    return clip_id or f"sample_{idx}"


def sort_samples(
    samples: List[Dict[str, Any]],
    stream_key: str,
    step_key: str,
    start_key: str,
) -> List[Tuple[int, Dict[str, Any]]]:
    indexed = list(enumerate(samples))

    def key_fn(item: Tuple[int, Dict[str, Any]]) -> Tuple[str, float, int]:
        idx, sample = item
        stream_id = get_stream_id(sample, stream_key, idx)
        order = safe_float(sample.get(step_key), safe_float(sample.get(start_key), float(idx)))
        return stream_id, order, idx

    indexed.sort(key=key_fn)
    return indexed
