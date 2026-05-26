import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


REPO_ROOT = Path(__file__).resolve().parent
DATA_DIR = REPO_ROOT / "data"
TRAIN_JSONL = DATA_DIR / "train.jsonl"
VAL_JSONL = DATA_DIR / "val.jsonl"


def resolve_repo_path(path: str) -> Path:
    if path is None:
        return None
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return REPO_ROOT / candidate


def split_jsonl(split: str) -> Path:
    normalized = split.strip().lower()
    if normalized == "train":
        return TRAIN_JSONL
    if normalized in {"val", "valid", "validation"}:
        return VAL_JSONL
    raise ValueError(f"Unsupported split: {split}")


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    input_path = resolve_repo_path(path)
    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def dump_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    output_path = resolve_repo_path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_dataset_rows(path: str, drop_missing_video: bool = False) -> List[Dict[str, Any]]:
    rows = load_jsonl(path)
    if not drop_missing_video:
        return rows

    filtered: List[Dict[str, Any]] = []
    for row in rows:
        video_path = str(row.get("video_path") or "").strip()
        if not video_path:
            continue
        if resolve_repo_path(video_path).exists():
            filtered.append(row)
    return filtered
