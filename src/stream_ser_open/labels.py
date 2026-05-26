from typing import Any, Dict, Iterable, List


ALIASES = {
    "angry": "anger",
    "disgust": "anger",
    "contempt": "anger",
    "fear": "anxiety",
    "happiness": "joy",
    "happy": "joy",
    "affection": "joy",
    "awkwardness": "embarrassment",
    "guilt": "embarrassment",
    "resignation": "sadness",
}


def normalize_label(
    value: Any,
    valid_labels: Iterable[str],
    label_map: Dict[str, str],
    default_label: str,
) -> str:
    labels = {label.lower() for label in valid_labels}
    text = str(value or "").strip()
    low = text.lower()
    candidates = [low, label_map.get(text, ""), label_map.get(low, ""), ALIASES.get(low, "")]
    for item in candidates:
        if item and item.lower() in labels:
            return item.lower()
    return default_label


def canonicalize_prediction(text: str, valid_labels: List[str], default_label: str) -> str:
    cleaned = "".join(ch for ch in str(text).lower().strip() if ch.isalnum())
    if cleaned in valid_labels:
        return cleaned
    if cleaned in ALIASES and ALIASES[cleaned] in valid_labels:
        return ALIASES[cleaned]
    for label in valid_labels:
        if label in cleaned:
            return label
    return default_label


def extract_label(
    sample: Dict[str, Any],
    label_keys: List[str],
    valid_labels: List[str],
    label_map: Dict[str, str],
    default_label: str,
) -> str:
    sources = [sample]
    if isinstance(sample.get("original"), dict):
        sources.append(sample["original"])
    for source in sources:
        for key in label_keys:
            value = source.get(key)
            if value is not None and str(value).strip():
                return normalize_label(value, valid_labels, label_map, default_label)
    return default_label
