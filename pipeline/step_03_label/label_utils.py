"""Đối soát ID/nhãn thuần cục bộ, không kết nối Argilla."""

import math
import re
import unicodedata

import pandas as pd


def record_ids(frame, prefix="cmt_"):
    if "id" in frame.columns:
        ids = frame["id"].fillna("").astype(str).str.strip()
        if ids.eq("").any() or ids.duplicated().any():
            raise ValueError("CSV có ID rỗng hoặc trùng; cần đối soát trước khi gán nhãn.")
        return ids.tolist()
    return [f"{prefix}{i}" for i in range(len(frame))]


def normalize_prediction(item):
    label = str(item.get("label", "")).strip().lower()
    try:
        confidence = float(item.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    if not math.isfinite(confidence) or not 0 <= confidence <= 1:
        confidence = 0.0
    return {"label": label if label in {"spam", "ham"} else None,
            "confidence": confidence}


def join_metadata(labeled, lookup):
    """Nối bằng ID hoặc vị trí legacy; dừng thay vì âm thầm bỏ/misalign dòng."""
    if labeled.empty:
        return labeled.copy()
    result = labeled.copy()
    if "id" in lookup:
        lookup = lookup.copy()
        lookup["id"] = record_ids(lookup)
        positions = pd.Series(range(len(lookup)), index=lookup["id"])
        indices = result["id"].astype(str).map(positions)
    else:
        ids = result["id"].astype(str)
        namespaces = ids.str.extract(r"^((?:p2_)?cmt_)\d+$", expand=False)
        if namespaces.nunique() > 1:
            raise ValueError("Dataset chứa nhiều phase; CSV lookup phải có cột id để nối chính xác.")
        indices = pd.to_numeric(ids.str.extract(r"^(?:p2_)?cmt_(\d+)$", expand=False), errors="coerce")
    if indices.isna().any() or ((indices < 0) | (indices >= len(lookup))).any():
        raise ValueError("Có ID không nối được CSV lookup; không xuất thiếu dòng. Kiểm tra đúng CSV/phase.")
    matched = lookup.iloc[indices.astype(int)].reset_index(drop=True)
    if "cleaned_text" in matched and "cleaned_text" in result:
        norm = lambda s: unicodedata.normalize("NFC", str(s)).strip()
        expected_display = matched["cleaned_text"].fillna("").astype(str)
        if "text" in matched:
            # Uploader dùng raw text khi cleaner để rỗng (ví dụ bình luận chỉ emoji).
            expected_display = expected_display.where(expected_display.str.strip().ne(""), matched["text"].fillna("").astype(str))
        if not (result["cleaned_text"].map(norm).to_numpy() == expected_display.map(norm).to_numpy()).all():
            raise ValueError("ID nối được nhưng cleaned_text khác CSV lookup; từ chối ghép nhầm văn bản.")
    for column in ["text", "comment_type", "reaction_count"]:
        if column in matched:
            result[column] = matched[column].to_numpy()
    return result
