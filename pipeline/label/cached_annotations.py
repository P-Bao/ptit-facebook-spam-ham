"""pipeline/label/cached_annotations.py
Liên kết gợi ý gán nhãn với các bình luận sau làm sạch.

Đầu vào: bảng bình luận, chỉ mục annotation và cache dự đoán.
Xử lý: đối chiếu post_id/comment_type/text, giữ ID đúng, từ chối liên kết mâu thuẫn.
Đầu ra: bảng có ID ổn định và cache chỉ gồm các ID thuộc bảng đó.
Liên quan: auto_accept_labels; các file đầu vào local của notebook 01.
"""

import hashlib
import json
import unicodedata

from .label_utils import record_ids

KEY_COLUMNS = ("post_id", "comment_type", "text")


def _key(row):
    return tuple(str(row[column]) for column in KEY_COLUMNS)


def attach_annotation_ids(frame, annotation_index, predictions):
    """Dùng khóa nội dung thay vì coi số dòng của bảng mới là ID cache."""
    for name, data in [("comments", frame), ("annotation index", annotation_index)]:
        missing = {*KEY_COLUMNS, "cleaned_text"} - set(data)
        if missing:
            raise ValueError(f"{name} thiếu cột: {sorted(missing)}")
        if data.duplicated(list(KEY_COLUMNS)).any():
            raise ValueError(f"{name} có khóa nội dung trùng")
    source_ids = record_ids(annotation_index)
    if set(predictions) - set(source_ids):
        raise ValueError("Cache chứa ID không thuộc chỉ mục annotation")
    indexed = {_key(row): (rid, str(row["cleaned_text"]))
               for rid, (_, row) in zip(source_ids, annotation_index.iterrows())}
    result = frame.copy()
    ids = []
    norm = lambda value: unicodedata.normalize("NFC", value).strip()
    for _, row in result.iterrows():
        key = _key(row)
        if key in indexed:
            rid, expected = indexed[key]
            if norm(str(row["cleaned_text"])) != norm(expected):
                raise ValueError("Text làm sạch khác nguồn gợi ý; từ chối dùng nhãn cache")
        else:
            digest = hashlib.sha256(json.dumps(key, ensure_ascii=False).encode()).hexdigest()[:24]
            rid = "comment_"+digest
        ids.append(rid)
    result["id"] = ids
    record_ids(result)  # Kiểm tra cả trường hợp ID được cung cấp/generate bị trùng.
    present = set(ids)
    return result, {rid: item for rid, item in predictions.items() if rid in present}
