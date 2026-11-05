"""pipeline/modern/data.py
Chuẩn bị văn bản và cách chia tập cho viBERT.

Đầu vào: CSV v7 và chỉ số fold validation.
Xử lý: giữ nguyên test của notebook 02; chọn validation trong train theo nhóm;
chuẩn hóa Unicode/khoảng trắng, giữ chữ hoa/thường và từ dừng.
Đầu ra: bảng có id/text/target/outer_split/development_split và cột model_text.
Liên quan: modeling.training_utils; modern.train; notebook 03.
"""

import html
import re
import unicodedata

import numpy as np
import pandas as pd

from pipeline.modeling.training_utils import norm_key, bucket, grouped_folds


def normalize_text(text):
    text = html.unescape(unicodedata.normalize("NFC", str(text)))
    return re.sub(r"\s+", " ", text).strip()


def prepare_data(csv_path, validation_fold=0):
    data = pd.read_csv(csv_path, keep_default_na=False, dtype={"id": str})
    required = {"id", "text", "cleaned_text", "label"}
    if not required.issubset(data.columns):
        raise ValueError(f"CSV thiếu cột: {sorted(required - set(data.columns))}")
    if not data.id.is_unique or data.id.str.strip().eq("").any():
        raise ValueError("ID phải duy nhất và không rỗng.")
    if set(data.label) != {"ham", "spam"}:
        raise ValueError("Dữ liệu cần đúng hai nhãn ham/spam.")
    data["target"] = data.label.eq("spam").astype(int)
    data["model_text"] = data.text.map(normalize_text)
    keys = data.cleaned_text.map(norm_key).to_numpy()
    is_test = np.array([bucket(key, "test") == 0 for key in keys])
    train_indices = np.flatnonzero(~is_test)
    y_train = data.target.to_numpy()[train_indices]
    folds = grouped_folds(keys[train_indices], y_train)
    if not 0 <= validation_fold < len(folds):
        raise ValueError("validation_fold phải từ 0 đến 4.")
    fit, valid = folds[validation_fold]
    data["outer_split"] = np.where(is_test, "test", "train")
    data["development_split"] = "test"
    data.loc[train_indices[fit], "development_split"] = "fit"
    data.loc[train_indices[valid], "development_split"] = "validation"
    for left, right in [("fit", "validation"), ("fit", "test"), ("validation", "test")]:
        assert not set(keys[data.development_split.eq(left)]) & set(keys[data.development_split.eq(right)])
    if data.groupby("development_split").target.nunique().ne(2).any():
        raise ValueError("Mỗi tập cần đủ hai lớp.")
    return data


def class_weights(labels):
    counts = np.bincount(np.asarray(labels, dtype=int), minlength=2)
    if (counts == 0).any():
        raise ValueError("Không thể tính trọng số khi thiếu lớp.")
    return len(labels) / (2 * counts)


def split_summary(data):
    return data.groupby(["development_split", "label"]).size().unstack(fill_value=0).reindex(
        ["fit", "validation", "test"])
