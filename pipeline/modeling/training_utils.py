"""pipeline/modeling/training_utils.py
Tiện ích chia nhóm, CV, sampling, ngưỡng và dự đoán.

Đầu vào: Text key, nhãn, chỉ số fold và pipeline mô hình.
Xử lý: Giữ text trùng cùng nhóm; sampling chỉ phần train; tìm ngưỡng trên OOF của train.
Đầu ra: Fold/chỉ số, mô hình/tham số hoặc prediction/score.
Liên quan: pipeline.modeling.experiments; pipeline.inference.predict.
"""

import hashlib
import re
import unicodedata

import numpy as np
from sklearn.base import clone
from sklearn.metrics import f1_score
from sklearn.model_selection import ParameterGrid


def norm_key(text):
    normalized = unicodedata.normalize("NFC", str(text)).lower()
    normalized = re.sub(r"\s+", " ", normalized).strip()
    key = re.sub(r"\d+", "0", normalized)
    key = re.sub(r"[^\w\s]", " ", key)
    key = re.sub(r"\s+", " ", key).strip()
    # Câu chỉ dấu câu vẫn có khóa ổn định theo nội dung, không theo chỉ số dòng.
    return key or "__symbols__|" + normalized


def bucket(key, salt, n=5):
    return int(hashlib.md5(f"{salt}|{key}".encode()).hexdigest(), 16) % n


def grouped_folds(keys, y, n_splits=5):
    keys = np.asarray(keys)
    y = np.asarray(y)
    ids = np.array([bucket(key, "cv", n_splits) for key in keys])
    folds = []
    for fold in range(n_splits):
        train, valid = np.where(ids != fold)[0], np.where(ids == fold)[0]
        if len(np.unique(y[train])) != 2 or len(np.unique(y[valid])) != 2:
            raise ValueError("Fold thiếu một lớp; cần kiểm tra nhóm dữ liệu hoặc số fold.")
        assert not set(keys[train]) & set(keys[valid])
        folds.append((train, valid))
    return folds


def sample_training_indices(y, indices, seed=42, ratio=2):
    """Chỉ lấy mẫu từ indices được phép; không lấy từ validation/test."""
    indices = np.asarray(indices)
    spam = indices[np.asarray(y)[indices] == 1]
    ham = indices[np.asarray(y)[indices] == 0]
    rng = np.random.default_rng(seed)
    sampled = rng.choice(ham, size=min(ratio * len(spam), len(ham)), replace=False)
    selected = np.concatenate([spam, sampled])
    rng.shuffle(selected)
    return selected


def undersampled_search(pipe, grid, X, y, folds, seed=42):
    """Chọn tham số bằng các fold gốc; validation giữ phân bố gốc."""
    rows = []
    for params in ParameterGrid(grid):
        scores = []
        for train, valid in folds:
            selected = sample_training_indices(y, train, seed)
            assert not set(selected) & set(valid)
            model = clone(pipe).set_params(**params).fit(X[selected], y[selected])
            scores.append(f1_score(y[valid], model.predict(X[valid]), zero_division=0))
        rows.append((float(np.mean(scores)), params))
    score, params = max(rows, key=lambda row: row[0])
    selected = sample_training_indices(y, np.arange(len(y)), seed)
    fitted = clone(pipe).set_params(**params).fit(X[selected], y[selected])
    return fitted, score, params, len(selected)


def tune_nb_threshold(pipe, X, y, folds):
    """OOF trên train; chọn ngưỡng bằng mean F1 cùng các fold, không nhận test."""
    probabilities = np.zeros(len(y))
    covered = np.zeros(len(y), dtype=int)
    for train, valid in folds:
        model = clone(pipe).fit(X[train], y[train])
        probabilities[valid] = model.predict_proba(X[valid])[:, 1]
        covered[valid] += 1
    assert (covered == 1).all()
    rows = []
    for threshold in np.linspace(0.05, 0.95, 200):
        score = np.mean([f1_score(y[valid], probabilities[valid] >= threshold, zero_division=0)
                         for _, valid in folds])
        rows.append((float(score), float(threshold)))
    score, threshold = max(rows, key=lambda row: row[0])
    return clone(pipe).fit(X, y), score, threshold


def predict_record(record, X):
    model = record["_model"]
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(X)[:, 1]
        predictions = (scores >= record.get("_threshold", 0.5)).astype(int)
    else:
        scores = model.decision_function(X)
        predictions = model.predict(X)
    return predictions, scores
