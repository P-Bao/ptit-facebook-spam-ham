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
from sklearn.metrics import f1_score, precision_score, recall_score
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
    """Trả model, chỉ số CV, tham số và số mẫu; chỉ lấy mẫu phần train."""
    rows = []
    for params in ParameterGrid(grid):
        predictions = np.zeros(len(y), dtype=int)
        for train, valid in folds:
            selected = sample_training_indices(y, train, seed)
            assert not set(selected) & set(valid)
            model = clone(pipe).set_params(**params).fit(X[selected], y[selected])
            predictions[valid] = model.predict(X[valid])
        rows.append((mean_cv_scores(y, predictions, folds), params))
    scores, params = max(rows, key=lambda row: row[0]["CV F1 (spam)"])
    selected = sample_training_indices(y, np.arange(len(y)), seed)
    fitted = clone(pipe).set_params(**params).fit(X[selected], y[selected])
    return fitted, scores, params, len(selected)


def mean_cv_scores(y, predictions, folds):
    """Trung bình chỉ số trên các phần validation, cùng cách tổng hợp GridSearchCV."""
    return {f"CV {name} (spam)": float(np.mean([
        metric(y[valid], predictions[valid], zero_division=0) for _, valid in folds]))
        for name, metric in [("Precision", precision_score), ("Recall", recall_score), ("F1", f1_score)]}


def tune_svm_threshold(pipe, X, y, folds):
    """Chọn ngưỡng decision score từ OOF của train; không nhận dữ liệu test.

    Giữ siêu tham số SVM đã chọn bằng CV. Khảo sát 199 phân vị 1–99%,
    ngưỡng 0 và hai biên; đồng điểm F1 ưu tiên ngưỡng gần 0 hơn.
    CV ở đây là điểm phục vụ lựa chọn, không phải ước lượng nested CV độc lập.
    """
    scores = np.zeros(len(y))
    covered = np.zeros(len(y), dtype=int)
    for train, valid in folds:
        model = clone(pipe).fit(X[train], y[train])
        scores[valid] = model.decision_function(X[valid])
        covered[valid] += 1
    assert (covered == 1).all()
    thresholds = np.unique(np.r_[0.0, np.quantile(scores, np.linspace(0.01, 0.99, 199)),
                                  np.nextafter(scores.min(), -np.inf), scores.max()])
    rows = [(mean_cv_scores(y, (scores > threshold).astype(int), folds), float(threshold))
            for threshold in thresholds]
    metrics, threshold = max(rows, key=lambda row: (row[0]["CV F1 (spam)"], -abs(row[1])))
    return clone(pipe).fit(X, y), metrics, threshold


def predict_record(record, X):
    model = record["_model"]
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(X)[:, 1]
        predictions = (scores >= record.get("_threshold", 0.5)).astype(int)
    else:
        scores = model.decision_function(X)
        predictions = (scores > record.get("_threshold", 0.0)).astype(int)
    return predictions, scores
