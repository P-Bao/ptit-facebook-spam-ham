"""Giai đoạn 0 của Phi: chia train/test rồi tạo các lượt train/validation.

Notebook 02 và scripts/run_phi_pipeline.py dùng cùng các hàm ở đây. Bản CSV
đầu vào đã có nhãn và chuỗi token; file này chỉ quyết định *dòng nào* thuộc
train/test/fold, không tạo token hay học mô hình.
"""

import hashlib
import re
import unicodedata

import numpy as np


def norm_key(text):
    """Đưa hai bình luận gần như trùng văn bản về cùng khóa để chia cùng phía.

    Ví dụ: chữ hoa/thường, khoảng trắng và các chữ số được chuẩn hóa. Đây là
    khóa chia tập, không phải câu dùng trực tiếp để huấn luyện mô hình.
    """
    # NFC ghép các cách viết Unicode tương đương; lower bỏ khác biệt hoa/thường.
    normalized = unicodedata.normalize("NFC", str(text)).lower()
    # Gom nhiều dấu cách thành một và bỏ khoảng trắng ở hai đầu.
    normalized = re.sub(r"\s+", " ", normalized).strip()
    # Đưa mọi chuỗi số về 0 để hạn chế cùng câu khác một con số bị tách tập.
    key = re.sub(r"\d+", "0", normalized)
    # Bỏ dấu câu trong khóa, nhưng vẫn giữ chữ và khoảng trắng.
    key = re.sub(r"[^\w\s]", " ", key)
    key = re.sub(r"\s+", " ", key).strip()
    # Nếu chỉ toàn ký hiệu, dùng chính chuỗi đã chuẩn hóa để tránh khóa rỗng.
    return key or "__symbols__|" + normalized


def bucket(key, salt, n=5):
    """Gán cùng một nội dung vào cùng nhóm, ổn định qua nhiều lần chạy.

    salt chỉ là tên mục đích "test"/"cv" để hai lần chia độc lập. MD5 ở đây
    KHÔNG mã hóa bình luận hay bảo mật dữ liệu; nó chỉ cho ra số nhóm cố định.
    Giữ quy tắc này để train/test của mô hình truyền thống và viBERT trùng nhau.
    """
    digest = hashlib.md5(f"{salt}|{key}".encode("utf-8")).hexdigest()
    return int(digest, 16) % n


def split_train_test(cleaned_texts, labels):
    """Chia toàn bộ CSV: khoảng 4/5 train, 1/5 test theo nhóm nội dung.

    Đầu vào: văn bản đã làm sạch và nhãn số cùng thứ tự từng dòng.
    Đầu ra: khóa nội dung, vị trí train, vị trí test trong TOÀN BỘ CSV.
    """
    keys = np.array([norm_key(text) for text in cleaned_texts])
    labels = list(labels)  # Chỉ kiểm hai lớp; chưa dùng nhãn để quyết định chia.
    if len(keys) != len(labels):
        raise ValueError("Văn bản và nhãn phải có cùng số dòng")

    # Dùng vị trí dòng để sau này lấy đúng văn bản và nhãn ở CSV.
    train_indices = np.array([i for i, key in enumerate(keys)
                              if bucket(key, "test") != 0], dtype=int)
    test_indices = np.array([i for i, key in enumerate(keys)
                             if bucket(key, "test") == 0], dtype=int)
    train_labels = {labels[i] for i in train_indices}
    test_labels = {labels[i] for i in test_indices}
    if train_labels != {0, 1} or test_labels != {0, 1}:
        raise ValueError("Train/test phải có cả ham=0 và spam=1")
    # Một bình luận trùng khóa không được xuất hiện ở cả train và test.
    assert not set(keys[train_indices]) & set(keys[test_indices])
    return keys, train_indices, test_indices


def grouped_folds(keys, y, n_splits=5):
    """Tạo năm cặp (vị trí học, vị trí kiểm tra) CHỈ bên trong train.

    keys/y ở đây đã lấy theo train. Mỗi vị trí trả về tính từ 0 trong train,
    khác với vị trí của dòng trong toàn bộ CSV.
    """
    keys = list(keys)
    labels = list(y)
    if len(keys) != len(labels):
        raise ValueError("Khóa và nhãn train phải có cùng số dòng")
    # Cùng khóa luôn có cùng số fold; "cv" tách mục đích này khỏi chia test.
    fold_ids = [bucket(key, "cv", n_splits) for key in keys]
    folds = []
    for fold_id in range(n_splits):
        # Vị trí thuộc fold hiện tại là validation; phần còn lại để học.
        train = np.array([i for i, value in enumerate(fold_ids)
                          if value != fold_id], dtype=int)
        valid = np.array([i for i, value in enumerate(fold_ids)
                          if value == fold_id], dtype=int)
        # set lấy các nhãn khác nhau; cả hai phía phải có {ham=0, spam=1}.
        train_labels = {labels[i] for i in train}
        valid_labels = {labels[i] for i in valid}
        if train_labels != {0, 1} or valid_labels != {0, 1}:
            raise ValueError("Fold thiếu một lớp; cần kiểm tra nhóm dữ liệu hoặc số fold.")
        assert not {keys[i] for i in train} & {keys[i] for i in valid}
        folds.append((train, valid))
    return folds


# Tương thích với các import cũ; phần Phi đọc code ở file theo giai đoạn.
from pipeline.modeling.imbalance import (sample_training_indices, undersampled_search,
                                         mean_cv_scores, tune_svm_threshold)
from pipeline.modeling.evaluation import predict_record
