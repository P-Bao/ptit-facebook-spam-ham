"""Tạo BoW hoặc TF-IDF bằng hai lớp có sẵn của scikit-learn.

make_vectorizer chỉ tạo đối tượng và lưới n-gram. Việc học từ vựng/IDF diễn ra
khi Pipeline được fit trong experiments.py, riêng trên train của từng fold.
"""

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


# Giữ cả từ một ký tự và dấu riêng lẻ, đúng quy tắc thí nghiệm đã chạy.
TOKEN_PATTERN = r"(?u)\b\w+\b|[^\w\s]"
NGRAMS = [(1, 1), (1, 2)]  # chỉ từ đơn hoặc từ đơn + cặp từ liên tiếp
FEATURES = ("BoW", "TF-IDF")
MIN_DOCUMENT_FREQUENCY = 2


def make_vectorizer(feature):
    """Trả (bộ tạo vector CHƯA fit, lưới n-gram để CV thử).

    Đầu vào: "BoW" hoặc "TF-IDF". Cả hai dùng cùng quy tắc tách từ,
    loại đặc trưng chỉ gặp trong một câu train và trả ma trận thưa.
    """
    common = {
        "token_pattern": TOKEN_PATTERN,
        "min_df": MIN_DOCUMENT_FREQUENCY,
        "ngram_range": (1, 1),  # giá trị ban đầu; GridSearchCV thử thêm (1, 2)
        "lowercase": True,
        "dtype": np.float64,
    }
    if feature == "BoW":
        vectorizer = CountVectorizer(**common)
    elif feature == "TF-IDF":
        vectorizer = TfidfVectorizer(
            **common, sublinear_tf=True, smooth_idf=True, norm="l2"
        )
    else:
        raise ValueError(f"Đặc trưng không hỗ trợ: {feature}")
    return vectorizer, {"vec__ngram_range": NGRAMS}
