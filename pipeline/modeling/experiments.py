"""pipeline/modeling/experiments.py
Định nghĩa pipeline đặc trưng/mô hình và các phép đo thực nghiệm.

Đầu vào: văn bản đã tách từ, nhãn train, folds và cấu hình seed/cache/jobs.
Xử lý: BoW/TF-IDF + NB/SVM; chọn bằng CV train, đánh giá riêng trên test.
Đầu ra: record mô hình/tham số/chỉ số; bootstrap F1 là phân tích bổ sung.
Liên quan: modeling.training_utils; notebook 02.
"""

import time

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from pipeline.modeling.training_utils import predict_record

TOKEN_PATTERN = r"(?u)\b\w+\b|[^\w\s]"
NGRAMS = [(1, 1), (1, 2)]
FEATURES = ("BoW", "TF-IDF")
MODELS = ("Naive Bayes", "SVM")
CV_SCORING = {"precision": "precision", "recall": "recall", "f1": "f1"}


def make_pipe(feat, model_name, *, seed=42, cache=None, n_jobs=2, class_weight=None):
    """Tạo cấu hình cơ sở; bật balanced riêng cho đối chứng TF-IDF + SVM."""
    if feat not in FEATURES or model_name not in MODELS:
        raise ValueError("Đặc trưng/thuật toán không nằm trong cấu hình thực nghiệm")
    vectorizer = (CountVectorizer(token_pattern=TOKEN_PATTERN, min_df=2) if feat == "BoW"
                  else TfidfVectorizer(token_pattern=TOKEN_PATTERN, min_df=2, sublinear_tf=True))
    vector_grid = {"vec__ngram_range": NGRAMS}
    if model_name == "Naive Bayes":
        steps = [("clf", MultinomialNB())]
        model_grid = {"clf__alpha": [0.1, 0.3, 1.0]}
    else:
        steps = [("clf", LinearSVC(dual=False, max_iter=20000, random_state=seed))]
        model_grid = {"clf__C": [0.1, 0.3, 1.0, 3.0], "clf__class_weight": [class_weight]}
    jobs = n_jobs
    return Pipeline([("vec", vectorizer), *steps], memory=cache), {**vector_grid, **model_grid}, jobs


def fit_experiment(pre, feat, model_name, X, y, folds, *, seed=42, cache=None, n_jobs=2,
                   class_weight=None, strategy="Không điều chỉnh"):
    pipe, grid, jobs = make_pipe(feat, model_name, seed=seed, cache=cache, n_jobs=n_jobs,
                                 class_weight=class_weight)
    start = time.time()
    search = GridSearchCV(pipe, grid, scoring=CV_SCORING, cv=folds, n_jobs=jobs,
                          refit="f1", error_score="raise")
    search.fit(X, y)
    return {"Tiền xử lý": pre, "Đặc trưng": feat, "Thuật toán": model_name,
            "Chiến lược": strategy,
            **{f"CV {name} (spam)": float(search.cv_results_[f"mean_test_{key}"][search.best_index_])
               for key, name in [("precision", "Precision"), ("recall", "Recall"), ("f1", "F1")]},
            "Tham số": search.best_params_, "Thời gian (s)": time.time()-start,
            "_threshold": 0.5 if model_name == "Naive Bayes" else 0.0,
            "_model": search.best_estimator_}


def evaluate(record, X, y):
    """Chỉ predict trên test; không fit lại vectorizer/model hoặc chọn tham số."""
    pred, score = predict_record(record, X)
    record.update({"Accuracy": accuracy_score(y, pred),
        "Precision": precision_score(y, pred, zero_division=0),
        "Recall": recall_score(y, pred, zero_division=0),
        "F1": f1_score(y, pred, zero_division=0), "AP": average_precision_score(y, score),
        "_pred": pred, "_score": score})


def boot_ci(y, pred, n=500, seed=0):
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n):
        indices = rng.integers(0, len(y), len(y))
        values.append(f1_score(y[indices], pred[indices], zero_division=0))
    return np.percentile(values, [2.5, 97.5])
