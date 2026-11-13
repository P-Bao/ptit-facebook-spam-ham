"""Giai đoạn 1c: huấn luyện và chọn tham số bằng CV trên train.

text_features.py tạo bước văn bản -> vector; classifiers.py tạo NB/SVM.
File này GHÉP và CHẠY hai bước đó. Chỉ số test được tính ở evaluation.py.
"""

import time

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from pipeline.modeling.classifiers import MODELS, make_classifier
from pipeline.modeling.evaluation import evaluate, boot_ci
from pipeline.modeling.text_features import FEATURES, make_vectorizer


CV_SCORING = {"precision": "precision", "recall": "recall", "f1": "f1"}


def make_pipe(feat, model_name, *, seed=42, cache=None, n_jobs=2, class_weight=None):
    """Ghép hai bước chưa fit; trả (pipeline, lưới tham số, số tiến trình).

    Pipeline.fit(text, y) tự gọi vectorizer.fit_transform(text), sau đó
    classifier.fit(vector, y). Pipeline.predict/decision_function trên câu
    mới tự gọi vectorizer.transform trước bộ phân loại.
    """
    # Hai lệnh sau chỉ tạo đối tượng; chưa học từ vựng hoặc trọng số phân loại.
    if feat not in FEATURES or model_name not in MODELS:
        raise ValueError("Đặc trưng/thuật toán không nằm trong cấu hình thực nghiệm")
    vectorizer, vector_grid = make_vectorizer(feat)
    classifier, model_grid = make_classifier(model_name, seed=seed,
                                              class_weight=class_weight)
    # "vec" và "clf" là tên hai bước. Chưa học dữ liệu tại dòng này.
    pipe = Pipeline([("vec", vectorizer), ("clf", classifier)], memory=cache)
    # Ví dụ "vec__ngram_range" đổi n-gram; "clf__C" đổi C của SVM.
    # Dấu __ chỉ nói rõ tham số thuộc bước nào trong pipeline.
    return pipe, {**vector_grid, **model_grid}, n_jobs


def fit_experiment(pre, feat, model_name, X, y, folds, *, seed=42, cache=None, n_jobs=2,
                   class_weight=None, strategy="Không điều chỉnh"):
    """Chạy một cặp đặc trưng + thuật toán trên các fold của train.

    X: mảng CHUỖI TOKEN của train, chưa phải vector; y: nhãn 0/1 của train.
    folds: năm cặp vị trí (học, validation) trong mảng X/y.
    Trả một record gồm trung bình điểm validation, tham số thắng và pipeline
    đã fit lại trên TOÀN train. Không nhận và không đọc test.
    """
    # X/y ở đây đã là *train*; folds chỉ chứa vị trí bên trong train.
    pipe, grid, jobs = make_pipe(feat, model_name, seed=seed, cache=cache, n_jobs=n_jobs,
                                 class_weight=class_weight)
    start = time.time()
    # GridSearchCV liệt kê từng tổ hợp n-gram và alpha/C. Với mỗi tổ hợp:
    # fit trên bốn fold train, dự đoán fold validation, tính P/R/F1 spam;
    # lặp năm lần rồi lấy điểm trung bình. refit="f1" chọn bằng F1 spam.
    search = GridSearchCV(pipe, grid, scoring=CV_SCORING, cv=folds, n_jobs=jobs,
                          refit="f1", error_score="raise")
    # Lời gọi fit này mới thật sự học dữ liệu. Sau CV, thư viện tự fit lại
    # tổ hợp thắng trên toàn bộ X/y train để dùng cho bước test.
    search.fit(X, y)
    best_index = search.best_index_
    cv = search.cv_results_
    # Trong tên của scikit-learn, mean_test_* ở đây là điểm VALIDATION của CV;
    # chưa phải điểm trên tập test độc lập của project.
    cv_precision = float(cv["mean_test_precision"][best_index])
    cv_recall = float(cv["mean_test_recall"][best_index])
    cv_f1 = float(cv["mean_test_f1"][best_index])
    # Record giữ cả điểm CV để so sánh và _model đã fit để chấm test về sau.
    return {"Tiền xử lý": pre, "Đặc trưng": feat, "Thuật toán": model_name,
            "Chiến lược": strategy,
            "CV Precision (spam)": cv_precision,
            "CV Recall (spam)": cv_recall,
            "CV F1 (spam)": cv_f1,
            "Tham số": search.best_params_, "Thời gian (s)": time.time()-start,
            # NB quyết định theo xác suất; SVM theo decision score có dấu.
            "_threshold": 0.5 if model_name == "Naive Bayes" else 0.0,
            "_model": search.best_estimator_}
