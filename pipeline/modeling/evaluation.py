"""Giai đoạn 3: dự đoán và đo kết quả trên tập test đã giữ riêng.

Không hàm nào ở đây fit hoặc chọn lại cấu hình dựa vào test.
"""

import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             precision_score, recall_score)


def predict_record(record, X):
    """Trả (nhãn dự đoán, điểm liên tục) cho các câu trong X.

    X là chuỗi token; pipeline trong record tự biến chúng thành vector.
    record["_model"] là pipeline đã fit; record["_threshold"] là mốc quyết định.
    NB dùng xác suất spam (mặc định >=0,5); SVM dùng decision score (>0).
    """
    model = record["_model"]
    if hasattr(model, "predict_proba"):
        # Naive Bayes có predict_proba: lấy cột thứ 2, lớp 1 = spam.
        scores = model.predict_proba(X)[:, 1]
        threshold = record.get("_threshold", 0.5)
        predictions = (scores >= threshold).astype(int)
    else:
        # LinearSVC không có predict_proba: score âm/dương KHÔNG là xác suất.
        # S3 có thể đổi ngưỡng 0 sang một giá trị khác đã chọn trên train.
        scores = model.decision_function(X)
        threshold = record.get("_threshold", 0.0)
        predictions = (scores > threshold).astype(int)
    return predictions, scores


def evaluate(record, X, y):
    """Gắn chỉ số test vào một record đã có model/điểm CV.

    X: chuỗi token của test; y: nhãn thật tương ứng (ham=0, spam=1).
    Record có cấu hình, điểm CV, model đã fit và ngưỡng. Hàm thêm các điểm
    test bên dưới, không biến điểm CV thành điểm test và không chọn lại model.
    """
    pred, score = predict_record(record, X)
    # Accuracy: số đoán đúng / tổng số test; dễ gây lạc quan khi ham nhiều.
    record["Accuracy"] = accuracy_score(y, pred)
    # Ba chỉ số sau chỉ xét lớp dương 1=spam; zero_division=0 khi không
    # dự đoán ra spam nào, tránh phép chia cho 0.
    record["Precision"] = precision_score(y, pred, zero_division=0)
    record["Recall"] = recall_score(y, pred, zero_division=0)
    record["F1"] = f1_score(y, pred, zero_division=0)
    # AP dùng thứ tự các score trên nhiều ngưỡng, không dùng nhãn tại một ngưỡng.
    record["AP"] = average_precision_score(y, score)
    # Giữ nhãn và score từng dòng để vẽ ma trận, đường PR, phân tích lỗi.
    record["_pred"] = pred
    record["_score"] = score


def boot_ci(y, pred, n=500, seed=0):
    """Ước lượng khoảng tham khảo 95% cho F1 của test đã có.

    Mỗi lượt rút len(y) dòng CÓ hoàn lại từ test, tính F1; lặp n=500 lần.
    Không huấn luyện lại và không chứng minh chất lượng ở nguồn dữ liệu mới.
    """
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n):
        # Rút lại len(y) vị trí có hoàn lại; một dòng có thể xuất hiện nhiều lần.
        indices = rng.integers(0, len(y), len(y))
        values.append(f1_score(y[indices], pred[indices], zero_division=0))
    # Lấy hai phân vị biên của 500 điểm F1 đã rút mẫu.
    return np.percentile(values, [2.5, 97.5])
