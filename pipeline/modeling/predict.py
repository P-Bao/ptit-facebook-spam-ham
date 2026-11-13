"""Giai đoạn 4: dự đoán bình luận mới bằng mô hình truyền thống đã học.

Notebook 02 hoặc scripts/run_phi_pipeline.py gọi joblib.dump để tạo .joblib.
File này chỉ nhận bundle đã nạp, áp cùng tiền xử lý và ngưỡng để dự đoán;
không huấn luyện, lưu file, tạo lại từ vựng/IDF hay gọi AI gán nhãn.
"""

import numpy as np
import pandas as pd

from pipeline.modeling.evaluation import predict_record
from pipeline.prepare.preprocess_for_model import preprocess


def predict_spam(texts, model_bundle):
    """Nhận một câu hoặc danh sách câu thô; trả bảng nhãn/điểm từng câu.

    model_bundle chứa: model đã fit, threshold, remove_stop.
    Bảng trả về: text gốc, score, loại score, nhãn dự đoán spam/ham.
    """
    # Một chuỗi là MỘT bình luận; list(texts) trực tiếp sẽ tách từng ký tự.
    texts = [texts] if isinstance(texts, str) else list(texts)
    if not texts:
        return pd.DataFrame(columns=["text", "score", "score_type", "du_doan"])
    # Dùng cùng cách chuẩn bị câu và lựa chọn từ dừng của lúc train.
    X = [preprocess(text, text, model_bundle["remove_stop"]) for text in texts]
    # Model trong bundle gồm cả vectorizer đã học: X vẫn là CHUỖI token.
    # predict_record gọi pipeline để biến X thành số, lấy score, áp ngưỡng.
    record = {"_model": model_bundle["model"]}
    if "threshold" in model_bundle:
        record["_threshold"] = model_bundle["threshold"]
    pred, score = predict_record(record, X)
    # NB: score là xác suất ước lượng; SVM: score là điểm có dấu, không phải
    # xác suất. Nhãn 1 đổi thành "spam", nhãn 0 thành "ham" để dễ đọc.
    return pd.DataFrame({"text": texts, "score": score,
        "score_type": "probability_spam" if hasattr(model_bundle["model"], "predict_proba") else "decision_score",
        "du_doan": np.where(pred == 1, "spam", "ham")})
