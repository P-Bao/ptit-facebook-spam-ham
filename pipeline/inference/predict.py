"""pipeline/inference/predict.py
Dự đoán spam/ham cho bình luận mới bằng bundle đã lưu.

Đầu vào: list raw text và bundle model/threshold/remove_stop đã train.
Xử lý: cùng tiền xử lý với train → transform bằng vectorizer đã fit → predict.
Đầu ra: bảng text, score, score_type, du_doan; không train/gọi AI gán nhãn.
Liên quan: prepare.preprocess_for_model; modeling.training_utils; notebook 02.
"""

import numpy as np
import pandas as pd

from pipeline.modeling.training_utils import predict_record
from pipeline.prepare.preprocess_for_model import preprocess


def predict_spam(texts, model_bundle):
    texts = list(texts)
    if not texts:
        return pd.DataFrame(columns=["text", "score", "score_type", "du_doan"])
    X = [preprocess(text, text, model_bundle["remove_stop"]) for text in texts]
    record = {"_model": model_bundle["model"]}
    if "threshold" in model_bundle:
        record["_threshold"] = model_bundle["threshold"]
    pred, score = predict_record(record, X)
    return pd.DataFrame({"text": texts, "score": score,
        "score_type": "probability_spam" if hasattr(model_bundle["model"], "predict_proba") else "decision_score",
        "du_doan": np.where(pred == 1, "spam", "ham")})
