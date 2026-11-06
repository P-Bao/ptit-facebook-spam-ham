"""pipeline/modern/predict.py
Nạp viBERT đã fine-tune để phân loại bình luận mới.

Đầu vào: text gốc, thư mục save_pretrained và cấu hình đầu vào đã lưu.
Xử lý: chuẩn hóa nhẹ → tokenizer cùng model → softmax, ngưỡng 0,5.
Đầu ra: bảng text/probability_spam/du_doan; không huấn luyện lại.
Liên quan: modern.data; modern.train; notebook 03.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from pipeline.modern.data import normalize_text


class VibertPredictor:
    def __init__(self, model_dir, device=None):
        model_dir = Path(model_dir)
        self.settings = json.loads((model_dir / "inference_config.json").read_text(encoding="utf-8"))
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir, local_files_only=True)
        self.model.to(self.device).eval()
        if self.model.config.id2label != {0: "ham", 1: "spam"}:
            raise ValueError("Model không có ánh xạ nhãn ham/spam mong đợi.")

    @torch.inference_mode()
    def predict(self, texts, batch_size=16):
        texts = list(texts)
        probabilities = []
        for start in range(0, len(texts), batch_size):
            batch = [normalize_text(text) for text in texts[start:start + batch_size]]
            inputs = self.tokenizer(batch, padding=True, truncation=True,
                                    max_length=self.settings["max_length"], pad_to_multiple_of=8,
                                    return_tensors="pt")
            mixed = str(self.device).startswith("cuda") and self.settings.get("fp16", False)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=mixed):
                logits = self.model(**inputs.to(self.device)).logits
            probabilities.extend(torch.softmax(logits.float(), dim=-1)[:, 1].cpu().tolist())
        values = np.asarray(probabilities)
        return pd.DataFrame({"text": texts, "probability_spam": values,
                             "du_doan": np.where(values >= self.settings["threshold"], "spam", "ham")})
