# Dự đoán bằng mô hình truyền thống đã lưu

| File | Nhiệm vụ |
|---|---|
| `predict.py` | `predict_spam(texts, model_bundle)`: chuẩn bị bình luận mới, gọi pipeline đã học và trả DataFrame |
| `__init__.py` | Khai báo package |

Thứ tự: **`joblib.load()` bundle → `preprocess()` theo cấu hình `remove_stop`
→ vectorizer đã fit → score → ngưỡng trong bundle → ham/spam**. Đầu ra gồm
`text`, `score`, `score_type`, `du_doan`. S3 trả decision score, không phải xác
suất; ngưỡng là −0,1776, không phải 0,5.

Notebook 02 mục 8 lưu và nạp lại
`artifacts/spamham/spam_ham_ml_only.joblib`, kiểm dự đoán rồi demo. File `.py`
chứa cách thực hiện; `.joblib` mới chứa pipeline đã học và cấu hình ngưỡng.
Không huấn luyện lại hoặc gọi LLM khi dự đoán.

File này là thư viện, không có CLI. Ví dụ gọi từ Python/Terminal có trong
[README gốc](../../README.md). Để dự đoán viBERT, dùng
`pipeline.modern.predict.VibertPredictor` với thư mục `model/`, không nạp bằng
joblib.
