# Gợi ý và tiếp nhận nhãn Spam/Ham

| File | Nhiệm vụ |
|---|---|
| `ai_assisted_spam_ham.py` | Gọi API tương thích OpenAI qua OpenRouter/GLM, xử lý batch/đa luồng, lưu gợi ý `label` và `confidence` vào cache; có thể đưa lên Argilla |
| `upload_to_argilla.py` | Tạo/đọc dataset Argilla, khai báo hướng dẫn ham/spam và đưa bình luận lên giao diện |
| `auto_accept_labels.py` | Tiếp nhận gợi ý đủ ngưỡng, ghi `official_label`, `ai_confidence`, `is_auto_accepted`; có chế độ `--skip-argilla` |
| `download_labeled_data.py` | Lấy nhãn `submitted` từ Argilla và nối lại nội dung/metadata để xuất CSV |
| `cached_annotations.py` | `attach_annotation_ids()`: nối phản hồi đã lưu với bình luận theo `post_id/comment_type/text`, kiểm nội dung sạch khớp trước khi dùng gợi ý |
| `label_utils.py` | `record_ids`, `normalize_prediction`, `join_metadata`: kiểm ID, nhãn, confidence và nối metadata |
| `__init__.py` | Khai báo package |

**Notebook 01 mục 4–6:** đọc chỉ mục/phản hồi local → `attach_annotation_ids()` →
ghi `annotation_predictions.json` → gọi `auto_accept_labels` với ngưỡng **0,75**,
`--skip-argilla` → giữ dòng có `official_label` → xuất
`data/labeled/comments_labeled.csv`. Các cell này không gọi API mới. Bình luận
thiếu gợi ý hoặc chưa đạt ngưỡng không được tự coi là ham.

Khi gán nhãn cho dữ liệu mới, có hai cách:

1. **Không dùng giao diện:** AI với `--no-upload` → cache → `auto_accept_labels`
   với `--skip-argilla` → lọc nhãn đủ điều kiện. Các mẫu còn lại cần xử lý riêng.
2. **Có Argilla:** upload → AI tạo gợi ý → tiếp nhận theo ngưỡng hoặc duyệt trên
   giao diện → download nhãn đã submit. Giữ cùng ID/dataset/cache xuyên suốt.

Lệnh và cấu hình môi trường nằm trong [README gốc](../../README.md). `confidence`
ở đây do mô hình gán nhãn trả về, chưa được hiệu chỉnh thành độ tin cậy thực
nghiệm. **0,75 là ngưỡng tiếp nhận nhãn**, khác ngưỡng decision score S3
**−0,1776** và ngưỡng xác suất viBERT **0,5** khi phân loại.
