# Thu thập và chuyển dữ liệu

| File | Nhiệm vụ | Đầu ra |
|---|---|---|
| `scrape_ptit_groups.py` | Runner PTIT gọi bộ nền ở `external/facebook_scraper/` | Raw JSON/JSONL, state, text/media tùy chọn |
| `json_to_csv.py` | Làm phẳng comment/reply và bỏ nội dung rỗng; có tùy chọn lọc trùng tham chiếu | CSV chưa có nhãn |

Runner thu thập cần phiên Facebook và `requirements/collection.txt`. Mặc định
ghi `data/raw/posts.json`; converter tạo `data/intermediate/comments.csv`.

Notebook 01 đọc trực tiếp ba file local trong `.cache/collection/`:
`posts.json`, `annotation_index.csv` và `annotation_predictions.json`.
Không có bước tải dữ liệu hoặc kiểm checksum. Mã chuyển bảng/làm sạch được gọi
theo từng cell; gợi ý nhãn được liên kết theo ID/nội dung bằng
`pipeline/label/cached_annotations.py` trước khi tiếp nhận.

CSV cuối là `data/labeled/comments_labeled.csv`. Cách chạy và các đầu vào khác
xem [README](../../README.md); file có sẵn để tool gộp dùng là
`data/labeled/comments_labeled_phase1.csv`.
