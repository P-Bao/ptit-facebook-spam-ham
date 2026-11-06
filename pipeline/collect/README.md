# Thu thập và chuyển dữ liệu

| File | Nhiệm vụ | Đầu ra |
|---|---|---|
| `scrape_ptit_groups.py` | Runner PTIT gọi bộ nền ở `external/facebook_scraper/` | Raw JSON/JSONL, state, text/media tùy chọn |
| `json_to_csv.py` | Làm phẳng comment/reply và bỏ nội dung rỗng; có tùy chọn lọc trùng tham chiếu | CSV chưa có nhãn |
| `__init__.py` | Khai báo package để import | Không phải bước xử lý |

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

## Thứ tự sử dụng

1. **Khi lấy dữ liệu mới:** `run_scraper()` đọc `PTIT_GROUPS` và cấu hình phiên
   `FB_COOKIES`/`FB_DTSG` từ `.env`, gọi `fetch_posts` ở mã nền; callback
   `on_batch` gọi `fetch_comments_for_post`, sau đó `save_records` ghi bài kèm
   comment/reply. `on_page` cập nhật cursor/state. Chưa có nhãn spam/ham.
2. **Tạo bảng:** `flatten_posts()` giữ metadata bài/nhóm, tạo dòng cho comment
   và reply; bỏ nội dung rỗng rồi lưu CSV. Chọn output JSON dạng mảng nếu dùng
   converter, không đưa JSONL trực tiếp vào hàm đọc JSON mảng.
3. Chuyển CSV sang `pipeline.prepare.clean_for_labeling`, rồi `pipeline.label`.

CLI từ gốc project, ví dụ chuyển dữ liệu đã lưu giống Notebook 01:

```powershell
.\.venv\Scripts\python.exe -m pipeline.collect.json_to_csv --input .cache/collection/posts.json --keep-phase1-duplicates --output data/intermediate/comments.csv
```

Flag giữ trùng tham chiếu ở đây yêu cầu chuyển toàn bộ JSON đang minh họa;
Notebook 01 gọi trực tiếp `flatten_posts`, không lọc bằng CSV tham chiếu.
Crawler live cần kết nối/phiên hiện hành; lần rà soát offline không gọi lại
Facebook và không bảo đảm nguồn hiện tại cho cùng số mẫu lịch sử. Giới hạn
phân trang/resume xem [biên bản nguồn](../../docs/provenance/CRAWLER_VA_DU_LIEU_DRIVE.md).
