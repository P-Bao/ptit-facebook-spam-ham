# Thu thập và chuyển dữ liệu

| File | Nhiệm vụ | Đầu ra |
|---|---|---|
| `scrape_ptit_groups.py` | Runner PTIT gọi module nền ở `external/facebook_scraper/` | Raw JSON/JSONL, state, text/media tùy chọn |
| `json_to_csv.py` | Làm phẳng, bỏ rỗng, tùy chọn loại trùng tham chiếu | CSV chưa có label |
| `snapshots.py` | Đọc/tải snapshot Drive đã chốt hash, hỗ trợ folder flat/phân tầng | Đường dẫn file đúng hash |

Từ gốc repo gọi `python -m pipeline.collect.<tên_module>`. Runner live cần
`requirements/collection.txt` và phiên hợp lệ trong `.env` ở gốc repo. Notebook 01
chỉ đọc snapshot, không cần cookie. Giới hạn crawler xem `docs/provenance/`.

Mặc định runner ghi `data/raw/posts.json`; converter đọc đúng file đó và ghi
`data/intermediate/comments.csv`. Cleaner tạo `data/intermediate/comments_cleaned.csv`,
sau đó các công cụ `pipeline.label` quản lý/chốt/export nhãn.

Converter có lọc trùng với CSV tham chiếu qua `--phase1-csv`; dùng
`--keep-phase1-duplicates` nếu không cần bước này. Không dùng cache ID cũ sau khi
đổi số/thứ tự dòng. Tái hiện clean lịch sử có `--keep-punctuation-only`. Ví dụ vài
bài trong notebook không được nhận là toàn bộ CSV lịch sử. Audit ở
`tools/dataset_versions/audit_source.py`.
