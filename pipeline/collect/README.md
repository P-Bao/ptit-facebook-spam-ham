# Thu thập và chuyển dữ liệu

| File | Nhiệm vụ | Đầu ra |
|---|---|---|
| `scrape_ptit_groups.py` | Runner PTIT gọi module nền ở `external/facebook_scraper/` | Raw JSON/JSONL, state, text/media tùy chọn |
| `json_to_csv.py` | Làm phẳng, bỏ rỗng, tùy chọn loại trùng tham chiếu | CSV chưa có label |
| `snapshots.py` | Tìm/tải đầu vào theo manifest, kiểm hash, cung cấp các tên logic cho notebook | Đường dẫn file đúng hash |

Từ gốc repo gọi `python -m pipeline.collect.<tên_module>`. Runner live cần
`requirements/collection.txt` và phiên hợp lệ trong `.env` ở gốc repo. Notebook 01
chỉ đọc snapshot, không cần cookie. Giới hạn crawler xem `docs/provenance/`.

Mặc định runner ghi `data/raw/posts.json`; converter đọc đúng file đó và ghi
`data/intermediate/comments.csv`. Cleaner tạo `data/intermediate/comments_cleaned.csv`,
sau đó các công cụ `pipeline.label` quản lý/chốt/export nhãn.

Notebook 01 đọc đầu vào qua `collection_inputs()`, gọi các hàm chuyển bảng/làm
sạch rồi liên kết gợi ý bằng `pipeline/label/cached_annotations.py`. Khóa nội dung
và metadata giữ gợi ý đúng bình luận khi thứ tự dòng thay đổi; văn bản làm sạch
không khớp sẽ bị từ chối. Kết quả có nhãn được xuất thành `comments_labeled.csv`.

Các tùy chọn CLI bổ sung được mô tả trong `--help`. Luồng project, cách chạy và
đầu ra nằm trong `docs/guides/TONG_QUAN_PROJECT_VA_CACH_THUC_THI.md`.
