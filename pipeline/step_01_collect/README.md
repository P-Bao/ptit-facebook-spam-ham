# Bước 1: chuyển và đối soát dữ liệu đã thu

Crawler gốc được thành viên bàn giao nằm tại
`facebook_post_comment_scraper_code/facebook_post_comment_scraper/`. Thư mục này
không thực hiện yêu cầu Facebook; chỉ làm việc trên các file đã có.

- `convert_phase2_json.py`: tách từng comment/reply từ JSON thành một dòng CSV,
  bỏ text rỗng và text làm sạch trùng Phase 1; giữ metadata bài/comment type.
- `audit_source.py`: kiểm tra chuỗi 7 snapshot Drive, tái hiện chuyển/clean/gộp,
  đối chiếu cache và chênh nhãn tới v7; tùy chọn kiểm kê AST/hash mã crawler.
- `__init__.py`: cho phép gọi bằng `python -m` từ gốc repo.

```powershell
python -m pipeline.step_01_collect.convert_phase2_json --help
python -m pipeline.step_01_collect.audit_source --source-dir data/source_drive --training-csv data/labeled_v7.csv --crawler-dir facebook_post_comment_scraper_code/facebook_post_comment_scraper --output results/source_audit/audit.json
```

Tái hiện CSV cleaned lịch sử bằng cleaner với `--keep-punctuation-only` (xem
`python -m pipeline.step_02_preprocess.clean_data_facebook_spamham --help`). Mặc
định mới sẽ bỏ 42 dòng chỉ dấu câu/emoji, làm dịch ID vị trí: không áp cache cũ
vào CSV đó. Dùng snapshot đã khóa hash là cách an toàn khi chỉ cần kiểm tra nguồn.

Đối soát không tự đổi nhãn hoặc tạo ra lịch sử chỉnh v7. Chi tiết bằng chứng và
giới hạn: `docs/references/CRAWLER_VA_DU_LIEU_DRIVE.md`.
