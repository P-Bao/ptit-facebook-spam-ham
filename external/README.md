# Bộ crawler bàn giao

`facebook_scraper/` giữ nguyên 7 module Python và các tài liệu/cấu hình của bộ
thành viên cung cấp. Runner PTIT chuyển ra `pipeline/collect/scrape_ptit_groups.py`.
Từ gốc repo gọi `python -m pipeline.collect.scrape_ptit_groups --help`.

README bên trong là tài liệu snapshot cũ: ví dụ đặt runner cùng folder không phải
đường dẫn hiện tại. `.env` cho runner mới ở gốc project. Không nâng cấp thư viện
nền lên main upstream trong lượt tái cấu trúc.

Nguồn/mức thay đổi: `docs/provenance/CRAWLER_VA_DU_LIEU_DRIVE.md` và
`results/source_audit/upstream_comparison.json`. Giữ tham chiếu upstream, không
gọi toàn bộ bộ nền là mã nhóm tự viết.
