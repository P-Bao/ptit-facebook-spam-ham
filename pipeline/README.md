# Các bước xử lý của project

`pipeline/` chứa mã dùng chung cho notebook và chương trình chạy từ Terminal.
Các file `__init__.py` đánh dấu package để import; hướng dẫn được đặt trong
README của từng thư mục để dễ mở và đọc.

| Thư mục | Đầu vào → nhiệm vụ → đầu ra | Nơi điều phối |
|---|---|---|
| [collect](collect/README.md) | Nhóm Facebook → lấy bài/comment/reply, làm phẳng JSON → bảng bình luận | Notebook 01, hoặc CLI thu thập |
| [prepare](prepare/README.md) | Bảng bình luận → làm sạch cho nhãn; v7 → token cho mô hình truyền thống | Notebook 01 và 02 |
| [label](label/README.md) | Bình luận sạch → gợi ý AI/duyệt nhãn → CSV ham/spam | Notebook 01 hoặc CLI/API |
| [modeling](modeling/README.md) | Văn bản/nhãn/fold → huấn luyện, CV, chiến lược ngưỡng/cân bằng → mô hình và chỉ số | Notebook 02 |
| [inference](inference/README.md) | Bình luận mới + bundle `.joblib` → nhãn và decision score | Notebook 02 |
| [modern](modern/README.md) | V7 → tokenizer và fine-tune viBERT → trọng số, test và dự đoán | Colab, Notebook 03 |
| [presentation](presentation/README.md) | DataFrame hoặc mã nguồn → bảng/khối code dễ đọc | Cả ba notebook |

Luồng chính: **01 chuẩn bị CSV → 02 thực nghiệm truyền thống trên v7 → 03
fine-tune/đánh giá viBERT trên cùng test**. CSV của Notebook 01 và v7 là hai đầu
vào có vai trò riêng; mở Notebook 02 không tự thay v7 bằng CSV vừa xuất.

`collect`, `clean_for_labeling`, các công cụ nhãn và `modern.train` có CLI.
Các file còn lại chủ yếu cung cấp hàm/class; chạy riêng một file thư viện không
tự thực hiện toàn bộ thí nghiệm. Xem [hướng dẫn chạy](../README.md) và
[tổng quan](../docs/guides/TONG_QUAN_PROJECT_VA_CACH_THUC_THI.md).
