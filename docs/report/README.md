# Báo cáo học phần

Hai bản báo cáo hiện hành:

- [DOCX có thể chỉnh sửa](BAO_CAO_PTIT_SPAM_HAM.docx)
- [PDF để đọc và gửi](BAO_CAO_PTIT_SPAM_HAM.pdf)

**Lưu ý đường dẫn khi nộp:** hai bản báo cáo được xuất trước khi chuyển hàm dự đoán từ `pipeline/inference/predict.py` sang [`pipeline/modeling/predict.py`](../../pipeline/modeling/predict.py). Trong DOCX còn một chỗ ghi đường dẫn cũ; cần sửa chỗ đó và xuất lại PDF khi chốt báo cáo. Các hướng dẫn Markdown và Notebook 02 đã dùng đường dẫn mới.

**Đối chiếu trước khi nộp:** BoW/TF-IDF hiện dùng `CountVectorizer`/`TfidfVectorizer` đúng như phần mô tả trong báo cáo. Notebook 02 đã được chạy lại sau thay đổi này; các điểm test B1–B4/S1–S3 giữ nguyên đến bốn chữ số thập phân, một vài giá trị CV/AP có sai khác số học rất nhỏ. Nếu nhóm sửa thêm mô hình hoặc dữ liệu, cần đối chiếu lại bảng báo cáo với `artifacts/spamham/metrics.csv` và xuất lại PDF từ DOCX.

Báo cáo bám [ảnh đề](../assignment/de_bai.png), gồm Mở đầu, Chương 1 cơ sở lý
thuyết, Chương 2 thực nghiệm/kết quả, Kết luận và 24 tài liệu tham khảo. Phần
thực nghiệm đối chiếu Notebook 01–03 và các CSV/hình trong `artifacts/`;
hai nhánh S3 và viBERT M1 dùng cùng 7.474 mẫu test.

**Trước khi nộp:** điền họ tên đầy đủ, MSSV, lớp/nhóm và giảng viên vào trang
bìa. Hiện project chỉ có tên gọi ngắn Cương, Bảo, Phi, Phương. Nếu sửa DOCX,
cần cập nhật mục lục/danh mục bảng/hình trong Word và xuất lại PDF để hai bản
khớp với thông tin bìa cuối cùng.

[Mục lục và kế hoạch biên soạn](MUC_LUC_VA_KE_HOACH_BAO_CAO.md) là file làm việc
tạm; giữ lại đến khi nhóm duyệt báo cáo cuối, sau đó mới xóa.
