> Biên bản kiểm chứng nguồn theo phiên bản đã ghi dưới đây. Đường dẫn lưu dữ liệu hiện hành: `.cache/collection/` cho đầu vào tái lập, `data/` cho dữ liệu làm việc. Cách chạy mới nằm trong [README](../../README.md).

# Đối chiếu Kaggle và xác minh nguồn dữ liệu

Ghi chép đối chiếu ngày 04/11, trước lượt tái cấu trúc. Bản hiện tại gồm ba
notebook import module repo, viBERT đã hoàn tất; hướng dẫn mới ở `docs/guides/` và thay đổi ở
`docs/archive/changes/2026-11-05_TAI_CAU_TRUC_VA_DONG_BO.md`. Các so sánh phiên bản cũ
dưới đây được giữ làm lịch sử, không phải yêu cầu còn nhúng module vào notebook.

Ngày truy cập: 04/11/2026. Ghi nhận từ nội dung tải được; không sửa notebook Kaggle
hoặc nhắn tin cho thành viên.

## Notebook Cương

Nguồn: https://www.kaggle.com/code/itsmecuong/notebook82c07ebf7a

Đã tải source qua API public `https://www.kaggle.com/api/v1/kernels/pull/itsmecuong/notebook82c07ebf7a`.
Metadata: public, phiên bản 1, 32 cell, CPU; lastRunTime do Kaggle trả là
`2026-11-01T14:25:07.0175115Z`. Đây là metadata, không phải lần chạy mới của agent.

| Nội dung | Kaggle v1 | Repo trước sửa | Bản sau sửa |
|---|---|---|---|
| Đặc trưng | BoW, TF-IDF | BoW, TF-IDF | Giữ hai loại (BoW, TF-IDF) |
| Họ thuật toán | NB/SVM | NB/SVM | Giữ, tìm class weight bằng CV |
| Stop words | Hai biến thể, 8 tổ hợp | 8 tổ hợp | 8 tổ hợp, quyết định theo CV |
| Mất cân bằng | Baseline/balanced/ngưỡng/under-sampling | Đã có tương tự | Sửa CV under-sampling, lưu đúng ngưỡng |
| Che tên/emoji | Heuristic, tải từ điển | Kế thừa tương tự | Sửa che nhầm/emoji, dùng từ điển cố định |
| CSV local | Markdown có nói, code tải HF trực tiếp | Có cấu hình env/cache | Có kiểm tra cột/nhãn/ID/hash |
| Kết luận | Số liệu/nhận xét có chỗ không khớp; nói F1 là “trần”, balanced luôn tốt | Đã chỉnh một phần | Sinh nhận xét từ output thật, chọn bằng CV |
| Mô hình hiện đại | Chưa có | Chưa có | Mục chờ theo yêu cầu |

Không có kỹ thuật bắt buộc quan trọng ở Kaggle mà repo trước sửa hoàn toàn thiếu.
Tham khảo cách trình bày và lưu ý về nhiễu nhãn/ngữ cảnh; không chép số liệu/nhận xét
cũ. Các thí nghiệm mất cân bằng đã có trong repo, không ghi là vừa lấy mới từ Kaggle.

## Dataset đã xác minh

Nguồn: https://huggingface.co/datasets/p-storm/ptit-facebook-comments-spamham

API: https://huggingface.co/api/datasets/p-storm/ptit-facebook-comments-spamham

- Revision: `5b904713bb2896ab8f6ff336545b9debd2489724`.
- File: `labeled_v7.csv`, 5.780.794 byte.
- SHA-256: `7305d712996da3ff0dce1ea4142844b7629e9880bcac47ee38934f0b6e6cfbee`.
- API `gated=false`; tải CSV không token thành công ngày kiểm tra.
- 38.233 mẫu: ham 32.907, spam 5.326; 31.931 ID `cmt_`, 6.302 ID `p2_cmt_`.
- Không ID trùng, cleaned text rỗng hoặc nhãn ngoài spam/ham.
- 2.729 dòng thuộc nhóm text lặp theo lowercase; 27 nhóm text giống nhau có nhãn khác.
  Không tự sửa/xóa khi chưa có ngữ cảnh và xác nhận của người gán nhãn.

Card mô tả crawl group công khai PTIT, LLM/OpenRouter gợi ý, con người soát bằng
Argilla. Đây là thông tin bên công bố ghi. Card còn tên file/số mẫu cũ
(`labeled_spam_ham.csv`, ~32.000 và ~7.700); không dùng thay số đếm v7. Tên tác giả
trích dẫn trong card cần nhóm xác nhận trước khi đưa vào báo cáo.

## Ghi nhận ban đầu và bằng chứng bổ sung

**Cập nhật cùng ngày:** thành viên đã bàn giao mã crawler và thư mục Drive. Nguồn
mới được đối soát tại [CRAWLER_VA_DU_LIEU_DRIVE.md](CRAWLER_VA_DU_LIEU_DRIVE.md),
thay cho trạng thái chưa biết ở bảng lịch sử dưới đây. Đã xác định được 5 group,
timestamp và chuỗi xử lý của raw Phase 2. Chưa có raw/log Phase 1, phiên bản/lệnh
crawl chốt, tỷ lệ người duyệt và lịch sử 1.366 nhãn khác giữa Drive và v7.

Bảng dưới lưu thông tin **trước khi nhận bộ bàn giao**, không phải kết luận hiện tại:

| Thông tin | Nội dung đã được xác nhận |
|---|---|
| Tên/link group | |
| Khoảng thời gian thu thập | |
| Công cụ/phiên bản crawl thực dùng | |
| Số bài đăng và cách lấy comments/replies | |
| Người/quy tắc rà nhãn, tỷ lệ soát tay | |
| Lịch sử cập nhật nhãn tới v7 | |

Người dùng xác nhận hiện chưa biết, sẽ hỏi nhóm nếu tìm được. Không tự điền hoặc yêu
cầu làm lại dữ liệu chỉ để lấp chỗ trống. Nếu không có thì giữ ghi chú chưa xác nhận
hoặc lược bỏ chi tiết không có căn cứ khi trình bày.

README cũ dẫn `https://github.com/mohdtalal3/facebook_post_comment_scraper` ở phần
tham khảo; tại thời điểm kiểm tra ban đầu chưa có mã bàn giao để đối chiếu. Có thể hỏi người phụ
trách thu thập (phân công ghi Bảo) và người chốt nhãn v7; phân công không phải log
chứng minh ai đã chạy công cụ nào.
