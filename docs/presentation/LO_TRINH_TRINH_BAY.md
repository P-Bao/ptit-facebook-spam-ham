# Lộ trình mở code khi trình bày

## Notebook 01 — từ thu thập đến bảng nhãn

| Thứ tự | Mở file/hàm | Giải thích | Đề |
|---|---|---|---|
| 1 | `pipeline/collect/scrape_ptit_groups.py`: `PTIT_GROUPS`, `run_scraper` | Nguồn, cấu hình, runner | Thu thập |
| 2 | `external/facebook_scraper/group_post_scraper_v2.py`: `fetch_posts` | Lấy trang bài và metadata | Thu thập |
| 3 | `external/facebook_scraper/main.py`: `fetch_comments_for_post`; `comment_scraper.py` | Comment/reply của bài | Thu thập |
| 4 | Runner: `on_batch`, `SingleFileWriter.save_records` | Record JSON, state hỗ trợ tiếp tục; không chứng minh thu đủ | Thu thập |
| 5 | `pipeline/collect/json_to_csv.py`: `flatten_posts` | Mỗi comment/reply một dòng | Chuẩn bị dữ liệu |
| 6 | `pipeline/prepare/clean_for_labeling.py`: `preprocess`, `clean_facebook_comments` | Thêm cleaned_text, giữ raw, khử trùng/lọc | Ý 2 |
| 7 | `pipeline/label/upload_to_argilla.py`, `ai_assisted_spam_ham.py` | Upload và gợi ý label/confidence | Chuẩn bị nhãn |
| 8 | `auto_accept_labels.py`, `download_labeled_data.py`, `label_utils.py` | Chốt theo ngưỡng/người duyệt, export đúng ID | Chuẩn bị nhãn |

Notebook đọc snapshot thật, chuyển toàn bộ JSON thành CSV, chạy cleaner và CLI
chấp nhận nhãn từ cache; không gọi dịch vụ live. CSV mới và nguồn cache lịch sử
có phạm vi riêng được ghi rõ để không ghép nhầm ID. Mỗi bước hiện dữ liệu/file
đầu ra; các khối “Xem code” mở sau khi giải thích output.

Câu chuyển: “Phần đầu trình bày mã và snapshot của một đợt thu thập. Các thí nghiệm
tiếp theo dùng dataset PTIT phiên bản v7 do nhóm cung cấp.”

## Notebook 02 — từ v7 đến mô hình và demo

| Phần | File/hàm | Giải thích | Đề |
|---|---|---|---|
| Dataset | Cell nạp v7 | Revision/hash, số mẫu, kiểm tra nhãn/ID | Mô tả dữ liệu |
| Tiền xử lý | `pipeline/prepare/preprocess_for_model.py`: `preprocess` | Tách từ, giữ/bỏ stop words | Ý 2 |
| Chia tập | Notebook; `pipeline/modeling/training_utils.py` | Text trùng cùng nhóm, train/test và CV | Ý 4 |
| Đặc trưng | `pipeline/features/spam_features.py`; `experiments.make_pipe` | BoW/TF-IDF/FastText, fit trong train/fold | Ý 3 |
| Huấn luyện | `pipeline/modeling/experiments.py`: `make_pipe`, `fit_experiment` | NB/SVM, grid/CV, chọn trước test | Ý 4 |
| Mô hình hiện đại | Mục chờ notebook; `pipeline/modern/README.md` | Chưa có kết quả, triển khai sau | Ý 4 bổ sung |
| Đánh giá | `experiments.evaluate`, bảng/hình notebook | Accuracy/Precision/Recall/F1/Confusion Matrix, so sánh | Ý 5 |
| Phân tích thêm | Sampling/ngưỡng, AP/CI/OOV, FP/FN | Để sau phần bắt buộc | Cải thiện |
| Demo | `pipeline/inference/predict.py`: `predict_spam` | Raw → cùng preprocess → model đã fit → spam/ham | Ứng dụng |

`tools/dataset_versions/` dùng tổng hợp/kiểm chứng, không bắt buộc khi train v7.
Báo cáo dẫn nguồn và phụ lục; không dùng vị trí folder để suy diễn không có giới hạn dữ liệu.
