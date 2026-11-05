# Lộ trình trình bày mã nguồn

## Notebook 01 — Thu thập và chuẩn bị dữ liệu

| Phần | Mã nguồn để mở | Nội dung giải thích |
|---|---|---|
| Nguồn và bài viết | `pipeline/collect/scrape_ptit_groups.py`, `external/facebook_scraper/group_post_scraper_v2.py` | Cấu hình nhóm, lấy bài và metadata. |
| Bình luận và lưu trữ | `external/facebook_scraper/main.py`, `comment_scraper.py`; runner: `on_batch`, `save_records` | Lấy comment/reply, ghi cấu trúc JSON. |
| Chuẩn hóa bảng | `pipeline/collect/json_to_csv.py`: `flatten_posts` | Mỗi comment/reply thành một dòng. |
| Làm sạch | `pipeline/prepare/clean_for_labeling.py` | Chuẩn hóa, lọc/khử trùng, giữ nội dung gốc. |
| Gán nhãn | `pipeline/label/ai_assisted_spam_ham.py`, `cached_annotations.py`, `auto_accept_labels.py` | Gợi ý, liên kết đúng ID và áp ngưỡng tiếp nhận. |
| Xuất CSV | Cell xuất dữ liệu | Các cột, số mẫu, tỷ lệ nhãn và `data/labeled/comments_labeled.csv`. |

Notebook tái lập xử lý từ dữ liệu/phản hồi đã lưu. Giải thích output trước,
mở khối mã nguồn tương ứng khi cần phân tích cách thực hiện.

Khi cần tổng hợp dữ liệu, chạy `tools/dataset_versions/merge_labeled_data.py`
theo hướng dẫn README; file kết quả nằm ở `data/final/comments_merged.csv`.

## Notebook 02 — Thực nghiệm và ứng dụng

| Phần | Mã nguồn để mở | Nội dung giải thích |
|---|---|---|
| Dữ liệu | Cell nạp v7 | Đọc CSV local, số mẫu, phân bố nhãn và bình luận thật từ file. |
| Tiền xử lý | `pipeline/prepare/preprocess_for_model.py` | Tiền xử lý toàn bộ v7, so sánh cùng ID, lưu/đọc lại CSV token cho mô hình. |
| Thiết kế thực nghiệm | Cell chia tập; `pipeline/modeling/training_utils.py` | Nhóm văn bản, train/test và CV. |
| Đặc trưng và huấn luyện | `pipeline/modeling/experiments.py`, `pipeline/features/spam_features.py` | BoW/TF-IDF/FastText + NB/SVM; học trong train/fold. |
| Mô hình hiện đại | Mục chờ trong notebook; `pipeline/modern/` | Phần dành cho triển khai tiếp theo. |
| Đánh giá | Bảng/hình và `experiments.evaluate` | Chỉ số, confusion matrix, ảnh hưởng stop words/cân bằng lớp, FP/FN. |
| Ứng dụng | Cell lưu mô hình; `pipeline/inference/predict.py` | Bundle đã học, tải lại và dự đoán raw text. |

Hướng dẫn đầy đủ: [Tổng quan project](../guides/TONG_QUAN_PROJECT_VA_CACH_THUC_THI.md).
