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
| Thiết kế thực nghiệm | Mục 4; `pipeline/modeling/training_utils.py` | Nhóm văn bản, train/validation/test; CV F1 chọn cấu hình, test báo cáo kết quả. |
| Đặc trưng và huấn luyện | Mục 5; `pipeline/modeling/experiments.py`, `pipeline/features/spam_features.py` | Đối chiếu từ dừng bằng CV, chọn tiền xử lý rồi so sánh sáu tổ hợp cơ sở; học trong train/fold. |
| Mô hình hiện đại | Notebook 03; `pipeline/modern/` | viBERT, tokenizer cased, hai loss, chọn bằng validation rồi fit toàn bộ train. |
| Mất cân bằng | Mục 5.1; `training_utils.py`: `undersampled_search`, `tune_svm_threshold` | Cùng TF-IDF + SVM: không điều chỉnh, balanced, lấy mẫu 2:1, chỉnh ngưỡng; bảng CV và hình số mẫu/trọng số. |
| Đánh giá | Mục 6–6.3; `experiments.evaluate` | Test của sáu tổ hợp và bốn chiến lược; FP/FN, ma trận nhầm lẫn và PR. Đổi ngưỡng chỉ đổi điểm hoạt động trên đường PR. |
| Phân tích lỗi | Mục 7 | Ghi rõ cấu hình được chọn, ID/text gốc, nhãn thực tế, dự đoán, score và ngưỡng. |
| Ứng dụng | Cell lưu mô hình; `pipeline/inference/predict.py` | Bundle đã học, tải lại và dự đoán raw text. |

Hướng dẫn đầy đủ: [Tổng quan project](../guides/TONG_QUAN_PROJECT_VA_CACH_THUC_THI.md).

## Notebook 03 — Fine-tune viBERT

Mở lần lượt: cách chia ID và bình luận gốc → token/subword → checkpoint và đầu
phân loại → loss thường/có trọng số → lịch sử và validation F1 → test trên cùng
ID với S3 → FP/FN → model/tokenizer đã lưu → demo. Điểm validation không phải
CV F1 năm fold. `MODE="review"` đọc kết quả đã train; `MODE="train"` gọi module
huấn luyện. Code loss ở `ClassificationTrainer`, cách chia ở `modern.data`,
suy luận ở `VibertPredictor`.
