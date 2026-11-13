# Lộ trình trình bày mã nguồn và kết quả

Mở project cùng ba notebook. Giải thích **đầu vào → hàm xử lý → đầu ra**, dùng
khối mã thu gọn để dẫn tới file khi cần. Không đọc tuần tự mọi dòng code.
Vai trò cụ thể từng người ở [phân công học](../team/TEAM_ASSIGNMENTS.md).

## 1. Notebook 01 — Nguồn, bảng dữ liệu và nhãn

| Mục/người | File/hàm cần mở | Output và ý cần nói |
|---|---|---|
| 1–2 / Cương | `pipeline/collect/scrape_ptit_groups.py`: `PTIT_GROUPS`, `run_scraper`, `on_batch`; `external/facebook_scraper/group_post_scraper_v2.py`: `fetch_posts`; `main.py`: `fetch_comments_for_post` | Group/link từ JSON thực tế, một bài và comment/reply; 5 nhóm, 2.170 bài. Runner cấu hình nguồn; mã nền lấy nội dung |
| 2 / Cương | `SingleFileWriter.save_records`, `on_page` | Writer lưu JSON bài kèm bình luận; state/cursor phục vụ tiếp tục thu thập. Chưa tạo nhãn |
| 3 / Cương | `pipeline/collect/json_to_csv.py`: `flatten_posts` | 13.833 comment/reply → 13.704 text không rỗng, mỗi bình luận thành một dòng CSV |
| 4 / Cương | `pipeline/prepare/clean_for_labeling.py`: `preprocess`, `clean_facebook_comments` | Ví dụ trước/sau cùng dòng, giữ text/metadata, 10.974 mẫu còn lại |
| 4–5 / Bảo | `pipeline/label/cached_annotations.py`: `attach_annotation_ids`; `ai_assisted_spam_ham.py`: `predict_batch_with_glm` | Gợi ý label/confidence; nối theo nội dung/ID, 6.571 gợi ý khớp, không nối bằng số dòng mới |
| 5–6 / Bảo | `auto_accept_labels.py`; cell lọc nhãn/xuất CSV | Ngưỡng 0,75 → 6.289 mẫu (5.190 ham/1.099 spam); tên file, số dòng và cột thực tế |

Lời chuyển: “Quy trình chuẩn bị dữ liệu giữ nội dung gốc, tạo bản sạch và chỉ
tiếp nhận nhãn đủ điều kiện. Phần tiếp theo thực nghiệm trên tập v7 đã chốt.”
Không diễn giải JSON thu thập là CSV đã gán nhãn, hoặc khẳng định Run All 01
gọi Facebook/GLM; notebook dùng phản hồi đã lưu cho việc tái lập.

Các file Argilla (`upload_to_argilla`, `download_labeled_data`) giải thích khi
được hỏi về giao diện duyệt nhãn; chúng không tự chạy trong Notebook 01.
Tool tổng hợp dữ liệu là tiện ích riêng, không phải mô hình hoặc bước fit.

## 2. Notebook 02 — Chọn và đánh giá mô hình truyền thống

| Mục/người | File/hàm | Output và ý cần nói |
|---|---|---|
| 1–3 / Cương | Cell đọc `data/final/labeled_v7.csv`; `pipeline/prepare/preprocess_for_model.py`: `preprocess` | 38.233 mẫu, tỷ lệ ham/spam, comment gốc/sau xử lý cùng ID; lưu toàn bộ CSV token |
| 4 / Cương | `pipeline/modeling/training_utils.py`: `norm_key`, `bucket`, `grouped_folds` | 30.759 train/7.474 test, năm fold trong train; tránh văn bản cùng khóa ở hai tập; CV chọn/test báo cáo |
| 5 / Phi | `pipeline/modeling/experiments.py`: `make_pipe`, `fit_experiment` | Hai đặc trưng × hai họ mô hình; bảng đối chiếu từ dừng bằng CV; luồng tiếp giữ từ dừng |
| 5.1 / Phi | `sample_training_indices`, `undersampled_search`, `tune_svm_threshold` | B4 đối chứng, S1 trọng số, S2 giảm mẫu, S3 chỉnh ngưỡng; bảng CV và hình số mẫu/trọng số, không dùng test chọn |
| 5.2 / Phi | Trích `pipeline/modern/train.py`: `load_pretrained` | Dẫn nhập viBERT; chi tiết ở Notebook 03, không fine-tune trong 02 |
| 6–6.1 / Bảo | `experiments.evaluate`; cell bảng | Test B1–B4 và B4/S1/S2/S3; tổng bảy ứng viên, so sánh Precision/Recall/F1/FP/FN |
| 6.2–6.3 / Bảo | Cell ma trận/PR, `boot_ci` | Đọc một FP/FN; PR dùng score, đổi ngưỡng không đổi AP; khoảng bootstrap không chứng minh ưu thế thống kê |
| 7 / Bảo | Cell dùng `best`, đặc trưng hệ số và bảng lỗi | Cấu hình phân tích là S3, không phải mô hình tùy ý; text/nhãn/prediction/score/threshold cùng mẫu |
| 8–9 / Phi | `joblib.dump/load`, `pipeline/modeling/predict.py`: `predict_spam` | S3 `.joblib` giữ pipeline/ngưỡng; nạp lại/demo; đánh đổi Precision–Recall và giới hạn |

S3: **TF-IDF+SVM, giữ từ dừng, C=0,3, ngram=(1,2), ngưỡng −0,1776**. CV F1
0,7876, test F1 0,7552; không chọn mô hình bằng test. Score âm vẫn có thể lớn
hơn ngưỡng và được dự đoán spam. Không gọi score SVM là xác suất.

Lời chuyển: “S3 là cấu hình truyền thống được chọn trong phạm vi khảo sát.
Nhóm tiếp tục fine-tune mô hình ngôn ngữ viBERT, giữ cùng tập test để đối chiếu.”

## 3. Notebook 03 — viBERT và đối chiếu cuối

Phương trình bày chính; Cương hỗ trợ giải thích split ở mục 2.

| Mục | File/hàm | Output và ý cần nói |
|---|---|---|
| 1–2 | `pipeline/modern/data.py`: `prepare_data` | Kernel/mode; 24.815 fit/5.944 validation trong train; cùng 7.474 test S3 |
| 3 | `normalize_text`, tokenizer cased | Comment thật và token WordPiece; `##`/token đặc biệt không phải lỗi font; `[UNK]` mất thông tin từ vựng; 0,77% train quá 128 token |
| 4 | `train.py`: `load_pretrained`, `CommentDataset` | Encoder pretrained và head ham/spam; logits→softmax→ngưỡng 0,5; không ghép thêm TF-IDF |
| 5 | `ClassificationTrainer.compute_loss`, `fit_variant`, `select`, `finalize` | Hai loss, lịch sử loss/validation F1; chọn loss thường/epoch 3 rồi fit lại toàn train từ pretrained |
| 6; 02 mục 6.4 | `compare_with_svm`, `draw_figures`; CSV dự đoán theo ID | Test M1 F1 0,8493 so S3 0,7552; cùng ID/nhãn, ma trận/PR, không so validation F1 với CV F1 |
| 7 | Cell FP/FN đọc `test_predictions.csv` | M1 có 91 FP/182 FN; chỉ ra một ví dụ và giới hạn ngữ cảnh/nhãn |
| 8–9 | `VibertPredictor`; `model/model.safetensors` và tokenizer | Nạp model và dự đoán, không fine-tune ở review; giới hạn câu ngắn/chỉ dấu câu, xác suất cao có thể sai |

M1 bắt thêm 115 spam và giảm 36 ham nhận nhầm so S3 trên test hiện tại. Kết
luận trong phạm vi một dataset/split/seed; chưa bảo đảm mọi nhóm hoặc thời điểm.

## 4. Chuẩn bị buổi demo

- Kiểm tra dữ liệu và hai thư mục model đã có local; Git không chứa trọng số.
- Dùng `.venv` cho 01/02, `.venv-modern` cho 03; 03 giữ `MODE="review"`.
- Có thể chạy toàn bộ bằng runner trước buổi trình bày để lưu output. Trong
  buổi demo, mở kết quả và chạy lại cell suy luận sau phần khởi tạo phù hợp;
  không chạy lại GridSearch hoặc fine-tune chỉ để hiện bảng.
- Với Notebook 02, cell mục 8 cần các biến sau huấn luyện trong cùng kernel;
  nếu đã restart và chỉ cần dự đoán, dùng đoạn nạp `.joblib` ở README mục 8
  thay vì nhảy thẳng tới cell lưu model. Với 03 review, có thể chạy cell cấu
  hình mục 1 rồi cell nạp model/demo mục 8 nếu chỉ cần suy luận.
- Nếu Restart Kernel, chạy lại các cell phụ thuộc trước khi demo; xem
  [README](../../README.md) để tránh biến chưa khởi tạo. Không gọi output cũ là
  kết quả vừa huấn luyện trong buổi trình bày.
- Thống nhất ba ngưỡng: tiếp nhận nhãn 0,75; S3 decision score −0,1776; M1
  xác suất 0,5. Chuẩn bị một ví dụ đúng và một ví dụ sai để giải thích giới hạn.

Các yêu cầu kỹ thuật của đề đã có trong luồng trên; CV/bootstrap/PR và các
chiến lược mất cân bằng là bổ trợ. Báo cáo học phần sẽ viết riêng sau khi chốt
code, không coi tài liệu hướng dẫn là báo cáo nộp hoàn chỉnh.
