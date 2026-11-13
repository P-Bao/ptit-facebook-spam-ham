# Mô hình truyền thống — đọc theo giai đoạn

Các file này là **module Python chạy được khi được import và gọi hàm**. [Notebook 02](../../notebooks/02_thuc_nghiem_va_demo.ipynb) điều phối chúng theo thứ tự dưới đây. Để đọc một luồng code chạy từ trên xuống, mở [scripts/run_phi_pipeline.py](../../scripts/run_phi_pipeline.py); script nhận CSV đã có nhãn/token, tự chia tập, huấn luyện, đánh giá và lưu `.joblib` vào thư mục riêng. Chạy riêng `python pipeline/modeling/text_features.py` sẽ chỉ nạp định nghĩa, không tự huấn luyện.

| Giai đoạn | File | Hàm chính | Nhận → trả |
|---|---|---|---|
| 0: chia tập | `training_utils.py` | `split_train_test`, `grouped_folds` | CSV đã sạch/nhãn → vị trí train/test và năm lượt train/validation. `norm_key`/`bucket` giữ câu trùng cùng phía. |
| 1a: đặc trưng | `text_features.py` | `make_vectorizer` | Tạo `CountVectorizer` hoặc `TfidfVectorizer` chưa fit, kèm lưới n-gram 1 hoặc 1–2; ma trận thưa chỉ được tạo khi Pipeline fit/transform. |
| 1b: phân loại | `classifiers.py` | `make_classifier` | Tạo NB/SVM **chưa fit**, kèm các alpha hoặc C để thử bằng CV. Không học mô hình tại đây. |
| 1c: huấn luyện | `experiments.py` | `make_pipe`, `fit_experiment` | Đặc trưng + classifier + train/folds → Pipeline, điểm CV, tham số, model đã fit. |
| 2: lệch lớp | `imbalance.py` | `fit_balanced_svm`, `undersampled_search`, `tune_svm_threshold` | B4 + train/folds → S1 trọng số, S2 giảm mẫu, S3 đổi ngưỡng; ba cách riêng. |
| 3: đánh giá | `evaluation.py` | `predict_record`, `evaluate`, `boot_ci` | Model đã chốt + test → nhãn, score, chỉ số và khoảng F1. Không fit lại. |
| 4: suy luận | `predict.py` | `predict_spam` | Bình luận mới + bundle đã nạp → tiền xử lý, score, nhãn ham/spam. Không huấn luyện hoặc ghi `.joblib`. |

Trong `training_utils.py` còn các lệnh `import` chuyển tiếp tên hàm cũ của Phi sang `imbalance.py` và `evaluation.py`, để test hoặc notebook cũ không lỗi. **Đọc định nghĩa thực của hàm ở file mới**, không phải ở dòng import chuyển tiếp.

`vector.py` là ví dụ tính BoW/TF-IDF thủ công trên ba câu mẫu để học khái niệm. Luồng huấn luyện và Notebook 02 dùng `CountVectorizer`/`TfidfVectorizer` trong `text_features.py`; không import `vector.py`.

Notebook 02: cell 11 chia train/test và tạo folds; cell 13 fit bốn cấu hình B1–B4 trên hai nhánh từ dừng; cell 15–16 so B4 với S1/S2/S3 và chọn `best` bằng CV; cell 20 đánh giá test; cell 32 gọi `joblib.dump(bundle, model_path)` để lưu `.joblib`, sau đó nạp lại và demo bằng `predict.py`. [`scripts/run_phi_pipeline.py`](../../scripts/run_phi_pipeline.py) cũng có lệnh `joblib.dump` khi chạy riêng phần Phi. `predict.py` chỉ thực hiện suy luận trên bundle được truyền vào. Phi học toàn bộ luồng này; Bảo vẫn có thể trình bày sâu về biểu đồ và lỗi.

**Học bốn lớp thư viện đang dùng:** [PHI_HIEU_4_LOP_SCIKIT_LEARN.md](../../docs/team/PHI_HIEU_4_LOP_SCIKIT_LEARN.md) chỉ đường Ctrl+Click tới mã nguồn trong `.venv`, giải thích `CountVectorizer`, `TfidfVectorizer`, `MultinomialNB` và `LinearSVC` theo đúng cấu hình bài.

**Cách đọc:** bắt đầu ở `split_train_test`, tiếp đến `make_vectorizer`, `make_classifier`, rồi `fit_experiment`. Bốn lớp vector hóa/phân loại đều dùng scikit-learn; các file project chỉ chọn cấu hình và ghép luồng. `make_vectorizer`, `make_classifier`, `make_pipe` chỉ tạo đối tượng chưa học; `search.fit(X,y)` trong `fit_experiment` mới chạy CV và fit lại model cuối trên train. `ParameterGrid` ở S2 chỉ liệt kê tám cặp n-gram/C để thử; không tự học dữ liệu.
