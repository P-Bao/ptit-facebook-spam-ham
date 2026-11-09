# Huấn luyện và đánh giá mô hình truyền thống

| File | Hàm và vai trò |
|---|---|
| `experiments.py` | `make_pipe`: pipeline/lưới tham số; `fit_experiment`: GridSearchCV theo F1 spam; `evaluate`: đo test; `boot_ci`: khoảng tin cậy bootstrap F1 |
| `training_utils.py` | `norm_key/bucket/grouped_folds`: chia nhóm; `sample_training_indices/undersampled_search`: lấy mẫu chỉ trong train; `tune_svm_threshold`: ngưỡng OOF; `predict_record`: áp ngưỡng nhất quán |
| `__init__.py` | Khai báo package |

Thứ tự trong Notebook 02:

1. **Mục 4:** khóa nhóm từ `cleaned_text`; bucket test theo khóa, giữ các văn bản
   cùng khóa trong cùng tập. Train 30.759 mẫu, test 7.474; train chia năm fold.
   Đây là chia theo văn bản, không phải chia theo tác giả hoặc theo thời gian.
2. **Mục 5:** BoW/TF-IDF × NB/SVM, đối chiếu giữ/loại từ dừng bằng CV.
   Chọn giữ từ dừng cho luồng tiếp. Bốn tổ hợp B1–B4 dùng train nguyên bản và
   ngưỡng mặc định, SVM không dùng trọng số lớp. NB là MultinomialNB cho đặc trưng
   đếm từ/TF-IDF, SVM là LinearSVC.
3. **Mục 5.1:** cố định TF-IDF+SVM, dùng B4 làm đối chứng; thử S1 balanced, S2
   under-sampling ham:spam 2:1 và S3 chỉnh ngưỡng decision score. Không cộng dồn
   ba chiến lược. Vectorizer/sampling đều nằm trong phần train của mỗi fold.
4. **Mục 6–7:** test, FP/FN, ma trận, PR/AP, bootstrap và phân tích lỗi. Bảng bốn
   tổ hợp và bảng bốn chiến lược có chung B4, tổng cộng **bảy** ứng viên.
5. **Mục 8:** notebook lưu ứng viên được chọn theo CV vào bundle `.joblib`;
   `inference/predict.py` sử dụng bundle đó.

S3 được chọn với CV F1 **0,7876**; C=0,3, n-gram=(1,2), ngưỡng **−0,1776**.
Test F1 **0,7552**, Recall **0,6877**, Precision **0,8374**. Ngưỡng âm là điểm
decision score SVM, không phải xác suất âm. Điểm lớn hơn ngưỡng → spam.

`tune_svm_threshold()` fit mỗi phần train, gom điểm dự đoán OOF của validation,
khảo sát 199 phân vị cùng ngưỡng 0/hai biên và chọn trung bình F1 fold cao nhất;
đồng điểm ưu tiên gần 0. Điểm CV này phục vụ lựa chọn, không phải nested CV độc
lập. Không điều chỉnh tham số/ngưỡng từ test.

Hai file là thư viện, không có chương trình `main()` huấn luyện toàn bộ bài.
Chạy [Notebook 02 hoặc runner](../../README.md) để dựng dữ liệu, gọi hàm theo
đúng thứ tự và ghi kết quả. `modern/` là nhánh viBERT riêng, dùng cùng test.
