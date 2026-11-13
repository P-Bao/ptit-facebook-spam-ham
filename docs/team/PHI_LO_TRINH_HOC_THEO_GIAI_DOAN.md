# Lộ trình học phần Phi theo giai đoạn

Tài liệu này theo **phạm vi Phi vừa chốt**: nhận CSV **đã có nhãn và đã tiền xử lý**, tự chia tập rồi làm toàn bộ phần mô hình truyền thống đến đánh giá và xuất file. Mỗi giai đoạn nhận đầu ra của giai đoạn trước. Sau khi nắm luồng, đọc [lý thuyết và câu hỏi](PHI_HOC_HIEU_PROJECT.md), rồi đối chiếu [toàn bộ code theo thứ tự](PHI_DOC_CODE_THEO_LUONG.md). Bảng phân công nhóm cũ chưa được cập nhật theo phạm vi học mới này.

```text
CSV có nhãn và token
  → 0. Phi tự chia train/test và tạo các fold CV
  → 1. Tạo bốn mô hình cơ sở, chọn cách giữ/bỏ từ dừng bằng CV
  → 2. Khảo sát ba chiến lược trên TF-IDF + SVM, chốt mô hình bằng CV
  → 3. Đọc kết quả test và giải thích giới hạn
  → 4. Lưu mô hình, nạp lại và dự đoán bình luận mới
```

## Giai đoạn 0 — Nhận CSV rồi tự chia tập

**Đầu vào bàn giao:** `data/processed/v7_preprocessed.csv` có `id`, `label`, `text`, `cleaned_text`, `tokens_keep_stopwords`, `tokens_remove_stopwords`. Khi chạy Notebook 02 từ đầu, các cell trước mục 4 tạo và lưu CSV này từ v7 rồi đọc lại vào `prepared`. Phi nhận nội dung CSV và **tự chia tập**. Trong notebook, `raw`, hai nhánh chuỗi token `TEXT` và nhãn số `y_all` (`ham=0`, `spam=1`) đã có trong RAM; [script chạy độc lập](../../scripts/run_phi_pipeline.py) đọc trực tiếp CSV.

**Xử lý:** cell 11 gọi `split_train_test` rồi `grouped_folds` trong [training_utils.py](../../pipeline/modeling/training_utils.py). Hai bước bên trong là `norm_key` (tạo khóa nội dung) và `bucket` (gán nhóm ổn định); chúng không làm sạch lại câu đưa vào mô hình. Test được tách riêng; chỉ train chia thành năm fold. Giữ cách gán nhóm hiện tại để điểm so với viBERT trên cùng test và số liệu đã có không đổi.

**Đầu ra Phi tự tạo:** `train_idx`, `test_idx`, `y_train`, `y_test`, `CV_FOLDS` cùng `TEXT`. `train_idx`/`test_idx` là vị trí trong toàn CSV; chỉ số bên trong từng cặp của `CV_FOLDS` lại tính theo **mảng train**. Phân biệt hai hệ chỉ số này để không lấy sai dòng.

**Tác dụng:** có tập test giữ riêng và các phần validation trong train để chọn mô hình mà không dùng test vào lúc huấn luyện. **Kiến thức cần học:** train/validation/test; 5-fold CV; nhóm bình luận trùng; rò rỉ dữ liệu (*data leakage*); ý nghĩa của `numpy` index. **Ranh giới nhiệm vụ:** Cương bàn giao CSV và token; từ bước chia train/test và fold CV trở đi là phần Phi học, chạy và giải thích.

**Tự kiểm:** giải thích được vì sao `X_train_text = TEXT[pre].iloc[train_idx]` chỉ lấy train, và vì sao không được dùng `test_idx` như chỉ số của mảng train.

## Giai đoạn 1 — Tạo mô hình cơ sở và chọn nhánh từ dừng

**Đầu vào:** `TEXT` có hai nhánh giữ/bỏ từ dừng, `train_idx`, `y_train`, `CV_FOLDS` từ giai đoạn 0.

**Xử lý:** [text_features.py](../../pipeline/modeling/text_features.py) tự triển khai BoW/TF-IDF đúng các lựa chọn bài dùng: tách từ/cặp từ, loại đặc trưng chỉ có trong một câu, đếm, tính IDF và chuẩn hóa. [classifiers.py](../../pipeline/modeling/classifiers.py) tạo `MultinomialNB` hoặc `LinearSVC` chưa fit. `make_pipe` trong [experiments.py](../../pipeline/modeling/experiments.py) ghép hai bước; `fit_experiment` dùng `GridSearchCV` thử n-gram và `alpha`/`C`, tính Precision, Recall, F1 spam trên năm fold; `refit="f1"` chọn tham số theo F1 rồi fit lại toàn train. [Notebook 02](../../notebooks/02_thuc_nghiem_va_demo.ipynb) cell 13 chạy **2 nhánh từ dừng × 2 đặc trưng × 2 bộ phân loại = 8 lượt CV**.

Phần đặc trưng hiện dùng duy nhất `CountVectorizer`/`TfidfVectorizer` của scikit-learn. Đọc [tài liệu bốn lớp](PHI_HIEU_4_LOP_SCIKIT_LEARN.md) để hiểu mã thư viện được gọi khi `Pipeline.fit`, thay vì tìm một lớp vector hóa tự viết trong project.

**Đầu ra:** `PREPROCESS_RESULTS` chứa tám record; bảng CV so hai nhánh; `MAIN_PRE` là nhánh có ứng viên CV F1 cao nhất. Với dữ liệu hiện tại đó là **giữ từ dừng**. `RESULTS` còn bốn record của nhánh được chọn, mã B1–B4. `focus` là B4 = TF-IDF + SVM, mô hình cơ sở dẫn đầu theo CV. Mỗi record giữ điểm CV, tham số và `_model` đã fit.

**Tác dụng:** chọn một cách biểu diễn và một bộ phân loại từ bằng chứng validation trong train. B4 trở thành mốc để khảo sát mất cân bằng. **Kiến thức cần học:** BoW, TF-IDF, unigram/bigram, `min_df`, ma trận thưa; MultinomialNB và `alpha`; SVM tuyến tính và `C`; `Pipeline.fit/transform`, `GridSearchCV`, `refit`, F1 spam; vì sao không fit từ vựng/IDF trên toàn CSV trước CV.

**Tự kiểm:** chỉ ra B1–B4 là bốn **cặp đặc trưng–bộ phân loại**, còn tám là số lượt thử vì có thêm hai nhánh từ dừng; nói được CV F1 ở cell 13 **chưa phải điểm test**. FastText không còn trong code hiện tại.

## Giai đoạn 2 — Khảo sát mất cân bằng và chốt S3

**Đầu vào:** `focus=B4` đã fit, token train thuộc `MAIN_PRE`, `y_train` và `CV_FOLDS`. Không dùng `y_test` hay bình luận test để chọn chiến lược.

**Xử lý:** cell 15 của Notebook 02 hiển thị số ham/spam trước và sau khi giảm mẫu, cùng trọng số cân bằng. Cell 16 thử ba chiến lược **riêng biệt** trên TF-IDF + SVM:

| Cấu hình | Hàm/code chính | Thay đổi gì? | Giữ nguyên gì? |
|---|---|---|---|
| B4 | `fit_experiment` | Mốc không điều chỉnh | Toàn train, trọng số mặc định, ngưỡng score 0 |
| S1 | `fit_balanced_svm` trong `imbalance.py` | Phạt lỗi lớp ít mẫu nặng hơn khi fit | Số dòng train và test |
| S2 | `sample_training_indices`, `undersampled_search` trong [imbalance.py](../../pipeline/modeling/imbalance.py) | Lấy ham tối đa gấp đôi spam **chỉ từ fold train**; fit cuối trên tập train đã giảm mẫu | Validation và test |
| S3 | `tune_svm_threshold` trong `imbalance.py` | Tạo score ngoài fold (OOF), tìm ngưỡng F1 tốt nhất; fit cuối trên toàn train | Trọng số mặc định và số dòng train của B4 |

**Đầu ra:** hình `imbalance_training.png`; `STRATEGIES=[B4,S1,S2,S3]`; bảng và file `imbalance_cv.csv`; `CANDIDATES` gồm bốn B và ba S. `best` được chọn bằng **CV F1 spam lớn nhất trong bảy ứng viên**, hiện là S3. S3 dùng ngưỡng score khoảng `−0,1776`, không phải xác suất âm.

**Tác dụng:** kiểm tra liệu các cách ứng phó với lệch lớp có cải thiện việc phát hiện spam hay không, rồi khóa lựa chọn **trước khi đo test**. **Kiến thức cần học:** mất cân bằng lớp, `class_weight="balanced"`, under-sampling, OOF prediction, `decision_function`, ngưỡng phân loại, đánh đổi Precision–Recall. Hiểu rằng S3 đổi **quy tắc ra nhãn**, không tạo thêm dữ liệu và không kết hợp cả S1/S2.

**Tự kiểm:** giải thích được tại sao score SVM và ngưỡng có thể âm; vì sao S2 chỉ lấy mẫu từ fold train; và vì sao điểm CV sau chỉnh ngưỡng không phải một ước lượng *nested CV* độc lập.

## Giai đoạn 3 — Đọc kết quả test, nhận xét mô hình đã chọn

**Đầu vào:** bảy record B1–B4/S1–S3 đã chọn tham số, `test_idx`, `y_test` và token test cùng nhánh tiền xử lý tương ứng.

**Xử lý:** cell 20 của Notebook 02 gọi `evaluate` trong [evaluation.py](../../pipeline/modeling/evaluation.py) trên **cùng một test** cho từng record. Hàm gọi `predict_record` để tạo nhãn và score, rồi tính Accuracy, Precision, Recall, F1 spam và AP. Các cell sau hiển thị bảng, ma trận nhầm lẫn, đường Precision–Recall và ví dụ lỗi. Cell 34 dùng kết quả đã có để viết nhận xét.

**Đầu ra:** các trường `Accuracy`, `Precision`, `Recall`, `F1`, `AP`, `_pred`, `_score` của mỗi record; bảng `metrics.csv` được ghi ở giai đoạn 4. Lượt chạy hiện tại của S3 có CV F1 ≈ `0,7876`; trên test, Precision ≈ `0,8374`, Recall ≈ `0,6877`, F1 ≈ `0,7552`. Đây là **đánh giá sau lựa chọn**, không phải lý do để chọn lại S3.

**Tác dụng:** biết mô hình đã chọn làm việc ra sao trên dữ liệu giữ riêng và hiểu chi phí của FP/FN. **Kiến thức cần học:** TP/FP/FN/TN, Accuracy, Precision, Recall, F1, AP và đường PR; phân biệt điểm CV với điểm test; đọc ma trận nhầm lẫn; giới hạn của kết luận trên một test. **Ranh giới nhiệm vụ:** Phi chạy và hiểu phép tính chỉ số test, đọc được các lỗi S3; Bảo có thể trình bày sâu phần đồ thị và phân tích lỗi của cả nhóm.

**Tự kiểm:** giải thích vì sao một mô hình luôn dự đoán ham vẫn có Accuracy khá cao nhưng F1 spam bằng 0; đổi ngưỡng có thể đổi F1 ở một điểm nhưng không đổi thứ tự score/AP của cùng mô hình.

## Giai đoạn 4 — Lưu S3 và dự đoán bình luận mới

**Đầu vào:** `best=S3` đã fit, ngưỡng đã chọn, chế độ từ dừng, kết quả test và bình luận thô mới.

**Xử lý:** cell 32 của Notebook 02 tạo `bundle` gồm `_model` đã fit, `threshold`, `remove_stop` và manifest; `joblib.dump` ghi file, `joblib.load` nạp lại. Cell kiểm dự đoán trước/sau lưu và kiểm bình luận gốc của tập test. Khi dự đoán câu mới, [predict.py](../../pipeline/modeling/predict.py) gọi lại `preprocess` cùng chế độ từ dừng đã lưu, vectorizer trong pipeline biến câu thành số, SVM tạo `decision_score`, rồi so với ngưỡng S3 để ra `spam`/`ham`.

**Đầu ra:** `artifacts/spamham/spam_ham_ml_only.joblib`, `metrics.csv`, `run_manifest.json`, `stopword_cv.csv`, `test_predictions.csv` và bảng demo `text/score/score_type/du_doan`. File `.py` mô tả **cách xử lý**; `.joblib` chứa **pipeline đã học và cấu hình dự đoán**, nên nạp file là để suy luận, không huấn luyện lại.

**Tác dụng:** biến kết quả thực nghiệm thành mô hình có thể nạp lại và dùng với bình luận mới một cách nhất quán. **Kiến thức cần học:** `fit` so với `predict`, serialization bằng `joblib`, cấu trúc bundle, tiền xử lý khi suy luận, điểm `decision_score` so với xác suất, kiểm nhất quán trước/sau lưu.

**Tự kiểm:** lần theo một bình luận mới từ text thô → `preprocess` → vectorizer TF-IDF đã fit → SVM score → ngưỡng S3 → nhãn; chỉ ra `.joblib` được tạo ở cell nào và vì sao không thể thay bằng mỗi `experiments.py`.

## Học theo thứ tự nào?

Học kỹ **giai đoạn 0** để tự giải thích cách chia train/test, fold và chống trùng; tiếp đó học **giai đoạn 1 và 2** về thiết kế/huấn luyện. Học **giai đoạn 3** để tự tính và bảo vệ chỉ số test của S3; cuối cùng học **giai đoạn 4** bằng cách nạp file và tự dự đoán vài bình luận. Ở mỗi giai đoạn, hãy tự nói được bốn ý: **nhận gì → gọi hàm nào → trả gì → kết quả đó dùng cho giai đoạn kế tiếp ra sao**.
