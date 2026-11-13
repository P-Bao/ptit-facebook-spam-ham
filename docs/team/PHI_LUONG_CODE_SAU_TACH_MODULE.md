# Luồng code phần Phi sau khi tách module

Đây là bản chỉ đường để Phi **đọc code và trình bày đúng thứ tự**. Notebook 02 vẫn là nơi gọi hàm và chạy toàn bộ thí nghiệm; các file `.py` chứa định nghĩa chạy thực tế. Bản [đọc nguyên văn code](PHI_DOC_CODE_THEO_LUONG.md) đặt mã của từng file và cell ngay dưới giai đoạn tương ứng. Không có file nào ở đây là mô hình đã huấn luyện; mô hình được ghi ra `.joblib` ở cuối Notebook 02.

**Phạm vi Phi chốt hiện tại:** Cương bàn giao CSV có nhãn/token; Phi tự gọi `split_train_test` rồi `grouped_folds`, viết tay phần BoW/TF-IDF đúng phạm vi bài, huấn luyện NB/SVM, so lệch lớp, đánh giá và lưu `.joblib`. [Script chạy thẳng](../../scripts/run_phi_pipeline.py) giúp đọc từng giai đoạn. Khi nhóm thống nhất phân công trình bày, báo Bảo đường dẫn đánh giá mới để cùng dùng đúng file.

## 0. Ranh giới bàn giao: CSV, train/test và năm fold

**Cương bàn giao:** nhãn và văn bản đã được xử lý trong `data/processed/v7_preprocessed.csv`, gồm `text`, `cleaned_text`, `label` và hai chuỗi token giữ/bỏ từ dừng. Khi chạy Notebook 02 từ đầu, các cell trước mục 4 tạo lại CSV này từ v7 và đưa `raw`, `TEXT`, `y_all` vào RAM. **Phi bắt đầu tại đây:** gọi `split_train_test`, sau đó `grouped_folds` trong [training_utils.py](../../pipeline/modeling/training_utils.py).

**Thứ tự trong cell 11:** lấy khóa từ `cleaned_text` → dùng hash của khóa để đặt bình luận vào train hoặc test → kiểm hai phía không có cùng khóa → chia **train** thành năm cặp chỉ số `(train_fold, validation_fold)`. Tập test giữ riêng đến bước đánh giá. Chia theo nội dung đã chuẩn hóa giúp tránh bình luận trùng đi cả hai phía, nhưng chưa loại được mọi câu gần trùng hoặc trùng tác giả/bài đăng. Tỷ lệ chỉ xấp xỉ 80/20 vì một khóa được giữ nguyên nhóm. Kết quả hiện tại là 30.759 train và 7.474 test.

**Phi tự tạo:** `train_idx`, `test_idx`, `y_train`, `y_test`, `CV_FOLDS` từ CSV và `TEXT`. `TEXT[pre].iloc[train_idx]` mới là đầu vào của việc fit. Không đưa `test_idx` vào hàm `fit_experiment` hay `tune_svm_threshold`.

## 1. Từ bình luận sang bốn mô hình cơ sở

Luồng gọi hàm ở cell 13:

```text
TEXT[pre] → chọn các dòng train
  → experiments.fit_experiment(pre, feature, algorithm, X_train, y_train, CV_FOLDS)
      → text_features.make_vectorizer(feature): BoW hoặc TF-IDF chưa fit
      → classifiers.make_classifier(algorithm): NB hoặc SVM chưa fit
      → experiments.make_pipe(): Pipeline(vec, clf) và lưới tham số
      → GridSearchCV.fit(): mỗi fold fit vec và clf trên fold train,
        chấm validation bằng Precision/Recall/F1 spam
      → refit cấu hình dẫn đầu trên toàn train
  → record {điểm CV, tham số, _model đã fit, _threshold mặc định}
```

[text_features.py](../../pipeline/modeling/text_features.py) tạo `CountVectorizer` hoặc `TfidfVectorizer` của scikit-learn: `fit` học từ vựng/IDF trên train; `transform` biến văn bản theo trạng thái đã học và trả ma trận thưa. [classifiers.py](../../pipeline/modeling/classifiers.py) chọn `MultinomialNB` với lưới `alpha`, hoặc `LinearSVC` với lưới `C` và `class_weight`; chưa fit. [experiments.py](../../pipeline/modeling/experiments.py) ghép chúng vào `Pipeline`; `search.fit(X,y)` mới học từ vựng, IDF và trọng số phân loại trên từng fold train.

Project hiện chỉ dùng hai lớp vector hóa của scikit-learn, không có lựa chọn backend. Notebook 02 đã chạy lại đầy đủ; kết quả trong `artifacts/spamham/metrics.csv` là lượt chạy thư viện hiện hành. Xem [giải thích bốn lớp](PHI_HIEU_4_LOP_SCIKIT_LEARN.md).

Notebook thử hai nhánh từ dừng × hai đặc trưng × hai bộ phân loại = **8 lượt CV**. Dựa trên CV F1 spam, nhánh giữ từ dừng đi tiếp; bốn record của nhánh này được đặt B1–B4. **B4 = TF-IDF + SVM** dẫn đầu nhóm cơ sở theo CV, nên trở thành đối chứng của giai đoạn 2. Điểm CV là trung bình trên validation của train; đây chưa phải điểm test. BoW/TF-IDF tạo vector đặc trưng thưa của **cả bình luận**, không phải embedding FastText; module FastText đã bị bỏ.

**Phi phải giải thích được:** `ngram_range`, `min_df`, `alpha`, `C`, vì sao NB dùng đặc trưng không âm, SVM trả `decision_score` có dấu, `Pipeline`/`GridSearchCV` gọi `fit` lúc nào và tại sao có 8 lượt nhưng chỉ 4 mã B.

## 2. Khảo sát lệch lớp trên B4

Cell 15 hiển thị số mẫu và trọng số để thấy ham nhiều hơn spam; cell 16 tạo **ba cấu hình riêng** từ TF-IDF + SVM:

| Mã | File và hàm chạy | Thực hiện | Đầu ra |
|---|---|---|---|
| B4 | `experiments.fit_experiment` | Toàn train, `class_weight=None`, ngưỡng SVM 0 | Mốc so sánh |
| S1 | `imbalance.fit_balanced_svm` gọi `fit_experiment` với trọng số balanced | Tăng mức phạt lỗi của lớp ít mẫu khi fit; giữ toàn bộ train | Record CV/model S1 |
| S2 | [imbalance.py](../../pipeline/modeling/imbalance.py): `sample_training_indices` → `undersampled_search` | Mỗi fold chỉ giảm ham trong **fold train** về tối đa 2:1 so với spam; chọn tham số rồi fit trên train đã giảm mẫu | Record CV/model S2 |
| S3 | `imbalance.py`: `tune_svm_threshold` | Giữ tham số B4, tạo score ngoài fold (OOF) trên train, chọn ngưỡng theo F1, fit cuối toàn train | Record CV/model S3 và ngưỡng |

`CANDIDATES` gồm B1–B4/S1–S3. Notebook chọn `best` bằng **CV F1 spam cao nhất trong bảy cấu hình** trước khi chạy cell test. Lượt hiện tại chọn S3, ngưỡng score khoảng `−0,1776`. Score âm hợp lệ vì `LinearSVC.decision_function` không trả xác suất. S1/S2/S3 không được cộng vào một mô hình.

**Vì sao chỉ khảo sát B4?** Mục tiêu hiện tại là đo tác động ba chiến lược trên một cấu hình cơ sở dẫn đầu, với cùng cách biểu diễn/bộ phân loại. Áp dụng cả ba chiến lược cho bốn cấu hình sẽ mở thêm thí nghiệm và phải định nghĩa cách chỉnh ngưỡng/lấy mẫu tương ứng cho NB. Đó là hướng mở rộng hợp lý, nhưng không cần để trả lời câu hỏi hiện tại. Vì chưa thử hết, kết luận “S3 tốt nhất” chỉ đúng **trong bảy ứng viên đã thử**, không khẳng định mọi cấu hình khác không thể tốt hơn.

## 3. Đánh giá trên test sau khi đã chọn

Cell 20 gọi [evaluation.py](../../pipeline/modeling/evaluation.py): `evaluate(record,X_test,y_test)` cho cả bảy record. `predict_record()` lấy xác suất spam của NB hoặc decision score của SVM, áp ngưỡng từng record rồi trả nhãn. `evaluate()` tính Accuracy, Precision, Recall, F1 spam, AP và gắn `_pred`/`_score` vào record. `boot_ci()` tính khoảng bootstrap F1 cho phần trình bày kết quả. Hàm đánh giá **không fit** mô hình và không chọn lại `best` theo test.

Bảo phụ trách phần biểu đồ và phân tích FP/FN sâu; Phi phải giải thích được chỉ số của B4/S1/S2/S3, đặc biệt S3. Kết quả test là phép kiểm độc lập tương đối với bước chọn bằng CV; nó chưa chứng minh mô hình đúng trên mọi comment mới. Việc bỏ FastText chỉ bỏ hai cấu hình FastText; các cấu hình còn lại được chạy lại và so với bảng trước ở cuối tài liệu này.

## 4. Lưu S3 và dự đoán bình luận mới

Cell 32 dùng trực tiếp `joblib.dump()` lưu `bundle` gồm `best["_model"]` (Pipeline đã fit), ngưỡng, chế độ từ dừng và manifest vào `artifacts/spamham/spam_ham_ml_only.joblib`. `joblib.load()` nạp lại để kiểm nhãn test trước/sau lưu khớp nhau. [predict.py](../../pipeline/modeling/predict.py) nhận bình luận thô mới, gọi `preprocess()` đúng chế độ trong bundle, để TF-IDF đã fit biến câu thành vector, lấy score SVM và so với ngưỡng S3. Kết quả là bảng `text`, `score`, `score_type`, `du_doan`.

Đây là **mô hình có thể dùng để dự đoán qua code Python**, chưa phải ứng dụng web/API. Khi mang file sang máy khác vẫn cần mã tiền xử lý và phiên bản thư viện tương thích. File `.py` định nghĩa các bước; file `.joblib` lưu đối tượng đã học và ngưỡng. Demo một câu mới cho thấy code chạy đúng, không chứng minh câu nào cũng dự đoán đúng.

## Mở code để tự học và trình bày

Trong VS Code, chọn interpreter `.venv`, dùng **Ctrl+Click** hoặc **F12** ở tên hàm. `text_features.py` cho thấy cách nhóm cấu hình bộ vector hóa; Ctrl+Click tiếp vào `CountVectorizer`/`TfidfVectorizer` mở mã thư viện. [Tài liệu bốn lớp](PHI_HIEU_4_LOP_SCIKIT_LEARN.md) chỉ đúng các lớp cha/phương thức liên quan. `help(GridSearchCV)` cho giao diện thư viện; `inspect.getsource(fit_experiment)` cho mã nhóm viết. Chạy riêng toàn bộ phần Phi bằng [scripts/run_phi_pipeline.py](../../scripts/run_phi_pipeline.py), hoặc chạy Notebook 02 như [README](../../README.md).

Có thể thử riêng việc ghép đặc trưng và phân loại bằng đoạn Python ngắn dưới đây từ gốc project; đây chỉ là **dữ liệu tự tạo để học cách gọi hàm**, không phải số liệu đánh giá bài làm:

```python
from pipeline.modeling.experiments import make_pipe

X = ["lich thi tai lieu", "lich thi hoc tap", "mua ban lien he", "mua ban quang cao"]
y = [0, 0, 1, 1]
pipe, grid, _ = make_pipe("TF-IDF", "SVM", n_jobs=1)
pipe.fit(X, y)  # Lúc này vectorizer và SVM mới học từ bốn câu.
print(grid)     # Các tham số mà CV sẽ thử trong thí nghiệm thật.
print(pipe.predict(["lich thi", "mua ban"]))
```

## Kết quả sau khi chạy lại

Notebook 02 đã chạy đủ 36 cell (19 cell code) trong kernel mới. File `artifacts/spamham/spam_ham_ml_only.joblib` nạp lại được; `predict_spam()` trả nhãn cho bình luận mới. So với lượt chạy script riêng trên cùng dữ liệu, **Accuracy, Precision, Recall và F1 test của cả bảy cấu hình trùng nhau**. CV F1 của B2/S2 và AP của B2 chênh rất nhỏ giữa hai lượt chạy (dưới 0,001), không đổi cấu hình được chọn; đây không phải bằng chứng mọi lần chạy đều khớp từng chữ số. FastText đã được loại khỏi bài hiện tại.

| Mã | CV F1 spam | Test Precision | Test Recall | Test F1 | Ghi chú |
|---|---:|---:|---:|---:|---|
| B1: BoW + NB | 0,6077 | 0,7441 | 0,4953 | 0,5947 | Cơ sở |
| B2: BoW + SVM | 0,7091 | 0,7110 | 0,6803 | 0,6953 | Cơ sở |
| B3: TF-IDF + NB | 0,5960 | 0,8402 | 0,5142 | 0,6380 | Cơ sở |
| B4: TF-IDF + SVM | 0,7736 | 0,8818 | 0,6278 | 0,7334 | Đối chứng |
| S1: trọng số balanced | 0,7580 | 0,7139 | 0,7792 | 0,7451 | Trên TF-IDF + SVM |
| S2: giảm mẫu 2:1 | 0,7629 | 0,7791 | 0,7308 | 0,7542 | Trên TF-IDF + SVM |
| **S3: chỉnh ngưỡng** | **0,7876** | **0,8374** | **0,6877** | **0,7552** | **Chọn bằng CV** |

S3 có test F1 chỉ hơn S2 khoảng **0,0010**, nên không diễn giải đây là cách áp đảo trên mọi dữ liệu. Ngưỡng S3 là `−0,177628` trên **decision score**, không phải xác suất. Kết quả này hợp lý cho tập dữ liệu hiện tại: hạ ngưỡng từ 0 giúp tăng Recall so với B4, kèm giảm Precision. Test được dùng để báo hiệu năng sau khi khóa lựa chọn bằng CV.

Script chạy thẳng từ `data/processed/v7_preprocessed.csv` cũng chọn S3, tạo 30.759 dòng train/7.474 dòng test. Hai lượt chạy có cùng Accuracy, Precision, Recall và F1 test của cả bảy cấu hình. CV F1 của B2 lệch khoảng 0,00053, S2 lệch khoảng 0,00018; AP của B2 lệch khoảng 0,00009. Những sai khác nhỏ này không thay đổi cấu hình được chọn hay kết luận test. Đối chiếu chính xác từng lượt chạy bằng `artifacts/spamham/phi_pipeline/metrics.csv` và `artifacts/spamham/metrics.csv`.

**Giới hạn môi trường:** Notebook 03 ở chế độ review cần `torch`/`transformers`; `.venv` local hiện thiếu `torch`, nên lượt chạy Notebook 03 dừng ngay ở import. Module chia tập dùng chung với viBERT vẫn import được; chưa có bằng chứng chạy hết Notebook 03 trong môi trường này sau lần tách module. Đây là việc của môi trường nhánh viBERT, cần trao đổi với Phương nếu nhóm muốn demo local.
