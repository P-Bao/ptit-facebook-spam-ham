# BƯỚC 03 — `classifiers.py`: MULTINOMIAL NAIVE BAYES VÀ LINEAR SVM — HỌC TỪ MÃ NGUỒN VÀ VÍ DỤ

> **Tài liệu học và bảo vệ bài tập lớn Spam/Ham bình luận sinh viên PTIT — phần modeling của bạn.**  
> **Phạm vi:** Sau bước `text_features.py`, trước bước `experiments.py`. Trọng tâm là hiểu lý thuyết, tham số, **hàm nhóm viết và các phương thức thật của scikit-learn**, ma trận đầu vào, các trạng thái được học, tính score và nhãn đầu ra.  
> **Nguồn chính:** `classifiers(1).py` (phiên bản mới nhất), `PHI_HIEU_4_LOP_SCIKIT_LEARN.md` (đặc biệt mục 5, 6, 7), `text_features(1).py`, `experiments(1).py`, `imbalance(1).py`, `evaluation(1).py`, `02_thuc_nghiem_va_demo(1).ipynb` (đặc biệt cell 12–13, 14–16, 20, 32) và `README(1).md`.  
> **Phân biệt nguồn:** Các giá trị **38.233, 30.759, các CV F1 và các tham số B1–B4** dưới đây là từ *output đã lưu của Notebook*, **không phải kết quả chạy lại toàn dự án**. Các bảng vector nhỏ, xác suất và trọng số SVM bên dưới là **ví dụ tự dựng và đã kiểm chứng bằng Python/scikit-learn trong phiên làm việc này**.  
> **Phiên bản:** Tài liệu nhóm dẫn đường đến **scikit-learn 1.6.1 trong `.venv`**; máy dùng để kiểm chứng ví dụ minh họa hiện cài **scikit-learn 1.8.0**. Đường xử lý cơ bản và kết quả ví dụ giữ như mô tả; **số dòng và một số nội bộ thư viện có thể thay đổi theo phiên bản**. Mã nguồn trong nhóm được diễn giải theo tài liệu 1.6.1, không tuyên bố đã đọc trực tiếp mã C/C++ biên dịch của Liblinear.

---

## 1. Học xong file này cần trả lời được gì?

1. File `classifiers.py` làm việc gì; khi nào nó **chưa học**, khi nào thực sự học?
2. Đầu vào của `MultinomialNB.fit()` và `LinearSVC.fit()` là gì? Phân biệt chuỗi văn bản, ma trận đặc trưng `X`, nhãn `y`.
3. Thuật toán Naive Bayes dùng định lý Bayes và giả định độc lập có điều kiện như thế nào?
4. Phân biệt **MultinomialNB** với việc chỉ nhân xác suất chung chung: nó cộng đặc trưng theo lớp, làm mịn bằng alpha, giữ log prior và log likelihood.
5. Những phương thức thư viện NB nào được gọi: `_BaseDiscreteNB.fit`, `_count`, `_update_feature_log_prob`, `_update_class_log_prior`, `_joint_log_likelihood`, `predict`, `predict_proba`?
6. Nói rõ `class_count_`, `feature_count_`, `class_log_prior_`, `feature_log_prob_`, `classes_` được học từ đâu.
7. Tự tính một câu thành **hai log-score** và xác suất dự đoán Spam; hiểu vì sao `predict_proba` **không mặc nhiên là xác suất đã hiệu chuẩn**.
8. SVM là gì; siêu phẳng, margin, support vectors về mặt lý thuyết, **linear** trong dữ liệu nhiều chiều là sao?
9. `LinearSVC` dùng dữ liệu X/y để học `coef_`, `intercept_`, `n_iter_`; vai trò của `C`, `dual=False`, `max_iter=20000`, `class_weight`.
10. Những phương thức thư viện SVM nào được gọi: `LinearSVC.fit`, `_get_liblinear_solver_type`, `_fit_liblinear`, `liblinear.train_wrap`, `decision_function`, `predict`?
11. Tự tính score = `X @ coef_.T + intercept_` và giải thích vì sao score **không phải xác suất**.
12. Phân biệt `clf__alpha` và `clf__C`, hiểu vì sao `GridSearchCV` thử nhiều giá trị và chọn theo **CV F1 Spam**.
13. Giải thích sự khác nhau giữa bốn mô hình B1–B4, kết quả Notebook và kết nối S1/S2/S3.
14. Phân biệt huấn luyện (`fit`), dự đoán (`predict`, `predict_proba`, `decision_function`) và đánh giá (`evaluation.py`).

**Giới hạn trách nhiệm:** Bạn nhận dataset đã gán nhãn và cột chuỗi token; không cần học lại toàn bộ mã cào Facebook/tiền xử lý/gán nhãn ở Notebook 01. Notebook 03 viBERT chỉ liên quan đến việc dùng cùng chia Train/Test để so sánh công bằng.

---

## 2. Tổng thể: từ chuỗi token tới một classifier đã học

```text
Notebook 02 — cell 9/11
  TEXT[pre] = chuỗi token mỗi bình luận; train_idx, y_train, CV_FOLDS
         |
         v
Notebook 02 — cell 13:  for feature in FEATURES: for algorithm in MODELS
  fit_experiment(pre, feature, algorithm, X_train_text, y_train, CV_FOLDS)
         |
         v
experiments.make_pipe(...)
     /                                   \
text_features.make_vectorizer       classifiers.make_classifier
     |                                   |
CountVectorizer/TfidfVectorizer     MultinomialNB/LinearSVC
chưa fit                           chưa fit + lưới alpha/C
     \                                   /
        Pipeline([('vec', vec), ('clf', clf)])
                    |
              GridSearchCV.fit(X_text, y)
                    |
    TRÊN MỖI FOLD TRAIN:
       vec.fit_transform(X_text_fold) -> X_matrix (n_rows x n_features)
       clf.fit(X_matrix, y_fold)      -> tham số classifier đã học
                    |
    FOLD VALIDATION:
       vec.transform(X_text_valid) -> cùng số cột với Train
       clf.predict(X_matrix_valid) -> nhãn 0/1
       precision/recall/F1 Spam -> điểm CV
                    |
      Chọn n-gram + alpha/C bằng CV F1, fit lại trên toàn Train
                    |
     record['_model'] = fitted Pipeline; record['_threshold'] = 0.5/0.0
                    |
      evaluation.py trên Test: score -> ngưỡng -> nhãn -> metrics
                    |
      Notebook 02 cell 32 lưu .joblib để modeling/predict.py dùng lại
```

**Nhấn mạnh:** **`classifiers.py` KHÔNG có `classifier.fit()`**. Trong file này chỉ tạo đối tượng và danh sách giá trị tham số. `GridSearchCV(...).fit(X, y)` trong `experiments.py` mới gián tiếp gọi `clf.fit()` rất nhiều lần.

### 2.1. Biến đầu vào là gì?

- `X_train_text`: mảng các **chuỗi token** từ Notebook; chưa là số.
- `y_train`: mảng các nhãn **0=Ham, 1=Spam** cùng thứ tự với `X_train_text`.
- `X_matrix`: **ma trận thưa số** do `vec.fit_transform(X_train_text)` tạo ra bên trong Pipeline.
- `clf.fit(X_matrix, y_train)`: classifier nhận ma trận số và nhãn tương ứng; **không nhận trực tiếp `raw['text']`**.
- Trong inference/validation: `vec.transform(text_mới)` tạo ma trận cùng số cột; classifier dùng các tham số đã học, **không fit lại**.

Ví dụ về kích thước: nếu một fold Train có 24.815 câu và vectorizer học 12.000 đặc trưng, thì `clf.fit()` nhận `X.shape=(24815, 12000)`, `y.shape=(24815,)`. **12.000 chỉ là ví dụ**; số đặc trưng thật phụ thuộc từ điển, min_df và n-gram trong mỗi fold.

---

## 3. Đọc TOÀN BỘ `classifiers(1).py`, từng dòng quan trọng

```python
"""Giai đoạn 1b: chọn bộ phân loại và các giá trị tham số cần thử.

File này CHƯA huấn luyện. experiments.py ghép bộ phân loại với bộ tạo vector,
sau đó gọi fit trên từng fold train và chọn tham số bằng CV.
"""

from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

MODELS = ("Naive Bayes", "SVM")
NB_ALPHAS = [0.1, 0.3, 1.0]
SVM_CS = [0.1, 0.3, 1.0, 3.0]


def make_classifier(model_name, *, seed=42, class_weight=None):
    if model_name == "Naive Bayes":
        if class_weight is not None:
            raise ValueError("class_weight chỉ dùng cho SVM trong thí nghiệm này")
        classifier = MultinomialNB()
        grid = {"clf__alpha": NB_ALPHAS}
    elif model_name == "SVM":
        classifier = LinearSVC(dual=False, max_iter=20000, random_state=seed)
        grid = {"clf__C": SVM_CS,
                "clf__class_weight": [class_weight]}
    else:
        raise ValueError(f"Bộ phân loại không hỗ trợ: {model_name}")
    return classifier, grid
```

### 3.1. Hai lệnh import

- `from sklearn.naive_bayes import MultinomialNB`: lấy lớp Multinomial Naive Bayes từ scikit-learn. Chúng ta **không tự viết bộ ước lượng xác suất NB từ đầu**.
- `from sklearn.svm import LinearSVC`: lấy lớp Linear Support Vector Classification. Chúng ta **không tự viết thuật toán tối ưu trọng số SVM từ đầu**.

### 3.2. `MODELS = (...)` là gì?

`MODELS` là một tuple chứa **hai tên thuật toán được phép dùng**. Notebook cell 13 duyệt:

```python
for algorithm in MODELS:
    record = fit_experiment(..., algorithm, ...)
```

Do đó nó sẽ gọi `fit_experiment()` với hai tên: `"Naive Bayes"` và `"SVM"`. Đây là **tên cấu hình**, không phải hai mô hình đã huấn luyện.

### 3.3. `NB_ALPHAS` và `SVM_CS` là gì?

- `NB_ALPHAS = [0.1, 0.3, 1.0]`: danh sách **tham số làm mịn** cho MultinomialNB để thử.
- `SVM_CS = [0.1, 0.3, 1.0, 3.0]`: danh sách **mức phạt sai số/độ mạnh regularization** của LinearSVC để thử.

**Đây là siêu tham số do con người định ra trước**, không phải giá trị mà mô hình tự học như `feature_log_prob_` hay `coef_`.

### 3.4. Các đối số của `make_classifier(model_name, *, seed=42, class_weight=None)`

| Đối số | Kiểu/giá trị | Vai trò | Dùng ở đâu? |
|---|---|---|---|
| `model_name` | `"Naive Bayes"` hoặc `"SVM"` | Xác định thuật toán cần tạo | `if/elif` |
| Dấu `*` | cú pháp Python | Các đối số sau bắt buộc truyền theo tên nếu muốn thay | `seed=...`, `class_weight=...` |
| `seed=42` | số nguyên | Truyền sang `random_state` của SVM; không dùng khi `dual=False` | Nhánh SVM |
| `class_weight=None` | `None` hoặc `"balanced"` | Điều chỉnh trọng số lớp trong SVM; mặc định không điều chỉnh | B2/B4 và S1 |

Ví dụ `make_classifier("SVM", seed=42, class_weight="balanced")` hợp lệ. `make_classifier("SVM", 42, "balanced")` không hợp lệ do có dấu `*`.

### 3.5. Nhánh Naive Bayes

```python
classifier = MultinomialNB()
grid = {"clf__alpha": NB_ALPHAS}
```

- `MultinomialNB()` **chỉ tạo estimator chưa fit** với các giá trị mặc định như `alpha=1.0`, `fit_prior=True`.
- `grid` nói với GridSearchCV: thử `alpha` bằng **0.1, 0.3 và 1.0**.
- Dòng `if class_weight is not None: raise ValueError(...)` ngăn việc nhầm đối số SVM với Naive Bayes. MultinomialNB **không có tham số khởi tạo `class_weight`** như LinearSVC trong dự án này.
- Chưa biết lớp Spam/Ham phân bố ra sao, chưa có `feature_count_`, `feature_log_prob_`.

### 3.6. Nhánh SVM

```python
classifier = LinearSVC(dual=False, max_iter=20000, random_state=seed)
grid = {"clf__C": SVM_CS, "clf__class_weight": [class_weight]}
```

- Tạo **LinearSVC chưa fit**, không có `coef_` hoặc `intercept_` đã học.
- `dual=False`: dùng phương án giải **primal** theo cấu hình nhóm, không phải bài toán dual.
- `max_iter=20000`: cho phép tối đa 20.000 lượt lặp trong tối ưu; không phải 20.000 epoch dữ liệu theo định nghĩa của mạng neuron.
- `random_state=seed`: được truyền để cấu hình nhất quán, nhưng với `dual=False`, scikit-learn ghi nhận giá trị này **không tác động đến thuật toán tối ưu**.
- `clf__C`: GridSearchCV thử 4 giá trị C.
- `clf__class_weight`: trong B2/B4 chỉ là `[None]` (một lựa chọn); trong S1 là `["balanced"]`.

### 3.7. `return classifier, grid`: hai kết quả riêng

```python
nb, nb_grid = make_classifier("Naive Bayes")
# nb: MultinomialNB() CHƯA FIT
# nb_grid: {'clf__alpha': [0.1, 0.3, 1.0]}

svm, svm_grid = make_classifier("SVM")
# svm: LinearSVC(dual=False, max_iter=20000, random_state=42) CHƯA FIT
# svm_grid: {'clf__C': [0.1, 0.3, 1.0, 3.0],
#            'clf__class_weight': [None]}
```

Không được nhầm: danh sách C/alpha trong grid không phải các giá trị đã học; chúng là các **ứng viên để huấn luyện rồi so sánh**.

### 3.8. Vì sao khóa là `clf__alpha`/`clf__C`, không phải `alpha`/`C`?

`experiments.make_pipe()` đặt tên bước của Pipeline là `"clf"`:

```python
pipe = Pipeline([("vec", vectorizer), ("clf", classifier)], memory=cache)
```

Cú pháp `clf__alpha` có nghĩa **thuộc tính `alpha` của bước `clf`**; `clf__C` nghĩa là **thuộc tính `C` của bước `clf`**. `vec__ngram_range` ở bước trước là tham số của **vectorizer**. GridSearchCV tự gọi `set_params()` lên các bản clone của Pipeline khi thử từng cấu hình. Có thể kiểm tra qua `pipe.get_params()`.

---

## 4. Khái niệm chung: classifier học những gì từ BoW/TF-IDF?

### 4.1. Dữ liệu có dạng nào?

Mỗi bình luận có `m` đặc trưng, ví dụ theo vocabulary `['làm', 'lịch', 'nhận', 'xin']`:

```text
Bình luận "nhận làm"   -> [1, 0, 1, 0]   nhãn y=1 (Spam)
Bình luận "xin lịch"   -> [0, 1, 0, 1]   nhãn y=0 (Ham)
```

Toàn bộ tập gồm `n` câu thành ma trận `X` kích thước `n × m`. Nhãn là vector `y` kích thước `n`.

**Khi fit:** `clf.fit(X, y)` nhìn cả đặc trưng **và nhãn thật** để học quan hệ thống kê giữa hai lớp.

**Khi dự đoán:** `clf.predict(X_new)` hoặc `predict_proba`/`decision_function` **không nhận `y_new`**; nhãn đúng của Test chỉ dùng sau đó tại `evaluation.py`.

### 4.2. NB và SVM khác mục tiêu học

| Tiêu chí | MultinomialNB | LinearSVC |
|---|---|---|
| Ý tưởng | Mô hình xác suất đặc trưng **theo lớp** | Học siêu phẳng phân chia hai lớp |
| Tham số được học | Prior lớp và xác suất đặc trưng có điều kiện | Trọng số `w` và bias `b` |
| Khi fit | Cộng đặc trưng theo lớp, làm mịn, lấy log | Tối ưu hàm mất mát + regularization |
| Đầu vào | Ma trận không âm; BoW hoặc TF-IDF | Ma trận số thực; BoW hoặc TF-IDF |
| Score | `predict_proba`: xác suất theo mô hình NB | `decision_function`: score có dấu, **không phải xác suất** |
| Quy tắc mặc định | Spam nếu xác suất Spam ≥ 0.5 trong `evaluation.py` | Spam nếu decision score > 0 trong `evaluation.py` |
| Siêu tham số CV thử | `alpha` | `C` |
| Tính chất nổi bật | Nhanh, dễ truy vết xác suất, giả định độc lập | Học trọng số phân biệt, thường phù hợp biểu diễn TF-IDF nhiều chiều |

Không nên nói: “NB chỉ đếm từ” (thiếu bước làm mịn/tiên nghiệm/dự đoán) hoặc “SVM dùng xác suất” (sai với `LinearSVC` đang dùng).

---

# PHẦN A — MULTINOMIAL NAIVE BAYES

## 5. Lý thuyết Bayes cần biết

### 5.1. Bài toán đang giải

Với một bình luận đã vector hóa `x`, muốn quyết định nó thuộc lớp `c=0 (Ham)` hay `c=1 (Spam)`.

Định lý Bayes:

\[
P(c\mid x)=\frac{P(x\mid c)P(c)}{P(x)}.
\]

Khi **so sánh hai lớp trên cùng x**, mẫu số `P(x)` giống nhau nên đủ so sánh `P(x|c) × P(c)`.

- **Prior `P(c)`**: trong tập học, lớp Ham/Spam phổ biến bao nhiêu?
- **Likelihood `P(x|c)`**: nếu thuộc lớp c, những đặc trưng trong câu có khả năng xuất hiện ra sao?
- **Posterior `P(c|x)`**: sau khi nhìn các đặc trưng, xác suất thuộc lớp c theo mô hình là bao nhiêu?

### 5.2. Tại sao gọi *Naive*?

Mô hình đặt giả định đơn giản hóa rằng các lần xuất hiện đặc trưng độc lập có điều kiện khi đã biết lớp. Dưới mô hình MultinomialNB với đặc trưng đếm `x_j`:

\[
P(x\mid c) \propto \prod_{j=1}^{V} P(t_j\mid c)^{x_j}.
\]

Từ ngữ thực tế có quan hệ (ví dụ “lịch” và “thi” thường đi cùng nhau), nên giả định không hoàn toàn đúng. Dù vậy NB vẫn là baseline tốt trong phân loại văn bản.

**Lưu ý quan trọng:** Công thức dùng số mũ `x_j`, vì từ lặp nhiều lần thì đóng góp nhiều lần; đây là **MultinomialNB**, khác BernoulliNB chỉ quan tâm từ có/không.

### 5.3. Vì sao dùng log?

Nếu nhân rất nhiều xác suất nhỏ, kết quả có thể gần 0 tới mức máy không lưu chính xác được (*underflow*). Dùng log chuyển tích thành tổng:

\[
\log P(c\mid x) = \text{hằng số} + \log P(c) + \sum_j x_j\log P(t_j\mid c).
\]

Scikit-learn lưu `class_log_prior_`, `feature_log_prob_` và tính các **log-score** trước, sau đó chuẩn hóa sang xác suất nếu gọi `predict_proba`.

### 5.4. Vai trò của `alpha` (Laplace/additive smoothing)

Nếu từ `t` chưa xuất hiện trong lớp Spam, đếm bằng 0; dùng xác suất không làm mịn sẽ có `P(t|Spam)=0`. Khi nhân các xác suất, cả biểu thức của lớp Spam bằng 0 chỉ vì một từ chưa gặp.

Công thức MultinomialNB:

\[
P(t_j\mid c)=\frac{N_{cj}+\alpha}{\sum_k N_{ck}+\alpha V}
\]

- `N_cj`: tổng giá trị đặc trưng j của các câu lớp c trong Train.
- `V`: số cột đặc trưng của vectorizer đã fit.
- `alpha`: tham số làm mịn do GridSearchCV thử.

**Không nói `alpha` là tỷ lệ Train/Test, learning rate hay ngưỡng quyết định**; nó là lượng làm mịn pseudo-count trong NB. `alpha=1.0` tương ứng Laplace smoothing; `0.1`/`0.3` là additive smoothing yếu hơn.

---

## 6. Mã scikit-learn của `MultinomialNB`: đọc đường gọi hàm THẬT

Tài liệu `PHI_HIEU_4_LOP_SCIKIT_LEARN.md` dẫn đến `sklearn/naive_bayes.py` của **scikit-learn 1.6.1**. Cần nhớ các phương thức `fit`, `predict`/`predict_proba` được **kế thừa từ lớp cha**, vì vậy Ctrl+Click trực tiếp vào class `MultinomialNB` có thể chưa thấy chúng.

### 6.1. Chuỗi thực thi khi `MultinomialNB.fit(X, y)`

```text
MultinomialNB.fit(X, y)
  = _BaseDiscreteNB.fit(...)       [phương thức kế thừa]
      |
      +-- Kiểm tra X/y; lấy classes_=[0,1]; mã hóa y thành bảng chỉ báo Y
      +-- _init_counters(n_classes, n_features)  -> cấp vùng đếm rỗng
      +-- MultinomialNB._count(X, Y)              -> cộng số liệu theo lớp
      +-- _check_alpha()                           -> kiểm alpha
      +-- _update_feature_log_prob(alpha)          -> log P(term|class)
      +-- _update_class_log_prior(...)             -> log P(class)
      +-- return self (đối tượng đã có tham số học)
```

### 6.2. `_init_counters(n_classes, n_features)`

Hàm cấp bộ nhớ/khởi tạo các bộ đếm về 0 cho từng lớp và từng cột. Nếu có 2 lớp và 5 đặc trưng:

- `class_count_`: chiều `(2,)`, khởi tạo `[0,0]`.
- `feature_count_`: chiều `(2,5)`, ban đầu mọi ô 0.

Đây mới là bộ đếm **chưa chứa dữ liệu**; bước tiếp theo sẽ cập nhật.

### 6.3. `MultinomialNB._count(X, Y)` — thực sự cộng ma trận theo lớp

**Trích đoạn mã nguồn trong tài liệu nhóm (đã rút gọn):**

```python
check_non_negative(X, "MultinomialNB (input X)")
self.feature_count_ += safe_sparse_dot(Y.T, X)
self.class_count_ += Y.sum(axis=0)
```

**Giải thích dòng một:** `check_non_negative` xác nhận ma trận đặc trưng **không có giá trị âm**. BoW đếm số lần từ nên không âm; TF-IDF theo cấu hình project cũng không âm. `MultinomialNB` không phù hợp trực tiếp với vector đặc trưng chứa số âm.

**Dòng hai:** `Y.T @ X` là phép nhân ma trận. `Y` là **one-hot nhãn**, có hai cột Ham/Spam. Nhân `Y.T` với `X` tương đương **cộng mọi vector của những câu cùng lớp**.

**Dòng ba:** `Y.sum(axis=0)` đếm số câu thuộc từng lớp. Đây là lý do `class_count_` đếm số **bình luận**, còn `feature_count_` cộng số **đếm từ/trọng số đặc trưng**.

`safe_sparse_dot` là hàm tiện ích cho phép nhân dữ liệu thưa CSR; nó không ra nhãn Spam/Ham ở bước này.

### 6.4. `_check_alpha()`

Kiểm tra alpha đang dùng có hợp lệ không (với lưới project, alpha đều dương), và chuẩn bị giá trị làm mịn cho bước tính xác suất. `GridSearchCV` đã gán một alpha cụ thể vào từng bản sao estimator trước khi fit.

### 6.5. `_update_feature_log_prob(alpha)` — biến số đếm thành xác suất

**Trích đoạn từ tài liệu nhóm:**

```python
smoothed_fc = self.feature_count_ + alpha
smoothed_cc = smoothed_fc.sum(axis=1)
self.feature_log_prob_ = np.log(smoothed_fc) - np.log(
    smoothed_cc.reshape(-1, 1)
)
```

Giải thích:

1. `smoothed_fc`: cộng alpha vào **từng ô** của bảng tổng đặc trưng theo lớp.
2. `smoothed_cc`: lấy tổng các đặc trưng đã được cộng alpha trong mỗi lớp.
3. `np.log(smoothed_fc) - np.log(smoothed_cc)`: tương đương **log của tỷ số**; nhờ `reshape(-1,1)` mà mẫu số cho mỗi lớp tự phát theo các cột (broadcast).
4. Kết quả `feature_log_prob_` có kích thước `(số lớp, số đặc trưng)`.

Giả sử `feature_count_` hàng Spam `[2,0,2,0]`, `alpha=1`, V=4: hàng đã làm mịn `[3,1,3,1]`, tổng bằng 8, xác suất mỗi cột `[3/8,1/8,3/8,1/8]`.

### 6.6. `_update_class_log_prior(class_prior=...)` — học xác suất lớp

Trong project, `fit_prior=True` theo mặc định và không truyền `class_prior`, nên:

```text
class_count_ = [số Ham, số Spam]
class_prior   = class_count_ / tổng số câu
class_log_prior_ = log(class_prior)
```

Nếu Train có 900 Ham và 100 Spam, prior Spam 0.1, prior Ham 0.9. NB tự tính các prior đó từ Train, không phải công thức chia lớp ở `training_utils.py` đưa trực tiếp vào classifier.

### 6.7. Sau `fit` còn lại những thuộc tính nào?

| Thuộc tính | Kích thước | Chứa gì? | Học từ đâu? |
|---|---|---|---|
| `classes_` | `(2,)` | `[0,1]` ứng với Ham/Spam | `y` |
| `class_count_` | `(2,)` | Số bình luận từng lớp | `y` |
| `feature_count_` | `(2,V)` | Tổng mỗi đặc trưng ở mỗi lớp | `X` + `y` |
| `class_log_prior_` | `(2,)` | Log xác suất lớp | `class_count_` |
| `feature_log_prob_` | `(2,V)` | Log xác suất đặc trưng theo lớp | `feature_count_` + alpha |
| `n_features_in_` | 1 số | Số cột X trong lần fit | `X.shape[1]` |

**Rất dễ nhầm:** `feature_count_` là tổng các cột theo lớp; **nó không phải bảng dự đoán trên Test**, cũng không phải trực tiếp `X_train`.

### 6.8. `_joint_log_likelihood(X)` — cách mô hình tính hai điểm cho câu mới

Trích đoạn thư viện từ tài liệu nhóm:

```python
return safe_sparse_dot(X, self.feature_log_prob_.T) + self.class_log_prior_
```

Công thức này tạo ma trận `n_câu_mới × 2` chứa hai điểm log, tương ứng lớp Ham và Spam:

\[
J_c(x)=\log P(c) + \sum_{j=1}^{V} x_j\log P(t_j\mid c)
\]

Lưu ý: tên nội bộ là **joint log likelihood** nhưng phép tính còn chứa log prior theo lớp; không nên chỉ hiểu là xác suất của từ.

### 6.9. `predict(X)` — biến hai điểm thành nhãn

Mã nhóm dẫn từ `_BaseNB.predict`:

```python
jll = self._joint_log_likelihood(X)
return self.classes_[np.argmax(jll, axis=1)]
```

- Với mỗi bình luận, `argmax` tìm cột có score cao hơn.
- `self.classes_` đổi chỉ số cột thành nhãn thật `0` hoặc `1`.
- Đầu ra là **mảng nhãn** `(n_câu,)`, không phải xác suất.

### 6.10. `predict_proba(X)` — biến hai score thành xác suất mô hình

Tại lớp cha `_BaseNB`, phương thức `predict_proba` gọi `predict_log_proba`, dùng phép chuẩn hóa log-sum-exp rồi exponentiate:

\[
P(c\mid x)=\frac{e^{J_c(x)}}{e^{J_0(x)}+e^{J_1(x)}}.
\]

Với nhãn `[0,1]`, `predict_proba(X)` trả ma trận `(n_câu,2)`:

```text
cột 0: xác suất Ham theo NB
cột 1: xác suất Spam theo NB
```

Trong `evaluation.predict_record`, dự án lấy `[:,1]` để lấy xác suất Spam, sau đó gán Spam khi `score >= 0.5`. `predict_proba` **không bảo đảm xác suất đã hiệu chuẩn (calibrated)**; khi trình bày, gọi là “xác suất theo mô hình”.

---

## 7. VÍ DỤ CHẠY TAY: NB từ văn bản → ma trận → đếm theo lớp → xác suất → dự đoán

**Các câu này hoàn toàn là ví dụ giáo khoa, không phải mẫu thật từ dataset PTIT.** Để dễ tính, cố tình cho mỗi loại có hai câu giống nhau, giúp các từ đạt `min_df=2`:

| Dòng | Chuỗi token | Nhãn |
|---|---|---|
| 0 | `nhận làm` | 1 = Spam |
| 1 | `nhận làm` | 1 = Spam |
| 2 | `xin lịch` | 0 = Ham |
| 3 | `xin lịch` | 0 = Ham |

Dùng `CountVectorizer(token_pattern=TOKEN_PATTERN, min_df=2, ngram_range=(1,1), ...)` từ `text_features.py`.

### 7.1. Tạo vocabulary và ma trận BoW

Từ điển theo thứ tự cột:

```python
['làm', 'lịch', 'nhận', 'xin']
```

Ma trận `X`:

| Dòng | làm | lịch | nhận | xin | y |
|---|---:|---:|---:|---:|---:|
| 0 | 1 | 0 | 1 | 0 | 1 |
| 1 | 1 | 0 | 1 | 0 | 1 |
| 2 | 0 | 1 | 0 | 1 | 0 |
| 3 | 0 | 1 | 0 | 1 | 0 |

`X.shape=(4,4)`, `y.shape=(4,)`.

### 7.2. Đổi y thành bảng chỉ báo Y

Vì `classes_=[0,1]`:

```text
Y = [[0,1],  # câu 0 là Spam
     [0,1],  # câu 1 là Spam
     [1,0],  # câu 2 là Ham
     [1,0]]  # câu 3 là Ham
```

### 7.3. `_count`: lấy `Y.T @ X`

```text
feature_count_ =
            làm  lịch nhận xin
 Ham   [      0,   2,   0,  2 ]
 Spam  [      2,   0,   2,  0 ]

class_count_ = [2,2]
```

Tức Ham có tổng 2 lần `lịch`, 2 lần `xin`; Spam có tổng 2 lần `làm`, 2 lần `nhận`.

### 7.4. Làm mịn với alpha=1

V=4, tổng đặc trưng mỗi lớp là 4; cộng alpha=1 vào **mỗi trong bốn cột** khiến mẫu số bằng `4 + 4×1 = 8`.

| Cột | P(term \| Ham) | P(term \| Spam) |
|---|---:|---:|
| làm | 1/8 = 0.125 | 3/8 = 0.375 |
| lịch | 3/8 = 0.375 | 1/8 = 0.125 |
| nhận | 1/8 = 0.125 | 3/8 = 0.375 |
| xin | 3/8 = 0.375 | 1/8 = 0.125 |

Prior Ham=0.5, Spam=0.5 do số bình luận cân bằng.

### 7.5. Dự đoán bình luận mới `nhận làm ngay`

`CountVectorizer.transform()` dùng lại từ điển cũ, nên từ `ngay` (chưa có cột) bị bỏ. Vector mới:

```text
x = [1,0,1,0]
```

Tính điểm xác suất **chưa chuẩn hóa**:

```text
Score_Ham  = 0.5 × (1/8) × (1/8) = 0.0078125
Score_Spam = 0.5 × (3/8) × (3/8) = 0.0703125
```

Chuẩn hóa hai score:

\[
P(Spam\mid x)=\frac{0.0703125}{0.0703125+0.0078125}=\mathbf{0.9}.
\]

Vậy:

```text
predict_proba(x) = [0.1, 0.9]   # thứ tự [Ham, Spam]
predict(x)       = 1            # Spam
```

Đúng với scikit-learn khi chạy ví dụ minh họa.

### 7.6. Nếu câu là `xin lịch`?

Ma trận mới `[0,1,0,1]`. Xác suất Spam=0.1, Ham=0.9 → nhãn 0 (Ham).

### 7.7. Nếu câu là `nhận lịch`?

Vector `[0,1,1,0]` chứa một từ nghiêng Spam và một từ nghiêng Ham. Hai score bằng nhau → mỗi lớp 0.5. Trong ví dụ, `nb.predict` của sklearn chọn lớp 0 nếu hai score bằng nhau (thứ tự lớp đã sắp `[0,1]`); còn `evaluation.predict_record` của dự án dùng `score >= 0.5` nên sẽ chọn **Spam=1 khi đúng 0.5**. **Đây là một bất đồng biên hiếm nhưng có thật** giữa `nb.predict()` và quy tắc ngưỡng trong `evaluation.py`; nên nhớ khi đối chiếu kết quả từng câu.

### 7.8. Vì sao trong project có thể dùng TF-IDF + MultinomialNB?

MultinomialNB chuẩn gắn với số đếm rời rạc, nhưng scikit-learn chấp nhận **đặc trưng không âm** như TF-IDF. Khi đó, `_count` vẫn làm `Y.T @ X`, nhưng các ô trong `feature_count_` là **tổng trọng số TF-IDF**, không phải số lần từ xuất hiện nguyên. Vì thế kết quả B3 (TF-IDF + NB) vẫn chạy hợp lệ. Đừng nói rằng NB ở B3 “đếm số từ y hệt B1”.

---

# PHẦN B — LINEAR SUPPORT VECTOR CLASSIFIER

## 8. SVM là gì? Vì sao dùng cho bình luận?

BoW/TF-IDF biến bình luận thành một điểm trong không gian nhiều chiều, mỗi chiều là một từ/ngữ. SVM học một **ranh giới tuyến tính** để tách các điểm thuộc Ham và Spam.

Với hai đặc trưng, có thể vẽ đường thẳng. Với ba đặc trưng là mặt phẳng. Với hàng nghìn đặc trưng là **siêu phẳng** (hyperplane) — không thể vẽ trực tiếp nhưng công thức giữ nguyên:

\[
f(x)=w^Tx+b.
\]

- `x`: vector của một bình luận.
- `w`: trọng số của từng cột từ/ngữ, được **học từ Train**.
- `b`: hệ số bias/intercept được **học từ Train**.
- `f(x)`: decision score, so với ngưỡng để ra Ham/Spam.

### 8.1. Ý nghĩa dấu của score

Trong dự án `classes_=[0,1]`:

```text
f(x) > 0  -> lớp 1 = Spam
f(x) <= 0 -> lớp 0 = Ham
```

Đây là cách `LinearSVC.predict` quyết định nhãn mặc định. Với chiến lược S3, `evaluation.py` dùng **ngưỡng đã tối ưu từ Train**, ví dụ `-0.1776`, thay cho 0.

**Không nói `f(x)=1.3` tương đương “130% Spam” hoặc “xác suất 1.3”** — score SVM không phải xác suất.

### 8.2. Support vectors và margin về mặt lý thuyết

- **Margin**: khoảng cách từ ranh giới đến các điểm học quan trọng gần biên ở cách giải thích SVM truyền thống.
- **Support vectors**: các điểm nằm sát biên hoặc vi phạm biên, đóng vai trò quan trọng trong nghiệm SVM tối ưu.
- **Soft margin**: cho phép có điểm vi phạm margin hoặc bị phân loại sai để cân bằng giữa ranh giới và khả năng tổng quát.

Trong dữ liệu bình luận thực tế, hai lớp chồng lấn (chẳng hạn “inbox” có thể là Spam nhưng cũng xuất hiện trong Ham), vì vậy không nên kỳ vọng có thể chia hoàn toàn sạch hai lớp.

**Quan trọng:** `LinearSVC` dùng Liblinear và **không cung cấp thuộc tính `support_vectors_` như `sklearn.svm.SVC`**. Bạn có thể giải thích khái niệm support vectors, nhưng không được nói rằng code của nhóm gọi `svm.support_vectors_` để lấy danh sách chúng.

### 8.3. Hàm mất mát của LinearSVC đang dùng

Theo tài liệu nhóm về nhánh `penalty='l2'`, `loss='squared_hinge'`, `dual=False`, một cách viết lý tưởng hóa của hàm mục tiêu là:

\[
\min_{w,b}\; \frac12\|w\|^2 + C\sum_{i=1}^{n} \left[\max(0, 1-y_i^*(w^Tx_i+b))\right]^2,
\]

trong đó `y_i^*` là nhãn toán học **-1 cho Ham, +1 cho Spam**. Dataset gốc vẫn lưu **0 và 1**; thư viện mã hóa nhãn khi tối ưu.

- Phần `1/2 ||w||²`: **regularization**, tránh hệ số phình quá lớn, liên quan tới việc giữ margin lớn.
- Phần `max(0, 1-y_i^*f(x_i))²`: **squared hinge loss**, phạt điểm không đạt margin hoặc bị phân loại sai.
- `C`: mức đánh đổi giữa lỗi và regularization.

**Lưu ý kỹ thuật:** Công thức trên là biểu diễn đơn giản để hiểu mục tiêu, không phải dòng code optimizer. Với `fit_intercept=True` trong Liblinear, intercept được xử lý như một đặc trưng tổng hợp và có thể chịu regularization; khi cần diễn giải đúng tuyệt đối thuật toán nền, phải tham khảo mô tả Liblinear cho phiên bản cài đặt.

### 8.4. C khác alpha như thế nào?

| `alpha` của NB | `C` của SVM |
|---|---|
| Làm mịn thống kê số đếm/trọng số theo lớp | Điều chỉnh mức phạt các điểm vi phạm margin |
| Nhỏ hơn → ít làm mịn hơn | Nhỏ hơn → regularization mạnh hơn tương đối, chấp nhận vi phạm margin nhiều hơn |
| Chỉ áp dụng NB | Chỉ áp dụng SVM |
| Thử `[0.1,0.3,1.0]` | Thử `[0.1,0.3,1.0,3.0]` |

**C lớn không tự động làm Test F1 cao hơn**; tham số được chọn dựa vào validation trên Train.

---

## 9. Mã scikit-learn của `LinearSVC`: giải thích THẬT sự đi qua các phương thức nào

Theo tài liệu `PHI_HIEU_4_LOP_SCIKIT_LEARN.md` dùng scikit-learn 1.6.1, đường đọc thư viện gồm `svm/_classes.py`, `svm/_base.py`, `linear_model/_base.py`. Khác NB, phần tối ưu lõi chuyển vào **Liblinear được biên dịch**, nên không có toàn bộ vòng lặp tối ưu bằng Python để đọc qua Ctrl+Click.

### 9.1. Đường gọi của `LinearSVC.fit(X, y)`

```text
LinearSVC.fit(X, y)
    |
    +-- Kiểm tra X/y; lấy classes_ (trong bài: [0,1])
    +-- Chọn cách giải qua _get_liblinear_solver_type
    |      dựa vào penalty='l2', loss='squared_hinge', dual=False
    +-- Gọi _fit_liblinear(X, y, C, class_weight, max_iter, tol, ...)
           |
           +-- Chuẩn bị định dạng dữ liệu, mã hóa nhãn
           +-- Chuẩn bị trọng số lớp nếu class_weight='balanced'
           +-- Gọi liblinear.train_wrap(...)  [thành phần biên dịch]
           +-- Nhận nghiệm w, b và số vòng lặp
    |
    +-- Lưu coef_, intercept_, n_iter_, classes_, ...
    +-- return self
```

### 9.2. Đoạn mã được tài liệu nhóm trích từ `LinearSVC.fit`

```python
self.classes_ = np.unique(y)
self.coef_, self.intercept_, n_iter_ = _fit_liblinear(
    X, y, self.C, self.fit_intercept, self.intercept_scaling,
    self.class_weight, self.penalty, _dual, self.verbose,
    self.max_iter, self.tol, self.random_state, self.multi_class,
    self.loss, sample_weight=sample_weight,
)
```

Các giá trị quan trọng:

- `np.unique(y)`: trong bài phải cho `[0,1]`; lớp dương là `1=Spam`.
- `X`: ma trận BoW hoặc TF-IDF; có thể có hàng nghìn đến hàng chục nghìn cột.
- `y`: nhãn đúng của những câu tương ứng từng hàng.
- `self.C`: ứng viên C được GridSearchCV gán cho lần fit này.
- `self.class_weight`: `None` ở B2/B4; `"balanced"` ở S1.
- `self.penalty`: mặc định `l2` trong cấu hình nhóm.
- `self.loss`: mặc định `squared_hinge`.
- `_dual`: trong bài chọn `False`.
- `max_iter`: 20000 — giới hạn vòng lặp của bộ tối ưu.
- `self.coef_`, `self.intercept_`: kết quả cần thiết để dự đoán câu mới.

### 9.3. `_get_liblinear_solver_type` — chọn bộ giải nào?

Nội bộ kiểm tra tổ hợp tham số như penalty, loss, dual, rồi chọn nhánh Liblinear tương ứng. Theo tài liệu nhóm, với cấu hình **L2 + squared_hinge + dual=False**, solver code là **2**. Đây **không phải nhãn Spam=2**; chỉ là số hiệu thuật toán tối ưu bên trong Liblinear.

### 9.4. `_fit_liblinear` làm gì?

Tóm tắt theo tài liệu nhóm:

1. Chuẩn bị định dạng ma trận để Liblinear có thể nhận (hỗ trợ dữ liệu thưa).
2. Mã hóa nhãn 0/1 để đưa vào bài toán phân loại nhị phân của bộ tối ưu.
3. Xử lý `class_weight`/`sample_weight` nếu có.
4. Truyền dữ liệu cùng C, `tol`, `max_iter`, loại solver sang `liblinear.train_wrap(...)`.
5. Nhận hệ số, intercept và thông tin hội tụ rồi trả về phần Python.

**Giới hạn minh bạch:** tài liệu nhóm không chứa mã nguồn C/C++ của vòng lặp tối ưu Liblinear. Ta có thể giải thích đúng **đầu vào, bài toán tối ưu, thuật toán được chọn và đầu ra**; không nên giả vờ rằng một đoạn Python ngắn cho thấy toàn bộ cách Liblinear cập nhật từng hệ số tại mỗi iteration.

### 9.5. Những thuộc tính SVM lưu sau `fit`

| Thuộc tính | Ý nghĩa | Cách dùng |
|---|---|---|
| `classes_` | `[0,1]` tương ứng Ham/Spam | Giải mã nhãn |
| `coef_` | Mảng trọng số từng đặc trưng, với hai lớp có shape `(1,V)` | Tính decision score |
| `intercept_` | Bias thường có shape `(1,)` | Cộng vào score |
| `n_iter_` | Số vòng tối ưu được báo cáo | Kiểm tra hội tụ |
| `n_features_in_` | Số đặc trưng đầu vào đã fit | Kiểm tra số cột khi predict |

**Lưu ý:** Với LinearSVC, `coef_` có thể âm hoặc dương; đây là trọng số của **đường phân chia**, khác với bảng xác suất theo lớp của NB. Không lấy trực tiếp hệ số âm của SVM đưa vào MultinomialNB.

### 9.6. `decision_function(X)` — tính score bằng ma trận

Theo tài liệu nhóm, phương thức kế thừa từ `LinearClassifierMixin` dùng:

```python
scores = safe_sparse_dot(X, self.coef_.T, dense_output=True) + self.intercept_
```

Phép tính tương đương:

```text
Với một câu:
score = x_1*w_1 + x_2*w_2 + ... + x_V*w_V + b
```

Giả sử hai câu có 4 đặc trưng và `coef_=[0.4,-0.4,0.4,-0.4]`, `b=0`, câu `[1,0,1,0]` có score `0.8` → Spam; câu `[0,1,0,1]` có score `-0.8` → Ham. Những hệ số này là ví dụ minh họa, **không phải trọng số thật của dự án**.

### 9.7. `predict(X)` — chuyển score thành nhãn

Một đoạn mã của lớp cha (diễn giải nhánh hai lớp):

```python
score = self.decision_function(X)
indices = (score > 0).astype(int)
return self.classes_[indices]
```

Đây là hai bước: lấy score và so với 0. Với `classes_=[0,1]`, score dương → Spam.

### 9.8. Vì sao LinearSVC không có `predict_proba()`?

Cấu hình hiện tại tối ưu **ranh giới dự đoán** và trả score có dấu; nó không được định nghĩa mặc định thành xác suất 0–1. Nếu sau này thực sự cần xác suất đã hiệu chuẩn, phải áp thêm quy trình calibration có dữ liệu validation thích hợp. **Dự án này không làm bước đó**. Vì vậy tại `evaluation.py`, SVM dùng `decision_function()` và AP dùng ranking score.

---

## 10. VÍ DỤ CHẠY TAY: LinearSVC từ BoW → w/b → score → Spam/Ham

Ta dùng lại đúng bốn câu và vocabulary ở mục 7:

```text
vocab = ['làm','lịch','nhận','xin']
X = [
  [1,0,1,0],  # Spam
  [1,0,1,0],  # Spam
  [0,1,0,1],  # Ham
  [0,1,0,1],  # Ham
]
y = [1,1,0,0]
```

### 10.1. Fit một LinearSVC với cấu hình project

```python
svm = LinearSVC(dual=False, max_iter=20000, random_state=42, C=1.0)
svm.fit(X, y)
```

Ở ví dụ nhỏ này, tôi đã chạy kiểm chứng và nhận:

```text
svm.classes_   = [0, 1]
svm.coef_      = [[ 0.44444444, -0.44444444, 0.44444444, -0.44444444 ]]
svm.intercept_ = [ 0.0 ]
```

**Chú ý:** Đây là nghiệm do Liblinear tìm ra từ bộ dữ liệu ví dụ, không phải hệ số đặt thủ công. Nó có thể thay đổi rất nhỏ do phiên bản/sai số tối ưu.

### 10.2. Tính score cho `nhận làm ngay`

Vectorizer bỏ từ `ngay` ngoài từ điển → `x=[1,0,1,0]`.

\[
score = 1(0.444444)+0(-0.444444)+1(0.444444)+0(-0.444444)+0 \approx \mathbf{0.888889}.
\]

`score > 0` nên dự đoán **Spam=1**.

### 10.3. Tính score cho `xin lịch`

`x=[0,1,0,1]`:

\[
score=0-0.444444+0-0.444444\approx\mathbf{-0.888889}.
\]

`score ≤ 0` nên dự đoán **Ham=0**.

### 10.4. Câu pha trộn `nhận lịch`

`x=[0,1,1,0]`: hai đóng góp `-0.444444 + 0.444444 ≈ 0`, nằm sát ranh giới. Ví dụ đối xứng này có score thực tế bằng 0 theo độ chính xác hiển thị; mô hình chọn Ham khi `score <= 0`.

### 10.5. Lưu ý về bản chất hệ số

`0.444444` ở đây **không phải 44,44%**. Nó là hệ số để lấy tích vô hướng. Tất cả đặc trưng cùng đóng góp vào score. Vì vậy SVM học khác NB dù dùng **cùng X/y**.

### 10.6. Khi dùng TF-IDF thay vì BoW

Nếu thay vectorizer bằng TF-IDF, `X` chứa trọng số thập phân thay số đếm. `LinearSVC.fit(X_tfidf, y)` tối ưu các hệ số mới. Không thể lấy `coef_` của SVM fit trên BoW rồi áp bừa cho vector TF-IDF ở mô hình khác, kể cả có cùng số cột, vì biểu diễn và phân bố giá trị đã đổi.

---

## 11. Code chạy thử đầy đủ: nhìn rõ các thuộc tính học được của NB và SVM

**Đoạn này chạy độc lập, không cần dataset Facebook và không ghi đè file dự án.** Nó cố tình dùng cùng cấu hình `TOKEN_PATTERN`, `min_df`, `ngram_range`, `dtype` như `text_features.py` và cấu hình SVM trong `classifiers.py`.

```python
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

TOKEN_PATTERN = r"(?u)\b\w+\b|[^\w\s]"
train_text = ["nhận làm", "nhận làm", "xin lịch", "xin lịch"]
y_train = np.array([1, 1, 0, 0])  # 1=spam, 0=ham
new_text = ["nhận làm ngay", "xin lịch", "nhận lịch"]

# 1. Vectorizer học từ điển từ TRAIN, tạo ma trận số
vec = CountVectorizer(token_pattern=TOKEN_PATTERN,
                      min_df=2, ngram_range=(1,1),
                      lowercase=True, dtype=np.float64)
X_train = vec.fit_transform(train_text)
X_new = vec.transform(new_text)
print("Vocab:", vec.get_feature_names_out())
print("X_train:\n", X_train.toarray())
print("X_new:\n", X_new.toarray())

# 2. Naive Bayes
nb = MultinomialNB(alpha=1.0)
nb.fit(X_train, y_train)
print("NB classes_:", nb.classes_)
print("NB class_count_:", nb.class_count_)
print("NB feature_count_:\n", nb.feature_count_)
print("NB P(term | class):\n", np.exp(nb.feature_log_prob_))
print("NB prior:", np.exp(nb.class_log_prior_))
print("NB predict_proba:\n", nb.predict_proba(X_new))
print("NB predict:", nb.predict(X_new))

# 3. SVM
svm = LinearSVC(dual=False, max_iter=20000,
                random_state=42, C=1.0)
svm.fit(X_train, y_train)
print("SVM classes_:", svm.classes_)
print("SVM coef_:", svm.coef_)
print("SVM intercept_:", svm.intercept_)
print("SVM decision scores:", svm.decision_function(X_new))
print("SVM predict:", svm.predict(X_new))
```

### Output chính đã đối chiếu

```text
Vocab: ['làm' 'lịch' 'nhận' 'xin']
NB class_count_: [2. 2.]
NB feature_count_:
 [[0. 2. 0. 2.]
  [2. 0. 2. 0.]]
NB P(term|class):
 [[0.125 0.375 0.125 0.375]
  [0.375 0.125 0.375 0.125]]
NB predict_proba(new_text):
 [[0.1 0.9]
  [0.9 0.1]
  [0.5 0.5]]
SVM coef_:
 [[ 0.44444444 -0.44444444  0.44444444 -0.44444444]]
SVM intercept_: [0.]
SVM decision_function(new_text):
 [ 0.88888889 -0.88888889  0.        ]
```

Ở trường hợp xác suất 0.5/0.5, `nb.predict()` thường chọn lớp đầu `[0]`, nhưng `evaluation.py` của nhóm dùng `>=0.5` nên chọn Spam nếu đúng ngưỡng. Không nên đồng nhất hai đường dự đoán trong mọi trường hợp biên.

### 11.1. Thử n-gram `(1,2)` thì điều gì đổi?

Cùng bốn câu, `min_df=2`:

```text
ngram=(1,1): ['làm','lịch','nhận','xin']              -> 4 cột
ngram=(1,2): ['làm','lịch','nhận','nhận làm','xin','xin lịch'] -> 6 cột
```

Thêm cột bigram khiến `X` đổi từ 4 thành 6 cột. **NB và SVM sẽ fit lại với ma trận 6 cột**, nên `feature_log_prob_`/`coef_` cũng đổi kích thước. Đây chính là lý do n-gram là siêu tham số của *toàn Pipeline* và phải được thử cùng alpha/C, không tự chọn theo cảm giác.

Với `ngram=(1,2)`, kiểm chứng trên bộ ví dụ với `alpha=1`:

```text
P_NB(Spam | "nhận làm ngay") ≈ 0.964286
SVM decision score("nhận làm ngay") ≈ +0.923077
```

Hai giá trị tăng so với `(1,1)` **trong ví dụ giả lập**, nhưng điều đó **không có nghĩa `(1,2)` luôn tốt hơn trên mọi dataset**. Trên PTIT, lựa chọn được thực hiện bằng Cross-Validation.

---

## 12. Dùng TF-IDF với NB/SVM: dữ liệu đi qua bằng cách nào?

Sau khi `text_features.py` tạo vectorizer, Notebook không phải tự xuất ma trận số ra CSV. Ví dụ khi tạo một Pipeline:

```python
pipe = Pipeline([
    ("vec", TfidfVectorizer(token_pattern=TOKEN_PATTERN,
                            min_df=2, ngram_range=(1,2),
                            sublinear_tf=True, smooth_idf=True,
                            norm="l2")),
    ("clf", LinearSVC(dual=False, max_iter=20000, C=0.3))
])
```

Khi gọi `pipe.fit(X_text, y)`, nội bộ:

1. `vec.fit_transform(X_text)`: học vocabulary, IDF và trả ma trận thưa **không âm**.
2. `clf.fit(X_matrix, y)`: học `coef_`, `intercept_` của SVM từ ma trận và nhãn.
3. Pipeline lưu cả vectorizer **đã fit** và classifier **đã fit**.

Khi gọi `pipe.decision_function(["nhận làm bài tập"])`, nội bộ:

1. `vec.transform(...)` dùng lại vocabulary và IDF.
2. `clf.decision_function(X_matrix_new)` lấy `X_new @ coef_.T + intercept_`.

Nếu đổi `clf` sang MultinomialNB thì phần vectorizer giữ nguyên; NB sẽ học `feature_count_` và các log xác suất từ ma trận TF-IDF không âm.

### 12.1. Nơi kiểm tra từng bước đã fit

```python
# giả sử record được fit_experiment trả về
fitted_pipe = record["_model"]
vec = fitted_pipe.named_steps["vec"]
clf = fitted_pipe.named_steps["clf"]
print(vec.get_feature_names_out()[:20])
print(clf.classes_)

# Nếu NB:
print(clf.class_count_, clf.feature_count_.shape)
print(clf.feature_log_prob_.shape)

# Nếu SVM:
print(clf.coef_.shape, clf.intercept_, clf.n_iter_)
```

**Không được xem `coef_` hoặc `feature_log_prob_` trên classifier vừa khởi tạo nhưng chưa fit**, sẽ gặp `AttributeError`.

---

## 13. Liên hệ cụ thể `experiments.py` và Notebook cell 13

### 13.1. Các vòng lặp trong Notebook

```python
PREPROCESS_RESULTS = []
for pre in TEXT:                    # giữ / loại stopwords
    X_train_text = TEXT[pre].iloc[train_idx].to_numpy()
    for feature in FEATURES:        # "BoW", "TF-IDF"
        for algorithm in MODELS:    # "Naive Bayes", "SVM"
            record = fit_experiment(
                pre, feature, algorithm,
                X_train_text, y_train, CV_FOLDS,
                seed=SEED, cache=cache, n_jobs=N_JOBS
            )
            PREPROCESS_RESULTS.append(record)
```

Điều này tạo **8 lời gọi `fit_experiment`** (2 cách stopwords × 2 đặc trưng × 2 thuật toán). Các mã B1–B4 là **bốn tổ hợp cơ sở trong nhánh tiền xử lý được chọn**, không phải 8 tên mô hình khác nhau.

### 13.2. `make_pipe` ghép lưới như thế nào?

```python
vectorizer, vector_grid = make_vectorizer(feat)
classifier, model_grid = make_classifier(model_name, ...)
pipe = Pipeline([("vec", vectorizer), ("clf", classifier)])
grid = {**vector_grid, **model_grid}
```

Ví dụ B1:

```python
{
  "vec__ngram_range": [(1,1), (1,2)],
  "clf__alpha": [0.1, 0.3, 1.0]
}
```

B1 có `2×3=6` cấu hình cần thử. B2 có `2×4×1=8` (n-gram × C × class_weight None). B3 có 6; B4 có 8.

**Mỗi nhánh tiền xử lý:** 6+8+6+8 = 28 cấu hình. 2 nhánh = 56 cấu hình. Với 5 fold = **280 lượt fit trên fold**, chưa tính fit lại cấu hình thắng trong từng `GridSearchCV` và các chiến lược S1/S2/S3.

### 13.3. `GridSearchCV` tính điểm và chọn ở đâu?

```python
search = GridSearchCV(
    pipe, grid,
    scoring={"precision":"precision", "recall":"recall", "f1":"f1"},
    cv=folds,
    refit="f1",
    error_score="raise"
)
search.fit(X_train_text, y_train)
```

Với từng n-gram và alpha/C:

1. Clone Pipeline chưa fit.
2. Lấy 4 fold train, fit vectorizer và classifier.
3. Dự đoán fold validation, so với nhãn thật để tính P/R/F1 cho Spam (`pos_label=1`).
4. Lặp qua 5 fold, lấy **trung bình theo 5 lượt**.
5. Chọn tổ hợp có `mean_test_f1` (trong API tên `test` ở đây có nghĩa **validation trong CV**) cao nhất.
6. `refit="f1"` tự fit lại cấu hình thắng trên **toàn bộ Train**, chưa sử dụng Test độc lập.

**Nơi xem kết quả:**

```python
search.best_params_          # cặp ngram + alpha/C thắng
search.best_estimator_       # pipeline đã fit trên toàn Train
search.cv_results_           # điểm từng cấu hình ở từng fold
search.cv_results_["mean_test_f1"]  # F1 trung bình validation
```

`fit_experiment()` không trả trực tiếp đối tượng `search`, mà trả `record` có `"Tham số"`, `"CV F1 (spam)"`, `"_model"` (best estimator đã fit), `"_threshold"`. Muốn xem đầy đủ `cv_results_` thì thêm lệnh hiển thị/lưu trong hàm hoặc chạy một GridSearchCV minh họa riêng; không có dữ liệu CV đầy đủ trong record đã trả về.

---

## 14. Kết quả THẬT trong Notebook: B1–B4 chọn n-gram và alpha/C ra sao?

**Output đã lưu ở cell 13, nhánh được chọn `"giữ stop words"`:**

| Mã | Vectorizer | Classifier | CV Precision Spam | CV Recall Spam | CV F1 Spam | Tham số thắng |
|---|---|---|---:|---:|---:|---|
| **B1** | BoW | NB | 0.7582 | 0.5153 | **0.6077** | `alpha=1.0`, `ngram=(1,2)` |
| **B2** | BoW | SVM | 0.7290 | 0.6901 | **0.7087** | `C=1.0`, `class_weight=None`, `ngram=(1,2)` |
| **B3** | TF-IDF | NB | 0.8263 | 0.4723 | **0.5959** | `alpha=0.1`, `ngram=(1,2)` |
| **B4** | TF-IDF | SVM | 0.9001 | 0.6806 | **0.7736** | `C=0.3`, `class_weight=None`, `ngram=(1,2)` |

Lý do output chỉ hiện `(1,2)`: Notebook in **tham số tốt nhất** của mỗi mô hình, không in từng cấu hình đã thử. Không phải `(1,1)` không được chạy. Nó vẫn được thử trong GridSearchCV nhưng không được chọn trong lần chạy Notebook đã lưu.

**Không được kết luận “SVM luôn tốt hơn NB” chỉ từ bảng này.** Kết luận đúng là **trong phạm vi thử nghiệm và dữ liệu hiện tại**, TF-IDF + SVM cho CV F1 Spam cao nhất trong bốn cấu hình cơ sở.

### 14.1. Tại sao B1 dùng alpha=1.0 nhưng B3 dùng alpha=0.1?

B1 nhận **ma trận đếm BoW**, còn B3 nhận **ma trận trọng số TF-IDF**. Vì độ lớn/phân bố đặc trưng thay đổi, lượng làm mịn tốt nhất có thể đổi. Alpha được GridSearchCV chọn **riêng cho từng Pipeline**, không dùng chung một alpha cho mọi NB.

### 14.2. Tại sao B2 có C=1.0, B4 có C=0.3?

Hai mô hình học trên những biểu diễn đặc trưng khác nhau. Cũng như alpha, C tối ưu theo validation có thể khác nhau. Việc chọn `C=0.3` trong B4 không thể lý giải chỉ bằng một công thức lý thuyết mà phải dựa vào CV F1 thực tế của các cấu hình đã thử.

### 14.3. Kết quả lựa chọn giữ/bỏ stopwords

| Tổ hợp | CV F1 bỏ stopwords | CV F1 giữ stopwords |
|---|---:|---:|
| BoW + NB | 0.5859 | **0.6077** |
| BoW + SVM | 0.6305 | **0.7087** |
| TF-IDF + NB | 0.5835 | **0.5959** |
| TF-IDF + SVM | 0.7562 | **0.7736** |

Nhánh `"giữ stop words"` được chọn bằng **CV**. Đây là dữ liệu Notebook đã lưu, không phải do NB/SVM tự quyết định có bỏ từ dừng hay không.

---

## 15. Kết nối với mất cân bằng S1–S3 (chỉ ở mức cần biết trước bước 05)

B4 dẫn đầu nên Notebook cell 14–16 thử thêm ba chiến lược **trên TF-IDF + SVM**:

- **S1**: `class_weight="balanced"`; vẫn LinearSVC, nhưng phạt lỗi từ lớp ít mẫu nặng hơn khi fit.
- **S2**: giảm số mẫu Ham **chỉ trong fold train**, giữ validation/Test nguyên; vẫn dùng LinearSVC.
- **S3**: giữ siêu tham số B4, dùng score ngoài fold (OOF) để chọn **ngưỡng** thay vì mặc định 0; vẫn dùng LinearSVC.

Kết quả CV đã lưu:

| Mã | Chiến lược | CV F1 Spam | Ngưỡng |
|---|---|---:|---:|
| B4 | Không điều chỉnh | 0.7736 | 0 |
| S1 | Balanced | 0.7580 | 0 |
| S2 | Under-sampling 2:1 | 0.7629 | 0 |
| **S3** | Chọn ngưỡng | **0.7876** | **-0.1776** |

**Cảnh báo phương pháp:** S3 dùng cùng các score OOF để lựa chọn ngưỡng **và** tính CV F1 của ngưỡng đã chọn; đó không phải nested CV độc lập. Điểm Test được giữ riêng là đánh giá quan trọng để biết mô hình hoạt động ngoài Train ra sao. Không nói S3 chắc chắn hơn các mô hình khác trong mọi tình huống.

**Điểm cốt lõi cho bước này:** S1/S2/S3 **không thay thế `LinearSVC` bằng thuật toán khác**. Chúng thay đổi cách fit/trọng số/dữ liệu/ngưỡng xung quanh cùng họ SVM. Phần chi tiết sẽ học tại `imbalance.py`.

---

## 16. Dự đoán trong `evaluation.py`: NB và SVM được gọi khác nhau ra sao?

```python
def predict_record(record, X):
    model = record["_model"]
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(X)[:, 1]
        threshold = record.get("_threshold", 0.5)
        predictions = (scores >= threshold).astype(int)
    else:
        scores = model.decision_function(X)
        threshold = record.get("_threshold", 0.0)
        predictions = (scores > threshold).astype(int)
    return predictions, scores
```

`record["_model"]` là **Pipeline**, nên `X` ở lời gọi `predict_record` là **chuỗi token**. Pipeline tự làm `vec.transform(X)` trước khi gọi bộ phân loại.

- Với **NB**, lấy `predict_proba(X)[:,1]` vì lớp 1=Spam; ngưỡng mặc định 0.5; quy tắc Spam là `>=0.5`.
- Với **SVM**, lấy `decision_function(X)` là score thực có dấu; ngưỡng mặc định 0; quy tắc Spam là `>0`.
- Sau đó `evaluate(record, X_test, y_test)` so hai mảng `predictions` và nhãn thật `y_test` để tính Accuracy, Precision, Recall, F1; AP dùng **scores**.
- Tập Test **không được đưa vào `fit`**, không dùng chỉnh alpha/C/threshold.

### 16.1. Lưu ý quan trọng về NB ở đúng ngưỡng 0.5

`MultinomialNB.predict` mặc định chọn **lớp có log-score lớn nhất**; nếu bằng nhau và `classes_=[0,1]`, có thể chọn Ham=0. Nhưng `evaluation.predict_record` gán Spam cho `score >= 0.5` kể cả đúng 0.5. Khi so hai phương thức trên dữ liệu cân bằng/score tie, kết quả có thể khác. Quy tắc báo cáo của project là **`predict_record`**, không phải nhất thiết `classifier.predict`.

### 16.2. `score` và nhãn dự đoán khác nhau

| Model | Score ví dụ | Ngưỡng | Nhãn |
|---|---:|---:|---|
| NB | 0.91 (xác suất Spam theo mô hình) | 0.5 | 1 = Spam |
| NB | 0.10 | 0.5 | 0 = Ham |
| SVM | +0.75 (decision score) | 0.0 | 1 = Spam |
| SVM | -0.25 | 0.0 | 0 = Ham |
| S3 | -0.10 | -0.1776 | 1 = Spam |

**S3 có thể gán Spam cho score âm nhẹ** vì đã hạ ngưỡng; `LinearSVC` không đổi `decision_function` thành xác suất.

---

## 17. Thực nghiệm TEST thật: các model đã fit đem đi kiểm tra ra sao?

Notebook cell 20 gọi `evaluation.evaluate()` trên `test_idx` sau khi `best` đã được chọn theo CV. Kết quả Test của bốn tổ hợp cơ sở đã lưu:

| Mã | Accuracy | Precision Spam | Recall Spam | F1 Spam | AP |
|---|---:|---:|---:|---:|---:|
| B1 | 0.9141 | 0.7441 | 0.4953 | 0.5947 | 0.7320 |
| B2 | 0.9241 | 0.7110 | 0.6803 | 0.6953 | 0.7493 |
| B3 | 0.9257 | 0.8402 | 0.5142 | 0.6380 | 0.7988 |
| B4 | 0.9419 | 0.8818 | 0.6278 | 0.7334 | 0.8117 |

Các số này **khác CV F1** bởi Test là tập các bình luận khác với các fold validation của Train; không thể gọi CV F1=0.7736 là Test F1=0.7736.

Notebook chọn cuối cùng S3 (TF-IDF+SVM ngưỡng điều chỉnh), Test F1 Spam được báo cáo khoảng **0.7552**. Đây là lựa chọn theo CV và đánh giá sau; **không dùng Test để chọn S3**.

---

## 18. Những lỗi dễ nhầm khi đọc hoặc trình bày code

1. **`make_classifier` không fit**: nó chỉ khởi tạo object + grid. Lệnh fit thực nằm trong `GridSearchCV.fit()` ở `experiments.py`.
2. **`X` tại `fit_experiment` là chuỗi**, còn `X` tại `MultinomialNB.fit`/`LinearSVC.fit` là **ma trận số** do Pipeline tạo.
3. **`alpha` của NB khác `C` của SVM**; không đem dùng chéo.
4. **`grid` là ứng viên**, không phải tham số đã được học hoặc đã thắng. `search.best_params_` mới là cấu hình được chọn.
5. **`class_weight='balanced'` không dùng cho NB trong project**; nhánh NB sẽ raise `ValueError` nếu truyền.
6. **`LinearSVC` không mặc định có `predict_proba()`**; score âm/dương là số thực không giới hạn ở 0–1.
7. **`NB.predict_proba` không mặc nhiên đã hiệu chuẩn**; đừng gọi 0.9 là bảo đảm 90% trên tập ngoài thực tế.
8. **`LinearSVC` không có `support_vectors_` như `SVC`**. Khái niệm support vectors chỉ để hiểu thuật toán.
9. **Không fit trên Test, không học từ vựng Test**. Mỗi lượt CV tự fit vectorizer chỉ trên fold train.
10. **`mean_test_f1` trong cv_results_ là validation CV**, không phải điểm Test cuối cùng.
11. **`max_iter=20000` không bảo đảm mô hình đã hội tụ**; phải kiểm tra `ConvergenceWarning` và `n_iter_`.
12. **`random_state=42` không tự động quyết định chia Train/Test**: phép chia do `training_utils.bucket()` đảm nhiệm; với `dual=False`, random_state còn không dùng ở solver.
13. **SVM nhãn y trong CSV là 0/1, nhưng nhãn trong công thức margin là ±1** — chỉ là cách mã hóa toán học.
14. **TF-IDF + NB hợp lệ trong scikit-learn**, nhưng các tổng đặc trưng là tổng trọng số, không phải số từ nguyên.
15. **C lớn không luôn tốt hơn C nhỏ**. GridSearchCV chọn theo CV F1 Spam của từng kết hợp n-gram/C.
16. **Notebook có hai nhánh stopwords**, nhưng đây là dữ liệu đầu vào hai dạng token từ bước tiền xử lý, không phải `classifiers.py` tự lọc.
17. **Nếu có tie đúng 0.5 ở NB**, `predict_record()` có thể khác `nb.predict()`; trong dự án đường đánh giá là `predict_record()`.
18. **Score SVM ở S3 so với ngưỡng âm**: một score -0.10 có thể là Spam nếu ngưỡng -0.1776.
19. **Đừng nhầm `feature_count_` và `class_count_`**: cái đầu là tổng theo đặc trưng/lớp, cái sau là số văn bản từng lớp.
20. **Không nói các kết quả minh họa nhỏ là kết quả của dataset PTIT**; phân biệt bảng thật Notebook với ví dụ tính tay.

---

## 19. Cách đọc mã thư viện trong VS Code theo đúng tài liệu nhóm

Trong project chọn interpreter `.venv` tương ứng **scikit-learn 1.6.1**. Ctrl+Click/F12 vào tên lớp, sau đó tìm các phương thức liên quan:

| Nội dung | File trong `site-packages/sklearn/` | Vị trí theo tài liệu nhóm (1.6.1) |
|---|---|---|
| `MultinomialNB` class | `naive_bayes.py` | class ~779 |
| `_BaseDiscreteNB.fit` | `naive_bayes.py` | ~715 |
| `MultinomialNB._count` | `naive_bayes.py` | ~887 |
| `MultinomialNB._update_feature_log_prob` | `naive_bayes.py` | ~893 |
| `MultinomialNB._joint_log_likelihood` | `naive_bayes.py` | ~902 |
| `_BaseNB.predict` / `predict_proba` | `naive_bayes.py` | ~90, ~132 |
| `LinearSVC.fit` | `svm/_classes.py` | ~281 |
| `_get_liblinear_solver_type` | `svm/_base.py` | ~1009 |
| `_fit_liblinear` | `svm/_base.py` | ~1066 |
| `decision_function` / `predict` của `LinearClassifierMixin` | `linear_model/_base.py` | ~329, ~359 |

**Không bắt buộc phải thuộc số dòng**; quan trọng là biết tên class cha và nơi chuyển vào Liblinear. Số dòng có thể khác ở bản scikit-learn mới.

### 19.1. Nếu muốn xem phương thức từ terminal

```python
import inspect
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

print(inspect.getfile(MultinomialNB))
print(inspect.getfile(LinearSVC))
print(MultinomialNB.mro())  # xem quan hệ kế thừa
print(LinearSVC.mro())
```

Lưu ý `inspect.getsource(LinearSVC.fit)` chỉ cho phần Python gọi sang Liblinear, không in được optimizer đã biên dịch. Trong file Markdown nhóm, những đoạn trích từ nhiều phương thức đặt gần nhau **không phải một đoạn source code liền mạch để copy chạy**.

---

## 20. Câu hỏi bảo vệ kèm câu trả lời mẫu

**C1. Vì sao có `classifiers.py` trong khi Notebook chỉ gọi `fit_experiment()`?**  
Vì `experiments.make_pipe` cần module cung cấp thuật toán phân loại và lưới alpha/C. `classifiers.py` chuẩn bị estimator chưa fit; `GridSearchCV.fit()` mới huấn luyện.

**C2. Đầu vào thực của `MultinomialNB.fit()` là gì?**  
Ma trận BoW/TF-IDF không âm kích thước số bình luận × số đặc trưng, cộng nhãn 0/1 cho từng hàng. Không phải chuỗi bình luận thô.

**C3. NB học từ đâu ra xác suất Spam?**  
Nó đếm số mẫu Ham/Spam để tính prior, cộng đặc trưng theo nhãn để có `feature_count_`, làm mịn bằng alpha, lưu log xác suất và kết hợp các log xác suất khi dự đoán câu mới.

**C4. Tại sao NB dùng log?**  
Vì tích nhiều xác suất nhỏ dễ underflow; log biến phép nhân thành phép cộng và ổn định tính toán.

**C5. Vai trò của alpha?**  
Tham số làm mịn các thống kê đặc trưng, tránh xác suất của từ chưa gặp bằng 0. Giá trị 0.1/0.3/1.0 được chọn bằng validation CV.

**C6. Nếu một từ không có trong vocabulary của Test?**  
Vectorizer bỏ từ chưa có cột đã học; classifier nhận vector cùng số chiều với Train, không tự tạo đặc trưng mới.

**C7. Vì sao B3 TF-IDF + NB có thể chạy?**  
Scikit-learn MultinomialNB chấp nhận ma trận không âm; TF-IDF không âm. Nó cộng trọng số theo lớp thay cho số đếm nguyên.

**C8. SVM khác NB ở đâu?**  
NB học xác suất đặc trưng theo lớp; SVM học các trọng số của ranh giới phân loại bằng tối ưu regularization và squared hinge loss.

**C9. `LinearSVC` học ra cái gì?**  
`coef_` và `intercept_`, dùng để tính `decision_function(X) = X @ coef_.T + intercept_`, cùng thông tin số vòng tối ưu `n_iter_`.

**C10. SVM score 1.2 có phải xác suất Spam 120%?**  
Không. Là decision score có dấu; nó cho biết bình luận nằm về phía nào của ranh giới và cách xa về mặt điểm số, không phải xác suất.

**C11. `C=0.3` trong B4 từ đâu ra?**  
Nhóm đặt lưới C = 0.1, 0.3, 1.0, 3.0; GridSearchCV thử qua 5 fold cùng các n-gram. Cấu hình thắng có CV F1 Spam cao nhất trong các ứng viên được thử.

**C12. Tại sao `ngram=(1,2)` được chọn cùng C/alpha?**  
Vì tạo bigram làm thay đổi các cột đặc trưng và điều kiện học; phải tối ưu hai lựa chọn trong cùng Pipeline để so bằng CV, không chọn riêng theo một lần xem vocabulary.

**C13. `class_weight='balanced'` có làm tăng số mẫu Spam không?**  
Không. Nó thay đổi trọng số lỗi theo lớp trong tối ưu LinearSVC; chỉ S2 mới thực sự giảm số mẫu Ham trong Train.

**C14. `dual=False` nghĩa là gì?**  
Cấu hình bộ tối ưu dùng cách giải primal cho LinearSVC, không phải đổi từ nhị phân sang đa lớp. Random state thường không ảnh hưởng ở nhánh này.

**C15. `max_iter=20000` có nghĩa mô hình chắc chắn tốt?**  
Không. Nó chỉ đặt giới hạn vòng lặp; phải xem cảnh báo hội tụ và đánh giá CV/Test.

**C16. Tại sao `LinearSVC.predict` không có probability?**  
Vì LinearSVC mặc định trả điểm của hàm quyết định từ trọng số tuyến tính; nếu cần xác suất phải qua calibration bổ sung, dự án này không làm.

**C17. Vì sao S3 có ngưỡng -0.1776?**  
Nhóm tìm ngưỡng theo score ngoài fold trên Train để tối đa F1 Spam; ngưỡng âm giúp nhận thêm các bình luận có score hơi âm là Spam. Không chọn từ Test.

**C18. Test có giúp tìm alpha/C không?**  
Không. Alpha/C chọn bằng Cross-Validation trên Train; Test chỉ dùng đánh giá mô hình đã chốt.

**C19. `GridSearchCV.fit` gọi đến những lớp nào?**  
Nó clone Pipeline, `vec.fit_transform` biến văn bản thành ma trận, rồi `clf.fit` học NB/SVM trong từng fold, dự đoán validation tính điểm và fit lại cấu hình thắng trên toàn Train.

**C20. Làm sao truy vết mô hình đã học trong record?**  
Lấy `record['_model']`, sau đó `record['_model'].named_steps['vec']` và `.named_steps['clf']`. Từ clf xem `feature_log_prob_` nếu NB hoặc `coef_/intercept_` nếu SVM.

**C21. Vì sao `MultinomialNB.predict` đôi khi khác dự đoán của `evaluation.py`?**  
Ở trường hợp xác suất Spam đúng bằng 0.5: `predict` chọn score lớn nhất, còn `predict_record` định nghĩa Spam khi `score >= 0.5`. Đây là khác biệt xử lý điểm hòa.

**C22. Vì sao nhóm không tự viết Liblinear để train SVM?**  
Project tập trung vào quy trình phân loại và đánh giá; dùng scikit-learn để tận dụng bộ tối ưu tin cậy, còn nhiệm vụ của nhóm là hiểu cấu hình, đầu vào, thuật toán, cách chọn tham số và đánh giá công bằng.

**C23. B4 tốt nhất nghĩa là TF-IDF luôn tốt hơn BoW?**  
Không. Trong chính kết quả Notebook, B1 BoW+NB có CV F1 cao hơn B3 TF-IDF+NB; so sánh cần theo từng tổ hợp và dữ liệu cụ thể.

**C24. Sau khi fit xong, mô hình cuối lưu gì để dự đoán mới?**  
Lưu cả vectorizer và classifier đã fit trong Pipeline, cùng ngưỡng và quy tắc tiền xử lý. Dự đoán mới phải dùng đúng vocabulary/IDF và các hệ số đã học.

---

## 21. Bài tự luyện nhỏ để kiểm tra bạn đã hiểu thật chưa

**Bài 1 — Naive Bayes:** với vocabulary `['làm','lịch','nhận','xin']`, `class_count_=[2,2]`, `feature_count_` như mục 7, alpha=1. Hãy tự tính `P(Spam | 'xin lịch')`, giải thích vì sao bằng 0.1 và chỉ ra `class_log_prior_` đóng góp gì.

**Bài 2 — SVM:** với `w=[0.4444,-0.4444,0.4444,-0.4444]`, `b=0`, hãy tự tính score cho `['nhận','lịch']`, dự đoán với threshold=0 và với threshold=-0.2. So sánh khác nhau.

**Bài 3 — Notebook:** giải thích vì sao có 8 lượt `fit_experiment` trong cell 13 nhưng bảng B1–B4 có 4 dòng; nêu tổng số cấu hình thử khi tính cả hai lựa chọn stopwords và 5-fold.

**Bài 4 — Đọc code:** viết lại theo trí nhớ đường gọi `make_classifier` → `make_pipe` → `GridSearchCV.fit` → `vec.fit_transform` → `clf.fit` → dự đoán validation → chọn `best_estimator_`.

---

## 22. Thuật ngữ trọng tâm — cần nhớ ngay

| Thuật ngữ | Ý nghĩa một câu |
|---|---|
| Estimator | Đối tượng scikit-learn có thể `fit` dữ liệu để học tham số. |
| Classifier | Estimator dự đoán lớp Spam/Ham. |
| Feature vector | Hàng số mô tả một bình luận. |
| Sparse matrix | Ma trận chỉ lưu phần tử khác 0 để tiết kiệm bộ nhớ. |
| Prior | Xác suất một lớp trước khi nhìn câu mới. |
| Likelihood | Xác suất đặc trưng theo lớp trong mô hình NB. |
| Posterior | Xác suất lớp sau khi xét các đặc trưng. |
| Laplace smoothing | Cộng alpha để tránh xác suất 0 cho từ chưa gặp ở một lớp. |
| Log probability | Log xác suất dùng để tính ổn định. |
| Hyperplane | Ranh giới phân loại tuyến tính của SVM. |
| Margin | Mức đệm của ranh giới với các điểm train. |
| Regularization | Hạn chế độ phức tạp/trọng số để tổng quát tốt hơn. |
| Squared hinge loss | Hàm phạt vi phạm margin mà LinearSVC mặc định dùng. |
| Decision score | Giá trị có dấu từ SVM, không phải xác suất. |
| Hyperparameter | Cấu hình chọn trước khi fit, như alpha/C/n-gram. |
| Learned parameter | Tham số học từ dữ liệu, như `feature_log_prob_`/`coef_`. |
| Cross-Validation | Quy trình fit/validate lặp trên các fold Train để chọn tham số. |
| Pipeline | Ghép vectorizer và classifier, tránh xử lý lệch train/test. |
| Refit | Fit lại cấu hình thắng trên toàn Train sau CV. |

---

## 23. Kết luận và bước kế tiếp

> **`classifiers.py` chỉ chọn và cấu hình hai thuật toán chưa học: `MultinomialNB` với lưới alpha và `LinearSVC` với lưới C/class_weight. Trong `experiments.py`, Pipeline biến văn bản thành vector trước rồi mới gọi `clf.fit(X, y)`. MultinomialNB học bảng thống kê và log xác suất đặc trưng theo lớp; LinearSVC học các hệ số của ranh giới tuyến tính. Khi dự đoán, NB trả xác suất Spam theo mô hình, SVM trả decision score; `evaluation.py` dùng ngưỡng tương ứng để ra nhãn 0/1.**

**Bước kế tiếp:** `experiments.py` — đào sâu `Pipeline`, `make_pipe`, `fit_experiment`, `GridSearchCV`, `clone`, `set_params`, các lượt CV và `refit`; truy vết cách B1–B4 được fit, chấm validation và chọn `best_params_` theo F1 Spam.

---

## Phụ lục A. Tệp, hàm và mức độ nguồn

| Nguồn của nhóm | Phần được trích/diễn giải |
|---|---|
| `classifiers(1).py` | Cấu hình thật, `MODELS`, `NB_ALPHAS`, `SVM_CS`, `make_classifier`, `LinearSVC(dual=False, max_iter=20000, random_state=seed)` |
| `PHI_HIEU_4_LOP_SCIKIT_LEARN.md` | Các trích đoạn và đường gọi trong scikit-learn 1.6.1: `_count`, `_update_feature_log_prob`, `_joint_log_likelihood`, `_fit_liblinear`, `decision_function`, `predict`; chú ý phương thức kế thừa |
| `text_features(1).py` | `CountVectorizer`/`TfidfVectorizer`, `min_df=2`, hai lựa chọn n-gram, TF-IDF chuẩn hóa |
| `experiments(1).py` | `make_pipe`, `fit_experiment`, `GridSearchCV(refit='f1')`, CV F1 và record đã fit |
| `imbalance(1).py` | S1 weighted, S2 undersampling, S3 threshold |
| `evaluation(1).py` | `predict_record`: NB `predict_proba`/SVM `decision_function`, nhãn theo ngưỡng, Test metrics |
| Notebook 02 bản `(1)` | Cell 13 CV B1–B4 và stopwords; cell 16 S1–S3; cell 20 Test; cell 32 lưu và nạp lại mô hình |

**Tài liệu API liên quan trong dự án:**  
- [MultinomialNB scikit-learn 1.6](https://scikit-learn.org/1.6/modules/generated/sklearn.naive_bayes.MultinomialNB.html)  
- [LinearSVC scikit-learn 1.6](https://scikit-learn.org/1.6/modules/generated/sklearn.svm.LinearSVC.html)  
- [Pipeline](https://scikit-learn.org/1.6/modules/generated/sklearn.pipeline.Pipeline.html)  
- [GridSearchCV](https://scikit-learn.org/1.6/modules/generated/sklearn.model_selection.GridSearchCV.html)
