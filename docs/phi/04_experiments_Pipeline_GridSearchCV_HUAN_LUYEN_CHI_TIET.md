# BƯỚC 04 — `experiments.py`: GHÉP PIPELINE, HUẤN LUYỆN VÀ CHỌN SIÊU THAM SỐ BẰNG 5-FOLD CV

> **Phạm vi:** phần mô hình truyền thống do bạn phụ trách trong dự án phân loại bình luận sinh viên PTIT thành Ham (0) / Spam (1).  
> **Nguồn đối chiếu:** `experiments(1).py` (bản mới), `text_features(1).py`, `classifiers(1).py`, `training_utils(1).py`, `imbalance(1).py`, `evaluation(1).py`, `02_thuc_nghiem_va_demo(1).ipynb` (đặc biệt cell 4, 9–13, 15–16, 20, 32), `README(1).md`, `PHI_HIEU_4_LOP_SCIKIT_LEARN.md`.  
> **Giới hạn:** ví dụ tự tạo được ghi rõ **minh họa**, không phải các con số kiểm định thật. Kết quả thực tế lấy từ *output đã lưu trong Notebook*; không có dataset `labeled_v7.csv` đi kèm trong lượt này nên không huấn luyện lại dự án hoàn chỉnh.

---

## 0. MỤC TIÊU HỌC TẬP

Sau khi học xong, bạn phải trả lời được:

1. `experiments.py` nhận gì từ `training_utils.py`, `text_features.py`, `classifiers.py`?
2. `make_pipe()` khác `fit_experiment()` thế nào? Dòng nào thật sự fit?
3. `Pipeline([('vec', ...), ('clf', ...)])` hoạt động thế nào với `X` là chuỗi token chứ **chưa phải vector**?
4. `vec__ngram_range` và `clf__alpha` / `clf__C` nghĩa là gì? Tại sao một tổ hợp lại thử 6 hoặc 8 cấu hình?
5. `GridSearchCV` tạo bản sao cấu hình, chạy 5 fold, chấm Precision/Recall/F1 và chọn tham số thắng ra sao?
6. Tại sao từ điển / IDF phải học lại bên trong từng fold? `fit_transform` và `transform` đi theo nhánh nào?
7. `refit='f1'` có nghĩa gì? Vì sao sau CV còn cần fit lại trên toàn Train?
8. `cv_results_`, `best_index_`, `best_params_`, `best_estimator_` và `record` khác nhau thế nào?
9. Cell 13 gọi `fit_experiment` bao nhiêu lần, thử cả `(1,1)` và `(1,2)` chưa, chọn stopwords và B4 tại đâu?
10. Kết quả CV khác kết quả Test ở cell 20 thế nào, và file kế tiếp `imbalance.py` nhận dữ liệu gì?

**Tóm tắt một dòng:** `experiments.py` **tạo quy trình** (vector hóa + phân loại), **thử các siêu tham số bằng CV trên Train**, **chọn cấu hình theo F1 Spam**, **fit lại mô hình thắng trên toàn bộ Train**, rồi **trả `record` có Pipeline đã học** để các bước sau dùng.

---

## 1. VỊ TRÍ TRONG PROJECT — ĐỌC LUỒNG TRƯỚC KHI ĐỌC HÀM

```text
Notebook 02 — cell 6, 9: CSV có nhãn 0/1, chuỗi token đã được chuẩn bị
        │
        ├─ TEXT['giữ stop words']
        └─ TEXT['có loại stop words']
        │
Notebook 02 — cell 11: training_utils.split_train_test(...)
        ├─ train_idx, test_idx
        └─ CV_FOLDS = grouped_folds(keys[train_idx], y_train)
        │
Notebook 02 — cell 13: LẶP hai nhánh stopwords × hai feature × hai model
        │
        └─ experiments.fit_experiment(pre, feature, algorithm, X_train_text, y_train, CV_FOLDS)
               │
               ├─ make_pipe(feature, algorithm)
               │      ├─ text_features.make_vectorizer(feature)    → CountVectorizer / TfidfVectorizer
               │      ├─ classifiers.make_classifier(algorithm)   → MultinomialNB / LinearSVC
               │      └─ Pipeline([('vec', vectorizer), ('clf', classifier)])
               │
               ├─ GridSearchCV(..., scoring=P/R/F1, cv=CV_FOLDS, refit='f1')
               │      └─ search.fit(X_train_text, y_train)
               │             ├─ mỗi cấu hình × 5 fold: vec.fit_transform → clf.fit
               │             ├─ dự đoán validation, chấm Precision/Recall/F1
               │             ├─ lấy điểm F1 trung bình 5 fold, chọn cấu hình
               │             └─ REFIT cấu hình thắng trên toàn bộ Train
               │
               └─ record: thông tin cấu hình, điểm CV, tham số thắng,
                          _threshold, _model (Pipeline đã fit)
        │
Cell 13: chọn MAIN_PRE và B1–B4; focus = B4 trong output đã lưu
        │
Cell 16: imbalance.py thử S1/S2/S3 từ nhánh TF-IDF + SVM
        │
Cell 20: evaluation.py dự đoán TEST độc lập (không fit trên Test)
        │
Cell 32: lưu Pipeline + ngưỡng + cấu hình tiền xử lý vào .joblib
```

### Phân biệt dữ liệu tại mỗi chỗ

| Tên | Trong dự án | Kiểu / ý nghĩa |
|---|---|---|
| `TEXT[pre]` | Cell 9 | `pandas.Series` chứa các **chuỗi token** tương ứng với mọi dòng CSV |
| `train_idx` | Cell 11 | Chỉ số **trong toàn CSV** để lấy Train |
| `test_idx` | Cell 11 | Chỉ số **trong toàn CSV** để lấy Test |
| `X_train_text` | Cell 13 | `numpy.ndarray` các chuỗi token *chỉ của Train* |
| `y_train` | Cell 11 | Mảng 0/1 tương ứng với `X_train_text` |
| `CV_FOLDS` | Cell 11 | Danh sách 5 tuple `(train_fold_idx, valid_fold_idx)`; chỉ số **tính trong Train** |
| `X_vectorized` | Bên trong Pipeline | Ma trận thưa BoW/TF-IDF; **không** phải đối số trực tiếp truyền cho `fit_experiment` |
| `_model` | Kết quả `fit_experiment` | Pipeline **đã fit lại trên toàn Train**, dùng để chấm Test và dự đoán bình luận mới |
| `y_test` | Cell 11 | Nhãn thật dành riêng để chấm Test, **không** truyền vào `fit_experiment` |

**Ví dụ cách lấy văn bản:**

```python
X_train_text = TEXT[pre].iloc[train_idx].to_numpy()
# X_train_text[0] có thể là một chuỗi như 'xin lịch_thi học_kỳ'
# CHƯA PHẢI numpy vector số / ma trận TF-IDF.

# Chỉ số train/valid của CV dùng TRÊN X_train_text, KHÔNG dùng trực tiếp trên raw toàn CSV.
for tr, va in CV_FOLDS:
    fold_text = X_train_text[tr]
    fold_y = y_train[tr]
    validation_text = X_train_text[va]
    validation_y = y_train[va]
```

---

## 2. LÝ THUYẾT PHẢI HIỂU TRƯỚC: HUẤN LUYỆN, SIÊU THAM SỐ VÀ KIỂM TRA

### 2.1. Học cái gì? Khác giữa vectorizer và classifier?

- **Vectorizer:** nhận chuỗi token, học `vocabulary_`; TF-IDF học thêm `idf_`. Khi `transform()`, nó tạo ma trận số.
- **Classifier:** nhận **ma trận số + `y_train`**, học cách phân loại: Naive Bayes học bảng thống kê/log xác suất; LinearSVC học `coef_`/`intercept_`.
- **Pipeline:** một lớp điều phối: khi `fit(text, y)` được gọi, trước tiên `vec.fit_transform(text)` rồi `clf.fit(ma_trận, y)`. Khi `predict(text)` được gọi, `vec.transform(text)` trước `clf.predict(...)`.
- **GridSearchCV:** không phải mô hình phân loại thứ ba. Nó thử các cách cấu hình Pipeline, huấn luyện và kiểm tra từng cách, chọn cách có điểm validation cao nhất.

### 2.2. Tham số học được khác siêu tham số được thử

| Loại | Ví dụ | Ai tạo / khi nào? |
|---|---|---|
| **Siêu tham số** | `ngram_range=(1,2)`, `alpha=0.3`, `C=1.0` | Người thiết kế đặt các giá trị ứng viên; CV chọn một bộ |
| **Tham số học được** | `vocabulary_`, `idf_`, `feature_log_prob_`, `coef_`, `intercept_` | Chỉ xuất hiện sau khi mô hình `.fit()` trên dữ liệu |
| **Chỉ số đánh giá** | `CV F1`, `Test F1` | Tính từ nhãn thật và dự đoán (đúng tập đánh giá) |

`C=0.3` không phải trọng số của từ “inbox” do SVM tự học. `C` chỉ quyết định cách **phạt lỗi trong tối ưu**. Trọng số của từng cột TF-IDF là `coef_`, được SVM học khi `fit()`.

### 2.3. Vì sao không chọn tham số bằng Test?

Nếu cứ nhìn Test rồi đổi `C`, `alpha`, n-gram đến khi Test F1 đẹp, tập Test không còn là phép đánh giá độc lập. Vì thế:

1. Chia riêng Train/Test (cell 11).
2. Chỉ dùng Train + các fold validation để tìm tham số (cell 13).
3. Sau khi chọn, fit lại mô hình trên toàn bộ Train.
4. Giữ Test đến cell 20 để đánh giá.

### 2.4. Cross-Validation (CV) là gì?

Với 5 fold, mỗi lượt chọn một fold làm validation, bốn fold còn lại làm train. Ví dụ:

```text
Lượt 1: [VALID][TRAIN][TRAIN][TRAIN][TRAIN]
Lượt 2: [TRAIN][VALID][TRAIN][TRAIN][TRAIN]
Lượt 3: [TRAIN][TRAIN][VALID][TRAIN][TRAIN]
Lượt 4: [TRAIN][TRAIN][TRAIN][VALID][TRAIN]
Lượt 5: [TRAIN][TRAIN][TRAIN][TRAIN][VALID]
```

Mỗi lượt:

- **Học** vocabulary/IDF và classifier trên bốn fold Train.
- **Dùng lại** từ điển/IDF và classifier đó để dự đoán một fold Validation.
- **So** nhãn dự đoán với nhãn thật của Validation và tính Precision/Recall/F1 của **Spam = 1**.

Với 5 lần có F1 lần lượt `0.70, 0.76, 0.72, 0.80, 0.77`, điểm CV F1 trung bình bằng `(0.70 + 0.76 + 0.72 + 0.80 + 0.77)/5 = 0.75` (ví dụ giả định). Nó là **trung bình 5 điểm fold**, không phải bắt buộc bằng F1 tính từ gộp mọi dự đoán của 5 fold; vì kích thước fold và số mẫu Spam có thể khác nhau.

### 2.5. CV của nhóm có một điểm riêng: `grouped_folds`

`CV_FOLDS` trong bài **không** do `GridSearchCV` tự chia ngẫu nhiên. `training_utils.grouped_folds(keys[train_idx], y_train)` đã tạo sẵn 5 cặp chỉ số. Các bình luận có **cùng khóa chuẩn hóa** được đưa cùng fold, hạn chế một nội dung gần trùng xuất hiện ở cả phần train và validation của một lượt. Nhóm còn hướng tới dùng cùng cách chia dữ liệu với viBERT.

**Lưu ý:** fold có thể khác kích thước và tỷ lệ nhãn; hàm kiểm tra mỗi phía có cả Ham/Spam, không bảo đảm phân tầng chính xác.

---

## 3. ĐỌC FILE `experiments(1).py` THEO TỪNG PHẦN

### 3.1. Phần import và hằng số scoring

```python
import time

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from pipeline.modeling.classifiers import MODELS, make_classifier
from pipeline.modeling.evaluation import evaluate, boot_ci
from pipeline.modeling.text_features import FEATURES, make_vectorizer

CV_SCORING = {"precision": "precision", "recall": "recall", "f1": "f1"}
```

- `time`: đo thời gian chạy tìm tham số **cộng cả refit**.
- `GridSearchCV`: thử lưới tham số theo các fold và chọn cấu hình tối ưu.
- `Pipeline`: ghép bước tạo vector `vec` và phân loại `clf`.
- `MODELS`, `FEATURES`: kiểm tên được phép (`"Naive Bayes"`, `"SVM"`; `"BoW"`, `"TF-IDF"`).
- `make_classifier`: tạo NB/SVM và lưới alpha/C.
- `make_vectorizer`: tạo CountVectorizer/TfidfVectorizer và lưới n-gram.
- `evaluate`, `boot_ci`: **được import nhưng không được gọi trong `fit_experiment()` của file này**. Phép đánh giá Test thuộc file `evaluation.py` và cell 20.
- `CV_SCORING`: ba metric dùng trong từng lượt validation; vì `y∈{0,1}`, với nhãn dương mặc định `pos_label=1`, điểm P/R/F1 hướng đến **Spam**.

`scoring` là một dictionary vì code cần theo dõi **ba metric**, nhưng `refit='f1'` chọn cấu hình theo **F1**. F1 trung bình là tiêu chí chọn; Precision/Recall là thông tin bổ sung, không phải ba tiêu chí bỏ phiếu ngang nhau.

### 3.2. Hàm `make_pipe(...)`: nhận gì và trả gì?

**Mã nguồn đang dùng:**

```python
def make_pipe(feat, model_name, *, seed=42, cache=None, n_jobs=2, class_weight=None):
    if feat not in FEATURES or model_name not in MODELS:
        raise ValueError("Đặc trưng/thuật toán không nằm trong cấu hình thực nghiệm")

    vectorizer, vector_grid = make_vectorizer(feat)
    classifier, model_grid = make_classifier(
        model_name, seed=seed, class_weight=class_weight
    )

    pipe = Pipeline([("vec", vectorizer), ("clf", classifier)], memory=cache)
    return pipe, {**vector_grid, **model_grid}, n_jobs
```

#### Giải thích chữ ký và tham số

| Đối số | Ý nghĩa | Giá trị của Notebook |
|---|---|---|
| `feat` | Phương pháp đặc trưng cần tạo | `"BoW"` hoặc `"TF-IDF"` |
| `model_name` | Thuật toán phân loại | `"Naive Bayes"` hoặc `"SVM"` |
| `*` | Các tham số sau đó phải truyền bằng tên nếu truyền tường minh | Cú pháp Python, không liên quan dữ liệu |
| `seed=42` | Giá trị seed được truyền cho bộ phân loại nếu phù hợp | `SEED=42` |
| `cache=None` | Nơi cache phép biến đổi trung gian của Pipeline (joblib.Memory) | Biến `cache` ở cell 13 |
| `n_jobs=2` | Số tác vụ song song dành cho `GridSearchCV`, **không** phải số fold hay số mô hình | `N_JOBS` mặc định 2 |
| `class_weight=None` | Trọng số lớp; mặc định SVM không cân bằng trọng số | B2/B4: `None`; S1: `"balanced"` |

#### Giải thích từng lệnh

1. `if feat not in FEATURES or model_name not in MODELS`: chống truyền tên không được hỗ trợ; ví dụ `"Random Forest"` sẽ báo lỗi thay vì tự đoán.
2. `vectorizer, vector_grid = make_vectorizer(feat)`: tạo bước `vec` chưa fit. `vector_grid` có `{'vec__ngram_range': [(1,1), (1,2)]}`.
3. `classifier, model_grid = make_classifier(...)`: tạo bước `clf` chưa fit; có grid `clf__alpha` hoặc `clf__C`/`clf__class_weight`.
4. `Pipeline([('vec', vectorizer), ('clf', classifier)], memory=cache)`: nối **thứ tự xử lý**: văn bản → ma trận → bộ phân loại.
5. `{**vector_grid, **model_grid}`: nối các cặp khóa–giá trị của hai dictionary; không tính toán hay chọn tham số.
6. `return pipe, grid, n_jobs`: trả **ba giá trị**, không có nhãn dự đoán, không có điểm CV, không có model đã fit.

#### Ví dụ 1 — TF-IDF + SVM

```python
pipe, grid, jobs = make_pipe("TF-IDF", "SVM", seed=42)
print(pipe)
print(grid)
```

Về ý tưởng kết quả:

```python
Pipeline([
    ("vec", TfidfVectorizer(..., min_df=2, ngram_range=(1,1),
                            sublinear_tf=True, smooth_idf=True, norm="l2")),
    ("clf", LinearSVC(dual=False, max_iter=20000, random_state=42))
])

{
    "vec__ngram_range": [(1, 1), (1, 2)],
    "clf__C": [0.1, 0.3, 1.0, 3.0],
    "clf__class_weight": [None]
}
# jobs = 2
```

Lưu ý: `CountVectorizer`/`TfidfVectorizer` lúc mới tạo chưa có `vocabulary_`, và SVM chưa có `coef_`.

#### Ví dụ 2 — BoW + Naive Bayes

```python
pipe, grid, jobs = make_pipe("BoW", "Naive Bayes")
```

Kết quả lưới:

```python
{
    "vec__ngram_range": [(1, 1), (1, 2)],
    "clf__alpha": [0.1, 0.3, 1.0]
}
```

Hai lưới khác nhau vì Naive Bayes cần thử `alpha`, còn SVM cần thử `C`. Cả hai đều thử `ngram_range`.

### 3.3. Cú pháp `vec__...`, `clf__...` nằm ở đâu?

Tên `"vec"` và `"clf"` được đặt tại:

```python
Pipeline([("vec", vectorizer), ("clf", classifier)])
```

Vì vậy:

- `vec__ngram_range=(1,2)` tương đương đổi `ngram_range` ở *vectorizer*.
- `clf__C=0.3` tương đương đổi `C` ở *LinearSVC*.
- `clf__alpha=1.0` tương đương đổi `alpha` ở *MultinomialNB*.
- `clf__class_weight=None` là không dùng trọng số lớp SVM; khi S1 thì thành `'balanced'`.

Trong scikit-learn, `Pipeline.set_params(vec__ngram_range=(1,2), clf__C=0.3)` chuyển tham số cho đúng bước. `GridSearchCV` tự thực hiện hành động tương đương cho mỗi cấu hình mà nó thử.

### 3.4. `ngram_range` `(1,1)` xuất hiện ở `text_features.py`, `(1,2)` được thử từ đâu?

**Trong `text_features(1).py`:**

```python
NGRAMS = [(1, 1), (1, 2)]

common = {
    "token_pattern": TOKEN_PATTERN,
    "min_df": 2,
    "ngram_range": (1, 1),  # CHỈ LÀ GIÁ TRỊ KHỞI TẠO
    "lowercase": True,
    "dtype": np.float64,
}

# ...
return vectorizer, {"vec__ngram_range": NGRAMS}
```

**Trong `experiments.py`:** `make_pipe()` gộp `vector_grid` vào `grid`, rồi `GridSearchCV(pipe, grid, ...)` sử dụng nó.

Như vậy, **cả `(1,1)` và `(1,2)` được thử** cho mọi B1–B4, chứ không phải chỉ dùng `(1,1)`. `(1,1)` = unigram; `(1,2)` = unigram **cộng** bigram. Khi lựa chọn `(1,2)`, mô hình đã được kiểm tra rằng cấu hình đó cho F1 validation tốt nhất (hoặc là một lựa chọn thắng theo quy tắc xử lý hòa điểm).

---

## 4. SỐ CẤU HÌNH ĐƯỢC THỬ — TÍNH CỤ THỂ

Khi truyền `grid` vào GridSearchCV, nó xét **tích Descartes** của các danh sách tham số (tất cả cách kết hợp các giá trị), không chỉ cặp phần tử cùng vị trí.

### 4.1. Với Naive Bayes: 2 n-gram × 3 alpha = 6 cấu hình

| ID | `vec__ngram_range` | `clf__alpha` |
|---:|---|---:|
| 1 | `(1,1)` | 0.1 |
| 2 | `(1,2)` | 0.1 |
| 3 | `(1,1)` | 0.3 |
| 4 | `(1,2)` | 0.3 |
| 5 | `(1,1)` | 1.0 |
| 6 | `(1,2)` | 1.0 |

BoW+NB và TF-IDF+NB *mỗi tổ hợp* đều thử 6 cấu hình đó, nhưng số cột/giá trị ma trận và điểm CV có thể khác nhau.

### 4.2. Với SVM: 2 n-gram × 4 C × 1 class_weight = 8 cấu hình

| ID | `vec__ngram_range` | `clf__C` | `clf__class_weight` |
|---:|---|---:|---|
| 1 | `(1,1)` | 0.1 | None |
| 2 | `(1,2)` | 0.1 | None |
| 3 | `(1,1)` | 0.3 | None |
| 4 | `(1,2)` | 0.3 | None |
| 5 | `(1,1)` | 1.0 | None |
| 6 | `(1,2)` | 1.0 | None |
| 7 | `(1,1)` | 3.0 | None |
| 8 | `(1,2)` | 3.0 | None |

BoW+SVM và TF-IDF+SVM đều thử 8 cấu hình trên.

### 4.3. Vì sao số lượt fit lớn hơn 4?

- **Một nhánh stopwords:** B1 (6) + B2 (8) + B3 (6) + B4 (8) = **28 cấu hình**.
- **Mỗi cấu hình qua 5 fold:** 28 × 5 = **140 lượt fit fold**.
- **Notebook thử hai nhánh stopwords:** 140 × 2 = **280 lượt fit fold**.
- `refit='f1'`: sau mỗi trong **8 cuộc GridSearchCV**, nó fit lại cấu hình thắng trên toàn bộ Train (**8 lượt refit** ngoài 280 lượt fit fold).

Con số 280 không bao gồm thí nghiệm S1/S2/S3 ở cell 16. Với `cache`, một số phần biến đổi vector có thể được tái sử dụng, nhưng đó không có nghĩa GridSearchCV bỏ qua các tổ hợp tham số hay fold.

---

## 5. HÀM `fit_experiment(...)` — ĐỌC TỪNG ĐỐI SỐ VÀ TỪNG DÒNG

**Code nguồn (giữ đúng phần xử lý):**

```python
def fit_experiment(pre, feat, model_name, X, y, folds, *, seed=42,
                   cache=None, n_jobs=2, class_weight=None,
                   strategy="Không điều chỉnh"):
    pipe, grid, jobs = make_pipe(
        feat, model_name, seed=seed, cache=cache,
        n_jobs=n_jobs, class_weight=class_weight
    )
    start = time.time()
    search = GridSearchCV(
        pipe, grid, scoring=CV_SCORING, cv=folds,
        n_jobs=jobs, refit="f1", error_score="raise"
    )
    search.fit(X, y)
    best_index = search.best_index_
    cv = search.cv_results_
    cv_precision = float(cv["mean_test_precision"][best_index])
    cv_recall = float(cv["mean_test_recall"][best_index])
    cv_f1 = float(cv["mean_test_f1"][best_index])
    return {
        "Tiền xử lý": pre,
        "Đặc trưng": feat,
        "Thuật toán": model_name,
        "Chiến lược": strategy,
        "CV Precision (spam)": cv_precision,
        "CV Recall (spam)": cv_recall,
        "CV F1 (spam)": cv_f1,
        "Tham số": search.best_params_,
        "Thời gian (s)": time.time() - start,
        "_threshold": 0.5 if model_name == "Naive Bayes" else 0.0,
        "_model": search.best_estimator_
    }
```

### 5.1. Ý nghĩa của 11 đối số

| Tên | Cụ thể |
|---|---|
| `pre` | **Tên nhánh tiền xử lý** như `"giữ stop words"` để ghi vào record; hàm này KHÔNG tự xóa stopwords. |
| `feat` | Chọn `BoW` hoặc `TF-IDF`; hàm chỉ gọi vectorizer tương ứng. |
| `model_name` | Chọn `Naive Bayes` hoặc `SVM`. |
| `X` | **Mảng chuỗi token Train**; có thứ tự khớp với `y`. |
| `y` | Nhãn thật của Train (`0=ham`, `1=spam`), dùng để `classifier.fit` và chấm validation. |
| `folds` | Năm cặp chỉ số `(train, validation)` **trong Train** từ `grouped_folds`. |
| `seed` | Seed chuyển xuống bộ phân loại, theo cấu hình dự án. |
| `cache` | Đối tượng `joblib.Memory`, được dùng làm cache cho bước trung gian của Pipeline. |
| `n_jobs` | Mức chạy song song của GridSearchCV; không đổi logic train/validation. |
| `class_weight` | `None` ở B1–B4; S1 truyền `"balanced"` cho SVM. |
| `strategy` | Chuỗi nhãn thí nghiệm, ví dụ `"Không điều chỉnh"`; không tự thực hiện chiến lược cân bằng. |

### 5.2. `pipe, grid, jobs = make_pipe(...)`

Tạo bộ khung công việc (chưa học) và lưới cấu hình. Nếu là TF-IDF + SVM: `pipe` chứa `TfidfVectorizer` chưa có `vocabulary_` và `LinearSVC` chưa có `coef_`.

### 5.3. `start = time.time()`

Đánh dấu thời gian bắt đầu thí nghiệm. `"Thời gian (s)"` của record = thời gian **toàn bộ tìm kiếm CV + refit** tính từ đó, không phải chỉ thời gian fit một mô hình trên Train. Thời gian còn phụ thuộc cache, CPU, `n_jobs`, phiên bản thư viện.

### 5.4. `GridSearchCV(...)` làm gì ở thời điểm khởi tạo?

Ở **dòng tạo** `GridSearchCV`, thư viện ghi nhớ `pipe`, `grid`, scorer, folds, jobs, quy tắc refit; **chưa chạy huấn luyện**. Việc học chỉ bắt đầu ở `search.fit(X, y)`.

| Tham số | Giá trị dự án | Nghĩa |
|---|---|---|
| `estimator` | `pipe` | Một Pipeline (vectorizer + classifier) để nhân bản và thử. |
| `param_grid` | `grid` | Các danh sách tham số cần kết hợp. |
| `scoring` | `CV_SCORING` | Tính ba metric Precision/Recall/F1 trên validation. |
| `cv` | `folds` | Dùng đúng các cặp chỉ số train/validation nhóm đã định nghĩa, **không tự chia mới**. |
| `n_jobs` | `jobs` | Số tác vụ chạy song song. |
| `refit` | `'f1'` | Dùng **mean validation F1** để chọn và fit lại estimator thắng trên toàn Train. |
| `error_score` | `'raise'` | Có lỗi khi fit/scoring thì ném lỗi, không thay bằng điểm lỗi rồi tiếp tục im lặng. |

**Vì sao chọn F1 Spam?** Dataset lệch lớp: Ham nhiều hơn Spam. Accuracy cao có thể chỉ do đoán Ham; F1 quan tâm cùng lúc Precision và Recall đối với lớp Spam. Đây là lựa chọn thiết kế của dự án, không có nghĩa F1 luôn là mục tiêu tốt nhất cho mọi hệ thống thực tế.

### 5.5. `search.fit(X, y)` — CHỖ HUẤN LUYỆN THỰC SỰ

Đây là lời gọi quan trọng nhất. Diễn giải logic bên trong (pseudocode, không phải mã nguồn nguyên văn của thư viện):

```python
for params in every_combination(grid):
    fold_metrics = []
    for train_positions, valid_positions in folds:
        candidate = clone(pipe)             # model MỚI chưa fit
        candidate.set_params(**params)      # chọn ngram, alpha/C
        candidate.fit(X[train_positions], y[train_positions])
        # Pipeline.fit tự gọi vec.fit_transform và clf.fit
        pred = candidate.predict(X[valid_positions])
        # Pipeline.predict tự gọi vec.transform rồi clf.predict
        fold_metrics.append({
            'precision': precision_score(y[valid_positions], pred),
            'recall': recall_score(y[valid_positions], pred),
            'f1': f1_score(y[valid_positions], pred),
        })
    calculate_mean_validation_scores(fold_metrics)

pick_params_with_highest_mean_f1()
fit_winner_again_on_all_train()  # vì refit='f1'
```

**Cần nhớ:** mỗi tổ hợp × mỗi fold là một mô hình tạm thời **riêng**. Mô hình ở fold thứ hai không lấy vocabulary/IDF/trọng số từ fold thứ nhất. Điều này tránh leakage trong CV.

### 5.6. Trong một fold, Pipeline gọi những phương thức nào?

Giả sử một lượt của B4 đang thử `ngram_range=(1,2), C=0.3`.

```text
A. TRAIN FOLD (ví dụ 24 nghìn bình luận trong Train)
    X_tr_fold: chuỗi token
    y_tr_fold: nhãn 0/1
       │
       ├─ TfidfVectorizer.fit_transform(X_tr_fold)
       │      ├─ tokenizer + tạo unigram/bigram
       │      ├─ học vocabulary min_df=2 trên fold TRAIN này
       │      ├─ tính IDF trên fold TRAIN này
       │      └─ trả CSR (số câu fold_train × số đặc trưng)
       │
       └─ LinearSVC(C=0.3).fit(CSR, y_tr_fold)
              ├─ tính toán tối ưu
              └─ lưu coef_, intercept_

B. VALIDATION FOLD (không nằm trong dữ liệu vừa fit)
    X_valid: chuỗi token
       │
       ├─ TfidfVectorizer.transform(X_valid)
       │      └─ chỉ dùng vocabulary_/idf_ đã học ở A, KHÔNG FIT LẠI
       ├─ LinearSVC.predict(CSR_valid)
       └─ so y_pred với y_valid → Precision, Recall, F1 Spam
```

Số cột của CSR validation **bằng đúng** số cột của CSR train fold, vì dùng chung vocabulary. Nếu validation có từ mới không thuộc vocabulary fold train, vectorizer không tạo cột mới.

**Cảnh báo leakage:** Sai nếu gọi `TfidfVectorizer.fit_transform()` trên cả Train+Validation trước khi chia fold. Vì như vậy vocabulary và IDF đã “nhìn” validation. Dự án dùng Pipeline bên trong GridSearchCV để tránh lỗi này.

### 5.7. Vì sao phải `refit` sau 5 fold?

Trong mỗi lượt CV, bộ phân loại chỉ học trên **4/5 Train**. Sau khi đã chọn cấu hình thắng, muốn dùng toàn bộ dữ liệu đã được phép học để xây dựng mô hình cuối, thư viện fit lại:

```text
Best params theo CV: {'vec__ngram_range': (1,2), 'clf__C': 0.3, ...}
                             ↓
TẠO LẠI estimator theo best params
                             ↓
Pipeline.fit(X_train_text TOÀN BỘ 30.759, y_train TOÀN BỘ 30.759)
                             ↓
vectorizer có vocabulary/IDF của TOÀN Train
classifier có coef_/intercept_ của TOÀN Train
                             ↓
search.best_estimator_  (mô hình để chấm Test)
```

**CV không tạo một mô hình từ phép “trung bình 5 bộ trọng số SVM”.** Nó trung bình **điểm đánh giá**, chọn **siêu tham số**, rồi **fit lại một mô hình mới** với siêu tham số thắng trên toàn bộ Train.

### 5.8. Các thuộc tính kết quả của GridSearchCV

| Thuộc tính | Kiểu / tác dụng | Có phải Test độc lập? |
|---|---|---|
| `search.cv_results_` | Dictionary chứa kết quả mọi cấu hình × fold (và trung bình/độ lệch) | **Không**, validation của CV |
| `search.best_index_` | Chỉ số dòng cấu hình được chọn trong `cv_results_` | Không |
| `search.best_params_` | Dictionary bộ siêu tham số thắng | Không |
| `search.best_score_` | Giá trị mean CV F1 của cấu hình thắng (`refit='f1'`) | Không |
| `search.best_estimator_` | Pipeline đã **refit trên toàn Train** | Là mô hình dùng để chấm Test sau này |
| `search.cv_results_['mean_test_f1']` | F1 trung bình của từng cấu hình qua 5 fold validation | **Không** (tên nội bộ có chữ `test`) |
| `search.cv_results_['split0_test_f1']` | F1 fold validation đầu tiên | Không |

**Bẫy quan trọng:** `mean_test_f1` trong GridSearchCV có chữ `test` theo cách đặt tên của scikit-learn nhưng ở **dự án này** chính là **validation của CV trong Train**. Không được lấy nó gọi là F1 của 7.474 mẫu Test độc lập.

### 5.9. Đọc ba dòng lấy điểm cấu hình thắng

```python
best_index = search.best_index_
cv = search.cv_results_
cv_precision = float(cv["mean_test_precision"][best_index])
cv_recall = float(cv["mean_test_recall"][best_index])
cv_f1 = float(cv["mean_test_f1"][best_index])
```

Giả sử `best_index=5`. Khi đó lấy hàng thứ 6 của cả ba mảng trung bình để lấy Precision/Recall/F1 **của cùng cấu hình đã thắng theo F1**. Không phải lấy `max(Precision)` và `max(Recall)` độc lập từ hai cấu hình khác nhau!

- `float(...)`: chuyển giá trị numpy scalar thành Python float thuận tiện cho bảng/report.
- `cv_precision`, `cv_recall`, `cv_f1`: ba điểm của **cùng** bộ tham số thắng.
- `best_index` là vị trí cấu hình trong danh sách kết quả GridSearchCV, **không** phải ID bình luận hay vị trí fold.

### 5.10. Đọc `record` trả về: các trường theo mục đích

```python
{
    'Tiền xử lý': pre,
    'Đặc trưng': feat,
    'Thuật toán': model_name,
    'Chiến lược': strategy,
    'CV Precision (spam)': cv_precision,
    'CV Recall (spam)': cv_recall,
    'CV F1 (spam)': cv_f1,
    'Tham số': search.best_params_,
    'Thời gian (s)': time.time() - start,
    '_threshold': 0.5 if model_name == 'Naive Bayes' else 0.0,
    '_model': search.best_estimator_,
}
```

| Trường | Để làm gì? |
|---|---|
| `Tiền xử lý` | Cho Notebook biết phải lấy cột chuỗi token nào tương ứng khi chấm Test; đây là **nhãn metadata**. |
| `Đặc trưng`, `Thuật toán`, `Chiến lược` | Ghi bảng, xác định mô hình thuộc B1–B4 hay S1. |
| `CV Precision/Recall/F1` | So sánh ở cell 13/16; **chưa phải Test**. |
| `Tham số` | Tham số thắng, ví dụ `C=0.3`, `ngram_range=(1,2)`. |
| `Thời gian (s)` | Đo thời gian tìm kiếm CV + refit. |
| `_threshold` | Với NB = 0.5 cho xác suất Spam; SVM = 0 cho decision score (S3 thay ngưỡng ở `imbalance.py`). |
| `_model` | **Pipeline đã fit** dùng ở `evaluation.predict_record()` và lưu `.joblib`. |

Dấu `_` ở hai trường cuối là quy ước đặt tên trường nội bộ trong dictionary, không phải cú pháp ẩn đặc biệt của Python.

Ví dụ, sau khi gọi:

```python
record = fit_experiment("giữ stop words", "TF-IDF", "SVM", X_train_text, y_train, CV_FOLDS)

print(record['Tham số'])
print(record['CV F1 (spam)'])
model = record['_model']  # pipeline đã fit, không cần fit lần nữa để chấm Test
```

Chú ý: code `fit_experiment` **không trả về biến `search`**. Nó chỉ trả `record` với `best_estimator_`, `best_params_` và vài điểm **của cấu hình thắng**. Muốn xem F1 của **từng cấu hình thua/thắng**, cần có `search.cv_results_` từ một phiên tìm kiếm; xem mục 9.

---

## 6. MINH HỌA CỤ THỂ: 1 FOLD HOẠT ĐỘNG TỪ ĐẦU TỚI CUỐI

### 6.1. Dataset nhỏ (tự tạo để học, KHÔNG phải dataset của dự án)

Giả sử 20 bình luận đã là chuỗi token; 10 Spam (1), 10 Ham (0):

```python
import numpy as np

X = np.array([
    # Spam: chỉ số 0–9
    'nhận làm bài tập giá rẻ', 'nhận làm đồ án giá rẻ',
    'inbox mua tài khoản giá rẻ', 'tuyển cộng tác viên kiếm tiền',
    'tặng học bổng inbox ngay', 'nhận làm đồ án inbox',
    'bán tài liệu giá rẻ inbox', 'tuyển cộng tác viên online',
    'kiếm tiền online ngay', 'mua tài khoản inbox ngay',
    # Ham: chỉ số 10–19
    'xin lịch thi môn java', 'cho mình hỏi lịch học',
    'xin tài liệu ôn thi', 'mọi người cho hỏi lịch thi',
    'chia sẻ kinh nghiệm học java', 'mình cần tài liệu môn java',
    'lịch học kỳ này có thay đổi', 'xin đề thi cuối kỳ',
    'có ai biết lịch học không', 'mọi người chia sẻ tài liệu thi'
], dtype=object)
y = np.array([1]*10 + [0]*10)
```

Đây chỉ là dữ liệu học cách gọi hàm. Dự án thật lấy 30.759 Train từ `training_utils.py`, sử dụng chính `CV_FOLDS` chia theo khóa văn bản. Ví dụ này chủ động đặt vị trí fold để tính toán dễ kiểm tra, **không thay thế** grouped split của dự án.

### 6.2. Tạo 5 fold minh họa

```python
folds = []
for i in range(5):
    valid = np.array([i, i+5, i+10, i+15])
    train = np.setdiff1d(np.arange(20), valid)
    folds.append((train, valid))
```

Trong fold 1 (`i=0`):

- Train có 16 bình luận.
- Validation gồm chỉ số `[0, 5, 10, 15]`, tức hai Spam + hai Ham.
- Nhãn validation là `[1, 1, 0, 0]`.

### 6.3. Huấn luyện B1 — BoW + NB với `alpha=0.1` và `(1,1)` trên fold 1

```python
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import precision_score, recall_score, f1_score

tr, va = folds[0]
model = Pipeline([
    ('vec', CountVectorizer(
        token_pattern=r'(?u)\b\w+\b|[^\w\s]',
        min_df=2, lowercase=True, dtype=np.float64,
        ngram_range=(1,1)
    )),
    ('clf', MultinomialNB(alpha=0.1))
])

model.fit(X[tr], y[tr])
y_pred = model.predict(X[va])
print('Validation index:', va)
print('True:', y[va])
print('Pred:', y_pred)
print('P:', precision_score(y[va], y_pred))
print('R:', recall_score(y[va], y_pred))
print('F1:', f1_score(y[va], y_pred))
```

**Kết quả đã chạy kiểm tra bằng scikit-learn:**

```text
Validation index: [ 0  5 10 15]
True: [1 1 0 0]
Pred: [1 1 0 0]
P: 1.0
R: 1.0
F1: 1.0
```

**Dữ liệu thực sự đi qua các bước:**

1. `Pipeline.fit(X[tr], y[tr])` nhận **16 chuỗi văn bản** và **16 nhãn**.
2. `CountVectorizer.fit_transform(X[tr])` phân tích token, học vocabulary từ **16 câu này** (với min_df=2) và trả ma trận **16 × số đặc trưng**.
3. `MultinomialNB.fit(ma_trận_16_câu, y[tr])` cộng đặc trưng theo lớp, làm mịn và lưu log xác suất.
4. `model.predict(X[va])`: vectorizer **transform bốn câu validation** theo vocabulary của 16 câu Train, rồi NB tính lớp dự đoán.
5. So dự đoán `[1,1,0,0]` với nhãn thật `[1,1,0,0]` nên lượt này TP=2, TN=2, FP=0, FN=0; Precision=Recall=F1=1.

Dù fold 1 đạt 1.0, **không kết luận mô hình đạt F1=1 trên toàn Train hoặc Test**. Còn bốn fold khác.

### 6.4. Tự chạy GridSearchCV cho B1 trên cả 5 fold

```python
from sklearn.model_selection import GridSearchCV

pipe = Pipeline([
    ('vec', CountVectorizer(token_pattern=r'(?u)\b\w+\b|[^\w\s]',
                            min_df=2, dtype=np.float64)),
    ('clf', MultinomialNB())
])
grid = {
    'vec__ngram_range': [(1,1), (1,2)],
    'clf__alpha': [0.1, 0.3, 1.0]
}
scoring = {'precision':'precision', 'recall':'recall', 'f1':'f1'}
search = GridSearchCV(pipe, grid, scoring=scoring, cv=folds,
                      refit='f1', n_jobs=1, error_score='raise')
search.fit(X, y)

print('Best params:', search.best_params_)
print('Best CV F1:', search.best_score_)
```

Với 20 câu minh họa này, kết quả đã kiểm tra:

```text
Best params: {'clf__alpha': 0.1, 'vec__ngram_range': (1, 1)}
Best CV F1: 0.8
```

Chứng minh: **không phải lúc nào `(1,2)` cũng thắng**. Ví dụ này nhiều cấu hình *hòa điểm*, nên GridSearchCV có thể chọn cấu hình xuất hiện trước theo quy tắc thứ tự ứng viên. Dataset PTIT trong Notebook cho kết quả khác: các cấu hình B1–B4 thắng đều là `(1,2)`.

Đối với cấu hình `alpha=0.1, ngram=(1,1)`, F1 của 5 lượt validation trong ví dụ nhỏ này là:

| Fold | F1 validation |
|---|---:|
| 1 | 1.00 |
| 2 | 1.00 |
| 3 | 0.67 (xấp xỉ) |
| 4 | 0.67 (xấp xỉ) |
| 5 | 0.67 (xấp xỉ) |

Trung bình `≈0.8` (thực tế ba giá trị cuối là `2/3`). Trong ví dụ này các cấu hình NB cùng đạt mean F1 = 0.8; `best_index_` trỏ đến cấu hình đầu được chọn khi hòa điểm. **Không được suy ra từ ví dụ minh họa rằng n-gram `(1,1)` luôn tốt hơn.**

### 6.5. Sau khi search fit xong, xem mô hình cuối

```python
best = search.best_estimator_
vectorizer = best.named_steps['vec']
classifier = best.named_steps['clf']

print(len(vectorizer.get_feature_names_out()))
print(classifier.classes_)
print(classifier.class_count_)
```

Trên ví dụ nhỏ này, sau khi **refit toàn bộ 20 câu**, `best` có vocabulary 34 đặc trưng với cấu hình unigram thắng; `classes_` = `[0,1]`, `class_count_` = `[10,10]`. **Các thống kê này thuộc mô hình refit trên 20 câu, không phải mô hình của riêng fold 1.**

---

## 7. CELL 13 NOTEBOOK 02: ĐỌC TỪNG ĐOẠN THỰC TẾ

```python
cache_dir = tempfile.TemporaryDirectory(prefix="spamham_cache_", dir=ARTIFACT_DIR)
cache = Memory(cache_dir.name, verbose=0)
PREPROCESS_RESULTS = []
for pre in TEXT:
    X_train_text = TEXT[pre].iloc[train_idx].to_numpy()
    for feature in FEATURES:
        for algorithm in MODELS:
            record = fit_experiment(pre, feature, algorithm, X_train_text, y_train, CV_FOLDS,
                                    seed=SEED, cache=cache, n_jobs=N_JOBS)
            PREPROCESS_RESULTS.append(record)
    print("Hoàn tất CV:", pre, flush=True)
```

### 7.1. `cache_dir` và `cache`

- `tempfile.TemporaryDirectory(...)`: tạo thư mục tạm để cache, nằm dưới `ARTIFACT_DIR`.
- `Memory(cache_dir.name, verbose=0)`: đối tượng `joblib.Memory` dùng để cache phép biến đổi có thể tái sử dụng trong Pipeline, giảm thời gian xử lý trong một số trường hợp.
- Đây **KHÔNG** phải mô hình `.joblib` được lưu để demo! Việc lưu mô hình được thực hiện bằng `joblib.dump(bundle, model_path)` ở cell 32.
- `cache_dir.cleanup()` ở cuối Notebook xóa cache tạm sau khi chạy.

### 7.2. Vòng `for pre in TEXT`: hai nhánh stopwords

Từ cell 9:

```python
TEXT = {
    'giữ stop words': prepared.tokens_keep_stopwords,
    'có loại stop words': prepared.tokens_remove_stopwords
}
```

Tức là hai dạng đầu vào văn bản đã được chuẩn bị. **`fit_experiment()` không tự xóa stopwords**; tên `pre` chỉ giúp lấy đúng cột và ghi record.

### 7.3. Vòng `for feature in FEATURES` và `for algorithm in MODELS`

Từ hai module:

```python
FEATURES = ('BoW', 'TF-IDF')
MODELS = ('Naive Bayes', 'SVM')
```

Vì thế, mỗi nhánh stopwords thực hiện **bốn** thí nghiệm:

| Vòng | Feature | Algorithm |
|---:|---|---|
| 1 | BoW | Naive Bayes |
| 2 | BoW | SVM |
| 3 | TF-IDF | Naive Bayes |
| 4 | TF-IDF | SVM |

Hai nhánh là **8 lần gọi** `fit_experiment`. Mỗi lời gọi trả một `record` có mô hình đã refit trên toàn Train của nhánh tương ứng. `PREPROCESS_RESULTS.append(record)` giữ tám record để so sánh CV.

### 7.4. `cv_comparison` — bảng so sánh stopwords

```python
cv_comparison = pd.DataFrame(PREPROCESS_RESULTS).pivot(
    index=['Đặc trưng', 'Thuật toán'],
    columns='Tiền xử lý',
    values='CV F1 (spam)'
)
```

- `pd.DataFrame(PREPROCESS_RESULTS)`: chuyển tám dictionary record thành bảng tám dòng.
- `.pivot(...)`: lấy tổ hợp đặc trưng/thuật toán làm hàng, nhánh stopwords làm cột và CV F1 làm giá trị.

**Bảng thật trong Notebook (làm tròn 4 chữ số):**

| Tổ hợp | CV F1 — loại stopwords | CV F1 — giữ stopwords |
|---|---:|---:|
| BoW + Naive Bayes | 0.5859 | **0.6077** |
| BoW + SVM | 0.6305 | **0.7087** |
| TF-IDF + Naive Bayes | 0.5835 | **0.5959** |
| TF-IDF + SVM | 0.7562 | **0.7736** |

Đó là điểm **validation của CV**, chưa dùng 7.474 mẫu Test.

### 7.5. `MAIN_PRE` được chọn như thế nào?

```python
MAIN_PRE = max(PREPROCESS_RESULTS, key=lambda r: r['CV F1 (spam)'])['Tiền xử lý']
```

Diễn giải từ trong ra ngoài:

1. `PREPROCESS_RESULTS`: tám record từ hai nhánh stopwords × bốn tổ hợp.
2. `key=lambda r: r['CV F1 (spam)']`: chỉ lấy trường F1 CV làm tiêu chí so sánh.
3. `max(...)`: trả **một record có CV F1 cao nhất trong cả tám record**.
4. `['Tiền xử lý']`: lấy tên nhánh xử lý của record thắng.

Trong output hiện có, record đạt 0.7736 là **TF-IDF + SVM giữ stopwords**, nên `MAIN_PRE = 'giữ stop words'`.

**Điểm cần nhớ:** đoạn code **không** tính trung bình F1 của bốn thuật toán trong mỗi nhánh rồi so sánh hai trung bình. Nó chọn nhánh chứa **ứng viên đơn lẻ tốt nhất** theo F1 CV.

### 7.6. `RESULTS` và gán B1–B4

```python
RESULTS = [r for r in PREPROCESS_RESULTS if r['Tiền xử lý'] == MAIN_PRE]
for i, record in enumerate(RESULTS, 1):
    record['Mã'] = f'B{i}'
```

Lọc lại bốn record từ nhánh `'giữ stop words'`. Thứ tự vòng lặp feature/model ban đầu trở thành:

- B1: BoW + Naive Bayes.
- B2: BoW + SVM.
- B3: TF-IDF + Naive Bayes.
- B4: TF-IDF + SVM.

`record['Mã']` chỉ thêm cột định danh; **không thay đổi mô hình hay chạy fit lại**.

### 7.7. Bảng CV của B1–B4 trong Notebook

| Mã | Đặc trưng | Thuật toán | CV Precision Spam | CV Recall Spam | CV F1 Spam |
|---|---|---|---:|---:|---:|
| B1 | BoW | Naive Bayes | 0.7582 | 0.5153 | 0.6077 |
| B2 | BoW | SVM | 0.7290 | 0.6901 | 0.7087 |
| B3 | TF-IDF | Naive Bayes | 0.8263 | 0.4723 | 0.5959 |
| B4 | TF-IDF | SVM | **0.9001** | 0.6806 | **0.7736** |

**Thông số thắng của từng tổ hợp:**

```text
B1: {'clf__alpha': 1.0, 'vec__ngram_range': (1,2)}
B2: {'clf__C': 1.0, 'clf__class_weight': None, 'vec__ngram_range': (1,2)}
B3: {'clf__alpha': 0.1, 'vec__ngram_range': (1,2)}
B4: {'clf__C': 0.3, 'clf__class_weight': None, 'vec__ngram_range': (1,2)}
```

**Tại sao đầu ra toàn `(1,2)`?** Vì đây là **tham số thắng** trong mỗi lần GridSearchCV. Cả `(1,1)` cũng đã được thử cho từng cấu hình; bảng này chỉ in bộ thắng. Từ output hiện tại **không suy ra được** F1 cụ thể của mọi ứng viên `(1,1)`, vì code không lưu `cv_results_` đầy đủ trong record.

### 7.8. `focus` là gì và chọn ở đâu?

```python
focus = max(RESULTS, key=lambda r: r['CV F1 (spam)'])
assert (focus['Đặc trưng'], focus['Thuật toán']) == ('TF-IDF', 'SVM')
```

- `max` chọn record có F1 cao nhất trong **B1–B4** của nhánh stopwords vừa chọn.
- Trong Notebook được lưu, đó là **B4: TF-IDF + SVM**, CV F1 = **0.7736**.
- `assert` kiểm tra kết quả có đúng B4 như phần khảo sát tiếp theo dự kiến. Nếu lần chạy khác chọn mô hình dẫn đầu khác thì chương trình báo lỗi để người làm xem lại phạm vi khảo sát, **không cố ép B4 trở thành mô hình thắng**.
- `focus['_model']` là Pipeline B4 đã fit lại trên toàn Train.
- `focus['Tham số']` giữ C=0.3, ngram=(1,2), class_weight=None.

Cell 16 nhận `focus` này để so với S1/S2/S3. **Đó là đầu ra quan trọng nhất của `experiments.py` cho giai đoạn cân bằng.**

---

## 8. TẠI SAO F1 TRONG CELL 13 VÀ CELL 20 KHÁC NHAU?

Hai con số có nguồn khác nhau:

```text
CELL 13 (fit_experiment)
    → chỉ có X_train_text, y_train, CV_FOLDS
    → chọn siêu tham số bằng trung bình F1 trên validation fold
    → B4 CV F1 = 0.7736
    → trả model đã refit trên toàn Train

CELL 20 (evaluate)
    → nhận model đã fit + X_test_text + y_test
    → dự đoán 7.474 bình luận Test độc lập
    → B4 Test F1 = 0.7334
```

**Kết quả Test của bốn tổ hợp cơ sở trong Notebook:**

| Mã | Accuracy Test | Precision Spam | Recall Spam | F1 Spam | AP |
|---|---:|---:|---:|---:|---:|
| B1 | 0.9141 | 0.7441 | 0.4953 | 0.5947 | 0.7320 |
| B2 | 0.9241 | 0.7110 | 0.6803 | 0.6953 | 0.7493 |
| B3 | 0.9257 | 0.8402 | 0.5142 | 0.6380 | 0.7988 |
| B4 | **0.9419** | **0.8818** | 0.6278 | **0.7334** | **0.8117** |

Vì test là dữ liệu chưa dùng trong việc chọn tham số, F1 Test thường khác F1 CV (có thể thấp hơn hoặc cao hơn). Không lấy giá trị Test để sửa lại alpha/C/ngram rồi báo đó vẫn là kiểm tra độc lập.

**Chỗ nối sang `imbalance.py`:** B4 dẫn đầu theo CV, nên cell 16 thử:

| Mã | Chiến lược | CV F1 Spam (Notebook) | Tham số thắng / ngưỡng |
|---|---|---:|---|
| B4 | Không điều chỉnh | 0.7736 | C=0.3, `(1,2)`, threshold 0 |
| S1 | `class_weight='balanced'` | 0.7580 | C=0.1, `(1,2)`, threshold 0 |
| S2 | Under-sampling Ham | 0.7629 | C=0.1, `(1,2)`, threshold 0 |
| S3 | Chỉnh ngưỡng score | **0.7876** | Giữ tham số B4; threshold ≈ -0.1776 |

Sau đó `best = max(CANDIDATES, key=lambda r: r['CV F1 (spam)'])` **ở cell 16** chọn **S3**. Đây là lựa chọn thêm ngoài phạm vi `fit_experiment` cơ sở; chưa phải bước Test.

---

## 9. MỤC RẤT DỄ BỊ HỎI: XEM ĐIỂM TỪNG `(1,1)` / `(1,2)` Ở ĐÂU?

Trong `fit_experiment`, đối tượng `search` là biến **cục bộ**; kết quả `record` chỉ giữ điểm/siêu tham số thắng và Pipeline refit. Vì vậy:

```python
record['Tham số']  # Có tham số thắng
record['_model']  # Có Pipeline đã fit
# record KHÔNG có đầy đủ search.cv_results_ của mọi cấu hình
```

Muốn xuất **tất cả kết quả của `(1,1)` và `(1,2)`** trên đúng Train, đúng folds, có thể đặt một cell kiểm tra bổ sung trong Notebook (sẽ chạy lại CV cho thí nghiệm được chọn):

```python
import pandas as pd
from sklearn.model_selection import GridSearchCV
from pipeline.modeling.experiments import make_pipe, CV_SCORING

X_train_text = TEXT['giữ stop words'].iloc[train_idx].to_numpy()
pipe, grid, jobs = make_pipe('TF-IDF', 'SVM', seed=SEED, cache=cache, n_jobs=N_JOBS)

search_dbg = GridSearchCV(
    pipe, grid, scoring=CV_SCORING,
    cv=CV_FOLDS, n_jobs=jobs, refit='f1', error_score='raise'
)
search_dbg.fit(X_train_text, y_train)

cols = ['param_vec__ngram_range', 'param_clf__C',
        'param_clf__class_weight', 'mean_test_precision',
        'mean_test_recall', 'mean_test_f1', 'rank_test_f1']
all_scores = pd.DataFrame(search_dbg.cv_results_)[cols]
print(all_scores.sort_values(['rank_test_f1', 'param_clf__C']).to_string(index=False))
print('Tham số thắng:', search_dbg.best_params_)
```

**Hiểu kết quả:** mỗi dòng = một cấu hình n-gram/C; `mean_test_f1` = F1 validation trung bình của cấu hình đó, `rank_test_f1` = xếp hạng dựa trên F1. Dòng có rank 1 là một phương án thắng. Bảng hiện có trong Notebook chỉ chứa dòng thắng của B4; các điểm ứng viên chưa được xuất.

Nếu muốn tránh chạy lại, có thể thay đổi `fit_experiment` *ở lần chạy sau* để lưu thêm `cv_results_` vào một bảng audit riêng, nhưng không nên nhét cả đối tượng GridSearchCV vào `record` dùng làm file mô hình demo nếu không cần. Để giải thích/cam kết số liệu cũ, không tự tạo điểm `(1,1)` khi chưa có bảng gốc.

---

## 10. KIỂM TRA BÊN TRONG PIPELINE ĐÃ FIT — BẠN NÊN BIẾT CÁCH DEBUG

**Lưu ý:** đoạn dưới chạy sau cell 13 vì lúc đó `focus['_model']` đã được fit.

```python
model = focus['_model']
print(model.named_steps.keys())
# dict_keys(['vec', 'clf'])

vec = model.named_steps['vec']
clf = model.named_steps['clf']

print(type(vec).__name__)  # TfidfVectorizer
print(type(clf).__name__)  # LinearSVC

print('Best params:', focus['Tham số'])
print('Vocabulary size:', len(vec.get_feature_names_out()))
print('Classes:', clf.classes_)
print('Coef shape:', clf.coef_.shape)
print('Intercept:', clf.intercept_)

X_some = TEXT[focus['Tiền xử lý']].iloc[train_idx[:3]].to_numpy()
Z_some = vec.transform(X_some)  # Chỉ transform, không học lại
print('Vector shape:', Z_some.shape)
print('Pred:', clf.predict(Z_some))
# Hoặc để Pipeline tự transform + predict:
print('Pipeline pred:', model.predict(X_some))
```

Các điều kiện cần đúng:

- `Z_some.shape[0] = 3` vì có 3 bình luận.
- `Z_some.shape[1]` bằng số đặc trưng trong `vec.vocabulary_`.
- SVM `coef_` có số cột tương ứng đúng số đặc trưng, với hai lớp thường có một hàng trọng số.
- `clf.predict(Z_some)` và `model.predict(X_some)` phải khớp khi dùng cùng ngưỡng **mặc định 0**. Nếu là S3, **quy tắc chọn threshold** trong `evaluation.predict_record` có thể cho nhãn khác `model.predict` mặc định.

### Vấn đề cache có làm mô hình bị chia sẻ sai giữa các fold không?

Không. `memory=cache` cho phép cache kết quả xử lý trung gian của Pipeline; GridSearchCV vẫn tạo/clone estimator để fit cho từng cấu hình/fold. Không được coi `cache` là hành động học trước trên cả CSV.

### `n_jobs=2` có nghĩa chạy 2 fold đồng thời hay 2 mô hình đồng thời?

Là số tác vụ song song do joblib quản lý trong GridSearchCV; lịch chạy cụ thể có thể thay đổi. **Không đồng nghĩa** “có hai mô hình”, “chỉ chạy hai fold” hoặc “hai lớp”.

### `seed=42` có đủ để mọi kết quả bất biến không?

Seed hỗ trợ tái lập, nhưng còn phụ thuộc dữ liệu, quy tắc chia, phiên bản scikit-learn/Liblinear, môi trường và các bước xử lý khác. Với LinearSVC `dual=False` ở cấu hình dự án, tài liệu scikit-learn chỉ ra `random_state` không ảnh hưởng tới bộ tối ưu đang dùng.

---

## 11. NHỮNG HIỂU NHẦM CẦN TRÁNH KHI THUYẾT TRÌNH

| Hiểu nhầm | Cách nói đúng |
|---|---|
| “`make_pipe()` đã huấn luyện NB/SVM.” | Chưa; mới tạo Pipeline. `search.fit()` mới thực sự học. |
| “`X` vào `fit_experiment` là vector TF-IDF.” | Không; là các **chuỗi token** của Train. Pipeline tự vector hóa trong từng fold. |
| “Mô hình dùng nguyên từ điển chung học trên cả dataset.” | Sai; mỗi fold học vocabulary/IDF riêng trên fold Train; cuối cùng refit trên toàn Train. |
| “GridSearchCV tự chia năm fold ngẫu nhiên.” | Không ở dự án này; `cv=CV_FOLDS` do `grouped_folds` cung cấp. |
| “Thử mỗi `(1,1)` vì file khởi tạo n-gram là `(1,1)`.” | Sai; lưới trả cả `(1,1)` và `(1,2)`, GridSearchCV thử cả hai. |
| “Một tổ hợp SVM có bốn mô hình vì có bốn C.” | Có bốn giá trị C × hai n-gram = **8 cấu hình**, mỗi cấu hình qua năm fold; cuối cùng chỉ một cấu hình thắng được refit. |
| “CV Precision tốt nhất, CV Recall tốt nhất, CV F1 tốt nhất đều do ba cấu hình riêng.” | Record lấy cả ba số từ **một cấu hình thắng theo F1**. |
| “`mean_test_f1` là điểm trên Test cuối.” | Sai; là điểm **validation trong CV**, vì toàn bộ dữ liệu đưa cho GridSearchCV chỉ là Train. |
| “Mô hình cuối bằng trung bình 5 mô hình đã fit.” | Sai; chọn **siêu tham số** bằng trung bình điểm, rồi fit lại một mô hình trên toàn Train. |
| “Chạy GridSearchCV một lần cho cả B1–B4.” | Thực tế cell 13 gọi `fit_experiment` **8 lần** (hai nhánh stopwords × bốn cặp), mỗi lần một GridSearchCV. |
| “`MAIN_PRE` là nhánh có F1 trung bình của bốn mô hình lớn nhất.” | Sai; là nhánh của **record đơn lẻ có CV F1 cao nhất trong 8 record**. |
| “B4 là mô hình cuối để demo.” | B4 dẫn đầu mô hình cơ sở; sau cell 16 dự án chọn **S3** bằng CV F1 rồi mới demo. |
| “Đánh giá Test xong mới chọn C.” | Sai; chọn C trước ở cell 13 bằng CV; Test được chấm ở cell 20. |
| “`cache` là file .joblib lưu mô hình.” | Sai; `cache` tối ưu tính toán trung gian; file `.joblib` của sản phẩm được lưu ở cell 32. |
| “Đặt `class_weight='balanced'` làm thay đổi tỷ lệ Train/Test.” | Sai; nó tác động mức phạt lớp khi fit SVM, không chia lại Train/Test. |

---

## 12. BẢNG PHÂN BIỆT BƯỚC NÀY VỚI HAI FILE TRƯỚC VÀ SAU

| Module | Gọi hàm chính | Nhận | Trả về | Chạy fit trực tiếp? |
|---|---|---|---|---|
| `text_features.py` | `make_vectorizer()` | `"BoW"`/`"TF-IDF"` | Vectorizer **chưa fit** + lưới n-gram | Không |
| `classifiers.py` | `make_classifier()` | `"Naive Bayes"`/`"SVM"` | Classifier **chưa fit** + lưới alpha/C | Không |
| **`experiments.py`** | **`make_pipe()`** | Loại đặc trưng & classifier | Pipeline chưa fit + lưới gộp | Không |
| **`experiments.py`** | **`fit_experiment()`** | Chuỗi token Train, y Train, folds | CV metrics + best params + **Pipeline đã fit** | **Có, trong `search.fit()`** |
| `imbalance.py` | `fit_balanced_svm`, `undersampled_search`, `tune_svm_threshold` | Train, folds, cấu hình B4 | Record/model chiến lược | Có fit trong các nhánh của Train, không fit Test |
| `evaluation.py` | `evaluate()` | Record đã fit, chuỗi Test, y Test | Cập nhật record với Accuracy/P/R/F1/AP/Test predictions | **Không** |
| `predict.py` | `predict_spam()` | Bình luận mới + model bundle | DataFrame score + nhãn Spam/Ham | **Không** |

---

## 13. CÂU HỎI ÔN TẬP KÈM TRẢ LỜI NGẮN

1. **Mục tiêu của `experiments.py`?** Ghép vectorizer và classifier, tìm siêu tham số bằng CV trên Train, fit lại cấu hình thắng và trả Pipeline đã học.
2. **`make_pipe` có biết dữ liệu chưa?** Chưa. Nó chỉ biết chọn kiểu vector hóa và thuật toán nào.
3. **Đầu vào `X` của `fit_experiment`?** Mảng chuỗi token của tập Train; ma trận số được tạo bên trong Pipeline.
4. **Đầu vào `y`?** Nhãn thật 0/1 khớp từng câu Train, được classifier học và metric dùng chấm validation.
5. **`folds` từ đâu?** `grouped_folds(keys[train_idx], y_train)` ở cell 11.
6. **Vì sao phải grouped folds?** Hạn chế nội dung cùng khóa xuất hiện ở cả train và validation; cho quy trình chia dữ liệu ổn định và phục vụ đối chiếu với mô hình khác.
7. **`GridSearchCV` thử tham số bằng cách nào?** Lấy tích Descartes của các danh sách tham số và fit/đánh giá mỗi cấu hình trên từng fold.
8. **`refit='f1'` nghĩa gì?** Lựa chọn theo F1 Spam trung bình của validation và fit lại cấu hình thắng trên toàn Train.
9. **Một B4 thử bao nhiêu cấu hình?** 8 (hai n-gram × bốn C, class_weight một lựa chọn None); qua năm fold = 40 lượt fit CV + 1 refit.
10. **Một B1 thử bao nhiêu?** 6 (hai n-gram × ba alpha); qua năm fold = 30 lượt fit CV + 1 refit.
11. **Hai nhánh stopwords và B1–B4 có bao nhiêu lượt fit CV?** 280 (56 cấu hình × 5 fold), cộng 8 lần refit.
12. **Chọn stopwords ở đâu?** Cell 13: `MAIN_PRE = max(PREPROCESS_RESULTS, key=CV F1)['Tiền xử lý']`.
13. **B4 được chọn ở đâu?** `focus = max(RESULTS, key=CV F1)` ở cell 13, sau khi lọc nhánh stopwords thắng.
14. **B4 có là sản phẩm cuối cùng không?** Chưa. Cell 16 so B4 với S1/S2/S3, chọn S3 trong Notebook hiện có.
15. **`best_params_` là gì?** Dictionary tham số thắng của GridSearchCV.
16. **`best_estimator_` là gì?** Pipeline đã fit lại trên toàn Train bằng tham số thắng, không phải tập hợp 5 mô hình.
17. **`mean_test_f1` là Test F1 không?** Không. Là validation CV từ dữ liệu Train.
18. **Làm sao xem F1 tất cả n-gram đã thử?** Xem `search.cv_results_`; record hiện tại chỉ giữ điểm và tham số thắng.
19. **Tại sao `fit_experiment` không có `X_test`/`y_test`?** Để không dùng Test trong việc tối ưu cấu hình.
20. **SVM có xác suất Spam không?** `LinearSVC` trả decision score (không phải xác suất); ngưỡng mặc định 0, S3 thay đổi ngưỡng dựa trên Train.
21. **NB `predict_proba` khác F1 thế nào?** Xác suất NB là score cho từng câu; F1 được tính từ nhãn dự đoán so với nhãn thật trên một tập đánh giá.
22. **`pre` có gọi hàm xóa stopwords không?** Không; `pre` là nhãn chỉ nhánh chuỗi token đã được chuẩn bị ở cell 9.
23. **Tại sao còn fit lại toàn Train sau CV?** Để mô hình cuối có thể học từ toàn bộ 30.759 dòng Train, thay vì chỉ 4/5 Train của một fold.
24. **Khi giảng viên hỏi “Vì sao CV F1 B4 0.7736 nhưng Test F1 0.7334?”** Vì hai điểm tính trên hai tập khác nhau: trung bình validation trong Train và Test độc lập; chọn tham số trên CV không bảo đảm Test đạt cùng điểm.

---

## 14. CÁCH TỰ ĐỌC THƯ VIỆN SCiKIT-LEARN (CÓ LIÊN KẾT TÀI LIỆU NHÓM)

- `Pipeline.fit`, `Pipeline.predict`, `Pipeline.named_steps`: xem [`sklearn.pipeline.Pipeline`](https://scikit-learn.org/1.6/modules/generated/sklearn.pipeline.Pipeline.html).
- `GridSearchCV.fit`, `best_params_`, `best_index_`, `cv_results_`, `best_estimator_`: xem [`sklearn.model_selection.GridSearchCV`](https://scikit-learn.org/1.6/modules/generated/sklearn.model_selection.GridSearchCV.html).
- `ParameterGrid`: xem [`sklearn.model_selection.ParameterGrid`](https://scikit-learn.org/1.6/modules/generated/sklearn.model_selection.ParameterGrid.html) — cách liệt kê các cấu hình.
- `clone`: xem [`sklearn.base.clone`](https://scikit-learn.org/1.6/modules/generated/sklearn.base.clone.html), tạo đối tượng estimator mới chưa fit.
- Để đọc nguồn của vectorizer và classifier **đã học ở bước 02/03**, tra `PHI_HIEU_4_LOP_SCIKIT_LEARN.md` (số dòng ứng với scikit-learn **1.6.1 của `.venv` nhóm**, có thể khác khi dùng phiên bản khác).

**Phân biệt mã thật và diễn giải:** Phần trên trích các đoạn `make_pipe`, `fit_experiment`, cell 13 đúng theo module/Notebook bạn gửi; vòng `for params in every_combination(...)` ở mục 5.5 là **pseudocode để học**, không phải hàm có tên đó trong scikit-learn.

---

## 15. KẾT LUẬN DỄ NHỚ ĐỂ BẢO VỆ

> “Sau khi đã có dữ liệu Train và các fold, em dùng `make_pipe()` để ghép bộ tạo vector BoW hoặc TF-IDF với Naive Bayes hoặc SVM thành Pipeline. Hàm `fit_experiment()` dùng `GridSearchCV` thử hai cấu hình unigram/unigram+bigram và nhiều alpha/C, trên đúng năm fold đã chia. Trong mỗi fold, vectorizer học vocabulary và IDF từ phần Train, classifier học từ ma trận đặc trưng và nhãn, rồi dự đoán Validation để tính Precision, Recall và F1 của Spam. Nhóm chọn cấu hình có F1 trung bình cao nhất, sau đó `refit='f1'` huấn luyện lại mô hình thắng trên toàn bộ Train. Hàm trả record gồm điểm CV, tham số thắng và Pipeline đã fit. Notebook tiếp tục chọn nhánh stopwords và B4 theo CV, thử chiến lược cân bằng ở cell 16, rồi mới đánh giá Test tại cell 20.”

**Từ khóa học thuộc:** `Pipeline` = nối các bước; `GridSearchCV` = thử cấu hình trên CV; `search.fit` = bắt đầu học; `refit='f1'` = chọn theo F1 và fit lại toàn Train; `best_estimator_` = Pipeline cuối đã fit; `cv_results_` = bảng điểm từng cấu hình trên validation; Test chỉ được đánh giá sau khi đã chốt mô hình.

**Bước học tiếp theo:** `imbalance.py` + cell 15–16 để hiểu rõ S1 `class_weight`, S2 undersampling, S3 threshold tuning (vì đó là ba thử nghiệm riêng sử dụng kết quả B4 của bước này).
