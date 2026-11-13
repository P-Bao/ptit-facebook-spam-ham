# BƯỚC 05 — `imbalance.py`: xử lý dữ liệu mất cân bằng Spam/Ham (bản học chi tiết)

> **Phạm vi:** Phần xây dựng mô hình truyền thống của bạn, kết hợp **`pipeline/modeling/imbalance.py` (bản `imbalance(1).py`)**, **cell 14–16** của `02_thuc_nghiem_va_demo(1).ipynb`, với những điểm nối cần thiết sang `training_utils.py`, `experiments.py`, `classifiers.py`, `text_features.py`, `evaluation.py`. Không phân tích cào dữ liệu/gán nhãn hay huấn luyện viBERT.
>
> **Quy ước nguồn:** Các kết quả thực tế trong những bảng bên dưới **lấy từ output lưu trong Notebook 02**, không phải được huấn luyện lại. Các ví dụ có số lượng nhỏ là **ví dụ để học**, không phải dữ liệu thật. Bản của nhóm ghi tài liệu scikit-learn **1.6.1**; nếu chạy trên phiên bản khác, chi tiết mã nguồn/thứ tự dòng có thể thay đổi.

---

## Mục lục

1. Bước này đứng ở đâu trong dự án?
2. Bối cảnh thực tế, định nghĩa và mục tiêu xử lý mất cân bằng
3. Các khái niệm và công thức bắt buộc
4. Sơ đồ đầu vào–đầu ra của `imbalance.py` và Notebook
5. Tổng quan code, import và thư viện liên quan
6. S1 — `fit_balanced_svm`: trọng số lớp
7. `sample_training_indices`: hàm lấy mẫu của S2
8. `mean_cv_scores`: tổng hợp điểm CV đúng cách
9. `undersampled_search`: toàn bộ vòng lặp S2
10. S3 — `tune_svm_threshold`: chọn ngưỡng từ score OOF
11. Các hàm thư viện được gọi ở bước này, đi sâu từng hàm
12. Giải thích cell 14, 15, 16 theo luồng chạy thật
13. Các record, biến và cách chọn ứng viên thắng
14. Đọc chính xác kết quả CV và Test từ Notebook
15. So sánh S1/S2/S3; vì sao S3 được chọn
16. Hai ví dụ tính tay và mô phỏng Python
17. Chuyện Train/Validation/Test và chống data leakage
18. Điểm cần thận trọng, giới hạn đánh giá và độ công bằng
19. Câu hỏi giảng viên có thể hỏi (có đáp án)
20. Bản nói miệng và checklist tự kiểm tra
21. Nguồn gốc code và hướng đọc tiếp

---

# 1. Bước này nằm ở đâu?

```text
Notebook 02
  ↓ Cell 11: training_utils.py
Train: 30.759 bình luận; y_train; CV_FOLDS
  ↓ Cell 13: text_features.py + classifiers.py + experiments.py
B1 = BoW + NB
B2 = BoW + SVM
B3 = TF-IDF + NB
B4 = TF-IDF + SVM   ← CV F1 cơ sở tốt nhất = 0.7736
  ↓ Cell 14–16: imbalance.py (BƯỚC HIỆN TẠI)
   ├─ S1: TF-IDF + SVM + class_weight="balanced"
   ├─ S2: TF-IDF + SVM + chỉ giữ 2 Ham / 1 Spam trong TRAIN
   └─ S3: TF-IDF + SVM, GIỮ siêu tham số B4, tìm threshold
  ↓ so CV F1 spam giữa B1–B4 và S1–S3
S3 thắng CV với F1 = 0.7876, threshold ≈ -0.1776
  ↓ Cell 20: evaluation.py, lần đầu xem TEST độc lập
Test F1 S3 = 0.7552
  ↓ Cell 32: lưu bundle và predict.py sử dụng để dự đoán mới
```

**Đừng nhầm:** `imbalance.py` không đọc CSV, không tự tạo fold, không huấn luyện cả 4 tổ hợp. Nó **nhận** `X_tr_text`, `y_train`, `CV_FOLDS` đã có; tận dụng các hàm xây Pipeline và thử **ba cách cải tiến dành riêng cho TF-IDF + SVM**. Mọi chiến lược chỉ sử dụng Train để chọn cấu hình; Test chưa được xem.

## 1.1. Những biến phải có sẵn trước khi chạy

| Biến từ Notebook | Kiểu dữ liệu / ý nghĩa | Được tạo từ đâu |
|---|---|---|
| `MAIN_PRE` | Tên nhánh giữ/bỏ stopwords được chọn bằng CV; thực tế là **giữ stopwords** | Cell 13 |
| `TEXT[MAIN_PRE]` | `pandas.Series` chứa chuỗi token đã chuẩn bị | Cell 9 |
| `train_idx`, `test_idx` | Vị trí dòng trong **toàn CSV** | Cell 11 |
| `X_tr_text` | `numpy.ndarray` gồm chuỗi token của **Train** (30.759 câu) | Cell 16, lấy từ TEXT |
| `y_train` | `numpy.ndarray` nhãn 0/1 cho Train | Cell 11 |
| `CV_FOLDS` | 5 cặp `(train, valid)` vị trí **bên trong X_tr_text/y_train** | `grouped_folds()` cell 11 |
| `focus` | Record B4 đã fit, chứa `focus['_model']`, `focus['Tham số']`, điểm CV | Cell 13 |
| `SEED` | Seed cố định của project, thường 42 | Cell cấu hình |
| `cache`, `N_JOBS` | Cấu hình cache Pipeline và mức song song | Cell 13 |

Lưu ý: `X_tr_text` vẫn là **chuỗi token**, chưa phải vector TF-IDF. Khi S1/S2/S3 gọi `Pipeline.fit`, `TfidfVectorizer` sẽ tự học từ vựng/IDF trên phần Train được truyền vào và tạo ma trận số cho `LinearSVC.fit`.

---

# 2. Tại sao phải xử lý mất cân bằng lớp?

## 2.1. Số liệu thực tế

| Tập | Ham (0) | Spam (1) | Tổng | % Spam |
|---|---:|---:|---:|---:|
| **Train** | **26.384** | **4.375** | **30.759** | **14,22%** |
| **Test** | **6.523** | **951** | **7.474** | **12,72%** |

Train có xấp xỉ **6,03 Ham trên 1 Spam**. Mô hình dễ học thiên hướng lớp Ham vì có nhiều ví dụ Ham hơn.

Ví dụ cực đoan: mô hình luôn dự đoán Ham cho mọi bình luận Test sẽ đúng 6.523/7.474 ≈ **87,28% Accuracy**, nhưng **không phát hiện được Spam nào** → Recall Spam = 0; F1 Spam = 0. Accuracy một mình không đủ.

## 2.2. Mất cân bằng là gì? Có nhất thiết gây lỗi không?

**Class imbalance** là tình trạng số lượng quan sát của các lớp chênh lệch đáng kể. Không phải mọi trường hợp mất cân bằng đều khiến thuật toán sai. Điều quan trọng là đánh giá xem lớp cần quan tâm (Spam) có bị bỏ sót quá nhiều hay không.

Trong bài này Spam là **lớp dương `1`** và nhóm ưu tiên so sánh **F1 Spam**. Mục tiêu không phải ép số mẫu bằng nhau bằng mọi giá: ta đang thử nhiều lựa chọn rồi đánh giá tác động tới Precision–Recall.

## 2.3. Tại sao chỉ thử cải tiến trên B4?

Ở cell 13, **B4 = TF-IDF + SVM** là ứng viên cơ sở có CV F1 Spam cao nhất. Cell 14 quyết định dùng B4 làm đối chứng, sau đó thử S1/S2/S3. Đây là lựa chọn thiết kế/thời gian của nhóm; **không có nghĩa** S1/S2/S3 đã được áp dụng, tối ưu trên cả B1–B3.

---

# 3. Lý thuyết và công thức cần nắm

## 3.1. Nhãn và bốn loại dự đoán

- Ham = `0` (negative).
- Spam = `1` (positive).
- **TP**: Spam thật, dự đoán Spam.
- **FP**: Ham thật, dự đoán Spam → cảnh báo nhầm.
- **FN**: Spam thật, dự đoán Ham → bỏ sót Spam.
- **TN**: Ham thật, dự đoán Ham.

## 3.2. Precision, Recall, F1 — để so các chiến lược

\[
Precision_{spam}=\frac{TP}{TP+FP}
\]

\[
Recall_{spam}=\frac{TP}{TP+FN}
\]

\[
F1_{spam}=\frac{2PR}{P+R}=\frac{2TP}{2TP+FP+FN}
\]

**Precision cao:** trong số các câu bị cảnh báo Spam, nhiều câu thực sự là Spam → ít FP.

**Recall cao:** trong tổng Spam thực tế, phát hiện được nhiều câu → ít FN.

**F1:** dung hòa hai chỉ số. Khi hạ ngưỡng hoặc tăng độ ưu tiên Spam, Recall thường tăng nhưng có thể làm FP tăng và Precision giảm. Không được khẳng định chiến lược nào lúc nào cũng tốt hơn.

## 3.3. Hyperparameter và model parameter

- **Hyperparameter**: do người thiết kế đặt/thử trước khi fit, như `C`, `ngram_range`, `class_weight`, `ratio`, `threshold` được chọn bằng CV.
- **Model parameter**: thư viện học từ Train trong `fit`, như `LinearSVC.coef_`, `intercept_`; `TfidfVectorizer.vocabulary_`, `idf_`.
- **Threshold**: ngưỡng sau khi có score, dùng quyết định ra nhãn; **không phải hệ số `coef_` của SVM**.

## 3.4. Tham số C của SVM

`LinearSVC` học ranh giới tuyến tính, điểm quyết định xấp xỉ

\[
s(x)=w^Tx+b.
\]

Với cấu hình SVM của dự án (phạt L2, `squared_hinge` mặc định, `dual=False`), ý tưởng hàm mục tiêu là

\[
\frac12\lVert w\rVert^2 + C\sum_i\max(0,1-y_i'(w^Tx_i+b))^2,
\]

trong đó nhãn toán học `y_i'` là −1 hoặc +1 (thư viện tự ánh xạ từ nhãn 0/1). **C** càng cao thì nhấn mạnh giảm vi phạm/lỗi trên Train càng nhiều; CV phải lựa chọn C phù hợp. Trong S1, trọng số lớp làm mức phạt hiệu dụng phụ thuộc nhãn.

## 3.5. Three strategies — khác nhau ở đối tượng bị thay đổi

| Phương pháp | Số mẫu đưa vào fit | Trọng số lỗi trong SVM | Ngưỡng dự đoán |
|---|---:|---|---|
| B4 | Giữ tất cả Train | Mặc định | `0.0` |
| S1 | Giữ tất cả Train | `class_weight='balanced'` | `0.0` |
| S2 | Chỉ giữ toàn Spam và tối đa 2×Spam Ham | Không cân bằng trọng số | `0.0` |
| S3 | Giữ tất cả Train | Mặc định, cùng siêu tham số B4 | Ngưỡng học từ OOF Train |

**Quan trọng:** S1 thay đổi quá trình `fit`; S2 thay đổi tập mẫu dùng `fit`; S3 thay đổi cách biến **score** thành nhãn (SVM của S3 vẫn được fit trên dữ liệu Train).

---

# 4. Sơ đồ đầu vào — đầu ra của file

```text
              X_tr_text, y_train, CV_FOLDS
                          │
                    [ B4 / focus ]
                          │
            ┌─────────────┼─────────────┐
            ▼             ▼             ▼
          S1             S2             S3
 fit_balanced_svm  undersampled_search  tune_svm_threshold
            │             │             │
  class_weight=     chỉ lấy mẫu       score OOF từ
  'balanced'        fold Train        các fold Train
            │             │             │
     GridSearchCV    ParameterGrid    thử nhiều ngưỡng
            │             │             │
      fit toàn       fit toàn Train    fit toàn Train
      Train          đã giảm Ham       + threshold riêng
            │             │             │
         record       tuple 4 phần     tuple 3 phần
            │             │             │
            └─────────────┼─────────────┘
                          ▼
 Notebook 02 cell 16 đóng gói record, so CV F1
                          ▼
              best = mô hình có CV F1 cao nhất
                          ▼
 Notebook cell 20 đánh giá trên TEST giữ riêng
```

**Trả về không đồng nhất:**

- `fit_balanced_svm()` → **dictionary record** vì gọi thẳng `fit_experiment()`.
- `undersampled_search()` → `(fitted_pipeline, cv_scores, best_params, n_fit_samples)`.
- `tune_svm_threshold()` → `(fitted_pipeline, cv_scores, threshold)`.

Do đó, **cell 16 phải tự đóng gói tuple của S2/S3 thành dictionary record** để so sánh chung, chứ không phải bản thân mọi hàm đều trả record cùng cấu trúc.

---

# 5. Đọc import trong `imbalance.py`

```python
import numpy as np
from sklearn.base import clone
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import ParameterGrid
```

| Import | Tác dụng cụ thể |
|---|---|
| `numpy as np` | Mảng chỉ số, lọc nhãn, random có seed, nối/trộn mảng, score, phân vị (`quantile`) |
| `clone` | Tạo estimator/Pipeline **mới chưa fit**, cùng tham số nhưng không thừa hưởng mô hình đã học |
| `precision_score`, `recall_score`, `f1_score` | Tính chỉ số trên **nhãn validation**, lớp dương mặc định `1=Spam` |
| `ParameterGrid` | Biến dictionary các lựa chọn thành từng bộ tham số cụ thể để thử thủ công trong S2 |

**Vì sao `clone()` rất quan trọng?** Nếu dùng tiếp mô hình đang chứa từ vựng, IDF, trọng số từ một fold khác, CV có thể sai vì rò rỉ trạng thái học. `clone(pipe)` giúp mỗi lượt fit có một bản mới và chỉ được học từ fold Train hiện tại.

Các hàm `fit_balanced_svm`, `sample_training_indices`, `mean_cv_scores`, `undersampled_search`, `tune_svm_threshold` nằm trong `imbalance.py`; phần khai báo chọn chiến lược, nối record và tạo bảng nằm **ở Notebook**, không phải trong file Python này.

---

# 6. S1 — `fit_balanced_svm()`: cân bằng bằng trọng số lớp

## 6.1. Code gốc (đúng bản nhóm)

```python
def fit_balanced_svm(pre, X, y, folds, *, seed=42, cache=None, n_jobs=2):
    """S1: fit TF-IDF + SVM với class_weight='balanced' trên toàn train.

    Chỉ đổi trọng số lỗi giữa hai lớp. Không bớt mẫu, không đổi ngưỡng.
    Bên trong vẫn dùng CV để chọn n-gram/C như cấu hình cơ sở.
    """
    from pipeline.modeling.experiments import fit_experiment

    return fit_experiment(pre, "TF-IDF", "SVM", X, y, folds,
                          seed=seed, cache=cache, n_jobs=n_jobs,
                          class_weight="balanced", strategy="Trọng số balanced")
```

## 6.2. Đầu vào, tác dụng, đầu ra

**Đầu vào:** `pre` (nhãn nhánh stopwords), `X` (chuỗi token Train), `y` (0/1 Train), `folds` (CV trong Train), thêm seed/cache/n_jobs.

**Tác dụng:** Gọi lại `fit_experiment()` từ bước 04, nhưng với `class_weight='balanced'`; cố định đặc trưng **TF-IDF**, bộ phân loại **SVM**. `GridSearchCV` vẫn thử 2 n-gram × 4 C = **8 cấu hình**, trên 5 fold → 40 lần fit validation, sau đó fit lại cấu hình thắng trên **toàn Train**.

**Đầu ra:** `record` chứa tên chiến lược, CV Precision/Recall/F1, best_params, `_model` đã fit và `_threshold=0.0`.

**Không làm:** Không thay đổi số dòng Train, không chọn ngưỡng mới, không thay đổi nhãn, không đọc Test.

## 6.3. Bên trong `class_weight='balanced'` hoạt động như thế nào?

Scikit-learn tính trọng số lớp theo công thức:

\[
w_c = \frac{N}{K\,N_c},
\]

với `N` là số mẫu trong **tập đang fit**, `K=2` là số lớp, `N_c` là số mẫu lớp `c`.

Với toàn Train của dự án:

- N = 30.759.
- Ham = 26.384.
- Spam = 4.375.

\[
w_{ham}=\frac{30759}{2\cdot26384}\approx0.58291
\]

\[
w_{spam}=\frac{30759}{2\cdot4375}\approx3.51531
\]

**Ý nghĩa:** Mẫu Spam nhận trọng số phạt cao hơn Ham (tỷ lệ khoảng 6,03 lần) khi SVM học. Không phải thuật toán nhân xác suất Spam lên 3,5 ở bước dự đoán; trọng số tác động trong **bài toán tối ưu khi fit**.

Với `C=0.1` (C thắng của S1 trong Notebook), mức phạt hiệu dụng theo lớp tỷ lệ:

- Ham: `0.1 × 0.58291 ≈ 0.05829`.
- Spam: `0.1 × 3.51531 ≈ 0.35153`.

Đây chỉ là minh họa từ **toàn Train**. **Khi chạy CV, mỗi fold tính lại trọng số theo chính bốn fold Train của lượt đó.** Không dùng số liệu toàn Train để áp cho mọi fold một cách thủ công.

Ví dụ nhỏ: 8 Ham + 2 Spam, tổng 10:

- Ham: `10/(2×8)=0.625`.
- Spam: `10/(2×2)=2.5`.

Mất cân bằng càng lớn, trọng số lớp ít mẫu càng được tăng tương đối.

## 6.4. `fit()` của LinearSVC dùng trọng số ra sao?

Từ bước 03, bạn biết SVM học `coef_` và `intercept_`. Khi `class_weight='balanced'`, `LinearSVC.fit(X_vector, y)` chuyển thông tin lớp và trọng số vào phần huấn luyện Liblinear (`_fit_liblinear`). Về mặt ý tưởng, hàm mục tiêu dùng `C × weight_c` cho các ví dụ lớp `c` thay vì C giống nhau cho mọi lớp.

**Tác động thường gặp:** SVM có động lực tránh bỏ sót Spam, nên Recall Spam có thể tăng, nhưng cũng dễ gắn nhầm Ham thành Spam → Precision giảm. Kết quả thực tế của Notebook xác nhận kiểu đánh đổi này, nhưng không phải định luật bắt buộc.

## 6.5. Cách Notebook gọi S1

```python
balanced = fit_balanced_svm(
    MAIN_PRE, X_tr_text, y_train, CV_FOLDS,
    seed=SEED, cache=cache, n_jobs=N_JOBS
)
balanced["Mã"] = "S1"
```

`balanced` là dictionary từ `fit_experiment()`. Notebook thêm khóa `"Mã":"S1"` để in bảng.

**Kết quả CV thật:** S1 thắng ở `clf__C=0.1`, `vec__ngram_range=(1,2)`, `clf__class_weight='balanced'`; CV Precision=0.7233, Recall=0.7964, F1=0.7580.

---

# 7. S2 — `sample_training_indices()`: chỉ giảm Ham ở tập được phép học

## 7.1. Code gốc

```python
def sample_training_indices(y, indices, seed=42, ratio=2):
    """S2: giữ spam, lấy tối đa ratio ham/spam từ indices được phép."""
    indices = np.asarray(indices)
    spam = indices[np.asarray(y)[indices] == 1]
    ham = indices[np.asarray(y)[indices] == 0]
    rng = np.random.default_rng(seed)
    ham_count = min(ratio * len(spam), len(ham))
    sampled = rng.choice(ham, size=ham_count, replace=False)
    selected = np.concatenate([spam, sampled])
    rng.shuffle(selected)
    return selected
```

## 7.2. Vì sao `indices` rất quan trọng?

- `y` là danh sách nhãn **toàn bộ Train**.
- `indices` chỉ rõ **những dòng nào hiện được phép dùng để học**.

Khi CV, `indices` là tập chỉ số **train của fold**; validation không được xuất hiện trong danh sách đó.

Khi đã chọn tham số và fit mô hình cuối, `indices = np.arange(len(y))` cho phép xét toàn bộ Train.

Do đó, hàm này **không nhận Test**, **không trực tiếp lấy từ Validation** và **không làm rò rỉ nhãn validation vào huấn luyện**.

## 7.3. Từng dòng xử lý

### `indices = np.asarray(indices)`

Chuyển danh sách/array index về NumPy array. Điều này cần cho việc truy cập nhiều vị trí cùng lúc và lọc bằng boolean mask.

### `spam = indices[np.asarray(y)[indices] == 1]`

1. `np.asarray(y)` biến nhãn thành NumPy array.
2. `[indices]` lấy nhãn **chỉ tại các vị trí được phép**.
3. `==1` tạo boolean mask `True/False`.
4. `indices[mask]` trả về **vị trí các dòng Spam**, không trả về văn bản hay nhãn.

Ví dụ:

```python
y       = np.array([0, 1, 0, 1, 0, 0, 0, 0])
indices = np.array([0, 1, 2, 3, 4, 6])  # chỉ những dòng được phép

y[indices]                     # [0, 1, 0, 1, 0, 0]
y[indices] == 1                # [F, T, F, T, F, F]
indices[y[indices] == 1]       # [1, 3] = vị trí Spam
indices[y[indices] == 0]       # [0, 2, 4, 6] = vị trí Ham
```

### `ham_count = min(ratio*len(spam), len(ham))`

Giữ **tối đa** `ratio` Ham cho mỗi Spam; nếu dữ liệu Ham vốn đã ít thì giữ hết, không tạo thêm mẫu Ham.

- Nếu 2 Spam và 6 Ham, `ratio=2`: chọn 4 Ham.
- Nếu 2 Spam và 3 Ham: chọn 3 Ham, vì `min(4,3)=3`.

### `rng = np.random.default_rng(seed)`

Tạo bộ sinh số ngẫu nhiên. Giữ seed để tái lập quá trình lấy mẫu khi chạy lại cùng dữ liệu/thứ tự.

### `rng.choice(ham, size=ham_count, replace=False)`

Chọn `ham_count` vị trí trong `ham` **không hoàn lại**: một dòng Ham không bị chọn trùng hai lần trong cùng lượt.

### `np.concatenate([spam, sampled])`

Nối **toàn bộ vị trí Spam** với các vị trí Ham được chọn.

### `rng.shuffle(selected)`

Trộn thứ tự các vị trí đã chọn; chỉ đổi thứ tự, **không đổi thành phần mẫu hay nhãn**. Với mô hình tuyến tính lý tưởng, chỉ đổi thứ tự không thay mục tiêu tối ưu, nhưng thứ tự có thể ảnh hưởng một số chi tiết thực thi; điều quan trọng ở đây là dữ liệu đầu vào được xáo thay vì nhóm sẵn theo lớp.

### `return selected`

Hàm trả **mảng chỉ số**, không trả DataFrame, không trả `(X,y)` và không fit mô hình. Caller dùng `X[selected]`/`y[selected]` để lấy mẫu thật.

## 7.4. Dữ liệu thật của dự án trước/sau S2

| | Trước S2 | Sau S2 |
|---|---:|---:|
| Ham | 26.384 | **8.750** |
| Spam | 4.375 | **4.375** |
| Tổng | 30.759 | **13.125** |
| Spam / tổng | 14,22% | **33,33%** |

Vì `8750 = 2 × 4375`, S2 tạo tỷ lệ **2 Ham : 1 Spam**. Nó **không cân bằng 1:1** và **không sinh Spam mới**.

Có `26.384−8.750 = 17.634` dòng Ham không được dùng trong lần fit cuối của S2; chúng **không bị xóa khỏi CSV**. Test và các fold validation vẫn giữ nguyên dữ liệu gốc.

## 7.5. Một lưu ý liên quan tới TF-IDF

Khi S2 fit trên tập đã giảm Ham, `TfidfVectorizer.fit_transform()` học `vocabulary_`, `idf_` từ **chính tập con đó**. Vì `min_df=2`, vocabulary S2 có thể khác B4/S1/S3; không được lấy vector toàn Train đã fit trước rồi mới under-sampling nếu muốn mô phỏng đúng Pipeline CV đang dùng.

---

# 8. `mean_cv_scores()`: tính điểm trên từng validation rồi lấy trung bình

## 8.1. Code gốc

```python
def mean_cv_scores(y, predictions, folds):
    result = {}
    for name, metric in [
        ("Precision", precision_score),
        ("Recall", recall_score),
        ("F1", f1_score)
    ]:
        fold_scores = []
        for _, valid in folds:
            fold_scores.append(
                metric(y[valid], predictions[valid], zero_division=0)
            )
        result[f"CV {name} (spam)"] = float(np.mean(fold_scores))
    return result
```

## 8.2. Đầu vào

- `y`: Nhãn thật của **toàn Train**.
- `predictions`: Mảng dự đoán **ngoài fold (out-of-fold/OOF)** có độ dài bằng `y`.
- `folds`: Mảng các cặp `(train, valid)` cho 5 lượt.

**OOF:** Với mỗi vị trí `i` trong Train, dự đoán `predictions[i]` đến từ mô hình **không học từ dòng i**, mà dùng dòng i làm validation ở một lượt CV.

## 8.3. Cách thực hiện

1. Lặp qua ba hàm metric: Precision, Recall, F1.
2. Với mỗi metric, lặp qua `folds` và lấy `valid` (bỏ qua `_` là chỉ số Train của fold không cần dùng ở đây).
3. Gọi `metric(y[valid], predictions[valid], zero_division=0)`.
4. Gom 5 điểm vào `fold_scores`.
5. Lấy `np.mean(fold_scores)`.
6. Trả dictionary gồm `CV Precision (spam)`, `CV Recall (spam)`, `CV F1 (spam)`.

`zero_division=0` giải quyết trường hợp mẫu validation không có dự đoán dương hoặc mẫu số không hợp lệ: trả 0 theo lựa chọn trong code, tránh lỗi phép chia 0.

## 8.4. Ví dụ tính tay với 2 fold để dễ nhìn

```text
Fold 1: y_true = [1,0,1,0]   y_pred = [1,1,0,0]
       TP=1, FP=1, FN=1 → P=0.5, R=0.5, F1=0.5

Fold 2: y_true = [1,0,1,0]   y_pred = [1,0,1,0]
       TP=2, FP=0, FN=0 → P=1, R=1, F1=1
```

Kết quả trung bình theo fold:

`CV Precision = (0.5+1)/2 = 0.75`

`CV Recall = (0.5+1)/2 = 0.75`

`CV F1 = (0.5+1)/2 = 0.75`

**Chú ý:** Điểm F1 khi ghép toàn bộ dự đoán OOF rồi tính một lần **không bắt buộc bằng** trung bình F1 của từng fold. Code đang lấy trung bình **không trọng số theo fold**. Với dự án, kích thước các validation fold không bằng nhau, nên điểm trung bình này khác về định nghĩa so với F1 gộp toàn bộ Train OOF.

## 8.5. Vì sao S2/S3 dùng hàm này?

- S2 tự viết CV do mỗi lượt phải giảm Ham **chỉ trên fold Train**, rồi vẫn chấm validation nguyên bản.
- S3 tự tạo score OOF, thay threshold nhiều lần, nên cần chấm F1 trên các validation fold.
- S1 dùng `GridSearchCV` trong `experiments.py` nên scikit-learn tự tính các điểm CV; không gọi `mean_cv_scores` ở nhánh S1.

---

# 9. S2 — `undersampled_search()`: tìm C và n-gram sau khi giảm Ham

Đây là **hàm dài nhất của nhánh S2**. Bạn cần hiểu nó như **GridSearchCV viết thủ công** để đặt bước lấy mẫu vào đúng thời điểm **sau khi chia fold, trước khi fit**.

## 9.1. Code gốc (đúng bản nhóm)

```python
def undersampled_search(pipe, grid, X, y, folds, seed=42):
    """S2: tìm tham số bằng CV với sampling chỉ trên fold train."""
    rows = []
    for params in ParameterGrid(grid):
        predictions = np.zeros(len(y), dtype=int)
        for train, valid in folds:
            selected = sample_training_indices(y, train, seed)
            assert not set(selected) & set(valid)
            model = clone(pipe).set_params(**params).fit(X[selected], y[selected])
            predictions[valid] = model.predict(X[valid])
        rows.append((mean_cv_scores(y, predictions, folds), params))
    scores, params = max(rows, key=lambda row: row[0]["CV F1 (spam)"])
    selected = sample_training_indices(y, np.arange(len(y)), seed)
    fitted = clone(pipe).set_params(**params).fit(X[selected], y[selected])
    return fitted, scores, params, len(selected)
```

## 9.2. Đầu vào từ Notebook 02

```python
pipe, grid, _ = make_pipe("TF-IDF", "SVM", seed=SEED, cache=cache, n_jobs=N_JOBS)
us_model, us_metrics, us_params, us_count = undersampled_search(
    pipe, grid, X_tr_text, y_train, CV_FOLDS, SEED
)
```

- `pipe`: **Pipeline chưa fit**, gồm `("vec", TfidfVectorizer(...))` và `("clf", LinearSVC(...))`.
- `grid`: n-gram `(1,1)`/`(1,2)`, C `[0.1,0.3,1.0,3.0]`, class_weight cố định `None`.
- `X`: mảng **chuỗi token** chỉ có các dòng Train.
- `y`: nhãn 0/1 tương ứng.
- `folds`: 5 cặp chỉ số **trong Train**, không phải index toàn CSV.
- `seed`: cố định thứ tự lựa chọn Ham.

**S2 không sử dụng `GridSearchCV.fit()`**, vì nhóm tự viết vòng lặp đảm bảo undersampling nằm **bên trong fold Train**. `ParameterGrid` chỉ tạo danh sách tham số, không huấn luyện gì cả.

## 9.3. Dòng 1 — `rows = []`

Tạo danh sách rỗng để gom:

```python
[(scores_cau_hinh_1, params_1), (scores_cau_hinh_2, params_2), ...]
```

Mỗi phần tử tương ứng **một tổ hợp siêu tham số** và điểm CV của nó.

## 9.4. `for params in ParameterGrid(grid):` diễn ra gì?

Ở dự án, grid S2 tạo 8 cấu hình:

| Thứ tự | `vec__ngram_range` | `clf__C` | `clf__class_weight` |
|---:|---|---:|---|
| 1 | `(1,1)` | 0.1 | None |
| 2 | `(1,2)` | 0.1 | None |
| 3 | `(1,1)` | 0.3 | None |
| 4 | `(1,2)` | 0.3 | None |
| 5 | `(1,1)` | 1.0 | None |
| 6 | `(1,2)` | 1.0 | None |
| 7 | `(1,1)` | 3.0 | None |
| 8 | `(1,2)` | 3.0 | None |

Trình tự trên là ví dụ theo `ParameterGrid` trên Python/scikit-learn được kiểm tra trong môi trường hiện tại; ý chính phải nhớ là **đủ 8 tổ hợp**, không nên dựa vào thứ tự cho logic thuật toán.

Mỗi phần tử `params` là một dictionary, chẳng hạn:

```python
{
    'clf__C': 0.1,
    'clf__class_weight': None,
    'vec__ngram_range': (1, 2)
}
```

Dấu `__` có nghĩa: `vec__ngram_range` là tham số của bước `vec`; `clf__C` là tham số của bước `clf` trong Pipeline. Tham số tạo từ `make_pipe()` của bước 04.

## 9.5. `predictions = np.zeros(len(y), dtype=int)`

Tạo mảng 30.759 số 0 (nếu chạy trên Train thật), để **gắn nhãn dự đoán vào đúng vị trí của từng mẫu khi mẫu đó làm validation**.

Ban đầu:

```text
predictions = [0, 0, 0, 0, 0, ...]  # chỉ là giá trị khởi tạo
```

Qua 5 fold, mỗi vị trí phải được ghi bằng dự đoán của đúng lượt validation. Không được hiểu số 0 ban đầu là mô hình đã dự đoán Ham.

## 9.6. `for train, valid in folds:`

Lặp qua năm bộ chỉ số:

- `train`: vị trí được phép dùng để fit trong lượt hiện tại.
- `valid`: vị trí được phép **đánh giá**, tuyệt đối không được sử dụng để fit.

Giả sử trong một ví dụ minh họa có 10 câu:

```python
train = np.array([0,1,2,3,4,5,6,7])
valid = np.array([8,9])
```

Các chỉ số này **tính từ 0 trong `X_tr_text`**, không phải trong toàn bộ CSV gốc.

## 9.7. `selected = sample_training_indices(y, train, seed)`

**Chỉ giảm Ham từ phần Train của fold.** Ví dụ fold Train có 800 Ham + 100 Spam → chỉ chọn 200 Ham + 100 Spam để fit; vẫn dự đoán trên fold Validation nguyên bản.

Nếu giảm luôn cả validation, điểm CV sẽ đánh giá một phân bố nhân tạo với tỷ lệ Ham/Spam khác thực tế và gây so sánh sai lệch.

## 9.8. `assert not set(selected) & set(valid)`

Giải thích toán tử:

- `set(selected)`: tập vị trí đã chọn huấn luyện.
- `set(valid)`: tập vị trí validation.
- `&`: giao của hai tập.
- `not ...`: giao phải rỗng.
- `assert`: ném lỗi nếu điều kiện không đúng.

Nghĩa là **không được có một dòng nào được fit đồng thời nằm trong validation**. Đây là rào chắn kiểm tra data leakage.

## 9.9. `clone(pipe).set_params(**params).fit(X[selected],y[selected])`

Cần đọc **từ trái sang phải**:

**(A) `clone(pipe)`**: tạo một Pipeline mới cùng cấu hình, **chưa mang vocabulary/IDF/trọng số SVM của lần fit trước**.

**(B) `.set_params(**params)`**: thay đổi `vec__ngram_range`, `clf__C`, `clf__class_weight` theo cấu hình hiện tại. `**params` trải các cặp khóa-giá trị của dict thành keyword arguments.

**(C) `.fit(X[selected], y[selected])`**: lúc này mới học dữ liệu.

Bên trong Pipeline:

```text
X[selected] = chuỗi token của 2 Ham : 1 Spam trong fold Train
    ↓
vec.fit_transform(X[selected])
    → TF-IDF học vocabulary + IDF từ tập con đã chọn
    ↓
clf.fit(TF-IDF_vector_matrix, y[selected])
    → LinearSVC học coef_, intercept_ từ đúng tập con đó
```

**Không dùng** `X[train]` (chưa giảm Ham), **không dùng** `X[valid]` để fit, và **không fit vectorizer trên toàn CSV trước CV**.

## 9.10. `predictions[valid] = model.predict(X[valid])`

Pipeline nhận **chuỗi token validation**, tự gọi `vec.transform` bằng vocabulary/IDF **đã fit ở fold Train**, rồi `clf.predict` mặc định score > 0 → Spam; ngược lại Ham.

Nhãn dự đoán được gắn lại **đúng các vị trí validation** trong `predictions`.

Sau năm lượt, ta có mảng dự đoán OOF cho toàn Train của **một cấu hình params**.

## 9.11. `rows.append((mean_cv_scores(...), params))`

Gọi `mean_cv_scores` tính ba chỉ số trên từng fold validation rồi lấy trung bình. Sau đó lưu điểm và cấu hình.

Ví dụ giả định:

```python
rows = [
    ({'CV F1 (spam)': 0.70, ...}, {'clf__C':0.1,'vec__ngram_range':(1,1), ...}),
    ({'CV F1 (spam)': 0.75, ...}, {'clf__C':0.1,'vec__ngram_range':(1,2), ...}),
    # ... các cấu hình còn lại
]
```

Đây là ví dụ cấu trúc, không phải số CV thật.

## 9.12. Chọn tham số bằng F1 Spam

```python
scores, params = max(
    rows,
    key=lambda row: row[0]["CV F1 (spam)"]
)
```

- `max` tìm phần tử có giá trị lớn nhất theo quy tắc key.
- `row[0]` là dictionary các điểm CV.
- `row[0]['CV F1 (spam)']` là tiêu chí chọn.
- Kết quả tách ra `scores` và `params` của cấu hình thắng.

Nếu nhiều cấu hình có F1 bằng nhau, Python `max` giữ phần tử đầu tiên đạt cực đại trong vòng lặp. Code không có tiebreak riêng cho S2.

**Kết quả thật của S2:** `C=0.1`, `ngram_range=(1,2)`, `class_weight=None`, CV F1 = **0.7629**.

## 9.13. Fit mô hình S2 cuối cùng

```python
selected = sample_training_indices(y, np.arange(len(y)), seed)
fitted = clone(pipe).set_params(**params).fit(X[selected], y[selected])
```

Đến đây tham số đã chốt, không cần lặp 5 fold nữa. Nhưng S2 vẫn **chỉ fit trên tập toàn Train đã giảm mẫu** (13.125 câu), chứ **không fit trên đủ 30.759**.

Tại sao không đưa 30.759 câu vào? Vì nếu fit toàn bộ Train thì đã bỏ chiến lược undersampling, mô hình cuối không còn đại diện đúng S2.

## 9.14. Hàm trả về

```python
return fitted, scores, params, len(selected)
```

| Biến | Kiểu / nội dung | Dùng tiếp ở đâu? |
|---|---|---|
| `fitted` | Pipeline TF-IDF+SVM đã fit trên 13.125 câu được chọn | `_model` của record S2, sau dùng dự đoán Test |
| `scores` | Dict điểm CV Precision/Recall/F1 Spam | So sánh ứng viên |
| `params` | Dict tham số thắng | Báo cáo và lưu cấu hình |
| `len(selected)` | `13125` | Kiểm chứng số mẫu được fit; bảng trong Notebook |

**Đánh giá đúng:** `mean_cv_scores` tính trên validation **không giảm mẫu**; còn mô hình cuối được fit trên Train **đã giảm mẫu**. Cách thiết kế này cho phép chấm chất lượng trên phân bố gần thực tế thay vì đánh giá trên dữ liệu đã cân đối nhân tạo.

---

# 10. S3 — `tune_svm_threshold()`: giữ TF-IDF+SVM, tìm ngưỡng quyết định

S3 là chiến lược được Notebook chọn. Bạn cần hiểu được **score, ngưỡng, OOF và lý do khác biệt với huấn luyện lại mô hình**.

## 10.1. Ý nghĩa score của SVM

SVM dự đoán dựa trên điểm:

\[
s(x)=w^T x+b.
\]

Với nhãn `[0,1]`:

- Score `> 0` → Spam mặc định.
- Score `<= 0` → Ham mặc định.
- Score **không phải xác suất**. Ví dụ `score=-0.1776` **không có nghĩa** xác suất Spam âm 17,76%.

Ví dụ:

| Bình luận | Score | Ngưỡng 0 | Ngưỡng −0.2 |
|---|---:|---|---|
| A | -0.9 | Ham | Ham |
| B | -0.1 | Ham | **Spam** |
| C | +0.3 | Spam | Spam |

Hạ ngưỡng từ 0 xuống −0.2 khiến bình luận B chuyển từ Ham sang Spam **mặc dù SVM không đổi trọng số `coef_` hoặc `intercept_`**.

## 10.2. Code gốc (đúng bản nhóm)

```python
def tune_svm_threshold(pipe, X, y, folds):
    scores = np.zeros(len(y))
    covered = np.zeros(len(y), dtype=int)
    for train, valid in folds:
        model = clone(pipe).fit(X[train], y[train])
        scores[valid] = model.decision_function(X[valid])
        covered[valid] += 1
    assert (covered == 1).all()
    thresholds = [0.0]
    thresholds.extend(np.quantile(scores, np.linspace(0.01, 0.99, 199)))
    thresholds.extend([np.nextafter(scores.min(), -np.inf), scores.max()])
    thresholds = np.unique(thresholds)
    rows = []
    for threshold in thresholds:
        predictions = (scores > threshold).astype(int)
        metrics = mean_cv_scores(y, predictions, folds)
        rows.append((metrics, float(threshold)))
    metrics, threshold = max(
        rows,
        key=lambda row: (row[0]["CV F1 (spam)"], -abs(row[1]))
    )
    return clone(pipe).fit(X, y), metrics, threshold
```

## 10.3. Đầu vào quan trọng: `pipe` là `focus['_model']`

Notebook gọi:

```python
threshold_model, threshold_metrics, threshold = tune_svm_threshold(
    focus["_model"], X_tr_text, y_train, CV_FOLDS
)
```

`focus['_model']` là **Pipeline B4 đã fit trên toàn Train từ cell 13**, có siêu tham số B4 tốt nhất: `ngram_range=(1,2)` và `C=0.3`, `class_weight=None`.

**Nhưng chú ý kỹ:** Hàm **không trực tiếp tận dụng các hệ số đã fit của `focus['_model']` để tạo score OOF**. Nó dùng `clone(pipe)` để tạo bản **chưa fit** nhưng giữ cấu hình, sau đó fit mới ở từng fold và fit lại trên toàn Train. Vì thế, cách nói chính xác là **S3 giữ siêu tham số B4**, chứ không phải hoàn toàn không chạy `fit()`.

## 10.4. Tạo mảng score và `covered`

```python
scores = np.zeros(len(y))
covered = np.zeros(len(y), dtype=int)
```

- `scores[i]` chứa **decision score OOF** của bình luận Train thứ `i` sau khi lượt chứa nó trong validation chạy xong.
- `covered[i]` đếm số lượt mà vị trí `i` đã làm validation; phải là `1` với mọi i.

Tại sao cần `covered`? Để chắc chắn rằng không dòng nào bị bỏ quên hoặc được chấm nhiều lần bởi lỗi tạo fold.

## 10.5. Tạo score OOF bằng 5 lượt fit

```python
for train, valid in folds:
    model = clone(pipe).fit(X[train], y[train])
    scores[valid] = model.decision_function(X[valid])
    covered[valid] += 1
```

Ví dụ một lượt:

```text
Lượt CV 1:
4 fold Train → clone(TF-IDF+SVM cấu hình B4).fit()
              → học vocabulary/IDF trên 4 fold Train
              → học coef_/intercept_ SVM trên 4 fold Train
Fold Validation 1 → model.decision_function(X[valid])
                  → score liên tục, không phải nhãn
                  → ghi vào scores[valid]
```

Sau 5 lượt, mọi dòng Train nhận đúng một score từ mô hình **không fit trên chính dòng đó**. Score này gọi là **out-of-fold decision score (OOF score)**.

### Khác với dự đoán trên chính Train đã fit

Nếu fit mô hình trên toàn Train rồi dùng nó tính score cho cũng Train và lựa chọn ngưỡng, mô hình đã nhìn thấy nhãn và đặc trưng của từng dòng, dẫn tới rủi ro chọn ngưỡng quá đẹp trên dữ liệu đã học.

OOF hạn chế điều đó: score dòng i sinh ra khi i chỉ là validation. Tuy nhiên, việc **tìm ngưỡng tối đa F1 trên chính các score OOF** vẫn khiến điểm F1 báo cáo cho S3 có khả năng lạc quan; xem mục 18.

## 10.6. `assert (covered == 1).all()`

- `covered == 1` tạo mảng True/False.
- `.all()` True khi tất cả phần tử đều bằng 1.
- `assert` dừng nếu có bất kỳ dòng nào không được validation đúng một lượt.

Ví dụ `covered=[1,1,1,0]` → False, tức có một dòng chưa được nhận score.

## 10.7. Tạo danh sách ngưỡng cần thử

```python
thresholds = [0.0]
thresholds.extend(
    np.quantile(scores, np.linspace(0.01, 0.99, 199))
)
thresholds.extend([
    np.nextafter(scores.min(), -np.inf),
    scores.max()
])
thresholds = np.unique(thresholds)
```

**(A) `thresholds=[0.0]`:** Luôn kiểm tra ngưỡng mặc định của SVM.

**(B) `np.linspace(0.01,0.99,199)`:** Sinh 199 mức phân vị từ 1% đến 99%, gồm cả hai đầu.

**(C) `np.quantile(scores, ...)`:** Lấy score tương ứng mỗi phân vị. Ví dụ ngưỡng phân vị 0.50 là trung vị score, phân vị 0.90 là score mà khoảng 90% giá trị nằm phía dưới hoặc bằng.

**(D) `np.nextafter(scores.min(), -np.inf)`:** Lấy một số thực biểu diễn được nằm **ngay phía dưới** score nhỏ nhất → với điều kiện `score > threshold`, mọi mẫu OOF đều thành Spam.

**(E) `scores.max()`:** Ngưỡng bằng score lớn nhất → không score nào `>` ngưỡng, mọi mẫu OOF thành Ham.

**(F) `np.unique(thresholds)`:** Bỏ ngưỡng trùng lặp, sắp xếp lại; tối đa **202 ngưỡng khác nhau** (199 phân vị + 0 + 2 biên), có thể ít hơn nếu trùng.

**Tại sao dựa vào phân vị chứ không chỉ thử −1, 0, 1?** Phân vị trải các mức ngưỡng theo **phân bố score thực tế**, giúp khảo sát những điểm cắt phù hợp với score hiện có.

## 10.8. Chuyển score thành nhãn theo từng ngưỡng

```python
for threshold in thresholds:
    predictions = (scores > threshold).astype(int)
    metrics = mean_cv_scores(y, predictions, folds)
    rows.append((metrics, float(threshold)))
```

- `scores > threshold`: mảng boolean True/False.
- `.astype(int)`: True→1 (Spam), False→0 (Ham).
- `mean_cv_scores`: đánh giá ba metric trên các phần validation OOF (không dùng Test).
- `rows`: lưu một cặp `(điểm_cv, ngưỡng)` cho mỗi ngưỡng.

Không gọi thêm `fit()` trong vòng `for threshold`: **một bộ score OOF có thể được dùng để thử rất nhiều ngưỡng**. Đây là điểm làm S3 khác với việc grid-search siêu tham số classifier.

## 10.9. Ví dụ tính tay: ngưỡng thay đổi FP/FN và F1

Giả sử 6 bình luận có điểm OOF và nhãn thật:

| i | y thật | score SVM |
|---:|---|---:|
| 0 | Ham | −1.0 |
| 1 | Spam | −0.3 |
| 2 | Ham | −0.1 |
| 3 | Spam | 0.2 |
| 4 | Spam | 0.5 |
| 5 | Ham | 0.9 |

**Ngưỡng 0.0:** dự đoán Spam ở i=3,4,5.

- TP=2 (i=3,4), FP=1 (i=5), FN=1 (i=1).
- Precision=2/3≈0.667; Recall=2/3≈0.667; F1≈0.667.

**Ngưỡng −0.4:** dự đoán Spam ở i=1,2,3,4,5.

- TP=3 (i=1,3,4), FP=2 (i=2,5), FN=0.
- Precision=3/5=0.6; Recall=3/3=1.0; F1=0.75.

**Ngưỡng −0.2:** dự đoán Spam ở i=2,3,4,5.

- TP=2, FP=2, FN=1.
- Precision=2/4=0.5; Recall=2/3≈0.667; F1≈0.571.

Qua ví dụ ta thấy: hạ ngưỡng **không bảo đảm** F1 liên tục tăng. Có ngưỡng sẽ bắt thêm Spam thật; cũng có ngưỡng làm tăng báo động nhầm. Vì vậy code phải đánh giá nhiều ngưỡng và chọn bằng F1.

Lưu ý: Đây là ví dụ F1 **trên toàn 6 câu để minh họa**; code thật dùng `mean_cv_scores()` → **trung bình F1 theo từng fold**.

## 10.10. Chọn ngưỡng cao F1 nhất, nếu hòa thì gần 0 nhất

```python
metrics, threshold = max(
    rows,
    key=lambda row: (
        row[0]["CV F1 (spam)"],
        -abs(row[1])
    )
)
```

Ý nghĩa tuple dùng trong `key`:

1. So điểm `CV F1 (spam)` trước: lớn nhất thắng.
2. Nếu điểm ngang nhau, so `-abs(threshold)`; threshold có trị tuyệt đối nhỏ hơn (gần 0 hơn) thắng.

Vì `max()` so tuple theo thứ tự từ trái sang phải, đây là cách tiebreak hợp lệ. Ví dụ F1 đều 0.78, ngưỡng −0.1 thắng −0.8 do `−abs(−0.1)=−0.1 > −0.8`.

## 10.11. Fit mô hình cuối và trả về

```python
return clone(pipe).fit(X, y), metrics, threshold
```

- Tạo Pipeline chưa fit với **siêu tham số B4**.
- Fit lại trên **toàn bộ X/y Train** (30.759 câu).
- Trả Pipeline đã fit, điểm CV và ngưỡng đã chọn.

Không có `threshold` bên trong `LinearSVC.coef_` hoặc `intercept_`. Ngưỡng được **lưu riêng trong record `_threshold`**, rồi `evaluation.predict_record()` dùng `score > _threshold` để tạo nhãn.

## 10.12. Kết quả S3 thật của Notebook

- Giữ n-gram `(1,2)`, C=0.3, `class_weight=None` theo B4.
- Threshold được chọn: `-0.17762792522140153` (làm tròn **−0.1776**).
- CV Precision Spam: **0.8490**.
- CV Recall Spam: **0.7353**.
- CV F1 Spam: **0.7876** (cao nhất trong các ứng viên đã khảo sát).

**Tại sao ngưỡng âm lại có thể hợp lý?** Với B4, score âm nhẹ mặc định bị coi Ham. Hạ ngưỡng xuống −0.1776 cho phép một phần các câu score trong khoảng `(−0.1776, 0]` được dự đoán Spam; điều đó cải thiện Recall và có thể cải thiện F1 tùy số FP phát sinh.

---

# 11. Đi sâu các hàm thư viện được dùng ở bước 05

Ở bước trước, bạn đã học `CountVectorizer`, `TfidfVectorizer`, `MultinomialNB` và `LinearSVC`. Bước này cần hiểu thêm **cách những hàm thư viện dưới đây tham gia vào chiến lược cân bằng**.

## 11.1. `sklearn.base.clone(estimator)`

**Đầu vào:** một estimator hoặc `Pipeline`, đã fit hoặc chưa fit.

**Đầu ra:** một estimator **mới chưa học**, giữ cùng cấu hình hyperparameter.

**Bên trong về ý tưởng:** scikit-learn lấy tham số khởi tạo qua `get_params`, tạo đối tượng mới có cùng tham số; trạng thái học như `vocabulary_`, `idf_`, `coef_`, `intercept_` không được sao chép.

Ví dụ:

```python
from sklearn.base import clone

# pipe_fitted đã fit trước đó
new_pipe = clone(pipe_fitted)
print(new_pipe.get_params()['clf__C'])   # C vẫn giống
# new_pipe chưa có SVM đã học (chưa fit)
```

**Trong S2:** `clone` tạo model mới cho mỗi cấu hình + mỗi fold, và cho fit cuối.

**Trong S3:** `clone` tạo model mới với siêu tham số B4 để fit từng fold và fit lại trên toàn Train.

**Không được nhầm:** `clone()` **không sao chép một mô hình đã học để dự đoán ngay**. Nếu muốn nạp lại mô hình đã fit, ta dùng `joblib.load()` trên bundle đã lưu, không dùng clone để thay thế.

## 11.2. `Pipeline.set_params(**params)`

**Đầu vào:** tên tham số có prefix của bước (`vec__ngram_range`, `clf__C`, `clf__class_weight`).

**Đầu ra:** chính Pipeline/estimator đã được cập nhật cấu hình, **chưa fit** nếu gọi trên clone chưa fit.

Ví dụ:

```python
pipe2 = clone(pipe)
pipe2.set_params(vec__ngram_range=(1, 2), clf__C=0.3)
pipe2.fit(X_fold_train_text, y_fold_train)
```

**Tại sao dùng `**params`:** Vì `ParameterGrid` trả dictionary. Dấu `**` bung các khóa dictionary thành keyword arguments.

## 11.3. `ParameterGrid(grid)`

**Đầu vào:** dictionary các danh sách giá trị. Ví dụ:

```python
{
    'vec__ngram_range': [(1,1),(1,2)],
    'clf__C': [0.1,0.3,1.0,3.0],
    'clf__class_weight': [None]
}
```

**Đầu ra:** đối tượng iterable liệt kê đủ `2 × 4 × 1 = 8` tổ hợp.

**Không thực hiện:** Không chia tập, không tính điểm, không gọi fit. S2 tự lặp qua 8 tổ hợp bằng vòng `for` và tự gọi `clone(...).fit(...)`.

## 11.4. `np.asarray`, indexing, `np.arange`

`np.asarray(y)` → biến list/array thành ndarray để có thể dùng boolean indexing `y[indices]==1`.

`np.arange(len(y))` → tạo `[0,1,...,len(y)-1]` là vị trí tất cả dòng Train.

`X[selected]`, `y[selected]` → chọn đồng thời **cùng một tập vị trí** từ văn bản và nhãn. Nếu lệch thứ tự, mô hình có thể học sai nhãn.

`np.concatenate([spam,sampled])` → nối các mảng chỉ số.

## 11.5. `np.random.default_rng`, `rng.choice`, `rng.shuffle`

`default_rng(42)` tạo bộ sinh ngẫu nhiên tái lập được khi giữ cùng đầu vào và seed.

`rng.choice(ham, size=k, replace=False)` chọn k vị trí Ham **không lặp**.

`rng.shuffle(selected)` trộn thứ tự trong cùng mảng. Khác với `np.random.choice` dùng trạng thái random toàn cục: code dự án chủ động tạo một `Generator` cục bộ.

## 11.6. `precision_score`, `recall_score`, `f1_score`

Các hàm nhận `y_true`, `y_pred`; vì nhãn dương theo mặc định là `1`, chúng đo lớp **Spam**.

```python
precision_score(y_valid, pred_valid, zero_division=0)
recall_score(y_valid, pred_valid, zero_division=0)
f1_score(y_valid, pred_valid, zero_division=0)
```

Chúng **không fit mô hình**. Chúng chỉ so nhãn thật/dự đoán.

## 11.7. `LinearSVC.decision_function(X)`

**Đầu vào:** ma trận vector đã được đưa qua TfidfVectorizer (nếu gọi qua Pipeline, bạn truyền **chuỗi token** vào Pipeline).

**Đầu ra:** score có dấu, với lớp `[0,1]` score dương nghiêng về Spam; **không phải xác suất**.

Bên trong, `LinearClassifierMixin` sử dụng `X @ coef_.T + intercept_` để tính điểm tuyến tính. S3 ghi score OOF để có thể thử các mốc quyết định khác nhau **mà không cần huấn luyện lại cho từng ngưỡng**.

## 11.8. `np.linspace`, `np.quantile`, `np.unique`, `np.nextafter`

| Hàm | Ví dụ | Công dụng trong S3 |
|---|---|---|
| `np.linspace(a,b,n)` | 199 số trải từ 0.01 đến 0.99 | Tạo các mức phân vị cần khảo sát |
| `np.quantile(scores,q)` | q=0.5 → trung vị score | Chuyển mức phân vị thành **ngưỡng score thật** |
| `np.unique(values)` | Bỏ giá trị trùng, thường sắp tăng dần | Loại các ngưỡng lặp |
| `np.nextafter(x,-np.inf)` | Số thực biểu diễn được kế tiếp nhỏ hơn x | Thử ngưỡng ngay dưới score nhỏ nhất |
| `np.zeros(n)` | n phần tử 0 | Tạo kho chứa score/pred/covered |

## 11.9. Những thứ không phải thư viện mới

- `max(rows, key=...)` là **hàm Python built-in**, chọn phần tử tốt nhất theo quy tắc `key`.
- `lambda row: ...` là **hàm ẩn danh** để lấy giá trị dùng so sánh.
- `.append()` và `.extend()` là phương thức **list**; `.append(x)` thêm **một phần tử**, `.extend(iterable)` thêm **nhiều phần tử**.
- `.astype(int)` là phép đổi kiểu NumPy từ boolean sang 0/1.
- `assert` là kiểm tra điều kiện chạy đúng, không phải một bước xử lý dữ liệu.

---

# 12. Notebook 02 — giải thích cell 14, 15 và 16 theo luồng thực thi

## 12.1. Cell 14 — phần định hướng thí nghiệm

Cell 14 là Markdown giải thích thiết kế. Sau khi B4 dẫn đầu, Notebook giữ **cùng TF-IDF+SVM** và so:

- Không điều chỉnh (B4) làm đối chứng.
- S1 tăng trọng số Spam bằng balanced.
- S2 giảm Ham chỉ trong fold Train theo tỷ lệ 2:1.
- S3 tìm ngưỡng mới từ score OOF, với siêu tham số B4.

Đây không phải cam kết mọi chiến lược chắc chắn cải thiện: kết luận chỉ có giá trị trong **tập ứng viên đã thực nghiệm**.

## 12.2. Cell 15 — thống kê và vẽ hình trước khi cân bằng

Các dòng liên quan trực tiếp đến hiểu dữ liệu:

```python
sampled_indices = sample_training_indices(
    y_train, np.arange(len(y_train)), seed=SEED, ratio=2
)
before_counts = pd.Series(
    np.where(y_train == 1, "spam", "ham")
).value_counts().reindex(["ham", "spam"])
after_counts = pd.Series(
    np.where(y_train[sampled_indices] == 1, "spam", "ham")
).value_counts().reindex(["ham", "spam"])
weights = len(y_train) / (2 * before_counts)
```

**Giải thích:**

- `sampled_indices` mô phỏng việc giảm mẫu trên **toàn Train** để tính số liệu và vẽ biểu đồ.
- `np.where(condition, "spam", "ham")` chuyển nhãn số sang chuỗi dùng hiển thị.
- `value_counts()` đếm số dòng từng lớp.
- `.reindex(["ham","spam"])` cố định thứ tự hai lớp trên bảng/hình.
- `weights` tính trọng số `n/(2*n_c)` **trên toàn Train để minh họa**.
- `sampling_table` so trước/sau undersampling; `weight_table` so trọng số mặc định 1.0 với balanced; các lệnh `plot.bar`, `ax.bar_label`, `fig.savefig` chỉ để vẽ và lưu hình.

**Rất quan trọng:** Cell 15 **không huấn luyện SVM** và **không thay dữ liệu gốc**. Trong CV, việc tính trọng số/giảm mẫu thực tế được thực hiện bên trong mỗi fold Train ở cell 16.

## 12.3. Cell 16 — chuẩn bị cùng dữ liệu Train

```python
X_tr_text = TEXT[MAIN_PRE].iloc[train_idx].to_numpy()
```

`TEXT[MAIN_PRE]` là Series token tương ứng nhánh **giữ stopwords** đã được chọn ở cell 13.

`.iloc[train_idx]` lấy những dòng thuộc Train theo **index của toàn CSV**.

`.to_numpy()` tạo ndarray để truy cập như `X_tr_text[train]` hoặc `X_tr_text[selected]` với các vị trí **bên trong Train**.

`X_tr_text` chưa phải ma trận TF-IDF; Pipeline S1/S2/S3 tự vector hóa.

## 12.4. Cell 16 — S1

```python
balanced = fit_balanced_svm(
    MAIN_PRE, X_tr_text, y_train, CV_FOLDS,
    seed=SEED, cache=cache, n_jobs=N_JOBS
)
balanced["Mã"] = "S1"
```

Hàm trả **record hoàn chỉnh** từ `fit_experiment`, sau đó Notebook gắn mã S1.

## 12.5. Cell 16 — S2

```python
pipe, grid, _ = make_pipe(
    "TF-IDF", "SVM", seed=SEED,
    cache=cache, n_jobs=N_JOBS
)
start = time.time()
us_model, us_metrics, us_params, us_count = undersampled_search(
    pipe, grid, X_tr_text, y_train, CV_FOLDS, SEED
)
under = {
    "Mã": "S2", "Tiền xử lý": MAIN_PRE,
    "Đặc trưng": "TF-IDF", "Thuật toán": "SVM",
    "Chiến lược": "Under-sampling 2:1",
    **us_metrics,
    "Tham số": us_params,
    "Thời gian (s)": time.time() - start,
    "_model": us_model, "_threshold": 0.0
}
assert us_count == len(sampled_indices)
```

- `make_pipe` tạo Pipeline và grid **chưa fit**.
- `undersampled_search` tự chạy 8×5 fold rồi fit mô hình cuối trên tập đã giảm.
- `us_model` = mô hình đã fit, `us_metrics` = CV scores, `us_params` = C/ngram thắng, `us_count` = số mẫu fit.
- `**us_metrics` trải các khóa CV Precision/Recall/F1 vào dictionary `under`.
- `time.time()` đo khoảng thời gian theo đồng hồ, khác với thời gian của S1 vì cách triển khai/lợi ích cache/song song khác nhau.
- `_threshold=0.0`: S2 vẫn dùng ngưỡng mặc định SVM.
- `assert us_count == len(sampled_indices)` đối chiếu số mẫu thực tế với con số khảo sát ở cell 15.

## 12.6. Cell 16 — S3

```python
start = time.time()
threshold_model, threshold_metrics, threshold = tune_svm_threshold(
    focus["_model"], X_tr_text, y_train, CV_FOLDS
)
tuned = {
    "Mã": "S3", "Tiền xử lý": MAIN_PRE,
    "Đặc trưng": "TF-IDF", "Thuật toán": "SVM",
    "Chiến lược": "Điều chỉnh ngưỡng",
    **threshold_metrics,
    "Tham số": {**focus["Tham số"], "threshold": threshold},
    "Thời gian (s)": time.time() - start,
    "_model": threshold_model,
    "_threshold": threshold
}
```

- `focus['_model']` chứa cấu hình B4 đã chọn, được clone và fit lại.
- `threshold_model` = Pipeline TF-IDF+SVM đã fit toàn Train.
- `threshold_metrics` = điểm CV khi áp ngưỡng tối ưu lên các score OOF của Train.
- `threshold` = mốc score được chọn (−0.1776...).
- `"Tham số"` gồm cả C/ngram của B4 và ngưỡng mới để báo cáo.
- `_threshold` được lưu riêng ngoài estimator để `evaluate()`/`predict_spam()` áp dụng đúng quyết định khi chấm Test hoặc dự đoán bình luận mới.

**Đừng nhầm** việc `{**focus['Tham số'], 'threshold': threshold}` là chỉnh tham số của LinearSVC: đây **chỉ là dictionary ghi nhận cấu hình**, ngưỡng không được tự truyền vào `LinearSVC.fit()`.

## 12.7. Cell 16 — gom các ứng viên và chọn best

```python
SUPPLEMENTARY = [balanced, under, tuned]
STRATEGIES = [focus, *SUPPLEMENTARY]
CANDIDATES = RESULTS + SUPPLEMENTARY
best = max(CANDIDATES, key=lambda r: r["CV F1 (spam)"])
```

- `RESULTS` có **B1–B4** trong nhánh stopwords đã chọn.
- `SUPPLEMENTARY` có **S1/S2/S3**.
- `STRATEGIES` chỉ có **B4 + S1/S2/S3** để lập bảng so chiến lược.
- `CANDIDATES` có **7 ứng viên**: B1, B2, B3, B4, S1, S2, S3. Không phải 8 ứng viên vì B4 xuất hiện trong `RESULTS`, không được thêm lại trong `SUPPLEMENTARY`.
- `best = max(...)` chọn **CV F1 Spam cao nhất trên 7 record**.

Sau đó cell tạo `strategy_cv` để hiển thị, lưu `imbalance_cv.csv` và in `SELECTED_NAME`, ví dụ: `S3 — TF-IDF + SVM — Điều chỉnh ngưỡng`.

**Nguyên tắc đúng:** Cell 16 chọn `best` khi chưa xem Test; cell 20 sau đó mới đo Test.

---

# 13. Các biến và record nối sang file tiếp theo

| Record / biến | Tác dụng |
|---|---|
| `focus` | B4 dùng làm mốc đối chứng; `_threshold=0` |
| `balanced` | S1, record từ `fit_experiment()` |
| `under` | S2, record Notebook tự đóng từ 4 giá trị trả về |
| `tuned` | S3, record Notebook tự đóng từ 3 giá trị trả về |
| `STRATEGIES` | B4 và 3 chiến lược → dùng tạo bảng/báo cáo so sánh |
| `CANDIDATES` | B1–B4 và S1–S3 → chọn `best` và chấm Test |
| `best` | Record có CV F1 Spam lớn nhất, **S3** trong output đã lưu |
| `record['_model']` | Pipeline đã fit, gồm vectorizer + LinearSVC |
| `record['_threshold']` | Mốc dùng đổi decision score thành Spam/Ham |
| `record['Tham số']` | Dict cấu hình để giải thích và lưu báo cáo |

Ở bước 06, `evaluation.py` sẽ nhận record của từng ứng viên:

```python
for record in CANDIDATES:
    evaluate(
        record,
        TEXT[record["Tiền xử lý"]].iloc[test_idx].to_numpy(),
        y_test
    )
```

Hàm `evaluate()` không fit lại mô hình: nó sử dụng `_model`, `_threshold` và Test đã giữ riêng để trả score, nhãn rồi tính các chỉ số.

---

# 14. Kết quả thật từ Notebook 02

## 14.1. So sánh bằng Cross-Validation trên Train — Cell 16

| Mã | Cách xử lý | Số mẫu fit cuối | Ngưỡng | CV Precision Spam | CV Recall Spam | **CV F1 Spam** |
|---|---|---:|---:|---:|---:|---:|
| **B4** | Không điều chỉnh | 30.759 | 0 | **0.9001** | 0.6806 | 0.7736 |
| **S1** | Balanced | 30.759 | 0 | 0.7233 | **0.7964** | 0.7580 |
| **S2** | Under-sampling 2:1 | **13.125** | 0 | 0.8038 | 0.7276 | 0.7629 |
| **S3** | Điều chỉnh ngưỡng | 30.759 | **−0.1776** | 0.8490 | 0.7353 | **0.7876** |

> Các điểm CV thực tế là **trung bình theo fold validation bên trong Train**. Không được gọi đây là điểm Test.

### 14.1.1. Hyperparameter thắng — cũng từ cell 16

| Mã | `vec__ngram_range` | `clf__C` | `clf__class_weight` | Ngưỡng ngoài model |
|---|---|---:|---|---:|
| B4 | `(1,2)` | 0.3 | None | 0 |
| S1 | `(1,2)` | 0.1 | balanced | 0 |
| S2 | `(1,2)` | 0.1 | None | 0 |
| S3 | `(1,2)` | 0.3 | None | −0.17762792522140153 |

**Giải thích khác biệt C:** S1 và S2 tối ưu lại C bằng CV sau thay đổi chiến lược; S3 giữ siêu tham số C/ngram của B4 và chỉ tìm threshold.

## 14.2. Đánh giá Test sau khi đã chọn `best` — Cell 22

| Mã | Accuracy | Precision Spam | Recall Spam | F1 Spam | AP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|
| B4 | 0.9419 | **0.8818** | 0.6278 | 0.7334 | 0.8117 | **80** | 354 |
| S1 | 0.9322 | 0.7139 | **0.7792** | 0.7451 | 0.8127 | 297 | **210** |
| S2 | 0.9394 | 0.7791 | 0.7308 | 0.7542 | 0.7826 | 197 | 256 |
| S3 | **0.9433** | 0.8374 | 0.6877 | **0.7552** | 0.8117 | 127 | 297 |

**Không dùng bảng Test để chọn lại best.** Nó giúp đánh giá kết quả của lựa chọn đã thực hiện trước bằng CV. S3 có Test F1 nhỉnh hơn S2 một chút (0.7552 vs 0.7542); đó không phải lý do chính S3 được chọn, vì lựa chọn đã chốt ở cell 16 bằng CV F1.

## 14.3. Giải thích thay đổi khi so B4 → S3

- B4: FP=80, FN=354, Precision 0.8818, Recall 0.6278, F1 0.7334.
- S3: FP=127, FN=297, Precision 0.8374, Recall 0.6877, F1 0.7552.

Đổi ngưỡng khiến S3 **phát hiện thêm 57 Spam thật** (`354−297=57` FN ít hơn), đồng thời **cảnh báo nhầm thêm 47 Ham** (`127−80=47` FP nhiều hơn). F1 tăng từ 0.7334 lên 0.7552.

Đây là ví dụ rõ của **trade-off Precision–Recall**. S3 không hoàn hảo: 297 Spam thật vẫn bị bỏ sót trên Test.

## 14.4. Vì sao AP của B4 và S3 đều 0.8117?

AP dựa trên **thứ tự xếp hạng các score liên tục**, không dựa vào một ngưỡng để ra nhãn. B4 và S3 dùng cùng siêu tham số và toàn bộ Train; S3 về cơ chế chỉ thay ngưỡng quyết định nên score/ranking có thể giống B4 khi fit tái lập, do đó **AP giống nhau là hợp lý**. Trong Notebook, hai giá trị đều là 0.8117.

Không nói rằng **AP là F1** hoặc AP đo độ đúng tại ngưỡng −0.1776. F1 đổi khi threshold đổi; AP phụ thuộc ranking score trên nhiều ngưỡng.

## 14.5. Sự khác biệt giữa điểm CV và Test

S3: CV F1=0.7876, Test F1=0.7552.

Đây là hai bộ dữ liệu/vai trò khác nhau. CV dùng để chọn tham số và threshold; Test là dữ liệu độc lập. Điểm CV không phải lời đảm bảo Test đạt đúng mức đó.

Đặc biệt, điểm CV S3 sử dụng cùng OOF predictions để tìm threshold và báo cáo điểm tốt nhất; điều này có thể khiến điểm CV **lạc quan** hơn cách đánh giá nested CV độc lập. Không nên dùng chính điểm thắng đó làm ước lượng không chệch tuyệt đối cho hiệu suất chung.

---

# 15. Khi nào nên chọn B4, S1, S2 hay S3?

Đây là **giải thích kết quả thực nghiệm của nhóm**, không phải khuyến nghị mọi bài toán.

- **B4:** Precision cao nhất trên Test, FP ít nhất (80); phù hợp khi giảm cảnh báo nhầm được đặt nặng. Nhưng Recall thấp (0.6278), bỏ sót 354 Spam.
- **S1:** Recall cao nhất (0.7792), FN ít nhất (210), nhưng FP nhiều nhất (297) và Precision giảm mạnh. Phù hợp khi mục tiêu ưu tiên phát hiện nhiều Spam hơn, chấp nhận cảnh báo nhầm.
- **S2:** Trade-off khá cân bằng trên Test, nhưng phải bỏ nhiều Ham khỏi Train; mô hình cuối chỉ fit trên 13.125 câu, có nguy cơ mất thông tin đa dạng của Ham. Không làm mất dòng khỏi dataset gốc.
- **S3:** Trong các ứng viên đã khảo sát, CV F1 cao nhất; trên Test F1 cao nhất 0.7552, đánh đổi FP tăng so với B4 để bỏ sót ít Spam hơn. Giữ toàn bộ Train, sửa ngưỡng quyết định; được Notebook chọn.

**Không nên nói:** “S3 tốt nhất tuyệt đối”, “S1 luôn làm mô hình tốt hơn”, “undersampling chắc chắn tăng accuracy”, “chỉnh threshold tăng Recall mà không ảnh hưởng Precision”.

**Nên nói:** “Trong tập các phương án và dữ liệu đã thực nghiệm, S3 có CV F1 Spam cao nhất nên được chọn. Kết quả Test giúp quan sát thêm đánh đổi Recall–Precision và số FP/FN.”

---

# 16. Ví dụ thực hành: theo dõi dữ liệu qua các hàm

## 16.1. Demo đơn giản nhất — giảm mẫu Ham

Chạy trong Notebook/project sau khi import module của nhóm:

```python
import numpy as np
from pipeline.modeling.imbalance import sample_training_indices

y = np.array([0,0,1,0,1,0,0,0,1,0,0,0])
indices = np.arange(len(y))
selected = sample_training_indices(y, indices, seed=42, ratio=2)

print('Chỉ số được chọn:', selected)
print('Nhãn được chọn:', y[selected])
print('Số Ham/Spam:', np.bincount(y[selected], minlength=2))
```

Tập này có 9 Ham và 3 Spam. S2 giữ 3 Spam và chọn 6 Ham:

```text
Số Ham/Spam: [6 3]
```

**Điều cần nhìn:** `selected` là **vị trí dòng**, không phải vector đã transform. Hàm không động đến Test; ở CV hãy truyền vào `indices=train` thay vì `np.arange(len(y))`.

## 16.2. Demo class weight bằng số thật của nhóm

```python
n_train = 30759
n_ham = 26384
n_spam = 4375

w_ham = n_train / (2*n_ham)
w_spam = n_train / (2*n_spam)

print(round(w_ham, 5))   # 0.58291
print(round(w_spam, 5))  # 3.51531
print(round(w_spam/w_ham, 2))  # 6.03
```

**Đừng nhầm** `6.03` là “SVM tự nhân xác suất Spam lên 6 lần”. SVM không có xác suất mặc định; đây là tỷ lệ trọng số tương đối trong **bài toán tối ưu**.

## 16.3. Demo threshold với dữ liệu nhỏ — chạy được không cần train

```python
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

y_true = np.array([0,1,0,1,1,0])
scores = np.array([-1.0,-0.3,-0.1,0.2,0.5,0.9])

for threshold in [0.0, -0.4, -0.2]:
    y_pred = (scores > threshold).astype(int)
    print(
        'threshold', threshold,
        'pred', y_pred.tolist(),
        'P', round(precision_score(y_true,y_pred,zero_division=0),3),
        'R', round(recall_score(y_true,y_pred,zero_division=0),3),
        'F1', round(f1_score(y_true,y_pred,zero_division=0),3)
    )
```

Kết quả mong đợi:

```text
threshold  0.0: P=0.667 R=0.667 F1=0.667
threshold -0.4: P=0.600 R=1.000 F1=0.750
threshold -0.2: P=0.500 R=0.667 F1=0.571
```

**Ý chính:** threshold **không huấn luyện SVM**, chỉ quyết định mẫu nào được gọi Spam từ mảng score đã có. F1 có thể tăng hoặc giảm tùy score và nhãn thật.

## 16.4. Demo vòng lặp S2/S3 dùng đúng các hàm của dự án

Đoạn này tạo **40 bình luận giả lập**, không phải dữ liệu thật của nhóm. Mỗi fold có 8 câu (6 Ham, 2 Spam), để bạn có thể lần theo luồng. Nó tạo folds bằng tay cho bài tập nhỏ; **trong Notebook thật phải dùng `grouped_folds()`** nhằm tách theo khóa nội dung.

```python
import numpy as np
from pipeline.modeling.experiments import make_pipe
from pipeline.modeling.imbalance import (
    undersampled_search, tune_svm_threshold
)

X, y = [], []
subjects = ['java','toan','mang','python','ai']
for subject in subjects:
    for j in range(6):
        X.append(f'xin lich thi mon {subject} lop {j} sinh vien')
        y.append(0)
    for j in range(2):
        X.append(f'nhan lam thue do an mon {subject} gia re inbox {j}')
        y.append(1)

X = np.asarray(X)
y = np.asarray(y)

# Minh họa: 5 fold, mỗi fold 8 câu. KHÔNG thay thế grouped_folds thực tế.
folds = []
for f in range(5):
    train = np.array([i for i in range(40) if i//8 != f])
    valid = np.array([i for i in range(40) if i//8 == f])
    folds.append((train, valid))

pipe, grid, _ = make_pipe('TF-IDF', 'SVM', seed=42, n_jobs=1)

# S2: 2 n-gram × 4 C × 5 fold + fit cuối
us_model, us_scores, us_params, n_selected = undersampled_search(
    pipe, grid, X, y, folds, seed=42
)
print('S2 - số mẫu fit cuối:', n_selected)
print('S2 - params:', us_params)
print('S2 - CV:', us_scores)

# S3: ví dụ cố định siêu tham số B4 (minh họa C=0.3, n-gram=(1,2))
pipe.set_params(vec__ngram_range=(1,2), clf__C=0.3)
s3_model, s3_scores, s3_threshold = tune_svm_threshold(
    pipe, X, y, folds
)
print('S3 - threshold:', s3_threshold)
print('S3 - CV:', s3_scores)

# Đây là điểm quyết định trên câu mới, KHÔNG phải xác suất:
new_scores = s3_model.decision_function(['nhan lam do an gia re inbox'])
new_pred = (new_scores > s3_threshold).astype(int)
print('Score:', new_scores, 'Nhãn:', new_pred)
```

Tôi đã chạy kiểm tra ví dụ trên với đúng logic `imbalance(1).py` và scikit-learn khả dụng trong môi trường: S2 chọn 30 mẫu (`20 Ham + 10 Spam`) để fit cuối, còn S3 chạy được qua năm fold và trả về một ngưỡng. Dữ liệu ví dụ rất dễ tách lớp nên cả hai có thể có F1 hoàn hảo; **không dùng kết quả demo này để suy ra chất lượng mô hình thực tế**.

## 16.5. Mini-trace — một bình luận đi qua S2 và S3 khác nhau thế nào?

Giả sử bình luận Spam `"nhận làm thuê đồ án giá rẻ"` có nhãn thật `1`:

**Với S2**:

1. `y[train]==1` giữ lại vị trí bình luận nếu nó thuộc fold Train.
2. `sample_training_indices` thêm một tập Ham ngẫu nhiên sao cho Ham tối đa gấp đôi Spam.
3. Pipeline TF-IDF học vocabulary/IDF từ các câu được chọn.
4. LinearSVC học trọng số `coef_`, `intercept_` từ tập đã giảm Ham.
5. Khi bình luận đó ở validation của một lượt khác, Pipeline dùng `transform` và `predict`, không học từ bình luận validation.

**Với S3**:

1. Giữ nguyên tất cả câu thuộc fold Train, không giảm mẫu.
2. Pipeline B4 với C/ngram đã chốt từ bước 04 được clone và fit.
3. Bình luận ở fold Validation nhận decision score OOF, ví dụ `−0.10`.
4. Với ngưỡng mặc định `0.0`, dự đoán Ham; với ngưỡng được chọn `−0.1776`, dự đoán Spam.
5. Score OOF chỉ dùng để chọn threshold; mô hình cuối được fit lại trên **toàn Train** rồi áp threshold lên Test.

---

# 17. Quy tắc bảo vệ tính độc lập Train / Validation / Test

## 17.1. Bảng quyền được phép xem dữ liệu

| Giai đoạn | Được dùng để fit vectorizer/classifier? | Được dùng để chọn C/ngram/threshold? | Dùng để báo cáo cuối? |
|---|---|---|---|
| Fold Train (bên trong Train) | Có | Dùng gián tiếp thông qua các lượt fit | Không phải Test |
| Fold Validation (bên trong Train) | **Không ở lượt đó** | **Có**, chấm điểm các cấu hình/ngưỡng | Chỉ là điểm CV |
| Test độc lập | **Không** | **Không** | **Có**, sau khi best đã chốt |

Dữ liệu một bình luận Train có thể đóng vai **fold Train ở lượt này** và **fold Validation ở lượt khác**; đó là đặc điểm bình thường của 5-fold CV. Vấn đề là **trong cùng một lượt**, nó không được có mặt ở cả hai phía.

## 17.2. Sai lầm: giảm mẫu trước khi chia Train/Test

Nếu lấy mẫu trên toàn dataset trước rồi mới chia Train/Test, phân bố Test bị can thiệp và có thể không phản ánh bộ dữ liệu gốc. Trong dự án S2 chỉ giảm trên Train, còn Test luôn giữ nguyên 7.474 câu.

## 17.3. Sai lầm: giảm Ham của toàn Train trước rồi mới chạy CV

Cách đó làm phân bố validation cũng bị thay đổi do tập đã giảm. Dự án làm ngược lại: tạo `folds` trước, bên trong từng fold gọi `sample_training_indices(y, train, seed)` chỉ trên các chỉ số `train`, rồi chấm toàn `valid` nguyên bản.

## 17.4. Sai lầm: fit TF-IDF trên toàn Train trước rồi CV S2/S3

Nếu fit vocabulary/IDF trên dữ liệu chứa fold Validation trước, validation đã tác động vào biểu diễn đặc trưng. Dự án giữ vectorizer **bên trong Pipeline** và `clone(pipe).fit(X[train],y[train])`, đảm bảo mỗi fold học vocabulary/IDF từ đúng phần được phép.

## 17.5. S3 có dùng Test để tìm threshold không?

**Không.** Score dùng tìm threshold là score OOF từ **Train**. Khi cell 16 hoàn tất, `best` đã được chốt. Cell 20 mới chạy `evaluate` trên Test.

---

# 18. Giới hạn và những điểm phải trình bày chính xác

1. **S3 không phải nested CV độc lập.** CV score cùng lúc dùng để chọn threshold và báo cáo giá trị tối đa; điều đó có thể tạo ra ước lượng hơi lạc quan. Ngoài ra, C/ngram B4 đã được chọn theo CV trước. Đừng nói CV F1 S3 là hiệu suất chắc chắn ngoài mẫu.
2. **Ba chiến lược không được kết hợp.** File chỉ so ba cách riêng trên TF-IDF+SVM. Không có thí nghiệm `balanced + undersampling + threshold` hoặc thử tất cả chiến lược trên B1–B3.
3. **S3 giữ hyperparameter chứ không giữ nguyên estimator đã fit.** Nó nhận `focus['_model']`, nhưng `clone` rồi fit mới từng fold và full Train. Không nói "S3 không fit gì nữa".
4. **S2 chỉ under-sample Train của fold.** Validation/Test tuyệt đối không giảm; nếu giảm sẽ làm sai phân bố đánh giá.
5. **S1 không tăng xác suất Spam.** Nó thay đổi chi phí/lỗi trong tối ưu SVM; với cùng dữ liệu và C, mô hình có thể học ranh giới khác.
6. **S2 không thêm Spam, không loại bỏ dữ liệu khỏi CSV.** Chỉ chọn vị trí dùng fit; full Train còn 13.125 mẫu cho mô hình S2 cuối.
7. **Mọi score SVM đều không phải xác suất.** `−0.1776` là ngưỡng trên `decision_function`; âm/dương không phải phần trăm.
8. **Vì sao S3 Test AP giống B4?** Nếu score/ranking như nhau, đổi threshold chỉ đổi nhãn cứng; AP dùng score ranking nên có thể giữ nguyên. Đó là kết quả Notebook quan sát được.
9. **CV tính trung bình theo fold.** Do mỗi fold có số mẫu khác nhau, trung bình từng F1 không tương đương F1 gộp OOF. Code thực hiện trung bình không trọng số, điều đó nhất quán giữa các phương án S1/S2/S3 về nguyên tắc scoring.
10. **So sánh thời gian không hoàn toàn đồng nhất.** S1 dùng GridSearchCV có `n_jobs`; S2 dùng vòng lặp CV thủ công và không sử dụng `n_jobs` cho các lần fit; S3 thử nhiều ngưỡng nhưng không fit lại cho mỗi ngưỡng. Cần hiểu khác biệt khi nhận xét tốc độ.
11. **`min_df=2` thay đổi vocabulary theo dữ liệu fit.** S2 giảm Ham nên TF-IDF có thể học tập từ khác với B4/S1; đây là một hệ quả bình thường.
12. **Các dữ liệu và kết quả đang có trong Notebook, không có dataset thật trong bộ file này để tái fit hoàn chỉnh.** Chỉ đọc các output có sẵn, không tự nhận đã chạy lại 38.233 dòng.
13. **Khi đánh giá toàn bộ dự án so với viBERT**, cả hai cần dùng đúng tập Test tương ứng và tránh rò rỉ văn bản trùng theo khóa; đó là nhiệm vụ thiết kế nhóm. Bước này chỉ thay đổi học/phân loại mô hình truyền thống trên cùng Train đã chia.
14. **Giá trị `class_weight='balanced'` được tính lại từ từng fold Train.** Không dùng đúng cặp 0.58291/3.51531 cho mọi lần fit CV vì tỷ lệ lớp trong từng fold khác nhau.
15. **Threshold thuộc record/bundle.** Khi dự đoán một câu mới bằng `predict.py`, cần sử dụng **cùng ngưỡng S3 đã chọn** và **cùng preprocessing**; nếu quên ngưỡng và dùng 0 thì S3 sẽ bị áp như B4 mặc định.

---

# 19. Câu hỏi bảo vệ — học thuộc ý hiểu, không học thuộc máy móc

**C1. Vì sao phải có bước cân bằng dữ liệu khi SVM đã có thể phân loại?**  
Vì Train có 26.384 Ham và 4.375 Spam, mô hình có thể ưu tiên lớp Ham. Bước này thử ba cách để giảm bỏ sót Spam hoặc cải thiện F1, nhưng phải đánh đổi giữa Recall và Precision.

**C2. Đầu vào thực sự của `imbalance.py` là văn bản hay ma trận TF-IDF?**  
Là mảng **chuỗi token** `X_tr_text` và nhãn `y_train`, thêm folds. Pipeline bên trong fit sẽ tự gọi TfidfVectorizer để tạo ma trận.

**C3. S1 có đổi số bình luận Train không?**  
Không. S1 giữ đủ 30.759 câu; chỉ đổi `class_weight='balanced'` trong cách SVM học.

**C4. Scikit-learn tính trọng số balanced thế nào?**  
`w_c = N/(K*N_c)` trên dữ liệu đang fit. Full Train: Ham≈0.58291, Spam≈3.51531. Trọng số được tính lại theo từng fold Train.

**C5. `class_weight='balanced'` có làm score thành xác suất không?**  
Không. `LinearSVC.decision_function` vẫn trả score có dấu, không phải xác suất.

**C6. S2 là random undersampling hay oversampling?**  
Random undersampling: giữ mọi Spam, lấy ngẫu nhiên tối đa 2 Ham/1 Spam, không hoàn lại; không tự tạo Spam mới.

**C7. Vì sao S2 chỉ giảm Ham trong phần Train của từng fold?**  
Để phần Validation vẫn giữ phân bố gốc, tránh rò rỉ/chấm điểm lạc quan do thay đổi tập kiểm tra.

**C8. Vì sao S2 không dùng `GridSearchCV` trực tiếp?**  
Nhóm viết vòng lặp `ParameterGrid` + CV thủ công để đặt bước sampling **sau khi chia fold và chỉ trên fold Train**, rồi mới fit Pipeline.

**C9. Có thử `(1,1)` và `(1,2)` trong S2 không?**  
Có. ParameterGrid ghép 2 n-gram với 4 C, thử 8 cấu hình × 5 fold. S2 thắng `(1,2)` và `C=0.1`.

**C10. `clone(pipe)` khác `pipe` đã fit thế nào?**  
Clone giữ tham số nhưng bỏ trạng thái học; nhờ vậy mỗi fold phải học vocabulary/IDF và SVM mới, không thừa hưởng từ fold trước.

**C11. `sample_training_indices` trả gì?**  
Mảng vị trí mẫu được chọn; caller lấy `X[selected]`, `y[selected]` để đưa vào `fit`.

**C12. S2 fit cuối có bao nhiêu câu?**  
13.125: 4.375 Spam + 8.750 Ham, tỷ lệ 2 Ham/1 Spam. Các câu Ham không được chọn vẫn còn trong CSV gốc.

**C13. `mean_cv_scores` lấy trung bình gì?**  
Tính Precision, Recall, F1 Spam riêng từng validation fold rồi lấy **trung bình cộng 5 fold**, không gộp hết nhãn OOF rồi tính một lần.

**C14. S3 làm thay đổi C hoặc n-gram không?**  
Không; S3 giữ siêu tham số B4 (C=0.3, ngram=(1,2)), thử threshold trên score OOF Train.

**C15. `focus['_model']` đã fit, sao S3 còn `clone(pipe).fit`?**  
Vì muốn score OOF, cần một mô hình mới không nhìn thấy chính câu validation. Clone giữ cấu hình B4 nhưng bỏ trạng thái học rồi fit từng fold; cuối cùng fit lại toàn Train.

**C16. Thế nào là OOF?**  
Out-of-fold: dự đoán cho mỗi dòng Train bởi mô hình fit trên các fold **khác**, không fit trên dòng đó. Sau 5 lượt, mọi dòng nhận đúng một score.

**C17. S3 có bao nhiêu ngưỡng?**  
199 phân vị từ 1% đến 99% + ngưỡng 0 + hai biên, rồi `np.unique`; tối đa 202 mức khác nhau.

**C18. Vì sao cần `np.nextafter(min, -inf)` và `max`?**  
Để có biên mà toàn bộ OOF có thể bị gọi Spam hoặc toàn bộ bị gọi Ham, xem hai trường hợp cực đoan khi tối ưu threshold.

**C19. Tại sao threshold âm lại có thể tăng Recall?**  
Vì những câu score âm nhẹ trước đây thành Ham sẽ chuyển sang Spam nếu `score > threshold âm`; bắt thêm Spam thật nhưng có thể làm tăng FP.

**C20. S3 có thử lại SVM cho từng ngưỡng không?**  
Không. S3 fit năm mô hình fold để tạo một mảng score OOF, thử rất nhiều threshold trên mảng này, rồi fit lại một model cuối.

**C21. Tại sao chọn S3 chứ không phải S1 khi Recall S1 lớn hơn?**  
Tiêu chí lựa chọn đã định trước là **CV F1 Spam**, không phải Recall riêng. S3 F1=0.7876 cao hơn S1=0.7580 và S2=0.7629.

**C22. Khi nào Test được sử dụng?**  
Sau khi chốt `best` tại cell 16. Cell 20 mới `evaluate` trên `test_idx`. Không chọn threshold/C theo kết quả Test.

**C23. Vì sao kết quả CV F1 S3 = 0.7876 mà Test F1=0.7552?**  
Hai tập dữ liệu/vai trò khác nhau; CV được dùng để chọn cấu hình và ngưỡng nên có thể lạc quan; Test là đánh giá độc lập.

**C24. B4 và S3 cùng F1 khác nhau nhưng AP bằng nhau có mâu thuẫn không?**  
Không. F1 dùng nhãn tại một threshold, còn AP dùng thứ tự score qua nhiều ngưỡng. S3 thay threshold nên F1 đổi mà ranking score và AP có thể giữ nguyên.

**C25. Cái gì trong Notebook quyết định mô hình tốt nhất?**  
`CANDIDATES = RESULTS + SUPPLEMENTARY` rồi `best = max(CANDIDATES, key=lambda r: r['CV F1 (spam)'])`.

**C26. Dùng threshold đã chọn ở đâu khi predict bình luận mới?**  
Threshold được lưu trong record/bundle, `predict_record()` lấy `_threshold`, tính `decision_function` rồi áp `(score > threshold).astype(int)`.

**C27. Điểm yếu của cách chọn ngưỡng S3?**  
Tối ưu threshold trên OOF rồi cũng báo cáo chính CV F1 tối ưu ấy; đây không phải nested CV hoàn toàn độc lập. Test giữ riêng mới là đánh giá quan trọng.

**C28. Nếu muốn so công bằng S1/S2/S3 thì cần giữ nguyên gì?**  
Cùng nguồn Train, cùng split/folds, cùng nhãn, cùng nhánh tiền xử lý; chỉ thay đổi yếu tố chiến lược. Validation/Test không được giảm mẫu.

---

# 20. Bản trình bày nói miệng và checklist

## 20.1. Nói miệng khoảng 1–2 phút

> Sau khi so sánh bốn tổ hợp cơ sở bằng Cross-Validation, nhóm chọn TF-IDF + Linear SVM làm đối chứng B4 vì có CV F1 Spam cao nhất. Tuy nhiên trong Train có hơn 26 nghìn Ham và chỉ khoảng 4,4 nghìn Spam, nên nhóm khảo sát ba cách xử lý mất cân bằng trong `imbalance.py`.
>
> S1 dùng `class_weight='balanced'`, nghĩa là giữ nguyên dữ liệu nhưng tăng trọng số lỗi với Spam khi SVM học, đồng thời dùng CV để chọn lại n-gram và C. S2 dùng random under-sampling: với mỗi lượt CV, giữ toàn bộ Spam trong fold Train và lấy tối đa hai Ham trên một Spam. Validation vẫn giữ nguyên; nhóm tự viết vòng lặp `ParameterGrid` để đảm bảo lấy mẫu đúng chỗ. S3 giữ cấu hình n-gram và C của B4, tạo decision score ngoài fold trên Train rồi tìm ngưỡng dự đoán có CV F1 Spam cao nhất.
>
> Cuối cùng, Notebook so sánh B4, S1, S2, S3 theo CV F1. S3 đạt 0.7876 và được chọn với ngưỡng khoảng −0.1776. Mô hình được chốt trước khi chạy Test. Trên Test, S3 đạt F1 Spam 0.7552, giảm số Spam bị bỏ sót so với B4 nhưng số Ham bị cảnh báo nhầm tăng. Điều này cho thấy sự đánh đổi giữa Precision và Recall khi xử lý mất cân bằng.

## 20.2. Checklist tự học

- [ ] Nói được lý do Accuracy không đủ khi Spam là lớp thiểu số.
- [ ] Giải thích được TP, FP, FN, TN; tính Precision/Recall/F1 bằng số nhỏ.
- [ ] Tính được class_weight của Ham/Spam theo công thức.
- [ ] Đọc được `sample_training_indices`, đặc biệt là `indices[y[indices]==1]`.
- [ ] Phân biệt undersampling **chỉ trên Train của fold** với giảm toàn bộ dataset.
- [ ] Giải thích được `ParameterGrid`, `clone`, `set_params`, `fit`, `predict` trong S2.
- [ ] Tự viết được quy trình tạo score OOF trong S3.
- [ ] Biết tại sao `np.linspace` → `np.quantile` → `np.unique` tạo ngưỡng.
- [ ] Phân biệt threshold trên score với C của SVM và xác suất.
- [ ] Hiểu `mean_cv_scores` lấy trung bình theo fold.
- [ ] Trình bày chính xác `best` được chọn **trước Test**, chỉ từ CV F1.
- [ ] Đọc được bảng CV và Test thật, không đánh đồng hai kết quả.
- [ ] Nhận ra giới hạn non-nested CV của S3.

---

# 21. Liên hệ các file và bước kế tiếp

| File/cell | Đóng góp cho bước 05 |
|---|---|
| `training_utils.py`, cell 11 | Chia Train/Test và 5 fold nhóm nội dung, đảm bảo cùng đầu vào đánh giá |
| `text_features.py` | `TfidfVectorizer`; fit riêng trên mỗi fold Train, không học trước từ Test |
| `classifiers.py` | `LinearSVC` và các giá trị C; S1 đổi `class_weight` |
| `experiments.py` | `make_pipe()`, `fit_experiment()` — S1 gọi lại GridSearchCV; S2 tạo Pipeline từ `make_pipe()` |
| **`imbalance.py`** | Toàn bộ S1/S2/S3 và các hàm hỗ trợ |
| **Notebook 02 cell 14–16** | Gọi ba chiến lược, đóng record, so CV F1, chốt `best` |
| `evaluation.py`, cell 20–22 | **Bước kế tiếp:** dùng model đã chốt dự đoán Test và tính Accuracy, Precision, Recall, F1, AP, FP/FN |
| `predict.py`, cell 32 | Sử dụng model và threshold đã lưu để dự đoán câu mới |

**File code gốc để đối chiếu từng dòng:** `pipeline/modeling/imbalance.py` (upload `imbalance(1).py`).  
**Notebook để đối chiếu thực nghiệm:** `02_thuc_nghiem_va_demo(1).ipynb`, các cell **14, 15, 16**, xem thêm 11, 13, 20 và 22.  
**Tài liệu thư viện liên quan:** `PHI_HIEU_4_LOP_SCIKIT_LEARN.md`, phần `LinearSVC`, `_fit_liblinear`, `decision_function`; hàm bổ sung đọc ngay trong thư viện là `sklearn.base.clone`, `sklearn.model_selection.ParameterGrid`, `sklearn.metrics.f1_score`/`precision_score`/`recall_score`.

**Kết luận một câu:** `imbalance.py` không chỉ “cân bằng số lượng lớp”; nó triển khai **ba giả thuyết khác nhau** để cải thiện phát hiện Spam: **đổi trọng số học (S1)**, **đổi tập mẫu học (S2)** hoặc **đổi ngưỡng ra quyết định (S3)**, và Notebook chọn bằng **CV F1 Spam trước khi đánh giá Test**.
