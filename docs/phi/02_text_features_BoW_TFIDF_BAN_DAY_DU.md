# BƯỚC 02 — `text_features.py`: TRÍCH CHỌN ĐẶC TRƯNG BOW VÀ TF-IDF — BẢN GIẢI THÍCH CHUYÊN SÂU

> **Tài liệu học và bảo vệ bài tập lớn: Phân loại bình luận sinh viên PTIT thành Ham (0) và Spam (1).**  
> **Bản này viết lại và mở rộng:** đi sâu vào mã thư viện scikit-learn (không chỉ gọi API), mô phỏng từng bước để tự tạo ma trận, giải thích đầy đủ (1,1)–(1,2), đường GridSearchCV thử và chọn, cùng cách xem điểm từng cấu hình. Các mục còn lại của bản trước được giữ và tích hợp trong cùng tài liệu.
> 
> **Phiên bản code làm căn cứ:** `text_features(1).py` (bản mới **dùng scikit-learn**, không dùng `ProjectTextVectorizer` tự viết), `02_thuc_nghiem_va_demo(1).ipynb` (đặc biệt cell 4, 9, 11, 12, 13, 16, 20, 32), `experiments(1).py`, `classifiers(1).py`, `predict(1).py`, `README(1).md` và `PHI_HIEU_4_LOP_SCIKIT_LEARN.md`.  
> **Phạm vi:** Người phụ trách modeling **nhận dataset đã gán nhãn và chuỗi token đã chuẩn bị**; không đi sâu vào cách thu thập, gán nhãn hay tự viết bộ tiền xử lý.  
> **Quy ước:** “Số liệu thực nghiệm” là kết quả đã lưu trong Notebook, **không phải kết quả tôi huấn luyện lại**. Các bình luận ngắn, vector và phép tính minh họa là ví dụ tự dựng và được ghi rõ.  
> **Phiên bản thư viện của Notebook:** scikit-learn **1.6.1**. Nếu mở mã nguồn qua Ctrl+Click trên một phiên bản khác, vị trí dòng có thể khác.

---

## 1. Học xong bước này phải trả lời được những câu gì?

1. **Tại sao cần trích chọn đặc trưng?** Một thuật toán học máy nhận ma trận số, không thể trực tiếp áp phép tính SVM/NB lên câu chữ chưa biểu diễn.
2. **Đầu vào của `text_features.py` đến từ đâu?** Chuỗi token do bước tiền xử lý tạo; Notebook chọn dòng Train qua `train_idx`.
3. **BoW là gì, TF-IDF là gì, khác nhau thế nào?** Cùng biểu diễn bằng cột từ/ngữ; BoW lưu số đếm, TF-IDF lưu trọng số.
4. **Vocabulary, unigram, bigram, document frequency (`df`), `min_df`, TF, IDF, L2, sparse matrix là gì?**
5. **`CountVectorizer` và `TfidfVectorizer` đến từ đâu?** Thư viện `sklearn.feature_extraction.text`.
6. **Hàm nhóm viết `make_vectorizer(feature)` nhận gì, làm gì, trả gì?** Nó chỉ tạo vectorizer *chưa fit* và lưới n-gram.
7. **`fit`, `fit_transform`, `transform` khác nhau thế nào?** Học từ vựng/IDF trên phần Train, rồi tạo vector; khi xử lý Validation/Test không học lại.
8. **Vì sao `text_features.py` không gọi `fit()`?** Việc fit thật xảy ra ở `Pipeline` trong `experiments.py` khi `GridSearchCV.fit()` chạy.
9. **`vec__ngram_range` nghĩa là gì?** Siêu tham số `ngram_range` của bước đặt tên `vec` trong Pipeline.
10. **Vector được chuyển sang classifier thế nào?** `vec.fit_transform(text_train)` tạo ma trận, `clf.fit(matrix, y_train)` học nhãn Spam/Ham.
11. **Sao biết phương pháp nào tốt hơn?** Dựa trên **CV F1 Spam** của *toàn bộ Pipeline*, không dựa vào cảm giác vector có vẻ đẹp hơn.
12. **Tại sao phải lưu vectorizer đã fit cùng mô hình?** Để văn bản mới dùng đúng từ điển/các IDF mà mô hình đã học.

### Ranh giới trách nhiệm

- **Thành viên tiền xử lý:** Chuẩn hóa và tách từ tiếng Việt, tạo hai cột token giữ/bỏ stopwords.
- **Bạn (modeling):** Sử dụng các cột này, chọn Train/fold, tạo BoW/TF-IDF, ghép classifier, fit/tối ưu, đánh giá, lưu và dự đoán.
- **Notebook 03 / viBERT:** Một hướng biểu diễn và mô hình khác; việc so sánh công bằng dựa trên phân vùng Train/Test tương ứng. Không cần học sâu viBERT ở bước này.

---

## 2. Bản đồ chính xác: văn bản đi qua đâu và kết quả chuyển sang file nào?

```text
Notebook 02 — cell 9: TEXT = {
    'giữ stop words':      Series chứa chuỗi token,
    'có loại stop words':  Series chứa chuỗi token khác
}
                                |
Notebook 02 — cell 11: train_idx, test_idx, CV_FOLDS
                                |
Notebook 02 — cell 13:
    X_train_text = TEXT[pre].iloc[train_idx].to_numpy()
                                |
                fit_experiment(pre, feature, algorithm,
                               X_train_text, y_train, CV_FOLDS)
                                |
                     experiments.py / make_pipe
                 _________|_________________________
                |                                   |
       text_features.py                     classifiers.py
       make_vectorizer(feat)               make_classifier(model)
       CountVectorizer hoặc                MultinomialNB hoặc
       TfidfVectorizer                     LinearSVC
                |                                   |
                +----------- Pipeline --------------+
                       [('vec', ...), ('clf', ...)]
                                  |
                     GridSearchCV(...).fit(X, y)
                                  |
                  MỖI FOLD TRAIN (chuỗi token, nhãn):
                   vec.fit_transform(text_train)
                          → ma trận số
                   clf.fit(matrix, y_fold_train)
                                  |
                  FOLD VALIDATION (chuỗi token):
                   vec.transform(text_valid)
                   clf.predict / decision_function
                          → nhãn/score → CV F1
                                  |
                  Chọn cấu hình CV F1 cao nhất
                                  |
                     REFIT trên toàn bộ Train
                                  |
                   Pipeline đã fit trong record
                                  |
              evaluation.py: transform trên Test,
              dự đoán nhãn và so với y_test
                                  |
       Notebook cell 32: joblib.dump(bundle chứa Pipeline)
                                  |
                 modeling/predict.py: bình luận mới
          preprocess → vec.transform → clf score → Spam/Ham
```

**Vì sao không phải chạy `python text_features.py` riêng để train?** File này chỉ có khai báo tham số, `import` và định nghĩa `make_vectorizer`. Hàm chỉ chạy khi được file khác **gọi**. Trong Notebook, nó được gọi gián tiếp bởi `fit_experiment()` → `make_pipe()`.

---

## 3. Đầu vào thật trong Notebook: cần biết vừa đủ

### 3.1. Cell 9: Dữ liệu đã được thành viên khác chuẩn bị

Các dòng quan trọng:

```python
prepared["tokens_keep_stopwords"] = [
    preprocess(c, t, False)
    for c, t in zip(raw.cleaned_text, raw.text)
]

prepared["tokens_remove_stopwords"] = (
    prepared.tokens_keep_stopwords.map(
        lambda text: " ".join(token for token in text.split()
                              if token not in STOP_WORDS)
    )
)

TEXT = {
    "giữ stop words": prepared.tokens_keep_stopwords,
    "có loại stop words": prepared.tokens_remove_stopwords,
}
```

**Chỉ cần hiểu:**

- `TEXT` là dictionary chứa **hai cột chuỗi**, chưa phải ma trận đặc trưng.
- Mỗi vị trí là một bình luận tương ứng cùng vị trí nhãn trong dataset.
- `ngram_range` của `CountVectorizer/TfidfVectorizer` được áp lên **chuỗi token đã có**, không thay thế bước tách từ tiếng Việt của `underthesea`.
- `stop_words` **không được truyền** vào hai vectorizer của bạn; việc loại stopwords đã diễn ra trước đó, khi tạo cột token tương ứng.
- Chi tiết `preprocess()` là phần của thành viên khác; riêng khi `predict.py` dự đoán câu mới, phải dùng lại chế độ tiền xử lý đã chọn.

### 3.2. Cell 11: Tập được phép học

```python
keys, train_idx, test_idx = split_train_test(raw["cleaned_text"], y_all)
y_train, y_test = y_all[train_idx], y_all[test_idx]
CV_FOLDS = grouped_folds(keys[train_idx], y_train)
```

Notebook ghi nhận **30.759 Train** và **7.474 Test**, tổng **38.233** bình luận.

### 3.3. Cell 13: Vectorizer thực sự được sử dụng

```python
for pre in TEXT:
    X_train_text = TEXT[pre].iloc[train_idx].to_numpy()
    for feature in FEATURES:
        for algorithm in MODELS:
            record = fit_experiment(
                pre, feature, algorithm,
                X_train_text, y_train, CV_FOLDS,
                seed=SEED, cache=cache, n_jobs=N_JOBS
            )
```

- `pre`: nhánh **giữ** hoặc **bỏ** stopwords.
- `X_train_text`: mảng chuỗi token **chỉ của Train**.
- `feature`: `"BoW"` hoặc `"TF-IDF"`.
- `algorithm`: `"Naive Bayes"` hoặc `"SVM"`.
- `y_train`: nhãn thật của mỗi chuỗi Train, **chưa được dùng trong vector hóa**.
- `CV_FOLDS`: 5 cặp chỉ số (train-fold, validation-fold) chỉ thuộc tập Train.

Cell 13 thử **2 nhánh stopwords × 2 phương pháp biểu diễn × 2 classifier = 8 thí nghiệm cơ sở theo cặp**; bên trong mỗi cặp còn thử n-gram và alpha/C bằng CV. Sau so sánh, Notebook chọn **giữ stopwords** cho các bước tiếp theo.

---

## 4. Bảng thuật ngữ nền tảng

| Thuật ngữ | Nghĩa | Ví dụ trong bài |
|---|---|---|
| **Document/sample** | Một văn bản cần phân loại | Một bình luận Facebook |
| **Corpus** | Tập nhiều văn bản | Các bình luận Train |
| **Token** | Đơn vị sau bước tách từ | `lịch_thi`, `java`, `!` |
| **Term/feature** | Đặc trưng làm cột cho vector | `lịch_thi` hoặc `xin lịch_thi` |
| **Unigram** | Một token làm đặc trưng | `lịch_thi` |
| **Bigram** | Hai token đứng cạnh nhau làm một đặc trưng | `xin lịch_thi` |
| **Vocabulary** | Ánh xạ đặc trưng → cột số | `{'lịch_thi': 120, ...}` |
| **Vector** | Dãy số biểu diễn một bình luận | `[0, 2, 0, 1, ...]` |
| **Feature matrix `X`** | Các vector ghép thành ma trận | Mỗi hàng một bình luận, mỗi cột một term |
| **Label/target `y`** | Nhãn thật, không phải vector đặc trưng | Ham=`0`, Spam=`1` |
| **Term frequency (TF)** | Tần suất của term trong một văn bản | `inbox` lặp 3 lần trong một câu |
| **Document frequency (DF)** | Số văn bản chứa term, không đếm lặp trong cùng câu | `inbox` có mặt trong 100 bình luận |
| **Inverse document frequency (IDF)** | Thành phần giảm trọng số đặc trưng gặp ở nhiều văn bản | Từ gặp ít tài liệu có IDF lớn hơn |
| **Sparse/CSR matrix** | Ma trận thưa, lưu chủ yếu những ô khác 0 | Mỗi câu chỉ dùng vài trong hàng chục nghìn cột |
| **OOV (out-of-vocabulary)** | Từ ngoài từ vựng đã học | Từ mới chỉ xuất hiện trong Test |
| **Fit** | Học trạng thái cần thiết từ Train | Vocabulary, IDF |
| **Transform** | Áp trạng thái đã học lên văn bản | Chuỗi token → vector theo từ điển cũ |

### 4.1. Vì sao máy phải có vector?

Một bộ phân loại như `LinearSVC` tính theo các đặc trưng số; `MultinomialNB` cũng cộng, ước lượng phân bố đặc trưng theo nhãn. Dòng chữ `"nhận làm đồ án"` tự nó chưa phải ma trận số có số cột cố định để hai thuật toán này học.

**Bộ tạo vector** có nhiệm vụ chuyển câu chữ thành các giá trị số theo một tập đặc trưng nhất quán. Còn **bộ phân loại** học liên hệ giữa các giá trị đó và nhãn Ham/Spam.

### 4.2. Vì sao có `vocabulary_`?

Ví dụ vocabulary:

```python
{"làm": 0, "lịch": 1, "nhận": 2, "thi": 3, "xin": 4}
```

Nó định nghĩa rằng trong **mọi** vector sau khi fit:

- Cột 0 luôn chỉ đặc trưng `làm`.
- Cột 1 luôn chỉ đặc trưng `lịch`.
- ...

Nếu validation/test tự học từ điển khác, cột 0 có thể chỉ từ khác. Khi đó mô hình đã học sẽ đọc sai ý nghĩa các hệ số. Vì vậy **test chỉ gọi `transform()`**, không `fit_transform()`.

### 4.3. Bigram không đồng nghĩa với thuật toán tách từ tiếng Việt

Giả sử token đã chuẩn bị là:

```text
xin lịch_thi java
```

`ngram_range=(1, 1)` tạo: `xin`, `lịch_thi`, `java`.

`ngram_range=(1, 2)` tạo cả từ đơn trên **và** cặp kề nhau: `xin lịch_thi`, `lịch_thi java`.

`ngram_range` không trực tiếp hiểu ngữ nghĩa một cụm từ; nó chỉ ghép token kề nhau. Nếu tokenizer trước đó chia câu không tốt, n-gram cũng có thể kém hữu ích.

---


### 4.4. ĐÀO SÂU: `(1,1)` và `(1,2)` là gì, được tạo ra ở khâu nào?

**Cú pháp:** `ngram_range=(min_n, max_n)` nghĩa là tạo mọi n-gram có số token từ `min_n` đến `max_n` (bao gồm cả hai đầu).

- `(1,1)` chỉ có **unigram**: một token là một đặc trưng.
- `(1,2)` gồm **unigram + bigram**: giữ token đơn, đồng thời thêm cặp hai token đứng cạnh nhau.
- `(2,2)` mới là **chỉ bigram**, nhưng dự án **không thử** lựa chọn này.

**Ví dụ phải phân biệt tách từ tiếng Việt với tạo n-gram:** Giả sử thành viên tiền xử lý đưa vào chuỗi `"xin lịch_thi học_kỳ"`. Dấu gạch dưới trong `lịch_thi`/`học_kỳ` xuất phát từ tokenizer tiếng Việt trước đó. Theo regex `token_pattern` của project, mỗi cụm có dấu gạch dưới là **một token**.

| Thao tác | Đầu ra |
|---|---|
| `build_tokenizer` | `['xin', 'lịch_thi', 'học_kỳ']` |
| `ngram_range=(1,1)` | `['xin', 'lịch_thi', 'học_kỳ']` |
| `ngram_range=(1,2)` | `['xin', 'lịch_thi', 'học_kỳ', 'xin lịch_thi', 'lịch_thi học_kỳ']` |

Điều này xảy ra **trước khi** thư viện xem xét `min_df=2`. Một bigram chỉ xuất hiện trong một bình luận Train có thể bị xóa; không phải bật `(1,2)` là tất cả bigram chắc chắn tồn tại trong vocabulary.

**Một ví dụ có dấu câu:** `"xin lịch_thi !"` có 3 token `['xin','lịch_thi','!']`, do regex còn giữ ký hiệu `!` như một token. `(1,2)` còn tạo `"xin lịch_thi"` và `"lịch_thi !"`. Vì vậy một số bigram gắn với dấu câu có thể xuất hiện. Tính chất này là kết quả của `TOKEN_PATTERN`, không phải lỗi thư viện.

**Câu hỏi bạn đặt: "Ở file này tôi chỉ thấy `(1,1)`, `(1,2)` ở đâu?"** Đáp án chính xác:

```python
# Ở text_features(1).py
NGRAMS = [(1, 1), (1, 2)]
# Bên trong common: chỉ là GIÁ TRỊ BAN ĐẦU của đối tượng
"ngram_range": (1, 1)
# Giá trị được trả cho GridSearchCV thử NẰM TẠI ĐÂY
return vectorizer, {"vec__ngram_range": NGRAMS}
```

`CountVectorizer(**common)`/`TfidfVectorizer(**common)` ban đầu có `(1,1)`. Nhưng khi `GridSearchCV` thử từng tham số, nó gọi cơ chế `set_params(vec__ngram_range=(1,2))` trên một bản sao Pipeline, tức thay đúng tham số bước `vec` **trước khi fit**. Bạn không nhìn thấy dòng thay đổi đó trong `text_features.py` vì nó được scikit-learn điều phối bên trong `GridSearchCV`.

**Một bẫy cần nhớ:** N-gram là đặc trưng được bổ sung, không phải một phép nối các câu lại hoặc làm xuất hiện thêm dòng dữ liệu. `(1,2)` tăng **số cột** tiềm năng của ma trận, không tăng số bình luận (hàng).

---

## 5. Toàn bộ mã nguồn `text_features(1).py` cần hiểu

```python
"""Tạo BoW hoặc TF-IDF bằng hai lớp có sẵn của scikit-learn.

make_vectorizer chỉ tạo đối tượng và lưới n-gram. Việc học từ vựng/IDF diễn ra
khi Pipeline được fit trong experiments.py, riêng trên train của từng fold.
"""

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

TOKEN_PATTERN = r"(?u)\b\w+\b|[^\w\s]"
NGRAMS = [(1, 1), (1, 2)]
FEATURES = ("BoW", "TF-IDF")
MIN_DOCUMENT_FREQUENCY = 2


def make_vectorizer(feature):
    """Trả (bộ tạo vector CHƯA fit, lưới n-gram để CV thử)."""
    common = {
        "token_pattern": TOKEN_PATTERN,
        "min_df": MIN_DOCUMENT_FREQUENCY,
        "ngram_range": (1, 1),
        "lowercase": True,
        "dtype": np.float64,
    }
    if feature == "BoW":
        vectorizer = CountVectorizer(**common)
    elif feature == "TF-IDF":
        vectorizer = TfidfVectorizer(
            **common, sublinear_tf=True, smooth_idf=True, norm="l2"
        )
    else:
        raise ValueError(f"Đặc trưng không hỗ trợ: {feature}")
    return vectorizer, {"vec__ngram_range": NGRAMS}
```

**Đọc code theo ba phần:**

1. `import` thư viện.
2. Khai báo quy tắc chung và phương án cần thử.
3. `make_vectorizer()` tạo đúng đối tượng vector hóa và lưới tham số.

Phân biệt **cấu hình** với **trạng thái đã học**: các dòng trên đều chưa biết vocabulary hoặc IDF của dataset. Những thuộc tính ấy chỉ xuất hiện sau `fit()`.

---

## 6. Giải thích từng import và hằng số

### 6.1. `import numpy as np`

```python
import numpy as np
```

Sử dụng `np.float64` để quy định kiểu dữ liệu số trong ma trận được tạo. Kể cả khi BoW đếm 1, 2, 3 lần, các ô vẫn được lưu theo kiểu `float64` (ví dụ `2.0`).

**`np.float64` không tự tạo TF-IDF và cũng không phải thuật toán xử lý văn bản.** Nó chỉ là kiểu số.

### 6.2. Import lớp `CountVectorizer`, `TfidfVectorizer`

```python
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
```

Hai lớp do **scikit-learn** cung cấp:

- `CountVectorizer`: text → ma trận số đếm BoW.
- `TfidfVectorizer`: text → ma trận TF-IDF.

Cả hai đều cung cấp các phương thức quan trọng `fit()`, `fit_transform()`, `transform()`, `get_feature_names_out()`.

**Đừng nhầm:** bạn không tự viết vòng lặp đếm từ hoặc tính IDF trong `text_features.py` phiên bản mới. Những bước này nằm **bên trong** các lớp thư viện.

### 6.3. `TOKEN_PATTERN`

```python
TOKEN_PATTERN = r"(?u)\b\w+\b|[^\w\s]"
```

Đây là **regular expression (regex)** làm quy tắc nhận diện token khi `analyzer='word'` (mặc định). Giải thích:

- Tiền tố `r"..."`: chuỗi *raw string*, giúp viết `\b`, `\w` mà không cần Python diễn giải escape theo cách khác.
- `(?u)`: bật hành vi Unicode cho regex (trong Python 3 mặc định đã thiên về Unicode).
- `\b`: ranh giới từ.
- `\w+`: một hoặc nhiều ký tự dạng chữ/số/gạch dưới, có hỗ trợ Unicode.
- `|`: “hoặc”.
- `[^\w\s]`: một ký tự **không phải** chữ/số/gạch dưới và **không phải** khoảng trắng; do đó có thể bắt dấu câu/ký hiệu như `!`, `?`.

Ví dụ minh họa:

```text
Đầu vào:  "xin lịch_thi!!!"
Token:    ["xin", "lịch_thi", "!", "!", "!"]
```

Vì `\w` bao gồm `_`, chuỗi `lịch_thi` đã tách từ bằng `underthesea` thường được xem là **một token**.

**Vì sao cần pattern riêng?** Pattern mặc định của `CountVectorizer` chủ yếu lấy token chữ/số dài từ hai ký tự và bỏ dấu câu. Cấu hình của bài cho phép giữ cả token một ký tự và dấu câu để khảo sát tín hiệu văn bản. Giữ dấu câu **không đảm bảo** dấu câu hữu ích; các đặc trưng vẫn phải vượt qua `min_df=2`.

**Điểm cần lưu ý:** Với `(1,2)`, bigram cũng có thể ghép token dấu câu kề từ, ví dụ `thi !`. Đây là hiệu ứng của quy tắc token hóa đang dùng, không phải lỗi thư viện.

### 6.4. `NGRAMS`

```python
NGRAMS = [(1, 1), (1, 2)]
```

Đây là **lưới hai phương án**:

- `(1, 1)`: unigram.
- `(1, 2)`: unigram **cộng** bigram.

Không có cấu hình chỉ bigram `(2,2)` hoặc trigram `(1,3)` trong dự án hiện tại.

`GridSearchCV` sẽ thử cả hai phương án và lựa chọn dựa trên **CV F1 Spam** của toàn bộ Pipeline (bước huấn luyện sẽ học tiếp ở `experiments.py`).

### 6.5. `FEATURES`

```python
FEATURES = ("BoW", "TF-IDF")
```

Đây là hai **tên hợp lệ** mà `make_vectorizer()` nhận. Ví dụ nếu truyền `"Word2Vec"`, hàm sẽ báo lỗi `ValueError`, không tự chuyển sang thuật toán khác.

### 6.6. `MIN_DOCUMENT_FREQUENCY = 2`

```python
MIN_DOCUMENT_FREQUENCY = 2
```

Một term chỉ được giữ nếu xuất hiện trong **ít nhất hai văn bản khác nhau thuộc tập được fit**.

Ví dụ trong fold train:

| Term | Số lần lặp trong một bình luận | Số **bình luận khác nhau** có term | Giữ? |
|---|---:|---:|---|
| `xin` | Có thể lặp nhiều lần | 20 | Có |
| `lịch_thi` | 1 | 3 | Có |
| `abc_xyz_hiem` | 20 trong cùng một câu | 1 | **Không** |

Quan trọng: `min_df` sử dụng **document frequency**, không phải tổng số lần xuất hiện của từ. Một term lặp 20 lần trong cùng một câu vẫn có DF=1.

**Mục đích:** bớt từ quá hiếm/lỗi gõ, giảm số chiều và độ nhiễu.

**Đánh đổi:** có thể loại term Spam hiếm nhưng rất quan trọng. Đây là một quyết định của nhóm, không phải quy luật rằng `min_df=2` luôn tối ưu.

**Trong CV:** min_df được tính lại trên *phần train của từng fold*. Không tính từ toàn CSV trước rồi áp lên các fold, vì như thế sẽ nhìn trước thông tin validation/Test.

---

## 7. Hàm duy nhất nhóm viết: `make_vectorizer(feature)`

### 7.1. Chữ ký, đầu vào và đầu ra

```python
def make_vectorizer(feature):
    ...
    return vectorizer, {"vec__ngram_range": NGRAMS}
```

**Đầu vào:** Một chuỗi `feature` bằng `"BoW"` hoặc `"TF-IDF"`.

**Đầu ra:** Một *tuple gồm 2 thành phần*:

1. `vectorizer`: Đối tượng `CountVectorizer` hoặc `TfidfVectorizer` **chưa fit**.
2. `vector_grid`: Dictionary cho phép `GridSearchCV` thử `(1,1)` và `(1,2)`.

Ví dụ:

```python
vec, grid = make_vectorizer("BoW")
print(type(vec))       # sklearn...CountVectorizer
print(grid)            # {'vec__ngram_range': [(1, 1), (1, 2)]}
```

Chưa thể lấy `vec.vocabulary_` lúc này, vì chưa học trên bộ văn bản nào.

### 7.2. `common = {...}` để làm gì?

```python
common = {
    "token_pattern": TOKEN_PATTERN,
    "min_df": MIN_DOCUMENT_FREQUENCY,
    "ngram_range": (1, 1),
    "lowercase": True,
    "dtype": np.float64,
}
```

Dictionary này chứa các tham số **chung** cho cả BoW và TF-IDF, để hai phương pháp được so sánh trên cùng quy tắc từ vựng ban đầu.

- `token_pattern`: cách nhận diện token.
- `min_df=2`: loại term chỉ có trong một tài liệu ở lần fit.
- `ngram_range=(1,1)`: **giá trị ban đầu**, sẽ được `GridSearchCV` thay bằng `(1,2)` ở cấu hình khác khi thử.
- `lowercase=True`: chuyển văn bản thành chữ thường trước khi tách token.
- `dtype=np.float64`: kiểu số ma trận.

### 7.3. Vì sao có `**common`?

```python
vectorizer = CountVectorizer(**common)
```

`**common` **mở rộng dictionary thành các tham số có tên**. Tương đương về ý tưởng với:

```python
vectorizer = CountVectorizer(
    token_pattern=TOKEN_PATTERN,
    min_df=2,
    ngram_range=(1, 1),
    lowercase=True,
    dtype=np.float64,
)
```

Nó không nhân số, không “trộn đặc trưng” và không gọi `fit()`.

### 7.4. Nhánh BoW

```python
if feature == "BoW":
    vectorizer = CountVectorizer(**common)
```

Tạo một đối tượng chuyển văn bản thành **ma trận số đếm**. Lệnh này **chỉ khởi tạo**.

### 7.5. Nhánh TF-IDF

```python
elif feature == "TF-IDF":
    vectorizer = TfidfVectorizer(
        **common, sublinear_tf=True, smooth_idf=True, norm="l2"
    )
```

Tạo bộ vector hóa TF-IDF với **ba tham số bổ sung**:

- `sublinear_tf=True`: khi một term xuất hiện `c` lần (`c>0`), phần TF được chuyển thành `1 + ln(c)` thay vì giữ nguyên c.
- `smooth_idf=True`: thêm 1 vào tử và mẫu trong IDF, giúp làm mịn.
- `norm="l2"`: chuẩn hóa **mỗi vector bình luận** theo độ dài Euclid (nếu khác zero).

`TfidfVectorizer` còn có các mặc định không ghi rõ trong code, chẳng hạn `use_idf=True`, `binary=False`, `stop_words=None`, `analyzer='word'`.

### 7.6. Nhánh không hợp lệ

```python
else:
    raise ValueError(f"Đặc trưng không hỗ trợ: {feature}")
```

Dừng với lỗi rõ ràng, giúp tránh việc Notebook âm thầm chạy một tên đặc trưng sai.

### 7.7. Vì sao trả `{"vec__ngram_range": NGRAMS}`?

`experiments.py` tạo:

```python
pipe = Pipeline([
    ("vec", vectorizer),
    ("clf", classifier),
])
```

Tên bước vector hóa là **`vec`**. Trong scikit-learn, `"vec__ngram_range"` nghĩa là “đổi `ngram_range` thuộc bước có tên `vec`”.

`GridSearchCV` dùng key này để tạo những cấu hình thử khác nhau. Đây không phải tên thuộc tính Python bình thường; `__` là quy tắc đặt tên tham số lồng trong Pipeline.

### 7.8. Kết luận một dòng

**`make_vectorizer()` chọn lớp thư viện + thiết lập thông số + đưa ra hai phương án n-gram; việc học vocabulary/IDF và tạo ma trận xảy ra sau đó khi `Pipeline.fit` được gọi.**

---

## 8. Bên trong `CountVectorizer`: BoW được tính ra sao?

### 8.1. Ý tưởng lý thuyết

Tạo bộ đặc trưng từ Train, rồi đếm số lần mỗi đặc trưng có mặt trong mỗi bình luận.

Với văn bản `d` và term `t`:

\[
BoW(d,t)=count(t,d).
\]

Ví dụ vocabulary `["nhận", "lịch", "thi"]`:

- `"nhận nhận"` → `[2, 0, 0]`.
- `"lịch thi"` → `[0, 1, 1]`.

**BoW không cần nhãn** để tạo vector. Nhãn chỉ được dùng ở `classifier.fit(X_vector, y)`.

### 8.2. Thứ tự hoạt động của lớp scikit-learn

Khi gọi:

```python
bow = CountVectorizer(**common)
X_bow = bow.fit_transform(train_documents)
```

Thư viện về mặt logic thực hiện:

1. Nhận danh sách chuỗi token, chuyển chữ thường nếu cần.
2. Tách token theo `token_pattern`.
3. Tạo unigram/bigram theo `ngram_range`.
4. Đếm những term xuất hiện ở từng văn bản và hình thành từ vựng ứng viên.
5. Tính **document frequency** rồi loại term không đạt `min_df=2`.
6. Ấn định thứ tự cột trong `vocabulary_`.
7. Trả ma trận thưa, mỗi hàng là một văn bản, mỗi cột là term.

Trong tài liệu `PHI_HIEU_4_LOP_SCIKIT_LEARN.md`, đường gọi mã nguồn của scikit-learn 1.6.1 là `CountVectorizer.fit_transform` → `_count_vocab` và `_limit_features` (cùng module `sklearn/feature_extraction/text.py`). `build_analyzer()` kết hợp tiền xử lý, tokenizer và sinh n-gram. Đây là mô tả nhánh mã nguồn mà **cấu hình của nhóm đang sử dụng**, không phải mọi khả năng của CountVectorizer.

### 8.3. `fit`, `fit_transform`, `transform`

```python
bow.fit(train_docs)
```

- Học vocabulary từ `train_docs`.
- Trả về chính đối tượng `bow` đã fit.
- Không trả ra ma trận của `train_docs` cho biến được gán (nếu chỉ gọi như trên).

```python
X_train_bow = bow.fit_transform(train_docs)
```

- Fit vocabulary **và** trả ma trận số đếm của `train_docs`.
- Đây là cách được Pipeline dùng trên **fold train**.

```python
X_valid_bow = bow.transform(valid_docs)
```

- Chỉ đếm theo **vocabulary đã học**.
- Không thêm cột, không fit lại, không lọc `min_df` lại trên Validation.

### 8.4. Thuộc tính quan trọng sau fit

```python
bow.vocabulary_              # dict term -> column_index
bow.get_feature_names_out()  # tên term theo thứ tự cột
X_train_bow.shape            # (số câu, số đặc trưng)
```

Chỉ số trong `vocabulary_` quyết định nghĩa của **từng cột**. Bộ phân loại sau này gán trọng số theo chính những cột đó.

### 8.5. Ma trận thưa CSR là gì?

Giả sử 30.759 câu và rất nhiều term, mỗi câu chỉ chứa một phần nhỏ. Hầu hết ô trong ma trận là 0. CSR (*Compressed Sparse Row*) lưu các giá trị khác 0 và các chỉ số cần thiết, tiết kiệm bộ nhớ hơn so với lưu toàn bộ ô.

Ví dụ:

```python
print(type(X_train_bow))  # Thường là scipy.sparse.csr_matrix
print(X_train_bow.shape)
print(X_train_bow.nnz)    # số ô khác 0
```

**Không dùng `X_train_bow.toarray()` cho toàn bộ dataset lớn**, vì chuyển thành ma trận đặc có thể sử dụng rất nhiều RAM. `.toarray()` chỉ phù hợp để kiểm tra vài câu ví dụ.

---

## 9. Bên trong `TfidfVectorizer`: từ đếm từ sang trọng số

### 9.1. Tại sao không chỉ dùng BoW?

BoW thấy term `"xin"` xuất hiện 3 lần thì cho trọng số 3. Nhưng nó chưa xét rằng `"xin"` có thể rất phổ biến trong toàn bộ bình luận, còn một cụm như `"nhận_làm"` có thể đặc trưng hơn cho một kiểu nội dung.

TF-IDF kết hợp:

- **TF:** term xuất hiện thế nào trong **một câu**.
- **IDF:** term phổ biến thế nào trong **tập câu dùng để fit**.

Nó tạo trọng số đặc trưng, **không tự học nhãn Spam/Ham**.

### 9.2. TF trong đúng cấu hình của nhóm

Do `sublinear_tf=True`, với term đã xuất hiện `c ≥ 1` lần:

\[
TF(t,d)=1+\ln(c).
\]

| Số lần xuất hiện `c` | TF dưới tuyến tính |
|---:|---:|
| 1 | 1,0000 |
| 2 | 1,6931 |
| 3 | 2,0986 |
| 4 | 2,3863 |

Nếu c=0 thì term có giá trị 0, không áp biểu thức log(0).

**Ý nghĩa:** một từ lặp ba lần có trọng số tăng nhưng không lớn gấp ba lần từ xuất hiện một lần. Điều này hạn chế tác động của việc lặp từ liên tục.

### 9.3. IDF đúng cấu hình `smooth_idf=True`

Với N văn bản trong phần Train được fit và `df(t)` văn bản chứa term:

\[
IDF(t)=1+\ln\left(\frac{N+1}{df(t)+1}\right).
\]

- `df` khác `count` trong một câu.
- IDF được học từ **các văn bản của fold train**, chưa bao gồm fold validation hoặc Test.
- Term có mặt ở càng nhiều câu thì IDF càng nhỏ, **nhưng không nhỏ hơn 1** trong cấu hình này.
- Với `min_df=2`, term chỉ xuất hiện trong một văn bản sẽ bị loại trước khi lưu `idf_`.

### 9.4. TF × IDF, sau đó chuẩn hóa L2

Ban đầu, với term xuất hiện:

\[
raw\_tfidf(t,d)=TF(t,d)\times IDF(t).
\]

Sau đó, `norm="l2"` chia mỗi ô của hàng cho độ dài:

\[
\|v_d\|_2=\sqrt{\sum_j v_{d,j}^{\,2}}.
\]

\[
TFIDF(d,j)=\frac{v_{d,j}}{\|v_d\|_2}
\]

(với hàng không toàn 0; nếu toàn 0, thư viện giữ nguyên hàng toàn 0).

**Không nhầm:** Chuẩn hóa L2 tác động **trên từng vector của một bình luận**, không phải lấy tổng của toàn dataset.

### 9.5. Các bước thư viện thực hiện

```python
tfidf = TfidfVectorizer(**common, sublinear_tf=True,
                         smooth_idf=True, norm="l2")
X_train_tfidf = tfidf.fit_transform(train_docs)
```

Về logic của scikit-learn:

1. Dùng cơ chế **CountVectorizer** để tách token, tạo n-gram, học vocabulary, lọc `min_df=2` và xây ma trận đếm.
2. Dùng thành phần **TfidfTransformer** để tính và lưu `idf_` từ các cột của ma trận đếm Train.
3. Trên những ô khác 0, áp dụng TF dưới tuyến tính rồi nhân IDF tương ứng.
4. Chuẩn hóa L2 theo từng hàng.
5. Trả ma trận thưa TF-IDF.

Theo tài liệu scikit-learn 1.6.1 nhóm gửi, `TfidfVectorizer.fit_transform` dùng bước đếm từ lớp cha `CountVectorizer` rồi gọi transformer TF-IDF bên trong. Không phải nhóm tự viết lại vòng lặp tính những công thức này.

### 9.6. Thuộc tính sau fit

```python
tfidf.vocabulary_             # term -> số cột
tfidf.get_feature_names_out() # tên cột
tfidf.idf_                   # IDF theo thứ tự các cột
X_train_tfidf.shape
```

**Quan trọng:** Khi gọi `tfidf.transform(test_docs)`, thư viện sử dụng `vocabulary_` và `idf_` đã học, **không tính IDF mới dựa trên test**.

### 9.7. Ba nhận định cần phân biệt

- **IDF cao** nghĩa là term có trong tương đối ít văn bản của Train, **không có nghĩa** term chắc chắn phân biệt Spam/Ham tốt.
- **TF-IDF không tạo xác suất Spam**. Nó chỉ tạo ma trận số không âm; NB/SVM mới dùng nhãn để học.
- **Vì L2**, giá trị TF-IDF cuối cùng của term còn phụ thuộc các term khác trong **cùng bình luận**. Không nên chỉ nhân TF×IDF rồi tưởng đó là kết quả cuối.

---

## 10. VÍ DỤ CỤ THỂ ĐÃ KIỂM TRA — 5 BÌNH LUẬN, TỪ ĐIỂN, VECTOR

> **Ví dụ hoàn toàn tự dựng để học.** Không phải 5 dòng lấy từ CSV thật. Các giá trị ma trận dưới đây đã được kiểm tra bằng `CountVectorizer` và `TfidfVectorizer` với các tham số tương ứng.

### 10.1. Tập Train minh họa

```python
docs = [
    "xin xin lịch thi",     # D0
    "xin lịch học",         # D1
    "nhận làm thuê",        # D2
    "nhận làm đồ án",       # D3
    "xin lịch thi",         # D4
]
```

Bước này chưa cần nhãn, vì ta đang nghiên cứu **trích chọn đặc trưng**. Sau này NB/SVM mới cần `y` đi kèm.

### 10.2. Đếm DF để áp dụng `min_df=2`

| Term (unigram) | Xuất hiện ở câu nào? | DF | Giữ? |
|---|---|---:|---|
| `xin` | D0, D1, D4 | 3 | Có |
| `lịch` | D0, D1, D4 | 3 | Có |
| `thi` | D0, D4 | 2 | Có |
| `nhận` | D2, D3 | 2 | Có |
| `làm` | D2, D3 | 2 | Có |
| `học` | D1 | 1 | Không |
| `thuê` | D2 | 1 | Không |
| `đồ` | D3 | 1 | Không |
| `án` | D3 | 1 | Không |

Dù `xin` xuất hiện **2 lần** trong D0, D0 vẫn chỉ đóng góp **1 văn bản** cho DF. Term `xin` có DF=3 (D0, D1, D4).

### 10.3. BoW với `ngram_range=(1,1)`

Tên cột thực tế sau fit, theo thứ tự trong ví dụ:

```python
["làm", "lịch", "nhận", "thi", "xin"]
```

Khi đó ma trận `X_bow` là:

| Bình luận | làm | lịch | nhận | thi | xin |
|---|---:|---:|---:|---:|---:|
| D0: xin xin lịch thi | 0 | 1 | 0 | 1 | **2** |
| D1: xin lịch học | 0 | 1 | 0 | 0 | 1 |
| D2: nhận làm thuê | 1 | 0 | 1 | 0 | 0 |
| D3: nhận làm đồ án | 1 | 0 | 1 | 0 | 0 |
| D4: xin lịch thi | 0 | 1 | 0 | 1 | 1 |

**Kích thước:** `X_bow.shape == (5, 5)`.

D2 và D3 có **cùng vector BoW** với vocabulary này, dù nội dung khác nhau, vì `thuê`, `đồ`, `án` đều không vượt qua min_df. Đây là ví dụ trực quan cho việc biểu diễn BoW có thể làm mất một phần thông tin.

### 10.4. TF-IDF với `ngram_range=(1,1)`

Vì N=5, IDF của từng term giữ lại:

- DF=3 (như `xin`, `lịch`): `1 + ln(6/4) ≈ 1,4055`.
- DF=2 (như `làm`, `nhận`, `thi`): `1 + ln(6/3) ≈ 1,6931`.

Ở D0 `"xin xin lịch thi"`:

- `xin`: count=2 → TF=`1+ln(2)≈1,6931`, IDF≈1,4055 → TF×IDF≈2,3797.
- `lịch`: count=1 → TF=1, IDF≈1,4055 → TF×IDF≈1,4055.
- `thi`: count=1 → TF=1, IDF≈1,6931 → TF×IDF≈1,6931.

Sau chuẩn hóa L2, vector D0 theo thứ tự cột `làm,lịch,nhận,thi,xin` xấp xỉ:

```text
[0, 0.4336, 0, 0.5224, 0.7342]
```

Với cùng năm câu, ma trận TF-IDF làm tròn 4 chữ số:

| Bình luận | làm | lịch | nhận | thi | xin |
|---|---:|---:|---:|---:|---:|
| D0 | 0 | 0,4336 | 0 | 0,5224 | 0,7342 |
| D1 | 0 | 0,7071 | 0 | 0 | 0,7071 |
| D2 | 0,7071 | 0 | 0,7071 | 0 | 0 |
| D3 | 0,7071 | 0 | 0,7071 | 0 | 0 |
| D4 | 0 | 0,5383 | 0 | 0,6485 | 0,5383 |

Giữa BoW và TF-IDF, **cột đặc trưng không tự thay đổi tên chỉ vì thuật toán đổi** (nếu các cấu hình lọc từ và n-gram giống nhau). Điều khác là **giá trị trong các ô** và cách thuật toán phía sau học từ chúng.

### 10.5. Nếu dùng `(1,2)` thì điều gì thay đổi?

Với cùng năm câu, một số bigram có DF≥2:

| Bigram | Có ở | DF | Giữ? |
|---|---|---:|---|
| `xin lịch` | D0, D1, D4 | 3 | Có |
| `lịch thi` | D0, D4 | 2 | Có |
| `nhận làm` | D2, D3 | 2 | Có |
| `lịch học` | D1 | 1 | Không |
| `làm thuê` | D2 | 1 | Không |

Vocabulary của ví dụ bây giờ có **8 cột**:

```python
["làm", "lịch", "lịch thi", "nhận", "nhận làm", "thi", "xin", "xin lịch"]
```

Ma trận BoW:

| Bình luận | làm | lịch | lịch thi | nhận | nhận làm | thi | xin | xin lịch |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| D0 | 0 | 1 | 1 | 0 | 0 | 1 | 2 | 1 |
| D1 | 0 | 1 | 0 | 0 | 0 | 0 | 1 | 1 |
| D2 | 1 | 0 | 0 | 1 | 1 | 0 | 0 | 0 |
| D3 | 1 | 0 | 0 | 1 | 1 | 0 | 0 | 0 |
| D4 | 0 | 1 | 1 | 0 | 0 | 1 | 1 | 1 |

**Kích thước:** `(5,8)` thay vì `(5,5)`.

Vậy lựa chọn n-gram làm thay đổi **vocabulary, số cột và cách biểu diễn mỗi câu**. Vì thế `ngram_range` là tham số cần CV lựa chọn, không thể coi nó chỉ là tùy chọn hiển thị.

### 10.6. Chuỗi mới đi vào `transform()` như thế nào?

Sau khi fit BoW unigram trên năm câu ở trên:

```python
new_doc = ["xin lịch mới nhận làm"]
X_new = bow.transform(new_doc)
```

- `mới` **không có** trong vocabulary → bỏ qua khi chuyển thành vector.
- Các term `xin`, `lịch`, `nhận`, `làm` có → đếm bình thường.

Vector theo `["làm","lịch","nhận","thi","xin"]`:

```text
[1, 1, 1, 0, 1]
```

Hình dạng `(1,5)` vẫn đúng **5 cột cũ**. Đó là nguyên tắc bắt buộc để bộ phân loại đã fit đọc đúng trọng số.

---


### 10.7. TF-IDF với `(1,2)` — thêm bigram thì trọng số thay đổi ra sao?

Trong cùng năm bình luận minh họa, `TfidfVectorizer` với `(1,2)` tạo **8 cột** giống tên cột của BoW `(1,2)` nhưng tính số thực. Tôi đã chạy code kiểm chứng trên scikit-learn; các số sau làm tròn bốn chữ số:

| Văn bản | làm | lịch | lịch thi | nhận | nhận làm | thi | xin | xin lịch |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| D0 | 0 | 0,3588 | 0,4322 | 0 | 0 | 0,4322 | 0,6074 | 0,3588 |
| D1 | 0 | 0,5774 | 0 | 0 | 0 | 0 | 0,5774 | 0,5774 |
| D2 | 0,5774 | 0 | 0 | 0,5774 | 0,5774 | 0 | 0 | 0 |
| D3 | 0,5774 | 0 | 0 | 0,5774 | 0,5774 | 0 | 0 | 0 |
| D4 | 0 | 0,4116 | 0,4959 | 0 | 0 | 0,4959 | 0,4116 | 0,4116 |

**Vì sao trọng số của `xin` ở D0 lại giảm từ 0,7342 ở `(1,1)` xuống 0,6074 ở `(1,2)`?** Bởi khi thêm các bigram, vector D0 có thêm các giá trị khác 0. L2 chuẩn hóa trên **cả hàng mới (8 cột)** nên mẫu số độ dài vector thay đổi; kể cả một unigram không đổi TF và IDF vẫn có trọng số *sau L2* khác trước.

**Ví dụ tính D0 ở `(1,2)`** (số trước L2):

- `xin`: TF = `1 + ln(2) = 1,6931`; IDF = `1 + ln(6/4) = 1,4055`; tích ≈ `2,3797`.
- `lịch` và `xin lịch`: count 1, DF 3 nên tích ≈ `1,4055` mỗi term.
- `thi` và `lịch thi`: count 1, DF 2 nên tích ≈ `1,6931` mỗi term.
- Chuẩn L2 của năm giá trị khác 0 xấp xỉ `3,918`.
- Sau chia cho chuẩn L2, `xin ≈ 0,6074`, `lịch ≈ 0,3588`, `thi ≈ 0,4322`, `xin lịch ≈ 0,3588`, `lịch thi ≈ 0,4322`.

**Kết luận:** `(1,2)` vừa thêm cột, vừa có thể làm thay đổi *mọi trọng số TF-IDF đã chuẩn hóa* của câu; không phải chỉ gắn các cột mới vào ma trận `(1,1)` rồi để nguyên số cũ.

---

## 11. CODE THỬ NHỎ CÓ THỂ CHẠY ĐỂ TỰ QUAN SÁT

> Chạy trong Python/scikit-learn. Đây là minh họa độc lập, không cần file CSV hay Notebook thật.

```python
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

DOCS = [
    "xin xin lịch thi", "xin lịch học",
    "nhận làm thuê", "nhận làm đồ án", "xin lịch thi",
]

common = {
    "token_pattern": r"(?u)\b\w+\b|[^\w\s]",
    "min_df": 2,
    "ngram_range": (1, 1),
    "lowercase": True,
    "dtype": np.float64,
}

# (A) BAG OF WORDS
bow = CountVectorizer(**common)
X_bow = bow.fit_transform(DOCS)
print("Từ điển BoW:", bow.vocabulary_)
print("Thứ tự cột:", bow.get_feature_names_out())
print("Kích thước:", X_bow.shape)
print("Ma trận nhỏ:")
print(X_bow.toarray())

# (B) TF-IDF
vec_tfidf = TfidfVectorizer(**common, sublinear_tf=True,
                             smooth_idf=True, norm="l2")
X_tfidf = vec_tfidf.fit_transform(DOCS)
print("Thứ tự cột TF-IDF:", vec_tfidf.get_feature_names_out())
print("IDF:", np.round(vec_tfidf.idf_, 4))
print("TF-IDF:")
print(np.round(X_tfidf.toarray(), 4))

# (C) KHÔNG FIT LẠI KHI GẶP CÂU MỚI
new = ["xin lịch mới nhận làm"]
print("BoW câu mới:", bow.transform(new).toarray())
print("TF-IDF câu mới:", np.round(vec_tfidf.transform(new).toarray(), 4))

# (D) SO UNIGRAM VỚI UNIGRAM + BIGRAM
bow_12 = CountVectorizer(**{**common, "ngram_range": (1, 2)})
X_bow_12 = bow_12.fit_transform(DOCS)
print("Tên cột khi có bigram:", bow_12.get_feature_names_out())
print("Shape unigram:", X_bow.shape, "/ shape (1,2):", X_bow_12.shape)
```

**Bài tập tự kiểm tra:** Trước khi chạy, hãy dự đoán vì sao `"thuê"` bị loại dù nó có thể là tín hiệu Spam; vì sao `"xin"` có số đếm 2 ở D0 nhưng DF chỉ tăng 1 cho D0; và vì sao số cột tăng từ 5 lên 8 khi bật bigram.

---

### 11.1. Xem tận mắt analyzer trước khi tạo vector

Đoạn sau **chưa huấn luyện mô hình**, chỉ giúp bạn thấy `CountVectorizer` biến *một chuỗi* thành n-gram nào. Sử dụng cấu hình đúng bài nhưng kiểm tra analyzer trước khi `min_df` lọc cột:

```python
from sklearn.feature_extraction.text import CountVectorizer

v11 = CountVectorizer(token_pattern=r"(?u)\b\w+\b|[^\w\s]",
                      ngram_range=(1,1), min_df=2)
v12 = CountVectorizer(token_pattern=r"(?u)\b\w+\b|[^\w\s]",
                      ngram_range=(1,2), min_df=2)

sentence = "xin lịch_thi !"
print(v11.build_analyzer()(sentence))
# ['xin', 'lịch_thi', '!']
print(v12.build_analyzer()(sentence))
# ['xin', 'lịch_thi', '!', 'xin lịch_thi', 'lịch_thi !']
```

Bạn có thể thử đổi câu có từ lặp, dấu câu, từ nối `lịch_thi`, hoặc `ngram_range`; sau đó đối chiếu với các cột cuối cùng của `get_feature_names_out()` sau `fit_transform(DOCS)` để thấy sự khác nhau giữa **ngram ứng viên** và **ngram được giữ lại sau `min_df`**.

---

## 12. ĐỌC SÂU MÃ THƯ VIỆN SCIKIT-LEARN — TỪ CÂU CHỮ TỚI MA TRẬN SỐ

> **Đây là phần bổ sung trọng tâm theo yêu cầu.** Tôi đối chiếu đường xử lý trong tài liệu nhóm `PHI_HIEU_4_LOP_SCIKIT_LEARN.md` với các tham số thực tế ở `text_features(1).py`. Tài liệu nhóm dẫn tới **scikit-learn 1.6.1 trong `.venv`**, còn môi trường kiểm tra ví dụ của tôi dùng phiên bản cài tại đây; các hàm/mô hình số trong ví dụ đã kiểm chứng, nhưng **số dòng mã nguồn thư viện có thể thay đổi theo phiên bản**.
>
> Những đoạn đánh dấu "trích rút gọn" là **một phần đường xử lý do tài liệu nhóm cung cấp**, không phải toàn bộ phương thức trong thư viện. Những đoạn đánh dấu "mô phỏng" là **code giải thích** do tôi viết để bạn nhìn được thuật toán, không phải nguyên văn code scikit-learn.

### 12.1. Bốn thứ cần phân biệt khi đọc code thư viện

1. **Đối tượng chưa fit:** `CountVectorizer(...)`, `TfidfVectorizer(...)` mới giữ cấu hình, chưa có `vocabulary_`.
2. **Học ở `fit` / `fit_transform`:** học *vocabulary*, lọc cột; riêng TF-IDF còn học `idf_`.
3. **Đầu ra `fit_transform`:** ma trận số `X` (CSR/sparse) có `số bình luận × số đặc trưng`.
4. **Áp dụng ở `transform`:** dùng trạng thái đã học trên câu mới, không thêm cột/không tính lại IDF.

Không nhầm `fit` của **vectorizer** với `fit` của **NB/SVM**: vectorizer học cách biểu diễn *mà không cần nhãn*, classifier mới học mối liên hệ giữa vector và nhãn 0/1.

### 12.2. `CountVectorizer`: từ một câu đến các token/ngram được đếm như thế nào?

**Đường gọi thực tế (nhánh mặc định `analyzer='word'` mà bài này dùng):**

```text
CountVectorizer.fit_transform(raw_documents)
  ├── build_analyzer()
  │    ├── build_preprocessor(): lowercase=True, tiền xử lý chuỗi
  │    ├── build_tokenizer(): biên dịch TOKEN_PATTERN, dùng findall
  │    └── _word_ngrams(): thêm unigram/bigram theo ngram_range
  ├── _count_vocab(...): duyệt từng văn bản → đếm từng ngram → gán chỉ số cột
  │    └── tạo ma trận thưa CSR từ vị trí cột + số đếm
  ├── _sort_features(): sắp xếp lại thứ tự cột ổn định
  ├── _limit_features(): lọc đặc trưng có document frequency < min_df
  └── lưu vocabulary_ → trả về ma trận X_bow
```

Tên/phạm vi này xuất phát từ **mục 3** của tài liệu `PHI_HIEU_4_LOP_SCIKIT_LEARN.md` (trong `.venv` 1.6.1, `_count_vocab` khoảng dòng 1246, `fit_transform` khoảng dòng 1331, `transform` khoảng dòng 1398, `_word_ngrams` khoảng dòng 238 của `sklearn/feature_extraction/text.py`).

#### 12.2.1. `build_preprocessor()` làm gì?

Vì bài truyền `lowercase=True`, nó chuyển `"XIN lịch_thi!"` thành `"xin lịch_thi!"`. Ở đây dữ liệu vốn đã qua tiền xử lý tiếng Việt trong Notebook, nên không phải một bước *tách từ tiếng Việt lần hai*. Nó chỉ là phần xử lý chuỗi theo cấu hình của vectorizer.

#### 12.2.2. `build_tokenizer()` làm gì với `TOKEN_PATTERN`?

Regex của bạn:

```python
TOKEN_PATTERN = r"(?u)\b\w+\b|[^\w\s]"
```

- `(?u)`: dùng quy tắc Unicode trong regex Python.
- `\b\w+\b`: lấy một cụm ký tự từ (chữ/số/dấu `_`) có ranh giới từ; kể cả token một ký tự.
- `|`: hoặc.
- `[^\w\s]`: lấy từng ký hiệu đơn lẻ không phải ký tự từ hoặc khoảng trắng, chẳng hạn `!`, `.`, `?`.

Ví dụ:

```python
text = "Xin lịch_thi!!!"
# sau lowercase + regex: ['xin', 'lịch_thi', '!', '!', '!']
```

Ở đây **`lịch_thi` là một token** vì `_` thuộc `\w`. Không được nhầm `lịch_thi` với bigram `"lịch thi"` tạo ra bởi `ngram_range` ở bước sau. Tokenizer tiếng Việt đã tách/gộp từ trước; vectorizer chỉ đọc chuỗi token.

#### 12.2.3. `_word_ngrams()` tạo unigram/bigram ra sao?

Ví dụ token:

```python
['xin', 'lịch_thi', '!']
```

- `(1,1)` trả: `['xin', 'lịch_thi', '!']`.
- `(1,2)` trả: `['xin','lịch_thi','!','xin lịch_thi','lịch_thi !']`.

Thư viện duyệt các vị trí kề nhau và ghép bằng dấu cách; nó **không tự xét các cặp không kề nhau** như `"xin !"` trong ví dụ này.

#### 12.2.4. `_count_vocab()` — đây là chỗ dữ liệu thật biến thành số đếm

**Trích rút gọn từ đường xử lý trong tài liệu nhóm:**

```python
for doc in raw_documents:
    feature_counter = {}
    for feature in analyze(doc):
        feature_idx = vocabulary[feature]  # tra tên term -> cột
        if feature_idx not in feature_counter:
            feature_counter[feature_idx] = 1
        else:
            feature_counter[feature_idx] += 1
    j_indices.extend(feature_counter.keys())
    values.extend(feature_counter.values())
    indptr.append(len(j_indices))
```

Ở đây cần hiểu **từng biến**:

| Biến | Đóng vai trò gì? | Ví dụ |
|---|---|---|
| `doc` | Một bình luận văn bản | `"xin xin lịch thi"` |
| `analyze(doc)` | Tách token và tạo n-gram | `['xin','xin','lịch','thi']` nếu `(1,1)` |
| `vocabulary` | Ánh xạ tên đặc trưng sang chỉ số cột | `{'xin':4,'lịch':1,'thi':3}` |
| `feature_idx` | Cột của đặc trưng hiện tại | `xin -> 4` |
| `feature_counter` | Bộ đếm **riêng bình luận** | `{4:2,1:1,3:1}` |
| `j_indices` | Danh sách chỉ số cột có giá trị khác 0 | `[4,1,3,...]` |
| `values` | Giá trị đếm đi với cột | `[2,1,1,...]` |
| `indptr` | Ranh giới nơi mỗi hàng bắt đầu/kết thúc | `[0,3,5,...]` |

**Mô phỏng rất nhỏ:** Nếu vocabulary `{'xin':0, 'lịch':1, 'thi':2}` và bình luận `"xin xin lịch thi"`, vòng lặp thấy `xin` hai lần, nên `feature_counter={0:2,1:1,2:1}`. Đây chính là hàng `[2,1,1]` theo thứ tự `xin,lịch,thi`.

**Tại thời điểm fit**, `vocabulary` được phép mở rộng khi gặp một đặc trưng mới. **Tại thời điểm transform**, bộ từ vựng cố định; gặp một từ không có trong vocabulary thì nhánh `KeyError` bỏ qua. Không thể thêm cột Test rồi đưa ma trận có số cột khác vào mô hình đã học.

#### 12.2.5. Vì sao còn `_limit_features()` và `_sort_features()`?

Ở giai đoạn đầu, `_count_vocab` có thể thu thập cả các term chỉ xuất hiện một lần. Sau đó `fit_transform` tính document frequency của các term: đếm **bao nhiêu bình luận chứa term**, không phải term lặp bao nhiêu lần.

**Trích rút gọn từ tài liệu nhóm:**

```python
vocabulary, X = self._count_vocab(raw_documents, self.fixed_vocabulary_)
if not self.fixed_vocabulary_:
    n_doc = X.shape[0]
    min_doc_count = min_df if isinstance(min_df, Integral) else min_df * n_doc
    X = self._limit_features(
        X, vocabulary, max_doc_count, min_doc_count, max_features
    )
    self.vocabulary_ = vocabulary
return X
```

Ở bài này `min_df=2`, vì vậy:

- `xin` nằm trong D0, D1, D4 → **DF=3** → giữ.
- `thi` nằm trong D0, D4 → **DF=2** → giữ.
- `thuê` chỉ ở D2 → **DF=1** → bỏ.
- `xin` lặp 2 lần ở D0 vẫn chỉ đóng góp **1** tài liệu cho DF.

Kết quả là `vocabulary_` chỉ chứa các term thỏa điều kiện. Cách sắp tên cột là chi tiết nội bộ, không nên tự suy số thứ tự cột: hãy lấy bằng `get_feature_names_out()`.

#### 12.2.6. Vậy ma trận CSR được tạo như thế nào?

Với ma trận nhỏ:

```text
           xin  lịch  thi
D0          2     1    1
D1          1     1    0
D2          0     0    0
```

Cách lưu thưa (minh họa theo thứ tự cột chuẩn):

```text
data     = [2,1,1, 1,1]
indices  = [0,1,2, 0,1]
indptr   = [0, 3, 5, 5]
shape    = (3,3)
```

- Hàng D0 lấy phần dữ liệu giữa `indptr[0]` và `indptr[1]` → ba ô khác 0.
- Hàng D1 giữa `indptr[1]` và `indptr[2]` → hai ô khác 0.
- Hàng D2 có `indptr[2] == indptr[3]` → không có ô khác 0.

Với hàng chục nghìn bình luận và rất nhiều từ, lưu cả số 0 sẽ tốn bộ nhớ; ma trận CSR hiệu quả hơn. `X.toarray()` chỉ nên dùng cho **ví dụ nhỏ**, không tùy tiện chuyển toàn bộ dataset sang ma trận đặc.

#### 12.2.7. `fit`, `fit_transform`, `transform`, `get_feature_names_out` khác nhau ra sao?

```python
vec = CountVectorizer(token_pattern=TOKEN_PATTERN, min_df=2)
vec.fit(train_docs)                    # học vocabulary_; không trả ma trận số
X_train = vec.transform(train_docs)    # tạo ma trận dùng vocab đã học
# hoặc gộp hai bước: X_train = vec.fit_transform(train_docs)
X_valid = vec.transform(valid_docs)    # KHÔNG học vocabulary_ mới
names = vec.get_feature_names_out()    # tên cột, đúng thứ tự
```

`CountVectorizer.fit()` trong thư viện sử dụng phần lớn logic `fit_transform()` nhưng bỏ ma trận trả về; vì vậy Ctrl+Click vào class rồi chỉ tìm `fit` có thể khiến bạn chưa thấy đường đếm từ nằm ở đâu.

**Bài tự kiểm tra:** Nếu Train đã học vocabulary `['làm','lịch','nhận','thi','xin']`, một câu validation `"xin lịch học mới"` chỉ có `xin` và `lịch` được đếm. Số cột không thay đổi.

### 12.3. `TfidfVectorizer`: chuỗi văn bản biến thành các trọng số cụ thể như thế nào?

Đầu tiên cần hiểu quan hệ kế thừa: `TfidfVectorizer` sử dụng chức năng **đếm từ của `CountVectorizer`**, sau đó dùng **`TfidfTransformer`** để tính IDF và biến đổi ma trận đếm.

**Đường gọi trong đúng cấu hình project:**

```text
TfidfVectorizer.fit_transform(train_docs)
   ├── CountVectorizer.fit_transform(train_docs)
   │       └── tokenize → ngram → vocabulary → min_df → ma trận count CSR
   ├── TfidfTransformer.fit(X_count)
   │       └── đếm DF từng cột → smooth_idf → lưu idf_
   └── TfidfTransformer.transform(X_count)
           ├── sublinear_tf: count thành 1 + ln(count) (cho count > 0)
           ├── nhân IDF của đúng cột
           ├── chuẩn hóa L2 từng hàng
           └── trả X_tfidf (ma trận thưa, cùng tên cột)
```

#### 12.3.1. `TfidfVectorizer.fit_transform()` gọi những hàm gì?

**Trích rút gọn từ tài liệu nhóm về mã scikit-learn 1.6.1:**

```python
X = super().fit_transform(raw_documents)   # CountVectorizer → ma trận số đếm
self._tfidf.fit(X)                         # TfidfTransformer học idf_
return self._tfidf.transform(X, copy=False) # áp TF × IDF, chuẩn hóa
```

- `super()` gọi xử lý của lớp cha `CountVectorizer` để tạo ma trận đếm.
- `_tfidf` là một đối tượng `TfidfTransformer` được dùng nội bộ.
- `.fit(X)` của transformer học trọng số IDF trên **ma trận count của Train**.
- `.transform(X)` trả về ma trận TF-IDF cuối cùng.

**Nghĩa là với TF-IDF, thư viện không bỏ qua BoW**: nó bắt đầu từ số đếm, rồi biến đổi các số đếm ấy.

#### 12.3.2. `TfidfTransformer.fit()` học `idf_` như thế nào?

Trích rút gọn đường xử lý trong tài liệu nhóm:

```python
n_samples, _ = X.shape
# df là số văn bản có giá trị khác 0 ở từng cột
df = _document_frequency(X)
df += float(self.smooth_idf)
n_samples += int(self.smooth_idf)
self.idf_ = np.full_like(df, fill_value=n_samples, dtype=dtype)
self.idf_ /= df
np.log(self.idf_, out=self.idf_)
self.idf_ += 1.0
```

Với `smooth_idf=True` của bài, đây là:

```text
idf(term) = 1 + ln( (1 + N) / (1 + DF(term)) )
```

Ví dụ 5 văn bản, `xin` có ở 3 văn bản, `thi` có ở 2 văn bản:

```text
idf('xin') = 1 + ln(6/4) ≈ 1,4055
idf('thi') = 1 + ln(6/3) ≈ 1,6931
```

- `xin` phổ biến hơn nên IDF nhỏ hơn `thi`.
- `idf_` là vector số dài bằng **số cột sau lọc min_df**.
- Không dùng nhãn spam/ham để tính IDF; chỉ dùng sự phân bố các từ trên tập đang được fit.

#### 12.3.3. `TfidfTransformer.transform()` biến số đếm thế nào?

Trích rút gọn từ tài liệu nhóm:

```python
if self.sublinear_tf:
    np.log(X.data, X.data)
    X.data += 1.0
if hasattr(self, "idf_"):
    X.data *= self.idf_[X.indices]
if self.norm is not None:
    X = normalize(X, norm=self.norm, copy=False)
return X
```

Từng biến trong CSR:

- `X.data` là các giá trị **khác 0** (số đếm) của ma trận.
- `X.indices` cho biết **mỗi giá trị thuộc cột nào**, từ đó chọn `idf_[cột]` để nhân đúng.
- `np.log(X.data)` biến count thành log, rồi cộng 1: `1 + ln(count)`.
- `normalize(X, norm='l2')` chia mỗi hàng cho độ dài L2.

**Vì sao tính log trên `X.data` mà không gặp `log(0)`?** CSR chỉ lưu những ô khác 0 trong `.data`. Những ô vốn bằng 0 không phải đi qua phép `log`, chúng tiếp tục bằng 0.

#### 12.3.4. Đi theo một câu từ đầu tới cuối — ví dụ D0

Tập Train minh họa 5 bình luận ở mục 10. Xét câu D0 `"xin xin lịch thi"` với `(1,1)`, `min_df=2`:

**Bước 1 — CountVectorizer tạo hàng số đếm:** (thứ tự cột `['làm','lịch','nhận','thi','xin']`)

```text
D0: [0, 1, 0, 1, 2]
```

**Bước 2 — TfidfTransformer.fit tính IDF từ toàn bộ năm câu Train:**

```text
idf(lịch) = idf(xin) ≈ 1,4055   (DF 3)
idf(thi)  ≈ 1,6931              (DF 2)
```

**Bước 3 — TF dưới tuyến tính `1+ln(count)`:**

```text
xin:  count=2 → tf=1+ln(2)≈1,6931
lịch: count=1 → tf=1
thi:  count=1 → tf=1
```

**Bước 4 — Nhân TF với IDF:**

```text
lịch = 1 × 1,4055 = 1,4055
thi  = 1 × 1,6931 = 1,6931
xin  = 1,6931 × 1,4055 ≈ 2,3797
```

**Bước 5 — L2 của hàng:**

```text
norm = sqrt(1,4055² + 1,6931² + 2,3797²) ≈ 3,241
```

**Bước 6 — Chia các ô khác 0 cho `norm`:**

```text
TF-IDF(D0) ≈ [0, 0,4336, 0, 0,5224, 0,7342]
```

**Kết quả cuối:** `fit_transform(docs)` trả về ma trận **5 × 5**, không phải một xác suất Spam. Sau này Naive Bayes/SVM mới nhận hàng số này cùng với nhãn thật để học phân loại.

#### 12.3.5. Khi `transform(test_docs)` thì sao?

**Đường gọi:**

```text
TfidfVectorizer.transform(new_docs)
   → CountVectorizer.transform(new_docs)
       → tokenize/ngram → đếm THEO VOCAB CŨ (không thêm cột)
   → TfidfTransformer.transform(X_counts)
       → dùng IDF CŨ → sublinear TF → L2 → ma trận X_new
```

Không hề có `TfidfTransformer.fit()` trên Test. Đây là cách giữ tập Test độc lập và bảo đảm số cột luôn khớp classifier đã học.

#### 12.3.6. Các thuộc tính sau fit và cách tự kiểm tra

```python
vec = TfidfVectorizer(**common, sublinear_tf=True,
                      smooth_idf=True, norm='l2')
X = vec.fit_transform(DOCS)
print(vec.get_feature_names_out())  # tên cột
print(vec.vocabulary_)              # tên -> vị trí cột
print(vec.idf_)                     # IDF theo thứ tự cột
print(X.shape)                      # số bình luận × số đặc trưng
print(X[0].toarray())               # chỉ in hàng D0, an toàn cho ví dụ nhỏ
```

**Không thể** xem `idf_` trước fit, và không được fit lại vectorizer trên riêng `X_test`. Những từ Test chưa có trong vocabulary bị bỏ qua; các bigram Test chưa có cũng bị bỏ qua.

### 12.4. Bảng đối chiếu sâu hai lớp thư viện

| Thành phần | CountVectorizer | TfidfVectorizer |
|---|---|---|
| Đầu vào | Danh sách chuỗi token | Danh sách chuỗi token |
| `fit_transform` học | `vocabulary_` và cột qua `min_df` | `vocabulary_` và `idf_` |
| Tạo token/ngram | Cùng analyzer/tokenizer | Kế thừa đường đếm từ CountVectorizer |
| Ma trận sơ bộ | Số đếm 0,1,2,... | Cũng bắt đầu từ số đếm |
| Bước tính tiếp | Không | `1+ln(tf)` → `× idf` → L2 |
| Đầu ra | Ma trận thưa số đếm | Ma trận thưa số thực không âm |
| `transform` văn bản mới | Tra từ điển cũ, đếm lại | Tra từ điển cũ, đếm, áp `idf_` cũ, L2 |
| Có tự biết Spam/Ham? | Không | Không |
| Cần `y_train` trong lúc học từ điển? | Không | Không |
| Bộ phân loại nhận tiếp | NB hoặc SVM | NB hoặc SVM |

### 12.5. Cách mở đúng mã nguồn trong VS Code (theo tài liệu nhóm)

Theo `PHI_HIEU_4_LOP_SCIKIT_LEARN.md`, đặt Python interpreter là `.venv` của project, Ctrl+Click vào tên lớp, sau đó tìm tên phương thức trong các tệp sau:

| Thành phần | File trong `.venv/Lib/site-packages/sklearn/` | Vị trí tham khảo **scikit-learn 1.6.1** |
|---|---|---|
| `CountVectorizer`, `_count_vocab`, `fit_transform`, `transform` | `feature_extraction/text.py` | Class khoảng 928; `_count_vocab` 1246; `fit_transform` 1331; `transform` 1398 |
| `build_analyzer`, `build_tokenizer`, `_word_ngrams` | `feature_extraction/text.py` | Khoảng 415, 346, 238 |
| `TfidfVectorizer.fit_transform`, `transform` | `feature_extraction/text.py` | Khoảng 2078, 2110 |
| `TfidfTransformer.fit`, `transform` | `feature_extraction/text.py` | Khoảng 1635, 1682 |

**Lưu ý:** Chỉ đọc `__init__` sẽ thấy tham số nhưng không thấy thuật toán tạo vector. Muốn hiểu tính toán phải lần tới `_count_vocab`, `_limit_features`, `TfidfTransformer.fit` và `.transform`.

---

## 13. `make_vectorizer()` được gọi trong `experiments.py` thế nào?

Đoạn liên quan ở `experiments(1).py`:

```python
from pipeline.modeling.text_features import FEATURES, make_vectorizer
from pipeline.modeling.classifiers import MODELS, make_classifier


def make_pipe(feat, model_name, *, seed=42,
              cache=None, n_jobs=2, class_weight=None):
    if feat not in FEATURES or model_name not in MODELS:
        raise ValueError("Đặc trưng/thuật toán không nằm trong cấu hình thực nghiệm")

    vectorizer, vector_grid = make_vectorizer(feat)
    classifier, model_grid = make_classifier(
        model_name, seed=seed, class_weight=class_weight
    )

    pipe = Pipeline([("vec", vectorizer), ("clf", classifier)], memory=cache)
    return pipe, {**vector_grid, **model_grid}, n_jobs
```

### 13.1. Khi gọi `make_pipe("TF-IDF", "SVM")`

1. Gọi `make_vectorizer("TF-IDF")` → tạo `TfidfVectorizer` **chưa fit**, kèm hai `ngram_range`.
2. Gọi `make_classifier("SVM")` → tạo `LinearSVC` **chưa fit**, kèm bốn giá trị `C` của mô hình cơ sở.
3. Gói hai đối tượng vào `Pipeline([('vec', ...), ('clf', ...)])`.
4. Trả `pipe`, `grid` và `n_jobs`.

**Cho đến đây chưa học vocabulary và cũng chưa biết trọng số SVM.**

### 13.2. Khi gọi `fit_experiment(...)`

```python
search = GridSearchCV(
    pipe, grid,
    scoring={"precision": "precision", "recall": "recall", "f1": "f1"},
    cv=folds, n_jobs=jobs,
    refit="f1", error_score="raise"
)
search.fit(X, y)
```

Chính **`search.fit(X, y)`** mới bắt đầu các lượt huấn luyện.

Trong mỗi lượt:

```text
FOLD TRAIN:      câu token → vec.fit_transform() → X_vector → clf.fit(X_vector, y_train_fold)
FOLD VALIDATION: câu token → vec.transform()     → X_vector → clf.predict/score(...)
                                                             ↓
                                                        CV Precision / Recall / F1
```

Sau khi thử xong, `refit="f1"` chọn bộ tham số theo trung bình F1 Spam và tự fit lại toàn bộ Train với cấu hình thắng. `search.best_estimator_` là **Pipeline đã fit**, trong đó vectorizer đã nhớ vocabulary (và IDF nếu TF-IDF), classifier đã nhớ thông tin phân loại.

### 13.3. Có bao nhiêu cấu hình?

Trong `classifiers(1).py`:

```python
NB_ALPHAS = [0.1, 0.3, 1.0]
SVM_CS = [0.1, 0.3, 1.0, 3.0]
```

Kết hợp lưới 2 n-gram của `text_features.py`:

| Tổ hợp | n-gram | Tham số phân loại | Cấu hình / nhánh stopwords |
|---|---:|---:|---:|
| B1 BoW + NB | 2 | 3 alpha | 6 |
| B2 BoW + SVM | 2 | 4 C | 8 |
| B3 TF-IDF + NB | 2 | 3 alpha | 6 |
| B4 TF-IDF + SVM | 2 | 4 C | 8 |
| **Tổng** | | | **28** |

Notebook thử **hai nhánh stopwords** → **56 cấu hình tham số theo tổ hợp cơ sở**, mỗi cấu hình qua 5 fold → **280 lượt fit theo fold**, chưa tính các lần refit mô hình thắng và chiến lược S1/S2/S3. `Pipeline(memory=cache)` cho phép lưu đệm một phần biến đổi khi có thể, chứ không có nghĩa số cấu hình được thử giảm đi.

---


### 13.4. BẰNG CHỨNG CODE: `(1,1)` và `(1,2)` thực sự được thử ở đâu?

**Đường đi của tham số, theo đúng ba file**:

```python
# text_features(1).py
NGRAMS = [(1, 1), (1, 2)]
return vectorizer, {"vec__ngram_range": NGRAMS}

# experiments(1).py, trong make_pipe(...)
vectorizer, vector_grid = make_vectorizer(feat)
classifier, model_grid = make_classifier(model_name, ...)
pipe = Pipeline([("vec", vectorizer), ("clf", classifier)], memory=cache)
return pipe, {**vector_grid, **model_grid}, n_jobs

# experiments(1).py, trong fit_experiment(...)
search = GridSearchCV(
    pipe, grid,
    scoring={"precision": "precision", "recall": "recall", "f1": "f1"},
    cv=folds, n_jobs=jobs, refit="f1", error_score="raise"
)
search.fit(X, y)
```

Bạn thấy file `text_features.py` chỉ **khai báo một vectorizer khởi tạo ở `(1,1)`** và **trả ra lưới có hai giá trị**. `experiments.py` mới đưa lưới đó cho `GridSearchCV`. Thư viện GridSearchCV thực hiện logic giống sau (đây là **mã mô phỏng tư duy**, không phải nguyên văn mã nguồn của thư viện):

```python
for params in ParameterGrid(grid):          # ví dụ vec__ngram_range=(1,2), clf__C=0.3
    scores = []
    for idx_train, idx_valid in folds:       # 5 lượt CV
        candidate = clone(pipe)
        candidate.set_params(**params)       # thay tham số BÊN TRONG bước vec, clf
        candidate.fit(X[idx_train], y[idx_train])
        y_pred = candidate.predict(X[idx_valid])
        scores.append(f1_score(y[idx_valid], y_pred, pos_label=1))
    mean_cv_f1 = np.mean(scores)
    # Ghi kết quả cấu hình này vào bảng để so với mọi cấu hình khác
# Chọn cấu hình có mean_cv_f1 cao nhất, fit lại trên toàn bộ X,y train
```

Ở mỗi lượt `candidate.fit`, bên trong Pipeline chạy:

```text
vec.fit_transform(X_fold_train) → matrix số
clf.fit(matrix_số, y_fold_train)
```

Với validation:

```text
vec.transform(X_fold_valid) → matrix số theo VOCAB FOLD TRAIN
clf.predict(matrix_valid) → nhãn 0/1
```

Chính vì vectorizer **nằm trong Pipeline**, mỗi lựa chọn n-gram học từ điển trên đúng fold train của cấu hình đó. Không tính từ điển một lần trên toàn CSV rồi mới chia fold, vì làm thế có nguy cơ rò rỉ thông tin validation/test.

### 13.5. Từng cấu hình của B4 được thử cụ thể như thế nào?

Với B4 (TF-IDF + SVM), lưới thực tế là:

```python
{
    "vec__ngram_range": [(1,1), (1,2)],
    "clf__C": [0.1, 0.3, 1.0, 3.0],
    "clf__class_weight": [None],
}
```

Tức tám cấu hình:

| Thử | n-gram | C | Cách chấm |
|---:|---|---:|---|
| 1 | `(1,1)` | 0,1 | Fit 5 fold → trung bình CV F1 Spam |
| 2 | `(1,1)` | 0,3 | Fit 5 fold → trung bình CV F1 Spam |
| 3 | `(1,1)` | 1,0 | Fit 5 fold → trung bình CV F1 Spam |
| 4 | `(1,1)` | 3,0 | Fit 5 fold → trung bình CV F1 Spam |
| 5 | `(1,2)` | 0,1 | Fit 5 fold → trung bình CV F1 Spam |
| 6 | `(1,2)` | 0,3 | Fit 5 fold → trung bình CV F1 Spam |
| 7 | `(1,2)` | 1,0 | Fit 5 fold → trung bình CV F1 Spam |
| 8 | `(1,2)` | 3,0 | Fit 5 fold → trung bình CV F1 Spam |

Mỗi cấu hình là một **Pipeline được fit riêng**, không đơn thuần thay `(1,1)` sang `(1,2)` trên một vector có sẵn. `min_df=2` được tính lại cho từng lần fit phù hợp với phần train của fold; với TF-IDF, `idf_` cũng tính lại.

Với B1/B3 thì logic tương tự, chỉ thay `C` bằng `alpha ∈ {0.1,0.3,1.0}`; mỗi tổ hợp có 6 cấu hình. B2 giống B4 ở số cấu hình, nhưng dùng BoW.

### 13.6. Chọn **ở đâu**, output thể hiện **cái gì**?

Trong `experiments.py`, sau `search.fit(X,y)`:

```python
best_index = search.best_index_
cv = search.cv_results_
cv_f1 = float(cv["mean_test_f1"][best_index])
return {
    "CV F1 (spam)": cv_f1,
    "Tham số": search.best_params_,
    "_model": search.best_estimator_,
    # ...
}
```

- `cv_results_`: chứa bảng **mọi cấu hình** đã được thử, gồm `params`, điểm từng fold và các `mean_test_*`.
- `mean_test_f1`: **F1 trung bình validation của 5 fold**, chữ `test` ở đây là thuật ngữ nội bộ của scikit-learn chứ không phải tập Test độc lập của bài.
- `best_index_`: chỉ số hàng có **F1 trung bình cao nhất** theo `refit='f1'`.
- `best_params_`: dictionary chứa **chỉ các tham số của cấu hình thắng**.
- `best_estimator_`: Pipeline thắng **đã tự fit lại trên toàn bộ Train**.

**Notebook cell 13** gom record của các lần thử và hiển thị `record['Tham số']`, nên bạn thấy toàn bộ B1–B4 hiện `(1,2)`; **đó là lựa chọn thắng đã được so với `(1,1)`, không có nghĩa code bỏ qua `(1,1)`**.

### 13.7. Vậy F1 của `(1,1)` chính xác là bao nhiêu?

**Không thể đọc ra chỉ từ bảng output hiện được lưu trong Notebook.** Notebook đã lưu các record chứa tham số thắng và điểm CV của cấu hình thắng, chứ không lưu toàn bộ `cv_results_` của mọi candidate vào các record.

Vì vậy ta biết **cả hai được thử** và **`(1,2)` thắng trong lần chạy đã lưu**, nhưng chưa có căn cứ để khẳng định con số điểm F1 riêng của các cấu hình `(1,1)` nếu không xem `cv_results_` hoặc chạy lại thí nghiệm.

**Nếu muốn hiện đầy đủ kết quả từng `(1,1)` và `(1,2)`** ngay trong Notebook, có thể chạy đoạn dưới với các biến `MAIN_PRE`, `TEXT`, `train_idx`, `y_train`, `CV_FOLDS` đã được tạo ở các cell trước:

```python
import pandas as pd
from sklearn.model_selection import GridSearchCV
from pipeline.modeling.experiments import make_pipe, CV_SCORING

# Ví dụ kiểm tra B4 trên nhánh giữ stopwords.
X_tr = TEXT["giữ stop words"].iloc[train_idx].to_numpy()
pipe, grid, jobs = make_pipe("TF-IDF", "SVM", seed=SEED,
                             cache=cache, n_jobs=N_JOBS)
search = GridSearchCV(pipe, grid, scoring=CV_SCORING,
                      cv=CV_FOLDS, n_jobs=jobs,
                      refit="f1", error_score="raise")
search.fit(X_tr, y_train)

# Đây là mọi cấu hình, không chỉ bộ tham số thắng.
comparison = pd.DataFrame(search.cv_results_)[
    ["param_vec__ngram_range", "param_clf__C", "mean_test_precision",
     "mean_test_recall", "mean_test_f1", "rank_test_f1"]
]
print(comparison.sort_values("rank_test_f1").to_string(index=False))
print("Tham số thắng:", search.best_params_)
```

**Lưu ý thực thi:** Đoạn này **chạy lại việc huấn luyện CV**, không chỉ truy xuất một bảng đã lưu sẵn. Nếu `cache` đã được dọn ở cuối Notebook, tạo `Memory`/`cache` mới hoặc dùng `cache=None`. Cần dataset gốc và môi trường chạy phù hợp; từ các file code/Notebook đính kèm *không có dữ liệu CSV thô ở đây*, nên tôi không tự bịa điểm cho `(1,1)`.

### 13.8. Đếm đúng số lượt thử trong Notebook

- **Mỗi nhánh stopwords:** B1 6 + B2 8 + B3 6 + B4 8 = **28 cấu hình**.
- **Hai nhánh stopwords:** 28 × 2 = **56 cấu hình**.
- **Mỗi cấu hình qua 5 fold:** 56 × 5 = **280 lượt fit trên fold**.
- **`refit='f1'`:** thêm các lượt fit lại cấu hình thắng trên toàn Train (mỗi lần gọi GridSearchCV một lần). `cache` có thể giúp tái sử dụng một số kết quả vector hóa nhưng không làm mất các cấu hình trong lưới.

Điểm số để **chọn n-gram/alpha/C là CV F1 Spam**, không phải điểm Test cuối. Sau cùng, Notebook dùng điểm cao nhất giữa các nhánh và mô hình để xác định `MAIN_PRE='giữ stop words'` và chọn ứng viên B4, trước khi thử S1/S2/S3.

---

## 14. Kết quả THẬT ở cell 13 liên quan đến `text_features.py`

> Các giá trị dưới đây lấy từ **output đã lưu của Notebook 02 bản mới**, không phải đo từ ví dụ 5 câu.

### 14.1. So sánh hai nhánh stopwords bằng CV F1 Spam

| Tổ hợp | Bỏ stopwords | Giữ stopwords |
|---|---:|---:|
| BoW + Naive Bayes | 0,5859 | **0,6077** |
| BoW + SVM | 0,6305 | **0,7087** |
| TF-IDF + Naive Bayes | 0,5835 | **0,5959** |
| TF-IDF + SVM | 0,7562 | **0,7736** |

Notebook chọn nhánh **`"giữ stop words"`** vì cấu hình có CV F1 cao nhất nằm ở nhánh này.

**Không kết luận sai:** Không phải “giữ stopwords luôn tốt hơn trong mọi bài toán”. Đây là kết quả **trong dataset và thiết lập cụ thể của nhóm**.

### 14.2. Kết quả của bốn mô hình cơ sở trên nhánh giữ stopwords

| Mã | Biểu diễn | Bộ phân loại | CV Precision Spam | CV Recall Spam | CV F1 Spam |
|---|---|---|---:|---:|---:|
| B1 | BoW | Naive Bayes | 0,7582 | 0,5153 | 0,6077 |
| B2 | BoW | SVM | 0,7290 | 0,6901 | 0,7087 |
| B3 | TF-IDF | Naive Bayes | 0,8263 | 0,4723 | 0,5959 |
| **B4** | **TF-IDF** | **SVM** | **0,9001** | **0,6806** | **0,7736** |

`B4` là ứng viên cơ sở dẫn đầu theo **CV F1 Spam**. Sau đó nhóm tiếp tục thử **S1/S2/S3** để cải thiện việc phát hiện Spam.

### 14.3. Tham số vector hóa và phân loại thắng theo CV

| Mã | `vec__ngram_range` | Tham số classifier được chọn |
|---|---|---|
| B1 | `(1,2)` | NB `alpha=1.0` |
| B2 | `(1,2)` | SVM `C=1.0`, `class_weight=None` |
| B3 | `(1,2)` | NB `alpha=0.1` |
| B4 | `(1,2)` | SVM `C=0.3`, `class_weight=None` |

**Kết luận đúng cho phần text features:** Trong bốn tổ hợp cơ sở của lần chạy đã lưu, lựa chọn **unigram + bigram** đều thắng lựa chọn chỉ unigram. Điều này cho thấy các cặp từ kế tiếp có ích theo tiêu chí CV F1 của bài hiện tại; không phải bằng chứng `(1,2)` luôn tốt trong mọi tập dữ liệu.

### 14.4. Sau khi vượt qua bước này, điều gì xảy ra?

Cell 16 tiếp tục thử ba chiến lược mất cân bằng trên nhánh TF-IDF + SVM. Notebook cuối cùng chọn **S3 (điều chỉnh ngưỡng)** theo CV F1. Việc **thay ngưỡng SVM không thay đổi cơ chế TF-IDF**: vectorizer đã fit vẫn biến câu thành vector theo vocabulary/IDF cũ.

Điểm Test độc lập là trách nhiệm của `evaluation.py` và sẽ được học trong bài riêng, **không được dùng ngược để chọn vocabulary hay n-gram**.

---

## 15. Validation/Test và dự đoán bình luận mới đi qua vectorizer ra sao?

### 15.1. Trong CV

- Trong mỗi fold, `fit_transform()` chỉ đọc các câu **train của fold**.
- Các câu validation chỉ gọi `transform()` bằng vocabulary/IDF vừa học.
- Sang fold khác, GridSearchCV tạo bản mô hình mới cho fold đó và học vocabulary/IDF phù hợp phần train mới.

### 15.2. Khi fit mô hình cuối

Sau khi chọn cấu hình thắng, `GridSearchCV(refit='f1')` fit lại **toàn bộ Train**. Vectorizer trong model cuối có vocabulary/IDF được học từ toàn Train (không bao gồm Test).

### 15.3. Khi chấm Test độc lập (Notebook cell 20)

```python
for record in CANDIDATES:
    evaluate(
        record,
        TEXT[record["Tiền xử lý"]].iloc[test_idx].to_numpy(),
        y_test
    )
```

`evaluate` gọi `predict_record` → Pipeline đã fit → tự gọi `vec.transform(test_text)` → classifier đưa score/nhãn → so với `y_test`.

**Test không gọi `vec.fit` hoặc `vec.fit_transform`.**

### 15.4. Khi lưu Joblib và dự đoán mới (Notebook cell 32, `modeling/predict.py`)

Bundle lưu gần như sau:

```python
bundle = {
    "model": best["_model"],
    "threshold": best["_threshold"],
    "remove_stop": best["Tiền xử lý"] == "có loại stop words",
    # và metadata khác...
}
joblib.dump(bundle, model_path)
```

Pipeline bên trong `bundle['model']` chứa cả **vectorizer đã fit** và **classifier đã fit**. Khi dự đoán câu Facebook mới, `predict_spam()` gọi tiền xử lý phù hợp, sau đó `predict_record()` gọi Pipeline để `transform` theo vocabulary/IDF cũ và tạo score.

**Không cần tự lưu một CSV vector khổng lồ làm đầu vào cho `predict_spam`**; mô hình lưu trong `.joblib` đã biết cách tự chuyển câu thành vector.

---

## 16. Những điểm dễ hiểu sai — cần tránh khi bảo vệ

1. **“Trong text_features.py ta đã huấn luyện BoW/TF-IDF.”** → Sai. File này mới **khởi tạo**. Lúc `search.fit()` ở `experiments.py`, Pipeline mới fit từ vựng/IDF.
2. **“BoW/TF-IDF phân loại Spam.”** → Sai. Chúng tạo **ma trận đặc trưng**, không sử dụng nhãn để ra quyết định.
3. **“Hai vectorizer nhận nhãn `y_train` để học.”** → Không theo cơ chế của các lớp này; phần học vocabulary/IDF chỉ dựa trên văn bản. Classifier phía sau mới nhận cả ma trận và nhãn.
4. **“TF-IDF chỉ nhân `count × IDF`.”** → Không đủ với cấu hình nhóm: còn **TF dưới tuyến tính** và **chuẩn hóa L2**.
5. **“`min_df=2` nghĩa là từ xuất hiện hai lần trong một câu thì giữ.”** → Sai. Phải xuất hiện trong **ít nhất hai văn bản khác nhau**.
6. **“`ngram_range=(1,2)` chỉ lấy bigram.”** → Sai. Nó lấy **cả unigram và bigram**.
7. **“`ngram_range=(1,1)` được cố định.”** → Sai. Chỉ là **giá trị khởi tạo**; GridSearchCV còn thử `(1,2)`.
8. **“`CountVectorizer` tự tách từ tiếng Việt theo nghĩa.”** → Sai. Nó dùng regex/đặc trưng token; chuỗi đầu vào đã được thành viên khác tách từ bằng `underthesea` trước đó.
9. **“`TfidfVectorizer` tự bỏ stopwords.”** → Không trong cấu hình project này (`stop_words` không được truyền). Hai nhánh đã được chuẩn bị ngoài vectorizer.
10. **“Test có từ mới thì thêm cột.”** → Sai. OOV bị bỏ, số chiều phải giữ nguyên để classifier đọc đúng trọng số.
11. **“IDF càng cao thì từ càng chắc là Spam.”** → Sai. IDF không sử dụng nhãn; tính phân biệt Spam/Ham do classifier học.
12. **“Fit TF-IDF một lần trên toàn CSV rồi dùng CV.”** → Có nguy cơ leakage. Phải fit riêng ở mỗi train-fold, đúng như Pipeline của project.
13. **“`X_train_text` tại `fit_experiment` đã là vector.”** → Sai. Nó vẫn là mảng **chuỗi token**; Pipeline tự chuyển thành số.
14. **“`np.float64` biến BoW thành TF-IDF.”** → Sai. Nó chỉ là **kiểu dữ liệu lưu giá trị**.
15. **“Vocabulary có kích thước giống nhau ở mọi fold.”** → Không chắc. Mỗi fold train có thể giữ các term khác nhau do `min_df`.
16. **“CV F1 của B4 chính là Test F1.”** → Sai. CV F1 được dùng chọn cấu hình; Test F1 đánh giá trên tập giữ riêng.
17. **“TF-IDF luôn hơn BoW.”** → Sai. Notebook chỉ chứng minh B4 thắng **trong bốn tổ hợp đang thử** với classifier và cấu hình cụ thể.
18. **“Chỉ cần giữ model SVM, không cần vectorizer.”** → Sai. Phải giữ cùng **cách ánh xạ văn bản thành cột** và IDF; Pipeline đã fit làm việc này.

### 16.1. Một số giới hạn kỹ thuật cần biết

- `min_df=2` loại một số term hiếm có thể hữu ích.
- Việc giữ ký hiệu và bigram dấu câu có thể tạo thêm nhiều cột không cần thiết.
- Khi phần lớn từ của câu mới nằm ngoài từ điển, vector có thể rất ít ô khác 0 hoặc **toàn 0**, khiến mô hình ít thông tin để dự đoán.
- BoW bỏ quan hệ thứ tự dài; bigram chỉ khôi phục một phần thứ tự cục bộ, chưa hiểu ngữ nghĩa sâu, phủ định phức tạp hay ngữ cảnh bài đăng.
- TF-IDF giảm ảnh hưởng tần suất và độ dài theo cấu hình, nhưng vẫn là biểu diễn từ vựng, không phải mô hình hiểu ngữ cảnh như viBERT.
- `Min_df=2` tính trên văn bản sau khi chia fold, nên nếu dữ liệu thiếu hoặc rất nhỏ, một fold có thể không còn term nào và thư viện sẽ báo lỗi.
- Bản thân token có thể chứa `_` từ tách từ tiếng Việt. Nếu truyền câu thô không qua bước tiền xử lý như lúc train, biểu diễn thay đổi, kéo theo chất lượng dự đoán khó đảm bảo.

---

## 17. Câu hỏi bảo vệ thường gặp và cách trả lời

**Câu 1. File `text_features.py` có vai trò gì?**  
Trả lời: File này chọn và cấu hình hai phương pháp trích chọn BoW/TF-IDF bằng `CountVectorizer` và `TfidfVectorizer`, đồng thời tạo lưới n-gram để Cross-Validation thử. Nó chưa fit; `experiments.py` ghép bộ vector hóa với NB/SVM và gọi fit.

**Câu 2. Vì sao phải chuyển văn bản thành vector?**  
Trả lời: NB/SVM trong dự án nhận ma trận đặc trưng số; vector hóa ánh xạ các từ/ngữ thành cột và giá trị số nhất quán giữa Train và các câu mới.

**Câu 3. `make_vectorizer` nhận và trả gì?**  
Trả lời: Nhận `"BoW"` hoặc `"TF-IDF"`; trả đối tượng vectorizer **chưa fit** và dictionary `{'vec__ngram_range': [(1,1),(1,2)]}`.

**Câu 4. Vì sao dùng CountVectorizer thay vì tự viết vòng đếm?**  
Trả lời: Thư viện chuẩn cung cấp token hóa theo regex, học từ vựng, lọc term hiếm, đếm, xây ma trận thưa và giao diện tương thích Pipeline/CV. Nhóm tập trung vào thiết kế thí nghiệm và giải thích cấu hình.

**Câu 5. TF-IDF gồm những phép tính nào?**  
Trả lời: Tạo ma trận đếm, tính TF dạng `1+ln(count)`, IDF làm mịn `1+ln((N+1)/(df+1))`, nhân TF×IDF rồi chuẩn hóa L2 từng vector.

**Câu 6. `min_df=2` có tác dụng gì?**  
Trả lời: Chỉ giữ term nằm trong ít nhất hai văn bản của phần được fit; giúp giảm term hiếm nhưng có thể bỏ tín hiệu tốt nếu quá hiếm.

**Câu 7. `ngram_range=(1,2)` có nghĩa gì?**  
Trả lời: Vừa dùng unigram vừa dùng bigram. Nó bổ sung cụm hai token kề nhau để giữ một phần thông tin cụm từ.

**Câu 8. Sao không học từ vựng trên cả dataset cho phong phú?**  
Trả lời: Vì validation/Test là dữ liệu giữ riêng. Học từ vựng/IDF trên toàn CSV trước CV sẽ nhìn trước thống kê của dữ liệu dùng để kiểm tra.

**Câu 9. Sau `fit_transform()` thu được gì?**  
Trả lời: Ma trận thưa `n_documents × n_features`; BoW chứa số đếm, TF-IDF chứa trọng số thực. Chưa có nhãn dự đoán.

**Câu 10. Sau khi có ma trận, làm gì?**  
Trả lời: Pipeline đưa ma trận cùng `y_train_fold` vào `MultinomialNB.fit` hoặc `LinearSVC.fit` để học bộ phân loại.

**Câu 11. Khi dự đoán một câu mới, phải gọi gì?**  
Trả lời: Tiền xử lý giống lúc Train, sau đó vectorizer đã fit gọi `transform`; classifier đã fit dùng vector ấy để tính nhãn/score. Pipeline tự nối các bước.

**Câu 12. Tại sao phải giữ stopwords trong project?**  
Trả lời: Notebook thử cả giữ và bỏ rồi so CV F1. Kết quả đã lưu cho thấy nhánh giữ stopwords dẫn đầu theo tiêu chí lựa chọn đang sử dụng. Không phải giả định giữ từ dừng luôn tốt.

**Câu 13. Vì sao B4 TF-IDF + SVM tốt hơn B3 TF-IDF + NB?**  
Trả lời: Cùng biểu diễn TF-IDF nhưng hai classifier học quy tắc khác nhau. SVM học ranh giới tuyến tính, NB học thống kê đặc trưng theo lớp. Trên tập hiện tại, B4 có CV F1 Spam cao hơn; đây là kết quả thực nghiệm, không chứng minh tính ưu việt phổ quát.

**Câu 14. Khi dùng TF-IDF với MultinomialNB có được không?**  
Trả lời: Scikit-learn chấp nhận ma trận đặc trưng không âm; nhiều bài toán text classification dùng TF-IDF với MultinomialNB. Khi ấy thống kê `feature_count_` là tổng **trọng số**, không còn đơn thuần là số đếm nguyên.

**Câu 15. `vocabulary_` và `idf_` được lưu ở đâu khi triển khai?**  
Trả lời: Trong đối tượng vectorizer đã fit, nằm bên trong Pipeline được lưu bằng `joblib` cùng classifier và ngưỡng, vì vậy câu mới dùng lại đúng hệ đặc trưng.

**Câu 16. Nếu input chỉ chứa từ chưa có trong vocabulary thì sao?**  
Trả lời: `transform` không mở rộng số cột; từ lạ bị bỏ. Vector có thể toàn 0, bộ phân loại vẫn có thể trả nhãn nhưng sẽ dựa trên rất ít hoặc không có tín hiệu từ vựng của câu mới.

**Câu 17. `dtype=np.float64` có phải tham số học không?**  
Trả lời: Không. Nó chọn kiểu số của ma trận, không phải siêu tham số được GridSearchCV thử ở dự án.

**Câu 18. Có phải Notebook huấn luyện chỉ bốn mô hình không?**  
Trả lời: Có bốn loại tổ hợp cơ sở B1–B4 nhưng Notebook còn chạy mỗi tổ hợp với hai nhánh stopwords và thử nhiều n-gram/alpha/C trong CV. S1/S2/S3 là các chiến lược bổ sung trên TF-IDF + SVM.

---

## 18. Tự kiểm tra nhanh (có đáp án)

**Bài A.** Một từ xuất hiện năm lần trong một bình luận và không xuất hiện ở bất kỳ bình luận train khác. Với `min_df=2`, giữ hay bỏ?  
**Đáp án:** Bỏ, vì document frequency = 1 dù tổng số lần xuất hiện = 5.

**Bài B.** Với `ngram_range=(1,2)`, chuỗi ba token `"xin lịch thi"` tạo ra những term nào trước lọc `min_df`?  
**Đáp án:** `xin`, `lịch`, `thi`, `xin lịch`, `lịch thi`.

**Bài C.** Sau `bow.fit(train_docs)`, có thể gọi `bow.transform(test_docs)` không?  
**Đáp án:** Có. Đó là đúng quy trình. Không fit lại trên test.

**Bài D.** Sau `make_vectorizer('TF-IDF')`, đọc `vec.idf_` ngay được không?  
**Đáp án:** Không, vì chưa fit. `idf_` được tạo trong quá trình học dữ liệu văn bản.

**Bài E.** Hai câu ở Test có thêm từ `"khuyến_mãi_mới"` không có trong vocabulary. Có thêm cột mới không?  
**Đáp án:** Không. Từ OOV không được thêm vào ma trận test; số cột giữ như lúc train.

**Bài F.** `GridSearchCV` chọn `(1,2)` cho B4 có nghĩa là gì?  
**Đáp án:** Trong lưới `(1,1)` và `(1,2)` đã thử, `(1,2)` có kết quả CV F1 Spam tốt hơn cho cấu hình được chọn ở B4.

**Bài G.** Score Spam ở đâu xuất hiện?  
**Đáp án:** Sau khi vector đi qua classifier đã fit, gọi `predict_proba` của NB hoặc `decision_function` của SVM (đúng theo `evaluation.py`). `CountVectorizer` và `TfidfVectorizer` không trả score Spam.

---

## 19. Ghi nhớ để nối sang `classifiers.py`

**Đầu vào sau khi kết thúc bước text features:**

```text
N bình luận dạng chuỗi token
          |
     CountVectorizer hoặc TfidfVectorizer
          |
   X_vector: ma trận thưa (N, V)
   V = số đặc trưng được giữ sau fit
```

Song song có `y_train`:

```text
y_train = [0, 1, 0, 0, 1, ...]
```

Bước kế tiếp, classifier nhận:

```python
classifier.fit(X_vector, y_train)
```

Đối với Naive Bayes: học thống kê lớp/đặc trưng và log xác suất.  
Đối với SVM: học hệ số ranh giới phân loại.  
Trong project, **Pipeline điều phối tự động** nên `fit_experiment()` vẫn nhận `X_train_text` dạng chuỗi; bạn không phải tự xuất ma trận ra file trung gian.

### Một câu tóm tắt để thuộc

> `text_features.py` khai báo cách biến chuỗi token thành vector BoW hoặc TF-IDF bằng scikit-learn. `make_vectorizer()` chỉ tạo đối tượng chưa fit và các lựa chọn n-gram. Khi `experiments.py` chạy Cross-Validation, mỗi fold train sẽ học vocabulary (và IDF nếu TF-IDF), rồi vector hóa để Naive Bayes/SVM học nhãn. Validation/Test chỉ được transform theo thông tin đã học, giúp tránh data leakage.

---

## 20. Bảng tra nhanh: nguồn code và tài liệu học

| Mục cần học | File/cell để mở | Nội dung |
|---|---|---|
| Đầu vào dạng token | Notebook 02, **cell 9** | Hai cột giữ/bỏ stopwords |
| Chỉ số Train/Test/fold | Notebook 02, **cell 11** | `train_idx`, `test_idx`, `CV_FOLDS` |
| Mô tả thiết kế | Notebook 02, **cell 12 (Markdown)** | BoW/TF-IDF, NB/SVM, CV |
| Lệnh chạy thực tế | Notebook 02, **cell 13** | Lặp hai nhánh từ dừng × hai feature × hai thuật toán |
| Cấu hình vector hóa | **`text_features(1).py`, toàn file** | `TOKEN_PATTERN`, `NGRAMS`, `make_vectorizer` |
| Tạo classifier | `classifiers(1).py` | MultinomialNB, LinearSVC, alpha/C |
| Ghép và fit | `experiments(1).py` | `make_pipe`, `GridSearchCV`, `fit_experiment` |
| Dự đoán Test | `evaluation(1).py`, Notebook **cell 20** | Từ text qua Pipeline đến nhãn/score |
| Lưu và dùng câu mới | `predict(1).py`, Notebook **cell 32** | Bundle có Pipeline đã fit và cấu hình tiền xử lý |
| Mã nguồn 4 lớp thư viện | `PHI_HIEU_4_LOP_SCIKIT_LEARN.md` | Đường Ctrl+Click tới scikit-learn 1.6.1; chức năng nội bộ |

**Bước tiếp theo trong lộ trình:** `classifiers.py` — hai thuật toán NB/SVM nhận ma trận số từ bước này, học gì khi `fit`, lưu thông tin nào và tạo score/nhãn như thế nào. Sau đó đến `experiments.py`, ta sẽ ghép toàn bộ các hàm vào đúng đường chạy CV của Notebook.


## 21. BẢNG TỔNG KẾT THEO YÊU CẦU BỔ SUNG — MỖI HÀM THỰC SỰ NHẬN/TRẢ GÌ?

| Bước | Hàm/phương thức thật | Đầu vào | Hành động | Trạng thái lưu/đầu ra |
|---|---|---|---|---|
| 1 | `make_vectorizer('BoW')` | Chuỗi tên phương pháp | Tạo `CountVectorizer` chưa fit + grid | `(vec, {'vec__ngram_range':[(1,1),(1,2)]})` |
| 2 | `make_vectorizer('TF-IDF')` | Chuỗi tên phương pháp | Tạo `TfidfVectorizer` chưa fit + grid | `(vec, grid)` |
| 3 | `vec.build_analyzer()` | Cấu hình | Ghép preprocessor + tokenizer + n-gram | Callable tạo term khi nhận một câu |
| 4 | `CountVectorizer._count_vocab(...)` | Danh sách câu + analyzer | Đếm term theo từng câu, gán cột | Candidate vocabulary + CSR count |
| 5 | `CountVectorizer.fit_transform(train_docs)` | Câu Train | Học vocab, lọc `min_df`, trả số đếm | `vocabulary_`, `X_count` |
| 6 | `TfidfTransformer.fit(X_count)` | Ma trận đếm Train | Đếm DF, tính IDF làm mịn | `idf_` |
| 7 | `TfidfTransformer.transform(X_count)` | Ma trận đếm | TF dưới tuyến tính × IDF × L2 | `X_tfidf` |
| 8 | `vec.transform(valid_docs)` | Câu Validation/Test | Đếm theo từ điển cũ (và áp IDF cũ nếu TF-IDF) | Ma trận đúng số cột đã fit |
| 9 | `make_pipe(feat, model)` | Tên vector + classifier | Ghép hai bước chưa fit và lưới | `Pipeline`, `grid`, `n_jobs` |
| 10 | `GridSearchCV.fit(X_train_text, y_train)` | Câu Train, nhãn 0/1, folds | Thử `(1,1)`/`(1,2)` với alpha/C; chấm validation; refit | `cv_results_`, `best_params_`, `best_estimator_` |
| 11 | `clf.fit(X_vector, y_train)` bên trong Pipeline | Ma trận số + nhãn | Học phân loại Spam/Ham | Thống kê NB hoặc hệ số SVM |
| 12 | `record['_model']` | Record đã fit | Truy xuất Pipeline thắng | Vectorizer và classifier đã cùng fit |

## 22. CÁC CÂU HỎI KIỂM TRA MỨC ĐỘ HIỂU SÂU

**Q1. Hàm tạo vector trả luôn ma trận hay không?** `make_vectorizer()` **không** trả ma trận; nó trả đối tượng vectorizer chưa fit và grid. Ma trận được tạo khi `Pipeline.fit` gọi `fit_transform` trên fold train.

**Q2. Có bao nhiêu vocabulary khi chạy 5-fold CV với hai cấu hình n-gram?** Mỗi ứng viên và mỗi fold được fit riêng, nên có nhiều vocabulary tạm thời; cấu hình thắng được **fit lại trên toàn Train** để có vocabulary của mô hình cuối. Không chỉ có một vocabulary dùng chung cho mọi fold.

**Q3. `min_df=2` là `xin` lặp hai lần trong một câu thì được giữ?** Không. Phải có `xin` trong **hai văn bản khác nhau** của tập đang fit.

**Q4. Tại sao `lịch_thi` không phải bigram?** Trong chuỗi đầu vào nó là **một token** ghép từ trước; `"xin lịch_thi"` mới là bigram gồm hai token đứng cạnh nhau.

**Q5. `(1,2)` có nghĩa chỉ bigram?** Không, nó gồm unigram **và** bigram.

**Q6. Tại sao `common` ghi `(1,1)` nhưng mô hình cuối có `(1,2)`?** Vì `common` là khởi tạo; `GridSearchCV` áp từng `vec__ngram_range` từ grid trước fit. Bộ thắng được chọn bằng CV F1.

**Q7. Dựa vào đâu biết thư viện đã thử cả hai?** `NGRAMS=[(1,1),(1,2)]`, `make_vectorizer` trả `vec__ngram_range`, `make_pipe` ghép grid, `GridSearchCV.fit` duyệt grid qua folds. Output `best_params_` chỉ hiển thị bộ thắng.

**Q8. Nếu muốn xem F1 `(1,1)` có cần file nào?** Cần `search.cv_results_` sau một lần GridSearchCV chạy. Bản Notebook hiện lưu record điểm thắng, không lưu mọi hàng cv_results_; cần chạy lại hoặc thay code để xuất bảng đó.

**Q9. TF-IDF có bỏ bước đếm từ không?** Không. `TfidfVectorizer` sử dụng CountVectorizer tạo ma trận số đếm, rồi TfidfTransformer tính TF×IDF và L2.

**Q10. Vì sao `X.data` dùng log không lỗi khi X có nhiều số 0?** Vì `X.data` của CSR chỉ chứa các giá trị khác 0; số 0 không lưu rõ như trong ma trận đặc.

**Q11. Vì sao một từ có IDF cao chưa chắc phân loại Spam tốt?** IDF chỉ phản ánh số **văn bản chứa từ**; không xem nhãn Spam/Ham. Classifier mới dùng nhãn thật để học mối liên hệ đó.

**Q12. Vì sao so sánh BoW/TF-IDF phải đánh giá cả Pipeline?** Vì BoW/TF-IDF chỉ tạo vector. Kết quả phân loại phụ thuộc đồng thời biểu diễn đặc trưng, n-gram, alpha/C và cách classifier học nhãn.

**Q13. Biến `y_train` đi đâu?** Không cần để xây vocabulary/IDF; `Pipeline` chuyển ma trận số cùng `y_train` đến `clf.fit`.

**Q14. Có nên `fit_transform` trên toàn CSV trước khi chia Train/Test?** Không. Dù chỉ học vocabulary/IDF, như vậy đã dùng thông tin của Test và gây rò rỉ vào quá trình chuẩn bị đặc trưng.

**Q15. Tại sao lúc dự đoán văn bản mới không tính lại từ điển?** Vì mỗi cột của mô hình đã học tương ứng một đặc trưng xác định. Thêm/đổi cột sẽ làm trọng số NB/SVM không còn khớp ý nghĩa.

**Q16. L2 chuẩn hóa trên từng cột hay từng hàng?** Từng **hàng** (mỗi bình luận), không phải trên cả bộ dữ liệu hoặc từng cột.

**Q17. Số cột TF-IDF `(1,2)` luôn gấp đôi `(1,1)`?** Không. Số bigram duy nhất phụ thuộc corpus và bước lọc `min_df`; ví dụ 5 câu chỉ tăng từ 5 lên 8 cột.

**Q18. Tại sao tất cả B1–B4 đều có `(1,2)` trong output?** Vì đó là các **cấu hình thắng CV** của lần chạy lưu trong Notebook. Không phải vì chỉ chạy `(1,2)`, cũng không phải kết luận tổng quát mọi bài toán nên dùng bigram.

## 23. CẦN CUNG CẤP GÌ NẾU MUỐN XEM SÂU HƠN NỮA?

**Các file hiện có đã đủ để giải thích đúng đường mã nguồn thư viện ở mức Python**, đặc biệt vì bạn đã cung cấp `PHI_HIEU_4_LOP_SCIKIT_LEARN.md` ghi phương thức và đoạn code của bản scikit-learn 1.6.1. Không cần gửi thêm mới hiểu `CountVectorizer`/`TfidfVectorizer` hoạt động.

Nếu bạn cần **bảng số thực tế từng n-gram/candidate cho bộ 38.233 bình luận**, hiện Notebook không lưu `cv_results_` đầy đủ trong output. Khi đó cần **file CSV và môi trường huấn luyện để chạy lại CV**, hoặc một bản `cv_results_.csv` được xuất ra từ máy nhóm. Không có các dữ liệu đó thì không thể nêu F1 từng cấu hình `(1,1)` bằng số cụ thể một cách đáng tin cậy.

**Tài liệu đối chiếu:** `text_features(1).py`, `experiments(1).py`, `classifiers(1).py`, `02_thuc_nghiem_va_demo(1).ipynb` (cell 9, 11–13), `PHI_HIEU_4_LOP_SCIKIT_LEARN.md` (mục 2–4, 7). Các bảng vector năm câu được kiểm tra bằng Python/scikit-learn cài trong môi trường hiện tại; số liệu mô hình B1–B4 và lựa chọn n-gram lấy từ output Notebook đã lưu.
