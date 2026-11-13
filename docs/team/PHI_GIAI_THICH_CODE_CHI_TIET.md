# Phi: đọc code từ CSV đến mô hình dự đoán

Tài liệu này dành riêng cho phần Phi phụ trách. Đọc từ trên xuống theo luồng chạy; mỗi mục có **toàn bộ mã nguồn của file ở phiên bản hiện tại**, rồi giải thích các câu lệnh theo nhóm dòng. Mã được chép tự động từ file gốc để dễ đối chiếu; khi thực hành, hãy chạy file gốc hoặc Notebook 02. Không cần học lại crawler, gán nhãn hay viBERT để hiểu luồng dưới đây.

> **Ranh giới đầu vào:** `data/processed/v7_preprocessed.csv` đã có `id`, `label`, `cleaned_text`, `tokens_keep_stopwords` và `tokens_remove_stopwords`. Phi nhận dữ liệu đó, tự chia tập, huấn luyện và đánh giá. **Ranh giới đầu ra:** bảng điểm, dự đoán trên test và `spam_ham_ml_only.joblib`. Notebook 02 tạo CSV tiền xử lý ở các cell trước phần Phi; khi chạy script riêng, CSV này phải có sẵn.

**Mở file gốc khi học:** [chia tập](../../pipeline/modeling/training_utils.py) → [đặc trưng](../../pipeline/modeling/text_features.py) → [bộ phân loại](../../pipeline/modeling/classifiers.py) → [thực nghiệm](../../pipeline/modeling/experiments.py) → [lệch lớp](../../pipeline/modeling/imbalance.py) → [đánh giá](../../pipeline/modeling/evaluation.py) → [dự đoán](../../pipeline/modeling/predict.py). [Script chạy thẳng](../../scripts/run_phi_pipeline.py) và [Notebook 02](../../notebooks/02_thuc_nghiem_va_demo.ipynb) gọi các hàm này.

## 0. Nhìn toàn bộ luồng trước khi đọc từng dòng

```text
CSV đã có nhãn + hai chuỗi token
    ↓ training_utils.py: khóa nội dung → train/test → 5 fold trong train
    ↓ text_features.py: BoW/TF-IDF, học từ vựng và IDF trên phần được phép học
    ↓ classifiers.py: tạo NB/SVM và danh sách tham số cần thử
    ↓ experiments.py: Pipeline + GridSearchCV.fit → B1–B4, điểm CV
    ↓ imbalance.py: S1 trọng số / S2 giảm ham / S3 chỉnh ngưỡng
    ↓ chọn B1–B4/S1–S3 bằng CV F1 spam, chưa xem test
    ↓ evaluation.py: dự đoán và đo Accuracy, Precision, Recall, F1, AP trên test
    ↓ Notebook 02 hoặc run_phi_pipeline.py: lưu .joblib, nạp lại để kiểm
    ↓ modeling/predict.py: bình luận thô mới → tiền xử lý → vector → score → ham/spam
```

**Các ký hiệu dùng xuyên suốt:** `X` trong bước huấn luyện là **mảng chuỗi token**, chưa phải ma trận số; `y` là mảng nhãn số cùng vị trí (`0=ham`, `1=spam`). `train_idx/test_idx` là vị trí trong **toàn CSV**; `train/valid` trong mỗi fold là vị trí trong **mảng train**. `record` là một `dict` chứa tên cấu hình, điểm CV, mô hình đã fit, ngưỡng và về sau thêm điểm test. `fit` nghĩa là học trạng thái từ dữ liệu; `transform` áp trạng thái đã học lên văn bản; `predict`/`decision_function` cho kết quả từ mô hình đã học.

| Thời điểm | Được phép học từ dữ liệu nào? | Dùng để làm gì? |
|---|---|---|
| Một lượt CV | Phần train của fold | Học từ vựng, IDF, tham số NB/SVM và trọng số phân loại. |
| Validation của lượt đó | Không fit | Tính điểm để so tham số/cấu hình. |
| Sau CV | Toàn bộ train | Fit lại cấu hình đã chọn. |
| Test | Không fit, không chọn lại | Báo hiệu năng cuối và xem lỗi. |

### Học thư viện ở mức nào?

Không cần đọc hàng nghìn dòng mã tối ưu của scikit-learn. Với hàm thư viện, Phi cần trả lời được: **đầu vào**, **trạng thái nó học/tạo**, **đầu ra**, và **lúc nào nó được gọi trong bài**. BoW/TF-IDF dùng `CountVectorizer` và `TfidfVectorizer` của scikit-learn; đọc [tài liệu bốn lớp](PHI_HIEU_4_LOP_SCIKIT_LEARN.md) để theo các phương thức thực sự được gọi. Phi cũng cần phân biệt **đối tượng chưa fit** với **mô hình đã fit**.

## 1. `pipeline/modeling/training_utils.py` — chia tập theo nội dung

**Đầu vào:** cột `cleaned_text` và nhãn số cùng thứ tự. **Đầu ra:** khóa nội dung từng dòng, vị trí train/test, rồi năm cặp vị trí train/validation. File này **không sửa cột token dùng để huấn luyện**. Việc chuẩn hóa trong `norm_key` chỉ để quyết định câu nào đi cùng tập.

### Mã nguồn đầy đủ

```python
"""Giai đoạn 0 của Phi: chia train/test rồi tạo các lượt train/validation.

Notebook 02 và scripts/run_phi_pipeline.py dùng cùng các hàm ở đây. Bản CSV
đầu vào đã có nhãn và chuỗi token; file này chỉ quyết định *dòng nào* thuộc
train/test/fold, không tạo token hay học mô hình.
"""

import hashlib
import re
import unicodedata

import numpy as np


def norm_key(text):
    """Đưa hai bình luận gần như trùng văn bản về cùng khóa để chia cùng phía.

    Ví dụ: chữ hoa/thường, khoảng trắng và các chữ số được chuẩn hóa. Đây là
    khóa chia tập, không phải câu dùng trực tiếp để huấn luyện mô hình.
    """
    # NFC ghép các cách viết Unicode tương đương; lower bỏ khác biệt hoa/thường.
    normalized = unicodedata.normalize("NFC", str(text)).lower()
    # Gom nhiều dấu cách thành một và bỏ khoảng trắng ở hai đầu.
    normalized = re.sub(r"\s+", " ", normalized).strip()
    # Đưa mọi chuỗi số về 0 để hạn chế cùng câu khác một con số bị tách tập.
    key = re.sub(r"\d+", "0", normalized)
    # Bỏ dấu câu trong khóa, nhưng vẫn giữ chữ và khoảng trắng.
    key = re.sub(r"[^\w\s]", " ", key)
    key = re.sub(r"\s+", " ", key).strip()
    # Nếu chỉ toàn ký hiệu, dùng chính chuỗi đã chuẩn hóa để tránh khóa rỗng.
    return key or "__symbols__|" + normalized


def bucket(key, salt, n=5):
    """Gán cùng một nội dung vào cùng nhóm, ổn định qua nhiều lần chạy.

    salt chỉ là tên mục đích "test"/"cv" để hai lần chia độc lập. MD5 ở đây
    KHÔNG mã hóa bình luận hay bảo mật dữ liệu; nó chỉ cho ra số nhóm cố định.
    Giữ quy tắc này để train/test của mô hình truyền thống và viBERT trùng nhau.
    """
    digest = hashlib.md5(f"{salt}|{key}".encode("utf-8")).hexdigest()
    return int(digest, 16) % n


def split_train_test(cleaned_texts, labels):
    """Chia toàn bộ CSV: khoảng 4/5 train, 1/5 test theo nhóm nội dung.

    Đầu vào: văn bản đã làm sạch và nhãn số cùng thứ tự từng dòng.
    Đầu ra: khóa nội dung, vị trí train, vị trí test trong TOÀN BỘ CSV.
    """
    keys = np.array([norm_key(text) for text in cleaned_texts])
    labels = list(labels)  # Chỉ kiểm hai lớp; chưa dùng nhãn để quyết định chia.
    if len(keys) != len(labels):
        raise ValueError("Văn bản và nhãn phải có cùng số dòng")

    # Dùng vị trí dòng để sau này lấy đúng văn bản và nhãn ở CSV.
    train_indices = np.array([i for i, key in enumerate(keys)
                              if bucket(key, "test") != 0], dtype=int)
    test_indices = np.array([i for i, key in enumerate(keys)
                             if bucket(key, "test") == 0], dtype=int)
    train_labels = {labels[i] for i in train_indices}
    test_labels = {labels[i] for i in test_indices}
    if train_labels != {0, 1} or test_labels != {0, 1}:
        raise ValueError("Train/test phải có cả ham=0 và spam=1")
    # Một bình luận trùng khóa không được xuất hiện ở cả train và test.
    assert not set(keys[train_indices]) & set(keys[test_indices])
    return keys, train_indices, test_indices


def grouped_folds(keys, y, n_splits=5):
    """Tạo năm cặp (vị trí học, vị trí kiểm tra) CHỈ bên trong train.

    keys/y ở đây đã lấy theo train. Mỗi vị trí trả về tính từ 0 trong train,
    khác với vị trí của dòng trong toàn bộ CSV.
    """
    keys = list(keys)
    labels = list(y)
    if len(keys) != len(labels):
        raise ValueError("Khóa và nhãn train phải có cùng số dòng")
    # Cùng khóa luôn có cùng số fold; "cv" tách mục đích này khỏi chia test.
    fold_ids = [bucket(key, "cv", n_splits) for key in keys]
    folds = []
    for fold_id in range(n_splits):
        # Vị trí thuộc fold hiện tại là validation; phần còn lại để học.
        train = np.array([i for i, value in enumerate(fold_ids)
                          if value != fold_id], dtype=int)
        valid = np.array([i for i, value in enumerate(fold_ids)
                          if value == fold_id], dtype=int)
        # set lấy các nhãn khác nhau; cả hai phía phải có {ham=0, spam=1}.
        train_labels = {labels[i] for i in train}
        valid_labels = {labels[i] for i in valid}
        if train_labels != {0, 1} or valid_labels != {0, 1}:
            raise ValueError("Fold thiếu một lớp; cần kiểm tra nhóm dữ liệu hoặc số fold.")
        assert not {keys[i] for i in train} & {keys[i] for i in valid}
        folds.append((train, valid))
    return folds


# Tương thích với các import cũ; phần Phi đọc code ở file theo giai đoạn.
from pipeline.modeling.imbalance import (sample_training_indices, undersampled_search,
                                         mean_cv_scores, tune_svm_threshold)
from pipeline.modeling.evaluation import predict_record
```

### Đọc các lệnh theo thứ tự

| Đoạn trong mã | Làm gì, vì sao cần |
|---|---|
| `hashlib` | Cung cấp MD5 để biến một khóa văn bản thành số nhóm ổn định. Ở đây **không mã hóa/giấu** bình luận. |
| `re` | Chạy biểu thức chính quy: `\s+` là một hoặc nhiều khoảng trắng, `\d+` là một hoặc nhiều chữ số, `[^\w\s]` là ký tự không phải chữ/số/gạch dưới/khoảng trắng. |
| `unicodedata` | `normalize("NFC", ...)` ghép các cách biểu diễn Unicode tương đương trước khi so câu. |
| `numpy as np` | Đóng vị trí thành mảng số để lấy nhiều dòng cùng lúc, ví dụ `y_all[train_idx]`. Không tự tính điểm hoặc học mô hình. |

**`norm_key(text)`:** `str(text)` bảo đảm đầu vào thành chuỗi, `.lower()` hạ chữ hoa. Lệnh `re.sub(r"\s+", " ", ...).strip()` gộp khoảng trắng và bỏ hai đầu. `re.sub(r"\d+", "0", ...)` khiến “lớp 1” và “lớp 2” có thể cùng khóa; sau đó bỏ dấu câu và gộp khoảng trắng lần nữa. `return key or "__symbols__|" + normalized` tránh khóa rỗng khi câu chỉ có biểu tượng. **Trả về một khóa để chia tập**, không phải câu token cho NB/SVM. Hai câu có khóa giống nhau sẽ luôn đi cùng phía; đổi số thành 0 cũng có thể gộp hai câu vốn khác nghĩa, nên đây là lựa chọn chống rò rỉ gần trùng, không phải quy tắc khẳng định hai câu đồng nghĩa.

**`bucket(key, salt, n=5)`:** ghép tên mục đích (`"test"` hoặc `"cv"`) với khóa, mã hóa chuỗi thành byte UTF-8 vì `hashlib.md5` nhận byte, lấy `hexdigest()` (chuỗi số hex), `int(...,16)` đổi sang số nguyên, `% n` lấy phần dư 0–4. Ví dụ cùng `key` và cùng `salt` luôn ra cùng số. `salt` chỉ là nhãn cho hai lần chia độc lập; không phải mật khẩu. Dùng hàm băm để kết quả chia không đổi theo thứ tự dòng/ngẫu nhiên và thống nhất với nhánh viBERT. Cách chia này **xấp xỉ 80/20**, không ép đúng 80/20 hay đúng tỷ lệ spam ở từng phần.

**`split_train_test(cleaned_texts, labels)`:** comprehension `[norm_key(text) for text in cleaned_texts]` tạo một khóa cho từng dòng; `np.array` bọc thành mảng. `list(labels)` để kiểm độ dài và tra nhãn theo vị trí. Hai comprehension với `enumerate(keys)` tạo chỉ số train khi bucket khác 0, test khi bucket bằng 0. `dtype=int` bảo đảm đó là chỉ số nguyên. Hai `set` kiểm cả train và test có đủ `{0,1}`; dấu `&` là giao hai tập khóa, `assert not ...` kiểm không có khóa xuất hiện ở cả hai phía. Trả `(keys, train_indices, test_indices)`; hàm **không trả hai DataFrame đã cắt**, nên bên gọi phải dùng các vị trí này để cắt văn bản/nhãn.

**`grouped_folds(keys, y, n_splits=5)`:** chỉ nhận `keys[train_idx]` và `y_train`. `fold_ids` gán mỗi khóa train vào một trong năm nhóm bằng `bucket(key,"cv",5)`. Với mỗi `fold_id`, comprehension tạo `valid` gồm vị trí thuộc nhóm đó, `train` gồm bốn nhóm còn lại. Kiểm cả hai phía có đủ nhãn và không trùng khóa. `folds.append((train,valid))` tạo danh sách năm cặp cho `GridSearchCV`. Các vị trí trong cặp **tính lại từ 0 trong mảng train**, không tra trực tiếp vào toàn CSV.

Hai dòng `from pipeline.modeling...` ở cuối file chỉ giữ tương thích với import cũ. Hàm lệch lớp thực sự nằm trong `imbalance.py`; `predict_record` nằm trong `evaluation.py`. Để học luồng hiện tại, hãy mở đúng hai file đó.

**Tự kiểm:** với hai bình luận có cùng `norm_key`, chúng có thể vào train và test khác nhau không? **Không**: `bucket` cho cùng số. `norm_key` có làm đổi chuỗi đưa vào TF-IDF không? **Không**: TF-IDF nhận `tokens_keep_stopwords` hoặc `tokens_remove_stopwords` sau khi đã chọn nhánh.

## 2. `pipeline/modeling/text_features.py` — tạo BoW/TF-IDF bằng thư viện

**Đầu vào:** tên `BoW` hoặc `TF-IDF`. **Đầu ra:** đối tượng `CountVectorizer`/`TfidfVectorizer` chưa fit và lưới thử n-gram. Văn bản train chỉ đi vào khi `Pipeline.fit` ở mục 4; vectorizer học `vocabulary_` và, với TF-IDF, `idf_` từ đúng fold train. `transform` trả ma trận thưa cho validation/test theo từ vựng ấy.

### Mã nguồn đầy đủ

```python
"""Tạo BoW hoặc TF-IDF bằng hai lớp có sẵn của scikit-learn.

make_vectorizer chỉ tạo đối tượng và lưới n-gram. Việc học từ vựng/IDF diễn ra
khi Pipeline được fit trong experiments.py, riêng trên train của từng fold.
"""

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


# Giữ cả từ một ký tự và dấu riêng lẻ, đúng quy tắc thí nghiệm đã chạy.
TOKEN_PATTERN = r"(?u)\b\w+\b|[^\w\s]"
NGRAMS = [(1, 1), (1, 2)]  # chỉ từ đơn hoặc từ đơn + cặp từ liên tiếp
FEATURES = ("BoW", "TF-IDF")
MIN_DOCUMENT_FREQUENCY = 2


def make_vectorizer(feature):
    """Trả (bộ tạo vector CHƯA fit, lưới n-gram để CV thử).

    Đầu vào: "BoW" hoặc "TF-IDF". Cả hai dùng cùng quy tắc tách từ,
    loại đặc trưng chỉ gặp trong một câu train và trả ma trận thưa.
    """
    common = {
        "token_pattern": TOKEN_PATTERN,
        "min_df": MIN_DOCUMENT_FREQUENCY,
        "ngram_range": (1, 1),  # giá trị ban đầu; GridSearchCV thử thêm (1, 2)
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

### Đọc từng phần

- `import numpy as np`: lấy kiểu `np.float64` cho ma trận; `from sklearn.feature_extraction.text import ...` lấy hai lớp thư viện, không có lớp vector hóa nhóm tự cài.
- `TOKEN_PATTERN`: biểu thức chính quy giữ từ Unicode kể cả một ký tự và dấu riêng lẻ. `lowercase=True` hạ chữ trước tách token. `min_df=2` lọc đặc trưng chỉ thấy ở một bình luận của fold train.
- `NGRAMS`: CV thử chỉ từ đơn hoặc từ đơn cộng cặp từ kề nhau. Giá trị ban đầu `(1,1)` được `GridSearchCV` đổi qua tên `vec__ngram_range`.
- `make_vectorizer("BoW")` tạo `CountVectorizer(**common)`; `make_vectorizer("TF-IDF")` tạo `TfidfVectorizer` với TF dưới tuyến tính, IDF làm trơn, chuẩn L2. Tên khác gây `ValueError` để tránh chạy sai.
- `return vectorizer, {"vec__ngram_range": NGRAMS}` trả đối tượng **chưa học** cùng lưới; `fit_transform` của lớp thư viện mới nhận chuỗi train và cho ma trận số. `transform` của chính đối tượng đó biến validation/test. Từng phương thức nội bộ và trích đoạn mã gốc được giải thích trong [PHI_HIEU_4_LOP_SCIKIT_LEARN.md](PHI_HIEU_4_LOP_SCIKIT_LEARN.md).

**Ví dụ:** train ba câu “mua sim”, “mua sim”, “lịch thi”; `min_df=2` chỉ giữ `mua`, `sim` và, khi bật bigram, `mua sim`. BoW của câu đầu có ba ô bằng 1; TF-IDF nhân IDF rồi chuẩn hóa hàng. Câu “lịch thi” chỉ có các đặc trưng bị lọc nên có thể thành hàng 0.

## 3. `pipeline/modeling/classifiers.py` — tạo NB/SVM chưa học

**Đầu vào của `make_classifier`:** tên thuật toán và tùy chọn trọng số lớp. **Đầu ra:** một đối tượng phân loại **chưa fit** và một `dict` các tham số để CV thử. File này **chưa nhận X/y**, nên chưa thể biết đâu là spam hay học ranh giới. Việc fit diễn ra ở `experiments.py` và `imbalance.py`.

### Mã nguồn đầy đủ

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
    """Trả (bộ phân loại chưa fit, các giá trị tham số để CV thử).

    alpha: cộng thêm vào số đếm NB để một từ chưa gặp không có xác suất 0.
    C: mức phạt lỗi khi SVM học ranh giới; cần thử trên validation.
    Ba giá trị alpha và bốn giá trị C là lưới nhỏ do nhóm chọn để khảo sát,
    không phải kết quả đã học. class_weight mặc định None; S1 dùng balanced.
    """
    if model_name == "Naive Bayes":
        # NB nhận vector không âm; chưa biết nhãn ham/spam cho đến lúc fit.
        if class_weight is not None:
            raise ValueError("class_weight chỉ dùng cho SVM trong thí nghiệm này")
        classifier = MultinomialNB()
        grid = {"clf__alpha": NB_ALPHAS}
    elif model_name == "SVM":
        # dual=False giữ cách giải bài toán gốc của project; max_iter=20000 là
        # số vòng tối đa để bộ tối ưu hội tụ. random_state được giữ trong cấu
        # hình, nhưng scikit-learn không dùng nó khi dual=False.
        # Chưa có trọng số phân loại nào được học ở dòng này.
        classifier = LinearSVC(dual=False, max_iter=20000, random_state=seed)
        grid = {"clf__C": SVM_CS,
                "clf__class_weight": [class_weight]}
    else:
        raise ValueError(f"Bộ phân loại không hỗ trợ: {model_name}")
    return classifier, grid
```

### Giải thích từng lựa chọn

| Câu lệnh | Ý nghĩa |
|---|---|
| `from sklearn.naive_bayes import MultinomialNB` | Lấy lớp triển khai Multinomial Naive Bayes. Viết `MultinomialNB()` chỉ **khởi tạo**, chưa tính xác suất. |
| `from sklearn.svm import LinearSVC` | Lấy lớp SVM tuyến tính. Viết `LinearSVC(...)` chỉ **khởi tạo**, chưa có vector trọng số/ranh giới. |
| `MODELS` | Giới hạn đúng hai bộ phân loại của bài; vòng lặp bên ngoài đọc biến này. |
| `NB_ALPHAS=[0.1,0.3,1.0]` | Ba mức làm trơn để CV thử; đây là lựa chọn khảo sát, không phải ba mô hình cuối. |
| `SVM_CS=[0.1,0.3,1.0,3.0]` | Bốn mức phạt lỗi để CV thử. `C` lớn thường ưu tiên khớp nhãn train hơn, `C` nhỏ cho phép biên/lỗi mềm hơn; không thể biết mức tốt nhất nếu chưa xem validation. |
| Dấu `*` trong chữ ký hàm | Các tham số phía sau chỉ được truyền theo tên, ví dụ `seed=42`; tránh đổi nhầm vị trí. |
| `class_weight=None` | Không tự chỉnh trọng số lớp ở B2/B4; S1 truyền `"balanced"`. Nhánh NB báo lỗi nếu cố truyền trọng số vì cấu hình bài chỉ khảo sát cách này cho SVM. |
| `grid={"clf__alpha":...}` | `clf` là tên bước phân loại trong Pipeline, dấu `__` đi đến tham số `alpha` của bước đó. Với SVM, `clf__C` tương tự; `clf__class_weight` là danh sách một giá trị vì mỗi lượt khảo sát cố định cách cân bằng. |
| `else: raise ValueError` | Báo tên thuật toán ngoài hai lựa chọn, tránh âm thầm tạo sai cấu hình. |
| `return classifier, grid` | Trả đối tượng chưa học và danh sách cần thử; **không trả điểm hay nhãn dự đoán**. |

**MultinomialNB xử lý như nào sau khi `fit(vector,y)`?** Vector có một cột cho mỗi từ/cặp từ, mọi giá trị không âm. Bộ học đếm/tổng trọng số đặc trưng trong các câu ham và trong các câu spam, học tần suất lớp `P(c)` và trọng số đặc trưng theo lớp. Với mỗi lớp `c` và cột `i`, ý tưởng làm trơn là `θ(c,i)=(N(c,i)+alpha)/(∑j N(c,j)+alpha×V)`, trong đó `V` là số cột. Khi có câu mới, nó cộng log prior lớp với các log `θ(c,i)` có trọng số theo giá trị vector, so hai lớp và có thể cho `predict_proba`. “Naive” là giả định các đặc trưng độc lập có điều kiện theo lớp, một xấp xỉ để tính được. BoW là đầu vào phù hợp tự nhiên; TF-IDF là trọng số thực không âm, cũng được thư viện hỗ trợ trong thực hành, dù không còn là số đếm nguyên đúng nghĩa của phân phối multinomial. `alpha` ngăn xác suất đặc trưng chưa gặp thành 0. Đầu vào của phương thức `fit` là ma trận `n_samples×n_features` và nhãn `0/1`; sau fit đối tượng giữ thống kê/trọng số. `predict_proba` trả ma trận `n_samples×2`, cột lớp 0 rồi lớp 1 trong bài này.

**LinearSVC xử lý như nào sau khi `fit(vector,y)`?** Nó tìm ranh giới tuyến tính `score(x)=w·x+b` tách ham và spam, đồng thời phạt mẫu nằm sai phía hoặc quá gần ranh giới. Cấu hình hiện dùng mặc định `loss="squared_hinge"` và chuẩn hóa trọng số L2 của thư viện; `C` điều khiển mức phạt lỗi. `class_weight="balanced"` đổi mức phạt theo tần suất lớp khi fit, không thêm/bớt câu. `dual=False` chọn bài toán tối ưu primal; `max_iter=20000` là giới hạn số vòng tối đa, không phải số vòng luôn chạy. Với `dual=False`, `random_state=seed` không làm thay đổi kết quả của bộ tối ưu này. Sau fit, `decision_function` nhận ma trận vector của câu mới và trả một score có dấu cho mỗi câu: score lớn hơn 0 thường phía spam, nhỏ hơn 0 phía ham. **Score không phải xác suất**; ngưỡng S3 có thể âm vì nó chỉ dịch ranh giới quyết định. `LinearSVC` không có `predict_proba` mặc định.

## 4. `pipeline/modeling/experiments.py` — ghép, CV và fit thật

**Đầu vào:** `pre` (tên nhánh từ dừng), `feat` (BoW/TF-IDF), `model_name` (NB/SVM), `X` (chuỗi token **chỉ của train**), `y` (nhãn train), `folds` (năm cặp vị trí). **Đầu ra:** một `record` có điểm CV, tham số thắng, ngưỡng mặc định và `_model` đã fit lại trên toàn train. Không nhận test ở file này.

### Mã nguồn đầy đủ

```python
"""Giai đoạn 1c: huấn luyện và chọn tham số bằng CV trên train.

text_features.py tạo bước văn bản -> vector; classifiers.py tạo NB/SVM.
File này GHÉP và CHẠY hai bước đó. Chỉ số test được tính ở evaluation.py.
"""

import time

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from pipeline.modeling.classifiers import MODELS, make_classifier
from pipeline.modeling.evaluation import evaluate, boot_ci
from pipeline.modeling.text_features import FEATURES, make_vectorizer


CV_SCORING = {"precision": "precision", "recall": "recall", "f1": "f1"}


def make_pipe(feat, model_name, *, seed=42, cache=None, n_jobs=2, class_weight=None):
    """Ghép hai bước chưa fit; trả (pipeline, lưới tham số, số tiến trình).

    Pipeline.fit(text, y) tự gọi vectorizer.fit_transform(text), sau đó
    classifier.fit(vector, y). Pipeline.predict/decision_function trên câu
    mới tự gọi vectorizer.transform trước bộ phân loại.
    """
    # Hai lệnh sau chỉ tạo đối tượng; chưa học từ vựng hoặc trọng số phân loại.
    if feat not in FEATURES or model_name not in MODELS:
        raise ValueError("Đặc trưng/thuật toán không nằm trong cấu hình thực nghiệm")
    vectorizer, vector_grid = make_vectorizer(feat)
    classifier, model_grid = make_classifier(model_name, seed=seed,
                                              class_weight=class_weight)
    # "vec" và "clf" là tên hai bước. Chưa học dữ liệu tại dòng này.
    pipe = Pipeline([("vec", vectorizer), ("clf", classifier)], memory=cache)
    # Ví dụ "vec__ngram_range" đổi n-gram; "clf__C" đổi C của SVM.
    # Dấu __ chỉ nói rõ tham số thuộc bước nào trong pipeline.
    return pipe, {**vector_grid, **model_grid}, n_jobs


def fit_experiment(pre, feat, model_name, X, y, folds, *, seed=42, cache=None, n_jobs=2,
                   class_weight=None, strategy="Không điều chỉnh"):
    """Chạy một cặp đặc trưng + thuật toán trên các fold của train.

    X: mảng CHUỖI TOKEN của train, chưa phải vector; y: nhãn 0/1 của train.
    folds: năm cặp vị trí (học, validation) trong mảng X/y.
    Trả một record gồm trung bình điểm validation, tham số thắng và pipeline
    đã fit lại trên TOÀN train. Không nhận và không đọc test.
    """
    # X/y ở đây đã là *train*; folds chỉ chứa vị trí bên trong train.
    pipe, grid, jobs = make_pipe(feat, model_name, seed=seed, cache=cache, n_jobs=n_jobs,
                                 class_weight=class_weight)
    start = time.time()
    # GridSearchCV liệt kê từng tổ hợp n-gram và alpha/C. Với mỗi tổ hợp:
    # fit trên bốn fold train, dự đoán fold validation, tính P/R/F1 spam;
    # lặp năm lần rồi lấy điểm trung bình. refit="f1" chọn bằng F1 spam.
    search = GridSearchCV(pipe, grid, scoring=CV_SCORING, cv=folds, n_jobs=jobs,
                          refit="f1", error_score="raise")
    # Lời gọi fit này mới thật sự học dữ liệu. Sau CV, thư viện tự fit lại
    # tổ hợp thắng trên toàn bộ X/y train để dùng cho bước test.
    search.fit(X, y)
    best_index = search.best_index_
    cv = search.cv_results_
    # Trong tên của scikit-learn, mean_test_* ở đây là điểm VALIDATION của CV;
    # chưa phải điểm trên tập test độc lập của project.
    cv_precision = float(cv["mean_test_precision"][best_index])
    cv_recall = float(cv["mean_test_recall"][best_index])
    cv_f1 = float(cv["mean_test_f1"][best_index])
    # Record giữ cả điểm CV để so sánh và _model đã fit để chấm test về sau.
    return {"Tiền xử lý": pre, "Đặc trưng": feat, "Thuật toán": model_name,
            "Chiến lược": strategy,
            "CV Precision (spam)": cv_precision,
            "CV Recall (spam)": cv_recall,
            "CV F1 (spam)": cv_f1,
            "Tham số": search.best_params_, "Thời gian (s)": time.time()-start,
            # NB quyết định theo xác suất; SVM theo decision score có dấu.
            "_threshold": 0.5 if model_name == "Naive Bayes" else 0.0,
            "_model": search.best_estimator_}
```

### Đọc dòng quan trọng

| Đoạn | Cách hiểu khi chạy |
|---|---|
| `time.time()` | Ghi thời điểm bắt đầu/kết thúc để tính thời gian thí nghiệm; không ảnh hưởng mô hình. |
| `Pipeline` | Ghép `("vec",vectorizer)` rồi `("clf",classifier)`. Khi fit, Pipeline gọi bước vector hóa trước bộ phân loại; khi dự đoán câu mới, nó dùng từ vựng đã học để transform trước. |
| `GridSearchCV` | Tự duyệt tất cả tổ hợp trong `grid`, dùng đúng `folds` truyền vào, chấm các validation, chọn theo `refit="f1"` và fit lại tổ hợp thắng trên toàn train. |
| `CV_SCORING` | Yêu cầu ba điểm Precision/Recall/F1 cho lớp dương mặc định `1=spam`. `refit="f1"` là quy tắc **chọn**, còn Precision/Recall được ghi để giải thích đánh đổi. |
| `FEATURES, MODELS` | Danh sách hai lựa chọn mỗi loại để kiểm cấu hình hợp lệ. `make_vectorizer` và `make_classifier` trả đối tượng chưa fit cùng lưới riêng. |
| `evaluate, boot_ci` | Import tương thích cũ trong module này; **`fit_experiment` không gọi chúng**. Test được đánh giá rõ ràng ở `evaluation.py`. |

**`make_pipe`:** kiểm tên; lấy `vectorizer,vector_grid`; lấy `classifier,model_grid`; dựng `Pipeline([("vec",...),("clf",...)],memory=cache)`. `cache` là bộ nhớ tạm tránh tính lại một số bước vector hóa, không phải file model nộp. `{**vector_grid,**model_grid}` gộp hai từ điển tham số: với NB có hai n-gram × ba alpha = **6** tổ hợp; SVM có hai n-gram × bốn C × một `class_weight` = **8**. `n_jobs` chỉ số tiến trình mà CV có thể dùng. Hàm trả `(pipe,grid,n_jobs)`, tất cả **chưa fit**.

**`fit_experiment`:** gọi `make_pipe`, lấy `start=time.time()`. `GridSearchCV(...,scoring=CV_SCORING,cv=folds,n_jobs=jobs,refit="f1",error_score="raise")` chỉ cấu hình thí nghiệm; `error_score="raise"` làm lỗi fit hiện rõ thay vì ghi điểm thiếu. Dòng `search.fit(X,y)` mới chạy thật: mỗi bộ tham số được thử trên từng fold; trong mỗi lượt vectorizer chỉ học từ fold train, NB/SVM học từ vector và nhãn fold train, rồi chấm fold validation. Sau khi lấy F1 trung bình, `refit` tự học lại tổ hợp thắng trên **toàn X/y train**. Đây là lý do không thể tạo vector trên toàn train trước CV: nếu làm vậy, từ vựng/IDF có thể đã thấy validation.

`search.best_index_` là vị trí bộ tham số thắng trong `cv_results_`. `mean_test_precision/recall/f1` là tên thuộc API GridSearchCV: từ “test” ở đây chỉ **phần validation của CV**, hoàn toàn khác test giữ riêng của bài. `float(...)` đổi số NumPy thành số Python cho record/JSON. `search.best_params_` là tham số thắng; `search.best_estimator_` là Pipeline **đã refit trên toàn train**; `time.time()-start` là thời gian chạy. Record dùng `_threshold=0.5` với NB và `0.0` với SVM, vì hai loại score khác nhau. Kết quả chưa có `Accuracy`/`F1` trên test cho đến khi gọi `evaluate`.

**Một vòng cụ thể:** nếu chọn TF-IDF + SVM, CV thử tối đa 8 cặp n-gram/C. Với mỗi cặp, lặp năm fold: học `vocabulary_/idf_` và `w,b` trên bốn phần, cho score/nhãn phần thứ năm, tính F1 spam; lấy trung bình năm F1. Bộ tham số tốt nhất được fit lại trên toàn train. Như vậy số “CV F1” là trung bình validation, **không phải** F1 của model refit đo trên chính train.

## 5. `pipeline/modeling/imbalance.py` — ba đối chứng trên TF-IDF + SVM

Sau B1–B4, bài dùng cấu hình TF-IDF + SVM dẫn đầu nhóm cơ sở làm mốc B4. S1/S2/S3 là **ba mô hình/chiến lược riêng**, không chồng cả ba lên nhau. Đầu vào mỗi hàm vẫn chỉ là train/folds; test được giữ nguyên để đánh giá ở bước sau.

### Mã nguồn đầy đủ

```python
"""Giai đoạn 2: so ba cách riêng để ứng phó ham nhiều hơn spam.

S1 tăng trọng số lỗi spam; S2 giảm số ham được dùng để học;
S3 giữ dữ liệu nhưng đổi ngưỡng quyết định. Chỉ nhận train/fold, không xem test.
"""

import numpy as np
from sklearn.base import clone
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import ParameterGrid


def fit_balanced_svm(pre, X, y, folds, *, seed=42, cache=None, n_jobs=2):
    """S1: fit TF-IDF + SVM với class_weight='balanced' trên toàn train.

    Chỉ đổi trọng số lỗi giữa hai lớp. Không bớt mẫu, không đổi ngưỡng.
    Bên trong vẫn dùng CV để chọn n-gram/C như cấu hình cơ sở.
    """
    from pipeline.modeling.experiments import fit_experiment

    return fit_experiment(pre, "TF-IDF", "SVM", X, y, folds,
                          seed=seed, cache=cache, n_jobs=n_jobs,
                          class_weight="balanced", strategy="Trọng số balanced")


def sample_training_indices(y, indices, seed=42, ratio=2):
    """S2: giữ spam, lấy tối đa ratio ham/spam từ indices được phép.

    Khi chạy CV, indices chỉ là phần train của fold; validation/test giữ nguyên.
    """
    # np.asarray cho phép tra nhiều vị trí một lần bằng mảng chỉ số.
    # indices chỉ chứa dòng được phép lấy mẫu (train của fold/toàn train).
    indices = np.asarray(indices)
    # Tách vị trí spam=1 và ham=0 để giữ toàn bộ spam, chỉ giảm ham.
    spam = indices[np.asarray(y)[indices] == 1]
    ham = indices[np.asarray(y)[indices] == 0]
    rng = np.random.default_rng(seed)
    ham_count = min(ratio * len(spam), len(ham))
    sampled = rng.choice(ham, size=ham_count, replace=False)
    selected = np.concatenate([spam, sampled])
    # Trộn thứ tự để classifier không thấy tất cả spam rồi mới đến ham.
    rng.shuffle(selected)
    return selected


def mean_cv_scores(y, predictions, folds):
    """Tính P/R/F1 trên mỗi validation rồi lấy trung bình năm lượt CV.

    predictions là nhãn dự đoán ngoài fold cho các dòng train, KHÔNG phải test.
    """
    result = {}
    for name, metric in [("Precision", precision_score),
                         ("Recall", recall_score), ("F1", f1_score)]:
        fold_scores = []
        for _, valid in folds:
            # Chỉ so nhãn thật/dự đoán của phần validation ở lượt này.
            fold_scores.append(metric(y[valid], predictions[valid], zero_division=0))
        result[f"CV {name} (spam)"] = float(np.mean(fold_scores))
    return result


def undersampled_search(pipe, grid, X, y, folds, seed=42):
    """S2: tìm tham số bằng CV với sampling chỉ trên fold train.

    Đầu ra: mô hình fit trên train đã giảm mẫu, điểm CV, tham số, số mẫu fit.
    """
    rows = []
    # ParameterGrid biến lưới thành các tổ hợp cụ thể:
    # n-gram (1,1)/(1,2) × C 0.1/0.3/1/3 = 8 tổ hợp để thử.
    for params in ParameterGrid(grid):
        # Mỗi dòng train sẽ nhận một nhãn dự đoán từ lượt mà nó là validation.
        predictions = np.zeros(len(y), dtype=int)
        for train, valid in folds:
            # Lấy mẫu CHỈ từ train của fold; validation vẫn giữ phân bố gốc.
            selected = sample_training_indices(y, train, seed)
            assert not set(selected) & set(valid)
            # clone tạo bản CHƯA FIT; không mang từ vựng/trọng số của fold trước.
            model = clone(pipe).set_params(**params).fit(X[selected], y[selected])
            predictions[valid] = model.predict(X[valid])
        rows.append((mean_cv_scores(y, predictions, folds), params))
    # Chọn bộ tham số có trung bình F1 spam trên validation cao nhất.
    scores, params = max(rows, key=lambda row: row[0]["CV F1 (spam)"])
    # Sau lựa chọn, fit mô hình cuối của S2 trên tập train đã giảm ham.
    selected = sample_training_indices(y, np.arange(len(y)), seed)
    fitted = clone(pipe).set_params(**params).fit(X[selected], y[selected])
    return fitted, scores, params, len(selected)


def tune_svm_threshold(pipe, X, y, folds):
    """S3: chọn ngưỡng decision score từ dự đoán ngoài fold trên train.

    Giữ siêu tham số B4. Thử 199 phân vị 1–99%, ngưỡng 0 và hai biên;
    cùng F1 thì ưu tiên ngưỡng gần 0. Đây không phải nested CV độc lập.
    Đầu ra: mô hình fit trên toàn train, điểm CV và ngưỡng score.
    """
    # scores[i] là decision score của dòng train i từ model không fit trên i.
    scores = np.zeros(len(y))
    covered = np.zeros(len(y), dtype=int)
    for train, valid in folds:
        model = clone(pipe).fit(X[train], y[train])
        scores[valid] = model.decision_function(X[valid])
        covered[valid] += 1
    # Mỗi dòng train phải được dự đoán ngoài fold đúng một lần.
    assert (covered == 1).all()
    # Thử 199 mức trải đều theo phân vị của score OOF, mốc 0 mặc định
    # và hai biên để không bỏ sót trường hợp rất ít/nhiều mẫu thành spam.
    thresholds = [0.0]
    thresholds.extend(np.quantile(scores, np.linspace(0.01, 0.99, 199)))
    thresholds.extend([np.nextafter(scores.min(), -np.inf), scores.max()])
    thresholds = np.unique(thresholds)
    rows = []
    for threshold in thresholds:
        predictions = (scores > threshold).astype(int)
        metrics = mean_cv_scores(y, predictions, folds)
        rows.append((metrics, float(threshold)))
    # Nếu F1 bằng nhau, ưu tiên ngưỡng gần 0 (ít lệch mốc SVM mặc định hơn).
    metrics, threshold = max(rows, key=lambda row: (row[0]["CV F1 (spam)"], -abs(row[1])))
    # Cuối cùng fit B4 trên toàn train; ngưỡng được lưu riêng trong record S3.
    return clone(pipe).fit(X, y), metrics, threshold
```

### S1: trọng số lớp

`fit_balanced_svm` gọi lại `fit_experiment` với `class_weight="balanced"`. Công thức trọng số scikit-learn cho lớp `c` là `n_samples/(n_classes×n_c)` tính từ **phần dữ liệu đang fit**; lớp ít mẫu nhận mức phạt lỗi lớn hơn. Hàm vẫn thử n-gram/C bằng CV, trả record S1 có model refit trên toàn train. S1 không đổi số dòng, không đổi ngưỡng score mặc định 0. `fit_experiment` được import ngay trong hàm; khi S1 được gọi, nó dùng đúng hàm thí nghiệm của B4 và thay duy nhất lựa chọn trọng số.

### S2: chỉ giảm ham trong phần được phép học

`sample_training_indices(y,indices,seed=42,ratio=2)` nhận nhãn toàn mảng train và danh sách vị trí **được phép lấy mẫu**. `np.asarray(indices)` chuyển danh sách vị trí thành mảng để tra cùng lúc. `np.asarray(y)[indices]==1` là mặt nạ Boolean chọn spam, `==0` chọn ham. `np.random.default_rng(seed)` tạo bộ lấy mẫu lặp lại được; `min(2×số spam,số ham)` quyết định số ham tối đa. `rng.choice(ham,size=...,replace=False)` bốc ham **không lặp lại**, `np.concatenate` ghép mọi spam với ham được bốc, `rng.shuffle` đảo thứ tự, trả chỉ số được giữ. Nó không xóa dòng khỏi CSV và không lấy mẫu từ validation/test.

`undersampled_search(pipe,grid,X,y,folds,seed)` tự chạy CV vì `GridSearchCV.fit` thông thường không tự đặt thao tác giảm mẫu **bên trong từng fold** theo cách bài này dùng. `ParameterGrid(grid)` chỉ liệt kê hai n-gram × bốn C = **8 dict tham số**, bản thân nó không fit. Với mỗi dict, `np.zeros(len(y))` chuẩn bị mảng dự đoán ngoài fold. Với mỗi `(train,valid)`: lấy mẫu trong `train`, kiểm không đụng `valid`, `clone(pipe)` tạo bản mới chưa fit, `set_params(**params)` gắn tham số, `fit(X[selected],y[selected])` học trên mẫu đã giảm, `predict(X[valid])` chấm validation nguyên bản. `mean_cv_scores` tính Precision/Recall/F1 trên từng fold rồi trung bình; `max(...CV F1...)` chọn bộ tham số. Sau đó lấy mẫu từ toàn train và fit lại model S2, trả `(fitted,scores,params,len(selected))`. **Không đánh giá test tại đây**.

`mean_cv_scores` duyệt ba hàm điểm. `for _,valid in folds` bỏ qua phần train, chỉ cắt `y[valid]` và `predictions[valid]`. `zero_division=0` trả 0 nếu Precision/Recall không tính được do không có dự đoán dương. `np.mean` lấy trung bình năm giá trị, nên “CV F1” không phải F1 gộp toàn bộ OOF.

### S3: giữ dữ liệu, đổi ngưỡng score

`tune_svm_threshold(pipe,X,y,folds)` nhận cấu hình B4 đã chọn nhưng dùng `clone(pipe)` để fit lại **một bản mới trên từng fold train**. `decision_function(X[valid])` trả score cho các câu mà bản đó chưa thấy; ghép lại thành `scores` ngoài fold (OOF). `covered[valid]+=1` và assert bảo đảm mỗi câu train được chấm đúng một lần. Score SVM có thể âm, ví dụ `-0.18`; đây không phải “xác suất âm”.

Hàm thử ngưỡng mặc định `0`, 199 phân vị score từ 1% đến 99%, hai biên và dùng `np.unique` bỏ mức trùng. Với mỗi ngưỡng, `(scores>threshold).astype(int)` biến score thành nhãn, `mean_cv_scores` đo trên năm validation. `max` chọn ngưỡng có CV F1 spam cao nhất; nếu bằng F1, ưu tiên ngưỡng gần 0 bằng khóa `-abs(threshold)`. Cuối cùng `clone(pipe).fit(X,y)` học model B4 trên **toàn train**, trả model, điểm CV và ngưỡng; ngưỡng được lưu riêng trong record/bundle. S3 **không đổi trọng số lớp hay giảm mẫu**. Vì ngưỡng được tối ưu trên chính score OOF dùng để báo điểm CV, CV F1 của S3 có thể hơi lạc quan; test riêng là phép kiểm sau lựa chọn.

## 6. `pipeline/modeling/evaluation.py` — đo test sau khi đã chọn

**Đầu vào:** record đã có mô hình fit, chuỗi token test `X`, nhãn thật test `y`. **Đầu ra:** record được thêm chỉ số test và các mảng dự đoán/score để vẽ biểu đồ, phân tích lỗi. File này **không gọi `fit`** và không đổi cấu hình được chọn.

### Mã nguồn đầy đủ

```python
"""Giai đoạn 3: dự đoán và đo kết quả trên tập test đã giữ riêng.

Không hàm nào ở đây fit hoặc chọn lại cấu hình dựa vào test.
"""

import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             precision_score, recall_score)


def predict_record(record, X):
    """Trả (nhãn dự đoán, điểm liên tục) cho các câu trong X.

    X là chuỗi token; pipeline trong record tự biến chúng thành vector.
    record["_model"] là pipeline đã fit; record["_threshold"] là mốc quyết định.
    NB dùng xác suất spam (mặc định >=0,5); SVM dùng decision score (>0).
    """
    model = record["_model"]
    if hasattr(model, "predict_proba"):
        # Naive Bayes có predict_proba: lấy cột thứ 2, lớp 1 = spam.
        scores = model.predict_proba(X)[:, 1]
        threshold = record.get("_threshold", 0.5)
        predictions = (scores >= threshold).astype(int)
    else:
        # LinearSVC không có predict_proba: score âm/dương KHÔNG là xác suất.
        # S3 có thể đổi ngưỡng 0 sang một giá trị khác đã chọn trên train.
        scores = model.decision_function(X)
        threshold = record.get("_threshold", 0.0)
        predictions = (scores > threshold).astype(int)
    return predictions, scores


def evaluate(record, X, y):
    """Gắn chỉ số test vào một record đã có model/điểm CV.

    X: chuỗi token của test; y: nhãn thật tương ứng (ham=0, spam=1).
    Record có cấu hình, điểm CV, model đã fit và ngưỡng. Hàm thêm các điểm
    test bên dưới, không biến điểm CV thành điểm test và không chọn lại model.
    """
    pred, score = predict_record(record, X)
    # Accuracy: số đoán đúng / tổng số test; dễ gây lạc quan khi ham nhiều.
    record["Accuracy"] = accuracy_score(y, pred)
    # Ba chỉ số sau chỉ xét lớp dương 1=spam; zero_division=0 khi không
    # dự đoán ra spam nào, tránh phép chia cho 0.
    record["Precision"] = precision_score(y, pred, zero_division=0)
    record["Recall"] = recall_score(y, pred, zero_division=0)
    record["F1"] = f1_score(y, pred, zero_division=0)
    # AP dùng thứ tự các score trên nhiều ngưỡng, không dùng nhãn tại một ngưỡng.
    record["AP"] = average_precision_score(y, score)
    # Giữ nhãn và score từng dòng để vẽ ma trận, đường PR, phân tích lỗi.
    record["_pred"] = pred
    record["_score"] = score


def boot_ci(y, pred, n=500, seed=0):
    """Ước lượng khoảng tham khảo 95% cho F1 của test đã có.

    Mỗi lượt rút len(y) dòng CÓ hoàn lại từ test, tính F1; lặp n=500 lần.
    Không huấn luyện lại và không chứng minh chất lượng ở nguồn dữ liệu mới.
    """
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n):
        # Rút lại len(y) vị trí có hoàn lại; một dòng có thể xuất hiện nhiều lần.
        indices = rng.integers(0, len(y), len(y))
        values.append(f1_score(y[indices], pred[indices], zero_division=0))
    # Lấy hai phân vị biên của 500 điểm F1 đã rút mẫu.
    return np.percentile(values, [2.5, 97.5])
```

### Từng hàm và các hàm thư viện bên trong

**`predict_record(record,X)`:** `record["_model"]` lấy Pipeline đã fit. `hasattr(model,"predict_proba")` hỏi mô hình có phương thức xác suất không. Với NB, `model.predict_proba(X)` thực chất gọi `vec.transform(X)` rồi `MultinomialNB.predict_proba`; kết quả có hai cột vì lớp đã được xác nhận là 0/1, `[:,1]` lấy cột **spam=1**. So `scores>=threshold`, mặc định 0,5, rồi `astype(int)` biến Boolean thành 0/1. Với SVM, Pipeline gọi `decision_function(X)`; hàm lấy score có dấu, so `scores>threshold`, mặc định 0. S3 đưa ngưỡng đã chọn trên train vào `record["_threshold"]`. Trả cả `(predictions,scores)` vì nhãn dùng tính F1/ma trận nhầm lẫn, còn score liên tục dùng AP và đường Precision–Recall. `predict_proba` của NB là ước lượng của mô hình, không cam kết đã hiệu chỉnh xác suất thực tế.

**`evaluate(record,X,y)`:** gọi `predict_record` rồi **thêm** các khóa vào cùng dict `record`; hàm không trả record mới. Các hàm `accuracy_score`, `precision_score`, `recall_score`, `f1_score` nhận `y_true` và `y_pred` có cùng số dòng. Đặt `TP` = spam đoán spam, `FP` = ham đoán spam, `FN` = spam đoán ham, `TN` = ham đoán ham:

| Điểm | Phép tính | Vì sao dùng |
|---|---|---|
| Accuracy | `(TP+TN)/(TP+TN+FP+FN)` | Tỷ lệ đúng tổng; ham nhiều nên một mô hình luôn đoán ham cũng có thể đạt Accuracy cao. |
| Precision spam | `TP/(TP+FP)` | Trong số câu báo spam, bao nhiêu thật sự spam; liên quan cảnh báo nhầm. |
| Recall spam | `TP/(TP+FN)` | Trong số spam thật, phát hiện được bao nhiêu; liên quan bỏ sót. |
| F1 spam | `2×P×R/(P+R)` | Cân bằng hai điểm trên tại **ngưỡng đang dùng**. |
| AP | Tóm tắt thứ tự xếp hạng theo đường Precision–Recall khi quét nhiều ngưỡng | Dùng `score` liên tục; khác F1 tại một ngưỡng. |

`zero_division=0` cho điểm 0 nếu mẫu dự đoán khiến mẫu số bằng 0, tránh lỗi chia 0. `average_precision_score(y,score)` dùng nhãn thật và score, không dùng nhãn nhị phân đã cắt. `record["_pred"]`, `record["_score"]` giữ một giá trị trên mỗi dòng test để các cell biểu đồ/lỗi sử dụng. `Accuracy/Precision/Recall/F1/AP` ở đây là **test**, còn các khóa `"CV ..."` đã có từ trước là **validation của train**.

**`boot_ci(y,pred,n=500,seed=0)`:** `np.random.default_rng(seed)` tạo bộ sinh ngẫu nhiên lặp lại được; mỗi lượt `rng.integers(0,len(y),len(y))` rút số lượng chỉ số bằng kích thước test **có hoàn lại**, một câu có thể xuất hiện nhiều lần hoặc không xuất hiện. Tính F1 trên tập rút lại, lặp 500 lần; `np.percentile(values,[2.5,97.5])` trả hai mốc tham khảo 95%. Đây là bootstrap của **dự đoán test đã có**, không fit lại model và không bảo đảm khoảng đúng cho dữ liệu Facebook tương lai khác nguồn.

## 7. `pipeline/modeling/predict.py` — bình luận mới đi qua mô hình đã lưu

**Đầu vào:** một chuỗi hoặc danh sách bình luận **thô**, cộng `model_bundle` đã nạp từ `.joblib`. **Đầu ra:** DataFrame có câu gốc, score, loại score và nhãn ham/spam. File này không tạo mô hình; mô hình được học ở `search.fit` hoặc các hàm S1–S3, sau đó Notebook 02/script lưu bằng `joblib.dump`.

### Mã nguồn đầy đủ

```python
"""Giai đoạn 4: dự đoán bình luận mới bằng mô hình truyền thống đã học.

Notebook 02 hoặc scripts/run_phi_pipeline.py gọi joblib.dump để tạo .joblib.
File này chỉ nhận bundle đã nạp, áp cùng tiền xử lý và ngưỡng để dự đoán;
không huấn luyện, lưu file, tạo lại từ vựng/IDF hay gọi AI gán nhãn.
"""

import numpy as np
import pandas as pd

from pipeline.modeling.evaluation import predict_record
from pipeline.prepare.preprocess_for_model import preprocess


def predict_spam(texts, model_bundle):
    """Nhận một câu hoặc danh sách câu thô; trả bảng nhãn/điểm từng câu.

    model_bundle chứa: model đã fit, threshold, remove_stop.
    Bảng trả về: text gốc, score, loại score, nhãn dự đoán spam/ham.
    """
    # Một chuỗi là MỘT bình luận; list(texts) trực tiếp sẽ tách từng ký tự.
    texts = [texts] if isinstance(texts, str) else list(texts)
    if not texts:
        return pd.DataFrame(columns=["text", "score", "score_type", "du_doan"])
    # Dùng cùng cách chuẩn bị câu và lựa chọn từ dừng của lúc train.
    X = [preprocess(text, text, model_bundle["remove_stop"]) for text in texts]
    # Model trong bundle gồm cả vectorizer đã học: X vẫn là CHUỖI token.
    # predict_record gọi pipeline để biến X thành số, lấy score, áp ngưỡng.
    record = {"_model": model_bundle["model"]}
    if "threshold" in model_bundle:
        record["_threshold"] = model_bundle["threshold"]
    pred, score = predict_record(record, X)
    # NB: score là xác suất ước lượng; SVM: score là điểm có dấu, không phải
    # xác suất. Nhãn 1 đổi thành "spam", nhãn 0 thành "ham" để dễ đọc.
    return pd.DataFrame({"text": texts, "score": score,
        "score_type": "probability_spam" if hasattr(model_bundle["model"], "predict_proba") else "decision_score",
        "du_doan": np.where(pred == 1, "spam", "ham")})
```

### Đọc lần lượt từng lệnh

| Lệnh | Tác dụng và kết quả |
|---|---|
| `import numpy as np` | Dùng `np.where` để đổi nhãn số 0/1 thành chữ. |
| `import pandas as pd` | Tạo bảng kết quả `DataFrame`, mỗi câu một hàng. |
| `from ...evaluation import predict_record` | Dùng lại đúng quy tắc NB/SVM/ngưỡng của lúc đánh giá test, tránh hai cách cắt nhãn khác nhau. |
| `from ...preprocess_for_model import preprocess` | Dùng hàm chuẩn bị văn bản đã có của project. `preprocess(cleaned,original,remove_stop)` trả chuỗi token; phần triển khai tiền xử lý thuộc giai đoạn trước của nhóm. Ở demo một câu mới, cùng câu thô được đưa vào hai tham số đầu để hàm tự làm sạch và giữ thông tin emoji. |
| `texts=[texts] if isinstance(texts,str) else list(texts)` | Một chuỗi là **một bình luận**; nếu gọi `list` thẳng trên chuỗi sẽ thành từng ký tự. Iterable nhiều câu được đổi thành danh sách. |
| `if not texts: return pd.DataFrame(...)` | Danh sách rỗng cho bảng rỗng đúng cấu trúc, không gọi mô hình với 0 câu. |
| `X=[preprocess(text,text,model_bundle["remove_stop"]) for text in texts]` | Tạo chuỗi token từng câu bằng đúng lựa chọn giữ/bỏ từ dừng đã lưu. `X` vẫn là **chuỗi**, chưa phải vector. |
| `record={"_model":model_bundle["model"]}` | Đưa Pipeline đã fit vào cấu trúc mà `predict_record` hiểu; nếu bundle có `threshold` thì gắn thêm ngưỡng. |
| `pred,score=predict_record(record,X)` | Pipeline biến chuỗi thành vector bằng từ vựng/IDF cũ, NB/SVM cho score, so ngưỡng ra nhãn. |
| `"score_type": ...` | NB có `predict_proba` nên ghi `probability_spam`; SVM ghi `decision_score` để người đọc không nhầm score âm thành xác suất. |
| `np.where(pred==1,"spam","ham")` | Đổi từng nhãn số thành chữ, trả bảng giữ cả văn bản gốc. |

**Một câu mới đi như nào?** Gọi `joblib.load(model_path)` lấy bundle → `predict_spam("Cho mình hỏi lịch thi...",bundle)` → `preprocess` tạo chuỗi token → `bundle["model"].named_steps["vec"].transform` dùng từ vựng/IDF đã học → `clf.decision_function` hoặc `predict_proba` tạo score → so với ngưỡng đã lưu → bảng một dòng. Không có nhãn thật của câu mới, nên hàm **không thể tính Accuracy/F1**. `.joblib` lưu trạng thái Python của vectorizer + classifier + ngưỡng/chính sách từ dừng; file `.py` giữ logic để nạp và chạy chúng.

## 8. `scripts/run_phi_pipeline.py` — file chạy thẳng toàn bộ phần Phi

Đây là **bộ điều phối**, không phải thuật toán thứ tám. Nó gọi các hàm vừa học theo thứ tự, cho phép kiểm phần Phi độc lập với giao diện Notebook. Lệnh từ gốc project: `.\.venv\Scripts\python.exe scripts/run_phi_pipeline.py`. Mặc định script đọc `data/processed/v7_preprocessed.csv` và ghi vào `artifacts/spamham/phi_pipeline/` để không ghi đè kết quả Notebook 02.

### Mã nguồn đầy đủ

```python
r"""Chạy phần Phi từ CSV đã có nhãn/token đến mô hình truyền thống.

Lệnh ở gốc project:
    .\.venv\Scripts\python.exe scripts/run_phi_pipeline.py

Đọc file từ trên xuống: nhận CSV → chia tập → CV → lệch lớp → test → lưu.
Không gọi Facebook, Argilla hoặc viBERT. Kết quả mặc định ghi vào thư mục
artifacts/spamham/phi_pipeline/ để không ghi đè file của Notebook 02.
"""

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from joblib import Memory


# Khi chạy file trong scripts/, thêm gốc project để import package pipeline/.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.modeling.predict import predict_spam
from pipeline.modeling.classifiers import MODELS
from pipeline.modeling.evaluation import evaluate, predict_record
from pipeline.modeling.experiments import fit_experiment, make_pipe
from pipeline.modeling.imbalance import (fit_balanced_svm, sample_training_indices,
                                         undersampled_search, tune_svm_threshold)
from pipeline.modeling.text_features import FEATURES
from pipeline.modeling.training_utils import grouped_folds, split_train_test


# argparse là thư viện chuẩn để người dùng đổi đường dẫn/số tiến trình khi cần.
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--csv", type=Path, default=PROJECT_ROOT / "data/processed/v7_preprocessed.csv")
parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "artifacts/spamham/phi_pipeline")
parser.add_argument("--jobs", type=int, default=2, help="Số tiến trình GridSearchCV")
args = parser.parse_args()

# Giống Notebook 02: cố định seed toàn cục trước mọi thao tác có yếu tố ngẫu nhiên.
np.random.seed(42)

csv_path = args.csv.resolve()
output_dir = args.output.resolve()
output_dir.mkdir(parents=True, exist_ok=True)

# GIAI ĐOẠN 0 — Nhận CSV đã có nhãn/token; chưa học đặc trưng/mô hình.
data = pd.read_csv(csv_path, keep_default_na=False, dtype={"id": str})
required = {"id", "label", "text", "cleaned_text", "tokens_keep_stopwords", "tokens_remove_stopwords"}
missing = required - set(data.columns)
if missing:
    raise ValueError(f"CSV thiếu cột: {sorted(missing)}")

data["label"] = data["label"].str.strip().str.lower()
if set(data["label"]) != {"ham", "spam"}:
    raise ValueError("CSV phải có đúng hai nhãn ham và spam")
if not data["id"].is_unique or data["id"].str.strip().eq("").any():
    raise ValueError("ID phải duy nhất và không rỗng")

# Hai chuỗi token đã được chuẩn bị sẵn; Phi chỉ chọn nhánh bằng CV ở giai đoạn 1.
text = {"giữ stop words": data["tokens_keep_stopwords"],
        "có loại stop words": data["tokens_remove_stopwords"]}
y_all = data["label"].eq("spam").astype(int).to_numpy()  # ham=0, spam=1

# Hàm chia giữ các câu trùng khóa cùng phía và trả vị trí trong TOÀN CSV.
keys, train_idx, test_idx = split_train_test(data["cleaned_text"], y_all)
y_train = y_all[train_idx]
y_test = y_all[test_idx]
if len(np.unique(y_train)) != 2 or len(np.unique(y_test)) != 2:
    raise ValueError("Train/test phải có cả ham và spam")
assert not set(keys[train_idx]) & set(keys[test_idx])

# Mỗi cặp (train, validation) bên dưới dùng chỉ số trong MẢNG TRAIN.
folds = grouped_folds(keys[train_idx], y_train)
print(f"Giai đoạn 0: {len(train_idx)} train, {len(test_idx)} test, {len(folds)} fold CV")

# Cache tạm giảm thời gian tính vector lặp lại; không phải file mô hình cuối.
with tempfile.TemporaryDirectory(prefix="phi_cache_", dir=output_dir) as temp_dir:
    cache = Memory(temp_dir, verbose=0)

    # GIAI ĐOẠN 1 — 2 nhánh token × 2 đặc trưng × 2 bộ phân loại = 8 CV.
    preprocess_results = []
    for pre, token_series in text.items():
        X_train = token_series.iloc[train_idx].to_numpy()
        for feature in FEATURES:
            for algorithm in MODELS:
                record = fit_experiment(pre, feature, algorithm, X_train, y_train, folds,
                                        seed=42, cache=cache, n_jobs=args.jobs)
                preprocess_results.append(record)
        print(f"Giai đoạn 1: xong CV cho nhánh {pre}", flush=True)

    # Chọn nhánh có ứng viên F1 spam cao nhất; chỉ nhánh đó được đánh mã B1–B4.
    main_pre = max(preprocess_results, key=lambda r: r["CV F1 (spam)"])["Tiền xử lý"]
    baselines = [r for r in preprocess_results if r["Tiền xử lý"] == main_pre]
    for number, record in enumerate(baselines, 1):
        record["Mã"] = f"B{number}"
    focus = max(baselines, key=lambda r: r["CV F1 (spam)"])
    if (focus["Đặc trưng"], focus["Thuật toán"]) != ("TF-IDF", "SVM"):
        raise RuntimeError("B4 không còn dẫn đầu; cần xem lại phạm vi khảo sát S1–S3")
    print(f"Giai đoạn 1: chọn {main_pre}; đối chứng {focus['Mã']} TF-IDF + SVM")

    # GIAI ĐOẠN 2 — Thử ba chiến lược riêng trên TF-IDF + SVM.
    X_train = text[main_pre].iloc[train_idx].to_numpy()
    s1 = fit_balanced_svm(main_pre, X_train, y_train, folds,
                          seed=42, cache=cache, n_jobs=args.jobs)
    s1["Mã"] = "S1"

    pipe, grid, _ = make_pipe("TF-IDF", "SVM", seed=42, cache=cache, n_jobs=args.jobs)
    start = time.time()
    s2_model, s2_cv, s2_params, s2_count = undersampled_search(pipe, grid, X_train, y_train, folds, 42)
    s2 = {"Mã": "S2", "Tiền xử lý": main_pre, "Đặc trưng": "TF-IDF", "Thuật toán": "SVM",
          "Chiến lược": "Under-sampling 2:1", **s2_cv, "Tham số": s2_params,
          "Thời gian (s)": time.time() - start, "_model": s2_model, "_threshold": 0.0}
    assert s2_count == len(sample_training_indices(y_train, np.arange(len(y_train)), seed=42))

    start = time.time()
    s3_model, s3_cv, threshold = tune_svm_threshold(focus["_model"], X_train, y_train, folds)
    s3 = {"Mã": "S3", "Tiền xử lý": main_pre, "Đặc trưng": "TF-IDF", "Thuật toán": "SVM",
          "Chiến lược": "Điều chỉnh ngưỡng", **s3_cv,
          "Tham số": {**focus["Tham số"], "threshold": threshold},
          "Thời gian (s)": time.time() - start, "_model": s3_model, "_threshold": threshold}

    candidates = baselines + [s1, s2, s3]
    best = max(candidates, key=lambda r: r["CV F1 (spam)"])
    print(f"Giai đoạn 2: chọn {best['Mã']} theo CV F1 spam = {best['CV F1 (spam)']:.4f}")

    # GIAI ĐOẠN 3 — Đo trên test một lần sau khi đã chốt lựa chọn bằng CV.
    for record in candidates:
        X_test = text[record["Tiền xử lý"]].iloc[test_idx].to_numpy()
        evaluate(record, X_test, y_test)
    print(f"Giai đoạn 3: {best['Mã']} test P={best['Precision']:.4f}, "
          f"R={best['Recall']:.4f}, F1={best['F1']:.4f}")

    # Chỉ xuất các cột có thể ghi CSV; _model và mảng _pred/_score ở trong RAM.
    columns = ["Mã", "Tiền xử lý", "Đặc trưng", "Thuật toán", "Chiến lược",
               "CV Precision (spam)", "CV Recall (spam)", "CV F1 (spam)",
               "Accuracy", "Precision", "Recall", "F1", "AP", "Tham số", "Ngưỡng"]
    rows = []
    for record in candidates:
        row = {}
        for name in columns:
            row[name] = record["_threshold"] if name == "Ngưỡng" else record[name]
        rows.append(row)
    pd.DataFrame(rows, columns=columns).to_csv(output_dir / "metrics.csv", index=False,
                                                encoding="utf-8-sig")

    # GIAI ĐOẠN 4 — Lưu pipeline đã fit + ngưỡng + quy tắc từ dừng.
    bundle = {"model": best["_model"], "threshold": best["_threshold"],
              "remove_stop": best["Tiền xử lý"] == "có loại stop words"}
    # Bỏ đường cache tạm trước khi lưu để file .joblib độc lập với thư mục tạm.
    bundle["model"].memory = None
    model_path = output_dir / "spam_ham_ml_only.joblib"
    joblib.dump(bundle, model_path)

    # Nạp lại và xác nhận nhãn trên test không đổi sau khi lưu.
    loaded = joblib.load(model_path)
    X_test = text[best["Tiền xử lý"]].iloc[test_idx].to_numpy()
    reloaded_pred, _ = predict_record({"_model": loaded["model"],
                                      "_threshold": loaded["threshold"]}, X_test)
    assert np.array_equal(reloaded_pred, best["_pred"])

    # Bảng này cho phép đối chiếu từng ID test và nhãn dự đoán của model chọn.
    pd.DataFrame({"id": data.iloc[test_idx]["id"].to_numpy(),
                  "label": data.iloc[test_idx]["label"].to_numpy(),
                  "prediction": np.where(best["_pred"] == 1, "spam", "ham"),
                  "score": best["_score"]}).to_csv(output_dir / "test_predictions.csv",
                                                    index=False, encoding="utf-8-sig")
    summary = {"source_csv": str(csv_path), "train_rows": len(train_idx),
               "test_rows": len(test_idx), "selected": best["Mã"],
               "cv_f1_spam": float(best["CV F1 (spam)"]), "test_f1_spam": float(best["F1"]),
               "threshold": float(best["_threshold"])}
    (output_dir / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                                  encoding="utf-8")
    print("Giai đoạn 4: đã lưu", model_path)
    print(predict_spam(["Cho mình hỏi lịch thi với ạ",
                        "Liên hệ mua bán qua 0901234567"], loaded).to_string(index=False))
```

### Từng cụm lệnh theo thứ tự chạy

| Cụm lệnh | Đầu vào → xử lý → đầu ra |
|---|---|
| `argparse`, `Path`, `sys.path` | `argparse` đọc `--csv`, `--output`, `--jobs` từ lệnh gọi. `Path(__file__).resolve().parents[1]` tìm gốc project; thêm vào `sys.path` để import được `pipeline` khi chạy file ở `scripts/`. |
| `json`, `tempfile`, `time` | `json` ghi tóm tắt lượt chạy; `TemporaryDirectory` tạo cache tự xóa khi thoát khối `with`; `time.time()` đo thời gian S2/S3. |
| `joblib` và `Memory` | `Memory` cache bước vector hóa trong lúc CV; `joblib.dump/load` lưu và nạp **mô hình cuối**. Cache tạm khác file `.joblib` cuối. |
| `numpy`, `pandas` | Mảng nhãn/chỉ số và đọc, tạo, ghi bảng CSV. `pd.read_csv(...,keep_default_na=False,dtype={"id":str})` giữ ID dạng chuỗi và không tự biến ô rỗng thành NaN; code kiểm cột, nhãn, ID trước khi huấn luyện. |
| `np.random.seed(42)` | Cố định phần ngẫu nhiên toàn cục; S2 còn dùng bộ sinh riêng với `seed=42`. Cùng seed chỉ giúp lặp lại khi dữ liệu/phiên bản/môi trường tương đương. |
| `required - set(data.columns)` | Phép trừ hai tập tên cột để tìm cột thiếu; báo lỗi sớm. `.str.strip().str.lower()` chuẩn hóa **nhãn chữ**, không làm sạch lại bình luận. `.eq("spam").astype(int).to_numpy()` tạo `y_all` 0/1. |
| `text={...}` | Trỏ đến hai chuỗi token đã có để CV chọn nhánh giữ hoặc bỏ stop words; script không tự tokenize lại. |
| `split_train_test`, `grouped_folds` | Tạo chỉ số toàn CSV rồi năm cặp chỉ số trong train. Các kiểm `np.unique`/`assert` trong script là kiểm thêm, không phải phương pháp chia mới. |
| `with TemporaryDirectory(...) as temp_dir` | Tạo `cache=Memory(...)` chỉ tồn tại trong khối; sau khi chạy xong cache bị dọn. |
| Ba vòng `for pre → feature → algorithm` | Hai nhánh từ dừng × BoW/TF-IDF × NB/SVM = **8 lượt thí nghiệm**. Mỗi lượt `fit_experiment` tự thử n-gram và alpha/C bằng năm fold CV; `preprocess_results` chứa tám record. |
| `main_pre=max(...)` | Chọn nhánh từ dừng theo **CV F1 spam cao nhất**, không nhìn test. `baselines` chỉ lấy bốn record thuộc nhánh thắng, gán B1–B4. `focus` là B cao nhất, kiểm đúng TF-IDF + SVM trước khi thử S1–S3. Nếu không đúng, script báo lỗi để người làm xem lại phạm vi khảo sát thay vì âm thầm gọi sai B4. |
| `s1`, `s2`, `s3` | Gọi ba chiến lược riêng, gán tên và lưu `_model`/`_threshold` vào record. `**s2_cv`/`**s3_cv` trải từ điển các điểm CV vào record; `**focus["Tham số"]` giữ tham số B4 khi thêm ngưỡng S3. |
| `candidates=baselines+[s1,s2,s3]` và `best=max(...)` | Bảy ứng viên cuối; chọn bằng CV F1 **trước** vòng `evaluate`. |
| Vòng `evaluate(record,X_test,y_test)` | Cắt đúng chuỗi token của mỗi record theo `test_idx`; thêm chỉ số test. Không lựa chọn lại `best` dựa vào test. |
| `columns`, `rows`, `to_csv("metrics.csv")` | Chỉ xuất trường số/chữ của bảy record; không nhét đối tượng `_model` hay mảng dự đoán vào CSV. `encoding="utf-8-sig"` giúp Excel mở tiếng Việt dễ hơn. |
| `bundle={"model":...,"threshold":...,"remove_stop":...}` | Gói Pipeline đã fit, ngưỡng và chính sách từ dừng. `bundle["model"].memory=None` bỏ đường cache tạm trước khi ghi, vì thư mục tạm sắp bị xóa. |
| `joblib.dump`, `joblib.load`, `np.array_equal` | Ghi `spam_ham_ml_only.joblib`, nạp lại và kiểm **mọi nhãn test** không đổi sau lưu. Đây là kiểm file thực sự dùng được, không phải chỉ in tên file. |
| `test_predictions.csv` | Mỗi dòng test: ID, nhãn thật, nhãn dự đoán của `best`, score; dùng phân tích FP/FN. |
| `run_summary.json` | Ghi đường CSV, số train/test, mô hình chọn, CV F1, test F1, ngưỡng; không chứa toàn bộ đối tượng model. |
| `predict_spam([...],loaded)` | Demo bình luận mới qua **model đã nạp lại**, khác với chấm test vì không có nhãn thật để tính điểm. |

## 9. Notebook 02: các cell tương ứng phần Phi

Notebook là cách **trình bày và chạy chính các hàm trên**, không phải một bộ thuật toán khác. Chạy cell theo thứ tự sẽ thực hiện thật các bước; riêng việc huấn luyện lại tốn thời gian. Khi học phần Phi, cần biết các cell trước đó đã đưa `raw`, `TEXT` và `y_all` vào RAM. Code của các cell liên quan được chép dưới đây; phần biểu đồ/nhận xét dùng lại record và score do Phi tạo.

### Cell 11 — chia tập

```python
keys, train_idx, test_idx = split_train_test(raw["cleaned_text"], y_all)
y_train, y_test = y_all[train_idx], y_all[test_idx]
CV_FOLDS = grouped_folds(keys[train_idx], y_train)

split_table = pd.DataFrame([
    {"Tập": name, "Số mẫu": len(labels), "Số mẫu spam": int(labels.sum()), "Tỷ lệ spam (%)": labels.mean() * 100}
    for name, labels in [("Train", y_train), ("Test", y_test)]
]).set_index("Tập")
display(split_table.round(2))
fold_table = pd.DataFrame([
    {"Fold": i + 1, "Số mẫu huấn luyện": len(tr), "Số mẫu validation": len(va),
     "Số mẫu spam trong validation": int(y_train[va].sum())}
    for i, (tr, va) in enumerate(CV_FOLDS)
]).set_index("Fold")
display(fold_table)
print("Mỗi dòng là một lượt CV; phần validation không được dùng để fit mô hình của lượt đó.")
```

`raw["cleaned_text"]` là câu đã sạch để tạo **khóa chia**, `y_all` là nhãn số toàn CSV. `y_all[train_idx]`/`y_all[test_idx]` lấy nhãn đúng vị trí. `grouped_folds(keys[train_idx],y_train)` chia **bên trong train**. Hai DataFrame chỉ để hiển thị số mẫu và tỷ lệ spam: `labels.sum()` đếm nhãn 1, `labels.mean()*100` cho phần trăm; bảng fold dùng `y_train[va].sum()` đếm spam của mỗi validation. `set_index` chọn cột làm nhãn hàng, `round` làm tròn hiển thị. Cell này không fit mô hình.

### Cell 13 — tám lượt CV, chốt bốn cấu hình cơ sở

```python
# text_features.py + classifiers.py tạo hai bước; experiments.py ghép và fit CV.
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

cv_comparison = pd.DataFrame(PREPROCESS_RESULTS).pivot(
    index=["Đặc trưng", "Thuật toán"], columns="Tiền xử lý", values="CV F1 (spam)")
cv_comparison = cv_comparison.rename(columns={"giữ stop words": "CV F1 — giữ từ dừng",
                                             "có loại stop words": "CV F1 — loại từ dừng"})
display(cv_comparison.round(4))
MAIN_PRE = max(PREPROCESS_RESULTS, key=lambda r: r["CV F1 (spam)"])["Tiền xử lý"]
RESULTS = [r for r in PREPROCESS_RESULTS if r["Tiền xử lý"] == MAIN_PRE]
for i, record in enumerate(RESULTS, 1):
    record["Mã"] = f"B{i}"
CV_COLS = ["CV Precision (spam)", "CV Recall (spam)", "CV F1 (spam)"]
display(pd.DataFrame(RESULTS)[["Mã", "Đặc trưng", "Thuật toán", *CV_COLS]].round(4))
display(pd.DataFrame(RESULTS)[["Mã", "Tham số"]])
focus = max(RESULTS, key=lambda r: r["CV F1 (spam)"])
assert (focus["Đặc trưng"], focus["Thuật toán"]) == ("TF-IDF", "SVM"), "Cần xem lại phạm vi khảo sát nếu ứng viên dẫn đầu thay đổi."
display(Markdown(f"**Cấu hình tiền xử lý:** {MAIN_PRE}. **Ứng viên dẫn đầu:** "
                 f"{focus['Mã']} — TF-IDF + SVM, CV F1 = {focus['CV F1 (spam)']:.4f}. "
                 "Các phần tiếp theo sử dụng cấu hình từ dừng này."))
```

`TemporaryDirectory`/`Memory` tạo cache tạm. Ba vòng gọi `fit_experiment` cho 8 lượt (mỗi lượt còn có lưới tham số bên trong). `cv_comparison.pivot` chỉ xoay tám record thành bảng so hai nhánh từ dừng; không tính điểm mới. `MAIN_PRE=max(...)` chọn nhánh theo CV F1 cao nhất, `RESULTS` giữ bốn cấu hình B1–B4 của nhánh đó. `CV_COLS` dùng để hiển thị Precision/Recall/F1 **validation**. `focus` chọn B cao nhất, assert bảo đảm nó là TF-IDF + SVM trước khi gắn S1–S3. `display`/`Markdown` chỉ định dạng output cho người xem, không thay model.

### Cell 15 — minh họa dữ liệu train bị lệch lớp

```python
sampled_indices = sample_training_indices(y_train, np.arange(len(y_train)), seed=SEED, ratio=2)
before_counts = pd.Series(np.where(y_train == 1, "spam", "ham")).value_counts().reindex(["ham", "spam"])
after_counts = pd.Series(np.where(y_train[sampled_indices] == 1, "spam", "ham")).value_counts().reindex(["ham", "spam"])
sampling_table = pd.DataFrame({"Train nguyên bản": before_counts, "Sau under-sampling": after_counts})
weights = len(y_train) / (2 * before_counts)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
sampling_table.plot.bar(ax=axes[0], rot=0, color=["#4c78a8", "#f2a541"])
axes[0].set(title="Số mẫu huấn luyện", ylabel="Số bình luận", xlabel="Nhãn")
weight_table = pd.DataFrame({"Không trọng số lớp": [1.0, 1.0], "Balanced": weights}, index=["ham", "spam"])
weight_table.plot.bar(ax=axes[1], rot=0, color=["#4c78a8", "#f2a541"])
axes[1].set(title="Trọng số trên toàn bộ train", ylabel="Trọng số", xlabel="Nhãn")
for ax in axes:
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f" if ax is axes[1] else "%.0f", padding=3, fontsize=9)
    ax.set_ylim(top=ax.get_ylim()[1] * 1.18)
plt.tight_layout()
fig.savefig(ARTIFACT_DIR / "imbalance_training.png", dpi=140, bbox_inches="tight")
plt.show()
print("Mỗi fold tính lại trọng số và lấy mẫu từ phần train của fold; validation/test giữ nguyên.")
```

`np.arange(len(y_train))` là mọi vị trí trong train; `sample_training_indices` cho biết nếu áp S2 sẽ giữ vị trí nào. `np.where` đổi 0/1 thành chữ để `value_counts` đếm ham/spam trước và sau. `weights=len(y_train)/(2*before_counts)` minh họa công thức trọng số balanced cho hai lớp trên **toàn train**; khi CV thật, bộ học tính lại theo từng fold train. `plt.subplots` và `plot.bar` vẽ hai ảnh: số mẫu sau giảm ham và trọng số khi giữ dữ liệu. `savefig` lưu hình; cell này chủ yếu giải thích S1/S2, chưa fit S3 hay chấm test.

### Cell 16 — S1/S2/S3 và chọn bằng CV

```python
X_tr_text = TEXT[MAIN_PRE].iloc[train_idx].to_numpy()
# S1: giữ toàn bộ train, tăng trọng số cho lớp ít mẫu.
balanced = fit_balanced_svm(MAIN_PRE, X_tr_text, y_train, CV_FOLDS,
                            seed=SEED, cache=cache, n_jobs=N_JOBS)
balanced["Mã"] = "S1"

# S2: lấy mẫu lại chỉ trên fold train, rồi fit lại sau khi chọn tham số.
pipe, grid, _ = make_pipe("TF-IDF", "SVM", seed=SEED, cache=cache, n_jobs=N_JOBS)
start = time.time()
us_model, us_metrics, us_params, us_count = undersampled_search(pipe, grid, X_tr_text, y_train, CV_FOLDS, SEED)
under = {"Mã": "S2", "Tiền xử lý": MAIN_PRE, "Đặc trưng": "TF-IDF", "Thuật toán": "SVM",
         "Chiến lược": "Under-sampling 2:1", **us_metrics, "Tham số": us_params,
         "Thời gian (s)": time.time() - start, "_model": us_model, "_threshold": 0.0}
assert us_count == len(sampled_indices)

start = time.time()
# S3: giữ B4, tạo score OOF và chọn ngưỡng trước khi dùng test.
threshold_model, threshold_metrics, threshold = tune_svm_threshold(focus["_model"], X_tr_text, y_train, CV_FOLDS)
tuned = {"Mã": "S3", "Tiền xử lý": MAIN_PRE, "Đặc trưng": "TF-IDF", "Thuật toán": "SVM",
         "Chiến lược": "Điều chỉnh ngưỡng", **threshold_metrics,
         "Tham số": {**focus["Tham số"], "threshold": threshold},
         "Thời gian (s)": time.time() - start, "_model": threshold_model, "_threshold": threshold}
SUPPLEMENTARY = [balanced, under, tuned]
STRATEGIES = [focus, *SUPPLEMENTARY]
CANDIDATES = RESULTS + SUPPLEMENTARY
best = max(CANDIDATES, key=lambda r: r["CV F1 (spam)"])

strategy_cv = pd.DataFrame([
    {"Mã": r["Mã"], "Chiến lược": r["Chiến lược"], "Số mẫu train": us_count if r is under else len(y_train),
     "Ngưỡng SVM": r["_threshold"], **{k: r[k] for k in CV_COLS}} for r in STRATEGIES
])
display(strategy_cv.round(4))
display(pd.DataFrame(STRATEGIES)[["Mã", "Tham số"]])
strategy_cv.to_csv(ARTIFACT_DIR / "imbalance_cv.csv", index=False, encoding="utf-8-sig")
SELECTED_NAME = f"{best['Mã']} — {best['Đặc trưng']} + {best['Thuật toán']} — {best['Chiến lược']}"
display(Markdown(f"**Lựa chọn theo CV:** {SELECTED_NAME}. "
                 f"CV F1 spam = {best['CV F1 (spam)']:.4f}; ngưỡng = {best['_threshold']:.4f}. "
                 "Lựa chọn này được xác định trước phần đánh giá test."))
```

`X_tr_text` là chuỗi token train của nhánh từ dừng thắng. `fit_balanced_svm` tạo S1 và record CV. `make_pipe` tạo TF-IDF + SVM **chưa fit** cho S2, `undersampled_search` tự chạy CV với sampling chỉ trên fold train; `under` ghép model/điểm/ngưỡng 0 thành record. `tune_svm_threshold` tạo score OOF của B4, tìm ngưỡng cho S3; `tuned` ghi cùng cấu hình B4 cộng ngưỡng. `SUPPLEMENTARY` là ba chiến lược, `STRATEGIES` thêm B4 để so tác động, `CANDIDATES` gồm **bốn B + ba S**. `best=max(...CV F1...)` chốt trước test. `strategy_cv` và `imbalance_cv.csv` chỉ trình bày điểm CV, số mẫu/ngưỡng; không phải bảng điểm test.

### Cell 20, 22, 24, 26 — chấm test và hình đánh giá

```python
# Đo test một lần sau khi best đã được chốt bằng CV ở mục 5.1.
for record in CANDIDATES:
    evaluate(record, TEXT[record["Tiền xử lý"]].iloc[test_idx].to_numpy(), y_test)
METRICS = ["Accuracy", "Precision", "Recall", "F1", "AP"]
REPORT_COLS = ["Mã", "Tiền xử lý", "Đặc trưng", "Thuật toán", "Chiến lược", *CV_COLS,
               *METRICS, "Tham số", "Thời gian (s)"]
metrics = pd.DataFrame([{k: r[k] for k in REPORT_COLS} | {"Ngưỡng": r["_threshold"]} for r in CANDIDATES])
sel = RESULTS
display(metrics.iloc[:len(RESULTS)][["Mã", "Đặc trưng", "Thuật toán", *METRICS]].round(4))
display(Markdown(f"*Đối chứng luôn dự đoán ham:* đúng {int((y_test == 0).sum()):,}/{len(y_test):,} mẫu, "
                 f"Accuracy = {1 - y_test.mean():.2%}, Recall/F1 spam = 0. "
                 "Accuracy cao chưa đủ phản ánh khả năng phát hiện spam khi dữ liệu lệch lớp."))
display(Markdown(f"**Cấu hình được chọn bằng CV:** {SELECTED_NAME}. "
                 f"Test Precision = {best['Precision']:.4f}, Recall = {best['Recall']:.4f}, "
                 f"F1 = {best['F1']:.4f}."))
```

Cell 20 mới gọi `evaluate` trên test cho từng B/S. `metrics` gom CV và test thành bảng; `metrics.iloc[:len(RESULTS)]` hiện riêng bốn B. Đoạn “luôn dự đoán ham” là **mốc giải thích**: Accuracy bằng tỷ lệ ham, nhưng Recall/F1 spam bằng 0; không phải một cấu hình được huấn luyện. `best` đã được chọn từ cell 16 nên dòng thông báo test không dùng để chọn lại.

```python
balance_rows = []
for record in STRATEGIES:
    tn, fp, fn, tp = confusion_matrix(y_test, record["_pred"], labels=[0, 1]).ravel()
    balance_rows.append({"Mã": record["Mã"], "Chiến lược": record["Chiến lược"],
                         **{k: record[k] for k in METRICS}, "FP": int(fp), "FN": int(fn)})
imbalance_report = pd.DataFrame(balance_rows)
display(imbalance_report.round(4))
imbalance_report.to_csv(ARTIFACT_DIR / "imbalance_comparison.csv", index=False, encoding="utf-8-sig")
plot_report = imbalance_report.set_index("Chiến lược")
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
plot_report[["Precision", "Recall", "F1"]].plot.barh(ax=axes[0], color=["#4c78a8", "#f2a541", "#59a14f"])
axes[0].set(title="Precision, Recall và F1 của spam", xlabel="Giá trị", ylabel="", xlim=(0, 1))
axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize=9)
plot_report[["FP", "FN"]].plot.barh(ax=axes[1], color=["#b279a2", "#e45756"])
axes[1].set(title="Cảnh báo nhầm và spam bị bỏ sót", xlabel="Số bình luận", ylabel="")
for container in axes[1].containers:
    axes[1].bar_label(container, padding=3, fontsize=9)
axes[1].set_xlim(right=plot_report[["FP", "FN"]].to_numpy().max() * 1.2)
plt.tight_layout()
fig.savefig(ARTIFACT_DIR / "imbalance_results.png", dpi=140, bbox_inches="tight")
plt.show()
```

Cell 22 lấy điểm test của B4/S1/S2/S3 để so tác động; cùng một tập test nên có thể so Precision/Recall/F1 tại ngưỡng từng chiến lược. Nếu xem một chỉ số cao hơn, vẫn phải xem chỉ số còn lại và số FP/FN.

```python
def plot_confusions(records, nrows, ncols, title, filename):
    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 3.4 * nrows), squeeze=False)
    for ax, record in zip(axes.ravel(), records):
        matrix = confusion_matrix(y_test, record["_pred"], labels=[0, 1])
        ax.imshow(matrix, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{matrix[i,j]}\n({matrix[i,j]/matrix[i].sum():.1%})", ha="center", va="center",
                        color="white" if matrix[i,j] > matrix.max()/2 else "#111827")
        ax.set_xticks([0, 1], ["ham", "spam"])
        ax.set_yticks([0, 1], ["ham", "spam"])
        ax.set_xlabel("Dự đoán"); ax.set_ylabel("Thực tế")
        name = f"{record['Đặc trưng']} + {record['Thuật toán']}" if records is RESULTS else record["Chiến lược"]
        ax.set_title(f"{record['Mã']} — {name} | F1={record['F1']:.3f}", fontsize=10)
    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(ARTIFACT_DIR / filename, dpi=140, bbox_inches="tight")
    plt.show()

plot_confusions(RESULTS, 2, 2, "Bốn tổ hợp cơ sở", "confusion_baselines.png")
plot_confusions(STRATEGIES, 2, 2, "Các chiến lược trên TF-IDF + SVM", "confusion_strategies.png")
print("Ma trận của cấu hình được chọn:", SELECTED_NAME)
display(pd.DataFrame(confusion_matrix(y_test, best["_pred"], labels=[0, 1]),
                     index=["Thực tế ham", "Thực tế spam"], columns=["Dự đoán ham", "Dự đoán spam"]))
```

Cell 24 dùng `confusion_matrix(y_test,record["_pred"],labels=[0,1])`: thứ tự hàng là nhãn thật ham/spam, cột là nhãn dự đoán ham/spam. `plot_confusions` vẽ/ghi ảnh cho các record đã chấm; hàm vẽ không fit, không đổi score. Từ bốn ô TN/FP/FN/TP có thể tự tính lại Precision, Recall và F1.

```python
ci = []
for record in CANDIDATES:
    lo, hi = boot_ci(y_test, record["_pred"])
    ci.append({"Mã": record["Mã"], "Đặc trưng": record["Đặc trưng"], "Thuật toán": record["Thuật toán"],
               "Chiến lược": record["Chiến lược"], "F1": record["F1"], "CI thấp": lo, "CI cao": hi})
display(pd.DataFrame(ci).round(4))

fig, ax = plt.subplots(figsize=(8, 5))
for record in RESULTS:
    precision, recall, _ = precision_recall_curve(y_test, record["_score"])
    ax.plot(recall, precision, label=f"{record['Mã']} — {record['Đặc trưng']} + {record['Thuật toán']} (AP={record['AP']:.3f})")
ax.axhline(y_test.mean(), ls=":", color="gray", label="Tỷ lệ spam")
ax.set(xlabel="Recall spam", ylabel="Precision spam", title="Precision–Recall: bốn tổ hợp cơ sở")
ax.legend(fontsize=8); ax.grid(alpha=.3); fig.tight_layout()
fig.savefig(ARTIFACT_DIR / "pr_baselines.png", dpi=140, bbox_inches="tight")
plt.show()

assert np.allclose(focus["_score"], tuned["_score"]), "Nhánh ngưỡng phải giữ nguyên decision score của mô hình đối chứng."
fig, ax = plt.subplots(figsize=(8, 5))
for record in [focus, balanced, under]:
    precision, recall, _ = precision_recall_curve(y_test, record["_score"])
    ax.plot(recall, precision, label=f"{record['Chiến lược']} (AP={record['AP']:.3f})")
for record, marker in [(focus, "o"), (tuned, "X")]:
    ax.scatter(record["Recall"], record["Precision"], marker=marker, s=95, zorder=5,
               label=f"Điểm hoạt động: ngưỡng {record['_threshold']:.3f}")
ax.axhline(y_test.mean(), ls=":", color="gray", label="Tỷ lệ spam")
ax.set(xlabel="Recall spam", ylabel="Precision spam", title="Precision–Recall: TF-IDF + SVM")
ax.legend(fontsize=8); ax.grid(alpha=.3); fig.tight_layout()
fig.savefig(ARTIFACT_DIR / "pr_strategies.png", dpi=140, bbox_inches="tight")
plt.show()
```

Cell 26 gọi `boot_ci` cho từng record, rồi vẽ đường Precision–Recall từ `record["_score"]`. B4 và S3 dùng cùng mô hình score, chỉ **khác ngưỡng**, nên đường PR/AP của chúng trùng; hai dấu trên đường cho thấy hai điểm vận hành. S1/S2 học model khác nên có thể có đường PR khác. `fig.savefig` lưu ảnh, `display` hiện bảng/biểu đồ; không huấn luyện lại.

### Cell 30, 32, 34 — lỗi, lưu model và nhận xét

```python
display(Markdown(f"**Cấu hình phân tích:** {SELECTED_NAME}; ngưỡng = {best['_threshold']:.4f}."))
selected_model = best["_model"]
if best["Đặc trưng"] in ("BoW", "TF-IDF"):
    vec, clf = selected_model.named_steps["vec"], selected_model.named_steps["clf"]
    names = vec.get_feature_names_out()
    importance = clf.coef_.ravel() if best["Thuật toán"] == "SVM" else clf.feature_log_prob_[1] - clf.feature_log_prob_[0]
    order = np.argsort(importance)
    display(pd.DataFrame({"Nghiêng về spam": names[order[::-1][:15]], "Nghiêng về ham": names[order[:15]]}))
errors = raw.iloc[test_idx][["id", "text"]].reset_index(drop=True).copy()
errors["Văn bản xử lý"] = TEXT[best["Tiền xử lý"]].iloc[test_idx].to_numpy()
errors["Thực tế"] = np.where(y_test == 1, "spam", "ham")
errors["Dự đoán"] = np.where(best["_pred"] == 1, "spam", "ham")
errors["Điểm mô hình"] = best["_score"]
print("Ham bị nhận nhầm thành spam (FP):")
display(errors[(errors["Thực tế"] == "ham") & (errors["Dự đoán"] == "spam")].sort_values("Điểm mô hình", ascending=False).head(8))
print("Spam bị bỏ sót (FN):")
display(errors[(errors["Thực tế"] == "spam") & (errors["Dự đoán"] == "ham")].sort_values("Điểm mô hình").head(8))
```

Cell 30 lấy `named_steps["vec"]` và `named_steps["clf"]` của model đã fit. `get_feature_names_out` trả tên cột theo đúng thứ tự; với SVM, `coef_.ravel()` là hệ số của từng cột; với NB, chênh `feature_log_prob_` giữa spam và ham là mức nghiêng về từng lớp. `np.argsort` sắp thứ tự để hiện các đặc trưng nghiêng mạnh về hai phía. Đây là **trọng số trong mô hình**, không phải bằng chứng một từ tự nó gây spam. `errors` ghép ID, câu gốc, câu xử lý, nhãn thật, nhãn đoán và score trên đúng test. Hai bộ lọc lấy **FP** (ham báo spam) và **FN** (spam bỏ sót), sắp score rồi hiện tám ví dụ; chỉ để phân tích lỗi, không được dùng để chỉnh lại tham số sau khi đã báo test.

```python
manifest = {"dataset_file": str(CSV_PATH), "processed_file": str(PROCESSED_PATH), "audit": DATA_AUDIT,
    "train_rows": len(train_idx), "test_rows": len(test_idx), "test_spam": int(y_test.sum()),
    "seed": SEED, "environment": ENVIRONMENT, "python": sys.version.split()[0],
    "completed_utc": datetime.now(timezone.utc).isoformat(), "modern_model": "reported_separately_in_notebook_03",
    "selection_rule": "max mean validation F1 on train CV; test not used for selection",
    "baseline_configurations": len(RESULTS), "additional_strategies": len(SUPPLEMENTARY),
    "selected": {k: best[k] for k in ["Mã", "Tiền xử lý", "Đặc trưng", "Thuật toán", "Chiến lược", *CV_COLS, *METRICS, "Tham số"]},
    "decision_threshold": best["_threshold"]}
# Lưu model đã fit, ngưỡng và đúng quy tắc tiền xử lý đã chọn.
bundle = {"model": best["_model"], "threshold": best["_threshold"],
    "remove_stop": best["Tiền xử lý"] == "có loại stop words", "manifest": manifest,
    "preprocessing_source": inspect.getsource(preprocessing_module)}
for record in PREPROCESS_RESULTS + SUPPLEMENTARY:
    record["_model"].memory = None
model_path = ARTIFACT_DIR / "spam_ham_ml_only.joblib"
joblib.dump(bundle, model_path)
loaded = joblib.load(model_path)
X_verify = TEXT[best["Tiền xử lý"]].iloc[test_idx].to_numpy()
reloaded_pred, _ = predict_record({"_model": loaded["model"], "_threshold": loaded["threshold"]}, X_verify)
assert np.array_equal(reloaded_pred, best["_pred"])
raw_predictions = predict_spam(raw.iloc[test_idx].text.tolist(), loaded)
assert np.array_equal(raw_predictions.du_doan.eq("spam").astype(int).to_numpy(), best["_pred"])

display(Markdown(f"**Mô hình lưu và demo:** {SELECTED_NAME}; ngưỡng = {loaded['threshold']:.4f}."))
display(predict_spam(["Cho mình hỏi lịch thi với ạ", "Liên hệ 0901234567 hoặc https://example.com", "Nguyễn Văn An", "."], loaded))
metrics.to_csv(ARTIFACT_DIR / "metrics.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(ci).to_csv(ARTIFACT_DIR / "bootstrap_ci.csv", index=False)
(ARTIFACT_DIR / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
cv_comparison.to_csv(ARTIFACT_DIR / "stopword_cv.csv", encoding="utf-8-sig")
errors.to_csv(ARTIFACT_DIR / "test_predictions.csv", index=False, encoding="utf-8-sig")
cache_dir.cleanup()
print("Đã lưu:", model_path.relative_to(PROJECT_ROOT).as_posix())
print("Dự đoán sau khi nạp lại và từ bình luận gốc khớp trên toàn bộ", len(test_idx), "mẫu test.")
```

`manifest` ghi bối cảnh chạy, số train/test, tiêu chí chọn, tham số, điểm và thời gian. `bundle` mới là thứ cần để dự đoán: `model` đã fit, `threshold`, `remove_stop`; `preprocessing_source` lưu bản mã nguồn tiền xử lý để truy xuất cách chuẩn bị văn bản tại thời điểm chạy. `record["_model"].memory=None` loại tham chiếu cache tạm. `joblib.dump` ghi file và `joblib.load` nạp lại. Hai assert kiểm dự đoán toàn test không đổi sau nạp và cùng nhãn khi đi từ bình luận gốc qua `predict_spam`. Sau đó cell demo vài bình luận mới và ghi `metrics.csv`, `bootstrap_ci.csv`, `run_manifest.json`, `stopword_cv.csv`, `test_predictions.csv`. `cache_dir.cleanup()` dọn cache tạm; không xóa model cuối.

```python
tn, fp, fn, tp = confusion_matrix(y_test, best["_pred"], labels=[0, 1]).ravel()
summary = f"""
- **Thiết kế:** so sánh bốn tổ hợp cơ sở với cấu hình {MAIN_PRE}; khảo sát riêng
  ba chiến lược trên TF-IDF + SVM. Mô hình được chọn theo CV F1 spam, không chọn lại bằng test.
- **Cấu hình truyền thống lựa chọn:** {SELECTED_NAME}; ngưỡng {best['_threshold']:.4f}, CV F1 {best['CV F1 (spam)']:.4f}.
- **Kết quả test:** {len(test_idx):,} bình luận, gồm {int(y_test.sum()):,} spam;
  Accuracy {best['Accuracy']:.2%}, Precision {best['Precision']:.2%}, Recall {best['Recall']:.2%},
  F1 {best['F1']:.4f}, AP {best['AP']:.4f}. Có {int(fp):,} ham bị nhận nhầm và {int(fn):,} spam bị bỏ sót.
- **Ảnh hưởng chiến lược:** so với không điều chỉnh (Recall {focus['Recall']:.2%}, F1 {focus['F1']:.4f}),
  balanced đạt Recall {balanced['Recall']:.2%}, F1 {balanced['F1']:.4f}; under-sampling đạt
  Recall {under['Recall']:.2%}, F1 {under['F1']:.4f}; chỉnh ngưỡng đạt Recall {tuned['Recall']:.2%}, F1 {tuned['F1']:.4f}.
  Các chỉ số thể hiện đánh đổi giữa phát hiện spam và cảnh báo nhầm, không chứng minh một chiến lược luôn tốt hơn.
- **viBERT:** mô hình hiện đại được đối chiếu tại mục 6.4; chi tiết huấn luyện và demo ở Notebook 03.
- **Phạm vi và giới hạn:** chưa thử mọi chiến lược trên cả bốn tổ hợp. Kết quả phụ thuộc
  nhãn hiện có; bình luận ngắn, gần trùng hoặc cần ngữ cảnh bài đăng vẫn có thể gây lỗi.
  Bundle lưu lại được dùng nhất quán cho dự đoán; demo không thay thế đánh giá trên test.
"""
display(Markdown(summary))
```

Cell 34 lấy TN/FP/FN/TP của mô hình cuối, lắp các số đã đo vào nhận xét. Đây là **tóm tắt kết quả đã có**, không chạy CV/test thêm lần nữa. Câu “S3 tốt nhất” phải đọc là tốt nhất theo **CV F1 trong bảy ứng viên đã thử** trên dữ liệu hiện có; test F1 của S3 chỉ nhỉnh hơn S2 rất ít, không chứng minh luôn tốt hơn ở nguồn bình luận khác.

## 10. Bảng tra nhanh những cú pháp thường gây khó hiểu

| Cú pháp / hàm | Hiểu bằng một hành động cụ thể trong bài |
|---|---|
| `import module as tên`, `from module import tên` | Nạp mã có sẵn; lệnh import **không huấn luyện**. `np`/`pd` chỉ là tên ngắn. |
| `def`, `return` | Định nghĩa hàm và trả kết quả; thân hàm chỉ chạy khi có lệnh gọi. |
| `class`, `self`, `self.vocabulary_` | Tạo kiểu đối tượng; `self` là chính đối tượng đang fit/transform; thuộc tính dấu `_` cuối là trạng thái học sau fit. |
| `list(...), np.array(...), np.asarray(...)` | Đổi/bao dữ liệu thành danh sách hoặc mảng; `asarray` không tạo bản sao nếu không cần. Mảng cho phép cắt bằng nhiều chỉ số. |
| `enumerate(items)` | Duyệt đồng thời vị trí `i` và giá trị; dùng để tạo các chỉ số dòng. |
| `[... for ... if ...]` | Comprehension: vòng lặp tạo danh sách; điều kiện `if` lọc phần tử. |
| `set(items)`, `A & B`, `A - B` | Lấy phần tử duy nhất; giao hai tập; phần ở A không nằm trong B. Dùng để chống khóa trùng hoặc kiểm cột thiếu. |
| `Counter(items)` | Từ điển đếm; trong `fit` đếm **số câu** nhờ bọc `set`, trong `transform` đếm **số lần xuất hiện trong câu**. |
| `arr[indices]`, `arr[mask]`, `[:,1]` | Lấy nhiều vị trí; lấy ô thỏa mặt nạ Boolean; lấy cột thứ hai của ma trận xác suất. |
| `.iloc[indices]`, `.to_numpy()` | Pandas lấy dòng theo **vị trí số**, rồi đổi Series thành mảng NumPy. |
| `.get(key,default)` | Lấy khóa trong dict; nếu thiếu dùng giá trị mặc định, ví dụ ngưỡng 0,5/0. |
| `hasattr(obj,"predict_proba")` | Hỏi đối tượng có phương thức đó không; dùng để tách nhánh NB/SVM. |
| `clone(pipe)` | Tạo bản cấu hình mới **chưa fit**; tránh mang từ vựng/trọng số của fold trước sang fold sau. |
| `set_params(**params)` | Trải dict tham số thành các đối số có tên và gắn vào Pipeline trước fit. |
| `max(rows,key=lambda r:...)` | Chọn phần tử có điểm lớn nhất theo phép tính trong `lambda`. Không tự huấn luyện. |
| `np.zeros(n)`, `np.arange(n)`, `np.unique(x)` | Tạo mảng 0; tạo vị trí 0…n−1; bỏ phần tử trùng. |
| `np.quantile`, `np.percentile` | Lấy phân vị score để thử ngưỡng; lấy mốc 2,5%/97,5% của các F1 bootstrap. |
| `pd.DataFrame(...).to_csv(...)` | Ghép bảng và ghi CSV; file CSV chỉ chứa số/chữ, không chứa đối tượng model đã fit. |
| `joblib.dump/load` | Lưu/nạp đối tượng Python; bundle chứa bộ biến văn bản, bộ phân loại, ngưỡng và quy tắc từ dừng. |

## 11. Những câu dễ bị hỏi khi trình bày

1. **Phi nhận gì từ thành viên trước?** CSV có nhãn và văn bản đã chuẩn bị; Phi tự chia train/test/fold. Notebook 02 có cell tạo lại CSV này, còn script chạy riêng đọc file đã có.
2. **`norm_key` có phải tiền xử lý hai lần?** Nó chỉ tạo khóa chia tập. Vector hóa dùng cột token riêng, không dùng khóa đó.
3. **Tại sao không chia ngẫu nhiên từng dòng?** Câu trùng/gần trùng có thể lọt cả train và test làm điểm lạc quan. Bucket theo khóa giữ chúng cùng phía; đồng thời hai nhánh truyền thống/viBERT dùng cùng test. Đây không phải chia bảo đảm cân bằng nhãn.
4. **`make_classifier` đã là mô hình chưa?** Nó là **đối tượng cấu hình chưa học**. `search.fit` mới học, `best_estimator_` mới là Pipeline đã học trên toàn train.
5. **BoW/TF-IDF dùng để làm gì nếu NB/SVM đã là thuật toán?** NB/SVM nhận số, còn đầu vào là văn bản. Bộ vector hóa biến câu thành ma trận số theo từ vựng học ở train.
6. **Vì sao `GridSearchCV` có khóa `mean_test_f1` dù ta chưa dùng test?** API gọi phần giữ ra trong mỗi fold là “test fold”; trong thiết kế bài, đó là **validation nằm trong train**. Test riêng chỉ được chấm ở `evaluate`.
7. **Tại sao S2 không gọi `GridSearchCV` trực tiếp?** Để giảm mẫu đúng **bên trong train của từng fold**, giữ validation nguyên phân bố; code tự duyệt tham số bằng `ParameterGrid` rồi fit.
8. **S3 có xác suất spam âm không?** Không. `LinearSVC.decision_function` trả score có dấu, không phải xác suất. Ngưỡng âm chỉ khiến nhiều câu có score âm nhẹ được nhận là spam.
9. **S3 được chọn vì test F1 cao nhất?** Không. Chọn theo CV F1 trước khi chấm test. Test F1 chỉ báo kết quả trên phần giữ riêng; S3 hơn S2 rất ít trên test.
10. **`.joblib` khác `.py` thế nào?** `.py` là logic chạy; `.joblib` giữ trạng thái đã học: từ vựng/IDF, hệ số phân loại, ngưỡng/chính sách từ dừng. Muốn dự đoán vẫn cần môi trường và các module Python tương thích.

## 12. Kiểm chứng và nguồn tra cứu

Lượt chạy đã kiểm trong project: 30.759 dòng train, 7.474 dòng test, năm fold; B1–B4 và S1–S3 được chấm trên cùng test. S3 được chọn theo CV F1 spam khoảng **0,7876**; trên test có Precision khoảng **0,8374**, Recall **0,6877**, F1 **0,7552**. `spam_ham_ml_only.joblib` đã nạp lại và kiểm nhãn trên toàn test. Những số này là **kết quả của lượt chạy hiện có**, có thể thay đổi nếu CSV, code hoặc cấu hình đổi.

Lý thuyết thư viện trong tài liệu được đối chiếu với bản scikit-learn **1.6.1** đang cài trong `.venv`:

- [MultinomialNB](https://scikit-learn.org/1.6/modules/generated/sklearn.naive_bayes.MultinomialNB.html): đầu vào không âm, `alpha`, `fit`, `predict_proba`; tài liệu nêu TF-IDF có thể dùng trong thực hành.
- [CountVectorizer](https://scikit-learn.org/1.6/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html) và [TfidfVectorizer](https://scikit-learn.org/1.6/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html): hai lớp thư viện được dùng trực tiếp trong `text_features.py`.
- [LinearSVC](https://scikit-learn.org/1.6/modules/generated/sklearn.svm.LinearSVC.html): `C`, `dual`, `max_iter`, `class_weight`, `decision_function`; `random_state` không tác động khi `dual=False`.
- [Pipeline](https://scikit-learn.org/1.6/modules/generated/sklearn.pipeline.Pipeline.html): thứ tự `fit/transform`, `memory` và tên tham số `bước__tham_số`.
- [GridSearchCV](https://scikit-learn.org/1.6/modules/generated/sklearn.model_selection.GridSearchCV.html): CV, `cv_results_`, `best_params_` và `refit`.
- [ParameterGrid](https://scikit-learn.org/1.6/modules/generated/sklearn.model_selection.ParameterGrid.html): liệt kê tổ hợp tham số, không tự fit.

**Cách học:** mở file nguồn cạnh tài liệu này, đi từng mục; với mỗi hàm hãy tự nói “nhận gì → gọi gì → học hay chỉ biến đổi → trả gì → ai dùng tiếp”. Sau đó chạy `scripts/run_phi_pipeline.py` hoặc Notebook 02 để nối lý thuyết với bảng kết quả thực.
