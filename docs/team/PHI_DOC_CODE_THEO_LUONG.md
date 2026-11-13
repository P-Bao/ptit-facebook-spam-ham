# Phi đọc toàn bộ code theo giai đoạn

Tài liệu chép **code hiện tại** của phần Phi từ CSV đã có nhãn/token đến `.joblib`: chia tập, đặc trưng, phân loại, chọn bằng CV, khảo sát lệch lớp, chấm test và dự đoán. Cương bàn giao CSV; **Phi tự chia train/test và fold**. Xem [bản đồ luồng](PHI_LUONG_CODE_SAU_TACH_MODULE.md) trước khi đọc từng dòng. Mã trong tài liệu chỉ để học; hãy chạy file/notebook gốc.

Đầu vào tại ranh giới của Phi là `raw`, `TEXT`, `y_all` đã được chuẩn bị từ v7; `data/processed/v7_preprocessed.csv` là CSV được cell 9 tạo và đọc lại. Không cần học crawler/Argilla/viBERT để hiểu các file bên dưới.

## 0. Nhận CSV và tự chia tập

Cell 11 gọi `split_train_test` để tự tạo `train_idx`/`test_idx`, rồi `grouped_folds` để có năm cặp train–validation. `norm_key` và `bucket` là hai bước nhỏ ở bên trong để câu trùng cùng phía, không phải tiền xử lý câu cho model. `train_idx/test_idx` trỏ vào toàn CSV; từng cặp trong `CV_FOLDS` trỏ vào mảng train. Test chỉ dùng sau khi đã chốt cấu hình bằng CV.

**Đọc theo bốn bước:** `norm_key` hạ chữ và gom khoảng trắng/số; `bucket` gán một khóa vào nhóm ổn định; `split_train_test` cho một nhóm làm test và bốn nhóm làm train; `grouped_folds` lần lượt dùng một trong năm nhóm train làm validation. `salt` chỉ tách mục đích chia test và chia CV, không phải mật khẩu hay mã hóa dữ liệu. Có thể chia ngẫu nhiên thẳng, nhưng câu trùng có nguy cơ lọt qua hai phía và viBERT sẽ không còn cùng test với mô hình này.

| Biến | Chứa gì | Hệ chỉ số |
|---|---|---|
| `keys` | Khóa nội dung đã chuẩn hóa của từng dòng CSV | Toàn CSV |
| `train_idx`, `test_idx` | Vị trí dòng được gán vào train/test | Toàn CSV |
| `y_train`, `y_test` | Nhãn số: ham = 0, spam = 1 | Từng tập |
| `CV_FOLDS` | Năm cặp `(train_fold, validation_fold)` | Chỉ trong mảng train |

### Toàn bộ `pipeline/modeling/training_utils.py`

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

### Notebook 02 cell 11 — Phi chia tập và tạo fold

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

## 1. Tạo BoW/TF-IDF và NB/SVM, chọn bằng CV

**Thứ tự dữ liệu trong một lần fit:** chuỗi token → `CountVectorizer` hoặc `TfidfVectorizer` học từ vựng/IDF trên fold train → `MultinomialNB` hoặc `LinearSVC` học từ vector và nhãn → chấm fold validation. `Pipeline.fit` thực hiện hai bước; `GridSearchCV` thử n-gram và alpha/C, chọn trung bình F1 spam trên năm validation rồi fit lại trên toàn train. Test chỉ được chấm sau khi chọn cấu hình.

`make_vectorizer` chỉ tạo một đối tượng thư viện **chưa fit**. `CountVectorizer` đếm BoW; `TfidfVectorizer` kế thừa bước đếm rồi tính TF-IDF. Cả hai nhớ từ vựng sau `fit`; TF-IDF còn nhớ `idf_`. `transform` của validation/test/bình luận mới dùng trạng thái đó mà không học thêm. Xem [giải thích mã nguồn bốn lớp scikit-learn](PHI_HIEU_4_LOP_SCIKIT_LEARN.md).

### Toàn bộ `pipeline/modeling/text_features.py`

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

`make_classifier(model_name)` tạo `MultinomialNB` hoặc `LinearSVC` **chưa fit**. NB thử `alpha`; SVM thử `C` và `class_weight` (mặc định None, S1 mới là balanced). Khi `Pipeline.fit()` chạy, classifier mới học từ vector/nhãn train. SVM `decision_function()` trả score có dấu, không phải xác suất.

### Toàn bộ `pipeline/modeling/classifiers.py`

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

`make_pipe` gọi hai hàm trên, ghép `Pipeline(vec,clf)` và lưới tham số. `fit_experiment` gọi `GridSearchCV(...,cv=folds,refit="f1")`: với từng tổ hợp tham số và fold, fit vectorizer/classifier trên fold train, chấm trên validation, chọn theo F1 spam rồi fit lại trên toàn train. Record trả về có điểm CV, tham số, `_model` đã fit và `_threshold` mặc định. Không truyền test vào đây. `evaluate`/`boot_ci` được import chuyển tiếp để code cũ còn chạy; định nghĩa thực nằm trong `evaluation.py`.

### Toàn bộ `pipeline/modeling/experiments.py`

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

Cell 13 chạy hai nhánh từ dừng × hai đặc trưng × hai bộ phân loại = tám lượt CV. `MAIN_PRE` giữ nhánh của ứng viên CV F1 cao nhất; bốn record nhánh đó được mã B1–B4. `focus=B4` là TF-IDF+SVM dẫn đầu nhóm cơ sở theo CV. Các bảng hiển thị điểm **validation trong train**, chưa phải test.

### Notebook 02 cell 13 — fit CV và chọn B4

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

## 2. Khảo sát lệch lớp trên B4

**Bốn trường hợp để so sánh:** B4 giữ nguyên dữ liệu và ngưỡng SVM bằng 0; S1 giữ nguyên dữ liệu nhưng đặt `class_weight="balanced"`; S2 lấy toàn bộ spam và tối đa hai ham cho một spam trong *train của từng fold*; S3 giữ mô hình B4 và chọn một ngưỡng decision score bằng dự đoán ngoài fold trên train. Ba cách này là ba thí nghiệm riêng, không cộng dồn. `clone` trong S2/S3 tạo model chưa học cho fold mới; nhờ vậy không có trọng số học từ fold trước lọt sang validation hiện tại. Ngưỡng S3 có thể âm vì decision score của SVM không phải xác suất; ngưỡng âm làm một số điểm ở phía âm gần 0 chuyển sang spam. Điểm CV của S3 dùng chính dự đoán OOF để **chọn và báo điểm ngưỡng**, nên có thể lạc quan đôi chút; điểm test giữ riêng mới là phép kiểm sau lựa chọn.

`imbalance.py` trình bày theo S1–S3: `fit_balanced_svm` gọi lại thí nghiệm TF-IDF + SVM với trọng số balanced; `sample_training_indices` và `undersampled_search` của S2 chỉ giảm ham trong fold train, dùng `ParameterGrid` để liệt kê tám cặp n-gram/C; `tune_svm_threshold` của S3 tạo score OOF, thử ngưỡng và fit lại toàn train. `mean_cv_scores` chỉ đo trên các fold validation, không đo test. Score/threshold âm của SVM là bình thường; đây không phải xác suất.

### Toàn bộ `pipeline/modeling/imbalance.py`

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

Cell 15 chỉ tạo bảng/hình để thấy tỷ lệ nhãn và trọng số; chưa fit các biến thể. `weights=len(y_train)/(2*before_counts)` là công thức balanced cho hai lớp. Hình lưu ở `imbalance_training.png`.

### Notebook 02 cell 15 — phân bố và trọng số

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

Cell 16 fit S1, S2, S3 riêng, tạo `CANDIDATES=[B1..B4,S1,S2,S3]`, rồi `best=max(...,CV F1)` **trước khi chấm test**. `strategy_cv`/`imbalance_cv.csv` là bằng chứng lựa chọn; `SELECTED_NAME` chỉ là tên để hiển thị. Không lấy mẫu validation/test; không cộng ba chiến lược vào một mô hình.

### Notebook 02 cell 16 — S1/S2/S3 và chốt theo CV

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

## 3. Chấm test sau lựa chọn

`predict_record` biến score thành nhãn theo ngưỡng của từng cấu hình. Với NB, score là xác suất spam và ngưỡng mặc định 0,5; với SVM, score là khoảng cách có dấu đến ranh giới và ngưỡng mặc định 0. `evaluate` so nhãn dự đoán với `y_test` để tính Accuracy, Precision, Recall, F1 lớp spam; AP dùng score liên tục, nên đổi riêng ngưỡng không nhất thiết đổi AP. Tất cả bảy ứng viên được báo cáo trên cùng test, nhưng **ứng viên được chọn trước theo CV**, không chọn lại bằng bảng test.

`predict_record` xem model có `predict_proba` (NB) hay `decision_function` (SVM), tạo score và áp ngưỡng từng record. `evaluate` chỉ dự đoán/tính Accuracy, Precision, Recall, F1, AP trên test rồi thêm `_pred`/`_score` vào record; không fit hay chọn lại. `boot_ci` lấy bootstrap 500 lần trên test để mô tả khoảng F1. Bảo phụ trách diễn giải sâu biểu đồ/lỗi; Phi cần đọc được các chỉ số và phạm vi chọn S3.

### Toàn bộ `pipeline/modeling/evaluation.py`

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

Cell 20 dùng cùng test cho bảy cấu hình. Dòng “luôn dự đoán ham” cho thấy Accuracy cao chưa đủ khi spam ít. Test Precision/Recall/F1 của `best` được in ra **sau** khi đã chọn `best` theo CV ở cell 16.

### Notebook 02 cell 20 — đánh giá bảy ứng viên

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

## 4. Lưu mô hình và dự đoán bình luận mới

`joblib` ghi đối tượng `Pipeline` đã fit, ngưỡng và cờ giữ/bỏ từ dừng. Vì vậy file `.py` là **cách huấn luyện/dự đoán**, còn file `.joblib` mới chứa từ vựng, IDF và trọng số đã học. `predict_spam` **nhận bundle đã nạp**, tiền xử lý comment mới bằng đúng quy tắc đã chọn, gọi `decision_function` hoặc `predict_proba`, rồi áp ngưỡng để trả nhãn. Cell 32 nạp lại file vừa lưu và kiểm dự đoán test không đổi; đó là phép kiểm tính dùng được của file xuất.

`predict_spam` nhận text thô, gọi lại `preprocess` với `remove_stop` trong bundle, đưa chuỗi token vào pipeline đã fit, lấy score qua `predict_record`, so với ngưỡng và trả bảng. Hàm không huấn luyện lại và không gọi LLM. `joblib.dump/load` nằm trực tiếp ở cell 32 để dễ theo dõi việc ghi/nạp `.joblib`.

### Toàn bộ `pipeline/modeling/predict.py`

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

Cell 32 tạo `bundle` từ `best["_model"]`, ngưỡng S3 và chính sách từ dừng; `joblib.dump` lưu mô hình, `joblib.load` nạp lại. Hai `assert` kiểm nhãn test trước/sau lưu và trên văn bản gốc. Sau đó cell demo bốn câu, ghi metrics/manifest và dọn cache. `.py` là cách làm; `.joblib` chứa pipeline đã fit và cấu hình suy luận.

### Notebook 02 cell 32 — lưu/nạp, kiểm và demo

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

Cell 34 chỉ tổng hợp TN/FP/FN/TP và nhận xét trên điểm CV/test đã đo; không fit model. Khi trình bày không dùng điểm test làm lý do chọn S3.

### Notebook 02 cell 34 — nhận xét cuối

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

## 5. Script chạy thẳng toàn bộ phần Phi

File này nối các module trên từ CSV đến `.joblib`. Hãy đọc từ trên xuống theo giai đoạn 0–4. Chạy từ gốc project bằng `.\.venv\Scripts\python.exe scripts/run_phi_pipeline.py`. Kết quả ghi riêng vào `artifacts/spamham/phi_pipeline/`, không ghi đè Notebook 02.

### Toàn bộ `scripts/run_phi_pipeline.py`

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

## Cách tự học và kiểm tra

Với mỗi hàm, trả lời bốn câu: **đầu vào gì, xử lý trên tập nào, đầu ra gì, hàm tiếp theo dùng nó ra sao**. Trong VS Code chọn `.venv`, Ctrl+Click hoặc F12 vào hàm nhóm viết để mở đúng file. `inspect.getsource(fit_experiment)` đọc mã Python của hàm này; `help(GridSearchCV)` đọc tài liệu thư viện. Có thể dùng [luồng tóm tắt](PHI_LUONG_CODE_SAU_TACH_MODULE.md) để nói lại toàn bộ mà không cần đọc từng dòng code trên slide.

| Lệnh/hàm phổ biến | Phi cần hiểu thao tác thực tế |
|---|---|
| `pd.read_csv`, `.iloc`, `.to_numpy()` | Đọc bảng; lấy dòng theo **vị trí**; chuyển cột thành mảng cho scikit-learn. |
| `set`, `Counter` | `set` giữ các giá trị khác nhau, dùng để kiểm có đủ hai nhãn hoặc đếm mỗi đặc trưng một lần trong một câu; `Counter` cộng số lần xuất hiện của từ/cặp từ. |
| `np.array`, `np.asarray` | Đưa dữ liệu thành mảng NumPy để lấy nhiều dòng bằng danh sách chỉ số; `asarray` giữ nguyên mảng nếu đầu vào đã là mảng phù hợp. Chúng không học mô hình. |
| `np.where`, `np.unique`, `np.array_equal` | Lấy vị trí thỏa điều kiện; kiểm các nhãn có mặt; đối chiếu dự đoán trước/sau khi lưu. |
| `bucket(key, "test"/"cv")` | Cho cùng nội dung một số nhóm cố định; `"test"` và `"cv"` chỉ là tên hai lần chia. Hàm băm ở đây không mã hóa hay ẩn danh bình luận. |
| `CountVectorizer` / `TfidfVectorizer` | Học từ vựng/IDF trên fold train; `transform` biến validation/test bằng trạng thái đã học. |
| `csr_matrix` | Chỉ lưu các ô khác 0 của vector; tránh ma trận đặc quá lớn. Không phải thuật toán phân loại. |
| `Pipeline.fit` | Gọi fit/transform của `vec` rồi fit của `clf`; `Pipeline.predict` hoặc `decision_function` áp dụng cả hai bước cho dữ liệu mới. |
| `GridSearchCV.fit` | Thử từng bộ tham số trên năm cặp fold; lấy trung bình điểm validation, chọn theo F1 rồi fit lại trên toàn train. |
| `clone`, `ParameterGrid` | Tạo bản model mới chưa fit; liệt kê các tổ hợp siêu tham số. |
| `joblib.dump`, `joblib.load` | Ghi và nạp lại đối tượng Python đã fit, gồm từ vựng/IDF/trọng số và ngưỡng đi kèm. |

Khi Ctrl+Click vào lớp thư viện, hãy tìm `fit`, `transform` và `predict`/`decision_function`; chỉ cần giải thích được **đầu vào, trạng thái học và đầu ra** của các hàm này. Không cần chép lại toàn bộ mã nội bộ của scikit-learn.

