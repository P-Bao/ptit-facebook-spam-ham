# Phi 02 — Luồng fit, các tổ hợp và cách tính điểm

Đọc sau [năm lớp thư viện](PHI_01_NAM_LOP_THU_VIEN_MODELING.md). Code trích dưới đây là code project hiện tại, không phải một bản tự viết thay thư viện. Phạm vi bắt đầu khi đã nhận CSV có nhãn và hai cột token, không học lại cách crawl/gán nhãn/tách từ.

## 1. Đầu vào trước khi fit

`training_utils.split_train_test` nhận khóa văn bản và nhãn, trả vị trí train/test, giữ các câu cùng khóa không bị chia sang hai phía. `grouped_folds` tạo năm cặp train/validation **bên trong tập train**. Các vị trí fold không phải ID gốc và không phải vị trí test.

`TEXT` là dict có hai Series: giữ từ dừng và bỏ từ dừng. `y_train` là mảng 0/1. Lấy `.iloc[train_idx]` chọn theo vị trí; `.to_numpy()` đổi Series thành mảng để hàm CV truy cập bằng chỉ số số nguyên. Kết quả vẫn là chuỗi token, chưa phải vector.

Trong các lượt chạy hiện tại: CSV 38.233 dòng, train 30.759, test 7.474; không nên suy ra đúng 80/20 từng lớp vì quy tắc chia giữ nhóm trùng. Test có 951 spam. Các con số này phụ thuộc dataset đang dùng.

## 2. Hàm fit_experiment nhận gì và trả gì?

| Tham số | Ý nghĩa |
|---|---|
| `pre` | Tên nhánh tiền xử lý dùng ghi kết quả; không tự tiền xử lý X |
| `feat` | BoW hoặc TF-IDF, dùng chọn vectorizer |
| `model_name` | Naive Bayes hoặc SVM, dùng chọn classifier |
| `X` | Chuỗi token chỉ thuộc train, đã chọn đúng nhánh trước khi gọi |
| `y` | Nhãn thật tương ứng X |
| `folds` | Năm cặp chỉ số train/validation trong X/y |
| `seed` | Cấu hình tái lập; không thay cho dữ liệu |
| `cache` | Bộ nhớ đệm kết quả vector hóa, giảm tính lại; không phải dữ liệu kiểm thử |
| `n_jobs` | Số công việc song song; không phải số folds |
| `class_weight` | None cho cơ sở, balanced cho S1 |
| `strategy` | Tên chiến lược ghi vào record |

Một lời gọi hàm này chạy **một cặp đặc trưng–thuật toán trên một nhánh token**, nhưng có nhiều bộ siêu tham số bên trong. Nó không chạy cả bốn cặp và không chấm test. Đầu ra là dict record: tên cấu hình, CV P/R/F1, best_params, thời gian, `_threshold`, `_model` đã fit.

## 3. Hai nhà máy cấu hình: chưa học dữ liệu

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

`make_vectorizer` trả tuple đối tượng+grid. `**common` trải dict thành các tham số có tên; không phải phép nhân. `(1,1)` là giá trị khởi tạo. Khi CV nhận `vec__ngram_range=(1,2)`, nó thực sự ghi đè giá trị khởi tạo trước khi fit bản clone. `return vectorizer, {...}` không tạo ma trận. Tương tự `make_classifier` tạo NB/SVM chưa có trọng số và grid alpha/C; dòng constructor chưa áp thuật toán lên X.

## 4. Toàn bộ experiments.py và cách đọc

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

Đọc theo thứ tự thực thi:

1. `CV_SCORING` yêu cầu ba chỉ số; mọi chỉ số có lớp dương mặc định 1=spam.
2. `make_pipe` kiểm tên; lấy hai đối tượng từ hai nhà máy; đặt tên bước `vec`, `clf`; `Pipeline` giữ thứ tự.
3. `{**vector_grid, **model_grid}` ghép dict grid. `vec__...` đi vào vectorizer, `clf__...` đi vào classifier. `n_jobs` chỉ chuyển tiếp số tiến trình.
4. `fit_experiment` gọi make_pipe rồi ghi thời điểm bắt đầu bằng `time.time()`.
5. Tạo GridSearchCV với Pipeline và các folds đã có. Dòng này chưa học.
6. `search.fit(X,y)` thực hiện mọi candidate×fold: clone → set_params → lấy X_train_fold → vectorizer.fit_transform → classifier.fit → vectorizer.transform(validation) → dự đoán validation → chấm P/R/F1. Không fit từ vựng/IDF trên validation.
7. Sau năm folds/candidate, lấy trung bình F1 và chọn best_index_. Do refit='f1', cấu hình thắng được fit lại trên toàn X/y train ngay trong search.fit.
8. Lấy mean_test_precision/recall/f1 cùng dòng thắng; không lấy cực đại từng metric từ các dòng khác nhau.
9. Tạo record: `float` đổi số NumPy thành số Python; `_model` giữ Pipeline; `_threshold` giữ 0,5 cho NB hoặc 0 cho SVM. Những tên có `_` ở đây là quy ước project, không phải cột dữ liệu đầu vào.

`evaluate`/`boot_ci` import trong experiments là tên chuyển tiếp để code khác có thể gọi; không được chạy chỉ vì import. Hàm evaluate thực định nghĩa trong evaluation.py.

## 5. Vòng lặp notebook thực sự chạy những tổ hợp nào?

Cell thứ 14 tính cả Markdown của Notebook 02:

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

`for pre in TEXT` chọn hai nhánh. `for feature in FEATURES` chọn hai biểu diễn. `for algorithm in MODELS` chọn hai thuật toán. Như vậy có tám lời gọi fit_experiment, mỗi lời gọi trả một record thắng grid riêng.

| Mỗi nhánh token | Siêu tham số được thử | Candidate | CV fit |
|---|---|---:|---:|
| BoW + NB | 2 n-gram × 3 alpha | 6 | 30 |
| BoW + SVM | 2 n-gram × 4 C × class_weight=None | 8 | 40 |
| TF-IDF + NB | 2 n-gram × 3 alpha | 6 | 30 |
| TF-IDF + SVM | 2 n-gram × 4 C × class_weight=None | 8 | 40 |
| Tổng hai nhánh từ dừng | 2 × (6+8+6+8) | **56** | **280** |

Ngoài 280 fit CV còn tám lượt refit của tám search. Cache có thể tái sử dụng phép biến đổi phù hợp để tiết kiệm thời gian; không bỏ lượt chấm nào.

`pivot` tạo bảng so giữ/bỏ từ dừng. `MAIN_PRE = max(...)[...]` lấy nhánh chứa record CV F1 cao nhất trong tám record; đây không phải lấy trung bình của cả bốn cặp để chọn nhánh. `RESULTS` chỉ giữ bốn record của MAIN_PRE. `enumerate(...,1)` gắn B1=BoW+NB, B2=BoW+SVM, B3=TF-IDF+NB, B4=TF-IDF+SVM theo thứ tự vòng lặp. Mỗi B là một pipeline có n-gram và alpha/C đã được CV chọn, không phải constructor mặc định.

`focus = max(...)` lấy ứng viên tốt nhất trong bốn B. Assert yêu cầu focus là TF-IDF+SVM vì phần khảo sát cân bằng đang thiết kế cho ứng viên đó; nếu dữ liệu mới làm ứng viên dẫn đầu đổi, cần xem lại phạm vi, không xóa assert rồi giữ nguyên lời kết luận cũ.

## 6. Tính điểm, chọn và test khác nhau ở đâu?

Với spam là lớp dương: TP=spam đoán spam; FP=ham đoán spam; FN=spam đoán ham; TN=ham đoán ham. Precision=TP/(TP+FP); Recall=TP/(TP+FN); F1=2PR/(P+R). `zero_division=0` trả 0 khi mẫu số không hợp lệ.

CV tính F1 riêng từng validation rồi lấy trung bình năm số F1. **Không** lấy P trung bình và R trung bình thế vào công thức F1, cũng không nhất thiết bằng F1 tính trên toàn bộ nhãn OOF gộp. Scoring precision/recall dùng nhãn dự đoán ở ngưỡng mặc định của classifier trong phần B/S1/S2.

Sau S1/S2/S3, `CANDIDATES=RESULTS+SUPPLEMENTARY` có bảy record. `best=max(...CV F1...)` chốt ứng viên trước khi evaluate test. Test đi qua vectorizer đã học, model đã học, ngưỡng đã chốt; không fit lại trên test. `evaluate` thêm Accuracy/P/R/F1/AP và `_pred`/`_score` vào record. AP dùng score liên tục để đánh giá thứ hạng trên nhiều ngưỡng; khác F1 tại một ngưỡng.

Ví dụ theo đầu ra local: B4 CV F1≈0,7736 và test F1≈0,7334; S3 CV F1≈0,7876 và test F1≈0,7552. Không so chéo CV của một model với test của model khác để xếp hạng. Điểm CV được dùng nhiều lần để chọn cấu hình nên không phải ước lượng hoàn toàn độc lập; test mới là đánh giá giữ riêng.

## 7. Xem toàn bộ điểm thật và lần ngược lời gọi

Notebook `04_kiem_tra_grid_search.ipynb` chạy lại lưới và ghi `artifacts/grid_audit/all_cv_configs.csv`; `best_by_group.csv` chứa tám cấu hình thắng theo từng nhánh/cặp. Không sửa hàm fit_experiment để in từng cấu hình khi chỉ cần kiểm bảng này.

Luồng cần nhớ: cell vòng lặp → fit_experiment → make_pipe → make_vectorizer/make_classifier → GridSearchCV.fit → Pipeline.fit → vectorizer.fit_transform → classifier.fit → validation.predict → scorer → chọn mean F1 → refit → record → S1/S2/S3 → chọn best → evaluation.evaluate(test) → lưu bundle joblib → modeling.predict.predict_spam cho câu mới.
