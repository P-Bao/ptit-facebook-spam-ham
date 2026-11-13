# BƯỚC 07 — `predict.py`: Lưu, nạp và sử dụng mô hình Spam/Ham PTIT

> **Mục tiêu học:** Hiểu toàn bộ quy trình từ một Pipeline đã huấn luyện đến file `.joblib`, cách nạp lại, cách `predict_spam()` xử lý bình luận thô, cách tính score/áp ngưỡng và đọc kết quả. Tài liệu giải thích **code dự án và các phương thức thư viện**; kèm ví dụ từng bước, kết quả thật từ Notebook, hạn chế, câu hỏi bảo vệ.
>
> **Nguồn code phiên bản mới:** `pipeline/modeling/predict.py` (`predict(1).py`), `pipeline/modeling/evaluation.py` (`evaluation(1).py`), `pipeline/prepare/preprocess_for_model.py`, Notebook `02_thuc_nghiem_va_demo(1).ipynb` (đặc biệt cell 4, 9, 13, 16, 20, 31–32), `experiments(1).py`, `text_features(1).py`, `PHI_HIEU_4_LOP_SCIKIT_LEARN.md`.
>
> **Phân định:** Tiền xử lý do thành viên khác phụ trách: ta chỉ nắm đủ để đảm bảo khi suy luận dùng đúng quy tắc khi train. Kết quả số trong mục 15 là **output đã lưu của Notebook**, không phải kết quả chạy lại `.joblib` vì model và dataset thật không được đính kèm. Ví dụ tính tay/code minh họa ở các mục khác được ghi là *minh họa*.

---

## Mục lục

1. Mục tiêu và ranh giới giữa Notebook/`predict.py`
2. Bản đồ toàn luồng Train → CV → Test → lưu → suy luận
3. Các khái niệm: inference, persistence, fitted pipeline, artifact, threshold, bundle
4. Đầu vào/đầu ra chính xác của `predict_spam()`
5. Đọc toàn bộ mã nguồn `predict.py`
6. Giải thích từng câu lệnh trong `predict_spam()`
7. Tiền xử lý của câu mới: so với hai cột token khi Train
8. Bên trong Pipeline: vectorizer.transform → classifier score
9. Naive Bayes vs SVM: score, xác suất và ngưỡng
10. Ví dụ tính tay một dự đoán SVM
11. Cell 31–32: lưu `bundle` bằng `joblib.dump()`
12. `manifest`, `preprocessing_source` và các trường thông tin
13. `joblib.load()` và kiểm tra đúng mô hình sau nạp
14. Bốn ví dụ demo thật từ Notebook và nhận xét
15. Những loại file được xuất ra, vai trò của chúng
16. So sánh Train/Validation/Test với bình luận mới
17. Rủi ro, các trường hợp đặc biệt và giới hạn của code
18. Hướng dẫn chạy thử và kiểm tra model trong project
19. Bài tập thực hành + câu trả lời
20. Câu hỏi có thể bị hỏi khi bảo vệ
21. Bản tóm tắt nói trong 90 giây
22. Nguồn liên quan và checklist hoàn thành

---

## 1. Mục tiêu và phạm vi

Sau bước `evaluation.py`, ta đã có mô hình truyền thống **S3 = TF-IDF + LinearSVC + threshold được chỉnh trên CV**, đang nằm trong biến `best` của Notebook. Không ai muốn chạy lại toàn bộ GridSearchCV mỗi lần có bình luận Facebook mới. Vì vậy cần **lưu trạng thái đã học**, rồi nạp lại khi cần.

Hai nơi phụ trách việc khác nhau:

| Nơi | Thực hiện | Không làm |
|---|---|---|
| Notebook 02, **cell 32** | Tạo `manifest`, đóng gói `bundle`, lưu `.joblib`, nạp lại và so kết quả, gọi demo, xuất CSV báo cáo | Không chọn lại mô hình trên Test |
| `pipeline/modeling/predict.py` | Hàm `predict_spam(texts, model_bundle)` nhận câu thô, chuyển về token, dùng bundle để chấm, trả bảng | **Không** tự chạy `joblib.load/dump`, **không** tự fit, **không** tự gán nhãn thật |
| `pipeline/modeling/evaluation.py` | `predict_record` phân luồng NB/SVM để lấy score và áp threshold | Không tiền xử lý văn bản thô |
| `pipeline/prepare/preprocess_for_model.py` | `preprocess` tạo chuỗi token nhất quán | Không phải bộ phân loại |

`predict.py` hiện nằm **cùng thư mục `pipeline/modeling/`** với các file bạn phụ trách, chứ không phải đường dẫn `pipeline/inference/predict.py` của phiên bản cũ.

---

## 2. Bản đồ luồng end-to-end

```text
DATASET ĐÃ CÓ NHÃN (Notebook 02)
   cleaned_text + text gốc + label 0/1
               |
   [chuẩn bị tokens — thành viên khác]
               |
    training_utils.py: train/test + 5 folds
               |
    text_features.py: CountVectorizer / TfidfVectorizer (chưa fit)
               |
    classifiers.py: MultinomialNB / LinearSVC (chưa fit)
               |
    experiments.py: Pipeline + GridSearchCV.fit()
               |
    imbalance.py: so S1/S2/S3, chốt S3 bằng CV F1
               |
    evaluation.py: dự đoán TEST, báo cáo F1/Precision/Recall/AP
               |
      best['_model'] = Pipeline đã fit
      best['_threshold'] = -0.1776... (S3)
               |
          NOTEBOOK CELL 32
      bundle = {model, threshold, remove_stop, manifest, preprocessing_source}
               |
      joblib.dump(bundle, '...spam_ham_ml_only.joblib')
               |
       joblib.load(...) → loaded
               |
    ┌──────────┴───────────────────────────────────────┐
    │                          │                      │
 kiểm tra trên Test      predict_spam(câu mới,loaded)  xuất CSV/JSON
    │                          │
 dùng token Test       raw comment (chưa token hóa)
    │                          │
    │                 preprocess(..., remove_stop)
    │                          │
    │                 chuỗi token (không có nhãn)
    │                          │
    │               fitted vec.transform(token)
    │                          │
    │               fitted SVM.decision_function
    │                          │
    │               score > threshold ?
    │                          │
    │                nhãn 1=spam / 0=ham
    │                          │
    └─> khớp nhãn       DataFrame: text | score | score_type | du_doan
```

**Chốt:** Sau khi lưu, bộ từ điển (`vocabulary_`), IDF (`idf_`), hệ số SVM (`coef_`), bias (`intercept_`) và threshold phải được giữ nguyên khi dự đoán. `Pipeline` đã fit chứa hai bước `vec` và `clf` nên không cần lưu từ điển hay IDF riêng bằng tay.

---

## 3. Các khái niệm phải hiểu

| Khái niệm | Giải thích theo dự án |
|---|---|
| **Training / fit** | Dùng Train có nhãn để học từ điển/IDF, thông số NB/SVM. Chỉ diễn ra trước khi lưu |
| **Inference / predict** | Sử dụng thông số đã học để dự đoán câu mới, **không cần nhãn thật** |
| **Fitted model** | Đối tượng đã `fit`; với SVM thường có `coef_`, `intercept_`; TF-IDF có `vocabulary_`, `idf_` |
| **Pipeline** | Chuỗi các bước `vec` (biến text thành số) rồi `clf` (chấm điểm, dự đoán) |
| **Model serialization / persistence** | Chuyển trạng thái Python đã học thành file có thể nạp lại; ở đây dùng `joblib` |
| **Artifact** | Sản phẩm được xuất sau train: mô hình `.joblib`, báo cáo CSV, manifest JSON... |
| **Threshold (ngưỡng)** | Mốc trên score để xác định Spam/Ham; **không** phải tham số `C` của SVM |
| **Score** | Giá trị liên tục thể hiện kết quả chấm từ mô hình; NB và SVM khác định nghĩa |
| **`model_bundle`** | Dictionary chứa model đã fit **và** cấu hình cần thiết cho suy luận |
| **Train–serving consistency** | Text mới phải qua quy tắc tiền xử lý và biểu diễn tương thích với lúc train |
| **Smoke test** | Kiểm tra nhanh rằng model nạp được và dự đoán chạy; không thay đánh giá Test |
| **Reproducibility** | Khả năng tái lập hoặc truy vết cách mô hình được tạo: dữ liệu, seed, version, tham số, mã... |

### Nhãn 0/1 và dữ liệu đầu vào

- `0` = Ham (bình luận bình thường)
- `1` = Spam (bình luận rác/chào mời)
- Lúc train cần **`X_train`, `y_train`**.
- Lúc inference chỉ cần **văn bản mới** và **bundle đã học**: không có `y_true` để chấm đúng/sai.
- Không lấy score âm của LinearSVC làm “xác suất âm”: score chỉ là số có dấu.

---

## 4. Hàm `predict_spam(texts, model_bundle)` nhận gì và trả gì?

**Hàm duy nhất tự viết trong `predict.py` phiên bản mới:**

```python
def predict_spam(texts, model_bundle):
    ...
```

### Đầu vào `texts`

Chấp nhận:

```python
predict_spam("Cho mình hỏi lịch thi với ạ", bundle)           # một chuỗi = một comment
predict_spam(["Xin lịch học", "Liên hệ 0901234567"], bundle) # danh sách hai comment
predict_spam([], bundle)                                    # danh sách rỗng
```

Hàm cũng gọi `list(texts)` đối với các iterable khác (ví dụ tuple), nhưng **không có cơ chế kiểm tra toàn bộ phần tử đều là chuỗi**. Do đó, nên chủ động nhập `str` hoặc `list[str]`.

### Đầu vào `model_bundle`

Tối thiểu cần có:

```python
{
    "model": <Pipeline đã fit>,
    "remove_stop": False,          # hoặc True
    "threshold": -0.1776...,     # nếu thiếu, predict_record dùng giá trị mặc định
}
```

Trong Notebook bundle thật còn có `manifest` và `preprocessing_source` để truy vết. **Chúng không phải đầu vào trực tiếp để `predict_record` chấm điểm.**

### Đầu ra `pandas.DataFrame`

| Cột | Nội dung |
|---|---|
| `text` | Giữ lại bình luận gốc ở đầu vào |
| `score` | Số liên tục lấy từ mô hình (NB probability / SVM decision score) |
| `score_type` | Chuỗi `probability_spam` hoặc `decision_score` |
| `du_doan` | Chuỗi `spam` hoặc `ham` |

Ví dụ kết quả (giá trị thật được phân tích riêng tại mục 14):

```text
text                               score    score_type      du_doan
Cho mình hỏi lịch thi với ạ       -1.2543   decision_score   ham
Liên hệ 0901234567 hoặc URL        2.9688   decision_score   spam
```

Đây là bảng hiển thị đầu ra, **không phải** `DataFrame` chứa nhãn thật hoặc metrics Test.

---

## 5. Đọc toàn bộ mã nguồn `predict.py`

Phiên bản mới `predict(1).py`, giữ nguyên cấu trúc logic:

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
    texts = [texts] if isinstance(texts, str) else list(texts)
    if not texts:
        return pd.DataFrame(columns=["text", "score", "score_type", "du_doan"])
    X = [preprocess(text, text, model_bundle["remove_stop"]) for text in texts]
    record = {"_model": model_bundle["model"]}
    if "threshold" in model_bundle:
        record["_threshold"] = model_bundle["threshold"]
    pred, score = predict_record(record, X)
    return pd.DataFrame({
        "text": texts,
        "score": score,
        "score_type": (
            "probability_spam" if hasattr(model_bundle["model"], "predict_proba")
            else "decision_score"
        ),
        "du_doan": np.where(pred == 1, "spam", "ham")
    })
```

Lưu ý phần xuống dòng cuối khác bố cục file gốc để dễ đọc, nhưng **không đổi ý nghĩa lệnh**.

---

## 6. Phân tích TỪNG CÂU LỆNH trong `predict_spam()`

### 6.1. `import numpy as np`, `import pandas as pd`

- `numpy`: dùng `np.where()` để đổi nhãn số sang chuỗi cho từng bình luận.
- `pandas`: tạo bảng `pd.DataFrame()` dễ hiển thị, lưu CSV, xử lý tiếp.
- Không có import `MultinomialNB` hoặc `LinearSVC` trong file này vì **mô hình đã nằm bên trong Pipeline** được nạp.

### 6.2. `from ...evaluation import predict_record`

Hàm `predict_spam` **không viết lại công thức** score của NB/SVM. Nó nhờ `predict_record` thực hiện việc lấy score và so ngưỡng, tránh hai nơi sử dụng hai quy tắc khác nhau.

### 6.3. `from ...preprocess_for_model import preprocess`

Mượn hàm tiền xử lý chung để text mới giống kiểu chuỗi token lúc train. Đây là chỗ duy nhất `predict.py` phụ thuộc phần tiền xử lý của thành viên khác; bạn chỉ cần hiểu giao diện và nguyên tắc nhất quán.

### 6.4. `texts = [texts] if isinstance(texts, str) else list(texts)`

**Vấn đề cần giải quyết:** Trong Python, chuỗi cũng là một iterable. Nếu gọi `list("spam")` sẽ nhận `['s','p','a','m']` — sai vì một bình luận bị tách thành bốn bình luận.

```python
texts = "Xin lịch thi"
isinstance(texts, str)  # True
texts = [texts]          # ["Xin lịch thi"] => một dòng
```

Với list/tuple, `list(texts)` giữ từng bình luận thành một phần tử. **Hàm không tách chuỗi thành từ tại bước này**; tách token nằm ở `preprocess`.

### 6.5. `if not texts: return pd.DataFrame(...)`

Nếu không nhận bình luận nào thì trả bảng rỗng có đúng bốn cột. Đây là **lối thoát sớm**, tránh gửi mảng rỗng vào Pipeline vốn có thể không nhận input rỗng. Không phải lỗi, không phải dự đoán Ham.

```python
# Output hình thức:
Empty DataFrame
Columns: [text, score, score_type, du_doan]
Index: []
```

### 6.6. `X = [preprocess(text, text, model_bundle["remove_stop"]) for text in texts]`

Đây là **list comprehension**: chạy `preprocess()` cho mỗi comment.

- Tham số thứ nhất `text`: được truyền vào `cleaned` (trong predict, dùng text gốc để hàm tự làm sạch).
- Tham số thứ hai `text`: được truyền vào `original`, dùng thêm để nhận dấu hiệu emoji và các thông tin ở text gốc.
- Tham số thứ ba `remove_stop`: cờ để giữ/bỏ stopwords **theo cấu hình đã chọn lúc huấn luyện**.

Đầu ra `X` là **list các chuỗi token**, ví dụ **minh họa**:

```python
X = ["cho mình hỏi lịch_thi với ạ", "liên_hệ phonetoken hoặc urltoken"]
```

Không coi chuỗi minh họa này là output underthesea chính xác cho mọi phiên bản; tách từ thực tế phụ thuộc phiên bản và biểu thức của `preprocess`.

**Quan trọng:** tại đây `X` **chưa** là ma trận số! Bộ vector hóa mới xử lý trong Pipeline.

### 6.7. `record = {"_model": model_bundle["model"]}`

`evaluation.predict_record()` trước đó được viết nhận một dictionary `record` có khóa `"_model"`, `"_threshold"`. Nhưng cell 32 lưu bundle bằng khóa `"model"`, `"threshold"`.

Do đó cần **bọc lại key** thành `record` đúng giao diện:

```text
bundle["model"]  → record["_model"]
bundle["threshold"] → record["_threshold"]
```

Không tạo thêm mô hình; hai khóa trỏ đến **cùng Pipeline đã fit**.

### 6.8. `if "threshold" in model_bundle: ...`

Nếu bundle chứa ngưỡng đã chọn (S3), chuyển sang record để `predict_record` sử dụng. Nếu thiếu ngưỡng:

- NB có default `0.5`.
- SVM có default `0.0`.

Vì S3 dùng ngưỡng **khác 0**, việc giữ `threshold` trong bundle là **bắt buộc về mặt tái tạo đúng kết quả S3**, dù code không yêu cầu khóa này để chạy.

### 6.9. `pred, score = predict_record(record, X)`

Hàm trả một **tuple** hai đối tượng tương ứng:

- `pred`: nhãn số 0/1 của từng bình luận.
- `score`: score liên tục của từng bình luận.

Thứ tự output phải được giữ đúng! Không nhầm `pred` với score; hai thứ không tương đương.

### 6.10. `hasattr(model_bundle["model"], "predict_proba")`

Dùng để phân biệt API score:

- Với `Pipeline(..., MultinomialNB)`, mô hình hỗ trợ `predict_proba` → tên loại score là `probability_spam`.
- Với `Pipeline(..., LinearSVC)`, không có `predict_proba` → `decision_score`.

`hasattr` kiểm tra xem phương thức có sẵn qua Pipeline (phụ thuộc classifier ở bước cuối). Không phải kiểm tra độ tin cậy hoặc tính đúng sai của dự đoán.

### 6.11. `np.where(pred == 1, "spam", "ham")`

Với `pred=[0,1,1,0]`, điều kiện `pred==1` là `[False,True,True,False]` → trả `['ham','spam','spam','ham']`.

Điều này **chỉ đổi cách hiển thị**. Không gán lại nhãn thật cho dataset; không huấn luyện thêm.

### 6.12. `pd.DataFrame({...})`

Pandas ghép bốn mảng cùng số phần tử thành bảng; mỗi hàng là một câu gốc. Nếu đầu vào n bình luận, đầu ra có n hàng (trừ đầu vào rỗng có 0 hàng).

Đây là điểm cuối của hàm.

---

## 7. `preprocess_for_model.py` liên quan thế nào?

**Bạn không phụ trách code tiền xử lý, nên chỉ cần nắm đường gọi này cho chính xác.**

Trong file kèm theo, giao diện là:

```python
def preprocess(cleaned, original=None, remove_stop=False):
    original = cleaned if original is None else original
    has_emoji = bool(EMOJI.search(str(cleaned)) or EMOJI.search(str(original)))
    text = mask_names(clean_text(cleaned)).lower()
    text = REPEAT_LETTER.sub(r"\1\1", text)
    tokens = word_tokenize(text, format="text").split() if text else []
    if remove_stop:
        tokens = [token for token in tokens if token not in STOP_WORDS]
    if has_emoji:
        tokens.append("emoji_token")
    return " ".join(tokens)
```

Các thao tác chính:

1. Dọn HTML, URL → `urltoken`, số điện thoại → `phonetoken`, Unicode, emoji.
2. Che một số username/tên dựa trên quy tắc `mask_names`, không hứa ẩn danh hoàn toàn.
3. Hạn chế ký tự lặp, tách từ bằng `underthesea.word_tokenize()`.
4. Nếu `remove_stop=True`, loại từ trong `STOP_WORDS`.
5. Thêm `emoji_token` nếu văn bản gốc có emoji, trả chuỗi token nối bằng khoảng trắng.

**Điểm nối với Notebook cell 9:**

```python
prepared["tokens_keep_stopwords"] = [
    preprocess(c, t, False)
    for c, t in zip(raw.cleaned_text, raw.text)
]
prepared["tokens_remove_stopwords"] = (
    prepared.tokens_keep_stopwords.map(
        lambda text: " ".join(token for token in text.split() if token not in STOP_WORDS)
    )
)
```

Cell 13 thử **cả hai** phiên bản token, rồi chọn `MAIN_PRE` bằng CV. Khi lưu:

```python
"remove_stop": best["Tiền xử lý"] == "có loại stop words"
```

Nghĩa là `remove_stop` trong bundle phải phản ánh lựa chọn này. Với lần chạy Notebook đã lưu, nhánh thắng là **giữ stopwords**, nên `remove_stop=False`.

**Cảnh báo tinh tế:** Khi Train, `preprocess(cleaned_text, text_gốc, False)` nhận hai nguồn dữ liệu khác nhau; khi dự đoán mới, `preprocess(text_gốc, text_gốc, remove_stop)` xử lý trực tiếp câu gốc. Do `cleaned_text` có sẵn có thể khác những gì `clean_text(text_gốc)` tạo ra, **score có thể khác nhau** dù nhãn đã kiểm chứng trùng trên Test. Notebook chính thức thừa nhận khả năng này trong cell 31.

---

## 8. Bên trong fitted `Pipeline`: `transform` trước, `decision_function` sau

### 8.1. Nơi Pipeline được tạo trong dự án

```python
# experiments.py
pipe = Pipeline([("vec", vectorizer), ("clf", classifier)], memory=cache)
```

Ở các bước trước, `GridSearchCV.fit()` đã fit Pipeline và `search.best_estimator_` là phiên bản thắng đã fit lại trên toàn Train.

Lúc mới dự đoán:

```python
model = loaded["model"]   # Pipeline đã fit
scores = model.decision_function(X)  # X vẫn là chuỗi token
```

Nhưng Pipeline **không** đưa chuỗi trực tiếp vào `LinearSVC`!

**Đường gọi khái niệm bên trong:**

```python
# ĐÂY LÀ PSEUDOCODE ĐỂ HỌC, KHÔNG PHẢI ĐOẠN NGUYÊN VĂN SKLEARN
vec = model.named_steps["vec"]
clf = model.named_steps["clf"]
X_matrix = vec.transform(X)               # KHÔNG gọi fit_transform
scores = clf.decision_function(X_matrix)  # SVM
```

Nếu model là NB, bước cuối có thể gọi `clf.predict_proba(X_matrix)[:, 1]`.

### 8.2. Vectorizer nhớ gì sau Train?

Đối với S3: `TfidfVectorizer` đã nhớ:

- `vocabulary_`: tên unigram/bigram → chỉ số cột;
- `idf_`: trọng số IDF đã học từ Train;
- `ngram_range=(1,2)`, `min_df=2`, regex tách token, `norm='l2'`, `sublinear_tf=True`, ...

`transform()` dùng **chính từ điển cũ**. Từ mới không có trong vocabulary sẽ không sinh cột mới. IDF không tính lại trên bình luận mới. Nếu vectorizer được `fit_transform()` lại trên một bình luận mới thì cột và IDF có thể đổi; SVM sẽ nhận sai hệ tọa độ hoặc lỗi số cột.

### 8.3. SVM nhớ gì sau Train?

`LinearSVC` đã nhớ:

- `classes_`: các nhãn `[0, 1]`;
- `coef_`: một trọng số cho mỗi cột đặc trưng;
- `intercept_`: hệ số chặn (bias);
- Cấu hình huấn luyện như `C=0.3` trong S3 được chọn bằng CV.

`decision_function(X_matrix)` tính dạng `X_matrix @ coef_.T + intercept_`. Không cập nhật `coef_`.

### 8.4. Kiểm tra thủ công hai bước trong project

```python
import joblib
import numpy as np

bundle = joblib.load("artifacts/spamham/spam_ham_ml_only.joblib")
pipe = bundle["model"]
print(pipe.named_steps)    # tên hai bước vec, clf

new_tokens = ["nhận làm đồ_án"]  # CHỈ MINH HỌA: cần preprocess thật trước
M = pipe.named_steps["vec"].transform(new_tokens)
print("vector shape:", M.shape, "nonzero:", M.nnz)

score_by_clf = pipe.named_steps["clf"].decision_function(M)
score_by_pipe = pipe.decision_function(new_tokens)
print(score_by_clf, score_by_pipe)
assert np.allclose(score_by_clf, score_by_pipe)
```

Đoạn trên áp dụng khi bundle chứa SVM như Notebook đang chọn; với NB, thay `decision_function` bằng `predict_proba`.

---

## 9. `evaluation.predict_record()` tính gì?

Mã thực của module liên quan:

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

### 9.1. Nhánh Naive Bayes

- `predict_proba(X)` trả ma trận **n×2** (hai cột Ham, Spam, nếu `classes_=[0,1]`).
- `[:, 1]` lấy **xác suất ước lượng** của Spam.
- Mặc định nhãn Spam khi `score >= 0.5`.
- Với xác suất `0.72`: Spam; với `0.32`: Ham.
- Đây là xác suất theo mô hình **chưa chắc đã hiệu chuẩn** với xác suất thực tế ngoài đời.

### 9.2. Nhánh LinearSVC / S3

- `decision_function(X)` trả một **decision score** trên mỗi bình luận.
- Với `classes_=[0,1]`, score dương nghiêng về Spam; score âm nghiêng về Ham theo ngưỡng mặc định 0.
- S3 dùng ngưỡng đã tối ưu từ OOF/CV trên Train, trong Notebook gần `−0.1776`.
- Khi `score > threshold`, nhãn `1=Spam`; còn lại `0=Ham`.

**Ví dụ ngưỡng (minh họa):**

| Score | Với B4 ngưỡng 0 | Với S3 ngưỡng −0.1776 |
|---:|---|---|
| −1.20 | Ham | Ham |
| −0.10 | Ham | **Spam** |
| +0.35 | Spam | Spam |

Việc đổi ngưỡng không đổi `coef_`, không biến score thành xác suất, không yêu cầu fit lại SVM cho từng bình luận. Chú ý NB dùng `>=`, còn SVM trong code dùng `>` — ở đúng ngưỡng, nhãn khác nhau theo phép so sánh.

---

## 10. Ví dụ tính tay một lần dự đoán (minh họa, KHÔNG phải trọng số của model PTIT)

Để hiểu cơ chế, giả sử vectorizer đã học ra ba đặc trưng:

```text
vocabulary = ["lịch", "nhận", "làm"]
```

Với câu đã tiền xử lý, `transform()` trả vector TF-IDF ví dụ:

```text
X_new = [0.0, 0.8, 0.6]
```

Và SVM đã học:

```text
w = [-0.8, +0.5, +1.0]
b = -0.2
```

Score là:

```text
s = 0.0*(-0.8) + 0.8*0.5 + 0.6*1.0 - 0.2
  = 0.8
```

Vì `0.8 > -0.1776` nên kết quả là `Spam`.

Nếu một câu khác có score `−0.10`, ngưỡng B4 (0) trả Ham nhưng ngưỡng S3 (−0.1776) trả Spam.

**Cần phân biệt:** Các trọng số và vector ba chiều trên do ta tự chọn để hiểu phép toán. Mô hình PTIT thật có số đặc trưng lớn hơn rất nhiều và các hệ số được học tự động.

### Ví dụ bằng code Python độc lập (không yêu cầu dataset PTIT)

```python
import numpy as np
from sklearn.svm import LinearSVC

X = np.array([
    [2., 0.], [1.5, 0.],      # ví dụ hai spam
    [0., 2.], [0., 1.5],      # ví dụ hai ham
])
y = np.array([1, 1, 0, 0])

clf = LinearSVC(C=0.3, dual=False, max_iter=20000)
clf.fit(X, y)

new_vector = np.array([[1.8, 0.2]])
score = clf.decision_function(new_vector)
threshold = -0.1776
pred = (score > threshold).astype(int)

print('coef_:', clf.coef_)
print('intercept_:', clf.intercept_)
print('score:', score, 'pred:', pred)
```

Code này tập trung vào bước `LinearSVC`; nó không thay thế vectorizer, cũng không đưa ra score thực tế của PTIT.

---

## 11. Đọc cell 32: `manifest`, `bundle`, `joblib.dump()`

### 11.1. `best` từ đâu ra?

Ở cell 16, trước khi Test:

```python
CANDIDATES = RESULTS + SUPPLEMENTARY
best = max(CANDIDATES, key=lambda r: r["CV F1 (spam)"])
```

Như các bước trước: mô hình được chọn **bằng CV F1 Spam**, không phải bằng Test F1. Trong output hiện có, `best` là S3 (TF-IDF + LinearSVC đã fit, có ngưỡng được chỉnh).

Ở cell 20, `evaluate(best, X_test, y_test)` đã thêm metrics và `_pred`. Đến cell 32 mới lưu lại.

### 11.2. `manifest` là gì?

Đoạn đầu cell 32:

```python
manifest = {
    "dataset_file": str(CSV_PATH),
    "processed_file": str(PROCESSED_PATH),
    "audit": DATA_AUDIT,
    "train_rows": len(train_idx),
    "test_rows": len(test_idx),
    "test_spam": int(y_test.sum()),
    "seed": SEED,
    "environment": ENVIRONMENT,
    "python": sys.version.split()[0],
    "completed_utc": datetime.now(timezone.utc).isoformat(),
    "modern_model": "reported_separately_in_notebook_03",
    "selection_rule": "max mean validation F1 on train CV; test not used for selection",
    "baseline_configurations": len(RESULTS),
    "additional_strategies": len(SUPPLEMENTARY),
    "selected": {
        k: best[k] for k in [
            "Mã", "Tiền xử lý", "Đặc trưng", "Thuật toán", "Chiến lược",
            *CV_COLS, *METRICS, "Tham số"
        ]
    },
    "decision_threshold": best["_threshold"],
}
```

**Ý nghĩa:** Đây là bản mô tả lần chạy (metadata), giúp biết model đến từ dataset nào, số mẫu, cấu hình, phiên bản, thời điểm, quy tắc lựa chọn và điểm đo. Không nên nhầm `manifest` với object học máy hoặc tập huấn luyện. Dù `manifest` có chỉ đến đường dẫn CSV, **`predict_spam()` không đọc lại CSV để dự đoán**.

### 11.3. Đóng gói `bundle`

```python
bundle = {
    "model": best["_model"],
    "threshold": best["_threshold"],
    "remove_stop": best["Tiền xử lý"] == "có loại stop words",
    "manifest": manifest,
    "preprocessing_source": inspect.getsource(preprocessing_module),
}
```

**Giải thích từng khóa:**

| Khóa | Giá trị/ý nghĩa | Cần để dự đoán trực tiếp? |
|---|---|---|
| `model` | Fitted Pipeline gồm `vec` và `clf` | **Có** |
| `threshold` | Ngưỡng quyết định của S3 | **Có để giữ đúng S3** |
| `remove_stop` | Cờ sử dụng đúng nhánh giữ/bỏ stopwords | **Có** |
| `manifest` | Thông tin lần train và đánh giá | Không, dùng để truy vết |
| `preprocessing_source` | Chuỗi mã nguồn module tiền xử lý tại lúc lưu | Không trực tiếp, dùng đối chiếu nguồn |

`inspect.getsource(preprocessing_module)` **ghi nội dung code vào chuỗi** để tham khảo. `predict_spam()` vẫn thực sự `import preprocess` từ module đang cài, **không tự thực thi chuỗi source đã lưu**. Nếu sau này mã tiền xử lý bị sửa, có nguy cơ khác biệt đầu vào dù dùng cùng bundle. Đây là điều nên nhấn mạnh khi bảo vệ.

### 11.4. Tại sao Notebook đặt `record["_model"].memory = None` trước khi lưu?

```python
for record in PREPROCESS_RESULTS + SUPPLEMENTARY:
    record["_model"].memory = None
```

Trong lúc chạy GridSearchCV, Pipeline sử dụng cache `joblib.Memory` để tránh tính lặp tốn thời gian. Khi đã fit xong, không cần mang tham chiếu thư mục cache tạm vào model được lưu. `memory=None` chỉ **tắt cấu hình cache của Pipeline**, không xóa hệ số classifier, không xóa `vocabulary_`, `idf_`, và không gọi `fit()`.

### 11.5. Lệnh `joblib.dump()` thực sự làm gì?

```python
model_path = ARTIFACT_DIR / "spam_ham_ml_only.joblib"
joblib.dump(bundle, model_path)
```

- `ARTIFACT_DIR` do cell 3 cấu hình, mặc định `artifacts/spamham/` trong project.
- Toán tử `/` của `pathlib.Path` ghép đường dẫn chứ không phải phép chia toán học.
- `joblib.dump(obj, path)` **tuần tự hóa** object Python (dict, pipeline và các NumPy arrays bên trong) ra file.
- File lưu **trạng thái đã fit** của vectorizer/classifier. Không cần tạo vectorizer mới, không cần chạy `GridSearchCV` khi tải lại.
- `.joblib` không phải file văn bản: không nên mở bằng text editor để chỉnh tay.

---

## 12. `joblib.load()`: nạp lại đúng trạng thái

```python
loaded = joblib.load(model_path)
```

`loaded` là dictionary có các khóa đã lưu. Bước này khôi phục model ở bộ nhớ để dùng tiếp.

```python
print(loaded.keys())
# kỳ vọng:
# dict_keys(['model','threshold','remove_stop','manifest','preprocessing_source'])
```

Nên kiểm tra:

```python
print(type(loaded["model"]))
print(loaded["model"].named_steps.keys())
print(loaded["threshold"])
print(loaded["remove_stop"])
print(loaded["manifest"]["selection_rule"])
```

### Yêu cầu quan trọng về an toàn và phiên bản

`joblib.load()` dùng cơ chế pickle: **chỉ nạp file `.joblib` đáng tin cậy**, bởi file độc hại có thể thực thi code khi nạp. Ngoài ra phiên bản Python/scikit-learn và các module liên quan cần phù hợp; model được lưu không cam kết chạy y hệt trên mọi phiên bản. Trong bundle có trường `environment` và `python` giúp truy vết, không tự khắc phục vấn đề tương thích.

Không gửi hoặc sửa `joblib` tùy tiện nếu file chứa thông tin nhạy cảm, kể cả khi mô hình chỉ học văn bản.

---

## 13. Kiểm tra sau nạp: Notebook làm NHỮNG GÌ?

### 13.1. Kiểm tra bằng chuỗi token Test đã chuẩn bị

```python
X_verify = TEXT[best["Tiền xử lý"]].iloc[test_idx].to_numpy()
reloaded_pred, _ = predict_record(
    {"_model": loaded["model"], "_threshold": loaded["threshold"]},
    X_verify
)
assert np.array_equal(reloaded_pred, best["_pred"])
```

- `X_verify`: những chuỗi token Test đúng cấu hình khi đánh giá ở cell 20.
- `reloaded_pred`: kết quả sau nạp lại bundle.
- `best['_pred']`: kết quả đã tính trước khi lưu.
- `np.array_equal` kiểm tra mọi nhãn tương ứng đều khớp, không chỉ tổng Accuracy.
- `assert` sẽ báo lỗi nếu không khớp. Đây là kiểm tra tính toàn vẹn của việc lưu/nạp.

**Không phải đang chấm Test lại để chọn mô hình**; chỉ xác nhận model được nạp lại cho cùng dự đoán như trước.

### 13.2. Kiểm tra đường đầu vào bình luận GỐC

```python
raw_predictions = predict_spam(raw.iloc[test_idx].text.tolist(), loaded)
assert np.array_equal(
    raw_predictions.du_doan.eq("spam").astype(int).to_numpy(),
    best["_pred"]
)
```

- `raw.iloc[test_idx].text.tolist()`: lấy bình luận gốc Test.
- `predict_spam()` tiền xử lý lại text gốc, vector hóa và dự đoán bằng bundle.
- `.eq('spam')` tạo True/False, `.astype(int)` đổi thành 1/0.
- So với `best['_pred']` nhằm kiểm chứng **đường suy luận thực tế từ text gốc** khớp nhãn của đường Test đã chuẩn bị trước.

**Kết quả Notebook in ra:**

```text
Đã lưu: artifacts/spamham/spam_ham_ml_only.joblib
Dự đoán sau khi nạp lại và từ bình luận gốc khớp trên toàn bộ 7474 mẫu test.
```

Đây là một điểm mạnh có thể trình bày. **Chỉ khẳng định nhãn khớp**, không suy ra mọi decision score đều bằng nhau hoặc rằng mọi dự đoán đều đúng so với nhãn thật.

### 13.3. Tại sao score có thể không trùng dù nhãn trùng?

Đường đánh giá dùng `preprocess(cleaned_text, text_gốc, False)` ở cell 9, còn đường suy luận gọi `preprocess(text_gốc,text_gốc,remove_stop)` ở `predict.py`. Nếu trường `cleaned_text` của dataset đã chuẩn hóa khác đầu ra làm sạch lại từ text gốc, hai chuỗi token có thể khác một chút; score SVM vì vậy cũng có thể khác, nhưng vẫn nằm cùng phía ngưỡng.

---

## 14. Bốn ví dụ demo THẬT trong cell 32

Lệnh Notebook:

```python
display(predict_spam([
    "Cho mình hỏi lịch thi với ạ",
    "Liên hệ 0901234567 hoặc https://example.com",
    "Nguyễn Văn An",
    "."
], loaded))
```

Bảng output đã lưu (giá trị đã làm tròn khi trình bày):

| Bình luận gốc | Decision score | Ngưỡng S3 | Nhãn mô hình |
|---|---:|---:|---|
| Cho mình hỏi lịch thi với ạ | −1.254256 | ~−0.1776 | `ham` |
| Liên hệ 0901234567 hoặc https://example.com | +2.968799 | ~−0.1776 | `spam` |
| Nguyễn Văn An | +0.688906 | ~−0.1776 | `spam` |
| . | +0.986103 | ~−0.1776 | `spam` |

### Đọc kết quả từng câu

**Câu 1**: score nhỏ hơn ngưỡng → Ham. Nội dung hỏi lịch thi phù hợp với nhãn dự đoán Ham; đây chỉ là nhận xét tình huống, chưa có nhãn thật độc lập cho câu demo.

**Câu 2**: score dương lớn → Spam. Có URL/SĐT, những dấu hiệu quan trọng được tiền xử lý thành token. Không nên kết luận cứ có URL hay SĐT là Spam: mô hình tính tổng các đặc trưng.

**Câu 3**: là một tên người, vẫn bị dự đoán Spam. Không có bằng chứng bản thân tên người là Spam; đó là ví dụ về khả năng **false positive nếu nhãn thật là Ham**. Chức năng `mask_names` có thể biến tên theo quy tắc thành `user_name`; không khẳng định câu này được token hóa chính xác thành từ gì khi chưa chạy `underthesea` trong đúng môi trường.

**Câu 4**: chỉ có một dấu câu, vẫn bị dự đoán Spam. Một bình luận gần như không có thông tin vẫn có thể nhận score theo bias và đặc trưng còn lại. Không thể kết luận nguyên nhân chính xác của score chỉ từ bảng output; muốn biết phải kiểm tra `preprocess` và `vec.transform` của model đã nạp.

**Kết luận quan trọng:** Demo cho thấy hàm chạy và trả kết quả, **không chứng minh mô hình đúng 100%**. Chất lượng phải dựa vào tập Test có nhãn thật và các metric trong bước `evaluation.py`.

---

## 15. Ngoài `.joblib`, Notebook còn xuất ra những gì?

Cuối cell 32:

```python
metrics.to_csv(ARTIFACT_DIR / "metrics.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(ci).to_csv(ARTIFACT_DIR / "bootstrap_ci.csv", index=False)
(ARTIFACT_DIR / "run_manifest.json").write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
)
cv_comparison.to_csv(ARTIFACT_DIR / "stopword_cv.csv", encoding="utf-8-sig")
errors.to_csv(ARTIFACT_DIR / "test_predictions.csv", index=False, encoding="utf-8-sig")
cache_dir.cleanup()
```

| File được tạo | Nội dung | Mục đích |
|---|---|---|
| `spam_ham_ml_only.joblib` | Bundle model + threshold + stopword flag + metadata | **Dùng cho dự đoán** |
| `metrics.csv` | Điểm CV/Test và thông số các ứng viên | So sánh, báo cáo |
| `bootstrap_ci.csv` | Khoảng bootstrap F1 | Phân tích độ bất định |
| `run_manifest.json` | Metadata lần train | Truy vết |
| `stopword_cv.csv` | So sánh hai cách xử lý stopwords trên CV | Giải thích lựa chọn tiền xử lý |
| `test_predictions.csv` | Bảng phân tích dự đoán/lỗi Test do Notebook chuẩn bị | Kiểm tra các trường hợp sai |

Ở cell 16 còn xuất `imbalance_cv.csv` để lưu so sánh CV của B4/S1/S2/S3. Các hình trong các cell trước đã được lưu ở các lệnh tạo biểu đồ; không nhầm chúng với nhiệm vụ `predict_spam()`.

`cache_dir.cleanup()` xóa tài nguyên tạm được dùng trong lúc GridSearchCV. Việc này không xóa model `.joblib` đã lưu ở `ARTIFACT_DIR`.

---

## 16. Các giai đoạn dùng văn bản khác nhau — đừng nhầm

| Giai đoạn | Đầu vào X | Có nhãn y? | Học tham số? | Phương thức |
|---|---|---|---|---|
| Train trên fold | Chuỗi token của fold Train | Có | Có | `vec.fit_transform` → `clf.fit` |
| Validation | Chuỗi token của fold Validation | Có để chấm | Không | `vec.transform` → `clf.predict` |
| Fit cuối trên toàn Train | Chuỗi token Train | Có | Có | `Pipeline.fit` |
| Test độc lập | Chuỗi token Test | Có để chấm | Không | `predict_record` + `evaluate` |
| **Dự đoán mới** | **Bình luận thô** | **Không** | **Không** | `preprocess` → `predict_record` |

Vì sao vẫn cần `text_features.py` và `classifiers.py` trong môi trường chạy? Các lớp và Pipeline đã được tuần tự hóa; khi nạp, Python/scikit-learn cần những định nghĩa lớp và phụ thuộc tương ứng. Tuy nhiên, **bản thân `predict_spam` không gọi `make_vectorizer`, không gọi `make_classifier` và không gọi `fit_experiment`**.

---

## 17. Giới hạn và lỗi dễ mắc

### 17.1. Dùng `model.predict()` của SVM có tương đương S3 không?

**Không hoàn toàn.** `LinearSVC.predict` sử dụng mốc mặc định **0**. Nhưng S3 chọn `threshold ≈ −0.1776`. Nếu dùng `pipe.predict(X)` thay vì `predict_record` thì bạn đã **bỏ qua ngưỡng S3**, một số nhãn có thể đổi.

### 17.2. Vì sao không fit lại vectorizer khi có câu mới?

SVM đã học trọng số theo đúng thứ tự và số cột của vocabulary Train. Học lại vocabulary sẽ thay hệ tọa độ; IDF cũng đổi, làm dự đoán không còn dựa trên hệ đặc trưng đã học.

### 17.3. `preprocessing_source` có bảo đảm dùng đúng preprocessing không?

Không. Nó chỉ **lưu bản văn mã nguồn để đối chiếu**. `predict.py` vẫn import hàm hiện tại từ module. Để reproducibility mạnh hơn cần lưu phiên bản/mã hash môi trường hoặc đóng gói cố định pipeline tiền xử lý, rồi kiểm tra đầu vào/đầu ra sau triển khai.

### 17.4. Score > 0 có phải 100% Spam?

Không. SVM score **không phải xác suất**; số càng cao chỉ là điểm nghiêng về lớp 1 theo ranh giới đã học. NB có `predict_proba` nhưng xác suất đó cũng không được đảm bảo đã hiệu chuẩn với thực tế.

### 17.5. Nếu input rỗng hoặc thiếu key?

- `predict_spam([], bundle)` trả bảng rỗng, không gọi model.
- `predict_spam(None, bundle)` có thể lỗi vì `list(None)` không hợp lệ.
- Thiếu `model` hoặc `remove_stop` trong bundle sẽ dẫn tới lỗi truy cập key.
- Thiếu `threshold` có thể vẫn chạy do `predict_record` có default, **nhưng không còn tái hiện được S3**.
- Dùng `joblib.load` với file không đáng tin có nguy cơ bảo mật.

### 17.6. Những lỗi mô hình quan sát được

- Có thể nhận nhầm tên người/dấu câu thành Spam trong demo.
- Bình luận có URL hoặc SĐT không tự động đồng nghĩa Spam; chỉ là dấu hiệu đầu vào.
- Không có nhãn thật ở bình luận mới nên không thể tự biết dự đoán đúng hay sai.
- Trên Test, S3 vẫn có **127 FP** và **297 FN**; cần ghi nhận giới hạn này thay vì chỉ nói Accuracy cao.

### 17.7. Sai khác đường đi `cleaned_text` vs text gốc

Notebook đã kiểm chứng nhãn của toàn bộ Test khớp giữa hai đường. Nhưng nếu sau này nguồn dữ liệu mới có nhiều định dạng khác biệt, nên có thêm kiểm thử đầu-cuối để bảo đảm tiền xử lý không làm lệch đáng kể score hoặc nhãn.

---

## 18. Code thực hành để bạn chạy trong project

> Các cell dưới đây dành cho môi trường project đã chạy Notebook 02 và đã có file `.joblib` trong `artifacts/spamham/`. Trong môi trường tài liệu hiện tại, file `.joblib` thực tế và dataset gốc không được cung cấp nên **không thể tự chạy lại** bốn giá trị score Notebook để xác minh độc lập.

### Thực hành 1: Nạp bundle, xem cấu hình

```python
from pathlib import Path
import joblib

path = Path("artifacts/spamham/spam_ham_ml_only.joblib")
loaded = joblib.load(path)

print('Keys:', loaded.keys())
print('Cấu hình:', loaded['manifest']['selected'])
print('Threshold:', loaded['threshold'])
print('Bỏ stopwords?:', loaded['remove_stop'])
print('Pipeline:', loaded['model'])
```

### Thực hành 2: Xem vectorizer và classifier đã fit

```python
model = loaded['model']
vec = model.named_steps['vec']
clf = model.named_steps['clf']

print('Số đặc trưng:', len(vec.vocabulary_))
print('N-gram:', vec.ngram_range)
print('Classifier:', type(clf).__name__)
print('Số lớp:', clf.classes_)
print('Shape coef:', clf.coef_.shape)   # chỉ dành cho LinearSVC
print('Intercept:', clf.intercept_)
```

### Thực hành 3: Dự đoán một câu và nhiều câu

```python
from pipeline.modeling.predict import predict_spam

print(predict_spam('Cho mình hỏi lịch thi với ạ', loaded))
print(predict_spam([
    'Cho mình hỏi lịch thi với ạ',
    'Liên hệ 0901234567 hoặc https://example.com',
    'Nguyễn Văn An',
    '.'
], loaded))
```

### Thực hành 4: Nhìn rõ text → token → vector → score → nhãn

```python
from pipeline.prepare.preprocess_for_model import preprocess
import numpy as np

raw_text = 'Liên hệ 0901234567 hoặc https://example.com'
tokens = preprocess(raw_text, raw_text, loaded['remove_stop'])
vec = loaded['model'].named_steps['vec']
clf = loaded['model'].named_steps['clf']

matrix = vec.transform([tokens])
score = float(clf.decision_function(matrix)[0])  # chỉ đúng nhánh SVM
pred = int(score > loaded['threshold'])

print('Raw:', raw_text)
print('Tokens:', tokens)
print('Vector shape:', matrix.shape)
print('Nonzero features:', matrix.nnz)
print('Score:', score)
print('Threshold:', loaded['threshold'])
print('Label:', 'spam' if pred else 'ham')
```

Với `matrix` là CSR sparse matrix, nên xem `matrix.nnz` hoặc liệt kê các đặc trưng khác 0 thay vì in toàn bộ ma trận vài chục nghìn cột.

### Thực hành 5: Liệt kê đặc trưng đóng góp vào score SVM

```python
feature_names = vec.get_feature_names_out()
row = matrix.tocsr().getrow(0)
weights = clf.coef_[0]
items = sorted(
    [(feature_names[j], float(v), float(weights[j]), float(v * weights[j]))
     for j, v in zip(row.indices, row.data)],
    key=lambda item: abs(item[3]), reverse=True
)
for name, value, weight, contribution in items[:15]:
    print(name, 'tfidf=', round(value,4), 'w=', round(weight,4),
          'contribution=', round(contribution,4))
print('bias:', float(clf.intercept_[0]))
print('score:', score)
```

**Ý nghĩa:** Đóng góp mỗi đặc trưng là `TF-IDF × trọng số SVM`. Tổng tất cả đóng góp + bias = decision score. Điều này giúp giải thích vì sao tên người hoặc dấu câu có thể bị nghiêng sang Spam, nhưng **không khẳng định nguyên nhân** trước khi bạn chạy phân tích này trên model thật.

### Thực hành 6: So ngưỡng S3 với mốc mặc định

```python
s = loaded['model'].decision_function([tokens])[0]
print('B4-like threshold=0:', 'spam' if s > 0 else 'ham')
print('S3 chosen threshold:', 'spam' if s > loaded['threshold'] else 'ham')
```

Chỉ thay đổi ngưỡng trên **cùng score**, không fit lại. Đây là thao tác minh họa ảnh hưởng của threshold; nó không biến S3 thành bản B4 đã lưu trong Notebook nếu mô hình S3 và B4 có khác biệt nhỏ do fit lại.

### Thực hành 7: Tránh nhầm `predict()` và `predict_spam()`

```python
# Sai về ý nghĩa S3: model.predict(...) mặc định ngưỡng SVM 0.
# Right: gọi predict_spam(...) hoặc predict_record(...) để dùng threshold từ bundle.
```

---

## 19. Bài tập kiểm tra hiểu sâu

**Bài 1.** `bundle['model']` chỉ có LinearSVC hay còn có vectorizer?

**Đáp án:** Có cả `Pipeline([('vec', TfidfVectorizer đã fit), ('clf', LinearSVC đã fit)])`.

**Bài 2.** Tại sao `predict_spam('Xin lịch thi', bundle)` không biến thành ba lần dự đoán?

**Đáp án:** Có `isinstance(texts,str)` và bọc chuỗi trong list, nên một chuỗi ứng với một bình luận.

**Bài 3.** Khi dự đoán, file có biết nhãn thật không?

**Đáp án:** Không. Nó chỉ có raw text + bundle. Nhãn `du_doan` là mô hình tự suy ra; nhãn thật chỉ có khi được con người/nguồn độc lập gán.

**Bài 4.** Nếu quên lưu threshold của S3 thì sao?

**Đáp án:** `predict_record` sẽ lấy default ngưỡng SVM `0`, có thể làm kết quả khác S3 đã được chốt.

**Bài 5.** Vì sao output `score_type='decision_score'`, không phải `'probability_spam'`?

**Đáp án:** Vì model S3 là `LinearSVC`, hỗ trợ `decision_function` nhưng không có `predict_proba`.

**Bài 6.** Nếu tập Test và text gốc trả cùng nhãn, có chứng minh score bằng nhau?

**Đáp án:** Không. Chúng có thể khác score nhưng cùng phía ngưỡng.

**Bài 7.** Nếu người dùng nhập câu chưa có trong dữ liệu, vectorizer phải làm gì?

**Đáp án:** Áp `transform` dựa trên vocabulary/IDF đã học, không thêm cột mới. Các n-gram lạ bị bỏ qua.

**Bài 8.** Có thể nói lưu `preprocessing_source` là đủ để đảm bảo sau này tiền xử lý không đổi?

**Đáp án:** Không. Chuỗi source được lưu để đối chiếu, còn `predict_spam` import module hiện hành.

**Bài 9.** `np.array_equal()` trong cell 32 nhằm gì?

**Đáp án:** Kiểm tra **từng nhãn dự đoán** sau nạp lại và từ text gốc khớp với kết quả trước khi lưu; không phải đánh giá lại để chọn mô hình.

**Bài 10.** Ngày mai model bị dùng với một nhánh khác tiền xử lý, có vấn đề gì?

**Đáp án:** Có thể thay đổi chuỗi token, vector số, score và nhãn. Cần dùng `remove_stop` và cùng `preprocess` như train.

---

## 20. Câu hỏi giảng viên có thể hỏi và câu trả lời mẫu

**Q1. Vì sao phải lưu model?** Để phục vụ dự đoán mới mà không huấn luyện lại toàn bộ GridSearchCV.

**Q2. File `predict.py` có train không?** Không; `fit()` đã thực hiện trước đó, `predict.py` nhận model đã fit từ bundle.

**Q3. Vì sao dùng `joblib`, không chỉ lưu `C`?** Vì cần cả vocabulary, IDF, trọng số/bias SVM, ngưỡng và các cấu hình khác. Chỉ lưu `C` không tái tạo được model đã học.

**Q4. Ai tạo file `.joblib`?** Cell 32 Notebook (hoặc script huấn luyện riêng được README nhắc tới), không phải hàm `predict_spam()`.

**Q5. `model_bundle` gồm những gì?** `model`, `threshold`, `remove_stop`, `manifest`, `preprocessing_source` trong Notebook hiện tại.

**Q6. Tại sao phải lưu `remove_stop`?** Vì Train đã so hai cách xử lý stopwords và chọn một; câu mới phải cùng cấu hình.

**Q7. `Pipeline` giúp gì?** Đảm bảo vector hóa bằng `vec.transform` trước rồi classifier xử lý, giữ cùng vocabulary và IDF.

**Q8. Khi nào mô hình học vocabulary và IDF?** Trong giai đoạn `fit` trên Train; khi predict chỉ dùng `transform`.

**Q9. Dự đoán mới bằng SVM dựa vào gì?** Score từ `X @ coef_.T + intercept_`, sau đó so với threshold đã lưu.

**Q10. Threshold S3 là gì?** Mốc decision score tìm từ kết quả OOF/CV trên Train nhằm cải thiện F1 Spam; không phải xác suất hay tham số C.

**Q11. Vì sao không dùng `LinearSVC.predict` trực tiếp?** Vì `predict` mặc định ngưỡng 0, trong khi S3 dùng ngưỡng đã điều chỉnh.

**Q12. Khác biệt score NB và SVM?** NB trả xác suất ước lượng của lớp; SVM trả điểm quyết định có dấu, không phải xác suất.

**Q13. Demo nhận nhầm tên người thành Spam nói lên điều gì?** Cho thấy mô hình có thể đưa ra dự đoán thiếu hợp lý; cần Test metric và phân tích lỗi, không nên tuyên bố demo luôn đúng.

**Q14. Nếu cần tính Accuracy cho bình luận mới thì sao?** Phải có nhãn thật từ nguồn độc lập; chỉ có dự đoán thì chưa tính Accuracy được.

**Q15. Làm sao biết model nạp lại không bị đổi kết quả?** So nhãn từng dòng trên cùng Test bằng `np.array_equal`, như cell 32.

**Q16. Sao test bằng câu thô lại khác đường Test chuẩn hóa?** Hai đầu vào có thể được làm sạch theo cách hơi khác; Notebook xác nhận nhãn trùng nhưng không bảo đảm score trùng hoàn toàn.

**Q17. Lưu code preprocessing vào bundle có tác dụng gì?** Dùng kiểm tra, đối chiếu bản code tại thời điểm train; không thay thế hàm đang import.

**Q18. `joblib.load` có nguy hiểm không?** Có thể thực thi mã độc khi nạp file pickle không đáng tin; chỉ mở artifact tin cậy.

**Q19. Có nhất thiết phải có dataset CSV khi dự đoán bình luận mới?** Không; cần bundle và hàm tiền xử lý trong môi trường tương thích, không cần đọc lại dữ liệu train.

**Q20. Tại sao chưa gọi đây là AI hiểu được ngữ cảnh?** BoW/TF-IDF + LinearSVC dựa trên đặc trưng từ/ngram và trọng số đã học, không đảm bảo hiểu ngữ nghĩa hay ngữ cảnh bài đăng như con người.

**Q21. Có phải `.joblib` tự chứa `preprocess_for_model.py` để chạy độc lập?** Không; có chuỗi `preprocessing_source` để tham khảo nhưng `predict.py` cần import module tiền xử lý ở runtime.

**Q22. Vì sao không chọn lại model bằng Test khi thấy S2 và S3 gần nhau?** Test dùng để báo cáo khả năng tổng quát hóa, không dùng để quyết định thêm; dùng Test để chọn sẽ tạo thiên lệch lựa chọn.

---

## 21. Bản giải thích trong 90 giây để bảo vệ

> Sau khi so sánh các mô hình bằng Cross-Validation, nhóm chọn S3 là TF-IDF kết hợp LinearSVC và ngưỡng quyết định được điều chỉnh. Mô hình được đánh giá trên tập Test độc lập ở phần `evaluation.py`.
>
> Tại cell 32 của Notebook, em tạo `bundle` chứa Pipeline đã huấn luyện, threshold, cấu hình stopwords và metadata lần chạy. Em dùng `joblib.dump()` để lưu bundle vào file `.joblib`, rồi `joblib.load()` nạp lại. Notebook kiểm tra các nhãn dự đoán sau nạp lại khớp với trước khi lưu trên toàn bộ 7.474 mẫu Test; đồng thời khi dự đoán từ text gốc, nhãn cũng khớp.
>
> Khi người dùng nhập bình luận mới, hàm `predict_spam()` gọi cùng bước tiền xử lý như lúc train để tạo chuỗi token. Pipeline tự gọi `TfidfVectorizer.transform()` với vocabulary và IDF đã học, rồi `LinearSVC.decision_function()` tính score. Em so score với threshold của S3 để trả nhãn spam hoặc ham trong DataFrame. Bước này **không fit lại mô hình** và **không cần nhãn thật**.
>
> Kết quả demo cho thấy các ví dụ đơn giản chạy được, nhưng những câu như tên người hoặc dấu câu vẫn có thể bị dự đoán Spam, nên em không khẳng định demo luôn đúng. Chất lượng mô hình phải đánh giá bằng Precision, Recall, F1 trên Test.

---

## 22. Checklist kiến thức và tài liệu nguồn

Học xong, bạn cần tự trả lời được:

- [ ] `predict.py` làm gì, cell 32 làm gì, khác `evaluation.py` ở đâu?
- [ ] Vì sao `bundle` phải lưu `model`, `threshold`, `remove_stop`?
- [ ] Vì sao text mới cần `preprocess` trước khi đưa vào Pipeline?
- [ ] Tại sao `X` trong `predict_spam` vẫn là chuỗi token và `vec.transform` mới tạo số?
- [ ] `TfidfVectorizer` và `LinearSVC` giữ thông tin gì sau Train?
- [ ] Có gì khác nhau giữa `decision_function`, `predict`, `predict_proba`?
- [ ] Ngưỡng S3 khác ngưỡng SVM mặc định ra sao?
- [ ] `joblib.dump/load` làm gì, các rủi ro nào cần lưu ý?
- [ ] Notebook kiểm tra 7.474 mẫu Test như thế nào và giới hạn của phép kiểm tra?
- [ ] Đọc được output demo, nhận diện kết quả có vẻ bất hợp lý?

### Các file nên mở cùng tài liệu

1. **`pipeline/modeling/predict.py` (mới nhất)** — định nghĩa duy nhất `predict_spam`, các dòng 15–37.
2. **`pipeline/modeling/evaluation.py`** — `predict_record` dòng 11–30, `evaluate` dòng 33–52.
3. **`pipeline/prepare/preprocess_for_model.py`** — `preprocess` dòng 93–104.
4. **`pipeline/modeling/text_features.py`** — `make_vectorizer`, cấu hình TF-IDF và n-gram.
5. **`pipeline/modeling/classifiers.py`** — `LinearSVC`/`MultinomialNB` chưa fit và lưới tham số.
6. **`pipeline/modeling/experiments.py`** — ghép Pipeline và `GridSearchCV.fit` để tạo fitted model.
7. **`02_thuc_nghiem_va_demo(1).ipynb`** — cell **4** import, **9** tạo token, **13** fit cơ sở, **16** chọn S3, **20** test, **31–32** lưu/demo, **34** nhận xét.
8. **`PHI_HIEU_4_LOP_SCIKIT_LEARN.md`** — đường gọi nguồn `TfidfVectorizer`, `LinearSVC`/`MultinomialNB`, cách kế thừa các phương thức `transform`, `decision_function`, `predict_proba`.

### Điều nên ghi nhớ cuối cùng

**`predict.py` không phải một thuật toán ML mới. Nó là lớp ứng dụng (inference) giúp sử dụng lại mô hình đã học, duy trì đúng xử lý đầu vào và ngưỡng để trả kết quả Spam/Ham cho từng bình luận mới.**
