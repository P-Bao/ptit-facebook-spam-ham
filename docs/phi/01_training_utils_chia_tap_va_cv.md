# BƯỚC 01 — `training_utils.py`: CHIA TRAIN/TEST VÀ 5-FOLD CROSS-VALIDATION

> **Tài liệu học và bảo vệ bài tập lớn — phân loại bình luận sinh viên PTIT thành Ham (0) / Spam (1).**  
> **Phiên bản làm căn cứ:** `training_utils(1).py` và `02_thuc_nghiem_va_demo(1).ipynb` (Notebook 02 cập nhật), cùng các file `text_features(1).py`, `experiments(1).py`, `imbalance(1).py`, `evaluation(1).py`, `README(1).md`, `clean_for_labeling.py`, `preprocess_for_model.py` để kiểm tra kết nối.  
> **Phạm vi:** phần của người xây dựng mô hình **bắt đầu từ dataset đã có nhãn**. Không đi sâu vào cách cào Facebook, gán nhãn, hoặc viết tiền xử lý của thành viên khác.  
> **Quan trọng:** ví dụ nhỏ tự dựng được ghi rõ là *minh họa*. Số liệu Train/Test/fold được lấy từ **output thật của cell 11** trong Notebook.

---

## 1. Đích đến: Học xong file này phải trả lời được gì?

Bạn phải giải thích được:

1. Vì sao dự án cần **Train, Validation và Test**; tại sao không fit tất cả rồi tự chấm trên chính những câu đã học.
2. Vì sao đã khử trùng ở bước thu thập mà vẫn cần `norm_key()`/`bucket()` khi chia dữ liệu.
3. Cách xác định nhóm bằng **Unicode NFC, regular expression (regex), hash MD5, phép `% 5`**; đây là *chia nhóm ổn định*, không phải mã hóa bảo mật và không phải loại bản ghi.
4. Bốn hàm `norm_key()`, `bucket()`, `split_train_test()`, `grouped_folds()` **nhận gì → làm gì → trả gì**.
5. Tại sao `split_train_test()` chỉ trả **chỉ số dòng**, còn `grouped_folds()` trả **năm cặp chỉ số tính tương đối bên trong Train**.
6. Cell 11 gọi những hàm nào; `keys`, `train_idx`, `test_idx`, `y_train`, `y_test`, `CV_FOLDS` thực sự là gì.
7. Các biến trên được cell 13, 16, 20 và 32 sử dụng ra sao; vì sao phải dùng cùng tập chia khi đối chiếu mô hình truyền thống với viBERT.
8. Cách đọc bảng số liệu thật trong Notebook và giới hạn của cách chia hiện tại: **không bảo đảm chính xác 80/20, không stratify nhãn, không phát hiện mọi câu giống nghĩa**.

### Bản đồ dữ liệu chỉ riêng bước này

```text
Notebook 02 (cell 6): CSV có nhãn
  raw["cleaned_text"]             y_all = (label == "spam").astype(int)
            |                              |
            +------------------------------+
                           |
                 split_train_test(...)
                      /      |       \
                    keys  train_idx  test_idx
                     |        |         |
                     |        |         +----> y_test = y_all[test_idx]
                     |        +--------------> y_train = y_all[train_idx]
                     |
          keys[train_idx] + y_train
                     |
             grouped_folds(...)
                     |
                   CV_FOLDS
              (5 cặp train/valid)
                     |
       cell 13: fit_experiment(..., CV_FOLDS)
       cell 16: chiến lược S1/S2/S3(..., CV_FOLDS)

  text của Train: TEXT[pre].iloc[train_idx].to_numpy()
  text của Test:  TEXT[pre].iloc[test_idx].to_numpy()
```

`training_utils.py` **không** tạo token, BoW, TF-IDF, Naive Bayes, SVM và cũng **không gọi `fit()`**. Nó chỉ quyết định *dòng nào thuộc phần nào*.

---

## 2. Kiến thức lý thuyết cần biết TRƯỚC khi đọc code

### 2.1. Dataset, sample, feature và label

- **Dataset**: toàn bộ bình luận đã được làm sạch và gán nhãn.
- **Sample**: một dòng/một bình luận.
- **Text đầu vào**: bình luận (ở bước chia, dùng `cleaned_text`; khi fit, dùng chuỗi token đã chuẩn bị).
- **Label/target `y`**: đáp án thật của người gán nhãn: `ham=0`, `spam=1`.
- **Feature**: đặc trưng bằng số (BoW/TF-IDF) sẽ được tạo **sau bước chia**, trong Pipeline của các bước tiếp theo.

Ví dụ:

| Dòng CSV | `cleaned_text` | `label` | `y_all` |
|---:|---|---|---:|
| 0 | `Xin lịch thi môn Java` | ham | 0 |
| 1 | `Nhận làm đồ án giá rẻ` | spam | 1 |
| 2 | `Ai có đề cương không` | ham | 0 |

**Vì sao phải có `y_all` từ sớm?** Chia tập và kiểm tra tập có đủ hai lớp cần đọc nhãn; sau đó `clf.fit(X_train_vector, y_train)` cũng cần chính nhãn này để học có giám sát.

### 2.2. Train / Validation / Test khác nhau thế nào?

| Phần | Dùng làm gì? | Có được dùng chọn siêu tham số/ngưỡng không? |
|---|---|---|
| **Train** | Học vocabulary/IDF và học trọng số/xác suất NB/SVM; là nguồn dữ liệu cho CV | Có, qua các lượt CV nội bộ |
| **Validation** (mỗi lượt CV) | Dùng để kiểm tra một cấu hình, tính Precision/Recall/F1, chọn alpha/C/n-gram/ngưỡng | Có |
| **Test độc lập** | Đánh giá cuối cùng sau khi chốt mô hình, kiểm tra khả năng dự đoán dữ liệu chưa học | **Không** |

Cần tách rõ hai khái niệm **"test"**:

- `mean_test_f1` trong **`GridSearchCV.cv_results_`** là cách scikit-learn đặt tên điểm trên **fold validation**; **không phải** tập Test độc lập của dự án.
- `record["F1"]` sau `evaluation.evaluate(record, X_test, y_test)` mới là **F1 trên Test cuối cùng**.

**Data leakage (rò rỉ dữ liệu)**: để mô hình vô tình thấy thông tin của những câu sẽ dùng để kiểm tra. Hai nguy cơ chính:

1. Bình luận trùng/gần trùng xuất hiện ở cả Train và Test → mô hình được kiểm tra trên nội dung rất giống nội dung đã học.
2. Fit vocabulary/IDF trên toàn bộ CSV **trước** khi chia Train/Test hoặc trước một lượt CV → vectorizer đã nhìn thấy thống kê từ tập validation/Test.

Thiết kế project ngăn ngừa hai nguy cơ này theo hai cách khác nhau: `training_utils` **nhóm nội dung** trước khi chia; `Pipeline` fit vectorizer **riêng trên train của từng fold**.

### 2.3. 5-fold Cross-Validation là gì?

Chỉ với tập **Train**, gán mỗi bình luận vào một trong **5 nhóm (fold)**. Ta có 5 lần chạy:

| Lượt | Dữ liệu dùng để `fit()` | Dữ liệu dùng để chấm |
|---|---|---|
| 1 | Fold 2 + 3 + 4 + 5 | Fold 1 |
| 2 | Fold 1 + 3 + 4 + 5 | Fold 2 |
| 3 | Fold 1 + 2 + 4 + 5 | Fold 3 |
| 4 | Fold 1 + 2 + 3 + 5 | Fold 4 |
| 5 | Fold 1 + 2 + 3 + 4 | Fold 5 |

Với mỗi cấu hình (ví dụ TF-IDF + SVM, C=0.3, unigram+bigram), chấm F1 trên từng validation, lấy **trung bình 5 điểm**. Có thể so nhiều cấu hình, chọn cấu hình CV F1 tốt nhất. Sau đó huấn luyện lại cấu hình được chọn trên **toàn bộ Train** và mới đánh giá trên Test độc lập.

**Group CV**: không chia tùy tiện theo dòng; những câu có **cùng khóa nội dung** thuộc cùng fold. Tránh một phiên bản câu vào phần fit và phiên bản giống nó vào validation.

**Out-of-fold (OOF)**: điểm/nhãn dự đoán cho mỗi câu Train được tính từ mô hình mà **không fit trên chính câu ấy**. Trong dự án, S3 sử dụng decision score OOF để chọn ngưỡng SVM; chưa cần đi vào cách tìm ngưỡng ở bước này.

### 2.4. Grouping khác stratification

- **Grouping**: giữ cùng một `norm_key` ở cùng một bên của phép chia.
- **Stratification**: cố gắng giữ tỷ lệ Spam/Ham gần nhau giữa các tập.

Trong file của bạn, code dùng **grouping bằng hash**, **không dùng stratified split**. Vì vậy nhóm được bảo vệ tốt hơn khỏi trùng nội dung nhưng **không đảm bảo mỗi tập có tỷ lệ Spam giống toàn bộ dữ liệu**. Code chỉ kiểm tra điều kiện tối thiểu: mỗi phía phải có cả nhãn `0` và `1`.

### 2.5. Hash, MD5, salt và modulo (`%`)

- **Hash**: chuyển một chuỗi thành một giá trị số dài theo quy tắc xác định.
- **MD5**: hàm hash dùng ở đây để có cách chia ổn định; **không dùng nhằm bảo mật**.
- **`salt`**: tên mục đích chia; code dùng **`"test"` hoặc `"cv"`**. Hai giá trị này khiến phép gán Train/Test khác phép gán fold.
- **`% 5`**: chia lấy dư; đầu ra nằm trong `{0, 1, 2, 3, 4}`.

Không phải "cứ 5 dòng lấy một dòng vào Test". `bucket()` được tính từ **nội dung đã chuẩn hóa**. Hai bình luận cùng khóa sẽ ra cùng bucket.

### 2.6. Chỉ số dòng (index) khác vector số

`train_idx = [3, 6, 9, ...]` là **vị trí dòng trong dataset**, không phải vector BoW/TF-IDF. Notebook dùng index để trích văn bản sau khi chia:

```python
X_train_text = TEXT[pre].iloc[train_idx].to_numpy()
```

Kết quả `X_train_text` **vẫn là mảng chuỗi token**. Đến `experiments.py` gọi `search.fit(X_train_text, y_train)` thì Pipeline mới tạo BoW/TF-IDF và huấn luyện bộ phân loại.

---

## 3. Notebook 02 chuẩn bị đầu vào và gọi file này thế nào?

Bạn không phụ trách viết tiền xử lý, nhưng phải biết biến nào được chuyển vào hàm mình viết.

### Cell 3 — Thiết lập đường dẫn, seed

```python
SEED = 42
np.random.seed(SEED)
CSV_PATH = Path(os.environ.get(
    "SPAMHAM_CSV", str(PROJECT_ROOT / "data/final/labeled_v7.csv")
))
```

- `CSV_PATH`: file CSV đã có nhãn.
- `SEED`: seed dùng trong nhiều thí nghiệm. **Lưu ý:** chia dữ liệu bằng `bucket()` dùng hash của nội dung, **không dựa vào `np.random.seed(42)`**.

### Cell 4 — Import hàm của bạn

```python
from pipeline.modeling.training_utils import split_train_test, grouped_folds
from pipeline.modeling.experiments import make_pipe, fit_experiment
from pipeline.modeling.imbalance import fit_balanced_svm, undersampled_search, tune_svm_threshold
from pipeline.modeling.evaluation import evaluate, boot_ci, predict_record
from pipeline.modeling.predict import predict_spam
```

`import` nạp định nghĩa hàm vào Notebook; **chưa tự chạy chia tập hoặc train**.

### Cell 6 — Đọc CSV và tạo `y_all`

```python
raw = pd.read_csv(CSV_PATH, keep_default_na=False, dtype={"id": str})
raw["label"] = raw["label"].str.strip().str.lower()
assert raw.id.is_unique and raw.id.str.strip().ne("").all()
assert set(raw.label) == {"ham", "spam"}
assert raw.cleaned_text.str.strip().ne("").all()
y_all = raw.label.eq("spam").astype(int).to_numpy()
```

- `raw`: bảng pandas gồm các dòng CSV.
- `raw["cleaned_text"]`: câu đã làm sạch, dùng tạo khóa chia tập.
- `raw.label.eq("spam")`: với mỗi dòng, trả `True` nếu Spam, `False` nếu Ham.
- `.astype(int)`: biến `True→1`, `False→0`.
- `.to_numpy()`: chuyển thành mảng số NumPy để dễ cắt theo index.
- **`y_all[i]` là nhãn thật của `raw.iloc[i]`.** Thứ tự dòng phải khớp.

Cell 6 cũng kiểm tra trùng ID và thống kê nhóm văn bản lặp/nhãn mâu thuẫn; **thống kê không tự xóa dòng**.

### Cell 9 — Hai nhánh token từ bước tiền xử lý

```python
prepared["tokens_keep_stopwords"] = [
    preprocess(c, t, False)
    for c, t in zip(raw.cleaned_text, raw.text)
]
prepared["tokens_remove_stopwords"] = ...
TEXT = {
    "giữ stop words": prepared.tokens_keep_stopwords,
    "có loại stop words": prepared.tokens_remove_stopwords
}
```

Đây là phần thành viên khác chuẩn bị. Cần nhớ rằng `TEXT` lưu **hai bản chuỗi token của cùng những bình luận**, giữ đúng thứ tự ID. Notebook lưu/đọc `data/processed/v7_preprocessed.csv` rồi kiểm tra ID và nhãn không bị đổi thứ tự. Cell 13 sẽ dùng cùng index chia tập cho cả hai nhánh.

### Cell 10 — Thiết kế thí nghiệm (Markdown)

Ghi rõ: nhóm theo nội dung, tỷ lệ xấp xỉ 80/20, CV trên Train, Test chỉ dùng đánh giá cuối; từ vựng/IDF được học bên trong fold train; không dùng Test để chọn tham số/ngưỡng.

### **Cell 11 — Điểm bắt đầu chính của `training_utils.py`**

Đây là code gốc, được giữ đầy đủ:

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

**Giải thích từng nhóm lệnh:**

1. `split_train_test(raw["cleaned_text"], y_all)`: từ toàn CSV, tạo khóa và trả **3 biến** gồm khóa và index Train/Test.
2. `y_all[train_idx]`, `y_all[test_idx]`: lấy nhãn thật theo đúng index vừa chia.
3. `grouped_folds(keys[train_idx], y_train)`: tạo **5 cặp index Train/Validation bên trong Train**.
4. `split_table`: `len(labels)` = số mẫu; `labels.sum()` = số Spam vì nhãn Spam=1; `labels.mean()*100` = tỷ lệ Spam; `.set_index("Tập")` thiết lập nhãn hàng; `.round(2)` làm tròn khi hiển thị.
5. `fold_table`: `enumerate(CV_FOLDS)` lặp từng cặp; `i+1` đánh số lượt từ 1; `len(tr)`/`len(va)` đếm mẫu fit/validation; `y_train[va].sum()` đếm Spam thực trong validation.
6. **Chưa có `fit()` ở cell 11.** Việc học sẽ diễn ra khi cell 13 gọi `fit_experiment()`.

---
## 4. Hàm `norm_key(text)` — Tạo khóa gần trùng

### 4.1. Tại sao cần?

`clean_for_labeling.py` đã khử trùng bản ghi trước gán nhãn theo bộ ba **`(post_id, comment_type, text)`**. Tuy nhiên, vẫn có thể có nội dung giống nhau đăng ở bài khác, hoặc một mẫu quảng cáo chỉ đổi giá, dấu câu, chữ hoa/thường. Nếu một câu ở Train và câu tương tự ở Test, điểm đánh giá có thể lạc quan.

**`norm_key()` không xóa bình luận.** Nó tạo một khóa phụ để các câu gần giống theo quy tắc ký tự rơi cùng một phía chia tập.

### 4.2. Code gốc

```python
def norm_key(text):
    normalized = unicodedata.normalize("NFC", str(text)).lower()
    normalized = re.sub(r"\s+", " ", normalized).strip()
    key = re.sub(r"\d+", "0", normalized)
    key = re.sub(r"[^\w\s]", " ", key)
    key = re.sub(r"\s+", " ", key).strip()
    return key or "__symbols__|" + normalized
```

### 4.3. Giải thích từng câu lệnh

**Đầu vào:** một bình luận, ví dụ `"NHẬN   làm đồ án 100k!!!"`.

1. `str(text)`: ép đầu vào thành chuỗi.
2. `unicodedata.normalize("NFC", ...)`: chuẩn hóa biểu diễn ký tự Unicode tiếng Việt; **không xóa dấu tiếng Việt**.
3. `.lower()`: biến `"NHẬN"` thành `"nhận"`.
4. `re.sub(r"\s+", " ", ...).strip()`: mọi chuỗi khoảng trắng liên tiếp được thay bằng một dấu cách, bỏ trắng đầu/cuối.
5. `re.sub(r"\d+", "0", normalized)`: biến **một hay nhiều chữ số liên tiếp** thành `0`, nên `"100k"` và `"200k"` đều thành `"0k"`.
6. `re.sub(r"[^\w\s]", " ", key)`: thay ký tự không phải ký tự chữ/số/gạch dưới hoặc khoảng trắng thành dấu cách. `\w` hỗ trợ chữ Unicode; dấu câu thường bị thay.
7. Gộp khoảng trắng một lần nữa vì dấu câu vừa bị thay có thể tạo khoảng cách thừa.
8. `return key or "__symbols__|" + normalized`: nếu kết quả `key` không rỗng thì trả nó; nếu kết quả rỗng do chỉ có dấu câu/ký hiệu thì tạo khóa dự phòng để tránh mọi câu kiểu đó cùng nhận khóa rỗng.

**Đầu ra minh họa:**

| Đầu vào | Khóa trả về |
|---|---|
| `NHẬN làm đồ án 100k!!!` | `nhận làm đồ án 0k` |
| `Nhận làm đồ án 200k` | `nhận làm đồ án 0k` |
| `Xin  lịch thi!!!` | `xin lịch thi` |
| `!!!` | `__symbols__|!!!` |

Hai câu quảng cáo đầu nhận **cùng khóa**. Vì vậy mọi phép chia `bucket()` với cùng salt sẽ đưa chúng vào cùng nhóm.

### 4.4. Các điểm nên nhớ

- Khóa này chỉ giúp **chia tập**; **không thay đổi** câu để vector hóa/huấn luyện.
- Không phải phát hiện gần trùng theo ngữ nghĩa: `"xin lịch thi"` và `"cho mình hỏi thời gian thi"` vẫn khác khóa.
- Có thể **gộp quá mức** những nội dung khác nhau chỉ vì thay mọi con số về `0`; đây là đánh đổi của quy tắc hiện tại.
- Nếu cùng khóa nhưng hai nhãn trái ngược, hàm không giải quyết mâu thuẫn; cần ghi nhận giới hạn chất lượng nhãn.

---

## 5. Hàm `bucket(key, salt, n=5)` — Gán nhóm bằng hash

### 5.1. Code gốc

```python
def bucket(key, salt, n=5):
    digest = hashlib.md5(f"{salt}|{key}".encode("utf-8")).hexdigest()
    return int(digest, 16) % n
```

**Đầu vào:** `key` (khóa bình luận), `salt` (`"test"` hoặc `"cv"`), `n` (số nhóm, mặc định 5).

**Các bước thực thi:**

1. `f"{salt}|{key}"` nối hai chuỗi, ví dụ `"test|nhận làm đồ án 0k"`.
2. `.encode("utf-8")` chuyển chuỗi thành bytes, vì `hashlib.md5` nhận bytes.
3. `.hexdigest()` tạo chuỗi hash dưới dạng hex (hệ 16).
4. `int(digest, 16)` chuyển chuỗi hex thành số nguyên.
5. `% n` chia lấy dư, đưa số rất lớn về **một số nhóm từ 0 đến n-1**.

**Đầu ra:** một số nguyên `0,1,2,3,4` khi `n=5`.

### 5.2. Ví dụ kiểm tra đúng theo hàm MD5 của code

| Bình luận | `norm_key` | `bucket(key,"test")` | `bucket(key,"cv")` |
|---|---|---:|---:|
| `NHẬN làm đồ án 100k!!!` | `nhận làm đồ án 0k` | **2** | **4** |
| `Nhận làm đồ án 200k` | `nhận làm đồ án 0k` | **2** | **4** |
| `Xin lịch thi` | `xin lịch thi` | **1** | **4** |
| `Hỏi lịch thi cuối kỳ` | `hỏi lịch thi cuối kỳ` | **0** | **4** |

Ví dụ `"Hỏi lịch thi cuối kỳ"` được gán `test` bucket 0 nên sẽ thuộc **Test**, không đưa vào CV mặc dù phép hash cho mục đích `"cv"` về mặt toán học là 4.

Hai câu `100k/200k` cùng `test` bucket 2 nên cả hai thuộc Train; chúng cũng cùng `cv` bucket 4 nên nếu chạy `grouped_folds` sẽ là validation ở lượt 5.

### 5.3. Hai cách dùng trong project

```python
bucket(key, "test")  # 0: Test; 1,2,3,4: Train
bucket(key, "cv", 5) # 0..4: năm nhóm validation trong Train
```

`"test"` và `"cv"` là hai salt khác nhau nên một bình luận có thể mang bucket 2 khi chia Train/Test nhưng bucket 4 khi chia CV. Hai phép chia phục vụ hai mục đích riêng.

**Không phải cứ 5 dòng có 1 dòng Test.** Hash được tính theo **khóa nội dung**; các nhóm câu cùng khóa đi chung một phía, nên số lượng dòng và tỷ lệ Spam có thể lệch.

**Không phải một bucket chỉ có một bình luận.** Nhiều khóa khác nhau có thể cùng bucket (đó chính là mục đích của `%5`). Điều cần đảm bảo là **cùng khóa không thể bị chia sang hai bucket trong cùng mục đích**.

### 5.4. Tại sao hữu ích khi so với viBERT?

Nếu mô hình truyền thống và viBERT sử dụng **cùng CSV, cùng khóa và cùng hàm chia**, cả hai sẽ có cùng tập Train/Test. Điều đó giúp so sánh kết quả công bằng vì chúng được kiểm tra trên **cùng bình luận và nhãn thật**.

Cần nói chính xác: đây là **mục đích thiết kế**; muốn khẳng định Notebook 03 đã làm đúng phải kiểm tra code Notebook 03. Cùng seed 42 hoặc cùng tỷ lệ 80/20 **không tự bảo đảm cùng tập**.

---
## 6. Hàm `split_train_test(cleaned_texts, labels)` — Chia toàn bộ CSV

### 6.1. Đầu vào/đầu ra

Notebook gọi:

```python
keys, train_idx, test_idx = split_train_test(raw["cleaned_text"], y_all)
```

**Nhận:**

- `cleaned_texts`: toàn bộ bình luận đã làm sạch, đúng thứ tự CSV, **không phải BoW/TF-IDF**.
- `labels`: nhãn 0/1, cùng thứ tự dòng với `cleaned_texts`.

**Trả một tuple có đúng ba phần:**

- `keys`: NumPy array khóa chuẩn hóa cho **tất cả bình luận**.
- `train_idx`: NumPy array chứa vị trí dòng Train **tính trên toàn CSV**.
- `test_idx`: NumPy array chứa vị trí dòng Test **tính trên toàn CSV**.

Hàm **không** trả trực tiếp `X_train`, `X_test`, `y_train`, `y_test`; Notebook sử dụng index để lấy các biến sau đó.

### 6.2. Code gốc

```python
def split_train_test(cleaned_texts, labels):
    keys = np.array([norm_key(text) for text in cleaned_texts])
    labels = list(labels)
    if len(keys) != len(labels):
        raise ValueError("Văn bản và nhãn phải có cùng số dòng")

    train_indices = np.array([i for i, key in enumerate(keys)
                              if bucket(key, "test") != 0], dtype=int)
    test_indices = np.array([i for i, key in enumerate(keys)
                             if bucket(key, "test") == 0], dtype=int)
    train_labels = {labels[i] for i in train_indices}
    test_labels = {labels[i] for i in test_indices}
    if train_labels != {0, 1} or test_labels != {0, 1}:
        raise ValueError("Train/test phải có cả ham=0 và spam=1")
    assert not set(keys[train_indices]) & set(keys[test_indices])
    return keys, train_indices, test_indices
```

### 6.3. Giải thích từng nhóm dòng

**`keys = np.array([...])`**: list comprehension gọi `norm_key` trên từng câu, `np.array` tạo mảng khóa để có thể trích theo index.

**`labels = list(labels)`**: chuyển nhãn sang list; `labels` chỉ phục vụ kiểm tra hợp lệ ở bước này, **không quyết định câu nào được chia vào đâu**.

**`if len(keys) != len(labels)`**: kiểm tra số lượng câu và nhãn bằng nhau; nếu lệch, `raise ValueError` vì không thể ánh xạ đúng từng nhãn.

**Hai list comprehension tạo `train_indices` và `test_indices`:** `enumerate(keys)` sinh `(i, key)` với `i` là chỉ số dòng; `bucket(key,"test") != 0` → Train, `== 0` → Test. `dtype=int` bảo đảm mảng chứa số nguyên dùng để index.

**`train_labels = {...}`** và **`test_labels = {...}`**: set comprehension lấy những nhãn **khác nhau** xuất hiện ở từng phía. Ví dụ `[0,0,1,1]` tạo `{0,1}`. Nếu Test chỉ có nhãn 0 thì `{0}` → `raise ValueError`.

**`assert not set(keys[train_indices]) & set(keys[test_indices])`**: `&` là **phép giao set**, đòi hỏi không có một khóa nào xuất hiện ở cả Train/Test. Nếu có, `assert` thất bại.

**`return keys, train_indices, test_indices`**: đưa kết quả cho Notebook, các bước sau dùng index để lấy text và nhãn.

### 6.4. Không nhầm lẫn với phép khử trùng

Giả sử hai bình luận giống template chỉ khác giá:

```text
id=10: NHẬN làm đồ án 100k!!!  (Spam)
id=27: Nhận làm đồ án 200k    (Spam)
```

`split_train_test()` **giữ cả hai dòng**, nhưng vì hai `norm_key` giống nhau, chúng **cùng Train hoặc cùng Test**. Nó không xóa một trong hai. Nếu đã xóa trùng ở bước thu thập, đó là cơ chế khác trong `clean_for_labeling.py`.

### 6.5. Số liệu THẬT từ output của Notebook cell 11

| Tập | Số dòng | Spam (1) | Ham (0) | Spam (%) |
|---|---:|---:|---:|---:|
| **Train** | **30.759** | **4.375** | **26.384** | **14,22%** |
| **Test** | **7.474** | **951** | **6.523** | **12,72%** |
| **Toàn CSV** | **38.233** | **5.326** | **32.907** | **13,93%** |

Dữ liệu lệch về Ham; đó là lý do không thể chỉ dùng Accuracy để đánh giá Spam. Nó cũng cho thấy **cách chia không stratify chính xác theo nhãn**: tỷ lệ Spam Train khác Test.

### 6.6. Giới hạn cần nói rõ

- Tỷ lệ Train/Test xấp xỉ 80/20, **không bị ép chính xác**.
- Mọi dòng đều thuộc đúng một phía; **không có thao tác xóa mẫu**.
- Không có bảo đảm hai phía cân bằng tỷ lệ lớp, chỉ kiểm tra cả hai có `0` và `1`.
- Không sửa nhãn mâu thuẫn, không phát hiện mọi câu tương đương về ngữ nghĩa.

---

## 7. Hàm `grouped_folds(keys, y, n_splits=5)` — Tạo 5 lượt CV trong Train

### 7.1. Đầu vào/đầu ra

Notebook gọi:

```python
CV_FOLDS = grouped_folds(keys[train_idx], y_train)
```

**Nhận:**

- `keys[train_idx]`: các khóa **chỉ của Train** (giữ đúng thứ tự đã lấy).
- `y_train`: nhãn của những dòng Train tương ứng.
- `n_splits=5`: số lượt/fold.

**Trả:** list 5 phần tử, mỗi phần tử là tuple `(tr, va)`:

```python
CV_FOLDS = [
    (tr_1, va_1),
    (tr_2, va_2),
    (tr_3, va_3),
    (tr_4, va_4),
    (tr_5, va_5)
]
```

`tr` và `va` là vị trí **bên trong mảng Train**, tức `0..len(y_train)-1` (không phải index gốc toàn CSV).

### 7.2. Code gốc

```python
def grouped_folds(keys, y, n_splits=5):
    keys = list(keys)
    labels = list(y)
    if len(keys) != len(labels):
        raise ValueError("Khóa và nhãn train phải có cùng số dòng")
    fold_ids = [bucket(key, "cv", n_splits) for key in keys]
    folds = []
    for fold_id in range(n_splits):
        train = np.array([i for i, value in enumerate(fold_ids)
                          if value != fold_id], dtype=int)
        valid = np.array([i for i, value in enumerate(fold_ids)
                          if value == fold_id], dtype=int)
        train_labels = {labels[i] for i in train}
        valid_labels = {labels[i] for i in valid}
        if train_labels != {0, 1} or valid_labels != {0, 1}:
            raise ValueError("Fold thiếu một lớp; cần kiểm tra nhóm dữ liệu hoặc số fold.")
        assert not {keys[i] for i in train} & {keys[i] for i in valid}
        folds.append((train, valid))
    return folds
```

### 7.3. Giải thích các lệnh

1. `keys = list(keys); labels = list(y)`: chuẩn hóa kiểu container, không thay nội dung.
2. `if len(keys) != len(labels)`: khóa và nhãn Train phải cùng số dòng.
3. `fold_ids = [bucket(key,"cv", n_splits) for key in keys]`: mỗi bình luận Train được gán một số fold từ 0 tới 4. **Cùng khóa → cùng fold**.
4. `for fold_id in range(n_splits)`: chạy qua 5 lượt, `fold_id=0,1,2,3,4`.
5. `train = np.array([... if value != fold_id])`: những dòng **ngoài** fold đang xét sẽ dùng fit.
6. `valid = np.array([... if value == fold_id])`: những dòng **thuộc** fold đang xét dùng dự đoán và chấm điểm.
7. `train_labels`/`valid_labels`: kiểm tra cả phần fit và validation đều có Ham/Spam.
8. `assert not {keys[i] for i in train} & {keys[i] for i in valid}`: bảo đảm không trùng **khóa** giữa hai phía trong **một lượt**.
9. `folds.append((train, valid))`: lưu cặp chỉ số, không fit gì tại đây.
10. `return folds`: chuyển 5 cặp index sang `GridSearchCV` hoặc các vòng lặp CV ở `imbalance.py`.

### 7.4. Vì sao 5 fold không có số mẫu bằng nhau?

Vì code chia **theo khóa hash**, không ép đúng số lượng mẫu và không stratify nhãn. Một khóa có thể xuất hiện ở nhiều bình luận; nhóm theo khóa không được cắt rời. Tất cả fold là các nhóm rời nhau nhưng kích thước từng nhóm có thể chênh lệch.

### 7.5. Số liệu THẬT của 5 fold ở output cell 11

| Lượt CV | Số mẫu fit | Số mẫu validation | Spam trong validation |
|---:|---:|---:|---:|
| 1 | 24.815 | 5.944 | 764 |
| 2 | 24.327 | 6.432 | 1.187 |
| 3 | 24.554 | 6.205 | 856 |
| 4 | 24.529 | 6.230 | 789 |
| 5 | 24.811 | 5.948 | 779 |

Kiểm tra: tổng số dòng validation qua 5 lượt bằng **30.759**; tổng số Spam validation bằng **4.375**. Đó là do mỗi dòng Train được làm validation đúng một lần. Trong những lượt khác, dòng đó nằm trong phần fit.

### 7.6. Sơ đồ luân phiên

```text
          [ Fold 1 ] [ Fold 2 ] [ Fold 3 ] [ Fold 4 ] [ Fold 5 ]
Lượt 1:     VALID       FIT        FIT        FIT        FIT
Lượt 2:     FIT         VALID      FIT        FIT        FIT
Lượt 3:     FIT         FIT        VALID      FIT        FIT
Lượt 4:     FIT         FIT        FIT        VALID      FIT
Lượt 5:     FIT         FIT        FIT        FIT        VALID

TEST độc lập (7.474 câu): KHÔNG xuất hiện trong bất kỳ lượt CV nào.
```

**Một câu có thể dùng để fit ở bốn lượt CV và làm validation ở một lượt.** Không phải một câu chỉ được sử dụng đúng một lần trong toàn quá trình.

---
## 8. Các biến chuyển sang file tiếp theo như thế nào?

| Nơi sử dụng | Câu lệnh / nội dung quan trọng | Dùng kết quả bước này để làm gì? |
|---|---|---|
| **Cell 13** — 4 tổ hợp cơ sở, 2 nhánh từ dừng | `X_train_text = TEXT[pre].iloc[train_idx].to_numpy()`; `fit_experiment(pre, feature, algorithm, X_train_text, y_train, CV_FOLDS, ...)` | Dùng **đúng các câu Train**, nhãn Train và 5 cặp index Train/Validation để chạy GridSearchCV. |
| **Cell 16** — S1/S2/S3 | `X_tr_text = TEXT[MAIN_PRE].iloc[train_idx].to_numpy()` | Giữ cùng Train và folds cho các chiến lược cân bằng; undersampling chỉ lấy mẫu phần Train của fold. |
| **Cell 20** — đánh giá Test | `evaluate(record, TEXT[record["Tiền xử lý"]].iloc[test_idx].to_numpy(), y_test)` | Dùng đúng **những câu Test** và nhãn Test để tính Accuracy/Precision/Recall/F1/AP. |
| **Cell 32** — kiểm tra mô hình lưu | `X_verify = TEXT[best["Tiền xử lý"]].iloc[test_idx].to_numpy()` | Dự đoán lại cùng Test bằng model nạp từ Joblib, so sánh với kết quả đã có. |

### Hai hệ tọa độ index — rất dễ nhầm

**Index toàn CSV:**

```python
train_idx  # vị trí dòng trong raw và TEXT[pre]
test_idx   # vị trí dòng trong raw và TEXT[pre]
```

**Index cục bộ bên trong Train:**

```python
for tr, va in CV_FOLDS:
    X_fold_train = X_train_text[tr]
    y_fold_train = y_train[tr]
    X_fold_valid = X_train_text[va]
    y_fold_valid = y_train[va]
```

**Lỗi nguy hiểm:** dùng `CV_FOLDS` để cắt trực tiếp `raw.iloc[va]`. `va` là vị trí **trong Train**, còn `raw` là toàn CSV. Nếu cần index gốc, phải lấy `train_idx[va]`; nếu đang làm CV, cách đúng là cắt từ `X_train_text[va]` và `y_train[va]`.

**Vì sao không chia riêng cho BoW và TF-IDF?** Bởi vì cả bốn mô hình và hai nhánh stopwords phải đối chiếu trên cùng nguồn dòng Train/Test; chỉ cách vector hóa và bộ phân loại thay đổi, không đổi tập Test mỗi lần.

---

## 9. Ví dụ code TỰ CHẠY để nhìn thấy kết quả từng hàm

> **Ví dụ học**, không phải dữ liệu thật và không phải cell có sẵn. Hãy chạy trong project có `pipeline/modeling/training_utils.py`.

```python
import string
import numpy as np
from pipeline.modeling.training_utils import (
    norm_key, bucket, split_train_test, grouped_folds
)

# 1. Xem rõ 2 câu quảng cáo cùng khóa
s1 = "NHẬN làm đồ án 100k!!!"
s2 = "Nhận làm đồ án 200k"
print(norm_key(s1))                # nhận làm đồ án 0k
print(norm_key(s2))                # nhận làm đồ án 0k
print(bucket(norm_key(s1), "test")) # 2
print(bucket(norm_key(s2), "test")) # 2

# 2. Tạo bộ dữ liệu học thử gồm 15 ham và 15 spam.
# Dùng hậu tố chữ cái để sau norm_key vẫn tạo nhiều khóa khác nhau.
letters = string.ascii_lowercase[:15]
texts = ([f"xin tài liệu môn {letter}" for letter in letters]
       + [f"nhận làm thuê môn {letter}" for letter in letters])
y_all = np.array([0] * 15 + [1] * 15)

# 3. Chạy đúng 2 hàm của training_utils.py
keys, train_idx, test_idx = split_train_test(texts, y_all)
y_train, y_test = y_all[train_idx], y_all[test_idx]
folds = grouped_folds(keys[train_idx], y_train)

print("Train:", len(train_idx), "Test:", len(test_idx))
print("Nhãn trong Train:", np.unique(y_train))
print("Nhãn trong Test:", np.unique(y_test))
for i, (tr, va) in enumerate(folds, 1):
    print(f"Fold {i}: fit={len(tr)}, valid={len(va)}, "
          f"spam_valid={int(y_train[va].sum())}")

# 4. Kiểm tra invariant: không có khóa chung giữa hai phía
assert not set(keys[train_idx]) & set(keys[test_idx])
for tr, va in folds:
    train_keys = keys[train_idx][tr]
    valid_keys = keys[train_idx][va]
    assert not set(train_keys) & set(valid_keys)
```

**Khi quan sát output cần rút ra:** (1) hai câu `100k/200k` có cùng khóa/bucket; (2) `train_idx/test_idx` là chỉ số của toàn bộ `texts`; (3) mỗi phần tử trong `folds` là cặp chỉ số **tương đối với Train**; (4) tất cả năm lượt phải có hai lớp trong fit/validation.

Không dùng dataset mô phỏng để suy ra chất lượng mô hình của nhóm; kết quả thực nghiệm phải dựa trên dữ liệu thật của Notebook.

---

## 10. Phân biệt với việc của các thành viên khác

### 10.1. "Khử trùng" trước gán nhãn khác "gom nhóm" trước Train

| Tiêu chí | `clean_for_labeling.py` | `training_utils.py` |
|---|---|---|
| Người phụ trách | Thành viên làm dữ liệu | Bạn làm mô hình |
| Thời điểm | Trước gán nhãn | Trước huấn luyện |
| Quy tắc | Trùng chính xác bộ `(post_id, comment_type, text)` | Cùng `norm_key(cleaned_text)` sau chuẩn hóa |
| Hành động | **Bỏ bản ghi trùng** | **Không xóa dòng; giữ cùng khóa cùng phía** |
| Mục đích | Giảm bản ghi thu thập bị lặp | Hạn chế rò rỉ nội dung sang Test/Validation |

Hai việc **bổ sung**, không trùng chức năng.

### 10.2. `preprocess_for_model.py` khác `norm_key()`

- `preprocess_for_model.preprocess(...)` tạo **chuỗi token để vector hóa và dùng lại khi dự đoán câu mới**; do thành viên khác đảm nhiệm.
- `training_utils.norm_key(...)` chỉ tạo **khóa chia tập**, không phải tokenizer và không thay input cho NB/SVM.

### 10.3. Liên quan viBERT

Về nguyên tắc, cách chia ổn định cho phép Notebook 03 dùng **cùng ID và nhãn của Test** như mô hình truyền thống. ViBERT có thể dùng cách tokenize/embedding khác, nhưng so sánh công bằng cần đánh giá trên cùng các bình luận. **Chưa có Notebook 03 để xác nhận cách phía viBERT lấy index thực tế**, nên đây là giải thích mục đích thiết kế, không phải kết luận kiểm toán notebook kia.

---

## 11. Bảng tổng kết 4 hàm và các hàm Python sử dụng

| Hàm | Đầu vào | Chức năng | Đầu ra | Có train model không? |
|---|---|---|---|---|
| `norm_key(text)` | Một câu | Chuẩn hóa thành khóa nội dung | `str` khóa | Không |
| `bucket(key, salt, n=5)` | Khóa, loại chia, số nhóm | Hash → nhóm ổn định | `int` 0..4 | Không |
| `split_train_test(cleaned_texts, labels)` | Toàn câu sạch + nhãn | Tạo khóa, phân index Train/Test, kiểm tra hai lớp | `(keys, train_idx, test_idx)` | Không |
| `grouped_folds(keys, y, n_splits=5)` | Khóa + nhãn **Train** | Gán 5 fold và tạo cặp index train/validation | `[(tr,va), ...]` | Không |

| Thành phần code | Ý nghĩa |
|---|---|
| `re.sub(...)` | Tìm và thay bằng regex |
| `unicodedata.normalize("NFC", ...)` | Chuẩn hóa Unicode |
| `hashlib.md5(...).hexdigest()` | Hash ổn định dạng hex |
| `int(digest, 16)` | Đổi chuỗi hệ 16 thành số nguyên |
| `% n` | Chia lấy dư để gán bucket |
| `np.array(...)` | Mảng NumPy để cắt theo index |
| `enumerate(...)` | Lấy cặp `(index, value)` trong vòng lặp |
| `{... for ...}` | Set comprehension: lấy nhãn phân biệt |
| `&` giữa hai `set` | Phép giao tập hợp |
| `assert` | Kiểm tra bất biến; có thể bị bỏ khi chạy Python với tối ưu `-O` |
| `raise ValueError(...)` | Báo điều kiện đầu vào không hợp lệ |
| `folds.append((train, valid))` | Ghi lại một cặp vị trí CV vào list |

### Cuối file còn những import gì?

```python
# Tương thích với các import cũ; KHÔNG phải định nghĩa mới của các hàm.
from pipeline.modeling.imbalance import (
    sample_training_indices, undersampled_search,
    mean_cv_scores, tune_svm_threshold
)
from pipeline.modeling.evaluation import predict_record
```

Những lệnh này chuyển tiếp tên hàm để code cũ vẫn import được. Chúng **không tạo thêm chiến lược hay thực hiện huấn luyện** khi chỉ đọc `training_utils.py`; hàm thực nằm trong `imbalance.py` và `evaluation.py`.

---
## 12. Những câu hỏi giảng viên có thể hỏi — đáp án để nói miệng

**C1. `training_utils.py` giải quyết bài toán gì?**  
Chia dataset đã có nhãn thành Train/Test theo nhóm nội dung và tạo 5 cặp Train/Validation trong Train. File không biến chữ thành vector và không fit mô hình.

**C2. Tại sao vẫn cần `norm_key()` sau khi nhóm đã xóa dòng trùng?**  
Bước thu thập chỉ xóa bản ghi trùng theo `(post_id, comment_type, text)`. Bước chia tập còn cần nhóm các bình luận chỉ khác giá, dấu câu, chữ hoa/thường vào cùng phía để hạn chế rò rỉ dữ liệu.

**C3. `norm_key()` có làm mất dấu tiếng Việt hoặc sửa dữ liệu train không?**  
Không. NFC chuẩn hóa biểu diễn Unicode, không xóa dấu; khóa được tạo riêng để chia tập, không thay nội dung train.

**C4. `bucket()` dùng làm gì?**  
Dùng MD5 của `salt|key` rồi `%5` để gán nội dung vào nhóm ổn định. Cùng khóa luôn cùng nhóm với cùng salt.

**C5. `salt="test"` và `salt="cv"` khác nhau?**  
Một bên xác định Train/Test toàn dataset; bên kia xác định năm fold bên trong Train. Salt khác giúp phân hai mục đích chia.

**C6. `split_train_test()` trả gì?**  
Ba mảng: khóa toàn CSV, index Train và index Test. Không trả vector BoW/TF-IDF.

**C7. `grouped_folds()` trả gì?**  
List năm cặp `(tr,va)` tính trên Train. Trong từng cặp, tập fit và tập validation không giao nhau về dòng hoặc khóa.

**C8. 5-fold CV có phải tạo năm mô hình cuối cùng không?**  
Không. CV thử các cấu hình, chấm trên năm validation và chọn tham số; sau đó cấu hình thắng được fit lại trên toàn bộ Train để tạo model cuối.

**C9. Vì sao Train 30.759 nhưng các lượt fit chỉ khoảng 24.000–25.000 dòng?**  
Mỗi lượt giữ riêng một fold validation (~5.900–6.400 dòng), bốn fold còn lại dùng để học. Cộng fit + validation ở mỗi lượt đều bằng 30.759.

**C10. Tại sao 5 fold không đều nhau?**  
Vì chia theo nhóm nội dung bằng hash, không ép số dòng và không stratify nhãn.

**C11. Tại sao tỷ lệ Spam Train 14,22% còn Test 12,72%?**  
Vì chia theo khóa hash, không dùng nhãn để cân bằng phân bố. Hàm chỉ kiểm tra mỗi phía có đủ cả Ham và Spam.

**C12. Một câu có được tham gia fit ở nhiều lượt CV không?**  
Có. Với năm fold, mỗi câu Train thường được dùng fit ở bốn lượt và dùng validation ở một lượt. Không được vừa fit vừa validation **trong cùng một lượt**.

**C13. `keys[train_idx]` khác `y_train` thế nào?**  
`keys[train_idx]` là khóa nội dung của Train dùng chia fold; `y_train` là nhãn thật tương ứng của Train, dùng kiểm tra hai lớp và về sau học mô hình.

**C14. `CV_FOLDS` khác `train_idx` thế nào?**  
`train_idx` là index **trong toàn CSV**; `CV_FOLDS` là index **trong tập Train**, cần dùng trên các mảng `X_train_text`/`y_train`.

**C15. `training_utils.py` đã xử lý mất cân bằng chưa?**  
Chưa. Nó chỉ nhận diện nhãn để kiểm tra; các cách cân bằng được làm trong `imbalance.py` về sau.

**C16. Khi nào mới được dùng Test?**  
Sau khi chốt cấu hình và mô hình dựa trên CV Train. `evaluation.py` dùng Test độc lập để báo cáo, không dùng Test tối ưu alpha/C/ngưỡng.

**C17. Vì sao viBERT cần dùng chung cách chia?**  
Để hai họ mô hình được đo trên cùng bình luận và nhãn Test, giảm sai lệch do một bên nhận tập dễ hơn bên kia. Phải đối chiếu đúng ID và cùng quy tắc chia, không chỉ dùng cùng seed.

**C18. Nhóm đã chặn hoàn toàn rò rỉ dữ liệu chưa?**  
Chưa thể khẳng định. Hàm chặn giao nhau theo khóa đã chuẩn hóa, không phát hiện mọi câu giống nghĩa, cùng tác giả hoặc cùng chiến dịch Spam. Đây là giới hạn cần báo cáo.

---

## 13. Sơ đồ tổng kết và chuyển sang file kế tiếp

```text
                 Toàn CSV: 38.233 câu, nhãn 0/1
                         |
               training_utils.py
                         |
               split_train_test(...)
                    /          \
       Train: 30.759            Test: 7.474 (giữ riêng)
              |
       grouped_folds(...)
              |
          5 cặp index
        (fold train, validation)
              |
      CELL 13 + experiments.py
    GridSearchCV(..., cv=CV_FOLDS)
              |
   fit vectorizer/classifier CHỈ trên fold train
   chấm F1 trên fold validation, chọn tham số
              |
    fit lại mô hình cuối trên toàn bộ Train
              |
    CELL 20 + evaluation.py: đo trên Test
```

**Một câu chốt để thuộc:** `training_utils.py` không quyết định bình luận có phải Spam hay không. Nó **quyết định những dòng nào dùng để dạy mô hình và những dòng nào dùng để kiểm tra mô hình**, đồng thời giữ các nội dung cùng khóa ở cùng phía chia.

## 14. File nguồn và phạm vi kiểm chứng

- `training_utils(1).py`: **mã nguồn chuẩn** của `norm_key` (15–31), `bucket` (34–42), `split_train_test` (45–67), `grouped_folds` (70–96) và import tương thích cũ (99–102).
- `02_thuc_nghiem_va_demo(1).ipynb`: cell 3, 4, 6, 9, 10, **11**, 13, 16, 20, 32; bảng Train/Test và 5 fold đọc trực tiếp từ **output cell 11**.
- `clean_for_labeling.py`: cơ chế xóa bản ghi trùng chính xác trước gán nhãn.
- `preprocess_for_model.py`: đầu vào là câu đã làm sạch và câu gốc, đầu ra là chuỗi token theo cấu hình từ dừng (phần chuẩn bị của thành viên khác).
- `experiments(1).py`, `imbalance(1).py`, `evaluation(1).py`, `README(1).md`: kiểm tra nơi tiêu thụ index/folds của bước này.

> **Bước học tiếp theo**: `text_features(1).py` kết hợp tài liệu `PHI_HIEU_4_LOP_SCIKIT_LEARN.md` và cell 13 của Notebook 02. Tập trung vào `CountVectorizer`, `TfidfVectorizer`, từ điển, n-gram, `min_df`, `fit_transform`, `transform`, ma trận thưa và việc tại sao chỉ fit trên fold train.
