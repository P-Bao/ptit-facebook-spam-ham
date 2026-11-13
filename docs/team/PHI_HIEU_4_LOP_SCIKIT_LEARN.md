# Phi đọc bốn lớp scikit-learn đang dùng trong project

Tài liệu này bắt đầu **sau khi đã có cột văn bản token và nhãn**. Phi dùng bốn lớp có sẵn: `CountVectorizer`, `TfidfVectorizer`, `MultinomialNB` và `LinearSVC`. Mục tiêu là đọc được **đường gọi hàm thật trong scikit-learn 1.6.1 của `.venv`**, hiểu đầu vào, trạng thái học và đầu ra. Các trích đoạn mã gốc được ghi rõ file/phương thức; đoạn tên hàm tiếng Việt là diễn giải để học, không phải hàm của thư viện. Số dòng giúp Ctrl+Click/ấn Ctrl+G đến toàn bộ phương thức.

## 1. Bốn lớp nằm ở đâu trong luồng của Phi?

```text
CSV: chuỗi token + label 0/1
   │
   ├─ text_features.make_vectorizer("BoW") → CountVectorizer
   └─ text_features.make_vectorizer("TF-IDF") → TfidfVectorizer
           ↓ fit_transform: ma trận thưa số câu × số đặc trưng
   ├─ classifiers.make_classifier("Naive Bayes") → MultinomialNB
   └─ classifiers.make_classifier("SVM") → LinearSVC
           ↓ fit(ma trận, nhãn): học tham số phân loại
       score/nhãn → CV → chọn cấu hình → test → lưu .joblib
```

Trong project, [text_features.py](../../pipeline/modeling/text_features.py) **chỉ tạo** vectorizer, [classifiers.py](../../pipeline/modeling/classifiers.py) **chỉ tạo** classifier. [experiments.py](../../pipeline/modeling/experiments.py) ghép hai đối tượng vào `Pipeline([("vec", ...), ("clf", ...)])`; `GridSearchCV.fit(X_train_text, y_train)` mới gọi các phương thức học. `X_train_text` vẫn là các **chuỗi token**, chưa phải vector. Trong mỗi fold, Pipeline gọi `vec.fit_transform(text_fold_train)` rồi `clf.fit(ma_tran, y_fold_train)`. Khi chấm validation/test hoặc dự đoán câu mới, nó gọi `vec.transform(text_moi)` trước classifier. Vì thế từ vựng và IDF được học trong **fold train**, không nhìn trước validation/test. [Pipeline của scikit-learn](https://scikit-learn.org/1.6/modules/generated/sklearn.pipeline.Pipeline.html) mô tả đúng thứ tự gọi này.

### Cấu hình đúng trong project

```python
# pipeline/modeling/text_features.py, rút từ make_vectorizer()
common = {
    "token_pattern": r"(?u)\b\w+\b|[^\w\s]",
    "min_df": 2,
    "ngram_range": (1, 1),   # GridSearchCV còn thử (1, 2)
    "lowercase": True,
    "dtype": np.float64,
}
bow = CountVectorizer(**common)
tfidf = TfidfVectorizer(**common, sublinear_tf=True,
                        smooth_idf=True, norm="l2")

# pipeline/modeling/classifiers.py, trước khi CV thay alpha/C
nb = MultinomialNB()
svm = LinearSVC(dual=False, max_iter=20000, random_state=42)
```

`CountVectorizer`/`TfidfVectorizer` **không tự bỏ stop words** trong cấu hình này (`stop_words=None`). Hai cột token giữ/bỏ từ dừng đã được chuẩn bị trước; Notebook 02 so hai cột bằng CV. Bốn cặp cơ sở B1–B4 là BoW/TF-IDF × NB/SVM; lưới còn thử `ngram_range` và `alpha`/`C`. `min_df=2` tính trên tập văn bản mỗi lần `fit`, không phải trên toàn CSV một lần.

| Khái niệm | Dữ liệu trước | Dữ liệu sau | Học gì? |
|---|---|---|---|
| `vec.fit_transform` | Danh sách chuỗi token của fold train | Ma trận thưa không âm | Từ vựng; TF-IDF học thêm IDF. |
| `clf.fit` | Ma trận trên và nhãn 0/1 | Chính đối tượng classifier đã fit | NB học thống kê lớp/từ; SVM học hệ số ranh giới. |
| `vec.transform` | Chuỗi validation/test/mới | Ma trận cùng số cột với train | Không học từ vựng/IDF mới. |
| `clf.predict_proba` hoặc `decision_function` | Ma trận đã vector hóa | Score cho mỗi câu | Không fit lại. |

## 2. Cách Ctrl+Click đúng chỗ trong mã thư viện

Chọn interpreter `.venv` của project trong VS Code. Từ tên lớp trong `text_features.py` hoặc `classifiers.py`, Ctrl+Click/F12 mở file cài đặt. Nếu vào đầu class dài hàng trăm dòng, dùng Ctrl+F tìm tên phương thức, hoặc Ctrl+G tới số dòng dưới đây. Đây là **số dòng của scikit-learn 1.6.1 đang cài**, có thể đổi nếu cập nhật phiên bản.

| Lớp/phần | File local trong `.venv/Lib/site-packages/sklearn/` | Dòng đáng đọc |
|---|---|---|
| CountVectorizer | [`feature_extraction/text.py`](../../.venv/Lib/site-packages/sklearn/feature_extraction/text.py) | class 928; `_count_vocab` 1246; `fit_transform` 1331; `transform` 1398. |
| Tách từ/ngram dùng chung | cùng `feature_extraction/text.py` | `_analyze` 68; `_word_ngrams` 238; `build_preprocessor` 319; `build_tokenizer` 346; `build_analyzer` 415. |
| TfidfVectorizer | cùng `feature_extraction/text.py` | class 1734; `fit_transform` 2078; `transform` 2110. |
| TfidfTransformer bên trong | cùng `feature_extraction/text.py` | class 1490; `fit` 1635; `transform` 1682. |
| MultinomialNB | [`naive_bayes.py`](../../.venv/Lib/site-packages/sklearn/naive_bayes.py) | class 779; `_count` 887; `_update_feature_log_prob` 893; `_joint_log_likelihood` 902. |
| Các phương thức NB kế thừa | cùng `naive_bayes.py` | `_BaseDiscreteNB.fit` 715; `_BaseNB.predict` 90; `_BaseNB.predict_proba` 132. |
| LinearSVC | [`svm/_classes.py`](../../.venv/Lib/site-packages/sklearn/svm/_classes.py) | class 35; `fit` 281. |
| Bộ tối ưu của LinearSVC | [`svm/_base.py`](../../.venv/Lib/site-packages/sklearn/svm/_base.py) | `_get_liblinear_solver_type` 1009; `_fit_liblinear` 1066. |
| Score/nhãn SVM kế thừa | [`linear_model/_base.py`](../../.venv/Lib/site-packages/sklearn/linear_model/_base.py) | `LinearClassifierMixin.decision_function` 329; `predict` 359. |

**Vì sao Ctrl+Click vào class chưa thấy tất cả?** Python cho phép **kế thừa**. `TfidfVectorizer` dùng lại nhiều hàm của `CountVectorizer`; `MultinomialNB` dùng `fit`/`predict_proba` từ lớp cha; `LinearSVC` dùng `decision_function`/`predict` từ `LinearClassifierMixin`. Tìm tên phương thức trong toàn file hoặc đọc bảng trên. Riêng phần tối ưu số của SVM chuyển sang **Liblinear đã biên dịch**, nên không thể hiểu toàn bộ chỉ từ một file Python.

## 3. CountVectorizer: từ chuỗi token sang BoW

### Trích đoạn mã gốc cần đọc khi Ctrl+Click

Các trích đoạn ở mục 3–6 lấy từ scikit-learn 1.6.1 cài trong `.venv` (giấy phép BSD-3-Clause). Chúng trích **đường xử lý được project sử dụng**, không chép các nhánh tùy chọn không dùng. Mở file và số dòng ở mục 2 để xem phương thức đầy đủ.

```python
# feature_extraction/text.py, CountVectorizer._count_vocab
for doc in raw_documents:
    feature_counter = {}
    for feature in analyze(doc):
        try:
            feature_idx = vocabulary[feature]
            if feature_idx not in feature_counter:
                feature_counter[feature_idx] = 1
            else:
                feature_counter[feature_idx] += 1
        except KeyError:
            continue  # ở transform: bỏ từ không thuộc vocabulary_ đã học
    j_indices.extend(feature_counter.keys())
    values.extend(feature_counter.values())
    indptr.append(len(j_indices))
```

`analyze(doc)` là tokenizer + tạo n-gram đã cấu hình. `feature_counter` là bộ đếm **một câu**; `vocabulary[feature]` tra tên đặc trưng ra số cột. `j_indices` giữ cột, `values` giữ số đếm, `indptr` chỉ vị trí bắt đầu/kết thúc mỗi hàng trong ma trận thưa CSR. Ở lúc fit, `vocabulary` cho phép thêm cột; lúc transform, nó cố định và `KeyError` bỏ từ ngoài từ vựng.

Trong `fit_transform`, sau khi có ma trận đếm, mã gọi `_limit_features(X, vocabulary, max_doc_count, min_doc_count, max_features)`. Ở bài này `min_doc_count=2` vì `min_df=2`, còn `max_df=1.0` và `max_features=None` không lọc thêm theo hai giới hạn đó. Phương thức xây lại `vocabulary_` và ma trận với các cột được giữ. Ở `transform`, nguồn gọi `_count_vocab(raw_documents, fixed_vocab=True)` nên không học lại từ vựng.

```python
# feature_extraction/text.py, CountVectorizer.fit_transform, đoạn lọc sau đếm
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

Trích đoạn bỏ các nhánh kiểm tra, `max_doc_count` và sắp cột để tập trung vào `min_df`; xem toàn bộ phương thức dòng 1331–1396. `n_doc` là số câu train, `min_doc_count` là số câu tối thiểu cần chứa một đặc trưng, `X` là ma trận thưa.

### 3.1. Gọi gì và trả gì?

- `CountVectorizer(**common)` chỉ tạo đối tượng; chưa có `vocabulary_`.
- `fit_transform(danh_sach_cau_train)` học từ vựng, đếm từ/cặp từ và trả **ma trận CSR** kích thước `số câu × số đặc trưng`. `y` không cần cho bước này.
- `transform(danh_sach_cau_moi)` đếm theo **từ vựng đã học**, giữ đúng số cột. Từ ngoài từ vựng bị bỏ qua.
- `get_feature_names_out()` cho tên cột theo đúng thứ tự của ma trận; `vocabulary_` ánh xạ tên đặc trưng → chỉ số cột.

### 3.2. Đọc các phương thức bên trong theo thứ tự

1. `fit_transform` kiểm `ngram_range` và các tham số, rồi gọi `_count_vocab(raw_documents, fixed_vocab=False)`. “False” nghĩa là **được phép xây từ vựng** từ fold train.
2. `_count_vocab` gọi `build_analyzer()`. Với `analyzer="word"` mặc định, analyzer ghép `build_preprocessor` (hạ chữ vì `lowercase=True`), `build_tokenizer` (biên dịch `token_pattern` rồi `findall`) và `_word_ngrams` (nối hai token kề nhau khi CV chọn `(1,2)`). `_analyze` điều phối ba bước.
3. Với mỗi câu, `_count_vocab` tạo bộ đếm đặc trưng của **riêng câu đó**, tăng số lần xuất hiện, và ghi vị trí cột/giá trị vào cấu trúc của ma trận CSR. Kết quả ban đầu là từ vựng ứng viên và ma trận đếm.
4. `_limit_features` tính **document frequency**: số câu có đặc trưng (khác tần suất lặp trong một câu). `min_df=2` xóa các cột chỉ gặp trong một câu train. `_sort_features` sắp lại cột theo tên để thứ tự ổn định. `vocabulary_` lưu từ vựng đã lọc.
5. `transform` gọi lại `_count_vocab(..., fixed_vocab=True)`: chỉ tra cột cũ, gặp từ lạ thì bỏ; **không chạy lại lọc `min_df`, không thêm cột**.

[Mã gốc `_count_vocab`](../../.venv/Lib/site-packages/sklearn/feature_extraction/text.py) nằm ở dòng 1246–1309 và `fit_transform` ở 1331–1396. Đọc trích đoạn ở đầu mục này rồi mở toàn bộ hai phương thức để xem thêm phần tạo ma trận CSR và lọc từ vựng.

**Ví dụ:** fold train có “mua sim”, “mua sim”, “lịch thi”. Với `min_df=2` và `(1,2)`, các cột còn `mua`, `sim`, `mua sim`; “lịch”, “thi” chỉ ở một câu nên bị bỏ. Hai câu đầu có giá trị 1 ở ba cột, câu cuối là hàng toàn 0. `CountVectorizer` không tự kết luận spam/ham; nó chỉ tạo số cho NB/SVM.

### 3.3. Tham số ta dùng

| Tham số | Trong bài | Tác dụng |
|---|---|---|
| `token_pattern` | `(?u)\b\w+\b|[^\w\s]` | Giữ từ Unicode, kể cả từ một ký tự, và dấu riêng lẻ. |
| `lowercase` | `True` | Đổi về chữ thường trước khi tra từ vựng. |
| `min_df` | `2` | Chỉ giữ đặc trưng có ở ít nhất hai câu trong lần fit. |
| `ngram_range` | CV thử `(1,1)`, `(1,2)` | Chỉ từ đơn hoặc thêm cặp kề nhau. |
| `dtype` | `np.float64` | Kiểu số của ma trận; đếm vẫn là 1, 2, 3… nhưng lưu dạng số thực để ghép bộ học nhất quán. |
| `stop_words` | Không truyền, mặc định `None` | Lựa chọn giữ/bỏ từ dừng đã ở cột token đầu vào. |

Nguồn API: [CountVectorizer của scikit-learn 1.6.1](https://scikit-learn.org/1.6/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html).

## 4. TfidfVectorizer: số đếm có trọng số

### Trích đoạn mã gốc cần đọc khi Ctrl+Click

Các đoạn dưới là các phương thức và nhánh mã thật mà `TfidfVectorizer` gọi; nguồn đầy đủ nằm ở số dòng trong mục 2.

```python
# feature_extraction/text.py, TfidfVectorizer.fit_transform
X = super().fit_transform(raw_documents)  # CountVectorizer học từ vựng, đếm từ
self._tfidf.fit(X)                        # học idf_ từ số câu train chứa mỗi cột
return self._tfidf.transform(X, copy=False)  # TF × IDF, rồi chuẩn hóa
```

```python
# feature_extraction/text.py, TfidfTransformer.fit, nhánh use_idf=True
n_samples, _ = X.shape
df = _document_frequency(X)
df += float(self.smooth_idf)
n_samples += int(self.smooth_idf)
self.idf_ = np.full_like(df, fill_value=n_samples, dtype=dtype)
self.idf_ /= df
np.log(self.idf_, out=self.idf_)
self.idf_ += 1.0
```

`df` là số câu có từng đặc trưng. Hai phép cộng `smooth_idf=True` thêm 1 ở tử và mẫu, `log` là log tự nhiên, dòng cuối thêm 1 để IDF của từ có ở mọi câu vẫn bằng 1. Nguồn đầy đủ còn có kiểm kiểu dữ liệu và nhánh `use_idf=False`, không dùng trong bài.

```python
# feature_extraction/text.py, TfidfTransformer.transform
if self.sublinear_tf:
    np.log(X.data, X.data)
    X.data += 1.0
if hasattr(self, "idf_"):
    X.data *= self.idf_[X.indices]
if self.norm is not None:
    X = normalize(X, norm=self.norm, copy=False)
return X
```

`X.data` là các giá trị **khác 0** của ma trận thưa; `X.indices` là cột tương ứng. Vì thế phép nhân lấy đúng IDF của từng đặc trưng mà không cần vòng lặp qua tất cả ô 0. `normalize` chia từng hàng theo chuẩn L2 trong cấu hình bài.

### 4.1. Đầu vào, trạng thái, đầu ra

`TfidfVectorizer` nhận **cùng chuỗi token** như `CountVectorizer`. Nó kế thừa bước tách token, tạo n-gram, học từ vựng và ma trận đếm của `CountVectorizer`; sau đó dùng `TfidfTransformer` để học `idf_` trên **fold train** và biến số đếm thành trọng số. `fit_transform(train_text)` trả ma trận thưa TF-IDF; `transform(validation_text)` dùng đúng `vocabulary_` và `idf_` đã học, không học lại trên validation. Các cột vẫn là đặc trưng từ đơn hoặc cặp từ; giá trị nay là số thực.

**Đường đọc nguồn:** [`TfidfVectorizer`](../../.venv/Lib/site-packages/sklearn/feature_extraction/text.py) dòng 1734, `fit_transform` 2078–2108, `transform` 2110–2130. Trong cùng file, đọc `TfidfTransformer.fit` 1635–1680 và `TfidfTransformer.transform` 1682–1724. `fit_transform` gọi `super().fit_transform` để lấy ma trận đếm; đó là phương thức của `CountVectorizer` đã giải thích ở mục 3.

### 4.2. Từng bước bên trong các phương thức

1. `__init__` giữ tham số `sublinear_tf=True`, `smooth_idf=True`, `norm="l2"` và các tham số chung. Tạo đối tượng chưa có từ vựng hoặc IDF.
2. `fit_transform` lấy ma trận đếm `X` từ lớp cha. `TfidfTransformer.fit(X)` đếm `df_j`, tức số **văn bản train** chứa đặc trưng `j`. Với `N` văn bản và `smooth_idf=True`, nó tính `idf_j = 1 + ln((1+N)/(1+df_j))`. Nó lưu dãy này vào `idf_`.
3. `TfidfTransformer.transform(X)` xử lý các ô khác 0. Với `sublinear_tf=True`, `tf=1` vẫn là `1`, còn `tf=3` thành `1+ln(3)`, giảm ảnh hưởng của lặp từ. Sau đó nhân từng ô của cột `j` với `idf_j`.
4. Với `norm="l2"`, mỗi hàng được chia cho căn bậc hai của tổng bình phương các ô trong hàng. Hàng toàn 0 vẫn toàn 0. Đây là chuẩn hóa **vector mỗi bình luận**, không phải chuẩn hóa toàn dataset.
5. `transform` của TfidfVectorizer trước hết gọi `CountVectorizer.transform` để đếm theo từ vựng cũ, rồi `TfidfTransformer.transform` với `idf_` cũ. Văn bản mới không thể thay đổi số cột hoặc IDF.

**Ví dụ:** trong 3 câu train, một từ ở cả 3 câu có `idf=1`; một từ ở 2 câu có `idf=1+ln(4/3)≈1,288`. Từ gặp ở ít câu hơn được tăng trọng số. Tuy vậy `min_df=2` đã xóa từ chỉ gặp ở 1 câu trước khi tính IDF. `TfidfVectorizer` không gán nhãn; ma trận của nó là đầu vào cho NB/SVM.

Nguồn API: [TfidfVectorizer](https://scikit-learn.org/1.6/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html).

## 5. MultinomialNB: học thống kê đặc trưng theo lớp

### Trích đoạn mã gốc cần đọc khi Ctrl+Click

```python
# naive_bayes.py, MultinomialNB._count
check_non_negative(X, "MultinomialNB (input X)")
self.feature_count_ += safe_sparse_dot(Y.T, X)
self.class_count_ += Y.sum(axis=0)

# MultinomialNB._update_feature_log_prob
smoothed_fc = self.feature_count_ + alpha
smoothed_cc = smoothed_fc.sum(axis=1)
self.feature_log_prob_ = np.log(smoothed_fc) - np.log(
    smoothed_cc.reshape(-1, 1)
)

# MultinomialNB._joint_log_likelihood
return safe_sparse_dot(X, self.feature_log_prob_.T) + self.class_log_prior_
```

```python
# naive_bayes.py, _BaseDiscreteNB.fit, đoạn sau chuẩn bị X/Y
self._init_counters(n_classes, n_features)
self._count(X, Y)
alpha = self._check_alpha()
self._update_feature_log_prob(alpha)
self._update_class_log_prior(class_prior=class_prior)
return self

# naive_bayes.py, _BaseNB.predict / predict_proba
jll = self._joint_log_likelihood(X)
return self.classes_[np.argmax(jll, axis=1)]  # nhánh predict
# nhánh predict_proba riêng: return np.exp(self.predict_log_proba(X))
```

`_init_counters` cấp bộ đếm rỗng; `_count` cộng thống kê; hai `_update` chuyển thống kê thành log xác suất. Trong `predict`, `argmax` chọn cột điểm lớn nhất rồi ánh xạ về nhãn 0/1. Dòng `predict_proba` nằm ở **phương thức khác**: nó gọi `predict_log_proba` để chuẩn hóa hai điểm trước khi lấy hàm mũ. Vì trích từ nhiều phương thức, khối trên dùng để **đọc**, không chạy liền như một hàm.

`Y` là bảng chỉ báo lớp của từng câu, tạo ở `_BaseDiscreteNB.fit`. `Y.T @ X` cộng vector của các câu cùng lớp. Phép trừ hai log tương đương log của một tỷ lệ. `safe_sparse_dot` là phép nhân ma trận hỗ trợ dữ liệu thưa; không phải hàm quyết định nhãn.

### 5.1. Đường gọi và dữ liệu

[`classifiers.py`](../../pipeline/modeling/classifiers.py) tạo `MultinomialNB()` và để GridSearchCV thử `alpha=0,1; 0,3; 1,0`. `X` khi fit là ma trận BoW/TF-IDF **không âm**; `y` là nhãn `0=ham, 1=spam`. Sau fit, mô hình nhớ `classes_`, `class_count_`, `feature_count_`, `class_log_prior_`, `feature_log_prob_`. Hai mảng sau là log xác suất tiên nghiệm của lớp và log xác suất đặc trưng theo lớp. `predict(text_vector)` trả nhãn; `predict_proba` trả xác suất hai lớp.

**Đường đọc nguồn:** [`MultinomialNB`](../../.venv/Lib/site-packages/sklearn/naive_bayes.py) dòng 779; `_count` 887, `_update_feature_log_prob` 893, `_joint_log_likelihood` 902. `fit` **kế thừa** từ `_BaseDiscreteNB` dòng 715–779; `predict` và `predict_proba` kế thừa từ `_BaseNB` dòng 90–150. Đây là lý do Ctrl+Click vào `MultinomialNB` không thấy toàn bộ hàm.

### 5.2. Đọc `fit` theo thứ tự

1. `fit(X, y)` kiểm tra ma trận/nhãn, tìm các lớp và mã hóa nhãn thành chỉ báo lớp. `_init_counters` tạo bộ đếm bằng 0.
2. `_count(X, Y)` cộng các giá trị ở mỗi cột `X` cho từng lớp: về ý tưởng `feature_count_[lớp, cột] = tổng X[câu,cột]` của các câu mang lớp đó. Lệnh ma trận tương ứng trong nguồn là `Y.T @ X`. `class_count_` đếm số câu ở mỗi lớp.
3. `_check_alpha` kiểm tra `alpha`. `_update_feature_log_prob` cộng `alpha` vào từng đặc trưng, chia cho tổng đặc trưng của lớp sau cộng, rồi lấy log. Với lớp `c` và đặc trưng `j`: `log P(j|c) = log((feature_count[c,j]+alpha)/(sum_j feature_count[c,j]+alpha·V))`, `V` là số cột. Làm trơn giúp xác suất không bằng 0 khi chưa gặp một từ ở lớp đó.
4. `_update_class_log_prior` lấy tần suất lớp trong train để tính `log P(c)`; trong bài không truyền `class_prior`. Nhãn spam ít hơn thì prior spam thấp hơn.

### 5.3. Đọc `predict`/`predict_proba`

`_joint_log_likelihood(X)` lấy `X @ feature_log_prob_.T + class_log_prior_`. Mỗi hàng có hai điểm log: ham và spam. `predict` chọn lớp có điểm lớn hơn. `predict_proba` chuẩn hóa hai log điểm về xác suất cộng thành 1 (qua `predict_log_proba` rồi hàm mũ). Khi notebook mặc định dùng ngưỡng `0,5` cho spam, hai lớp tương đương chọn điểm cao hơn; ở thí nghiệm chỉnh ngưỡng cho NB (nếu có), đó là quyết định **sau fit**, không đổi `feature_log_prob_`.

`MultinomialNB` gốc mô tả số đếm rời rạc. Scikit-learn cũng chấp nhận ma trận TF-IDF không âm trong thí nghiệm phân loại văn bản; khi đó `feature_count_` là tổng **trọng số** chứ không còn số lần xuất hiện nguyên. Không diễn giải `predict_proba` là xác suất đã hiệu chuẩn để dùng trực tiếp trong quyết định thực tế.

Nguồn API: [MultinomialNB](https://scikit-learn.org/1.6/modules/generated/sklearn.naive_bayes.MultinomialNB.html).

## 6. LinearSVC: học ranh giới ham/spam

### Trích đoạn mã gốc cần đọc khi Ctrl+Click

```python
# svm/_classes.py, LinearSVC.fit, lược bớt danh sách đối số ở đây
self.classes_ = np.unique(y)
self.coef_, self.intercept_, n_iter_ = _fit_liblinear(
    X, y, self.C, self.fit_intercept, self.intercept_scaling,
    self.class_weight, self.penalty, _dual, self.verbose,
    self.max_iter, self.tol, self.random_state, self.multi_class,
    self.loss, sample_weight=sample_weight,
)

# linear_model/_base.py, LinearClassifierMixin.decision_function/predict
scores = safe_sparse_dot(X, self.coef_.T, dense_output=True) + self.intercept_
indices = xp.astype(scores > 0, indexing_dtype(xp))  # trường hợp hai lớp
return xp.take(self.classes_, indices, axis=0)
```

`coef_` và `intercept_` là kết quả học, `n_iter_` là số vòng; `_fit_liblinear` gọi tiếp phần Liblinear biên dịch. Ba dòng cuối thuộc **hai phương thức khác nhau**: `decision_function` trả `scores`; `predict` gọi nó rồi ánh xạ chỉ số sang nhãn trong `classes_`. Chúng được đặt cạnh nhau để thấy luồng, không phải một thân hàm liên tục.

### 6.1. Đầu vào và tham số đang dùng

[`classifiers.py`](../../pipeline/modeling/classifiers.py) tạo `LinearSVC(dual=False, max_iter=20000, random_state=42)`. CV thử `C=0,1; 0,3; 1; 3` và `class_weight=None` cho B2/B4; S1 thử `class_weight="balanced"`. `X` là ma trận thưa BoW/TF-IDF, `y` là nhãn 0/1. `C` càng lớn thì phạt lỗi huấn luyện càng nặng; `class_weight="balanced"` tăng trọng số lớp ít mẫu theo tỷ lệ nghịch với số mẫu lớp. `max_iter` giới hạn số vòng tối ưu; `dual=False` chọn dạng bài toán đối ngẫu **không** dùng, phù hợp cấu hình đang chạy. Theo tài liệu scikit-learn, `random_state` không ảnh hưởng khi `dual=False`.

**Đường đọc nguồn:** [`LinearSVC.fit`](../../.venv/Lib/site-packages/sklearn/svm/_classes.py) dòng 281–350; [`_fit_liblinear`](../../.venv/Lib/site-packages/sklearn/svm/_base.py) dòng 1066–1270; [`LinearClassifierMixin.decision_function`](../../.venv/Lib/site-packages/sklearn/linear_model/_base.py) dòng 329–358 và `predict` dòng 359–385. `LinearSVC` kế thừa hai phương thức cuối, nên không thấy chúng khi chỉ nhìn class trong `_classes.py`.

### 6.2. Bên trong `fit`

1. Kiểm tra `X` và `y`, chuyển ma trận về định dạng và kiểu số mà bộ tối ưu cần; lấy `classes_ = [0, 1]`.
2. `_get_liblinear_solver_type` chọn bộ giải theo `penalty="l2"`, `loss="squared_hinge"` mặc định và `dual=False`. Trong nhánh bài này, nó dùng solver Liblinear mã `2`.
3. `_fit_liblinear` mã hóa nhãn, tạo trọng số từng lớp nếu `class_weight="balanced"`, rồi chuyển `X`, `y`, `C`, `tol`, `max_iter`, solver sang `liblinear.train_wrap`.
4. Bộ tối ưu Liblinear (phần biên dịch C/C++, **không phải def Python dễ đọc bằng Ctrl+Click**) tìm `w` và `b` của siêu phẳng. Về mục tiêu, nó cân bằng độ lớn trọng số và lỗi phân loại: `(1/2)||w||² + C·Σ max(0, 1-y_i(w·x_i+b))²` với nhãn toán học `y_i∈{-1,+1}`; khi có `class_weight`, mức phạt `C` thay đổi theo lớp. Đây là cách diễn giải toán học của cấu hình `l2/squared_hinge`, không phải mã Python của optimizer.
5. Python nhận kết quả vào `coef_` (trọng số mỗi cột đặc trưng), `intercept_` và `n_iter_`. Nếu không hội tụ sau giới hạn vòng lặp, thư viện phát cảnh báo; cần xem cảnh báo đó thay vì mặc định tin điểm.

### 6.3. Bên trong `decision_function` và `predict`

`decision_function(X)` tính `X @ coef_.T + intercept_`: một **điểm có dấu** cho mỗi câu. Với `classes_=[0,1]`, điểm dương nghiêng về spam, điểm âm nghiêng về ham; khoảng cách lớn hơn thường cho thấy quyết định mạnh hơn nhưng **không phải xác suất**. `predict(X)` của lớp cha chọn lớp 1 khi điểm `>0`, lớp 0 khi điểm `≤0`. `LinearSVC` không có `predict_proba`; code đánh giá lấy `decision_function` làm score để vẽ PR/AP.

Ở S3, nhóm **giữ nguyên mô hình B4 đã học**, thu score trên các lượt validation của train và chọn một ngưỡng tối ưu F1 spam; ví dụ ngưỡng âm `-0,1776` khiến cả một số câu có score hơi âm được gọi là spam. Cách đó thường tăng recall, có thể giảm precision. Ngưỡng được áp cho test và lưu cùng mô hình; **không** fit SVM trên test.

Nguồn API: [LinearSVC](https://scikit-learn.org/1.6/modules/generated/sklearn.svm.LinearSVC.html).

## 7. Bốn lớp đi vào Notebook 02 như thế nào?

`make_vectorizer` và `make_classifier` chỉ tạo đối tượng **chưa fit**. [`make_pipe`](../../pipeline/modeling/experiments.py) ghép `Pipeline([("vec", vectorizer), ("clf", classifier)])`. Khi [`fit_experiment`](../../pipeline/modeling/experiments.py) gọi `GridSearchCV(...).fit(X_train, y_train)`, mỗi cấu hình được thử trên năm fold:

```text
fold train (chuỗi token) → vec.fit_transform → ma trận X số
                         → clf.fit(X số, nhãn) → tham số đã học
fold validation           → vec.transform    → clf.predict/score → P, R, F1 spam
5 fold                   → trung bình điểm validation, chọn theo F1 spam
toàn train                → fit lại cấu hình thắng (refit)
test độc lập              → chỉ transform + predict/score; tính điểm báo cáo
```

Trong `cv_results_` của scikit-learn, trường `mean_test_f1` là điểm **validation của CV**, dù tên có chữ `test`; test độc lập của project chỉ xuất hiện ở bước đánh giá sau khi chọn cấu hình. `vec__ngram_range` và `clf__alpha`/`clf__C` là cách `Pipeline` chuyển tham số cho đúng bước. Ma trận BoW/TF-IDF không cần lưu thành một CSV lớn: vectorizer đã fit nằm **bên trong pipeline** và được lưu trong `.joblib`, nên bình luận mới dùng cùng từ vựng/IDF.

| Mã cơ sở | Vectorizer | Classifier | Đầu ra |
|---|---|---|---|
| B1 | `CountVectorizer` | `MultinomialNB` | BoW → NB |
| B2 | `CountVectorizer` | `LinearSVC` | BoW → SVM |
| B3 | `TfidfVectorizer` | `MultinomialNB` | TF-IDF → NB |
| B4 | `TfidfVectorizer` | `LinearSVC` | TF-IDF → SVM |

S1/S2/S3 trong Notebook 02 là các thí nghiệm mất cân bằng trên **TF-IDF + SVM**; chúng không phải bốn lớp thư viện mới. Lượt chạy xác minh mới nhất chọn **S3**, lưu một `Pipeline` có `TfidfVectorizer` và `LinearSVC`, cùng ngưỡng trong bundle `artifacts/spamham/spam_ham_ml_only.joblib`. CV F1 spam của S3 là `0,7876`; test Precision `0,8374`, Recall `0,6877`, F1 `0,7552`. Các chỉ số có thể đổi nếu dữ liệu, phiên bản hoặc quy tắc chia tập thay đổi.

### Khi bị hỏi hoặc cần đọc mã gốc

1. **`CountVectorizer.fit` ở đâu?** `fit` của nó gọi `fit_transform` rồi bỏ ma trận; đọc `fit_transform` và `_count_vocab` mới thấy đường làm việc chính.
2. **`TfidfVectorizer` có đếm từ lại không?** Có: nó kế thừa CountVectorizer để lấy ma trận đếm, rồi `TfidfTransformer` biến ma trận đó thành trọng số.
3. **`MultinomialNB.fit` ở đâu?** Ở lớp cha `_BaseDiscreteNB` trong `naive_bayes.py`; class con cung cấp công thức đếm và xác suất đặc trưng.
4. **`LinearSVC.predict` lấy xác suất?** Không. Nó so `decision_function` với 0; S3 so với ngưỡng đã chọn từ validation.
5. **Vì sao không thể đọc toàn bộ optimizer bằng Ctrl+Click?** `LinearSVC.fit` gọi Liblinear đã biên dịch. Hãy giải thích đầu vào, hàm mục tiêu, tham số và đầu ra; mã Python chỉ là lớp điều phối.
6. **Từ vựng/IDF có rò rỉ từ test?** Không: vectorizer nằm trong `Pipeline` được fit chỉ trên train từng fold; test chỉ gọi `transform`.

Các đường dẫn dòng ở trên ứng với `scikit-learn 1.6.1` trong `.venv` của project; cài phiên bản khác có thể đổi số dòng. Tài liệu chỉ diễn giải nhánh cấu hình đang dùng, không mô tả mọi tùy chọn của thư viện.
## Tài liệu mở rộng hiện hành

Để học cả **GridSearchCV** cùng bốn lớp bên dưới, đọc
[Phi 01 — Năm lớp thư viện](PHI_01_NAM_LOP_THU_VIEN_MODELING.md), rồi
[Phi 02 — Luồng fit và tính điểm](PHI_02_LUONG_FIT_TO_HOP_VA_DIEM_CV.md),
[Phi 03 — S1/S2/S3 theo code](PHI_03_CAN_BANG_S1_S2_S3_THEO_CODE.md).
Ba tài liệu này có mã nguồn thư viện/project và giải thích theo luồng hiện tại.

