# Phi 01 — Hiểu năm lớp thư viện trong modeling

Tài liệu đối chiếu mã nguồn scikit-learn **1.6.1** trong `.venv` và code project hiện tại. Các khối “Mã thư viện” được trích nguyên từ môi trường này; số dòng có thể đổi khi nâng phiên bản. Phần giải thích tập trung đường chạy project đang dùng. Không cần học thuộc các nhánh hỗ trợ metadata, nhiều lớp hay tham số mà bài không sử dụng.

## 1. Vị trí trong luồng của Phi

CSV có nhãn và token → chia train/test và folds → chuỗi token của train → **CountVectorizer hoặc TfidfVectorizer** → ma trận số → **MultinomialNB hoặc LinearSVC** → nhãn/score. **GridSearchCV** điều phối thử tham số và CV cho toàn Pipeline.

| Lớp | Nhận lúc học | Học được gì | Trả khi suy luận |
|---|---|---|---|
| CountVectorizer | Danh sách chuỗi | Từ vựng, ánh xạ đặc trưng → cột | Ma trận số đếm thưa |
| TfidfVectorizer | Danh sách chuỗi | Từ vựng và IDF | Ma trận TF-IDF thưa |
| MultinomialNB | Ma trận không âm X và nhãn y | Prior lớp, trọng số log xác suất đặc trưng theo lớp | Nhãn hoặc xác suất ước lượng |
| LinearSVC | Ma trận X và nhãn y | Vector trọng số w và hệ số b | Nhãn hoặc decision score |
| GridSearchCV | Pipeline chưa fit, grid, folds, X/y train | Cấu hình tốt nhất theo CV và model refit | Bảng CV và estimator tốt nhất |

`Class(...)` tạo đối tượng với cấu hình. `fit` học và thường trả chính đối tượng đó. `fit_transform` vừa học vừa chuyển dữ liệu. `transform` chỉ áp kiến thức đã học. `predict` phân loại. Dấu `_` cuối tên như `vocabulary_`, `coef_`, `best_params_` thường chỉ thuộc tính có sau khi fit. Tên bắt đầu `_` thường là phương thức nội bộ.

X có n hàng tương ứng n bình luận, d cột tương ứng d đặc trưng. y có n phần tử, ham=0, spam=1. Trước vector hóa X là chuỗi; khi vào classifier X đã là ma trận số. Ma trận CSR chỉ lưu vị trí và giá trị khác 0 để tiết kiệm RAM: `data` chứa giá trị, `indices` chứa chỉ số cột, `indptr` đánh dấu đoạn của từng hàng.

## 2. CountVectorizer — biến văn bản thành số đếm

Project tạo đối tượng ở `text_features.make_vectorizer('BoW')`. Khi Pipeline.fit chạy, bộ vector hóa nhận **chỉ phần train của fold**.

Các bước bằng ngôn ngữ tự nhiên:

1. Kiểm tra đầu vào là tập bình luận, không phải một chuỗi duy nhất.
2. Tạo analyzer: giải mã → chuyển chữ thường → tách token theo regex → tạo n-gram.
3. Với mỗi câu, đếm mỗi đặc trưng xuất hiện mấy lần.
4. Khi đang học, thêm đặc trưng mới vào từ điển và cấp chỉ số cột.
5. Ghép các số đếm thành CSR.
6. Tính số **bình luận** chứa mỗi đặc trưng; `min_df=2` bỏ đặc trưng chỉ xuất hiện trong một bình luận train. Hai lần lặp trong cùng một câu vẫn chỉ có document frequency bằng 1.
7. Sắp xếp/định lại cột, lưu `vocabulary_`, trả ma trận đã lọc.

Khi transform validation/test, từ điển đã cố định: từ chưa có bị bỏ qua, không sinh cột mới; bộ lọc min_df không được học lại. `y=None` vì vectorizer không cần biết nhãn.

`build_analyzer` ghép các bước; `build_preprocessor` chuẩn hóa chữ; `build_tokenizer` dùng regex; `_word_ngrams` ghép token liên tiếp; `_count_vocab` đếm và tạo CSR; `_limit_features` loại cột theo DF; `_sort_features` đổi thứ tự cột. Project không dùng `stop_words` của vectorizer: nhánh giữ/bỏ từ dừng đã được bàn giao trong hai cột token.

Ví dụ câu `xin lịch thi`: `(1,1)` sinh `xin`, `lịch`, `thi`; `(1,2)` sinh thêm `xin lịch`, `lịch thi`. Nếu từ ghép đã được underthesea nối như `sinh_viên`, regex giữ nó thành một token. Regex project còn giữ từ một ký tự và dấu câu đứng riêng. CountVectorizer không tự hiểu nghĩa từ.

### Mã thư viện: `CountVectorizer._count_vocab`

Nguồn local: `.venv/Lib/site-packages/sklearn/feature_extraction/text.py`, bắt đầu dòng 1246.

```python
    def _count_vocab(self, raw_documents, fixed_vocab):
        """Create sparse feature matrix, and vocabulary where fixed_vocab=False"""
        if fixed_vocab:
            vocabulary = self.vocabulary_
        else:
            # Add a new value when a new vocabulary item is seen
            vocabulary = defaultdict()
            vocabulary.default_factory = vocabulary.__len__

        analyze = self.build_analyzer()
        j_indices = []
        indptr = []

        values = _make_int_array()
        indptr.append(0)
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
                    # Ignore out-of-vocabulary items for fixed_vocab=True
                    continue

            j_indices.extend(feature_counter.keys())
            values.extend(feature_counter.values())
            indptr.append(len(j_indices))

        if not fixed_vocab:
            # disable defaultdict behaviour
            vocabulary = dict(vocabulary)
            if not vocabulary:
                raise ValueError(
                    "empty vocabulary; perhaps the documents only contain stop words"
                )

        if indptr[-1] > np.iinfo(np.int32).max:  # = 2**31 - 1
            if _IS_32BIT:
                raise ValueError(
                    (
                        "sparse CSR array has {} non-zero "
                        "elements and requires 64 bit indexing, "
                        "which is unsupported with 32 bit Python."
                    ).format(indptr[-1])
                )
            indices_dtype = np.int64

        else:
            indices_dtype = np.int32
        j_indices = np.asarray(j_indices, dtype=indices_dtype)
        indptr = np.asarray(indptr, dtype=indices_dtype)
        values = np.frombuffer(values, dtype=np.intc)

        X = sp.csr_matrix(
            (values, j_indices, indptr),
            shape=(len(indptr) - 1, len(vocabulary)),
            dtype=self.dtype,
        )
        X.sort_indices()
        return vocabulary, X
```

### Mã thư viện: `CountVectorizer.fit_transform`

Nguồn local: `.venv/Lib/site-packages/sklearn/base.py`, bắt đầu dòng 1330.

```python
    @_fit_context(prefer_skip_nested_validation=True)
    def fit_transform(self, raw_documents, y=None):
        """Learn the vocabulary dictionary and return document-term matrix.

        This is equivalent to fit followed by transform, but more efficiently
        implemented.

        Parameters
        ----------
        raw_documents : iterable
            An iterable which generates either str, unicode or file objects.

        y : None
            This parameter is ignored.

        Returns
        -------
        X : array of shape (n_samples, n_features)
            Document-term matrix.
        """
        # We intentionally don't call the transform method to make
        # fit_transform overridable without unwanted side effects in
        # TfidfVectorizer.
        if isinstance(raw_documents, str):
            raise ValueError(
                "Iterable over raw text documents expected, string object received."
            )

        self._validate_ngram_range()
        self._warn_for_unused_params()
        self._validate_vocabulary()
        max_df = self.max_df
        min_df = self.min_df
        max_features = self.max_features

        if self.fixed_vocabulary_ and self.lowercase:
            for term in self.vocabulary:
                if any(map(str.isupper, term)):
                    warnings.warn(
                        "Upper case characters found in"
                        " vocabulary while 'lowercase'"
                        " is True. These entries will not"
                        " be matched with any documents"
                    )
                    break

        vocabulary, X = self._count_vocab(raw_documents, self.fixed_vocabulary_)

        if self.binary:
            X.data.fill(1)

        if not self.fixed_vocabulary_:
            n_doc = X.shape[0]
            max_doc_count = max_df if isinstance(max_df, Integral) else max_df * n_doc
            min_doc_count = min_df if isinstance(min_df, Integral) else min_df * n_doc
            if max_doc_count < min_doc_count:
                raise ValueError("max_df corresponds to < documents than min_df")
            if max_features is not None:
                X = self._sort_features(X, vocabulary)
            X = self._limit_features(
                X, vocabulary, max_doc_count, min_doc_count, max_features
            )
            if max_features is None:
                X = self._sort_features(X, vocabulary)
            self.vocabulary_ = vocabulary

        return X
```

### Mã thư viện: `CountVectorizer.transform`

Nguồn local: `.venv/Lib/site-packages/sklearn/feature_extraction/text.py`, bắt đầu dòng 1398.

```python
    def transform(self, raw_documents):
        """Transform documents to document-term matrix.

        Extract token counts out of raw text documents using the vocabulary
        fitted with fit or the one provided to the constructor.

        Parameters
        ----------
        raw_documents : iterable
            An iterable which generates either str, unicode or file objects.

        Returns
        -------
        X : sparse matrix of shape (n_samples, n_features)
            Document-term matrix.
        """
        if isinstance(raw_documents, str):
            raise ValueError(
                "Iterable over raw text documents expected, string object received."
            )
        self._check_vocabulary()

        # use the same matrix-building strategy as fit_transform
        _, X = self._count_vocab(raw_documents, fixed_vocab=True)
        if self.binary:
            X.data.fill(1)
        return X
```

### Đọc code CountVectorizer vừa trích

- `defaultdict` cấp chỉ số cho token mới; ở chế độ `fixed_vocab=True` chỉ tra từ điển cũ.
- `feature_counter` gom số đếm trong một câu; `KeyError` ở từ chưa biết được bỏ qua khi transform.
- `j_indices`, `values`, `indptr` là các thành phần để tạo CSR, không phải nhãn.
- `X.sort_indices()` sắp chỉ số cột trong mỗi hàng CSR. Số lần lặp của một đặc trưng trong câu đã được cộng trong `feature_counter` trước đó.
- `max_df`, `min_df`: ngưỡng DF; project không giới hạn bằng `max_features`.
- `self.vocabulary_ = vocabulary` giữ ánh xạ đã học cho các câu mới.
- `check_is_fitted` ngăn transform khi chưa học từ vựng.

## 3. TfidfVectorizer — từ số đếm sang trọng số

TfidfVectorizer kế thừa CountVectorizer. Vì vậy `super().fit_transform` làm toàn bộ phần học từ vựng và đếm như trên; sau đó nó dùng **TfidfTransformer** để học IDF và đổi ma trận đếm.

Với cấu hình project:

- TF: số đếm c>0 đổi thành `1 + ln(c)` do `sublinear_tf=True`; phần tử 0 vẫn là 0.
- IDF: `ln((1+n)/(1+df_j)) + 1` do `smooth_idf=True`; n là số câu train của fold, df_j là số câu chứa đặc trưng j.
- Trọng số trước chuẩn hóa: TF × IDF.
- `norm='l2'`: chia mỗi hàng cho căn tổng bình phương của hàng. Hàng không có đặc trưng hợp lệ vẫn là hàng 0.

Ví dụ với n=3, một từ có df=2 và count=2 trong câu: IDF=`ln(4/3)+1≈1,2877`, TF=`1+ln(2)≈1,6931`; tích khoảng 2,1802. Đây chưa phải giá trị cuối vì còn chuẩn hóa toàn hàng.

Ở validation/test chỉ dùng IDF đã học trên train. Không tính IDF lại từ test. IDF dùng độ phổ biến của từ, không dùng nhãn spam/ham.

### Mã thư viện: `TfidfVectorizer.fit_transform`

Nguồn local: `.venv/Lib/site-packages/sklearn/feature_extraction/text.py`, bắt đầu dòng 2078.

```python
    def fit_transform(self, raw_documents, y=None):
        """Learn vocabulary and idf, return document-term matrix.

        This is equivalent to fit followed by transform, but more efficiently
        implemented.

        Parameters
        ----------
        raw_documents : iterable
            An iterable which generates either str, unicode or file objects.

        y : None
            This parameter is ignored.

        Returns
        -------
        X : sparse matrix of (n_samples, n_features)
            Tf-idf-weighted document-term matrix.
        """
        self._check_params()
        self._tfidf = TfidfTransformer(
            norm=self.norm,
            use_idf=self.use_idf,
            smooth_idf=self.smooth_idf,
            sublinear_tf=self.sublinear_tf,
        )
        X = super().fit_transform(raw_documents)
        self._tfidf.fit(X)
        # X is already a transformed view of raw_documents so
        # we set copy to False
        return self._tfidf.transform(X, copy=False)
```

### Mã thư viện: `TfidfVectorizer.transform`

Nguồn local: `.venv/Lib/site-packages/sklearn/feature_extraction/text.py`, bắt đầu dòng 2110.

```python
    def transform(self, raw_documents):
        """Transform documents to document-term matrix.

        Uses the vocabulary and document frequencies (df) learned by fit (or
        fit_transform).

        Parameters
        ----------
        raw_documents : iterable
            An iterable which generates either str, unicode or file objects.

        Returns
        -------
        X : sparse matrix of (n_samples, n_features)
            Tf-idf-weighted document-term matrix.
        """
        check_is_fitted(self, msg="The TF-IDF vectorizer is not fitted")

        X = super().transform(raw_documents)
        return self._tfidf.transform(X, copy=False)
```

### Mã thư viện: `TfidfTransformer.fit`

Nguồn local: `.venv/Lib/site-packages/sklearn/base.py`, bắt đầu dòng 1634.

```python
    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None):
        """Learn the idf vector (global term weights).

        Parameters
        ----------
        X : sparse matrix of shape (n_samples, n_features)
            A matrix of term/token counts.

        y : None
            This parameter is not needed to compute tf-idf.

        Returns
        -------
        self : object
            Fitted transformer.
        """
        # large sparse data is not supported for 32bit platforms because
        # _document_frequency uses np.bincount which works on arrays of
        # dtype NPY_INTP which is int32 for 32bit platforms. See #20923
        X = validate_data(
            self, X, accept_sparse=("csr", "csc"), accept_large_sparse=not _IS_32BIT
        )
        if not sp.issparse(X):
            X = sp.csr_matrix(X)
        dtype = X.dtype if X.dtype in (np.float64, np.float32) else np.float64

        if self.use_idf:
            n_samples, _ = X.shape
            df = _document_frequency(X)
            df = df.astype(dtype, copy=False)

            # perform idf smoothing if required
            df += float(self.smooth_idf)
            n_samples += int(self.smooth_idf)

            # log+1 instead of log makes sure terms with zero idf don't get
            # suppressed entirely.
            # Force the dtype of `idf_` to be the same as `df`. In NumPy < 2, the dtype
            # was depending on the value of `n_samples`.
            self.idf_ = np.full_like(df, fill_value=n_samples, dtype=dtype)
            self.idf_ /= df
            # `np.log` preserves the dtype of `df` and thus `dtype`.
            np.log(self.idf_, out=self.idf_)
            self.idf_ += 1.0

        return self
```

### Mã thư viện: `TfidfTransformer.transform`

Nguồn local: `.venv/Lib/site-packages/sklearn/feature_extraction/text.py`, bắt đầu dòng 1682.

```python
    def transform(self, X, copy=True):
        """Transform a count matrix to a tf or tf-idf representation.

        Parameters
        ----------
        X : sparse matrix of (n_samples, n_features)
            A matrix of term/token counts.

        copy : bool, default=True
            Whether to copy X and operate on the copy or perform in-place
            operations. `copy=False` will only be effective with CSR sparse matrix.

        Returns
        -------
        vectors : sparse matrix of shape (n_samples, n_features)
            Tf-idf-weighted document-term matrix.
        """
        check_is_fitted(self)
        X = validate_data(
            self,
            X,
            accept_sparse="csr",
            dtype=[np.float64, np.float32],
            copy=copy,
            reset=False,
        )
        if not sp.issparse(X):
            X = sp.csr_matrix(X, dtype=X.dtype)

        if self.sublinear_tf:
            np.log(X.data, X.data)
            X.data += 1.0

        if hasattr(self, "idf_"):
            # the columns of X (CSR matrix) can be accessed with `X.indices `and
            # multiplied with the corresponding `idf` value
            X.data *= self.idf_[X.indices]

        if self.norm is not None:
            X = normalize(X, norm=self.norm, copy=False)

        return X
```

### Đọc code TF-IDF vừa trích

`_document_frequency(X)` đếm số hàng có từng cột. `df += float(self.smooth_idf)` cộng 1 vào df khi bật smoothing; n cũng cộng 1. `np.full_like` tạo mảng cùng kích thước; phép chia rồi `np.log` thực hiện đúng công thức IDF. `X.data` chỉ gồm phần tử khác 0 nên log không tác động lên số 0. `self.idf_[X.indices]` lấy IDF đúng cột của từng phần tử. `normalize(..., norm='l2')` chuẩn hóa hàng. Các thao tác `copy=False`, `out=...` tái sử dụng bộ nhớ, không thay công thức.

## 4. MultinomialNB — học thống kê đặc trưng theo nhãn

Project dùng MultinomialNB cho cả BoW và TF-IDF vì cả hai không âm. TF-IDF không phải số đếm nguyên đúng nghĩa mô hình sinh multinomial; dùng trọng số không âm với NB là một cách áp dụng thực tế, không biến TF-IDF thành xác suất.

Các bước học:

1. Kiểm tra X/y và mã hóa nhãn thành ma trận chỉ báo Y.
2. Cộng vector của tất cả câu cùng lớp: thu `feature_count_[c,j]`.
3. Đếm số câu mỗi lớp: `class_count_`.
4. Làm trơn: `P(t_j|c)=(feature_count_[c,j]+alpha)/(tổng trọng số lớp c + alpha*d)`.
5. Lưu log của các xác suất trong `feature_log_prob_`.
6. Với `fit_prior=True` mặc định, học prior từ tỷ lệ mẫu: `P(c)=n_c/n` rồi lấy log trong `class_log_prior_`.

Dự đoán câu x: tính `score_c = log P(c) + Σ_j x_j log P(t_j|c)`. Lớp có score lớn nhất thắng. Dùng log tránh nhân nhiều số rất nhỏ. `predict_proba` trừ `logsumexp` rồi lấy exp để chuẩn hóa hai score thành tổng 1. Đó là xác suất ước lượng của mô hình, có thể chưa được hiệu chỉnh.

“Naive” là giả định độc lập có điều kiện giữa các đặc trưng khi biết lớp; không có nghĩa các từ trong tiếng Việt thực sự độc lập. Alpha lớn làm phân bố bớt cực đoan; không có một alpha tốt nhất cho mọi dataset. Project thử 0,1; 0,3; 1,0 bằng CV.

### Mã thư viện: `_BaseDiscreteNB.fit`

Nguồn local: `.venv/Lib/site-packages/sklearn/base.py`, bắt đầu dòng 714.

```python
    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y, sample_weight=None):
        """Fit Naive Bayes classifier according to X, y.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vectors, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        y : array-like of shape (n_samples,)
            Target values.

        sample_weight : array-like of shape (n_samples,), default=None
            Weights applied to individual samples (1. for unweighted).

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        X, y = self._check_X_y(X, y)
        _, n_features = X.shape

        labelbin = LabelBinarizer()
        Y = labelbin.fit_transform(y)
        self.classes_ = labelbin.classes_
        if Y.shape[1] == 1:
            if len(self.classes_) == 2:
                Y = np.concatenate((1 - Y, Y), axis=1)
            else:  # degenerate case: just one class
                Y = np.ones_like(Y)

        # LabelBinarizer().fit_transform() returns arrays with dtype=np.int64.
        # We convert it to np.float64 to support sample_weight consistently;
        # this means we also don't have to cast X to floating point
        if sample_weight is not None:
            Y = Y.astype(np.float64, copy=False)
            sample_weight = _check_sample_weight(sample_weight, X)
            sample_weight = np.atleast_2d(sample_weight)
            Y *= sample_weight.T

        class_prior = self.class_prior

        # Count raw events from data before updating the class log prior
        # and feature log probas
        n_classes = Y.shape[1]
        self._init_counters(n_classes, n_features)
        self._count(X, Y)
        alpha = self._check_alpha()
        self._update_feature_log_prob(alpha)
        self._update_class_log_prior(class_prior=class_prior)
        return self
```

### Mã thư viện: `_BaseDiscreteNB._update_class_log_prior`

Nguồn local: `.venv/Lib/site-packages/sklearn/naive_bayes.py`, bắt đầu dòng 583.

```python
    def _update_class_log_prior(self, class_prior=None):
        """Update class log priors.

        The class log priors are based on `class_prior`, class count or the
        number of classes. This method is called each time `fit` or
        `partial_fit` update the model.
        """
        n_classes = len(self.classes_)
        if class_prior is not None:
            if len(class_prior) != n_classes:
                raise ValueError("Number of priors must match number of classes.")
            self.class_log_prior_ = np.log(class_prior)
        elif self.fit_prior:
            with warnings.catch_warnings():
                # silence the warning when count is 0 because class was not yet
                # observed
                warnings.simplefilter("ignore", RuntimeWarning)
                log_class_count = np.log(self.class_count_)

            # empirical prior, with sample_weight taken into account
            self.class_log_prior_ = log_class_count - np.log(self.class_count_.sum())
        else:
            self.class_log_prior_ = np.full(n_classes, -np.log(n_classes))
```

### Mã thư viện: `MultinomialNB._count`

Nguồn local: `.venv/Lib/site-packages/sklearn/naive_bayes.py`, bắt đầu dòng 887.

```python
    def _count(self, X, Y):
        """Count and smooth feature occurrences."""
        check_non_negative(X, "MultinomialNB (input X)")
        self.feature_count_ += safe_sparse_dot(Y.T, X)
        self.class_count_ += Y.sum(axis=0)
```

### Mã thư viện: `MultinomialNB._update_feature_log_prob`

Nguồn local: `.venv/Lib/site-packages/sklearn/naive_bayes.py`, bắt đầu dòng 893.

```python
    def _update_feature_log_prob(self, alpha):
        """Apply smoothing to raw counts and recompute log probabilities"""
        smoothed_fc = self.feature_count_ + alpha
        smoothed_cc = smoothed_fc.sum(axis=1)

        self.feature_log_prob_ = np.log(smoothed_fc) - np.log(
            smoothed_cc.reshape(-1, 1)
        )
```

### Mã thư viện: `MultinomialNB._joint_log_likelihood`

Nguồn local: `.venv/Lib/site-packages/sklearn/naive_bayes.py`, bắt đầu dòng 902.

```python
    def _joint_log_likelihood(self, X):
        """Calculate the posterior log probability of the samples X"""
        return safe_sparse_dot(X, self.feature_log_prob_.T) + self.class_log_prior_
```

### Mã thư viện: `_BaseNB.predict_proba`

Nguồn local: `.venv/Lib/site-packages/sklearn/naive_bayes.py`, bắt đầu dòng 132.

```python
    def predict_proba(self, X):
        """
        Return probability estimates for the test vector X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        C : array-like of shape (n_samples, n_classes)
            Returns the probability of the samples for each class in
            the model. The columns correspond to the classes in sorted
            order, as they appear in the attribute :term:`classes_`.
        """
        return np.exp(self.predict_log_proba(X))
```

### Đọc code NB vừa trích

`LabelBinarizer` tạo Y; trong bài hai lớp, thư viện bảo đảm có cột tương ứng mỗi lớp. `_init_counters` tạo mảng đếm 0; `_count` cập nhật chúng. `safe_sparse_dot(Y.T, X)` cộng đặc trưng theo lớp bằng nhân ma trận, hỗ trợ dữ liệu thưa. `_check_alpha` kiểm mức làm trơn; `_update_feature_log_prob` cộng alpha, cộng tổng theo hàng rồi trừ log mẫu số. `axis=1` nghĩa là cộng dọc theo đặc trưng của mỗi lớp. `np.newaxis` giúp mảng mẫu số khớp số chiều để phép trừ áp cho mọi cột. `predict_proba` gọi `_joint_log_likelihood` rồi chuẩn hóa; nó không fit lại.

## 5. LinearSVC — học ranh giới tuyến tính

Đầu vào là vector X và nhãn y. Với bài hai lớp, mô hình học một vector w và b: `s(x)=w·x+b`. Theo `classes_=[0,1]`, score dương nghiêng spam; score âm nghiêng ham. Score không phải xác suất; độ lớn còn phụ thuộc biểu diễn và C.

Project dùng mặc định penalty L2, loss **squared_hinge**, `dual=False`, `max_iter=20000`. Diễn giải mục tiêu: giảm `1/2 ||w||² + C Σ_i a_i max(0, 1-y'_i s(x_i))²`, với nhãn nội bộ y' là −1/+1 và a_i là trọng số lớp. LIBLINEAR còn xử lý intercept qua đặc trưng tổng hợp theo `intercept_scaling`; intercept vì vậy cũng chịu regularization trong cách triển khai này.

Các bước học: kiểm X/y → xác định hai lớp → chọn solver → chuyển dữ liệu cho LIBLINEAR → lặp tối ưu trọng số để giảm mục tiêu → dừng khi đạt độ chính xác tối ưu hoặc max_iter → nhận `coef_`, `intercept_`, `n_iter_`. Với `dual=False`, đây là bài toán primal; không phải tự chia hai nhóm rồi tìm trung bình, cũng không phải duyệt mọi đường thẳng. `C` lớn tăng mức ưu tiên giảm lỗi/margin violation so với regularization; không bảo đảm điểm tốt hơn.

`fit` dưới đây chỉ là lớp Python điều phối; phép tối ưu lõi nằm trong `_fit_liblinear` → wrapper `_liblinear.train_wrap` → mã native LIBLINEAR. Ctrl+Click không thể biến phần đó thành một vòng for Python ngắn. Phần cần giải thích khi bảo vệ là mục tiêu tối ưu, dữ liệu vào, tham số và trọng số đầu ra; không nói code Python này tự tính nghiệm bằng công thức đóng.

### Mã thư viện: `LinearSVC.fit`

Nguồn local: `.venv/Lib/site-packages/sklearn/base.py`, bắt đầu dòng 280.

```python
    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y, sample_weight=None):
        """Fit the model according to the given training data.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vector, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        y : array-like of shape (n_samples,)
            Target vector relative to X.

        sample_weight : array-like of shape (n_samples,), default=None
            Array of weights that are assigned to individual
            samples. If not provided,
            then each sample is given unit weight.

            .. versionadded:: 0.18

        Returns
        -------
        self : object
            An instance of the estimator.
        """
        X, y = validate_data(
            self,
            X,
            y,
            accept_sparse="csr",
            dtype=np.float64,
            order="C",
            accept_large_sparse=False,
        )
        check_classification_targets(y)
        self.classes_ = np.unique(y)

        _dual = _validate_dual_parameter(
            self.dual, self.loss, self.penalty, self.multi_class, X
        )

        self.coef_, self.intercept_, n_iter_ = _fit_liblinear(
            X,
            y,
            self.C,
            self.fit_intercept,
            self.intercept_scaling,
            self.class_weight,
            self.penalty,
            _dual,
            self.verbose,
            self.max_iter,
            self.tol,
            self.random_state,
            self.multi_class,
            self.loss,
            sample_weight=sample_weight,
        )
        # Backward compatibility: _fit_liblinear is used both by LinearSVC/R
        # and LogisticRegression but LogisticRegression sets a structured
        # `n_iter_` attribute with information about the underlying OvR fits
        # while LinearSVC/R only reports the maximum value.
        self.n_iter_ = n_iter_.max().item()

        if self.multi_class == "crammer_singer" and len(self.classes_) == 2:
            self.coef_ = (self.coef_[1] - self.coef_[0]).reshape(1, -1)
            if self.fit_intercept:
                intercept = self.intercept_[1] - self.intercept_[0]
                self.intercept_ = np.array([intercept])

        return self
```

### Những phương thức kế thừa cần hiểu

`decision_function(X)` kiểm đã fit và số cột, tính tích ma trận X với `coef_.T`, cộng `intercept_`; trong bài nhị phân trả một score mỗi câu. `predict(X)` dùng dấu score để chọn `classes_`. Project gọi `decision_function` trong `evaluation.predict_record` để có thể đổi ngưỡng cho S3. LinearSVC không có `predict_proba`. `clone` không sao chép trọng số đã học: nó tạo model mới cùng siêu tham số.

## 6. GridSearchCV — điều phối thử cấu hình và chọn theo CV

Nó không phải thuật toán phân loại thứ ba. Nó nhận estimator là Pipeline, `param_grid`, năm folds, scoring, `refit='f1'`, n_jobs. `ParameterGrid` khai triển tích Descartes các lựa chọn. Với từng cấu hình và từng fold, thư viện clone estimator, đặt tham số, lấy train/validation đúng chỉ số, fit trên train rồi tính các score trên validation. Cuối cùng lấy trung bình, xếp hạng F1 và refit cấu hình thắng trên toàn train bàn giao.

Các thành phần nội bộ:

| Thành phần | Nhiệm vụ |
|---|---|
| `_get_scorers` | Đổi tên precision/recall/f1 thành bộ chấm điểm |
| `check_cv` và `cv.split` | Dùng các cặp chỉ số fold đã cung cấp |
| `clone` | Tạo bản chưa học, giữ cấu hình |
| `_fit_and_score` | Lấy train/validation, set_params, fit, chấm và ghi thời gian |
| `Parallel`, `delayed` | Chạy nhiều công việc khi n_jobs>1 |
| `_format_results` | Gom điểm các fold thành bảng và thống kê |
| `_select_best_index` | Chọn dòng theo tiêu chí refit |
| `best_estimator_.fit(X,y)` | Học lại cấu hình thắng trên toàn train |

`cv_results_` có một dòng mỗi cấu hình: params, split0_test_f1…split4_test_f1, mean_test_f1, std_test_f1, rank_test_f1 và các chỉ số khác. Chữ **test trong tên nội bộ này là validation của CV**, không phải tập test độc lập của project. `best_score_` là mean CV F1 khi refit='f1'; `best_params_` là siêu tham số thắng; `best_estimator_` là pipeline đã refit. `error_score='raise'` khiến lỗi fit được báo ra thay vì âm thầm ghi NaN.

### Mã thư viện: `GridSearchCV._run_search`

Nguồn local: `.venv/Lib/site-packages/sklearn/model_selection/_search.py`, bắt đầu dòng 1569.

```python
    def _run_search(self, evaluate_candidates):
        """Search all candidates in param_grid"""
        evaluate_candidates(ParameterGrid(self.param_grid))
```

### Mã thư viện: `BaseSearchCV.fit`

Nguồn local: `.venv/Lib/site-packages/sklearn/base.py`, bắt đầu dòng 887.

```python
    @_fit_context(
        # *SearchCV.estimator is not validated yet
        prefer_skip_nested_validation=False
    )
    def fit(self, X, y=None, **params):
        """Run fit with all sets of parameters.

        Parameters
        ----------

        X : array-like of shape (n_samples, n_features) or (n_samples, n_samples)
            Training vectors, where `n_samples` is the number of samples and
            `n_features` is the number of features. For precomputed kernel or
            distance matrix, the expected shape of X is (n_samples, n_samples).

        y : array-like of shape (n_samples, n_output) \
            or (n_samples,), default=None
            Target relative to X for classification or regression;
            None for unsupervised learning.

        **params : dict of str -> object
            Parameters passed to the ``fit`` method of the estimator, the scorer,
            and the CV splitter.

            If a fit parameter is an array-like whose length is equal to
            `num_samples` then it will be split by cross-validation along with
            `X` and `y`. For example, the :term:`sample_weight` parameter is
            split because `len(sample_weights) = len(X)`. However, this behavior
            does not apply to `groups` which is passed to the splitter configured
            via the `cv` parameter of the constructor. Thus, `groups` is used
            *to perform the split* and determines which samples are
            assigned to the each side of the a split.

        Returns
        -------
        self : object
            Instance of fitted estimator.
        """
        estimator = self.estimator
        scorers, refit_metric = self._get_scorers()

        X, y = indexable(X, y)
        params = _check_method_params(X, params=params)

        routed_params = self._get_routed_params_for_fit(params)

        cv_orig = check_cv(self.cv, y, classifier=is_classifier(estimator))
        n_splits = cv_orig.get_n_splits(X, y, **routed_params.splitter.split)

        base_estimator = clone(self.estimator)

        parallel = Parallel(n_jobs=self.n_jobs, pre_dispatch=self.pre_dispatch)

        fit_and_score_kwargs = dict(
            scorer=scorers,
            fit_params=routed_params.estimator.fit,
            score_params=routed_params.scorer.score,
            return_train_score=self.return_train_score,
            return_n_test_samples=True,
            return_times=True,
            return_parameters=False,
            error_score=self.error_score,
            verbose=self.verbose,
        )
        results = {}
        with parallel:
            all_candidate_params = []
            all_out = []
            all_more_results = defaultdict(list)

            def evaluate_candidates(candidate_params, cv=None, more_results=None):
                cv = cv or cv_orig
                candidate_params = list(candidate_params)
                n_candidates = len(candidate_params)

                if self.verbose > 0:
                    print(
                        "Fitting {0} folds for each of {1} candidates,"
                        " totalling {2} fits".format(
                            n_splits, n_candidates, n_candidates * n_splits
                        )
                    )

                out = parallel(
                    delayed(_fit_and_score)(
                        clone(base_estimator),
                        X,
                        y,
                        train=train,
                        test=test,
                        parameters=parameters,
                        split_progress=(split_idx, n_splits),
                        candidate_progress=(cand_idx, n_candidates),
                        **fit_and_score_kwargs,
                    )
                    for (cand_idx, parameters), (split_idx, (train, test)) in product(
                        enumerate(candidate_params),
                        enumerate(cv.split(X, y, **routed_params.splitter.split)),
                    )
                )

                if len(out) < 1:
                    raise ValueError(
                        "No fits were performed. "
                        "Was the CV iterator empty? "
                        "Were there no candidates?"
                    )
                elif len(out) != n_candidates * n_splits:
                    raise ValueError(
                        "cv.split and cv.get_n_splits returned "
                        "inconsistent results. Expected {} "
                        "splits, got {}".format(n_splits, len(out) // n_candidates)
                    )

                _warn_or_raise_about_fit_failures(out, self.error_score)

                # For callable self.scoring, the return type is only know after
                # calling. If the return type is a dictionary, the error scores
                # can now be inserted with the correct key. The type checking
                # of out will be done in `_insert_error_scores`.
                if callable(self.scoring):
                    _insert_error_scores(out, self.error_score)

                all_candidate_params.extend(candidate_params)
                all_out.extend(out)

                if more_results is not None:
                    for key, value in more_results.items():
                        all_more_results[key].extend(value)

                nonlocal results
                results = self._format_results(
                    all_candidate_params, n_splits, all_out, all_more_results
                )

                return results

            self._run_search(evaluate_candidates)

            # multimetric is determined here because in the case of a callable
            # self.scoring the return type is only known after calling
            first_test_score = all_out[0]["test_scores"]
            self.multimetric_ = isinstance(first_test_score, dict)

            # check refit_metric now for a callable scorer that is multimetric
            if callable(self.scoring) and self.multimetric_:
                self._check_refit_for_multimetric(first_test_score)
                refit_metric = self.refit

        # For multi-metric evaluation, store the best_index_, best_params_ and
        # best_score_ iff refit is one of the scorer names
        # In single metric evaluation, refit_metric is "score"
        if self.refit or not self.multimetric_:
            self.best_index_ = self._select_best_index(
                self.refit, refit_metric, results
            )
            if not callable(self.refit):
                # With a non-custom callable, we can select the best score
                # based on the best index
                self.best_score_ = results[f"mean_test_{refit_metric}"][
                    self.best_index_
                ]
            self.best_params_ = results["params"][self.best_index_]

        if self.refit:
            # here we clone the estimator as well as the parameters, since
            # sometimes the parameters themselves might be estimators, e.g.
            # when we search over different estimators in a pipeline.
            # ref: https://github.com/scikit-learn/scikit-learn/pull/26786
            self.best_estimator_ = clone(base_estimator).set_params(
                **clone(self.best_params_, safe=False)
            )

            refit_start_time = time.time()
            if y is not None:
                self.best_estimator_.fit(X, y, **routed_params.estimator.fit)
            else:
                self.best_estimator_.fit(X, **routed_params.estimator.fit)
            refit_end_time = time.time()
            self.refit_time_ = refit_end_time - refit_start_time

            if hasattr(self.best_estimator_, "feature_names_in_"):
                self.feature_names_in_ = self.best_estimator_.feature_names_in_

        # Store the only scorer not as a dict for single metric evaluation
        if isinstance(scorers, _MultimetricScorer):
            self.scorer_ = scorers._scorers
        else:
            self.scorer_ = scorers

        self.cv_results_ = results
        self.n_splits_ = n_splits

        return self
```

### Ví dụ ngắn và cách đọc khối fit dài

Grid gồm hai n-gram × ba alpha = 6 cấu hình NB. Với năm folds: 30 lượt fit; mỗi lượt chỉ học bốn phần và chấm phần còn lại. Ví dụ F1 của cấu hình A qua năm fold là 0,6; 0,7; 0,6; 0,7; 0,65, trung bình là 0,65. Cấu hình B có trung bình 0,67 thì B thắng, dù A có thể thắng ở một fold. Sau đó fit B một lần trên toàn train. Test vẫn chưa được dùng.

Trong `BaseSearchCV.fit`, các dòng đầu chuẩn bị scorer, X/y và tham số; `with parallel` tạo nơi chạy các công việc; closure `evaluate_candidates` tạo danh sách candidate×fold và gọi `_fit_and_score`; `self._run_search` ở GridSearchCV gọi closure đó với ParameterGrid. Đoạn cuối xác định `best_index_`, gán params/score và refit. Những nhánh `callable(refit)`, metadata routing, nhiều metric đặc biệt hỗ trợ cách dùng rộng hơn, không làm tăng số cấu hình của project.

## 7. Tự lần theo bằng Ctrl+Click

Chọn interpreter `.venv` trong VS Code trước. Tại file project: `make_vectorizer` → CountVectorizer/TfidfVectorizer → `fit_transform`/`transform`; `make_classifier` → MultinomialNB/LinearSVC → `fit`; `fit_experiment` → GridSearchCV → `_run_search` và BaseSearchCV.fit. Nếu chỉ thấy constructor hoặc chữ ký tham số, tìm tiếp phương thức `fit` và lớp cha theo tên ghi trong tài liệu này.

Nguồn chính là các file thư viện local được ghi ngay trước mỗi đoạn trích. Tham khảo API chính thức: [CountVectorizer](https://scikit-learn.org/1.6/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html), [TfidfVectorizer](https://scikit-learn.org/1.6/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html), [MultinomialNB](https://scikit-learn.org/1.6/modules/generated/sklearn.naive_bayes.MultinomialNB.html), [LinearSVC](https://scikit-learn.org/1.6/modules/generated/sklearn.svm.LinearSVC.html), [GridSearchCV](https://scikit-learn.org/1.6/modules/generated/sklearn.model_selection.GridSearchCV.html).
