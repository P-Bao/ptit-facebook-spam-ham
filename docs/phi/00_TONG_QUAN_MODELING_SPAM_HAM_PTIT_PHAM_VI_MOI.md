# 00 — TỔNG QUAN XÂY DỰNG MÔ HÌNH SPAM/HAM PTIT

> **Tài liệu học phiên bản cập nhật (13/11/2026)**  
> **Phạm vi chính do bạn phụ trách:** nhận dataset **đã có nhãn và do nhóm chuẩn bị**, chia Train/Test và fold, BoW/TF-IDF, Naive Bayes/SVM, CV, xử lý lệch lớp, đánh giá, lưu và dự đoán bình luận mới. Khâu thu thập, làm sạch, gán nhãn và tạo token do thành viên khác phụ trách; bạn chỉ cần hiểu giao diện nhận/bàn giao dữ liệu.  
> Đây là **bản đồ tổng quan**. Mỗi bước sẽ có file Markdown riêng để đào sâu lý thuyết, giải thích code theo từng dòng, ví dụ và câu hỏi bảo vệ.

## 0. RANH GIỚI PHẦN VIỆC VÀ LIÊN KẾT VỚI CÁC NOTEBOOK KHÁC (CẬP NHẬT)

**Phần việc trực tiếp:** nhận dataset `labeled_v7.csv` đã gán nhãn (Ham=0, Spam=1) và các cột token chuẩn bị từ Notebook 02; dùng `training_utils.py` chia Train/Test + grouped folds; `text_features.py` và `classifiers.py` chuẩn bị bốn cặp đặc trưng/phân loại; `experiments.py` fit/CV; `imbalance.py` xử lý lệch lớp; `evaluation.py` chấm Test; `predict.py` suy luận với bundle đã lưu.

**Phần liên quan cần hiểu sơ lược (không phụ trách):**
- **Notebook 01** theo mô tả của nhóm: thu thập bình luận → `clean_for_labeling.py` làm sạch và **loại trùng bản ghi** → gán nhãn → bàn giao CSV. Chưa được cung cấp file Notebook 01 nên không xác nhận được các cell gán nhãn cụ thể.
- **Chuẩn bị đầu vào cho model** trong Notebook 02: `preprocess_for_model.py` chuyển `cleaned_text` và văn bản gốc thành chuỗi token; có hai nhánh giữ/bỏ stopwords. Việc này **không huấn luyện**, không thay nhãn và không xóa bản ghi trùng.
- **Notebook 03 / viBERT** theo mô tả của nhóm: mô hình hiện đại do thành viên khác phụ trách. Muốn so sánh công bằng, cần dùng cùng bản dataset/snapshot, nhãn và cùng **các dòng Train/Test**; khi đối chiếu điểm CV nên dùng cùng nhóm fold nếu phù hợp. Chưa có Notebook 03 để kiểm tra thực tế.

**Hai kiểu “trùng” khác nhau:**

| Vị trí | Quy tắc | Có xóa dòng? | Mục tiêu |
|---|---|---|---|
| `clean_for_labeling.clean_facebook_comments()` | Khóa chính xác `(post_id, comment_type, text)` | **Có**: bỏ dòng trùng cùng bộ khóa | Tránh bản ghi lặp khi gộp nguồn trước gán nhãn |
| `training_utils.norm_key()` + `bucket()` | Chuẩn hóa nội dung (`NFC`, viết thường, xóa dấu câu, đưa số về 0) rồi băm khóa | **Không**: giữ tất cả dòng và đặt dòng cùng khóa vào cùng phía/fold | Hạn chế trùng/gần trùng theo dạng chữ giữa Train–Test hoặc train–validation |

**Ví dụ:** Hai bình luận cùng nội dung “Nhận làm thuê 100k” nhưng ở hai `post_id` khác nhau không bị coi là trùng theo khóa của `clean_for_labeling`, song có thể vào cùng nhóm chia ở `training_utils`; “Nhận làm thuê 100k” và “Nhận làm thuê 200k” có thể chung `norm_key` nhờ chuẩn hóa số.

**Ý nghĩa `bucket`:** `md5(f"{salt}|{norm_key}") % n` gán nhóm xác định, cùng nội dung chuẩn hóa sẽ có cùng nhóm; `salt="test"` dùng chia Train/Test, `salt="cv"` dùng 5 fold trong Train. Điều này giảm rò rỉ dữ liệu và cho phép các mô hình truyền thống, viBERT **áp dụng cùng phân vùng nếu chúng dùng cùng quy trình/đầu vào**. Bucket không tự đảm bảo tỷ lệ lớp cân bằng, không phải phân loại Spam/Ham và không có chức năng bảo mật.

**Chỗ bắt đầu trách nhiệm chuyên sâu:** Nhận `X` là chuỗi token và `y` là nhãn, rồi theo `training_utils → text_features + classifiers → experiments → imbalance → evaluation → predict`. Khi bảo vệ, không nhận mình là người cào, gán nhãn hoặc huấn luyện viBERT.

---

## 1. Tôi đã lấy những nguồn nào làm chuẩn?

**Bộ nguồn MỚI** (không dùng phiên bản `ProjectTextVectorizer` cũ):

| File được gửi | Vị trí logic trong dự án | Vai trò |
|---|---|---|
| `02_thuc_nghiem_va_demo(1).ipynb` | `notebooks/02_thuc_nghiem_va_demo.ipynb` | Chương trình điều phối, thực thi thí nghiệm, kết quả, báo cáo và lưu mô hình |
| `training_utils(1).py` | `pipeline/modeling/training_utils.py` | Chia tập có nhóm và tạo 5 fold |
| `text_features(1).py` | `pipeline/modeling/text_features.py` | Tạo **CountVectorizer / TfidfVectorizer của scikit-learn**, KHÔNG tự tính vector bằng lớp tự viết |
| `classifiers(1).py` | `pipeline/modeling/classifiers.py` | Tạo MultinomialNB / LinearSVC và lưới tham số |
| `experiments(1).py` | `pipeline/modeling/experiments.py` | Ghép Pipeline, GridSearchCV, lựa chọn và refit mô hình |
| `imbalance(1).py` | `pipeline/modeling/imbalance.py` | S1: balanced; S2: undersampling; S3: threshold tuning |
| `evaluation(1).py` | `pipeline/modeling/evaluation.py` | Lấy score/nhãn dự đoán, tính chỉ số Test và bootstrap |
| `predict(1).py` | **`pipeline/modeling/predict.py`** | Suy luận cho bình luận thô với model bundle đã lưu/nạp |
| `README(1).md` | README của thư mục modeling | Chỉ dẫn luồng module, điểm bắt đầu và cách đọc |
| `PHI_HIEU_4_LOP_SCIKIT_LEARN.md` | `docs/team/PHI_HIEU_4_LOP_SCIKIT_LEARN.md` | Giải thích bốn lớp thư viện, vị trí phương thức và cách chúng hoạt động trong scikit-learn **1.6.1** |

**Giới hạn kiểm chứng (cập nhật):** Đã đọc thêm `clean_for_labeling.py` và `preprocess_for_model.py`; chưa có mã nguồn Notebook 01, Notebook 03/viBERT, CSV `labeled_v7.csv`, `scripts/run_phi_pipeline.py` hoặc `.venv/` để kiểm chứng trực tiếp. Các **con số thực nghiệm** lấy từ outputs đã lưu trong Notebook 02, không phải huấn luyện lại.

### Những thay đổi so với phiên bản cũ

1. `text_features.py` dùng `CountVectorizer`/`TfidfVectorizer`; không còn `ProjectTextVectorizer` tự cài đặt.
2. `predict.py` nằm cùng `pipeline/modeling/` và Notebook import từ `pipeline.modeling.predict`.
3. Có tài liệu tham khảo riêng về bốn lớp scikit-learn; khi học sâu sẽ đối chiếu lời gọi trong project với phương thức của thư viện.
4. `training_utils.py`, `classifiers.py`, `experiments.py`, `imbalance.py`, `evaluation.py` giữ cấu trúc xử lý đang có; cần hiểu cách **các file gọi nhau**, không chạy mỗi file thành chương trình độc lập.

---

## 2. Bài toán và dữ liệu thật trong Notebook

**Bài toán:** phân loại bình luận sinh viên PTIT thành `ham` (bình luận bình thường) và `spam` (nội dung rác theo quy tắc gán nhãn dataset). Đây là **học có giám sát**, vì bình luận Train đã có nhãn.

- Đầu vào của phần việc: `data/final/labeled_v7.csv` — đã có `id`, `text`, `cleaned_text`, `label` và các metadata khác như `comment_type`, `reaction_count`.
- Mã hóa nhãn trong cell 6: `y_all = raw.label.eq("spam").astype(int).to_numpy()` → **ham = 0, spam = 1**.
- Cell 6 đọc CSV bằng `pd.read_csv`, kiểm tra ID không trùng/rỗng, `cleaned_text` không rỗng, chỉ hai nhãn, sau đó kiểm tra phân bố và các nhóm văn bản trùng.
- Cell 9 tạo hai cột đầu vào `tokens_keep_stopwords` và `tokens_remove_stopwords`, rồi lưu `data/processed/v7_preprocessed.csv`.

**Các con số đã xuất hiện trong Notebook mới:**

| Dữ liệu | Số bình luận | Số Spam | Số Ham |
|---|---:|---:|---:|
| Toàn bộ | **38.233** | **5.326** | **32.907** |
| Train | **30.759** | **4.375** | **26.384** |
| Test | **7.474** | **951** | **6.523** |

Tỷ lệ Spam toàn bộ ≈ **13,93%**, vì thế có vấn đề **mất cân bằng lớp** (Ham nhiều hơn Spam). Notebook còn thống kê **2.729 dòng thuộc các nhóm text lặp** theo một cách chuẩn hóa và **27 nhóm text có nhãn mâu thuẫn**. Đây là giới hạn dữ liệu thật, không phải lý do tự ý sửa nhãn trong phần xây dựng mô hình.

**Cần phân biệt bốn dạng dữ liệu:**

| Tên | Ý nghĩa | Ví dụ |
|---|---|---|
| `text` | Bình luận gốc | `"Em xin lịch thi với ạ!!!"` |
| `cleaned_text` | Bình luận đã làm sạch từ dataset | `"Em xin lịch thi với ạ"` |
| `tokens_keep_stopwords` | Chuỗi token phục vụ model, giữ từ dừng | `"em xin lịch thi với ạ"` (minh họa) |
| `tokens_remove_stopwords` | Chuỗi token đã bỏ một số stop words | `"xin lịch thi"` (minh họa) |

Giá trị token cụ thể phụ thuộc `preprocess()` trong `preprocess_for_model.py` (đã đọc): `clean_text` → `mask_names` → hạ chữ/giảm lặp ký tự → `underthesea.word_tokenize` → tùy chọn bỏ stopwords → thêm `emoji_token` nếu phát hiện emoji. Notebook gọi `preprocess(c, t, False/True)` để chuẩn bị hai nhánh. **Đây là phần thành viên khác phụ trách. Không được đồng nhất** `norm_key()` ở bước chia tập với `preprocess()` tạo token để train.

---

## 3. Một luồng đầu–cuối: các biến chuyển qua đâu?

```text
CSV đã có nhãn (Notebook cell 6)
  ├── raw["cleaned_text"] ───> training_utils.norm_key/bucket
  │                              └── split_train_test → train_idx, test_idx
  │                              └── grouped_folds → CV_FOLDS (trong Train)
  └── raw["text"], raw["cleaned_text"]
          └── preprocess → hai cột chuỗi token (cell 9)
                    └── TEXT[pre].iloc[train_idx] → X_train_text (CHUỖI)
                              └── experiments.fit_experiment (cell 13)
                                    ├── text_features.make_vectorizer
                                    │       └── CountVectorizer hoặc TfidfVectorizer
                                    ├── classifiers.make_classifier
                                    │       └── MultinomialNB hoặc LinearSVC
                                    ├── Pipeline([("vec",...), ("clf",...)])
                                    └── GridSearchCV.fit(X_train_text, y_train)
                                          ├── mỗi fold: vec.fit_transform → ma trận X số
                                          │             clf.fit(ma trận, nhãn)
                                          ├── validation: vec.transform → clf.predict
                                          │                 → Precision/Recall/F1 spam
                                          ├── chọn tham số theo CV F1 spam
                                          └── refit trên TOÀN Train → record["_model"]
                                └── B1/B2/B3/B4 (hai nhánh giữ/bỏ stop words)
                                      └── chọn nhánh giữ stopwords và B4 theo CV
                                            └── imbalance: S1/S2/S3 (cell 15–16)
                                                  └── chọn best theo CV F1
                                                        └── evaluation (cell 20)
                                                              ├── score + pred trên Test
                                                              ├── đối chiếu y_test
                                                              └── Accuracy, P, R, F1, AP
                                                        └── notebook (cell 22,24,26,30)
                                                              └── ma trận nhầm lẫn,
                                                                  PR, bootstrap, phân tích lỗi
                                                        └── notebook (cell 32)
                                                              ├── joblib.dump(bundle)
                                                              ├── joblib.load(bundle)
                                                              └── modeling/predict.py
                                                                    └── bình luận gốc → nhãn/score
```

### Cực kỳ quan trọng: `X` và `y` nghĩa là gì?

- `X_train_text`: **mảng chuỗi token của bình luận Train**, chưa phải vector số.
- `y_train`: **mảng nhãn 0/1 tương ứng theo vị trí**.
- `X_valid_text`: chuỗi token ở fold validation; không được dùng để fit bộ tạo vector ở lượt đó.
- `X_test_text`: chuỗi token của Test, chỉ dùng `transform` bằng vectorizer đã fit.
- `y_test`: đáp án đúng, chỉ dùng đối chiếu khi tính metric. **Không truyền y_test vào `predict` hoặc `fit`.**
- `Pipeline`: chứa cả `vec` (vectorizer) và `clf` (classifier). Khi `fit(X_train_text, y_train)`, Pipeline tự chuyển chuỗi thành số rồi cho classifier học.

---

## 4. Bản đồ Notebook: học ở cell nào?

> **Chỉ số cell ở đây bắt đầu từ 0 trong JSON Notebook**, không phải số `In [n]` của Jupyter.

| Cell | Mục tiêu | File/hàm liên quan | Cần nắm |
|---|---|---|---|
| 2–4 | Môi trường, đường dẫn, imports | toàn bộ modeling | Python 3.12, scikit-learn 1.6.1, seed, `ARTIFACT_DIR`, `CSV_PATH` |
| **6** | Đọc/kiểm toán dataset | Pandas | `label` → `y_all` 0/1, không rỗng, trùng, mâu thuẫn nhãn |
| **9** | Chuẩn bị văn bản token | `preprocess()` (module prepare), `STOP_WORDS` | giữ/bỏ từ dừng; lưu CSV processed |
| **11** | Train/Test, 5 folds | `training_utils.split_train_test`, `grouped_folds` | `keys`, `train_idx`, `test_idx`, `CV_FOLDS` |
| **13** | Bốn tổ hợp trên hai nhánh stopwords | `text_features`, `classifiers`, `experiments.fit_experiment` | B1–B4, Pipeline, GridSearchCV, CV F1, `focus` |
| **15** | Minh họa phân bố và lấy mẫu | `imbalance.sample_training_indices` | Ham/Spam mất cân bằng, ratio 2:1, balanced weights |
| **16** | S1/S2/S3 và chọn `best` | `fit_balanced_svm`, `undersampled_search`, `tune_svm_threshold` | thay trọng số, giảm mẫu, đổi ngưỡng; chọn bằng **CV** |
| **20** | Test độc lập | `evaluation.evaluate` | tính `y_pred`, score, Accuracy/P/R/F1/AP, không fit lại |
| 22, 24 | Báo cáo và confusion matrices | `confusion_matrix` | FP = Ham bị gắn Spam; FN = Spam bị bỏ sót |
| 26 | Bootstrap, PR | `boot_ci`, `precision_recall_curve` | độ biến động F1, PR và AP |
| 30 | Từ/cụm từ phân biệt, lỗi | `get_feature_names_out`, `coef_`, `feature_log_prob_` | những đặc trưng nghiêng Spam/Ham, ví dụ FP/FN |
| **32** | Lưu, nạp, demo | `joblib.dump/load`, `predict_record`, `modeling.predict_spam` | `bundle`, `threshold`, `remove_stop`, kiểm tra dự đoán sau nạp |
| 34 | Kết luận | các kết quả trên | phân biệt CV F1 và Test F1; giới hạn kết quả |

**Ngoài phạm vi chuyên sâu của bạn:** Notebook 02 có đoạn giới thiệu và đối chiếu **viBERT** (cell 17–18, 27–28), nhưng huấn luyện mô hình đó ở Notebook 03/Colab khác. Có thể biết để trả lời câu hỏi tổng thể, không nhầm với luồng NB/SVM.

---

## 5. Bản đồ file/hàm — đầu vào → xử lý → đầu ra

| File | Hàm/lớp chính | Đầu vào | Xử lý | Đầu ra / nơi nhận tiếp |
|---|---|---|---|---|
| `training_utils.py` | `norm_key(text)` | cleaned text | NFC, lowercase, chuẩn hóa số/khoảng trắng/dấu để nhóm gần trùng | chuỗi khóa |
| | `bucket(key,salt,n=5)` | khóa + salt | MD5 rồi `% n`, nhóm ổn định | ID bucket |
| | `split_train_test(cleaned_texts,labels)` | cleaned texts + nhãn 0/1 | bucket cùng nội dung về cùng phía | `(keys, train_idx, test_idx)` → Notebook |
| | `grouped_folds(keys,y,n_splits=5)` | khóa/nhãn của **Train** | 5 cặp vị trí train/validation, tách nhóm | `CV_FOLDS` → `fit_experiment` |
| `text_features.py` | `make_vectorizer(feature)` | `"BoW"` hoặc `"TF-IDF"` | tạo `CountVectorizer` / `TfidfVectorizer` **chưa fit**, kèm grid ngram | `(vectorizer, {"vec__ngram_range":...})` → `make_pipe` |
| `classifiers.py` | `make_classifier(model_name,seed,class_weight)` | `"Naive Bayes"` hoặc `"SVM"` | tạo `MultinomialNB` / `LinearSVC` **chưa fit**, grid alpha/C | `(classifier, grid)` → `make_pipe` |
| `experiments.py` | `make_pipe(feat,model_name,...)` | lựa chọn đặc trưng và thuật toán | ghép `Pipeline([("vec",...), ("clf",...)])`, ghép grid | `(pipe,grid,n_jobs)` |
| | `fit_experiment(pre,feat,model_name,X,y,folds,...)` | **chuỗi** Train + nhãn + CV_FOLDS | `GridSearchCV(...,refit="f1").fit(X,y)` | record: CV scores, tham số thắng, `_model`, `_threshold` |
| `imbalance.py` | `fit_balanced_svm(...)` | Train + folds | gọi `fit_experiment` với `class_weight="balanced"` | record S1 |
| | `sample_training_indices(y,indices,ratio=2)` | nhãn + chỉ số được phép | giữ Spam, lấy Ham tối đa 2 lần Spam | chỉ số được chọn |
| | `mean_cv_scores(y,predictions,folds)` | nhãn thật và dự đoán ngoài fold | P/R/F1 Spam từng validation, lấy mean | dictionary CV metrics |
| | `undersampled_search(pipe,grid,X,y,folds,...)` | Pipeline, grid, Train | CV thủ công, giảm Ham **chỉ tại train của fold** | `(fitted,scores,params,n_selected)` → cell 16 tạo record S2 |
| | `tune_svm_threshold(pipe,X,y,folds)` | Pipeline B4 + Train | score OOF, thử ngưỡng, tối ưu CV F1 | `(fitted,metrics,threshold)` → cell 16 tạo record S3 |
| `evaluation.py` | `predict_record(record,X)` | record đã fit + chuỗi Test | Pipeline tự transform; NB `predict_proba`, SVM `decision_function`; áp ngưỡng | `(pred,scores)` |
| | `evaluate(record,X,y)` | record, Test text, y_test | tính Accuracy, P, R, F1, AP | **sửa trực tiếp record; không có `return`** |
| | `boot_ci(y,pred,n=500)` | nhãn thật và nhãn dự đoán | Bootstrap 500 lần của Test F1 | 2 phân vị (2,5%; 97,5%) |
| `predict.py` | `predict_spam(texts,model_bundle)` | bình luận thô + bundle đã nạp | `preprocess` cùng cấu hình, `predict_record`, ánh xạ 0/1 | DataFrame `text, score, score_type, du_doan` |

`README.md` là sơ đồ hướng dẫn; `PHI_HIEU_4_LOP_SCIKIT_LEARN.md` là tài liệu đọc **bên trong thư viện**. Không có `fit()` chạy tự động chỉ vì import các module.

---

## 6. Những khái niệm/lý thuyết phải học theo từng giai đoạn

### 6.1. Dataset, tiền xử lý, chia tập

Cần hiểu: sample/bình luận; feature/đặc trưng; target/label; học có giám sát; missing data; trùng/near-duplicate; data leakage; Unicode NFC; tokenization tiếng Việt; từ ghép có dấu gạch dưới; stopwords; nhãn sai/mâu thuẫn; class imbalance; random seed; Train/Validation/Test; group split; 5-fold cross-validation; OOF (out-of-fold).

**Chỗ phải hiểu:** `split_train_test()` trả **chỉ số**, không trả văn bản; Notebook dùng `TEXT[pre].iloc[train_idx]` để lấy câu tương ứng. Cách chia bằng hash không đảm bảo cân bằng nhãn tuyệt đối. `norm_key` chỉ tạo khóa chia tập, không thay cho tokenizer dùng khi dự đoán.

### 6.2. Trích chọn BoW và TF-IDF (phiên bản DÙNG THƯ VIỆN)

Cần hiểu: vocabulary, token, unigram/bigram, n-gram, document frequency, `min_df`, sparse matrix CSR, số hàng = số bình luận, số cột = số đặc trưng, OOV (từ ngoài từ vựng), `.fit`, `.transform`, `.fit_transform`, `.get_feature_names_out()`.

**Code cấu hình thật trong `text_features.py`:**

```python
common = {
    "token_pattern": r"(?u)\b\w+\b|[^\w\s]",
    "min_df": 2,
    "ngram_range": (1, 1),  # CV còn thử (1, 2)
    "lowercase": True,
    "dtype": np.float64,
}

bow = CountVectorizer(**common)
tfidf = TfidfVectorizer(
    **common, sublinear_tf=True, smooth_idf=True, norm="l2"
)
```

- **BoW:** `CountVectorizer` học từ điển trên train, mỗi ô là **số lần đặc trưng xuất hiện trong một bình luận**.
- **TF-IDF:** `TfidfVectorizer` kế thừa bước đếm, rồi dùng `TfidfTransformer` để học IDF và biến các số đếm thành trọng số. Với đặc trưng xuất hiện `tf > 0`: `sublinear_tf = 1 + ln(tf)`; `smooth_idf = 1 + ln((1+N)/(1+df))`; TF-IDF nhân hai phần rồi `norm="l2"` chuẩn hóa **từng vector bình luận**.
- **`min_df=2`:** đặc trưng phải có trong ít nhất **hai bình luận của lần fit** (ví dụ fold train). Đặc trưng chỉ có ở Test không tự tạo cột mới.
- **`stop_words=None`:** `CountVectorizer` không tự bỏ stopwords ở đây; Notebook đã tạo hai cột token và thử hai nhánh.
- **N-gram:** `(1,1)` chỉ từ đơn; `(1,2)` gồm từ đơn + cặp từ kề nhau; dùng CV chọn.

**Ví dụ thực tế của `min_df`:** Bốn câu train đã token hóa: `"nhận làm bài_tập"`, `"nhận làm thuê"`, `"xin tài_liệu học"`, `"xin lịch học"`. Với `(1,1)` và `min_df=2`, vocabulary có các đặc trưng `học`, `làm`, `nhận`, `xin` (theo thứ tự tên); `bài_tập`, `thuê`, `tài_liệu`, `lịch` xuất hiện một câu nên bị loại. Hàng BoW của câu 1 là `[0,1,1,0]`, câu 3 là `[1,0,0,1]`. **Không có nhãn Spam/Ham nào được tạo tại bước vector hóa.**

**Đọc sâu trong tài liệu thư viện:** `CountVectorizer._count_vocab`, `CountVectorizer.fit_transform`, `TfidfVectorizer.fit_transform/transform`, `TfidfTransformer.fit/transform`, tokenizer/analyzer/`_word_ngrams`. Đường dẫn và số dòng tương ứng scikit-learn 1.6.1 nằm trong `PHI_HIEU_4_LOP_SCIKIT_LEARN.md`; số dòng có thể thay đổi nếu dùng phiên bản khác.

### 6.3. Naive Bayes và SVM

Cần hiểu: supervised classification; binary classification; xác suất tiên nghiệm/hậu nghiệm; độc lập có điều kiện; Laplace smoothing / `alpha`; log probability; hyperplane, margin, trọng số `w`, bias `b`, decision score; hinge/squared-hinge; tham số `C`; `class_weight`; underfitting/overfitting.

**Hai lớp đang dùng:**

```python
nb  = MultinomialNB()
svm = LinearSVC(dual=False, max_iter=20000, random_state=42)
```

- **`MultinomialNB.fit(X_matrix, y)`** học `class_count_`, `feature_count_`, `class_log_prior_`, `feature_log_prob_`. Về bản chất: lấy tổng các giá trị đặc trưng theo lớp, làm mịn rồi dùng log xác suất để tính điểm. `predict_proba` cho **xác suất theo mô hình** từng lớp; trong bài lấy **cột 1 = Spam**. TF-IDF không phải số đếm nguyên nhưng scikit-learn cho phép thử MNB với các trọng số không âm.
- **`LinearSVC.fit(X_matrix, y)`** tối ưu `coef_` và `intercept_`, tìm ranh giới tuyến tính. `decision_function` tính `X @ coef_.T + intercept_`; đây là **score có dấu, không phải xác suất**. Khi lớp là `[0,1]`, score > 0 nghiêng Spam; <= 0 nghiêng Ham. S3 được phép dùng ngưỡng khác 0.
- Grid của NB: `alpha = [0.1, 0.3, 1.0]`; grid của SVM: `C = [0.1, 0.3, 1.0, 3.0]`. Đây là **các giá trị đề xuất để thử**, chưa phải kết quả mô hình học được.

**Đọc sâu bốn lớp thư viện** (đường dẫn trong tài liệu đính kèm):

1. `CountVectorizer`: `_count_vocab`, `fit_transform`, `transform`.
2. `TfidfVectorizer`: `fit_transform`, `transform`; `TfidfTransformer` tính IDF.
3. `MultinomialNB`: `fit` kế thừa từ `_BaseDiscreteNB`; `_count`, `_update_feature_log_prob`, `_joint_log_likelihood`, `predict_proba`.
4. `LinearSVC`: `fit` gọi `_fit_liblinear`; `decision_function`/`predict` kế thừa từ lớp cha. Bộ tối ưu Liblinear có phần biên dịch nên không đọc được toàn bộ chỉ bằng Python Ctrl+Click.

### 6.4. Pipeline, `fit`, Cross-Validation, GridSearchCV

Cần hiểu: estimator chưa fit, trạng thái đã học, Pipeline, `vec__`/`clf__`, grid siêu tham số, GridSearchCV, scorer, mean validation F1, `best_params_`, `best_estimator_`, `refit`, CV vs Test, `clone`, caching bằng joblib `Memory`.

**Ghép hai bước:**

```python
pipe = Pipeline([("vec", vectorizer), ("clf", classifier)], memory=cache)
```

**Bên trong một fold:**

```text
X_fold_train (chuỗi) → vec.fit_transform → ma trận X_train_vector
                                      + y_fold_train → clf.fit(...)
X_fold_valid (chuỗi) → vec.transform → ma trận X_valid_vector
                                    → clf.predict / score
                                    → so với y_fold_valid → P, R, F1
```

`fit_experiment()` gọi `GridSearchCV(pipe, grid, scoring={...}, cv=folds, refit="f1")` rồi `search.fit(X,y)`. **Đây là lúc bắt đầu huấn luyện thật**. `mean_test_f1` trong `cv_results_` chính là **mean F1 validation ở CV**, dù tên scikit-learn chứa `test`. Sau CV, cấu hình thắng được fit lại trên **toàn bộ Train**, trả `record["_model"]` (Pipeline đã fit).

**Bốn tổ hợp cơ sở:**

| Mã | Vectorizer | Classifier | Số cấu hình tham số mỗi nhánh stopwords |
|---|---|---|---:|
| B1 | BoW (`CountVectorizer`) | `MultinomialNB` | `2 n-gram × 3 alpha = 6` |
| B2 | BoW (`CountVectorizer`) | `LinearSVC` | `2 n-gram × 4 C = 8` |
| B3 | TF-IDF (`TfidfVectorizer`) | `MultinomialNB` | `6` |
| B4 | TF-IDF (`TfidfVectorizer`) | `LinearSVC` | `8` |

Có **28 cấu hình** trên một nhánh stopwords; Notebook thử **hai nhánh**, tức **56 cấu hình**. Với 5 fold là **280 lượt fit của CV cho thí nghiệm cơ sở**, chưa kể lần refit cấu hình thắng, caching và các thí nghiệm S1/S2/S3.

### 6.5. Mất cân bằng lớp: ba chiến lược khác nhau

Cần hiểu: majority/minority class; precision–recall trade-off; class weight, undersampling, validation distribution, OOF score, threshold, `ParameterGrid`, `clone`.

| Tên | Cách làm | Dữ liệu dùng để fit | Điều gì thay đổi? |
|---|---|---|---|
| **B4** | TF-IDF + SVM gốc | toàn Train | C, n-gram do CV chọn; threshold=0 |
| **S1** | `class_weight="balanced"` | toàn Train | phạt lỗi của lớp ít mẫu nặng hơn |
| **S2** | undersampling 2:1 | chỉ Spam + tối đa 2 Ham / 1 Spam | số mẫu Ham được dùng để học giảm; validation/Test **không giảm** |
| **S3** | threshold tuning | toàn Train | **ngưỡng decision score**, không phải vectorizer hay thuật toán khác |

S3 tạo **OOF decision score** trên Train: mỗi dòng được chấm bởi mô hình không học chính dòng đó. Thử các ngưỡng, tính F1 theo 5 validation fold, chọn ngưỡng có CV F1 tốt nhất; sau đó clone và fit lại Pipeline B4 trên toàn Train. **S3 không phải thuật toán thứ ba** và **không học ngưỡng từ Test**. Thủ tục lựa chọn ngưỡng hiện tại không phải nested CV độc lập.

### 6.6. Evaluation: đánh giá dựa trên điều gì?

Cần hiểu: label thật `y_test`, label dự đoán `pred`, score; TP, TN, FP, FN; Accuracy, Precision, Recall, F1; AP (Average Precision), đường PR; bootstrap CI; FP/FN trong lọc Spam.

```text
Test text → record["_model"] (Pipeline đã fit)
          → NB.predict_proba[:,1] hoặc SVM.decision_function
          → score > hoặc >= threshold → pred 0/1
          → so sánh pred[i] và y_test[i] → TP/TN/FP/FN
          → Accuracy / Precision / Recall / F1
          → dùng score + y_test → AP / PR curve
```

- Với **Spam = 1**: TP = Spam dự đoán Spam; TN = Ham dự đoán Ham; FP = Ham dự đoán Spam; FN = Spam bị bỏ sót.
- `evaluate(record,X,y)` **chỉnh chính dictionary `record`**, không trả record mới.
- `predict_record` trả `(pred,scores)`: NB dùng xác suất Spam và threshold mặc định `0.5` (`>=`), SVM dùng decision score và threshold mặc định `0.0` (`>`).
- Score NB và SVM **khác thang đo**, không so sánh trực tiếp hai score; dùng metric để đánh giá.
- AP dựa trên xếp hạng score qua nhiều ngưỡng, không phải F1 tại một ngưỡng cố định.
- `boot_ci` rút lại các cặp `(y_test,pred)` 500 lần có hoàn lại, trả khoảng phân vị F1; **không fit lại mô hình**.

### 6.7. Lưu mô hình và inference

Cần hiểu: trained model vs fresh estimator; serialization; bundle; `joblib.dump`, `joblib.load`; khả năng tái lập preprocessing; model drift; score vs label; không fit trên dữ liệu mới.

**Cell 32 lưu:**

```python
bundle = {
    "model": best["_model"],
    "threshold": best["_threshold"],
    "remove_stop": best["Tiền xử lý"] == "có loại stop words",
    "manifest": manifest,
    "preprocessing_source": inspect.getsource(preprocessing_module),
}
joblib.dump(bundle, model_path)
loaded = joblib.load(model_path)
```

`model` chứa **Pipeline đã fit, gồm vectorizer với vocabulary/IDF và classifier với tham số đã học**. Vì vậy `predict.py` không cần xây vocabulary mới. Tệp `predict.py` **không ghi `.joblib`**; Notebook làm việc đó.

**`predict_spam(texts, model_bundle)` chạy:** bình luận **gốc** → `preprocess(text,text, remove_stop)` → `predict_record` (Pipeline tự vector hóa, tính score, so ngưỡng) → DataFrame `text`, `score`, `score_type`, `du_doan` (`spam`/`ham`). Cell 32 nạp lại `.joblib`, kiểm tra nhãn trên **toàn bộ 7.474 bình luận Test** khớp kết quả trước khi lưu; tuy nhiên score có thể khác khi đi từ `text` gốc thay vì cột cleaned của CSV. Không sử dụng demo để khẳng định mô hình luôn đúng. Chỉ nạp file joblib từ nguồn tin cậy vì đây là định dạng pickle có khả năng thực thi mã khi nạp.

---

## 7. Kết quả thực nghiệm notebook MỚI — cần thuộc số nào?

### 7.1. Lựa chọn tiền xử lý và mô hình cơ sở (cell 13)

- Thử hai nhánh giữ/bỏ stopwords. **Nhánh giữ stopwords** tốt hơn ở cấu hình dẫn đầu nên được chọn làm `MAIN_PRE`.
- Điểm CV F1 Spam trong nhánh giữ stopwords:

| Mã | Tổ hợp | CV F1 Spam | Tham số thắng |
|---|---|---:|---|
| B1 | BoW + Naive Bayes | 0,6077 | alpha=1,0; n-gram=(1,2) |
| B2 | BoW + SVM | 0,7087 | C=1,0; n-gram=(1,2) |
| B3 | TF-IDF + Naive Bayes | 0,5959 | alpha=0,1; n-gram=(1,2) |
| **B4** | **TF-IDF + SVM** | **0,7736** | C=0,3; n-gram=(1,2) |

**B4 được chọn làm mô hình cơ sở để khảo sát xử lý mất cân bằng**; đây chưa phải lựa chọn cuối cùng S3.

### 7.2. So sánh xử lý mất cân bằng (cell 16)

| Mã | Phương án | CV F1 Spam | Ngưỡng | Số mẫu fit Train cuối |
|---|---|---:|---:|---:|
| B4 | Không điều chỉnh | 0,7736 | 0 | 30.759 |
| S1 | Balanced | 0,7580 | 0 | 30.759 |
| S2 | Under-sampling 2:1 | 0,7629 | 0 | 13.125 |
| **S3** | **Chỉnh ngưỡng** | **0,7876** | **−0,1776** | **30.759** |

**Chọn S3 theo CV F1, trước khi đánh giá Test**. Ngưỡng âm chuyển thêm một số score hơi âm sang Spam, thường làm Recall tăng nhưng có thể làm Precision giảm.

### 7.3. Đánh giá tập Test độc lập (cell 20 và 22)

| Mã | Accuracy | Precision Spam | Recall Spam | F1 Spam | AP |
|---|---:|---:|---:|---:|---:|
| B1 | 0,9141 | 0,7441 | 0,4953 | 0,5947 | 0,7320 |
| B2 | 0,9241 | 0,7110 | 0,6803 | 0,6953 | 0,7493 |
| B3 | 0,9257 | 0,8402 | 0,5142 | 0,6380 | 0,7988 |
| B4 | 0,9419 | 0,8818 | 0,6278 | 0,7334 | 0,8117 |
| S1 | 0,9322 | 0,7139 | 0,7792 | 0,7451 | 0,8127 |
| S2 | 0,9394 | 0,7791 | 0,7308 | 0,7542 | 0,7826 |
| **S3** | **0,9433** | **0,8374** | **0,6877** | **0,7552** | **0,8117** |

S3 trên Test: **951 Spam / 7.474 mẫu**, có **127 Ham bị báo Spam (FP)** và **297 Spam bị bỏ sót (FN)**, nên **TP=654**, **TN=6.396**. Điểm CV F1 `0,7876` **không phải** F1 Test `0,7552`.

**Cảnh báo đọc kết quả:** S1 có Recall Spam cao nhất trong bốn phương án SVM so sánh ở đây (`0,7792`), nhưng F1 Test của S3 cao hơn một chút. Không được tuyên bố S3 tốt hơn trên mọi chỉ số hoặc luôn tốt nhất trên mọi tập dữ liệu. Đối chứng luôn đoán Ham đạt Accuracy khoảng **87,28%** nhưng Recall/F1 Spam bằng 0.

**Ví dụ demo của cell 32**: `"Cho mình hỏi lịch thi với ạ"` → Ham; `"Liên hệ 0901234567 hoặc https://example.com"` → Spam; `"Nguyễn Văn An"` và `"."` cũng bị dự đoán Spam. Đây là ví dụ hay để nói về hạn chế với bình luận thiếu ngữ cảnh/ngắn hoặc biểu diễn đặc trưng và nhãn có nhiễu. Không coi dự đoán demo là nhãn thật.

---

## 8. Những điểm giảng viên có thể hỏi và câu trả lời cốt lõi

**Hỏi: `fit` là gì?** → Đối tượng học trạng thái cần thiết từ Train. Vectorizer học vocabulary/IDF; NB học phân bố đặc trưng và lớp; SVM học trọng số ranh giới. Tạo đối tượng `CountVectorizer()` hay `LinearSVC()` **chưa phải fit**.

**Hỏi: Input vào `fit_experiment` đã là vector chưa?** → Chưa, vẫn là **chuỗi token**. `Pipeline.fit` gọi `vec.fit_transform` và `clf.fit` bên trong.

**Hỏi: Bốn tổ hợp kết hợp bằng cách nào?** → Vectorizer + classifier được nối trong một `Pipeline`, không trộn thuật toán. Mỗi tổ hợp là mô hình riêng; cùng Train/Test để so sánh.

**Hỏi: Vì sao không fit TF-IDF trên toàn dataset trước CV?** → Vì vocabulary/IDF từ validation/Test sẽ rò rỉ vào bước học. Pipeline đảm bảo mỗi fold chỉ học vectorizer trên fold train.

**Hỏi: `GridSearchCV.fit()` làm gì?** → Với từng tổ hợp n-gram + alpha/C, fit trên fold train và đo F1 Spam trên fold validation; lặp 5 fold, lấy trung bình, chọn best; `refit="f1"` tự huấn luyện lại trên toàn Train.

**Hỏi: `mean_test_f1` có phải Test F1 không?** → Không; đó là validation score của các fold trong CV, theo cách đặt tên trường của scikit-learn.

**Hỏi: S3 có huấn luyện lại SVM không?** → Có các lượt fit clone SVM để tạo OOF score và một lượt fit cuối trên Train; **không phải huấn luyện lại bằng Test**. Thay đổi chính để dự đoán là **ngưỡng score**; giữ bộ siêu tham số của B4.

**Hỏi: Điểm Accuracy/F1 từ đâu ra?** → Từ `y_test` nhãn thật và `pred` nhãn dự đoán (TP/TN/FP/FN). Không lấy trực tiếp vector TF-IDF hoặc decision score thế vào công thức F1; score dùng để quyết định nhãn và tính AP/PR.

**Hỏi: NB score và SVM score có cùng nghĩa?** → Không. NB dùng xác suất theo mô hình cho Spam (`predict_proba[:,1]`); SVM dùng `decision_function` có dấu, **không phải xác suất**.

**Hỏi: `predict.py` có fit/lưu mô hình không?** → Không. Nó nhận bundle đã nạp, tiền xử lý bình luận mới giống Train, gọi Pipeline đã fit và trả bảng nhãn/score. `joblib.dump` nằm ở Notebook/script.

**Hỏi: Có nhược điểm gì?** → Chất lượng phụ thuộc nhãn và nội dung bình luận, có nhóm trùng/lệch nhãn; tập bị lệch lớp; bình luận thiếu ngữ cảnh, nhiều từ lóng hoặc chỉ một dấu có thể sai. CV dùng để chọn mô hình, Test để báo cáo; kết quả chỉ áp dụng trong phạm vi dữ liệu/thiết kế được thử.

---

## 9. Lộ trình tạo từng file `.md` chuyên sâu sau tổng quan

Khi bạn yêu cầu một bước, tạo một file `.md` **độc lập và có thể học riêng**, theo mẫu chung:

1. **Mục tiêu của bước / kết quả cần có**.
2. **Khái niệm, lý thuyết, công thức, thuật ngữ** từ cơ bản đến đúng cấu hình dự án; phân biệt kiến thức từ code và kiến thức mở rộng.
3. **Cell Notebook tương ứng**, trích code gốc liên quan và giải thích biến/đầu vào/đầu ra.
4. **Hàm/lớp trong file Python**: khai báo, từng dòng hoặc từng nhóm dòng, tại sao cần, kiểu dữ liệu/chuyển đổi, gọi sang file nào.
5. **Đường đi của một bình luận PTIT cụ thể** qua hàm, bảng/vector/nhãn trung gian (ghi rõ số nào minh họa, số nào từ Notebook).
6. **Những chỗ dễ nhầm**, các giả định/giới hạn và tác động lên bước tiếp theo.
7. **Bảng tóm tắt + câu hỏi bảo vệ + câu trả lời mẫu**.

| Số | File `.md` dự kiến | Nội dung chính / cell Notebook |
|---|---|---|
| 01 | `01_nhan_dataset_va_tien_xu_ly.md` | Cell 6, 9; CSV, label, token, stopwords, `preprocess()` — sẽ cần nguồn module prepare nếu phân tích nội bộ hàm |
| 02 | `02_training_utils_chia_train_test_cv.md` | Cell 11; khóa, hash bucket, Train/Test, grouped 5 folds |
| 03 | `03_text_features_bow_tfidf_sklearn.md` | Cell 12–13; `CountVectorizer`, `TfidfVectorizer`, `fit_transform`, `transform`, vocabulary, n-gram, IDF/L2, bốn lớp doc (phần vector) |
| 04 | `04_classifiers_naive_bayes_svm.md` | Cell 12–13; MultinomialNB, LinearSVC, alpha/C, xác suất/decision score, bốn lớp doc (phần phân loại) |
| 05 | `05_experiments_pipeline_gridsearch_cv.md` | Cell 13; Pipeline, CV, grid 4 tổ hợp × 2 nhánh, record, refit |
| 06 | `06_imbalance_S1_S2_S3.md` | Cell 15–16; class weight, under-sampling, OOF threshold, lựa chọn best |
| 07 | `07_evaluation_metrics_test.md` | Cell 20, 22, 24, 26, 30; score, dự đoán, TP/FP/FN, P/R/F1/AP, bootstrap, phân tích lỗi |
| 08 | `08_predict_joblib_inference.md` | Cell 31–32; bundle, joblib.dump/load, kiểm tra consistency, `predict_spam` |

**Ưu tiên học:** 01 → 02 → 03 → 04 → 05 → 06 → 07 → 08. Nếu nền tảng về `fit`/`transform` chưa chắc, học 03 rồi mới sang 04–05.

---

## 10. Tóm tắt 60 giây để trình bày phần việc

> Em nhận dataset bình luận PTIT đã có nhãn Ham và Spam từ phần chuẩn bị dữ liệu của nhóm. Notebook tạo hai phương án văn bản giữ/bỏ stopwords bằng module của thành viên khác; phần em phụ trách bắt đầu ở khâu chia Train/Test theo nhóm nội dung để hạn chế trùng lặp, đồng thời tạo 5 fold validation bên trong Train. Sau đó em dùng BoW hoặc TF-IDF của scikit-learn để biểu diễn văn bản và Naive Bayes hoặc Linear SVM để phân loại, tạo bốn tổ hợp trong Pipeline. Em dùng GridSearchCV thử n-gram, alpha/C và lựa chọn theo F1 Spam trung bình trên validation, rồi fit lại cấu hình thắng trên toàn Train. Với TF-IDF + SVM, em thử thêm balanced, undersampling và điều chỉnh ngưỡng; phương án S3 được chọn theo CV. Sau đó em dự đoán Test độc lập, so với nhãn thật để tính Accuracy, Precision, Recall, F1, AP và phân tích FP/FN. Cuối cùng em lưu Pipeline, ngưỡng và cấu hình tiền xử lý bằng Joblib, nạp lại và dùng `predict_spam()` để phân loại bình luận mới mà không train lại.

**Điểm cần thuộc cuối cùng:** `CSV → token → split → CV → vec.fit_transform → clf.fit → chọn tham số → refit → Test score/pred → metrics → save → predict mới`. Các bước bên trong CV phải **chỉ nhìn Train**, không lấy Test để tối ưu.
