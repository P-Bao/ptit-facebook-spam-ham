# Phi học hiểu phần mô hình truyền thống

Phạm vi Phi chốt hiện tại: nhận CSV đã có nhãn/token → **tự chia train/test và CV** → BoW/TF-IDF + Naive Bayes/SVM → khảo sát lệch lớp → đánh giá test → lưu và demo S3. Học trước theo [các giai đoạn](PHI_LO_TRINH_HOC_THEO_GIAI_DOAN.md), đọc [script chạy thẳng](../../scripts/run_phi_pipeline.py), rồi xem [code đầy đủ theo luồng](PHI_DOC_CODE_THEO_LUONG.md). [Phân công nhóm](TEAM_ASSIGNMENTS.md) cần được thống nhất lại khi mọi người chốt phần trình bày.

## 1. Điểm nhận việc và kết quả bàn giao

Đầu vào của Phi là `data/processed/v7_preprocessed.csv`, đã có nhãn `ham`/`spam`, `text` gốc, `cleaned_text` và hai cột token giữ/bỏ từ dừng. Các cell trước mục 4 của Notebook 02 đã nạp CSV vào `raw`, tạo `TEXT` và `y_all` (`ham=0`, `spam=1`). Phi **không cần viết lại** crawler, quy trình gán nhãn hay hàm tách từ; cần hiểu định dạng đầu vào và tác động của nó đến mô hình.

Phi bàn giao: train/test và năm fold; bảng CV cho B1–B4 và S1–S3; chỉ số test; mô hình S3 đã fit, ngưỡng và quy tắc tiền xử lý trong `artifacts/spamham/spam_ham_ml_only.joblib`; hàm dự đoán bình luận mới. Bảo có thể trình bày sâu phần lỗi/biểu đồ chung; Phương phụ trách viBERT. Phi cần hiểu toàn bộ phép đo test của nhánh truyền thống.

```text
CSV có nhãn và token
  → Notebook 02 cell 11 / run_phi_pipeline.py: tự chia train/test, 5 fold CV
  → text_features.py + classifiers.py + experiments.py + cell 13: 4 cặp × 2 nhánh từ dừng
  → chọn nhánh giữ từ dừng; B1–B4, B4 = TF-IDF + SVM
  → imbalance.py + cell 15–16: B4, S1, S2, S3; chọn bằng CV F1 spam
  → evaluation.py + phần đánh giá chung: đo test sau khi đã chọn
  → cell 32 + modeling/predict.py: lưu S3, nạp lại, dự đoán bình luận mới
  → cell 34: nhận xét phần mô hình truyền thống
```

Đây là **hai giai đoạn đo khác nhau**: CV dùng các phần validation **nằm trong train** để chọn cấu hình; test được giữ riêng cho lần đánh giá sau lựa chọn. Kết quả hiện tại có thể thay đổi khi dữ liệu hoặc code đổi; số liệu dưới đây là của lượt chạy đã xác minh sau khi bỏ FastText.

## 2. File và cell Phi cần học

| Mức độ | File / vị trí | Phi cần giải thích được |
|---|---|---|
| Học kỹ | [`text_features.py`](../../pipeline/modeling/text_features.py), [`classifiers.py`](../../pipeline/modeling/classifiers.py) | `CountVectorizer`/`TfidfVectorizer` tạo đặc trưng; `make_classifier` tạo NB/SVM chưa fit; hiểu lưới n-gram/alpha/C. |
| Học kỹ | [`experiments.py`](../../pipeline/modeling/experiments.py) | `make_pipe`, `fit_experiment`, `GridSearchCV`; phân biệt record CV với điểm test. |
| Học kỹ | [`imbalance.py`](../../pipeline/modeling/imbalance.py) | `sample_training_indices`, `undersampled_search`, `mean_cv_scores`, `tune_svm_threshold`. |
| Học kỹ | [`evaluation.py`](../../pipeline/modeling/evaluation.py) | `predict_record`, `evaluate`, `boot_ci`; giải thích test Precision/Recall/F1/AP và tác dụng của score/ngưỡng. |
| Học kỹ | [`training_utils.py`](../../pipeline/modeling/training_utils.py) | `split_train_test`, `grouped_folds` là hai thao tác chính; `norm_key`, `bucket` giữ câu trùng cùng phía và giữ cách chia ổn định. |
| Học kỹ | [`pipeline/modeling/predict.py`](../../pipeline/modeling/predict.py) | `predict_spam`: nhận comment mới, gọi đúng tiền xử lý, lấy score từ bundle và áp ngưỡng S3. |
| Hiểu giao diện | [`pipeline/prepare/preprocess_for_model.py`](../../pipeline/prepare/preprocess_for_model.py) | Đầu ra `preprocess(...)` là chuỗi token và `remove_stop` quyết định nhánh; Cương phụ trách nội bộ hàm. |
| Trình bày | [Notebook 02](../../notebooks/02_thuc_nghiem_va_demo.ipynb) cell 13, 15, 16, 32, 34 | Các cell thực hiện CV, khảo sát lệch lớp, chọn S3, lưu/nạp và nhận xét. |
| Trình bày | Notebook 02 cell 11, 20 và mục 6 | Phi tự chia tập và chấm test. Bảo có thể phối hợp trình bày đồ thị và phân tích lỗi. |

Không còn `pipeline/features/spam_features.py`, `FastTextVectorizer`, `GaussianNB` hay `StandardScaler` của nhánh FastText. Không cần học chúng để trình bày **bài hiện tại**. Không tự sửa code trên nhánh Cương/Bảo/Phương; nếu thấy thay đổi về split, nhãn, quy trình đánh giá hay viBERT thì ghi lại để hỏi người phụ trách trước khi gộp.

## 3. Khái niệm phải hiểu, theo thứ tự học

### 3.1. Bài toán và thước đo

Đây là phân loại nhị phân. `spam=1` là lớp cần phát hiện; `ham=0` là bình luận bình thường. **TP**: spam đoán spam; **FP**: ham đoán spam; **FN**: spam bị bỏ sót; **TN**: ham đoán ham. Precision spam = `TP/(TP+FP)`; Recall spam = `TP/(TP+FN)`; F1 spam là trung bình điều hòa `2PR/(P+R)`. Accuracy có thể cao dù bỏ sót spam vì ham nhiều hơn. `AP` tóm tắt chất lượng xếp hạng trên đường Precision–Recall; AP dùng score, nên đổi ngưỡng đơn thuần thường không đổi AP.

### 3.2. Đặc trưng văn bản

Máy học truyền thống cần vector số. **BoW** do `CountVectorizer` đếm: mỗi chiều ứng với một từ/cặp từ trong train; giá trị là số lần xuất hiện. **TF-IDF** do `TfidfVectorizer` tính thêm IDF và chuẩn hóa độ dài vector. `ngram_range=(1,1)` là từ đơn; `(1,2)` thêm cặp từ liên tiếp. `min_df=2` loại đặc trưng chỉ có trong một văn bản **khi fit vectorizer**. Từ vựng và IDF được học riêng trên train của từng fold.

Project hiện dùng duy nhất `CountVectorizer` và `TfidfVectorizer` của scikit-learn. Xem [tài liệu giải thích mã nguồn bốn lớp](PHI_HIEU_4_LOP_SCIKIT_LEARN.md) khi muốn đọc tiếp phương thức thư viện qua Ctrl+Click.

### 3.3. Hai bộ phân loại

**Multinomial Naive Bayes** được tạo trong `classifiers.py` và fit qua `experiments.py`; nó học liên hệ giữa đặc trưng không âm và lớp. `alpha` làm trơn để đặc trưng chưa gặp không khiến xác suất bằng 0. Đây là mô hình thật sau khi gọi `fit`, không chỉ là tên thuật toán. **LinearSVC** học ranh giới tuyến tính có biên; `C` quyết định mức phạt lỗi/độ điều chuẩn. `decision_function` trả score có dấu, **không phải xác suất**: mặc định score `>0` dự đoán spam; S3 thay ngưỡng này bằng giá trị học từ train.

`Pipeline(vec → clf)` ghép vectorizer và bộ phân loại. Mỗi lần `fit` của CV sẽ fit lại cả hai bước trên **fold train**. Trong `GridSearchCV`, tên như `vec__ngram_range` và `clf__C` là tham số của bước `vec` và `clf`. Code thử hai n-gram, nhiều `alpha` hoặc `C`, chấm Precision/Recall/F1 nhưng chọn tham số theo `refit="f1"`.

### 3.4. Train, validation, test và CV

Cell 11 tạo `train_idx`, `test_idx` bằng `split_train_test`; khóa `norm_key(cleaned_text)` giữ các bình luận trùng nội dung ở cùng phía. `grouped_folds` chia **phần train** thành năm fold. Mỗi lượt: bốn phần fit, một phần validation; lặp năm lượt rồi lấy trung bình F1 spam. Tập test riêng chỉ dùng sau khi đã quyết định mô hình. Phương pháp chia theo văn bản không bảo đảm tách theo tác giả, bài đăng hay thời gian.

Cell 13 chạy **8 lượt thí nghiệm CV**: hai nhánh token (giữ/bỏ từ dừng) × hai đặc trưng (BoW/TF-IDF) × hai bộ phân loại (NB/SVM). Nhánh có ứng viên CV F1 cao nhất hiện là **giữ từ dừng**. Chỉ bốn record của nhánh đó được mã hóa B1–B4; B4 là TF-IDF + SVM dẫn đầu nhóm cơ sở. Không nói “8 mô hình cuối” hay dùng test để chọn nhánh từ dừng.

### 3.5. Mất cân bằng và ba chiến lược

Vì spam ít hơn ham, Phi so B4 (không điều chỉnh) với **ba cách riêng trên TF-IDF + SVM**, không cộng cả ba vào một model:

| Mã | Cách làm | Điểm cần nói |
|---|---|---|
| B4 | Không điều chỉnh | Mốc so sánh; `class_weight=None`, dùng toàn bộ train, ngưỡng 0. |
| S1 | `class_weight="balanced"` | Lỗi lớp ít mẫu bị phạt nặng hơn khi fit SVM; giữ nguyên số dòng train. |
| S2 | Under-sampling 2:1 | Trong mỗi fold, giữ spam và lấy ngẫu nhiên tối đa hai ham cho mỗi spam từ **fold train**, không đụng validation/test. Fit cuối cũng lấy mẫu từ toàn train. |
| S3 | Chỉnh ngưỡng bằng OOF | Giữ pipeline SVM của B4; fit từng fold để tạo score ngoài fold, thử các ngưỡng trên score train rồi chọn theo F1 spam. Fit cuối trên toàn train; khi dự đoán dùng ngưỡng đã chọn. |

OOF (*out of fold*) nghĩa là mỗi score huấn luyện được tạo bởi model **không fit trên chính dòng đó**. Ngưỡng S3 khoảng `−0,1776` là ngưỡng của `decision_function`, không phải xác suất âm. Hạ ngưỡng so với 0 làm một số bình luận trước đó bị đoán ham chuyển sang spam: thường tăng Recall và có thể giảm Precision. Code chọn bằng CV F1 trong phạm vi các ứng viên đã thử; đây **không phải nested CV độc lập**.

### 3.6. Chọn mô hình, lưu và suy luận

`best = max(CANDIDATES, key=CV F1)` chọn giữa bốn B và ba S **trước khi xem test**. Sau đó `evaluate` đo Accuracy, Precision, Recall, F1 và AP trên cùng test. Lượt chạy hiện tại: B4 CV F1 `0,7736`, test F1 `0,7334`; S3 CV F1 `0,7876`, test Precision `0,8374`, Recall `0,6877`, F1 `0,7552`. S3 chỉ nhỉnh hơn S2 trên test; không khẳng định nó sẽ luôn tốt nhất trên dữ liệu khác.

Cell 32 lưu `bundle` bằng `joblib.dump`: pipeline đã fit, ngưỡng S3, quy tắc giữ/bỏ từ dừng và manifest. File `.py` định nghĩa **cách làm**; `.joblib` chứa **trạng thái đã học**. Khi nạp lại, `predict_spam` nhận text mới, gọi `preprocess` cùng chế độ như lúc train, để vectorizer trong pipeline biến thành số, lấy `decision_function` và so với ngưỡng; trả bảng `text`, `score`, `score_type`, `du_doan`. Không gọi AI gán nhãn và không huấn luyện lại khi dự đoán.

## 4. Một lượt trình bày ngắn cho Phi

“Tôi nhận CSV đã có nhãn và hai cột token, tự chia train/test theo nhóm nội dung rồi chia train thành năm lượt kiểm định. Tôi dùng `CountVectorizer` cho BoW, `TfidfVectorizer` cho TF-IDF, `MultinomialNB` và `LinearSVC` để phân loại. Mỗi lượt chỉ học từ vựng, IDF và trọng số trên fold train rồi chấm validation. B4 TF-IDF + SVM dẫn đầu bốn cấu hình cơ sở. Trên B4, tôi so mốc không điều chỉnh với trọng số lớp, giảm mẫu ham và chỉnh ngưỡng từ score ngoài fold. S3 có CV F1 cao nhất trong bảy ứng viên. Sau khi chốt, nhóm đo trên test; F1 của S3 là 0,7552. Tôi lưu pipeline đã học cùng ngưỡng vào `.joblib`, nạp lại và demo bình luận mới.”

## 5. Câu hỏi dễ bị hỏi

1. **Vì sao chỉ bốn B?** — FastText đã bỏ; hiện là `2 đặc trưng × 2 bộ phân loại = 4`. Hai nhánh từ dừng chỉ phục vụ chọn cách tiền xử lý bằng CV, không làm thành tám B ở bảng cuối.
2. **Vì sao NB và SVM đều gọi là mô hình?** — `MultinomialNB()`/`LinearSVC()` là cấu hình thuật toán; sau `fit`, đối tượng chứa tham số đã học và là mô hình. `.joblib` lưu chính đối tượng đã học đó cùng vectorizer/ngưỡng.
3. **Vì sao không fit TF-IDF trên toàn bộ CSV trước CV?** — IDF và từ vựng sẽ nhìn thấy validation/test, làm đánh giá lạc quan. `Pipeline` fit trong từng fold để tránh điều này.
4. **CV F1 và test F1 khác nhau thế nào?** — CV F1 là trung bình validation trong train để chọn; test F1 là kết quả trên tập giữ riêng. Không chọn lại theo điểm test.
5. **S1/S2/S3 có dùng cùng lúc không?** — Không. Đó là ba đối chứng riêng so với B4. S3 cuối dùng ngưỡng OOF, không dùng trọng số balanced hay under-sampling.
6. **Vì sao S3 có ngưỡng âm?** — Score SVM là khoảng cách có dấu, không phải xác suất. Ngưỡng âm cho phép một số score âm nhẹ vẫn được nhận là spam.
7. **S3 có chắc tốt nhất?** — Chỉ tốt nhất theo CV F1 trong bảy ứng viên đã thử. Test cho biết hiệu năng trên tập giữ riêng, không bảo đảm cho mọi bình luận tương lai.
8. **Đổi ngưỡng có đổi AP không?** — Không nếu score và thứ tự xếp hạng giữ nguyên; AP được tính từ score liên tục. Precision/Recall/F1 tại một ngưỡng cụ thể có thể đổi.
9. **B4, S1, S2, S3 khác nhau ở dữ liệu nào?** — B4/S1/S3 fit cuối trên toàn train, S1 thay trọng số, S3 thay ngưỡng dự đoán; S2 giảm ham của train. Validation/test giữ nguyên trong mọi đối chứng.
10. **Phi cần hỏi ai khi có thay đổi?** — Cương về nguồn CSV/token; Bảo về nhãn và phân tích lỗi chung; Phương về viBERT/checkpoint. Phi tự chịu trách nhiệm chia tập và chỉ số test của mô hình truyền thống; ghi rõ thay đổi và kết quả chạy trước khi đề nghị gộp.

## 6. Thứ tự tự học và tự kiểm

Đọc `training_utils.py` phần chia train/test và fold → `text_features.py` phần cấu hình hai lớp thư viện và `classifiers.py` phần tạo NB/SVM → `experiments.py` phần Pipeline/CV → `imbalance.py` phần S1/S2/S3 → Notebook 02 cell 13, 15, 16 → `evaluation.py` và mục test → `modeling/predict.py` cùng cell 32. Sau mỗi bước, tự trả lời: đầu vào là gì, hàm fit trên tập nào, đầu ra là record/model/score/file gì, và test có được dùng để chọn không.

## 7. Giải đáp đúng vai trò từng file sau khi đơn giản hóa

### `training_utils.py`: chia tập, không làm sạch lại dữ liệu

`norm_key` chuẩn hóa câu **chỉ để quyết định nhóm chia**: khác chữ hoa, số hoặc dấu câu nhưng cùng nội dung gần giống sẽ có cùng khóa. Đây là một dạng xử lý văn bản cho thiết kế thực nghiệm, không thay cột token dùng để huấn luyện. `split_train_test` nhận văn bản đã sạch và nhãn số, trả khóa cùng vị trí train/test. `grouped_folds` nhận **chỉ phần train**, trả năm cặp vị trí học/validation tính trong train. `bucket` dùng một phép băm ổn định; `salt` chỉ phân biệt lần chia test với lần chia CV, không phải mật khẩu hay mã hóa. Giữ quy tắc này để so mô hình truyền thống và viBERT trên cùng test; chia ngẫu nhiên từng dòng sẽ dễ đưa câu trùng sang cả hai phía. `np.array` ở đây chỉ đóng gói các vị trí để các bước sau lấy nhiều dòng một lần.

### `text_features.py`: tạo BoW và TF-IDF bằng scikit-learn

`CountVectorizer` và `TfidfVectorizer` giữ **trạng thái đã học** giữa `fit` và `transform`. Chúng tách từ đơn; khi CV chọn `(1,2)` còn thêm cặp từ đứng cạnh nhau như “lịch thi”. `fit` đếm **số bình luận** chứa từng đặc trưng, giữ đặc trưng có ở ít nhất hai bình luận và học IDF nếu dùng TF-IDF. `transform` biến câu theo từ vựng đã học; TF-IDF còn tính `(1+log(tf))×idf` và chuẩn hóa mỗi dòng. Đầu ra là ma trận thưa. Xem [giải thích phương thức thư viện](PHI_HIEU_4_LOP_SCIKIT_LEARN.md).

`make_vectorizer` chọn `CountVectorizer` cho BoW hoặc `TfidfVectorizer` cho TF-IDF; cả hai được `Pipeline` fit riêng trong từng fold. Hàm chỉ tạo đối tượng chưa fit và trả lưới n-gram cho CV.

Ví dụ nhỏ: với ba câu train `mua sim`, `mua sim`, `lịch thi` và `min_df=2`, từ vựng chỉ còn `mua`, `sim` (thêm `mua sim` nếu bật bigram). Từ `lịch` và `thi` chỉ xuất hiện trong một câu nên bị loại. BoW của câu đầu có giá trị 1 ở cột `mua` và `sim`. TF-IDF lấy các số đếm ấy, nhân IDF đã học rồi chuẩn hóa; một câu mới chỉ dùng **đúng các cột đã học**, không tự mở rộng từ vựng.

### `classifiers.py`: chọn thuật toán, chưa huấn luyện

`make_classifier` chỉ trả **đối tượng NB hoặc SVM chưa fit** và một `grid` các giá trị cần thử. NB học mức phổ biến đặc trưng theo nhãn; `alpha` 0,1/0,3/1,0 là các mức làm trơn để không cho từ chưa gặp xác suất 0. SVM học ranh giới tuyến tính; `C` 0,1/0,3/1,0/3,0 là bốn mức phạt lỗi. Các giá trị này là **phạm vi thử của nhóm**, không phải chân lý cố định; CV chọn mức phù hợp với train. `class_weight=None` là bản cơ sở, `balanced` chỉ dùng trong S1. File này chưa nhận vector dữ liệu và chưa tạo mô hình đã học; việc đó xảy ra ở `experiments.py`.

### `experiments.py`: nơi hai bước được ghép và thực sự fit

`make_pipe` ghép bộ tạo vector và bộ phân loại theo thứ tự, chưa học gì. `X` truyền vào `fit_experiment` là **chuỗi token của train**, chưa phải vector; `y` là nhãn 0/1 tương ứng; `folds` là các vị trí học/validation bên trong train. `GridSearchCV` liệt kê từng tổ hợp n-gram và alpha/C, fit lại **cả vectorizer lẫn bộ phân loại** trên bốn fold, chấm fold còn lại, lặp năm lượt rồi lấy trung bình Precision/Recall/F1 spam. Nó chọn tổ hợp bằng CV F1 và tự fit lại trên toàn train. Record trả về chứa tên cấu hình, điểm **CV**, tham số thắng, ngưỡng mặc định và `_model` đã fit; chưa có điểm test. Tên `mean_test_f1` của scikit-learn ở đây chỉ **validation trong CV**, không phải test riêng của project.

### `imbalance.py`: ba cách riêng và điểm validation

S1 gọi lại thí nghiệm SVM với trọng số lỗi balanced; S2 giữ mọi spam, chọn tối đa hai ham cho mỗi spam **chỉ từ fold train**; S3 giữ dữ liệu B4, lấy score ngoài fold để tìm ngưỡng. `ParameterGrid` chỉ trải lưới của S2 thành **2 lựa chọn n-gram × 4 lựa chọn C = 8 tổ hợp**, rồi vòng lặp thử từng tổ hợp; nó không tự fit. `mean_cv_scores` đo Precision/Recall/F1 trên **fold validation thuộc train** để chọn S2/S3. Đây không phải điểm test. Mỗi chiến lược fit lại mô hình cuối sau khi chọn; không cộng S1, S2, S3 vào một mô hình.

### `evaluation.py`: đo test sau lựa chọn

`record` nhận vào đã có điểm CV, model đã fit và ngưỡng. `X` là **chuỗi token của test**; pipeline trong model tự biến chúng thành vector. `y` là nhãn thật của đúng các câu test đó. `predict_record` lấy xác suất spam nếu là NB, hoặc decision score có dấu nếu là SVM, rồi so với ngưỡng để tạo nhãn 0/1. Cần giữ **score liên tục** vì AP/đường PR đo chất lượng xếp hạng ở nhiều ngưỡng; còn Precision/Recall/F1 của một cấu hình dùng nhãn tại ngưỡng đã chốt. `evaluate` thêm Accuracy, Precision, Recall, F1, AP của **test** vào record, không fit và không chọn lại model. `boot_ci` rút 500 tập có hoàn lại từ test để ước lượng khoảng dao động F1; không huấn luyện lại.

### `predict.py`: dùng mô hình đã lưu, không tạo mô hình mới

Mô hình thành hình khi quá trình CV đã chọn tham số và **fit lại trên toàn train**; sau đánh giá, Notebook 02 hoặc script chạy thẳng lưu pipeline đã fit, ngưỡng và lựa chọn từ dừng vào `.joblib`. `predict_spam` nhận một câu hoặc danh sách câu thô cùng bundle đã nạp, áp đúng tiền xử lý đã chọn, để vectorizer đã học biến câu thành số, lấy score, so ngưỡng và trả bảng `text`, `score`, `score_type`, `du_doan`. Hàm không biết nhãn thật của câu mới, không đo F1 và không học lại.
