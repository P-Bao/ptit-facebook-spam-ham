# BƯỚC 06 — `evaluation.py`: Đánh giá, so sánh và giải thích kết quả Spam/Ham PTIT

> **Mục tiêu học:** Biết chính xác khi nào mô hình được chọn, Test được dùng lúc nào; giải thích từng dòng `evaluation.py`, các hàm `scikit-learn` tính metric; đọc được 5 biểu đồ thật trong Notebook 02; nhận xét B1–B4, B4/S1/S2/S3, S3 so với viBERT và lý giải lựa chọn cuối cùng.
>
> **Nguồn chính:** `pipeline/modeling/evaluation.py` (bản tải lên `evaluation(1).py`); Notebook `02_thuc_nghiem_va_demo(1).ipynb` cell 10, 13, 16, 19–30, 33–34; các file `experiments.py`, `imbalance.py`; tài liệu `PHI_HIEU_4_LOP_SCIKIT_LEARN.md` về NB/SVM. **Các điểm số, ma trận và hình trong tài liệu là kết quả đã lưu của Notebook, KHÔNG phải do tài liệu này huấn luyện lại.**
>
> **Phạm vi trách nhiệm:** Bạn phụ trách đánh giá mô hình truyền thống, giải thích kết quả và giao mô hình đã chọn cho `predict.py`. viBERT chỉ được nhắc tới để xác định vị trí so sánh chung trong nhóm.

---

## Mục lục

1. Bản đồ luồng — `experiments` → `imbalance` → `evaluation` → `predict`
2. CV vs Test: nguyên tắc chọn mô hình
3. Dữ liệu Test và baseline luôn dự đoán Ham
4. Khái niệm: TP, TN, FP, FN và confusion matrix
5. Lý thuyết các metric và ví dụ tính tay
6. Mã `evaluation.py`: imports, `predict_record`, `evaluate`, `boot_ci`
7. Đi sâu vào hàm thư viện scikit-learn (`accuracy_score`, `precision_score`, `recall_score`, `f1_score`, `average_precision_score`)
8. Kết nối với Notebook 02 cell 16 và cell 20
9. Nhận xét bốn tổ hợp cơ sở B1–B4
10. Nhận xét B4, S1, S2, S3
11. Đọc biểu đồ kết quả cân bằng — cell 22
12. Đọc hai nhóm confusion matrix — cell 24
13. Đọc Precision–Recall curve — cell 26
14. AP, threshold và vì sao B4/S3 cùng đường PR
15. Bootstrap 95% CI cho F1 — lý thuyết, code, giới hạn
16. Phân tích trọng số đặc trưng và lỗi FP/FN — cell 30
17. So sánh với viBERT — cell 28
18. Nhận xét mấu chốt: vì sao S3, CV vs Test, giới hạn kết luận
19. Code tự thực hành để kiểm chứng từ `record` và confusion matrix
20. Những câu hỏi thường gặp khi bảo vệ + trả lời mẫu
21. Bản tóm tắt học thuộc trong 90 giây

---

## 1. Bản đồ luồng và nhiệm vụ của file

```text
Notebook 02: sau tiền xử lý + chia tập
  |
  +--> 30.759 dòng TRAIN ──> grouped_folds() ──> 5 lượt CV
  |                                |
  |   experiments.py:              |
  |   B1 BoW+NB, B2 BoW+SVM, B3 TF-IDF+NB, B4 TF-IDF+SVM
  |                                |
  |   imbalance.py: B4, S1 balanced, S2 undersampling, S3 đổi threshold
  |                                |
  |   Chọn best = argmax(CV F1 spam) ──> S3 ──> CHỐT MÔ HÌNH
  |
  +--> 7.474 dòng TEST (giữ riêng) ──────────────────────────+
                                                             |
           evaluation.py                                     |
           predict_record(record, X_test)  <------------------+
                     |
                     +--> _pred: nhãn 0/1
                     +--> _score: số liên tục
                     |
           evaluate(record, X_test, y_test)
                     +--> Accuracy / Precision / Recall / F1 / AP
                     +--> record bị bổ sung kết quả (không fit)
                     |
           boot_ci(y_test, record['_pred'])
                     +--> khoảng bootstrap tham khảo 95% cho F1
                     |
           Notebook: metrics / bar charts / confusion matrices / PR / errors
                     |
           Notebook lưu model bundle → modeling/predict.py dùng cho câu mới
```

**Phân biệt:** `evaluation.py` không chọn tham số, không huấn luyện, không tạo mới vocabulary, không làm cân bằng và không lưu model. File đánh giá **các mô hình đã fit**. Trong Notebook, sau khi chọn `best` bằng CV, chương trình vẫn chấm **bảy ứng viên truyền thống đã có sẵn** trên cùng Test để báo cáo so sánh; không dùng bảng đó để chọn lại model.

Bảy ứng viên là: `RESULTS=[B1,B2,B3,B4]` và `SUPPLEMENTARY=[S1,S2,S3]`; `CANDIDATES=RESULTS+SUPPLEMENTARY`. Mảng `STRATEGIES=[B4,S1,S2,S3]` chỉ tập trung phân tích tác động cân bằng trên TF-IDF+SVM.

## 2. CV và Test: học, chọn, rồi mới đánh giá

### 2.1. Phân biệt chính xác

| Dữ liệu | Mục đích | Ai sử dụng? | Có được dùng chọn mô hình? |
|---|---|---|---|
| Train | Học vocabulary/IDF; trọng số NB/SVM | `Pipeline.fit()` trên fold Train hoặc toàn Train | Có, dùng để phát triển mô hình |
| Fold validation | Kiểm tra một cấu hình trên dữ liệu chưa được dùng fit trong lượt đó | `GridSearchCV`, S2/S3 | **Có**, là nguồn điểm CV |
| Test giữ riêng | Ước lượng hiệu năng sau khi quyết định xong mô hình | `evaluate`, biểu đồ, báo cáo | **Không**, nếu muốn giữ đánh giá độc lập |

Trong dự án, group split theo khóa nội dung giúp bình luận cùng khóa không nằm ở hai tập, không phải phép chia ngẫu nhiên trực tiếp từng dòng. Bởi vậy không thể bảo đảm Train/Test hoàn toàn không có nội dung gần nghĩa nhau; chỉ giảm rủi ro theo định nghĩa khóa nhóm.

### 2.2. Vì sao lấy F1 spam làm mục tiêu?

Dataset mất cân bằng: Ham nhiều hơn Spam. Nếu dự đoán toàn Ham thì Accuracy vẫn có thể cao nhưng Recall Spam bằng 0. Nhóm ưu tiên **F1 của lớp spam** vì đó là trung bình điều hòa giữa Precision Spam và Recall Spam. F1 cao đòi hỏi tránh đồng thời quá nhiều cảnh báo nhầm và quá nhiều Spam bị bỏ sót; nhưng F1 **không** mã hóa chi phí riêng của từng loại lỗi. Với ứng dụng thật, có thể cần tiêu chí khác theo nghiệp vụ.

### 2.3. Vì sao phải chốt bằng CV F1 chứ không dùng Test F1?

- Cell 13 so các tổ hợp và stopwords bằng CV, chỉ dùng Train.
- Cell 16 so B4, S1, S2, S3 và chọn `best = max(CANDIDATES, key=lambda r: r['CV F1 (spam)'])` **trước** phần Test.
- Cell 20 chấm Test với các ứng viên đã cố định để **báo cáo**, không sửa lựa chọn.
- Nếu dùng điểm Test để đổi mô hình / đổi ngưỡng rồi vẫn gọi Test đó là phép đánh giá độc lập, sẽ tạo **selection bias / test leakage**: ta đã điều chỉnh quyết định dựa trên dữ liệu cần giữ làm bài kiểm tra cuối cùng.

**Một sắc thái quan trọng:** CV F1 sau khi thử nhiều tham số cũng có thể lạc quan. S3 dùng chính OOF validation để vừa chọn ngưỡng vừa ghi nhận F1; đây không phải nested CV hoàn toàn độc lập. Vì vậy có thể thấy Test F1 thấp hơn CV F1. Không thể chỉ nhìn sự chênh lệch để khẳng định overfitting là nguyên nhân duy nhất.

## 3. Dữ liệu Test và đối chứng luôn Ham

Theo cell 11 và 20:

| Tập | Tổng | Ham (0) | Spam (1) | Spam % |
|---|---:|---:|---:|---:|
| Train | 30.759 | 26.384 | 4.375 | 14,22% |
| **Test** | **7.474** | **6.523** | **951** | **12,72%** |

Nếu một classifier rất lười **luôn dự đoán Ham**:

- Dự đoán đúng 6.523/7.474 bình luận → **Accuracy ≈ 87,28%**.
- Không phát hiện Spam nào → **TP=0; Recall=0; F1 Spam=0**.
- Kết luận: **Accuracy 87% có thể hoàn toàn vô dụng cho nhiệm vụ phát hiện Spam**.

Đây là lý do các bảng tập trung vào Precision, Recall, F1 và AP của lớp dương `1=spam`.

## 4. TP, TN, FP, FN và confusion matrix

### 4.1. Bốn trường hợp

- **TP — True Positive:** nhãn thật Spam và dự đoán Spam (phát hiện đúng).
- **TN — True Negative:** nhãn thật Ham và dự đoán Ham (giữ đúng bình luận thường).
- **FP — False Positive:** thật Ham nhưng dự đoán Spam (**cảnh báo nhầm**; có thể ẩn nhầm bình luận sinh viên).
- **FN — False Negative:** thật Spam nhưng dự đoán Ham (**bỏ sót Spam**).

Với `confusion_matrix(y_true, y_pred, labels=[0,1])`, hàng là **nhãn thật**, cột là **nhãn dự đoán**:

|  | Dự đoán Ham (0) | Dự đoán Spam (1) |
|---|---:|---:|
| **Thật Ham (0)** | TN | FP |
| **Thật Spam (1)** | FN | TP |

### 4.2. Ma trận thật của S3 (cell 24)

|  | Dự đoán Ham | Dự đoán Spam | Tổng |
|---|---:|---:|---:|
| **Thật Ham** | TN = **6.396** | FP = **127** | 6.523 |
| **Thật Spam** | FN = **297** | TP = **654** | 951 |
| **Tổng** | 6.693 | 781 | **7.474** |

Từ đây suy ra:

- Phát hiện đúng **654/951 Spam = 68,77%**.
- **297/951 Spam = 31,23%** bị bỏ sót.
- Trong **781 câu bị gắn cờ Spam**, có **654 thật Spam**, còn **127 là Ham**.
- Tỷ lệ Ham bị cảnh báo nhầm = `127/6523 ≈ 1,95%` (FPR); con số 127 vẫn có ý nghĩa vận hành nếu áp dụng ở quy mô lớn.
- Tổng dự đoán đúng = `6396 + 654 = 7050`; sai = `127+297=424`.

## 5. Các chỉ số: công thức, ý nghĩa, ví dụ thật

### 5.1. Accuracy — tỷ lệ đúng toàn bộ

`Accuracy = (TP+TN)/(TP+TN+FP+FN)`.

S3: `(654+6396)/7474 = 7050/7474 ≈ 0,9433` → 94,33%.

Không đủ dùng một mình vì lớp Ham chiếm đa số.

### 5.2. Precision Spam — độ chính xác trong nhóm bị gắn cờ Spam

`Precision = TP/(TP+FP)`.

S3: `654/(654+127)=654/781 ≈ 0,8374` → 83,74%.

Đọc bằng tiếng Việt: **Trong 100 bình luận mô hình báo Spam, khoảng 84 bình luận thật sự là Spam** (diễn giải theo tỷ lệ đo trên Test, không bảo đảm mọi 100 câu mới cũng như vậy).

Tăng Precision thường đòi hỏi cẩn trọng hơn khi báo Spam, có thể bỏ sót nhiều Spam hơn.

### 5.3. Recall Spam — khả năng tìm được Spam thực tế

`Recall = TP/(TP+FN)`.

S3: `654/(654+297)=654/951≈0,6877` → 68,77%.

Đọc bằng tiếng Việt: **Trong 100 bình luận thật sự là Spam, mô hình phát hiện khoảng 69 bình luận**.

Tăng Recall có thể khiến FP tăng nếu ngưỡng được hạ xuống.

### 5.4. F1 Spam — trung bình điều hòa của Precision và Recall

`F1 = 2PR/(P+R) = 2TP/(2TP+FP+FN)`.

S3: `2×654/(2×654+127+297)=1308/1732≈0,7552`.

F1 nhạy với bên thấp hơn: Precision cao nhưng Recall thấp vẫn có thể kéo F1 xuống. F1 không tính trực tiếp TN, do đó phù hợp hơn Accuracy để so sánh khả năng phát hiện lớp Spam trong bài này.

### 5.5. AP — Average Precision

**AP không phải Accuracy, cũng không phải Precision tại ngưỡng hiện tại.** AP tổng hợp chất lượng **xếp hạng bằng score** khi quét các ngưỡng và thu các cặp Precision–Recall. Trong scikit-learn, AP là tổng có trọng số theo các bước tăng Recall, không đơn giản là diện tích hình thang của đường PR.

Dạng ý tưởng: `AP = tổng (Recall_k - Recall_{k-1}) × Precision_k` trên các mốc xếp hạng phù hợp; thứ tự mốc phụ thuộc quy ước quét điểm.

- `MultinomialNB`: score là `predict_proba(X)[:,1]`, **ước lượng xác suất theo mô hình NB**, không mặc định là xác suất đã hiệu chuẩn.
- `LinearSVC`: score là `decision_function(X)`, **không phải xác suất**.
- Score càng cao thường càng nghiêng về Spam. `average_precision_score(y_true, scores)` đánh giá mức độ các Spam thật được xếp ở phần score cao.

**Nhớ:** đổi threshold mà giữ nguyên score → Precision/Recall/F1 tại một điểm thay đổi, nhưng đường PR tổng thể và AP không đổi.

### 5.6. Các chỉ số phụ để hiểu sâu

- **Specificity / TNR:** `TN/(TN+FP)` — tỷ lệ Ham được nhận diện đúng. S3: `6396/6523 ≈ 98,05%`.
- **FPR:** `FP/(FP+TN)`. S3 ≈ 1,95%; bằng `1-specificity`.
- **FNR:** `FN/(FN+TP)`. S3 ≈ 31,23%; bằng `1-recall`.
- **Support:** số câu thật của mỗi lớp (Ham 6.523, Spam 951). Nhìn số mẫu để tránh diễn giải chỉ số mà thiếu ngữ cảnh.

Các chỉ số phụ không được `evaluate()` lưu riêng trong record, nhưng có thể tính từ confusion matrix.

## 6. Mã nguồn `evaluation.py`: giải thích đến từng lời gọi

### 6.1. Toàn bộ code gốc (phiên bản mới)

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
        scores = model.predict_proba(X)[:, 1]
        threshold = record.get("_threshold", 0.5)
        predictions = (scores >= threshold).astype(int)
    else:
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
    record["Accuracy"] = accuracy_score(y, pred)
    record["Precision"] = precision_score(y, pred, zero_division=0)
    record["Recall"] = recall_score(y, pred, zero_division=0)
    record["F1"] = f1_score(y, pred, zero_division=0)
    record["AP"] = average_precision_score(y, score)
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
        indices = rng.integers(0, len(y), len(y))
        values.append(f1_score(y[indices], pred[indices], zero_division=0))
    return np.percentile(values, [2.5, 97.5])
```

### 6.2. `predict_record(record, X)` — đầu vào và đầu ra

**Đầu vào:**

- `record` là dictionary của một mô hình đã fit, ví dụ `{'Mã':'S3', '_model':fitted_pipeline, '_threshold':-0.1776, ...}`.
- `X` là **các chuỗi token** dành cho Test, KHÔNG phải ma trận số đã vector hóa bên ngoài; `Pipeline` tự gọi `vec.transform(X)`.

**Đầu ra:** hai mảng cùng chiều dài với `X`:

- `predictions`: nhãn nguyên `0` (Ham) hoặc `1` (Spam).
- `scores`: điểm liên tục cho từng câu (NB: xác suất Spam; SVM: decision score).

**Giải thích từng lệnh:**

1. `model = record['_model']`: lấy **Pipeline đã học** được lưu sau CV; trong Pipeline có vectorizer và classifier cùng cấu hình.
2. `hasattr(model,'predict_proba')`: kiểm tra Pipeline có phương thức dự đoán xác suất không; với NB có, với LinearSVC không.
3. NB: `model.predict_proba(X)` trả ma trận kích thước `(số_câu, 2)` với hai cột theo `classes_=[0,1]`; `[:,1]` chọn **xác suất Spam** của tất cả câu.
4. `record.get('_threshold',0.5)`: dùng ngưỡng được lưu, nếu không có thì mặc định 0,5.
5. `(scores >= threshold).astype(int)`: tạo mảng Boolean rồi đổi thành nhãn số: `True -> 1`, `False -> 0`.
6. SVM: `model.decision_function(X)` trả mảng score có dấu; 0 là ngưỡng mặc định.
7. `(scores > threshold).astype(int)`: dùng `>` cho SVM đúng như code, kể cả S3 khi ngưỡng âm; tại đúng ngưỡng, dự đoán Ham.
8. `return predictions, scores`: phải giữ cả **nhãn** cho F1/confusion matrix và **score** cho AP/đường PR.

**Lưu ý nhỏ nhưng quan trọng:** Code chọn phương thức dựa trên `hasattr(model,'predict_proba')`; đây là cơ chế phù hợp hai thuật toán hiện có, không có nghĩa mọi classifier Python đều trả score theo đúng định dạng NB/SVM.

### 6.3. Ví dụ `predict_record` dùng số đơn giản

**Ví dụ NB** (giả lập output của `predict_proba`; không phải dữ liệu thực của Notebook):

```text
scores = [0.10, 0.72, 0.49, 0.91]
threshold = 0.50
scores >= threshold → [False, True, False, True]
predictions → [0, 1, 0, 1]
```

**Ví dụ SVM** (giả lập output của `decision_function`):

```text
scores = [-1.20, -0.10, 0.35, 0.80]
threshold B4 = 0.0
B4 → [0, 0, 1, 1]
threshold S3 = -0.1776
S3 → [0, 1, 1, 1]
```

Điểm `-0.10` chưa đủ để B4 gọi Spam, nhưng vượt `-0.1776`, do đó S3 gọi Spam. **Score `-0.10` không phải -10% xác suất.**

### 6.4. `evaluate(record, X, y)` — chấm Test và ghi vào record

**Đầu vào:** `record` đã fit, `X` danh sách chuỗi token Test, `y` là nhãn thật Test cùng thứ tự.

`pred, score = predict_record(record, X)` lấy nhãn/score; sau đó:

| Code | Hàm thư viện | Ý nghĩa / Kết quả |
|---|---|---|
| `accuracy_score(y, pred)` | tỷ lệ nhãn đúng toàn Test | `record['Accuracy']` |
| `precision_score(y,pred,zero_division=0)` | TP/(TP+FP), lớp dương 1 | `record['Precision']` |
| `recall_score(y,pred,zero_division=0)` | TP/(TP+FN), lớp dương 1 | `record['Recall']` |
| `f1_score(y,pred,zero_division=0)` | trung bình điều hòa P/R | `record['F1']` |
| `average_precision_score(y,score)` | tổng hợp xếp hạng các Spam qua nhiều ngưỡng | `record['AP']` |
| `record['_pred']=pred` | lưu mảng nhãn dự đoán | vẽ confusion matrix và phân tích lỗi |
| `record['_score']=score` | lưu score liên tục | vẽ PR curve, xem ca khó |

**Quan trọng:** `evaluate()` **không có `return`**, mặc định trả `None`, nhưng **sửa dictionary `record` ngay tại chỗ**. Không viết `record = evaluate(record,X,y)` vì lúc đó `record` sẽ thành `None`.

**Ví dụ S3:** Sau `evaluate(best,X_test,y_test)`, có thể đọc `best['F1']==0.7552...`, `len(best['_pred'])==7474`, `best['_score']` là mảng số thực. Mô hình đã fit không đổi tham số và không học lại từ Test.

### 6.5. `boot_ci(y, pred, n=500, seed=0)` — bootstrap F1

**Đầu vào:** `y` thật và `pred` đã dự đoán (cùng độ dài), số lần rút mẫu `n=500`, seed mặc định 0.

1. `rng = np.random.default_rng(seed)` tạo bộ sinh số ngẫu nhiên tái lập.
2. `rng.integers(0,len(y),len(y))` lấy `len(y)` chỉ số trong `[0,len(y)-1]` **có hoàn lại**; chỉ số có thể lặp hoặc không xuất hiện.
3. Chọn cùng `indices` cho cả `y` và `pred` để cặp nhãn thật – dự đoán không bị lệch hàng.
4. Tính F1 trên mỗi tập rút mẫu và lưu vào `values`.
5. `np.percentile(values,[2.5,97.5])` lấy hai phân vị 2,5 và 97,5 để có **khoảng bootstrap percentile tham khảo 95% cho F1 trên Test**.

**Không** huấn luyện lại NB/SVM trong 500 lượt. Khoảng này không bao trùm mọi nguồn bất định như thay đổi tập Train, thay nhãn, sự dịch chuyển dữ liệu trên Facebook, và không phải phép kiểm định S3 có ý nghĩa thống kê hơn S2.

## 7. Bên trong các hàm metric của scikit-learn: cách hiểu theo công thức và đầu vào

Tài liệu nhóm giải thích rất sâu bốn lớp `CountVectorizer`, `TfidfVectorizer`, `MultinomialNB`, `LinearSVC`. Ở `evaluation.py`, thư viện được gọi qua `sklearn.metrics`. Các hàm score dưới đây tính toán **từ nhãn/score đã có**, không gọi lại vectorizer hay fit classifier.

### 7.1. `accuracy_score(y_true, y_pred)`

- Kiểm tra hai dãy cùng chiều dài và dạng mục tiêu phân loại.
- So sánh từng cặp nhãn: `y_true[i]==y_pred[i]`.
- Với mặc định `normalize=True`, trả tỷ lệ đúng toàn bộ; nếu `normalize=False` thì trả số lượng câu đúng.
- S3: 7050 cặp trùng trên 7474 cặp → khoảng 0,9433.

### 7.2. `precision_score`, `recall_score`, `f1_score`

- Cùng nhận hai mảng nhãn thật và dự đoán.
- Trong bài toán nhị phân, mặc định `average='binary'` và `pos_label=1`: đánh giá **Spam**, không phải tính trung bình cả hai lớp.
- Precision dùng TP và FP; Recall dùng TP và FN; F1 kết hợp hai con số.
- `zero_division=0` nói rằng nếu mẫu số bị bằng 0 thì trả 0 thay vì giá trị không xác định; chẳng hạn mô hình không dự đoán Spam nào sẽ có Precision=0.
- Gợi ý kiểm chứng: `sklearn.metrics.confusion_matrix(y,pred,labels=[0,1])` rồi thế bốn số TN,FP,FN,TP vào công thức.

### 7.3. `average_precision_score(y_true, y_score)`

- `y_score` phải là số liên tục phản ánh độ nghiêng về Spam, **không** phải mảng nhãn 0/1 nếu muốn đánh giá xếp hạng đầy đủ.
- Theo từng score từ cao xuống thấp, cân nhắc các mốc ngưỡng, cập nhật Precision/Recall và tổng hợp thành AP.
- Không giống Average Accuracy, không dùng threshold đã chốt của S3 để tính AP.
- Trên Notebook: B4 AP=0,8117 và S3 AP=0,8117, dù S3 F1 khác B4.

### 7.4. `confusion_matrix(y_true, y_pred, labels=[0,1])`

- Đếm số trường hợp tương ứng từng ô `[true_class,pred_class]`.
- Với `labels=[0,1]`, trả mảng 2×2 là `[[TN,FP],[FN,TP]]`.
- `.ravel()` biến 2×2 thành chuỗi 4 số theo thứ tự `tn,fp,fn,tp`.
- Ví dụ S3: `[[6396,127],[297,654]]` → `tn=6396`, `fp=127`, `fn=297`, `tp=654`.

### 7.5. `precision_recall_curve(y_true,y_score)`

- Trả `(precision, recall, thresholds)` cho các mốc score.
- Khi vẽ `ax.plot(recall, precision)`, trục X là Recall Spam, trục Y là Precision Spam.
- Đường PR dựa vào `y_score`, không vào `record['_pred']` tại ngưỡng cụ thể.
- Điểm hoạt động hiện tại có thể biểu diễn bằng `ax.scatter(record['Recall'],record['Precision'])`.

### 7.6. `np.percentile()` và bootstrap

- `np.percentile(values,[2.5,97.5])` **không** cộng/trừ cố định 1,96 độ lệch chuẩn như công thức xấp xỉ chuẩn.
- Nó lấy trực tiếp các phân vị của 500 F1 bootstrap đã tính.
- Với `seed=0`, lệnh lấy mẫu có thể lặp lại nhất quán trên cùng dữ liệu và môi trường.

### 7.7. Giới hạn trích nguồn thư viện

Tôi giải thích **đường xử lý quan trọng và ý nghĩa hàm** dựa trên các lệnh được dự án sử dụng, không chép nguyên toàn bộ file scikit-learn. Nếu muốn xem thân hàm Python gốc trong VS Code, Ctrl+Click từ `evaluation.py` vào `sklearn.metrics` (đặc biệt `sklearn/metrics/_classification.py` cho Accuracy/P/R/F1/Confusion Matrix, và `sklearn/metrics/_ranking.py` cho AP/PR). Vị trí dòng phụ thuộc phiên bản; Notebook dùng scikit-learn 1.6.1.

## 8. Cách các cell Notebook gọi `evaluation.py`

### 8.1. Cell 16 — chọn best trước Test

```python
STRATEGIES = [focus, *SUPPLEMENTARY]
CANDIDATES = RESULTS + SUPPLEMENTARY
best = max(CANDIDATES, key=lambda r: r["CV F1 (spam)"])
```

Điểm CV chọn S3 (TF-IDF + SVM, C=0,3, unigram+bigram, ngưỡng -0,1776). Thông tin này đã chốt **trước** cell 20.

### 8.2. Cell 20 — đánh giá Test trên các ứng viên đã fit

```python
for record in CANDIDATES:
    evaluate(
        record,
        TEXT[record["Tiền xử lý"]].iloc[test_idx].to_numpy(),
        y_test
    )

METRICS = ["Accuracy", "Precision", "Recall", "F1", "AP"]
```

**Từng thành phần của dòng đánh giá:**

- `CANDIDATES`: danh sách dictionary cho B1,B2,B3,B4,S1,S2,S3.
- `record['Tiền xử lý']`: dùng đúng nhánh token giữ/bỏ stopwords mà record đã học, tránh đổi tiền xử lý khi đánh giá.
- `TEXT[...].iloc[test_idx]`: lấy những chuỗi token Test, theo **index của toàn CSV**.
- `.to_numpy()`: chuyển Series pandas thành mảng để Pipeline tiếp nhận.
- `y_test`: nhãn thật 0/1 tương ứng cùng dòng.
- `evaluate(...)`: dự đoán và gắn 5 metric + `_pred` + `_score` vào record.
- `pd.DataFrame` ở cell 20 tạo bảng tổng hợp, không hề fit lại.

### 8.3. Cell 22 — các chiến lược và hai biểu đồ thanh ngang

Lặp qua `STRATEGIES`, gọi `confusion_matrix` để lấy FP/FN, gộp các metric và vẽ bằng Matplotlib, lưu `imbalance_results.png`.

### 8.4. Cell 24 — hai hình confusion matrix

`plot_confusions(RESULTS,...)` tạo 4 ma trận cơ sở. `plot_confusions(STRATEGIES,...)` tạo 4 ma trận B4/S1/S2/S3. Màu xanh đậm là số lượng ô lớn, **không có nghĩa ô đó 'tốt'**. Chú ý nhãn trục và các tỷ lệ phần trăm được chuẩn hóa **theo hàng** (mỗi lớp thật có tổng 100%).

### 8.5. Cell 26 — bootstrap CI và hai biểu đồ PR

Chạy `boot_ci` cho 7 mô hình rồi vẽ đường PR theo `record['_score']`. Nhánh SVM riêng được vẽ cho `[B4,S1,S2]`; B4 và S3 có đường trùng nhau, nhưng có thêm **hai dấu điểm hoạt động** (B4 ngưỡng 0 và S3 ngưỡng -0,1776).

### 8.6. Cell 30 — đặc trưng và ví dụ sai

Lấy `vec.get_feature_names_out()` cùng `clf.coef_.ravel()` (SVM) để liệt kê các n-gram có hệ số dương/âm lớn. Tạo bảng lỗi Test bằng `raw.iloc[test_idx]`, `best['_pred']`, `best['_score']`; lọc FP và FN.

## 9. Bảng đánh giá bốn tổ hợp cơ sở — số liệu thật cell 20

| Mô hình | CV F1 | Accuracy Test | Precision | Recall | F1 Test | AP |
|---|---:|---:|---:|---:|---:|---:|
| B1 — BoW + NB | 0,6077 | 0,9141 | 0,7441 | 0,4953 | 0,5947 | 0,7320 |
| B2 — BoW + SVM | 0,7087 | 0,9241 | 0,7110 | 0,6803 | 0,6953 | 0,7493 |
| B3 — TF-IDF + NB | 0,5959 | 0,9257 | 0,8402 | 0,5142 | 0,6380 | 0,7988 |
| **B4 — TF-IDF + SVM** | **0,7736** | **0,9419** | **0,8818** | **0,6278** | **0,7334** | **0,8117** |

### B1 — BoW + NB

- F1 Test 0,5947, thấp nhất nhóm cơ sở.
- Recall 49,53%: chỉ phát hiện khoảng một nửa số Spam thật.
- `FN=480` chứng minh mô hình bỏ sót rất nhiều Spam; dù Accuracy 91,41% nhìn khá cao, hiệu quả phát hiện Spam chưa tốt.

### B2 — BoW + SVM

- F1 0,6953 và Recall 68,03%; cao hơn đáng kể B1.
- Đổi NB sang SVM (trong cùng đặc trưng BoW) giúp giảm FN từ 480 xuống 304, nhưng FP tăng từ 162 lên 263.
- Nhận xét: phát hiện Spam tốt hơn nhưng cảnh báo nhầm Ham nhiều hơn.

### B3 — TF-IDF + NB

- Precision 84,02%, khá cao; FP chỉ 93.
- Recall 51,42%, FN=462; điều này cho thấy mô hình **bảo thủ khi báo Spam**.
- F1 0,6380, thấp hơn B2. Đặc biệt CV F1 B3 là 0,5959 nhưng Test F1 lại 0,6380 — chênh lệch dữ liệu đánh giá có thể làm thứ tự một số cặp thay đổi.

### B4 — TF-IDF + SVM

- F1 Test **0,7334**, cao nhất trong bốn mô hình cơ sở.
- Precision **88,18%**, cao nhất cả nhóm cơ sở; FP chỉ **80**, thấp nhất.
- Recall 62,78%, FN vẫn **354** → còn bỏ sót nhiều Spam.
- B4 được dùng làm **nền tảng** thử S1/S2/S3. Không được nói B4 phát hiện Spam nhiều nhất trong bốn mô hình: B2 có Recall cao hơn B4 (68,03% so với 62,78%).

**Kết luận cơ sở:** B4 cân bằng tốt nhất theo F1 của nhóm B1–B4 (cả CV và Test), đặc biệt mạnh ở hạn chế FP, nhưng còn yếu ở bỏ sót Spam; đó là động lực thử xử lý mất cân bằng.

## 10. So sánh B4 / S1 / S2 / S3 — kết quả thật cell 16 và 22

### 10.1. Điểm CV dùng để chọn

| Mã | Chiến lược | CV Precision | CV Recall | **CV F1** | Ngưỡng |
|---|---|---:|---:|---:|---:|
| B4 | Không điều chỉnh | 0,9001 | 0,6806 | 0,7736 | 0 |
| S1 | Trọng số balanced | 0,7233 | **0,7964** | 0,7580 | 0 |
| S2 | Under-sampling 2:1 | 0,8038 | 0,7276 | 0,7629 | 0 |
| **S3** | **Điều chỉnh ngưỡng** | 0,8490 | 0,7353 | **0,7876** | **−0,1776** |

`best = max(CANDIDATES, key=lambda r: r['CV F1 (spam)'])` nên **S3 được chọn**. Không chọn S1 vì Recall cao nhất, bởi mục tiêu đã chốt là **F1 Spam**, không đơn thuần tối đa Recall. Không chọn B4 vì Precision cao nhất, bởi nếu chỉ ưu tiên ít FP sẽ bỏ sót nhiều Spam hơn.

### 10.2. Điểm Test để báo cáo

| Mã | Accuracy | Precision | Recall | **F1** | AP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|
| B4 | 0,9419 | **0,8818** | 0,6278 | 0,7334 | 0,8117 | **80** | 354 |
| S1 | 0,9322 | 0,7139 | **0,7792** | 0,7451 | **0,8127** | 297 | **210** |
| S2 | 0,9394 | 0,7791 | 0,7308 | 0,7542 | 0,7826 | 197 | 256 |
| **S3** | **0,9433** | 0,8374 | 0,6877 | **0,7552** | 0,8117 | 127 | 297 |

### 10.3. Nhận xét từng chiến lược

**B4 — ít cảnh báo nhầm nhất:**

- Chỉ 80 Ham bị nhận nhầm Spam, Precision 88,18%.
- Đổi lại, bỏ sót 354 Spam; Recall chỉ 62,78%.
- Phù hợp hơn khi cực kỳ sợ báo nhầm, nhưng theo mục tiêu F1 đang dùng thì chưa tối ưu.

**S1 — bắt được nhiều Spam nhất:**

- Recall 77,92%; bỏ sót ít nhất (FN 210).
- Nhưng FP tăng mạnh lên 297, Precision giảm còn 71,39%; tức gần 3 trong 10 bình luận bị gắn cờ Spam là Ham (trên Test này).
- Nếu nghiệp vụ **ưu tiên tối đa bắt Spam** và chấp nhận nhiều cảnh báo nhầm, S1 là lựa chọn đáng cân nhắc. Trong nghiên cứu hiện tại, nó không có CV F1 cao nhất.

**S2 — thỏa hiệp ở mức trung gian:**

- Recall 73,08%, Precision 77,91%, F1 0,7542; FP 197, FN 256.
- Chỉ dùng 13.125 mẫu để fit cuối (thay vì 30.759), đánh đổi việc bỏ đi nhiều ví dụ Ham có thể mang thông tin.
- F1 Test gần S3, nhưng CV F1 thấp hơn S3 (0,7629 so với 0,7876).

**S3 — được chọn trong phạm vi mô hình truyền thống:**

- Chỉnh ngưỡng, không giảm mẫu huấn luyện và không thay C/n-gram đã chọn ở B4.
- F1 Test 0,7552, lớn nhất trong bốn chiến lược, nhưng **chỉ hơn S2 0,0010** — rất nhỏ.
- So B4: TP từ 597 lên 654 (**bắt thêm 57 Spam**); FN từ 354 xuống 297 (**bỏ sót ít hơn 57**); FP từ 80 lên 127 (**cảnh báo nhầm thêm 47 Ham**).
- F1 tăng 0,7334 → 0,7552; Recall tăng 62,78% → 68,77%, nhưng Precision giảm 88,18% → 83,74%.
- **Đây là đánh đổi, không phải cải thiện mọi chỉ số.**

### 10.4. Tại sao S3 không thể gọi là tốt nhất tuyệt đối?

Vì 'tốt nhất' phải gắn với **tiêu chí và phạm vi**. Theo CV F1 trên bảy mô hình truyền thống đã thử, S3 dẫn đầu. Nhưng S1 có Recall cao nhất, B4 có Precision cao nhất, và viBERT trong bảng dự án có F1 Test cao hơn. Nếu thay tiêu chí thành chi phí bỏ sót Spam hoặc yêu cầu tránh báo nhầm, lựa chọn có thể khác.

## 11. Biểu đồ 1: So sánh metric và FP/FN (cell 22)

![Biểu đồ cell 22: Precision/Recall/F1 và số lỗi FP/FN](06_evaluation_assets/01_imbalance_metrics_and_errors.png)

**Cách đọc hình trái:**

- Trục ngang 0–1 là giá trị metric; mỗi dòng là một chiến lược.
- B4 có Precision cao nhất nhưng Recall thấp nhất trong bốn cấu hình.
- S1 có Recall cao nhất nhưng Precision thấp nhất.
- S2 ở giữa, S3 có F1 nhỉnh hơn S2.
- F1 không đơn giản bằng trung bình cộng P và R: nó là **trung bình điều hòa**.

**Cách đọc hình phải:**

- Thanh FP: **cảnh báo nhầm Ham**; thanh FN: **Spam bị bỏ sót**.
- B4: FP 80 thấp, nhưng FN 354 cao.
- S1: FP 297 cao nhất, FN 210 thấp nhất.
- S2: FP 197, FN 256.
- S3: FP 127, FN 297. So B4, S3 giảm FN với mức tăng FP tương đối có kiểm soát.

**Câu nhận xét ngắn khi bảo vệ:** “Biểu đồ cho thấy quan hệ đánh đổi: muốn tăng khả năng phát hiện Spam thường phải chấp nhận thêm cảnh báo nhầm. S1 ưu tiên Recall; B4 ưu tiên Precision; S3 đạt F1 cân bằng tốt nhất theo tiêu chí CV của nhóm.”

## 12. Biểu đồ 2 và 3: Confusion matrix (cell 24)

### 12.1. Ma trận của bốn mô hình cơ sở

![Cell 24: confusion matrix B1 B2 B3 B4](06_evaluation_assets/02_confusion_baselines.png)

| Mã | TN | FP | FN | TP | Nhận xét |
|---|---:|---:|---:|---:|---|
| B1 | 6.361 | 162 | 480 | 471 | Bỏ sót tới 480/951 Spam |
| B2 | 6.260 | 263 | 304 | 647 | Bắt thêm Spam nhưng nhiều FP hơn B1 |
| B3 | 6.430 | 93 | 462 | 489 | Thận trọng khi gọi Spam, recall thấp |
| B4 | 6.443 | **80** | 354 | 597 | Ít FP nhất, F1 tốt nhất trong baseline |

**Điểm dễ nhìn nhầm trên hình:** Hàng đầu của confusion matrix có lượng lớn Ham nên ô TN xanh rất đậm. Đừng kết luận mô hình tốt chỉ vì ô TN lớn; phải xem **hàng Spam**: TP bao nhiêu và FN bao nhiêu. Các phần trăm trong mỗi hàng tính theo số bình luận thật của lớp đó, không tính theo tổng toàn dataset.

### 12.2. Ma trận của bốn chiến lược TF-IDF + SVM

![Cell 24: confusion matrix B4 S1 S2 S3](06_evaluation_assets/03_confusion_strategies.png)

| Mã | TN | FP | FN | TP | Nhận xét |
|---|---:|---:|---:|---:|---|
| B4 | 6.443 | 80 | 354 | 597 | Ít báo nhầm, nhưng bỏ sót nhiều |
| S1 | 6.226 | 297 | 210 | **741** | Phát hiện nhiều Spam nhất |
| S2 | 6.326 | 197 | 256 | 695 | Trung gian giữa B4 và S1 |
| S3 | 6.396 | 127 | 297 | 654 | Điểm F1 tốt nhất, FP thấp hơn S1/S2 |

**Nhận xét thực tế:** S3 không có TP cao nhất; S1 có TP=741 nhưng FP=297. S3 có TP=654 và FP=127. Chênh lệch này giải thích vì sao không thể chỉ nhìn số Spam phát hiện mà phán mô hình tốt nhất.

## 13. Biểu đồ 4 và 5: Precision–Recall curves (cell 26)

### 13.1. Hình PR của bốn tổ hợp cơ sở

![Cell 26: PR baseline B1 B2 B3 B4](06_evaluation_assets/04_pr_baselines.png)

**Trục X**: Recall Spam; **trục Y**: Precision Spam. Khi tăng độ bao phủ Recall (muốn bắt thêm Spam), Precision thường giảm do thêm cả những câu khó hoặc Ham có score tương tự Spam.

AP là chỉ số tổng hợp khả năng xếp hạng:

| Mô hình | AP Test |
|---|---:|
| B1 | 0,7320 |
| B2 | 0,7493 |
| B3 | 0,7988 |
| **B4** | **0,8117** |

B4 có AP cao nhất trong baseline; B3 có AP cao hơn B2 dù F1 tại ngưỡng của B3 thấp hơn. Đây là minh chứng **AP đánh giá xếp hạng qua các ngưỡng; F1 đánh giá nhãn tại một ngưỡng cụ thể**.

Đường tham chiếu ngang “Tỷ lệ spam” ở khoảng **12,72%** tương ứng tỷ lệ Spam trong Test: một cơ sở để so với thứ hạng ngẫu nhiên khi xếp hạng độc lập với nhãn (mang tính tham chiếu trực quan, không phải chất lượng baseline luôn Ham).

### 13.2. Hình PR các chiến lược TF-IDF + SVM

![Cell 26: PR strategies + B4 S3 operating points](06_evaluation_assets/05_pr_strategies.png)

Điều quan trọng nhất ở hình này:

1. Có ba **đường** score: B4 không điều chỉnh, S1 trọng số balanced, S2 under-sampling.
2. **Không có đường riêng cho S3** vì S3 và B4 dùng cùng mô hình cuối và cùng `_score`; Notebook còn `assert np.allclose(focus['_score'],tuned['_score'])`.
3. Trên đường của B4, hai **điểm hoạt động** được đánh dấu: B4 dùng ngưỡng 0, S3 dùng ngưỡng -0,1776.
4. Hạ ngưỡng từ 0 xuống -0,1776 dẫn đến Recall tăng và Precision giảm: vị trí hoạt động dịch theo đường PR.
5. AP: B4 = 0,8117; S3 = 0,8117, **không đổi** do danh sách score/xếp hạng không đổi.
6. S1 AP ≈ 0,8127 nhỉnh hơn B4; S2 AP ≈ 0,7826 thấp hơn. Nhưng **S1 không thắng tiêu chí CV F1 tại ngưỡng cuối**, nên không được chọn.

**Câu trả lời khi giảng viên hỏi “Tại sao đổi threshold mà AP y nguyên?”**: “Vì AP đánh giá thứ tự xếp hạng từ score trên mọi ngưỡng. S3 chỉ đổi một ngưỡng ra quyết định mà không đổi score, nên đường PR và AP không đổi; chỉ Precision, Recall, F1 tại ngưỡng được chọn thay đổi.”

## 14. Threshold, đường PR và một tình huống ví dụ

Giả sử mô hình SVM cho các bình luận score `[0.80, 0.25, -0.10, -0.30]`; nhãn thật tương ứng `[1,0,1,0]`.

- **Ngưỡng 0:** dự đoán `[1,1,0,0]` → TP=1, FP=1, FN=1, TN=1 → P=0,5; R=0,5; F1=0,5.
- **Ngưỡng -0,1776:** dự đoán `[1,1,1,0]` → TP=2, FP=1, FN=0, TN=1 → P=2/3; R=1; F1=0,8.
- **AP của danh sách score không đổi** vì thứ tự 4 score không đổi; chỉ ngưỡng gắn nhãn đổi.

Ví dụ này chỉ minh họa nguyên tắc, **không phải 4 câu thực tế trong dataset**.

## 15. Bootstrap 95% confidence interval — bảng thật cell 26

| Mã | F1 Test | CI thấp | CI cao |
|---|---:|---:|---:|
| B1 | 0,5947 | 0,5629 | 0,6227 |
| B2 | 0,6953 | 0,6724 | 0,7193 |
| B3 | 0,6380 | 0,6089 | 0,6655 |
| B4 | 0,7334 | 0,7106 | 0,7562 |
| S1 | 0,7451 | 0,7243 | 0,7647 |
| S2 | 0,7542 | 0,7325 | 0,7758 |
| **S3** | **0,7552** | **0,7337** | **0,7776** |

### Cách nhận xét cho đúng

- S3 F1 Test = 0,7552, khoảng bootstrap 95% tham khảo `[0,7337; 0,7776]`.
- S2 `[0,7325;0,7758]` gần như chồng lên khoảng S3. Chênh lệch F1 Test chỉ **0,0010**, không đủ căn cứ để tuyên bố S3 chắc chắn vượt trội S2 trong mọi điều kiện.
- Chồng lấp hai CI riêng **không phải** phép kiểm định trực tiếp hiệu số F1 giữa hai mô hình; muốn so sánh có ý nghĩa phải dùng phân tích ghép cặp về hiệu số (vì hai mô hình chấm trên cùng Test), cùng quy trình đánh giá đủ chặt.
- CI chỉ tái lấy mẫu **dự đoán Test đã có**, không đào tạo lại model cho 500 lượt. Không khẳng định kết quả sẽ giữ nguyên khi dữ liệu Facebook thay đổi chủ đề hoặc cách viết.

## 16. Phân tích đặc trưng và các lỗi thực tế — cell 30

### 16.1. Đặc trưng có hệ số dương/âm nổi bật

Notebook lấy `names = vec.get_feature_names_out()` và, do mô hình cuối là SVM, `importance = clf.coef_.ravel()`. `np.argsort(importance)` sắp xếp hệ số rồi hiển thị 15 đặc trưng mạnh nhất theo hai phía.

Một số đặc trưng có hệ số dương (nghiêng về Spam): `urltoken`, `phonetoken`, `ib chị`, `user_name`, `aptis`, `mở đơn`.

Một số đặc trưng âm (nghiêng về Ham): `thôi`, `cảm_ơn`, `quá`, `kìa`, `cũng`, `phải`.

**Giải thích cẩn thận:** Đó là **hệ số của mô hình trên dữ liệu đã học**, không chứng minh từ đó luôn là Spam/Ham hay có quan hệ nhân quả. Ví dụ `urltoken` có thể xuất hiện trong lời mời quảng cáo, nhưng cũng có trong đường dẫn nhóm học tập chính thức. `user_name` có thể do hàm che tên; dự đoán sai đối với câu chứa tên là một giới hạn của tập đặc trưng.

### 16.2. Lỗi FP — Ham bị nhận nhầm Spam

Từ bảng cell 30 (diễn đạt lại, không cần học tên riêng):

- **Bình luận dẫn link nhóm chuyên ngành** được gán Ham nhưng mô hình báo Spam. Nội dung có nhiều `urltoken` và ký hiệu gây giống quảng cáo.
- **Bình luận chỉ chứa tên hoặc tag người dùng** có thể bị biến thành `user_name`, làm mô hình thiên về Spam.
- **Lời nhắn 'ib page' hoặc 'nhắn ...'** là hội thoại thường nhưng có từ giống lời mời liên hệ quảng cáo.

**Kết luận:** Mô hình dựa trên văn bản/token, thiếu ngữ cảnh bài đăng và mục đích thực tế, nên có thể báo nhầm nội dung chứa link, số điện thoại hoặc lời mời nhắn tin.

### 16.3. Lỗi FN — Spam bị bỏ sót

Từ bảng cell 30:

- Một bình luận dài giới thiệu “trung tâm uy tín, cần thì nhắn để giới thiệu” được gán Spam nhưng có score Ham: cách quảng cáo **trộn vào câu chuyện tự nhiên**, khiến mẫu khó nhận diện.
- Nhiều câu rất ngắn như “có ạ”, “em xin”, “36k thì lấy” được gán Spam trong dữ liệu nhưng văn bản thiếu tín hiệu rõ ràng nếu đứng riêng.

**Kết luận:** Những bình luận Spam theo ngữ cảnh có thể trông như câu giao tiếp bình thường; với mô hình chỉ dựa trên văn bản, khó suy ra đầy đủ ý định người viết. Cũng cần hiểu hạn chế về nhãn: có trường hợp không thể suy diễn tính Spam từ một câu đơn lẻ mà thiếu bài đăng cha.

### 16.4. Nhận xét riêng với ngưỡng S3

Ngay cả khi hạ ngưỡng SVM, các câu có score rất âm vẫn bị bỏ sót; ngưỡng không thay đổi đặc trưng hoặc kiến thức của SVM. Muốn cải thiện lỗi kiểu này có thể cần thêm dữ liệu có ngữ cảnh hoặc cách biểu diễn tốt hơn, nhưng đó là gợi ý nghiên cứu tiếp, **chưa phải thành quả đã thực nghiệm trong Notebook này**.

## 17. Đối chiếu với viBERT ở cell 28

Notebook 02 hiển thị bảng được đọc từ `results/modern/comparison_with_svm.csv` khi `verification.json` báo hoàn tất:

| Model | Accuracy | Precision | Recall | F1 | AP |
|---|---:|---:|---:|---:|---:|
| **S3 — TF-IDF + SVM** | 0,9433 | 0,8374 | 0,6877 | 0,7552 | 0,8117 |
| **M1 — viBERT (unweighted)** | **0,9635** | **0,8942** | **0,8086** | **0,8493** | **0,9108** |

**Nhận xét đúng phạm vi:** Bảng Notebook báo cáo viBERT vượt S3 trên năm metric này, đặc biệt F1 hơn 0,0941 (9,41 điểm phần trăm) và Recall hơn 0,1209 (12,09 điểm phần trăm). Đây là đối chiếu chung của dự án, không thuộc phần bạn phụ trách huấn luyện viBERT.

**Giới hạn xác minh:** Cell 28 đọc bảng từ file của nhánh viBERT, kiểm `status='completed'` và số dòng Test. Chỉ riêng đoạn code cell 28 không cho phép tự kiểm chứng từng ID Test hoặc quy trình fine-tune của Notebook 03. Khi bảo vệ có thể nói theo mô tả của nhóm là “đối chiếu kết quả trên tập Test chung”, nhưng nếu giảng viên hỏi cách bảo đảm cùng ID thì cần trình `verification.json`/Notebook 03 hoặc danh sách ID đối chiếu — không suy luận tính đồng nhất chỉ từ số lượng 7.474.

## 18. Nhận xét tổng hợp theo đúng trật tự báo cáo

### 18.1. Phát hiện từ thí nghiệm

1. **Không thể dùng Accuracy một mình.** Luôn Ham đạt 87,28% Accuracy nhưng F1 Spam = 0.
2. **Trong bốn tổ hợp cơ sở, B4 là cấu hình F1 tốt nhất**, nhờ TF-IDF+SVM; nhưng Recall chưa cao (Test 62,78%).
3. **S1 thiên về Recall:** bắt được nhiều Spam nhất, đồng thời tạo nhiều cảnh báo nhầm nhất.
4. **S2** có F1 Test gần S3, nhưng CV F1 thấp hơn và phải loại nhiều mẫu Ham khỏi Train dùng học.
5. **S3 đạt CV F1 cao nhất (0,7876)** nên được chọn trước Test. Test F1 0,7552; Precision 83,74%, Recall 68,77%, AP 0,8117.
6. S3 **chỉ nhỉnh hơn S2 0,0010 F1 Test**; không thổi phồng khoảng cách.
7. **F1 S3 trên Test thấp hơn CV 0,0324**, có thể do khác phân bố/độ khó, tác động chọn tham số/ngưỡng trên CV; không thể quy nguyên nhân tuyệt đối chỉ từ bảng.
8. ViBERT trong bảng dự án có Test F1 0,8493 cao hơn mô hình truyền thống, nhưng không thuộc phần huấn luyện bạn đảm nhiệm.

### 18.2. Vì sao S3 được chọn nhưng không khẳng định tốt nhất vô điều kiện?

- Tiêu chí **đã đặt trước**: CV F1 của Spam.
- Phạm vi **đã khảo sát**: B1–B4 và ba chiến lược trên TF-IDF + SVM.
- Kết quả **thực tế**: S3 CV F1 = 0,7876 là cao nhất.
- Test **xác nhận mức hiệu năng ngoài Train**: S3 F1 = 0,7552, vẫn có 127 FP và 297 FN.
- Lựa chọn nghiệp vụ khác có thể khác: S1 nếu muốn bắt nhiều Spam hơn bằng mọi giá; B4 nếu đặc biệt ngại báo nhầm.

### 18.3. Không nên nói những điều sau

- “S3 là tốt nhất vì có Test F1 cao nhất.” → Không đúng quy trình; **chọn bằng CV**, Test để báo cáo.
- “S3 tốt hơn S2 một cách rõ rệt.” → Chênh chỉ 0,0010 trên Test; cần thận trọng.
- “S3 thay TF-IDF/SVM để học lại ngưỡng mỗi lần dự đoán.” → S3 **chọn ngưỡng trước**, lưu cùng model; không chọn lại khi dự đoán.
- “B4 và S3 AP khác nhau vì ngưỡng khác.” → Sai; hai mô hình có cùng score và AP.
- “S1 làm dataset ban đầu có thêm Spam.” → Sai; S1 đổi trọng số lỗi, không thêm/xóa mẫu.
- “S2 giảm số mẫu trong Test.” → Sai; chỉ lấy mẫu Train từng fold/toàn Train.
- “95% CI nghĩa là chắc chắn 95% các lần triển khai sau nằm trong khoảng.” → Diễn giải quá mức; CI bootstrap ở đây chỉ phản ánh sự biến thiên theo rút mẫu Test với dự đoán cố định, dưới giả định mẫu thích hợp.

## 19. Thực hành Python — kiểm chứng ngay trên Notebook

### 19.1. Tự tính các metric từ confusion matrix

```python
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Cần chạy sau cell 20 trong Notebook 02
pred = best['_pred']
y = y_test

tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()

accuracy_manual = (tp + tn) / (tp + tn + fp + fn)
precision_manual = tp / (tp + fp) if tp + fp else 0.0
recall_manual = tp / (tp + fn) if tp + fn else 0.0
f1_manual = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0

print(tn, fp, fn, tp)
print('Tự tính:', accuracy_manual, precision_manual, recall_manual, f1_manual)
print('scikit-learn:', accuracy_score(y,pred), precision_score(y,pred), recall_score(y,pred), f1_score(y,pred))

assert np.isclose(f1_manual, best['F1'])
```

**Kỳ vọng theo output Notebook đã lưu:** `6396 127 297 654` và F1 gần `0.7552`.

### 19.2. Xem CV vs Test và khoảng chênh

```python
from pandas import DataFrame
rows = []
for r in CANDIDATES:
    rows.append({
        'Mã': r['Mã'],
        'CV F1': r['CV F1 (spam)'],
        'Test F1': r['F1'],
        'Test - CV': r['F1'] - r['CV F1 (spam)']
    })
print(DataFrame(rows).round(4).to_string(index=False))
```

### 19.3. Kiểm tra vì sao B4 và S3 cùng AP

```python
from sklearn.metrics import average_precision_score
assert np.allclose(focus['_score'], tuned['_score'])
print('AP B4:', average_precision_score(y_test, focus['_score']))
print('AP S3:', average_precision_score(y_test, tuned['_score']))
print('F1 B4:', focus['F1'], 'F1 S3:', tuned['F1'])
```

### 19.4. Kiểm tra `evaluate()` sửa record tại chỗ

```python
# KHÔNG dùng trực tiếp best nếu chỉ đang thử ví dụ, hãy làm copy dictionary
record_demo = dict(best)
X_test_demo = TEXT[best['Tiền xử lý']].iloc[test_idx].to_numpy()
ret = evaluate(record_demo, X_test_demo, y_test)
print('Hàm trả:', ret)  # None
print('Record có F1:', record_demo['F1'])
```

### 19.5. Tự làm một confusion matrix nhỏ không cần Notebook

```python
from sklearn.metrics import confusion_matrix, classification_report

y_true = [0, 0, 0, 1, 1, 1]
y_pred = [0, 0, 1, 1, 1, 0]
print(confusion_matrix(y_true, y_pred, labels=[0, 1]))
# [[2, 1],
#  [1, 2]]
print(classification_report(y_true, y_pred, target_names=['ham','spam'], zero_division=0))
```

**Cần nhớ:** `classification_report` là ví dụ học bổ sung; hàm này không được gọi trong file `evaluation.py` hiện tại.

### 19.6. So sánh hai mô hình bằng paired bootstrap (gợi ý mở rộng, chưa có trong dự án)

Nếu thật sự muốn kiểm tra độ chắc chắn của **hiệu số F1 S3−S2** chứ không chỉ đọc hai CI riêng, có thể rút cùng vị trí Test cho cả hai mô hình rồi tính phân bố hiệu số. Đây là **phân tích mở rộng** ngoài Notebook hiện tại, không nên trình bày thành kết quả đã có.

```python
# MỞ RỘNG, CHƯA CHẠY TRONG NOTEBOOK GỐC
from sklearn.metrics import f1_score
rng = np.random.default_rng(0)
paired_diff = []
for _ in range(1000):
    idx = rng.integers(0, len(y_test), len(y_test))
    paired_diff.append(
        f1_score(y_test[idx], tuned['_pred'][idx], zero_division=0)
        - f1_score(y_test[idx], under['_pred'][idx], zero_division=0)
    )
print(np.percentile(paired_diff,[2.5,50,97.5]))
```

## 20. Câu hỏi ôn tập khi bảo vệ

**1. File `evaluation.py` có fit lại mô hình không?** Không. Nó lấy Pipeline đã fit và đo Test.

**2. `predict_record()` nhận `X` dạng gì?** Chuỗi token; Pipeline tự `transform` thành vector.

**3. Vì sao phải lưu cả `_pred` và `_score`?** `_pred` để tính F1/confusion và lọc lỗi; `_score` để tính AP, đường PR và xem độ nghiêng về Spam.

**4. Tại sao NB dùng `predict_proba`, SVM dùng `decision_function`?** NB cung cấp xác suất theo mô hình, LinearSVC không có `predict_proba`, chỉ có decision score.

**5. Score SVM = 0,8 có nghĩa 80% Spam?** Không. Nó là điểm tuyến tính có dấu, không phải xác suất.

**6. Ngưỡng của S3 có phải chọn bằng Test?** Không, chọn từ OOF trên Train trong cell 16; Test chỉ dùng ở cell 20.

**7. Vì sao không chọn theo Test F1?** Để giữ Test làm phép đánh giá độc lập và tránh chọn mô hình theo chính bài kiểm tra cuối.

**8. F1 Spam tính từ những ô nào?** TP, FP, FN: `2TP/(2TP+FP+FN)`.

**9. Accuracy cao có chứng minh mô hình bắt Spam tốt?** Không. Luôn dự đoán Ham đã đạt 87,28% Accuracy nhưng F1 Spam=0.

**10. Precision 83,74% nghĩa là gì?** Trong nhóm bị gắn cờ Spam trên Test, 83,74% thực sự là Spam.

**11. Recall 68,77% nghĩa là gì?** Trong số Spam thật của Test, nhận diện đúng 68,77%; còn 31,23% bị bỏ sót.

**12. FP 127 có tác động gì?** 127 bình luận Ham có thể bị hệ thống xử lý nhầm là Spam.

**13. FN 297 có tác động gì?** 297 Spam không bị phát hiện trên 951 Spam thật.

**14. B4 có ít FP nhất sao không chọn?** Vì nhóm ưu tiên CV F1 để cân bằng Precision/Recall, không chỉ cực tiểu FP.

**15. S1 có Recall cao nhất sao không chọn?** Vì Precision thấp và CV F1 không cao nhất; tuy nhiên nghiệp vụ ưu tiên Recall có thể chọn khác.

**16. S3 khác B4 ở điểm nào?** Giữ TF-IDF+SVM, C và n-gram, chỉ đổi ngưỡng từ 0 thành -0,1776.

**17. Sao S3 AP không tăng dù F1 tăng?** AP dùng toàn bộ score/xếp hạng; score của B4 và S3 giống nhau, ngưỡng quyết định khác.

**18. Sao Test F1 thấp hơn CV F1?** Khác tập và phân bố, độ khó của Test, và CV có thể lạc quan do quá trình chọn cấu hình/ngưỡng. Không kết luận một nguyên nhân duy nhất.

**19. `zero_division=0` dùng để làm gì?** Tránh trường hợp công thức có mẫu số 0 và quy định trả 0 cho metric tương ứng.

**20. `confusion_matrix(...).ravel()` theo thứ tự nào?** Khi `labels=[0,1]`, là TN, FP, FN, TP.

**21. Tại sao `evaluate()` không return?** Nó cập nhật trực tiếp dictionary record, rồi Python mặc định trả None.

**22. Bootstrap có huấn luyện lại 500 mô hình không?** Không. Rút lại các cặp nhãn thật/dự đoán Test và tính lại F1.

**23. 95% CI F1 của S3 nghĩa là thắng S2 có ý nghĩa thống kê?** Không, đó là hai CI riêng và chúng rất gần/chéo nhau; chưa phải kiểm định hiệu số ghép cặp.

**24. Tại sao cần biểu đồ PR khi đã có F1?** F1 chỉ là một điểm theo ngưỡng đã chọn; PR giúp xem quan hệ đánh đổi trên nhiều ngưỡng, AP tổng hợp chất lượng xếp hạng.

**25. Đường PR S3 nằm ở đâu?** Trùng với B4; Notebook đánh dấu thêm một điểm S3 tại ngưỡng -0,1776.

**26. Bảng `coef_` cho biết điều gì?** N-gram có hệ số thiên về Ham/Spam trong SVM, không chứng minh nhân quả hoặc quy tắc tuyệt đối.

**27. Vì sao link nhóm học có thể bị nhận nhầm Spam?** Token URL và các từ tương tự quảng cáo tạo tín hiệu Spam; mô hình chỉ nhìn văn bản, thiếu ngữ cảnh.

**28. Tại sao không khẳng định S3 tốt nhất toàn dự án?** Bảng viBERT báo Test F1 cao hơn, và tiêu chí chọn mô hình có thể khác; S3 chỉ dẫn đầu nhóm truyền thống theo CV F1.

**29. viBERT có được huấn luyện trong file này không?** Không; cell 28 đọc bảng kết quả nhánh khác. Việc xác minh toàn bộ quy trình viBERT thuộc Notebook 03.

**30. Có nên dùng bảng Test để sửa threshold sau khi xem lỗi?** Nếu làm vậy phải thiết kế tập đánh giá mới độc lập; không còn được báo cáo cùng Test như đánh giá cuối không bị ảnh hưởng lựa chọn.

## 21. Tóm tắt trong 90 giây để trình bày

> “Sau khi hoàn thành huấn luyện và chọn mô hình bằng 5-fold Cross-Validation, nhóm sử dụng `evaluation.py` để đánh giá các mô hình trên 7.474 bình luận Test giữ riêng, gồm 951 Spam. Hàm `predict_record` gọi Pipeline đã fit để lấy score và áp ngưỡng, trả nhãn Ham/Spam; hàm `evaluate` tính Accuracy, Precision, Recall, F1, AP, rồi lưu dự đoán/score phục vụ biểu đồ và phân tích lỗi. Kết quả cho thấy B4 là mô hình cơ sở dẫn đầu F1, nhưng Recall còn thấp. Nhóm thử S1, S2, S3 và chọn S3 bằng CV F1 Spam=0,7876 trước khi xem Test. S3 đạt Test F1=0,7552, Precision=83,74%, Recall=68,77%, có 127 Ham báo nhầm và 297 Spam bỏ sót. So với B4, S3 bắt thêm 57 Spam nhưng báo nhầm thêm 47 Ham. Đường PR của B4 và S3 giống nhau vì chỉ ngưỡng thay đổi, không đổi score. Kết quả Test thấp hơn CV một chút và gần S2, nên không khẳng định S3 vượt trội trong mọi tình huống. Báo cáo giới hạn dữ liệu và so sánh chung với viBERT, còn việc huấn luyện viBERT thuộc thành viên khác.”

---

## Phụ lục A. Bảng số dùng trực tiếp khi giải thích biểu đồ

| Mã | CV F1 | Test F1 | Test−CV | TP | FP | FN | TN | AP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B1 | 0,6077 | 0,5947 | -0,0130 | 471 | 162 | 480 | 6361 | 0,7320 |
| B2 | 0,7087 | 0,6953 | -0,0134 | 647 | 263 | 304 | 6260 | 0,7493 |
| B3 | 0,5959 | 0,6380 | +0,0421 | 489 | 93 | 462 | 6430 | 0,7988 |
| B4 | 0,7736 | 0,7334 | -0,0402 | 597 | 80 | 354 | 6443 | 0,8117 |
| S1 | 0,7580 | 0,7451 | -0,0129 | 741 | 297 | 210 | 6226 | 0,8127 |
| S2 | 0,7629 | 0,7542 | -0,0087 | 695 | 197 | 256 | 6326 | 0,7826 |
| S3 | 0,7876 | 0,7552 | -0,0324 | 654 | 127 | 297 | 6396 | 0,8117 |

## Phụ lục B. Nguồn và danh sách biểu đồ

- **Code:** `pipeline/modeling/evaluation.py`; `pipeline/modeling/experiments.py`; `pipeline/modeling/imbalance.py`.
- **Notebook:** `02_thuc_nghiem_va_demo(1).ipynb`, đặc biệt cell 16 (chọn CV), 20 (Test), 22 (so sánh chiến lược), 24 (ma trận), 26 (PR và bootstrap), 28 (viBERT), 30 (lỗi), 34 (nhận xét).
- **Hình 1:** `06_evaluation_assets/01_imbalance_metrics_and_errors.png` — lấy nguyên từ output cell 22.
- **Hình 2:** `06_evaluation_assets/02_confusion_baselines.png` — lấy nguyên từ output cell 24.
- **Hình 3:** `06_evaluation_assets/03_confusion_strategies.png` — lấy nguyên từ output cell 24.
- **Hình 4:** `06_evaluation_assets/04_pr_baselines.png` — lấy nguyên từ output cell 26.
- **Hình 5:** `06_evaluation_assets/05_pr_strategies.png` — lấy nguyên từ output cell 26.

> **Lưu ý khi tải:** Markdown tham chiếu năm hình theo đường dẫn tương đối. Để hình hiển thị trong VS Code, tải **gói ZIP đầy đủ** (gồm file `.md` và thư mục `06_evaluation_assets`), giải nén và mở Markdown tại đó. File `.md` tải riêng vẫn đọc được toàn bộ văn bản, bảng, công thức, code và nhận xét, nhưng không tự mang theo ảnh.
