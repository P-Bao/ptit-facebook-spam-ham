# Phi 03 — Hiểu ba cách xử lý mất cân bằng S1, S2, S3

Đọc sau [luồng fit](PHI_02_LUONG_FIT_TO_HOP_VA_DIEM_CV.md). Phạm vi hiện tại: khảo sát ba chiến lược trên ứng viên TF-IDF+SVM B4 sau khi chốt nhánh từ dừng bằng CV. Các cách là ba thí nghiệm riêng, không nối S1 → S2 → S3 thành một model dùng cả ba.

## 1. Vì sao cần khảo sát?

Ham nhiều hơn spam nên model có thể đạt Accuracy cao nhưng bỏ sót spam. Mục tiêu là cân nhắc Precision/Recall/F1 spam. “Xử lý mất cân bằng” không bắt buộc đưa số mẫu về 1:1 và không bảo đảm mọi cách đều tăng F1. S1 thay hàm mất mát, S2 thay mẫu dùng để học, S3 thay quyết định từ score. S3 thường được gọi là điều chỉnh quyết định trước lệch lớp; nó không cân bằng lại số lượng dữ liệu.

| Mã | Mẫu train | Trọng số lỗi | Ngưỡng SVM | Học lại gì? |
|---|---|---|---|---|
| B4 | Giữ nguyên | None | 0 | Chọn n-gram/C bằng CV |
| S1 | Giữ nguyên | balanced | 0 | Tìm lại n-gram/C với loss có trọng số |
| S2 | Giữ spam, giảm ham 2:1 tối đa | None | 0 | Tìm lại n-gram/C trên train fold đã lấy mẫu |
| S3 | Giữ nguyên | None | Chọn trên score OOF | Giữ tham số B4, chọn ngưỡng rồi refit |

Validation và test luôn giữ phân bố ban đầu, không giảm ham để làm đẹp điểm.

## 2. S1 — class_weight='balanced'

Mỗi lỗi của lớp ít mẫu chịu mức phạt lớn hơn. Trọng số lớp c là `n/(k*n_c)`; bài này k=2. Ví dụ train có 900 ham và 100 spam: trọng số ham=1000/(2×900)≈0,556, spam=1000/(2×100)=5. Một mẫu spam được nhấn mạnh gấp khoảng chín lần một mẫu ham trong thành phần loss. Không tạo thêm spam, không xóa ham và không biến score thành xác suất.

Các bước: nhận X/y train và folds → tạo TF-IDF+LinearSVC có balanced → với mỗi n-gram/C và fold, tính trọng số từ **y của train fold** khi classifier.fit chạy → học → chấm validation nguyên trạng → chọn mean F1 cao nhất → refit trên toàn train. Trọng số được tính lại cho lần refit, không dùng tỷ lệ test.

Hàm `fit_balanced_svm` chỉ là hàm gọi cấu hình; phần tối ưu vẫn do LinearSVC, còn thử lưới do fit_experiment. Các tham số `pre/X/y/folds` giữ nguyên nghĩa như tài liệu 02. `strategy` chỉ là tên ghi kết quả.

## 3. S2 — giảm mẫu ham trong train

Under-sampling bỏ bớt mẫu lớp nhiều để giảm áp lực từ lớp đó. Project giữ toàn bộ spam và chọn ngẫu nhiên **không hoàn lại** tối đa hai ham cho mỗi spam. Nếu train vốn có ít hơn hai ham/spam thì giữ tất cả ham, không tạo mẫu mới. Ví dụ 900 ham/100 spam → giữ 200 ham/100 spam. Có thể mất thông tin ham hữu ích, nên phải chấm trên validation/test nguyên trạng.

Các bước cho một candidate: tách fold → chỉ lấy chỉ số train fold → chia chỉ số thành ham/spam → chọn ham theo seed → gộp spam+ham được chọn → xáo thứ tự → clone Pipeline và set_params → học từ vựng/IDF/classifier trên mẫu đã chọn → dự đoán validation. Làm đủ năm fold, tính mean F1. Lặp tám candidate rồi lấy ứng viên tốt nhất. Cuối cùng lấy mẫu từ toàn train và fit một model S2 để chấm test.

## 4. S3 — điều chỉnh ngưỡng decision score

SVM thường quyết định spam khi score>0. Với ngưỡng t, project quyết định spam khi **score>t**. Hạ t xuống âm làm một số câu trước đây bị xem ham trở thành spam: Recall thường tăng, Precision có thể giảm. Ví dụ t=−0,18: score −0,10 chuyển thành spam, score −0,30 vẫn ham. Ngưỡng âm hợp lệ vì score SVM là giá trị trên trục thực, không phải xác suất.

Không chọn t từ score model dự đoán lại chính dữ liệu nó đã học. Project tạo **OOF**: chia train thành năm fold; mỗi lần clone B4, fit trên bốn fold và lấy score của fold bị giữ lại; ghi score vào đúng vị trí ban đầu. Mỗi dòng train có đúng một score từ model chưa học dòng đó (và theo cách chia nhóm, không học nhóm trùng của nó).

Từ score OOF: tạo 199 mức phân vị từ 1% tới 99%, thêm 0 và hai mốc biên, loại trùng; với mỗi t tạo nhãn score>t, tính P/R/F1 từng validation và lấy trung bình; chọn mean F1 cao nhất, hòa thì lấy t gần 0 hơn. Cuối cùng clone B4 và fit toàn train; lưu t riêng trong record/bundle. B4 và S3 cùng siêu tham số, cùng dữ liệu fit cuối; khác quy tắc chuyển score thành nhãn.

**Giới hạn cần nói đúng:** đây không phải nested CV. Tham số B4 và t đều được lựa chọn với các folds train, và điểm S3 báo trên chính OOF dùng để tìm t nên có thể lạc quan do lựa chọn. Không dùng điểm ấy làm bằng chứng đánh giá độc lập. Test vẫn giữ riêng để đo sau khi chốt. B4 và S3 có cùng thứ hạng score nên AP/đường PR có thể giống nhau dù F1 và confusion matrix khác.

## 5. Toàn bộ code imbalance.py hiện tại

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

## 6. Giải thích các dòng xử lý theo từng hàm

### 6.1. Import và fit_balanced_svm

- `numpy as np`: tạo mảng, chọn phần tử theo chỉ số, sinh ngẫu nhiên, phân vị, lấy trung bình.
- `clone`: tạo estimator mới cùng siêu tham số, không giữ từ vựng/trọng số đã fit.
- `f1_score`, `precision_score`, `recall_score`: so nhãn thật 0/1 với dự đoán, mặc định spam=1.
- `ParameterGrid`: chỉ liệt kê cấu hình; không fit, không chia fold, không chấm.
- `from ... import fit_experiment` trong thân hàm tránh vòng import khi các module đang nạp.
- `return fit_experiment(pre, 'TF-IDF', 'SVM',...)`: cố định cặp TF-IDF/SVM; chuyển X/y/folds cùng cache/jobs/seed; `class_weight='balanced'` đổi loss, `strategy` đặt tên record. Hàm trả nguyên record do fit_experiment tạo.

### 6.2. sample_training_indices(y, indices, seed=42, ratio=2)

Đầu vào y là nhãn của toàn train; indices là các vị trí **được phép lấy** trong y. Kết quả là mảng vị trí được giữ, không phải vector dữ liệu mới.

| Dòng | Xử lý và lý do |
|---|---|
| `indices = np.asarray(indices)` | Đổi list/array thành mảng NumPy để chọn nhiều vị trí cùng lúc |
| `np.asarray(y)[indices] == 1` | Lấy nhãn trong vùng cho phép và tạo mặt nạ spam |
| `spam = indices[...]` | Giữ vị trí gốc của spam trong mảng train |
| `ham = indices[... == 0]` | Làm tương tự cho ham |
| `rng = np.random.default_rng(seed)` | Bộ sinh ngẫu nhiên riêng, tái lập với cùng input/seed |
| `min(ratio * len(spam), len(ham))` | Giới hạn ham được chọn; không chọn quá số ham có sẵn |
| `rng.choice(..., replace=False)` | Chọn ham không hoàn lại, không lặp cùng vị trí |
| `np.concatenate([spam, sampled])` | Ghép toàn bộ spam với ham đã chọn |
| `rng.shuffle(selected)` | Đổi thứ tự tại chỗ; không đổi thành phần mẫu |
| `return selected` | Trả chỉ số để bên gọi lấy cả X và y tương ứng |

### 6.3. mean_cv_scores(y, predictions, folds)

`result={}` tạo dict kết quả. Vòng ngoài ghép tên Precision/Recall/F1 với hàm metric. Mỗi chỉ số có `fold_scores=[]` riêng. `for _,valid in folds` bỏ chỉ số train, chỉ lấy vùng validation. `metric(y[valid], predictions[valid], zero_division=0)` tính điểm đúng fold. `append` lưu từng điểm; `float(np.mean(...))` lấy trung bình và đổi về số Python. f-string tạo tên cột `CV F1 (spam)`… `return result` trả ba số, không trả model và không học lại.

### 6.4. undersampled_search(pipe, grid, X, y, folds, seed=42)

`rows=[]` giữ kết quả các candidate. `for params in ParameterGrid(grid)` duyệt tám dict cụ thể: hai n-gram × bốn C; `class_weight=None` chỉ có một lựa chọn.

`predictions=np.zeros(len(y),dtype=int)` cấp chỗ lưu nhãn OOF cho candidate hiện tại. Vòng `for train,valid in folds` lấy chỉ số từng lượt. `sample_training_indices(y,train,seed)` giảm ham chỉ trong train fold. `assert not set(selected) & set(valid)` kiểm giao hai tập vị trí rỗng; nếu không thì dừng vì có rò rỉ.

`clone(pipe).set_params(**params).fit(X[selected],y[selected])` gồm ba thao tác nối tiếp: tạo pipeline chưa học → trải dict đặt n-gram/C → fit trên đúng các mẫu được giữ. `predictions[valid]=model.predict(X[valid])` đặt nhãn của validation về đúng vị trí, không ghi vào vị trí train fold. `rows.append((mean_cv_scores(...),params))` ghi một bộ điểm trung bình+tham số sau khi đủ năm fold.

`max(rows,key=lambda row:row[0]['CV F1 (spam)'])` chọn tuple có F1 lớn nhất; `row[0]` là dict chỉ số, `row[1]` là params. Nếu hòa tuyệt đối, Python giữ phần tử gặp đầu; không có tie-break khác ở S2. `np.arange(len(y))` tạo mọi vị trí của toàn train; lấy mẫu lại trên vùng này để fit cuối. `return fitted,scores,params,len(selected)` trả bốn giá trị để notebook tạo record S2. Ngưỡng S2 vẫn là 0.

### 6.5. tune_svm_threshold(pipe, X, y, folds)

`pipe` nhận từ focus B4 đã fit, nhưng mỗi `clone(pipe)` chỉ giữ siêu tham số, xóa trạng thái đã học. Vì vậy score OOF không được tạo bởi model fit toàn train.

| Dòng | Ý nghĩa |
|---|---|
| `scores=np.zeros(len(y))` | Chỗ chứa một score thực cho mỗi dòng train |
| `covered=np.zeros(len(y),dtype=int)` | Đếm mỗi dòng được validation mấy lần |
| `clone(pipe).fit(X[train],y[train])` | Học B4 mới trên train fold |
| `scores[valid]=model.decision_function(X[valid])` | Điền score ngoài fold |
| `covered[valid] += 1` | Tăng đếm vùng vừa dự đoán |
| `assert (covered == 1).all()` | Mọi dòng có đúng một score OOF |
| `thresholds=[0.0]` | Đưa mốc mặc định vào danh sách |
| `np.linspace(0.01,0.99,199)` | Tạo 199 mức xác suất phân vị cách đều, không phải 199 score cách đều |
| `np.quantile(scores,...)` | Đổi các mức phân vị thành ngưỡng score thực |
| `np.nextafter(scores.min(),-np.inf)` | Số thực ngay dưới score nhỏ nhất; với dấu >, mọi dòng thành spam |
| `scores.max()` | Với dấu >, không dòng nào thành spam |
| `np.unique(thresholds)` | Loại ngưỡng trùng và sắp thứ tự |
| `(scores > threshold).astype(int)` | Tạo nhãn 0/1 cho một mốc |
| `mean_cv_scores(...)` | Chấm năm validation với nhãn tại mốc đó |
| `rows.append((metrics,float(threshold)))` | Giữ cả chỉ số và mốc tương ứng |
| `max(... key=(F1,-abs(t)))` | Ưu tiên F1 cao; hòa thì mốc gần 0 |
| `return clone(pipe).fit(X,y),metrics,threshold` | Refit toàn train; trả model mới, điểm OOF lựa chọn và ngưỡng |

Hai mốc biên được thêm để xét đầy đủ cả trường hợp tất cả ham/tất cả spam, không phải lỗi của bảng đánh giá. Project dùng **dấu >** cho SVM, bao gồm S3; NB dùng >= với xác suất. Khi tự viết lại phải giữ quy ước này để các câu bằng đúng ngưỡng được xử lý thống nhất.

## 7. Notebook gọi và lắp record như thế nào?

Cell thứ 17 tính cả Markdown của Notebook 02:

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

`X_tr_text` là chuỗi token nhánh MAIN_PRE trên train. S1 trả record sẵn nên chỉ gắn mã. S2 trả bốn giá trị; notebook ghép `**us_metrics` vào dict, đặt `_model`, `_threshold=0`. `us_count` là số mẫu fit cuối, được đối chiếu số mẫu đã thống kê. S3 nhận `focus['_model']` giữ cấu hình B4; `**focus['Tham số']` cộng thêm threshold vào dict ghi nhận. Không thay C/n-gram của B4 trong tune_svm_threshold.

`SUPPLEMENTARY` là ba record S1–S3. `STRATEGIES` là B4+S1+S2+S3 để so tác động trong cùng họ TF-IDF/SVM. `CANDIDATES` gồm B1–B4+S1–S3 để chọn mô hình cuối trong phạm vi khảo sát. `best=max(...CV F1...)` xảy ra trước test; `SELECTED_NAME` chỉ tạo chữ hiển thị. DataFrame strategy_cv tổng hợp điểm CV/số mẫu/ngưỡng, `to_csv` ghi bảng, không huấn luyện.

## 8. Chấm test, lưu và dùng ngưỡng

`evaluation.predict_record` nhận record đã chốt và chuỗi token test. Nếu classifier có predict_proba thì lấy cột spam và so >= threshold; nếu là LinearSVC thì lấy decision_function và so > threshold. `evaluation.evaluate` so nhãn dự đoán với y_test để tính P/R/F1, AP, Accuracy. Không thay ngưỡng bằng kết quả test.

Notebook lưu bundle có model, threshold, remove_stop và thông tin chạy bằng joblib.dump. `pipeline/modeling/predict.py` nạp theo bundle được truyền vào, tiền xử lý câu mới nhất quán rồi áp ngưỡng. File predict.py không tự chọn S3 và không tự train lại.

Theo kết quả local đã có: S1 tăng Recall nhưng giảm Precision; S2 có F1 test≈0,7542; S3 CV F1≈0,7876 và test F1≈0,7552 với ngưỡng≈−0,1776. Chênh lệch S2/S3 trên test nhỏ; không khẳng định S3 luôn tốt hơn ở mọi nguồn dữ liệu. S3 được chọn theo CV trong phạm vi đã thử, không vì nhìn test rồi lựa.

## 9. Các câu dễ bị hỏi

**Có dùng cả ba cách vào model cuối không?** Không. Chúng là ba ứng viên riêng; model cuối hiện là S3, giữ mẫu và trọng số mặc định, đổi ngưỡng.

**S3 có học lại không?** Có năm fit để tạo OOF và một fit toàn train; chọn ngưỡng không làm thay đổi w/b của model cuối so với cách fit B4 cùng tham số/dữ liệu, nhưng thay nhãn quyết định.

**S1 weighted khác S3 threshold thế nào?** S1 đổi mục tiêu học nên có thể đổi trọng số/ranking score; S3 đổi mốc quyết định trên score của cùng cấu hình.

**ParameterGrid khác GridSearchCV thế nào?** ParameterGrid chỉ cho các dict tham số. S2 tự viết vòng fit/chấm do cần lấy mẫu riêng từng train fold. GridSearchCV đã làm sẵn điều phối đó cho B/S1, nhưng pipeline hiện tại không có bước sampler.

**Vì sao không giảm mẫu trước khi chia validation?** Làm vậy thay phân bố validation và có thể làm sai thiết kế đo. Code chỉ giảm phần học của mỗi fold.

**Ngưỡng −0,1776 có phải xác suất âm?** Không. Nó là decision score SVM; xác suất chỉ nằm [0,1] còn decision score không có ràng buộc ấy.

**Có nên accept nhãn AI 0,75 giống threshold S3?** Hai đại lượng khác nhau: confidence AI cho quyết định tiếp nhận nhãn và decision score cho phân loại. Không so hoặc dùng thay nhau.
