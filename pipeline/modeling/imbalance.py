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
