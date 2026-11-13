r"""Chạy phần Phi từ CSV đã có nhãn/token đến mô hình truyền thống.

Lệnh ở gốc project:
    .\.venv\Scripts\python.exe scripts/run_phi_pipeline.py

Đọc file từ trên xuống: nhận CSV → chia tập → CV → lệch lớp → test → lưu.
Không gọi Facebook, Argilla hoặc viBERT. Kết quả mặc định ghi vào thư mục
artifacts/spamham/phi_pipeline/ để không ghi đè file của Notebook 02.
"""

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from joblib import Memory


# Khi chạy file trong scripts/, thêm gốc project để import package pipeline/.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.modeling.predict import predict_spam
from pipeline.modeling.classifiers import MODELS
from pipeline.modeling.evaluation import evaluate, predict_record
from pipeline.modeling.experiments import fit_experiment, make_pipe
from pipeline.modeling.imbalance import (fit_balanced_svm, sample_training_indices,
                                         undersampled_search, tune_svm_threshold)
from pipeline.modeling.text_features import FEATURES
from pipeline.modeling.training_utils import grouped_folds, split_train_test


# argparse là thư viện chuẩn để người dùng đổi đường dẫn/số tiến trình khi cần.
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--csv", type=Path, default=PROJECT_ROOT / "data/processed/v7_preprocessed.csv")
parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "artifacts/spamham/phi_pipeline")
parser.add_argument("--jobs", type=int, default=2, help="Số tiến trình GridSearchCV")
args = parser.parse_args()

# Giống Notebook 02: cố định seed toàn cục trước mọi thao tác có yếu tố ngẫu nhiên.
np.random.seed(42)

csv_path = args.csv.resolve()
output_dir = args.output.resolve()
output_dir.mkdir(parents=True, exist_ok=True)

# GIAI ĐOẠN 0 — Nhận CSV đã có nhãn/token; chưa học đặc trưng/mô hình.
data = pd.read_csv(csv_path, keep_default_na=False, dtype={"id": str})
required = {"id", "label", "text", "cleaned_text", "tokens_keep_stopwords", "tokens_remove_stopwords"}
missing = required - set(data.columns)
if missing:
    raise ValueError(f"CSV thiếu cột: {sorted(missing)}")

data["label"] = data["label"].str.strip().str.lower()
if set(data["label"]) != {"ham", "spam"}:
    raise ValueError("CSV phải có đúng hai nhãn ham và spam")
if not data["id"].is_unique or data["id"].str.strip().eq("").any():
    raise ValueError("ID phải duy nhất và không rỗng")

# Hai chuỗi token đã được chuẩn bị sẵn; Phi chỉ chọn nhánh bằng CV ở giai đoạn 1.
text = {"giữ stop words": data["tokens_keep_stopwords"],
        "có loại stop words": data["tokens_remove_stopwords"]}
y_all = data["label"].eq("spam").astype(int).to_numpy()  # ham=0, spam=1

# Hàm chia giữ các câu trùng khóa cùng phía và trả vị trí trong TOÀN CSV.
keys, train_idx, test_idx = split_train_test(data["cleaned_text"], y_all)
y_train = y_all[train_idx]
y_test = y_all[test_idx]
if len(np.unique(y_train)) != 2 or len(np.unique(y_test)) != 2:
    raise ValueError("Train/test phải có cả ham và spam")
assert not set(keys[train_idx]) & set(keys[test_idx])

# Mỗi cặp (train, validation) bên dưới dùng chỉ số trong MẢNG TRAIN.
folds = grouped_folds(keys[train_idx], y_train)
print(f"Giai đoạn 0: {len(train_idx)} train, {len(test_idx)} test, {len(folds)} fold CV")

# Cache tạm giảm thời gian tính vector lặp lại; không phải file mô hình cuối.
with tempfile.TemporaryDirectory(prefix="phi_cache_", dir=output_dir) as temp_dir:
    cache = Memory(temp_dir, verbose=0)

    # GIAI ĐOẠN 1 — 2 nhánh token × 2 đặc trưng × 2 bộ phân loại = 8 CV.
    preprocess_results = []
    for pre, token_series in text.items():
        X_train = token_series.iloc[train_idx].to_numpy()
        for feature in FEATURES:
            for algorithm in MODELS:
                record = fit_experiment(pre, feature, algorithm, X_train, y_train, folds,
                                        seed=42, cache=cache, n_jobs=args.jobs)
                preprocess_results.append(record)
        print(f"Giai đoạn 1: xong CV cho nhánh {pre}", flush=True)

    # Chọn nhánh có ứng viên F1 spam cao nhất; chỉ nhánh đó được đánh mã B1–B4.
    main_pre = max(preprocess_results, key=lambda r: r["CV F1 (spam)"])["Tiền xử lý"]
    baselines = [r for r in preprocess_results if r["Tiền xử lý"] == main_pre]
    for number, record in enumerate(baselines, 1):
        record["Mã"] = f"B{number}"
    focus = max(baselines, key=lambda r: r["CV F1 (spam)"])
    if (focus["Đặc trưng"], focus["Thuật toán"]) != ("TF-IDF", "SVM"):
        raise RuntimeError("B4 không còn dẫn đầu; cần xem lại phạm vi khảo sát S1–S3")
    print(f"Giai đoạn 1: chọn {main_pre}; đối chứng {focus['Mã']} TF-IDF + SVM")

    # GIAI ĐOẠN 2 — Thử ba chiến lược riêng trên TF-IDF + SVM.
    X_train = text[main_pre].iloc[train_idx].to_numpy()
    s1 = fit_balanced_svm(main_pre, X_train, y_train, folds,
                          seed=42, cache=cache, n_jobs=args.jobs)
    s1["Mã"] = "S1"

    pipe, grid, _ = make_pipe("TF-IDF", "SVM", seed=42, cache=cache, n_jobs=args.jobs)
    start = time.time()
    s2_model, s2_cv, s2_params, s2_count = undersampled_search(pipe, grid, X_train, y_train, folds, 42)
    s2 = {"Mã": "S2", "Tiền xử lý": main_pre, "Đặc trưng": "TF-IDF", "Thuật toán": "SVM",
          "Chiến lược": "Under-sampling 2:1", **s2_cv, "Tham số": s2_params,
          "Thời gian (s)": time.time() - start, "_model": s2_model, "_threshold": 0.0}
    assert s2_count == len(sample_training_indices(y_train, np.arange(len(y_train)), seed=42))

    start = time.time()
    s3_model, s3_cv, threshold = tune_svm_threshold(focus["_model"], X_train, y_train, folds)
    s3 = {"Mã": "S3", "Tiền xử lý": main_pre, "Đặc trưng": "TF-IDF", "Thuật toán": "SVM",
          "Chiến lược": "Điều chỉnh ngưỡng", **s3_cv,
          "Tham số": {**focus["Tham số"], "threshold": threshold},
          "Thời gian (s)": time.time() - start, "_model": s3_model, "_threshold": threshold}

    candidates = baselines + [s1, s2, s3]
    best = max(candidates, key=lambda r: r["CV F1 (spam)"])
    print(f"Giai đoạn 2: chọn {best['Mã']} theo CV F1 spam = {best['CV F1 (spam)']:.4f}")

    # GIAI ĐOẠN 3 — Đo trên test một lần sau khi đã chốt lựa chọn bằng CV.
    for record in candidates:
        X_test = text[record["Tiền xử lý"]].iloc[test_idx].to_numpy()
        evaluate(record, X_test, y_test)
    print(f"Giai đoạn 3: {best['Mã']} test P={best['Precision']:.4f}, "
          f"R={best['Recall']:.4f}, F1={best['F1']:.4f}")

    # Chỉ xuất các cột có thể ghi CSV; _model và mảng _pred/_score ở trong RAM.
    columns = ["Mã", "Tiền xử lý", "Đặc trưng", "Thuật toán", "Chiến lược",
               "CV Precision (spam)", "CV Recall (spam)", "CV F1 (spam)",
               "Accuracy", "Precision", "Recall", "F1", "AP", "Tham số", "Ngưỡng"]
    rows = []
    for record in candidates:
        row = {}
        for name in columns:
            row[name] = record["_threshold"] if name == "Ngưỡng" else record[name]
        rows.append(row)
    pd.DataFrame(rows, columns=columns).to_csv(output_dir / "metrics.csv", index=False,
                                                encoding="utf-8-sig")

    # GIAI ĐOẠN 4 — Lưu pipeline đã fit + ngưỡng + quy tắc từ dừng.
    bundle = {"model": best["_model"], "threshold": best["_threshold"],
              "remove_stop": best["Tiền xử lý"] == "có loại stop words"}
    # Bỏ đường cache tạm trước khi lưu để file .joblib độc lập với thư mục tạm.
    bundle["model"].memory = None
    model_path = output_dir / "spam_ham_ml_only.joblib"
    joblib.dump(bundle, model_path)

    # Nạp lại và xác nhận nhãn trên test không đổi sau khi lưu.
    loaded = joblib.load(model_path)
    X_test = text[best["Tiền xử lý"]].iloc[test_idx].to_numpy()
    reloaded_pred, _ = predict_record({"_model": loaded["model"],
                                      "_threshold": loaded["threshold"]}, X_test)
    assert np.array_equal(reloaded_pred, best["_pred"])

    # Bảng này cho phép đối chiếu từng ID test và nhãn dự đoán của model chọn.
    pd.DataFrame({"id": data.iloc[test_idx]["id"].to_numpy(),
                  "label": data.iloc[test_idx]["label"].to_numpy(),
                  "prediction": np.where(best["_pred"] == 1, "spam", "ham"),
                  "score": best["_score"]}).to_csv(output_dir / "test_predictions.csv",
                                                    index=False, encoding="utf-8-sig")
    summary = {"source_csv": str(csv_path), "train_rows": len(train_idx),
               "test_rows": len(test_idx), "selected": best["Mã"],
               "cv_f1_spam": float(best["CV F1 (spam)"]), "test_f1_spam": float(best["F1"]),
               "threshold": float(best["_threshold"])}
    (output_dir / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                                  encoding="utf-8")
    print("Giai đoạn 4: đã lưu", model_path)
    print(predict_spam(["Cho mình hỏi lịch thi với ạ",
                        "Liên hệ mua bán qua 0901234567"], loaded).to_string(index=False))
