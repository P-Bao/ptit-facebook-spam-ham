"""Kiểm tra kết quả notebook local và model joblib đã lưu; không huấn luyện lại."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    statuses = {}
    for number in ("01", "02", "03", "04"):
        status = json.loads((ROOT / f".work/runs/{number}/execution_status.json").read_text(encoding="utf-8"))
        assert status["success"], f"Notebook {number} chưa chạy thành công"
        statuses[number] = {key: status[key] for key in ("success", "seconds", "executed_code_cells")}
    folder = ROOT / "artifacts/spamham"
    predictions = pd.read_csv(folder / "test_predictions.csv", keep_default_na=False)
    bundle = joblib.load(folder / "spam_ham_ml_only.joblib")
    scores = bundle["model"].decision_function(predictions["Văn bản xử lý"].to_numpy())
    assert np.allclose(scores, predictions["Điểm mô hình"], rtol=0, atol=1e-12)
    labels = np.where(scores >= bundle["threshold"], "spam", "ham")
    assert np.array_equal(labels, predictions["Dự đoán"])
    manifest = json.loads((folder / "run_manifest.json").read_text(encoding="utf-8"))
    metrics = pd.read_csv(folder / "metrics.csv")
    audit = pd.read_csv(ROOT / "artifacts/grid_audit/all_cv_configs.csv")
    result = {"notebooks": statuses, "saved_joblib_matches_all_test_predictions": True,
              "test_rows": len(predictions), "grid_configurations": len(audit),
              "selected": manifest["selected"],
              "metrics": metrics.drop(columns=["Thời gian (s)"]).to_dict(orient="records"),
              "vibert_mode": "review_existing_checkpoint_not_retraining",
              "live_facebook_argilla_llm_calls": False}
    output = ROOT / "results/local_verification_2026-11-13.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"OK: {len(predictions)} test predictions match saved model; {len(audit)} CV configurations")
    print(output)


if __name__ == "__main__":
    main()
