import os
from pathlib import Path
import subprocess
import sys

import joblib
import numpy as np
import pytest
from sklearn.model_selection import StratifiedKFold

from pipeline.modeling.predict import predict_spam
from pipeline.modeling.experiments import evaluate, fit_experiment, make_pipe
from pipeline.prepare.preprocess_for_model import preprocess


@pytest.mark.parametrize("feature,algorithm", [("BoW", "Naive Bayes"), ("TF-IDF", "SVM")])
def test_extracted_training_and_saved_bundle_inference(tmp_path, feature, algorithm):
    raw = ["học lịch thi tài liệu bạn" if i % 2 == 0 else "quảng cáo mua bán liên hệ" for i in range(24)]
    X = np.array([preprocess(text, text) for text in raw])
    y = np.arange(len(X)) % 2
    folds = list(StratifiedKFold(3).split(X, y))
    record = fit_experiment("test", feature, algorithm, X, y, folds, n_jobs=1)
    evaluate(record, X, y)
    assert record["F1"] == 1
    assert record["CV Precision (spam)"] == record["CV Recall (spam)"] == record["CV F1 (spam)"] == 1
    bundle = {"model": record["_model"], "threshold": record["_threshold"], "remove_stop": False}
    path = tmp_path / "bundle.joblib"
    joblib.dump(bundle, path)
    predicted = predict_spam(raw[:2], joblib.load(path))
    assert predicted.du_doan.tolist() == ["ham", "spam"]
    assert predict_spam([], bundle).empty
    if algorithm == "SVM":
        # Nếu ngưỡng không được áp dụng trong inference, mẫu spam vẫn bị đoán spam.
        bundle["threshold"] = float(record["_model"].decision_function(X).max() + 1)
        joblib.dump(bundle, path)
        assert predict_spam(raw[:2], joblib.load(path)).du_doan.tolist() == ["ham", "ham"]


@pytest.mark.parametrize("feature", ["BoW", "TF-IDF"])
def test_svm_baselines_do_not_mix_class_weights(feature):
    _, baseline_grid, _ = make_pipe(feature, "SVM")
    _, balanced_grid, _ = make_pipe(feature, "SVM", class_weight="balanced")
    assert baseline_grid["clf__class_weight"] == [None]
    assert balanced_grid["clf__class_weight"] == ["balanced"]


def test_crawler_entrypoint_help_after_separation():
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, PROXY="")
    output = root / "data/raw/posts.json"
    before = output.stat().st_mtime_ns if output.exists() else None
    result = subprocess.run([sys.executable, "-m", "pipeline.collect.scrape_ptit_groups", "--help"],
                            cwd=root, env=env, capture_output=True, text=True, encoding="utf-8")
    assert result.returncode == 0, result.stderr
    assert "--state-file" in result.stdout
    assert (output.stat().st_mtime_ns if output.exists() else None) == before


def test_default_stage_paths_connect():
    from pipeline.collect import json_to_csv
    from pipeline.prepare import clean_for_labeling
    from pipeline.label import upload_to_argilla, ai_assisted_spam_ham, auto_accept_labels, download_labeled_data
    from tools.dataset_versions import merge_labeled_data

    assert json_to_csv.DEFAULT_OUTPUT_CSV in clean_for_labeling.DEFAULT_INPUT_FILES
    cleaned = clean_for_labeling.DEFAULT_OUTPUT_CSV
    assert cleaned == upload_to_argilla.DEFAULT_CSV_PATH == ai_assisted_spam_ham.DEFAULT_CSV_PATH
    assert cleaned == auto_accept_labels.DEFAULT_CSV_PATH == download_labeled_data.DEFAULT_CSV_LOOKUP
    assert download_labeled_data.DEFAULT_OUTPUT_CSV == merge_labeled_data.DEFAULT_PHASE2_CSV
    assert json_to_csv.DEFAULT_PHASE1_CSV == merge_labeled_data.DEFAULT_PHASE1_CSV
