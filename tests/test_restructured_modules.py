import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import joblib
import numpy as np
import pytest
from sklearn.model_selection import StratifiedKFold

from pipeline.collect import snapshots
from pipeline.inference.predict import predict_spam
from pipeline.modeling.experiments import evaluate, fit_experiment
from pipeline.prepare.preprocess_for_model import preprocess


def snapshot_manifest(tmp_path, monkeypatch):
    manifest = tmp_path / "results/source_audit/drive_manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps([{"name": "groups_data_phase2.json", "bytes": 2,
        "download_url": "https://example.invalid/source", "sha256": hashlib.sha256(b"[]").hexdigest()}]), encoding="utf-8")
    monkeypatch.setattr(snapshots, "PROJECT_ROOT", tmp_path)


def test_existing_snapshot_hash_mismatch_is_not_overwritten(tmp_path, monkeypatch):
    snapshot_manifest(tmp_path, monkeypatch)
    source = tmp_path / "data/raw/groups_data_phase2.json"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"wrong")
    monkeypatch.setattr(snapshots, "urlopen", lambda *a, **k: pytest.fail("Must not download over existing file"))
    with pytest.raises(ValueError, match="hash"):
        snapshots.ensure_snapshots(tmp_path / "data", [source.name])
    assert source.read_bytes() == b"wrong"


def test_download_rejects_wrong_payload_before_writing(tmp_path, monkeypatch):
    snapshot_manifest(tmp_path, monkeypatch)

    class BadResponse:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self, limit): return b"bad"

    monkeypatch.setattr(snapshots, "urlopen", lambda *a, **k: BadResponse())
    with pytest.raises(ValueError, match="hash"):
        snapshots.ensure_snapshots(tmp_path / "data", ["groups_data_phase2.json"])
    assert not (tmp_path / "data/raw/groups_data_phase2.json").exists()


def phase1_manifest(tmp_path, monkeypatch):
    snapshot_manifest(tmp_path, monkeypatch)
    path = tmp_path / "results/source_audit/drive_manifest.json"
    entries = json.loads(path.read_text(encoding="utf-8"))
    entries.append({"name": "comments_labeled_export.csv", "bytes": 6,
                    "download_url": "https://example.invalid/labels",
                    "sha256": hashlib.sha256(b"phase1").hexdigest()})
    path.write_text(json.dumps(entries), encoding="utf-8")


@pytest.mark.parametrize("content", [b"phase1", b"wrong"])
def test_phase1_preparation_preserves_existing_file(tmp_path, monkeypatch, content):
    phase1_manifest(tmp_path, monkeypatch)
    target = tmp_path / "data/labeled/comments_labeled_phase1.csv"
    target.parent.mkdir(parents=True)
    target.write_bytes(content)
    monkeypatch.setattr(snapshots, "urlopen", lambda *a, **k: pytest.fail("Existing source must not be replaced"))
    if content == b"phase1":
        assert snapshots.prepare_phase1(tmp_path / "data") == target
    else:
        with pytest.raises(ValueError, match="hash"):
            snapshots.prepare_phase1(tmp_path / "data")
    assert target.read_bytes() == content


def test_phase1_preparation_downloads_verified_bytes_without_duplicate_copy(tmp_path, monkeypatch):
    phase1_manifest(tmp_path, monkeypatch)

    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self, limit): return b"phase1"

    monkeypatch.setattr(snapshots, "urlopen", lambda *a, **k: Response())
    cache = tmp_path / "cache"
    target = snapshots.prepare_phase1(tmp_path / "data", cache)
    assert target.read_bytes() == b"phase1"
    assert list(cache.iterdir()) == []


@pytest.mark.parametrize("feature,algorithm", [("BoW", "Naive Bayes"), ("TF-IDF", "SVM")])
def test_extracted_training_and_saved_bundle_inference(tmp_path, feature, algorithm):
    raw = ["học lịch thi tài liệu bạn" if i % 2 == 0 else "quảng cáo mua bán liên hệ" for i in range(24)]
    X = np.array([preprocess(text, text) for text in raw])
    y = np.arange(len(X)) % 2
    folds = list(StratifiedKFold(3).split(X, y))
    record = fit_experiment("test", feature, algorithm, X, y, folds, n_jobs=1)
    evaluate(record, X, y)
    assert record["F1"] == 1
    bundle = {"model": record["_model"], "threshold": .5, "remove_stop": False}
    path = tmp_path / "bundle.joblib"
    joblib.dump(bundle, path)
    predicted = predict_spam(raw[:2], joblib.load(path))
    assert predicted.du_doan.tolist() == ["ham", "spam"]
    assert predict_spam([], bundle).empty


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


def test_default_stage_paths_connect_without_overwriting_snapshots():
    from pipeline.collect import json_to_csv
    from pipeline.prepare import clean_for_labeling
    from pipeline.label import upload_to_argilla, ai_assisted_spam_ham, auto_accept_labels, download_labeled_data
    from tools.dataset_versions import merge_labeled_data

    assert json_to_csv.DEFAULT_OUTPUT_CSV in clean_for_labeling.DEFAULT_INPUT_FILES
    cleaned = clean_for_labeling.DEFAULT_OUTPUT_CSV
    assert cleaned == upload_to_argilla.DEFAULT_CSV_PATH == ai_assisted_spam_ham.DEFAULT_CSV_PATH
    assert cleaned == auto_accept_labels.DEFAULT_CSV_PATH == download_labeled_data.DEFAULT_CSV_LOOKUP
    assert download_labeled_data.DEFAULT_OUTPUT_CSV.name not in snapshots.SOURCE_LOCATIONS
    assert merge_labeled_data.DEFAULT_OUTPUT_CSV.name not in snapshots.SOURCE_LOCATIONS
