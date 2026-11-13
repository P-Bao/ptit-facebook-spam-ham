import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from pipeline.collect.json_to_csv import flatten_posts
from pipeline.label import auto_accept_labels
from pipeline.label.label_utils import join_metadata, normalize_prediction, record_ids
from tools.dataset_versions.merge_labeled_data import load_phase2, validate_labeled


def test_flatten_preserves_comments_and_replies():
    rows = flatten_posts([{"post_id": "p1", "comments": [{"text": " first ", "reaction_count": "2",
                           "replies": [{"text": "reply", "reaction_count": None}]}]}])
    assert [row["text"] for row in rows] == ["first", "reply"]
    assert [row["comment_type"] for row in rows] == ["comment", "reply"]


@pytest.mark.parametrize("confidence", [None, "bad", np.nan, float("inf"), -1, 2])
def test_bad_confidence_cannot_auto_accept(confidence):
    result = normalize_prediction({"label": "spam", "confidence": confidence})
    assert result["confidence"] == 0


def test_unknown_label_is_not_accepted():
    assert normalize_prediction({"label": "unknown", "confidence": .99})["label"] is None


def test_record_ids_prefer_original_id():
    assert record_ids(pd.DataFrame({"id": ["p2_cmt_7"]})) == ["p2_cmt_7"]
    with pytest.raises(ValueError, match="ID"):
        record_ids(pd.DataFrame({"id": ["x", "x"]}))


@pytest.mark.parametrize("cache", [{}, {"p2_cmt_0": {"label": "spam", "confidence": .9}}])
def test_auto_accept_phase2_and_empty_cache_without_server(tmp_path, monkeypatch, cache):
    cache_path, csv_path, output = tmp_path / "cache.json", tmp_path / "data.csv", tmp_path / "out.csv"
    cache_path.write_text(json.dumps(cache), encoding="utf-8")
    pd.DataFrame({"cleaned_text": ["example"], "text": ["example"]}).to_csv(csv_path, index=False)
    monkeypatch.setattr("sys.argv", ["auto_accept_labels", "--cache", str(cache_path),
        "--csv-in", str(csv_path), "--export-csv", str(output), "--skip-argilla"])
    auto_accept_labels.main()
    result = pd.read_csv(output)
    if cache:
        assert result.loc[0, "official_label"] == "spam"
        assert result.loc[0, "id"] == "p2_cmt_0"
    else:
        assert result["official_label"].isna().all()


def test_metadata_mismatch_and_unknown_ids_do_not_drop_rows():
    lookup = pd.DataFrame({"cleaned_text": ["original"], "text": ["raw"]})
    with pytest.raises(ValueError, match="khác"):
        join_metadata(pd.DataFrame({"id": ["cmt_0"], "cleaned_text": ["wrong"]}), lookup)
    with pytest.raises(ValueError, match="ID"):
        join_metadata(pd.DataFrame({"id": ["other"], "cleaned_text": ["original"]}), lookup)
    result = join_metadata(pd.DataFrame({"id": ["p2_cmt_0"], "cleaned_text": ["original"]}), lookup)
    assert len(result) == 1 and result.loc[0, "text"] == "raw"


def test_metadata_mixed_phase_needs_explicit_ids():
    rows = pd.DataFrame({"id": ["cmt_0", "p2_cmt_0"], "cleaned_text": ["a", "b"]})
    with pytest.raises(ValueError, match="nhiều phase"):
        join_metadata(rows, pd.DataFrame({"cleaned_text": ["a"]}))
    lookup = rows.copy()
    lookup["text"] = ["raw a", "raw b"]
    assert join_metadata(rows, lookup)["text"].tolist() == ["raw a", "raw b"]


def test_merge_preserves_phase2_id_and_rejects_duplicate_ids(tmp_path):
    path = tmp_path / "phase2.csv"
    pd.DataFrame({"id": ["p2_cmt_7"], "official_label": ["spam"], "cleaned_text": ["example"]}).to_csv(path, index=False)
    result = load_phase2(path)
    assert result.id.tolist() == ["p2_cmt_7"]
    with pytest.raises(ValueError, match="ID"):
        validate_labeled(pd.concat([result, result]), "test")


def test_merge_rejects_missing_id_before_prefixing(tmp_path):
    path = tmp_path / "missing.csv"
    pd.DataFrame({"id": [None], "label": ["spam"], "cleaned_text": ["example"]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="ID"):
        load_phase2(path)


def test_ai_no_upload_uses_csv_without_argilla_or_http(tmp_path, monkeypatch):
    from pipeline.label import ai_assisted_spam_ham as ai
    csv_path, cache = tmp_path / "csv.csv", tmp_path / "cache.json"
    pd.DataFrame({"id": ["p2_cmt_0"], "cleaned_text": ["example"], "text": ["example"]}).to_csv(csv_path, index=False)
    cache.write_text(json.dumps({"p2_cmt_0": {"label": "ham", "confidence": .9}}), encoding="utf-8")
    def forbidden(*args, **kwargs):
        pytest.fail("Không được kết nối dịch vụ trong ca kiểm tra offline")
    monkeypatch.setattr(ai.rg, "Argilla", forbidden)
    monkeypatch.setattr(ai.requests, "post", forbidden)
    monkeypatch.setattr("sys.argv", ["ai", "--api-key", "test-not-a-real-key", "--no-upload",
                                    "--csv", str(csv_path), "--cache", str(cache)])
    ai.main()


def test_ai_argilla_environment_is_used(monkeypatch):
    from pipeline.label import ai_assisted_spam_ham as ai
    monkeypatch.setenv("ARGILLA_API_URL", "http://example.invalid:6900")
    monkeypatch.setenv("ARGILLA_API_KEY", "test-not-a-real-key")
    monkeypatch.setattr("sys.argv", ["ai"])
    args = ai.parse_args()
    assert args.argilla_url == "http://example.invalid:6900"
    assert args.argilla_key == "test-not-a-real-key"


def test_download_empty_valid_label_result_keeps_schema(monkeypatch):
    from pipeline.label import download_labeled_data as download
    response = SimpleNamespace(status="submitted", value="invalid")
    record = SimpleNamespace(responses=[response], id="cmt_0", fields={"cleaned_text": "example"})
    dataset = SimpleNamespace(records=[record])
    client = SimpleNamespace(me=SimpleNamespace(username="test", role="owner"), datasets=lambda **kw: dataset)
    monkeypatch.setattr(download.rg, "Argilla", lambda **kw: client)
    monkeypatch.setattr(download, "build_records_iter", lambda ds: [record])
    monkeypatch.setattr("sys.argv", ["download", "--dry-run"])
    download.main()  # 0 nhãn hợp lệ không được gây KeyError ở df['label'].


def test_restructured_csv_pipeline_cli_without_services(tmp_path):
    root = Path(__file__).resolve().parents[1]
    phase1, source, flattened = tmp_path / "p1.csv", tmp_path / "posts.json", tmp_path / "flat.csv"
    cleaned, cache, accepted, merged = [tmp_path / name for name in ["clean.csv", "cache.json", "accepted.csv", "merged.csv"]]
    pd.DataFrame({"id": ["cmt_0"], "label": ["ham"], "cleaned_text": ["existing"], "text": ["existing"]}).to_csv(phase1, index=False)
    source.write_text(json.dumps([{"post_id": "post", "comments": [
        {"text": "existing"}, {"text": "new https://example.com", "replies": []}]}]), encoding="utf-8")
    cache.write_text(json.dumps({"p2_cmt_0": {"label": "spam", "confidence": .9}}), encoding="utf-8")
    commands = [
        ["pipeline.collect.json_to_csv", "--input", str(source), "--phase1-csv", str(phase1), "--output", str(flattened)],
        ["pipeline.prepare.clean_for_labeling", "--input", str(flattened), "--output", str(cleaned)],
        ["pipeline.label.auto_accept_labels", "--cache", str(cache), "--csv-in", str(cleaned), "--export-csv", str(accepted), "--skip-argilla"],
        ["tools.dataset_versions.merge_labeled_data", "--phase1", str(phase1), "--phase2", str(accepted), "--output", str(merged)],
    ]
    for args in commands:
        result = subprocess.run([sys.executable, "-m", *args], cwd=root, capture_output=True, text=True, encoding="utf-8")
        assert result.returncode == 0, result.stderr
    result = pd.read_csv(merged)
    assert result.id.tolist() == ["cmt_0", "p2_cmt_0"]
    assert result.label.tolist() == ["ham", "spam"]
    assert result.source.tolist() == ["phase1", "phase2"]
    assert "urltoken" in result.iloc[1].cleaned_text


def test_presentation_notebooks_use_project_modules():
    root = Path(__file__).resolve().parents[1]
    names = ["01_thu_thap_va_gan_nhan.ipynb", "02_thuc_nghiem_va_demo.ipynb"]
    for name in names:
        nb = json.loads((root / "notebooks" / name).read_text(encoding="utf-8"))
        code = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
        assert not any(line.startswith("%%writefile") for line in code.splitlines())
        assert "from pipeline." in code
    experiment = json.loads((root / "notebooks" / names[1]).read_text(encoding="utf-8"))
    code = "\n".join("".join(c["source"]) for c in experiment["cells"] if c["cell_type"] == "code")
    assert "SPAMHAM_SOURCE_DIR" not in code
    assert "collection_audit" not in code
    assert "from pipeline.modeling.predict import predict_spam" in code
