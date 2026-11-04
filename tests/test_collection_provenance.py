import ast
import json
from pathlib import Path

import pandas as pd
import pytest

from pipeline.step_01_collect.audit_source import audit_source_bundle, read_configured_groups
from pipeline.step_01_collect.convert_phase2_json import flatten_posts
from pipeline.step_02_preprocess.clean_data_facebook_spamham import clean_facebook_comments, preprocess
from pipeline.step_03_label import auto_accept_labels
from pipeline.step_03_label.label_utils import join_metadata


def test_legacy_cleaning_keeps_row_positions(tmp_path):
    source, current, legacy = [tmp_path / name for name in ["raw.csv", "current.csv", "legacy.csv"]]
    pd.DataFrame({"post_id": ["p"] * 4, "comment_type": ["comment"] * 4,
                  "text": [".", "😂", "hello", "hello"]}).to_csv(source, index=False)
    clean_facebook_comments([source], current)
    clean_facebook_comments([source], legacy, keep_punctuation_only=True)
    assert pd.read_csv(current).text.tolist() == ["hello"]
    kept = pd.read_csv(legacy, keep_default_na=False)
    assert kept.text.tolist() == [".", "😂", "hello"]
    assert kept.cleaned_text.tolist() == [".", "", "hello"]


def test_metadata_allows_uploader_raw_fallback_but_not_wrong_text():
    lookup = pd.DataFrame({"cleaned_text": [""], "text": ["😂"], "comment_type": ["comment"]})
    labeled = pd.DataFrame({"id": ["cmt_0"], "cleaned_text": ["😂"], "label": ["ham"]})
    assert join_metadata(labeled, lookup).text.tolist() == ["😂"]
    labeled["cleaned_text"] = "wrong"
    with pytest.raises(ValueError, match="khác"):
        join_metadata(labeled, lookup)


def test_auto_export_uses_same_display_fallback(tmp_path, monkeypatch):
    source, cache, output = [tmp_path / name for name in ["clean.csv", "cache.json", "export.csv"]]
    pd.DataFrame({"cleaned_text": [""], "text": ["😂"]}).to_csv(source, index=False)
    cache.write_text(json.dumps({"cmt_0": {"label": "ham", "confidence": .9}}), encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["auto", "--csv-in", str(source), "--cache", str(cache),
                                    "--export-csv", str(output), "--skip-argilla"])
    auto_accept_labels.main()
    assert pd.read_csv(output).cleaned_text.tolist() == ["😂"]


def make_source_fixture(folder, text="new"):
    posts = [{"post_id": "p", "group_id": "g", "group_name": "PTIT fixture",
              "created_at": "2026-09-27 10:00:00", "scraped_at": "2026-09-27 12:00:00",
              "comments": [{"text": text, "reaction_count": "0", "replies": []}]}]
    (folder / "groups_data_phase2.json").write_text(json.dumps(posts), encoding="utf-8")
    flat = pd.DataFrame(flatten_posts(posts))
    flat.to_csv(folder / "comments_phase2.csv", index=False)
    cleaned = flat.copy()
    cleaned.insert(list(cleaned.columns).index("text") + 1, "cleaned_text", cleaned.text.map(preprocess))
    cleaned.to_csv(folder / "comments_phase2_cleaned.csv", index=False)
    phase1 = pd.DataFrame({"id": ["cmt_0"], "label": ["ham"], "cleaned_text": ["existing"],
                          "text": ["existing"], "comment_type": ["comment"], "reaction_count": ["0"]})
    phase1.to_csv(folder / "comments_labeled_export.csv", index=False)
    phase2 = pd.DataFrame({"id": ["cmt_0"], "label": ["spam"], "cleaned_text": [preprocess(text) or text],
                          "text": [text], "comment_type": ["comment"], "reaction_count": ["0"]})
    phase2.to_csv(folder / "comments_labeled_phase2.csv", index=False)
    (folder / "predictions_glm_cache_phase2.json").write_text(json.dumps({"cmt_0": {"label": "spam", "confidence": .9}}), encoding="utf-8")
    phase2["id"] = "p2_cmt_0"
    merged = pd.concat([phase1.assign(source="phase1"), phase2.assign(source="phase2")], ignore_index=True)
    merged.to_csv(folder / "comments_labeled_merged.csv", index=False)
    training = merged.drop(columns="source")
    training.loc[training.id == "p2_cmt_0", "label"] = "ham"
    path = folder / "v7.csv"
    training.to_csv(path, index=False)
    return path


@pytest.mark.parametrize("text", ["new", "😂"])
def test_audit_separates_exact_lineage_from_label_changes(tmp_path, text):
    path = make_source_fixture(tmp_path, text)
    result = audit_source_bundle(tmp_path, path, preprocess, flatten_posts, [{"id": "other", "name": "Configured"}])
    assert result["conversion"]["exact_reproduction"]
    assert result["cleaning"]["legacy_exact_reproduction"]
    assert result["merge"]["exact_reproduction"]
    assert result["raw_group_ids_not_in_config"] == ["g"]
    assert result["v7_comparison"]["changed_fields"]["label"] == 1
    assert result["v7_comparison"]["changed_fields"]["text"] == 0
    assert result["annotation"]["cleaned_text_fallback_to_raw"] == (1 if text == "😂" else 0)


def test_read_config_does_not_execute_handoff_code(tmp_path):
    (tmp_path / "scrape_ptit_groups.py").write_text(
        'raise RuntimeError("Do not execute")\nPTIT_GROUPS = [{"id": "123", "name": "PTIT", "urls": []}]\n', encoding="utf-8")
    assert read_configured_groups(tmp_path)[0]["id"] == "123"


def test_notebook_embeds_exact_snapshot_transform_functions():
    root = Path(__file__).resolve().parents[1]
    nb = json.loads((root / "notebooks/spam-ham-comments.ipynb").read_text(encoding="utf-8"))
    helper = next("".join(c["source"]).split("\n", 1)[1] for c in nb["cells"]
                  if c["cell_type"] == "code" and "".join(c["source"]).startswith("%%writefile spam_source_helpers.py"))
    extracted = {n.name: ast.dump(n) for n in ast.parse(helper).body if isinstance(n, ast.FunctionDef)}
    for path, names in [("pipeline/step_01_collect/convert_phase2_json.py", ["to_int", "flatten_posts"]),
                        ("pipeline/step_02_preprocess/clean_data_facebook_spamham.py", ["preprocess"])]:
        originals = {n.name: ast.dump(n) for n in ast.parse((root/path).read_text(encoding="utf-8")).body if isinstance(n, ast.FunctionDef)}
        assert all(extracted[name] == originals[name] for name in names)


def test_real_drive_snapshot_when_available():
    root = Path(__file__).resolve().parents[1]
    folder = root / "data/source_drive"
    if not (folder / "groups_data_phase2.json").exists() or not (root / "data/labeled_v7.csv").exists():
        pytest.skip("Snapshot thật không nằm trong Git; dùng tests fixture khi chưa tải")
    configured = read_configured_groups(root / "facebook_post_comment_scraper_code/facebook_post_comment_scraper")
    result = audit_source_bundle(folder, root / "data/labeled_v7.csv", preprocess, flatten_posts, configured)
    assert result["conversion"] == {"raw_rows": 13833, "empty_removed": 129,
        "phase1_duplicate_removed": 4135, "flat_rows": 9569, "exact_reproduction": True}
    assert result["cleaning"]["legacy_rows"] == 9398
    assert result["cleaning"]["current_filter_would_remove"] == 42
    assert result["merge"]["exact_reproduction"] and result["merge"]["rows"] == 38233
    assert result["v7_comparison"]["changed_fields"] == {"label": 1366, "cleaned_text": 0,
                                                       "text": 0, "comment_type": 0, "reaction_count": 0}
