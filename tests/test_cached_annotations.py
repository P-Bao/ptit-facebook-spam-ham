import pandas as pd
import pytest

from pipeline.label.cached_annotations import attach_annotation_ids


def fixture():
    return pd.DataFrame({"post_id": ["a", "b"], "comment_type": ["comment", "reply"],
                         "text": ["hello", "offer"], "cleaned_text": ["hello", "offer"]})


def test_cache_ids_follow_content_when_rows_are_reordered():
    index = fixture()
    frame = index.iloc[[1]].reset_index(drop=True)
    result, cache = attach_annotation_ids(frame, index, {"cmt_0": {"label": "ham"}, "cmt_1": {"label": "spam"}})
    assert result.id.tolist() == ["cmt_1"]
    assert set(cache) == {"cmt_1"}
    assert "id" not in frame


def test_unseen_comments_get_stable_unlabeled_ids():
    index = fixture()
    frame = index.iloc[[0]].copy()
    frame["post_id"] = "new"
    result, cache = attach_annotation_ids(frame, index, {"cmt_0": {"label": "ham"}})
    repeated, _ = attach_annotation_ids(frame, index, {})
    assert result.id.iloc[0].startswith("comment_")
    assert result.id.tolist() == repeated.id.tolist()
    assert cache == {}


def test_mismatched_text_and_unknown_cache_ids_are_rejected():
    index = fixture()
    frame = index.copy()
    frame.loc[0, "cleaned_text"] = "different"
    with pytest.raises(ValueError, match="khác"):
        attach_annotation_ids(frame, index, {})
    with pytest.raises(ValueError, match="ID"):
        attach_annotation_ids(index, index, {"unknown": {"label": "ham"}})
