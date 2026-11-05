from pathlib import Path

import pandas as pd

from pipeline.presentation.notebook_display import code_excerpt_html, file_record, table_html


def test_table_preserves_index_and_escapes_comment_html():
    frame = pd.DataFrame({"text": ["<script>unsafe</script>", "actual comment"]}, index=["ham", "spam"])
    before = frame.copy(deep=True)
    html = table_html(frame, max_rows=1)
    assert "&lt;script&gt;unsafe&lt;/script&gt;" in html and "<script>unsafe" not in html
    assert ">ham</td>" in html
    assert "1/2" in html
    assert "color:#111827!important" in html and "background:#ffffff!important" in html
    pd.testing.assert_frame_equal(frame, before)


def test_code_excerpt_is_escaped_and_not_executed(tmp_path):
    path = tmp_path / "example.py"
    path.write_text('raise RuntimeError("must not run")\ndef example():\n    return "<script>"\n', encoding="utf-8")
    html = code_excerpt_html(path, "example")
    assert "&lt;script&gt;" in html
    assert "must not run" not in html
    assert "background:#0f172a!important" in html and "color:#f1f5f9!important" in html


def test_file_summary_describes_real_written_file(tmp_path):
    path = tmp_path / "output.csv"
    path.write_text("text,label\nhello,ham\n", encoding="utf-8")
    record = file_record(path, 1, "text,label", root=tmp_path)
    assert record["file"] == "output.csv"
    assert record["số bản ghi"] == 1
