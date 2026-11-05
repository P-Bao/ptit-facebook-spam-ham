"""pipeline/presentation/notebook_display.py
Hiển thị code và bảng với màu tương phản cố định trong notebook.

Đầu vào: DataFrame/Series hoặc đường dẫn/hàm Python cần minh họa.
Xử lý: escape nội dung; đặt màu inline để không phụ thuộc theme sáng/tối của IDE.
Đầu ra: HTML dễ đọc và thông tin file; không thực thi đoạn code được show.
Liên quan: notebooks/01_thu_thap_va_gan_nhan.ipynb; notebooks/02_thuc_nghiem_va_demo.ipynb.
"""

import ast
from html import escape
from pathlib import Path

import pandas as pd
from IPython.display import HTML, display as ipython_display

CELL_STYLE = ("color:#111827!important;background:#ffffff!important;"
              "border:1px solid #cbd5e1!important;padding:9px 11px;"
              "vertical-align:top;text-align:left;white-space:pre-wrap;overflow-wrap:anywhere;")
HEAD_STYLE = ("color:#0f172a!important;background:#e2e8f0!important;"
              "border:1px solid #94a3b8!important;padding:9px 11px;text-align:left;")


def table_html(frame, title=None, max_rows=12):
    """Chỉ cắt phần xem trước; DataFrame/file đầy đủ không bị thay đổi."""
    if isinstance(frame, pd.Series):
        frame = frame.to_frame()
    shown = frame.head(max_rows)
    parts = ['<div class="ptit-table" style="color:#111827!important;background:#ffffff!important;'
             'padding:12px;border-radius:6px;overflow-x:auto;margin:8px 0;">']
    if title:
        parts.append('<div style="color:#0f172a!important;background:#ffffff!important;'
                     'font-weight:700;margin-bottom:8px;">'+escape(str(title))+'</div>')
    parts.append('<table style="border-collapse:collapse;width:100%;color:#111827!important;background:#ffffff!important;"><thead><tr>')
    parts.append('<th style="'+HEAD_STYLE+'">'+escape(str(frame.index.name or "#"))+'</th>')
    for column in shown.columns:
        parts.append('<th style="'+HEAD_STYLE+'">'+escape(str(column))+'</th>')
    parts.append('</tr></thead><tbody>')
    for row in shown.itertuples(index=True, name=None):
        parts.append('<tr>')
        for value in row:
            text = "" if value is None else str(value)
            if len(text) > 600:
                text = text[:600]+" …"
            parts.append('<td style="'+CELL_STYLE+'">'+escape(text)+'</td>')
        parts.append('</tr>')
    parts.append('</tbody></table><div style="color:#334155!important;background:#ffffff!important;'
                 'font-size:12px;margin-top:7px;">'
                 +f'Hiển thị {len(shown):,}/{len(frame):,} dòng · {len(frame.columns)} cột'
                 +'</div></div>')
    return "".join(parts)


def display_readable(*objects, **kwargs):
    """Thay cách render bảng; Markdown/hình/đối tượng khác dùng display bình thường."""
    for obj in objects:
        ipython_display(HTML(table_html(obj)) if isinstance(obj, (pd.DataFrame, pd.Series)) else obj, **kwargs)


def code_excerpt_html(path, function_name, max_lines=35, title=None):
    """Đọc AST rồi hiển thị source, không import hoặc gọi hàm trong file đó."""
    path = Path(path)
    source = path.read_text(encoding="utf-8")
    node = next(n for n in ast.walk(ast.parse(source))
                if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == function_name)
    lines = ast.get_source_segment(source, node).splitlines()
    caption = f"{title or path.name}:{node.lineno} — {function_name} · {min(len(lines), max_lines)}/{len(lines)} dòng"
    return ('<div class="ptit-code" style="background:#0f172a!important;color:#f1f5f9!important;'
            'padding:14px;border:1px solid #475569;border-radius:6px;margin:10px 0;">'
            '<div style="background:#0f172a!important;color:#cbd5e1!important;font-size:12px;margin-bottom:10px;">'
            +escape(caption)+'</div><pre style="background:#0f172a!important;color:#f1f5f9!important;'
            'margin:0;white-space:pre-wrap;overflow-wrap:anywhere;font:13px/1.6 Consolas,monospace;">'
            '<code style="background:#0f172a!important;color:#f1f5f9!important;">'
            +escape("\n".join(lines[:max_lines]))+'</code></pre></div>')


def file_record(path, rows, content, root=None, status="vừa ghi"):
    """Thông tin file thực trên đĩa để người xem nhận biết đầu vào/đầu ra."""
    path = Path(path).resolve()
    label = path.relative_to(Path(root).resolve()).as_posix() if root and path.is_relative_to(Path(root).resolve()) else str(path)
    return {"file": label, "trạng thái": status, "số bản ghi": rows,
            "KB": round(path.stat().st_size/1024, 1), "chứa": content}
