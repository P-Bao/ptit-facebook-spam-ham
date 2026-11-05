# Hiển thị bảng và mã nguồn trong notebook

| File | Hàm và nhiệm vụ |
|---|---|
| `notebook_display.py` | `table_html`: bảng nền sáng/chữ tối; `display_readable`: render bảng hoặc hình/Markdown; `code_excerpt_html`: trích source bằng AST; `file_record`: thông tin file đã ghi |
| `__init__.py` | Khai báo package |

Cả ba notebook gọi module này sau khi xử lý dữ liệu. Bảng xem trước mặc định
12 dòng, mỗi ô cắt hiển thị ở 600 ký tự; file/DataFrame đầy đủ không bị rút gọn.
Khối code dùng nền tối/chữ sáng và có số dòng nguồn để dễ mở file. Notebook 01
bọc các trích đoạn trong khối thu gọn; bản thân `code_excerpt_html` trả HTML.

`code_excerpt_html()` chỉ đọc/hiển thị mã, không import hoặc thực thi hàm được
trích. Cell gọi làm sạch, huấn luyện, ghi CSV hoặc suy luận mới thực hiện bước
đó. Không có CLI và không cần chạy module này trước các module khác.

`##`, `[CLS]`, `[SEP]`, `[UNK]` trong bảng token Notebook 03 là ký hiệu tokenizer,
không phải do CSS của module. Giải thích và ví dụ nằm tại mục 3 Notebook 03 và
[hướng dẫn đọc token](../../docs/guides/DOC_TOKEN_VA_KET_QUA.md).
