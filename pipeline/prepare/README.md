# Làm sạch và tiền xử lý văn bản

| File | Hàm chính | Đầu vào → đầu ra |
|---|---|---|
| `clean_for_labeling.py` | `preprocess`, `clean_facebook_comments` | CSV có `text` → giữ nội dung gốc, thêm `cleaned_text`, khử trùng và lọc dòng không có chữ/số |
| `preprocess_for_model.py` | `clean_text`, `mask_names`, `preprocess` | `cleaned_text` và text gốc → chuỗi token underthesea, với lựa chọn giữ/loại từ dừng |
| `__init__.py` | Khai báo package | Cho phép import `pipeline.prepare` |

Thứ tự sử dụng:

1. **Notebook 01 mục 4:** `clean_facebook_comments()` xử lý bảng vừa chuyển từ
   JSON. Khóa khử trùng gồm `post_id`, `comment_type`, `text`. URL/số điện thoại
   được thay bằng `urltoken`/`phonetoken`, không xóa mất tín hiệu. Sau đó nối ID
   annotation và chuyển sang bước gán nhãn.
2. **Notebook 02 mục 3:** `preprocess()` chuẩn bị đầu vào BoW/TF-IDF/FastText:
   chuẩn hóa, che một số tên theo quy tắc, chuyển chữ thường, tách từ, thêm tín
   hiệu emoji. Lưu toàn bộ v7 và hai cột token vào
   `data/processed/v7_preprocessed.csv`; mục 5 chọn giữ từ dừng bằng CV.
3. **Khi dự đoán S3:** `pipeline.inference.predict` gọi lại cùng `preprocess()`
   theo lựa chọn từ dừng trong bundle. Không fit lại vectorizer.

Đây là hai mục đích xử lý khác nhau: chuẩn bị bảng cho người/AI gán nhãn và tạo
biểu diễn đầu vào bộ phân loại. Không gán nhãn lại v7, không tự đổi nhãn qua
tiền xử lý. Che tên bằng quy tắc không bảo đảm ẩn danh hoàn toàn.

CLI làm sạch, chạy từ gốc project:

```powershell
.\.venv\Scripts\python.exe -m pipeline.prepare.clean_for_labeling --input data/intermediate/comments.csv --output data/intermediate/comments_cleaned.csv
```

`preprocess_for_model.py` là thư viện được Notebook 02 gọi, không có CLI huấn
luyện. viBERT dùng chuẩn hóa nhẹ trong `modern/data.py`, không dùng chuỗi token
underthesea. Xem [hướng dẫn chạy](../../README.md).
