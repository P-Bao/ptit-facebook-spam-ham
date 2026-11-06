# Biểu diễn văn bản bằng FastText

| File | Nhiệm vụ |
|---|---|
| `spam_features.py` | Class `FastTextVectorizer` tương thích scikit-learn: `fit()` học vector từ/IDF, `transform()` tổng hợp thành vector câu |
| `__init__.py` | Khai báo package |

Notebook 02 mục 5 gọi `modeling.experiments.make_pipe()` để đưa vectorizer vào
pipeline. FastText dùng skip-gram, vector 100 chiều, subword 3–5 ký tự, seed 42,
một worker. Tìm hai cách pooling: trung bình và trung bình có trọng số TF-IDF.

Thứ tự: **chuỗi token → `fit()` trên train của fold → `transform()` validation
→ NB/SVM**. Sau khi chọn cấu hình, fit lại trên toàn bộ train rồi transform
test. Không học FastText hoặc IDF trên test. Subword có thể tạo vector cho từ
chưa gặp, không có nghĩa dự đoán spam luôn đúng.

BoW/TF-IDF dùng trực tiếp `CountVectorizer`/`TfidfVectorizer` trong
[`modeling/experiments.py`](../modeling/experiments.py), không có hai file
vectorizer tự viết riêng. Thư mục này không có CLI; chạy
`scripts/run_notebook.py --notebook 02` để thực hiện đầy đủ thí nghiệm.
