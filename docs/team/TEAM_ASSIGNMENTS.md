# Phân công nhiệm vụ dự án Spam/Ham

## Thành viên: Bảo – Cương – Phi – Phương

### Tổng quan phân công

| Thành viên | Phần việc |
|---|---|
| **Bảo** | Bước 1 (thu thập dữ liệu, JSON → CSV) + Bước 4 (huấn luyện 12 thí nghiệm) |
| **Cương** | Bước 2 (tiền xử lý) + BoW/TF‑IDF ở Bước 3 |
| **Phi** | FastText ở Bước 3, phân tích FastText, chia train/test |
| **Phương** | Bước 5 (đánh giá, so sánh, xử lý mất cân bằng) |

---

### Bước 1 – Thu thập dữ liệu
**Bảo**

- Tìm kiếm nguồn dữ liệu, thu thập bình luận thô từ Facebook, lưu dữ liệu gốc dưới dạng JSON.
- Chuyển đổi dữ liệu JSON sang CSV, kiểm tra tính toàn vẹn (loại bỏ bản ghi rỗng/trùng lặp), chuẩn bị file CSV sạch để các bước sau sử dụng.

---

### Bước 2 – Tiền xử lý dữ liệu
**Cương**

Áp dụng các kỹ thuật đã học: tách từ (tokenisation), loại bỏ stop‑words, chuẩn hoá văn bản (lower‑case, xoá ký tự đặc biệt). Kết quả là DataFrame sạch sẵn sàng để trích đặc trưng.

---

### Bước 3 – Trích chọn đặc trưng
Thử nghiệm đủ **3 phương pháp** trích chọn đặc trưng, chia làm hai phần:

**Cương – Biểu diễn dựa trên tần suất từ (nối tiếp ngay sau bước tách từ)**
- **Bag‑of‑Words (BoW)** – `CountVectorizer`, n‑gram (1,1) / (1,2), `min_df=2`, giữ dấu câu làm token.
- **TF‑IDF** – `TfidfVectorizer` (`sublinear_tf`), n‑gram (1,1) / (1,2), `min_df=2`.
- Giải thích sự khác nhau giữa BoW và TF‑IDF (vì sao TF‑IDF giảm trọng số từ phổ biến, nhấn mạnh token đặc trưng như `urltoken`, `phonetoken`, `zalo`).

**Phi – Word Embeddings (FastText)**
- Tự huấn luyện FastText Skip‑gram 100 chiều (subword n‑gram 3–5 ký tự) bằng `gensim` trên tập train, đóng gói thành `FastTextVectorizer` trong `spam_features.py`; vector câu = trung bình có trọng số TF‑IDF hoặc trung bình thường của vector từ. Giúp mô hình nắm được **ngữ nghĩa** và xử lý teencode/từ viết sai (OOV) như `ib`, `inbox`, `inboxxx`.
- Embedding được học **bên trong Pipeline** ở từng fold CV để không rò rỉ dữ liệu test; giảm `bucket=100_000` và chỉ giữ `model.wv` để tiết kiệm bộ nhớ.
- Phân tích riêng FastText: từ gần nghĩa (`most_similar`), kiểm tra vector cho từ chưa gặp khi train (OOV), đối chiếu các mẫu spam FastText bắt được mà TF‑IDF bỏ sót.
- Chia dữ liệu 80% train / 20% test theo hash văn bản (chống rò rỉ bình luận trùng lặp).

---

### Bước 4 – Xây dựng và huấn luyện mô hình
**Bảo**

Sử dụng 2 thuật toán phân loại cho cả 3 loại đặc trưng (6 tổ hợp × 2 kiểu tiền xử lý = 12 thí nghiệm):
- **Naive Bayes**: `MultinomialNB` cho BoW/TF‑IDF; `GaussianNB` cho FastText (vì vector embedding là số thực, có thể âm).
- **SVM**: `LinearSVC(class_weight="balanced")`; với FastText thêm `StandardScaler` trước SVM.

Tinh chỉnh siêu tham số bằng `GridSearchCV` 5 fold (tối ưu F1 lớp spam), dự đoán trên tập test, lưu mô hình tốt nhất (`spam_ham_ml_only.joblib`).

---

### Bước 5 – Đánh giá mô hình
**Phương**

Tính các chỉ số: Accuracy, Precision, Recall, F1‑Score, PR‑AUC, Confusion Matrix (lưới 3×2), khoảng tin cậy bootstrap 95%. So sánh hiệu suất giữa **3 phương pháp trích chọn đặc trưng** và 2 thuật toán; ảnh hưởng của việc loại stop words. Xử lý mất cân bằng dữ liệu (class‑weight, under‑sampling, threshold tuning) và đánh giá lại.

---

*Mỗi bước hoàn thành → bàn giao kết quả cho bước tiếp theo.*


## Trạng thái sau chỉnh sửa ngày 04/11/2026

- Notebook chính: `notebooks/spam-ham-comments.ipynb`.
- Công cụ chuẩn bị dữ liệu: `pipeline/step_*`, từ gốc repo chạy bằng `python -m ...`.
- Mô hình hiện đại tạm hoãn theo yêu cầu; chưa phân công, chưa ghi là hoàn thành.
- Link group, thời gian/công cụ crawl và lịch sử nhãn v7 để trống/chưa xác nhận; bổ sung nếu nhóm tìm được.
