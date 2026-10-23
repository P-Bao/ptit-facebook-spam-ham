# Phát hiện Spam/Ham — Bình luận Facebook PTIT

> **Bài tập lớn** — Môn: *Phân tích & Khai phá Dữ liệu Văn bản*  
> **Yêu cầu thực hiện:** Phân loại Spam/Ham trên bình luận của sinh viên PTIT tại các Facebook group.  
> **Phiên bản:** v2.0 (Đã tích hợp dữ liệu Phase 2 & cập nhật mô hình).

---

## 📌 1. Mô tả bài toán

Hệ thống phân loại tự động bình luận trong các **Facebook group của sinh viên PTIT** thành hai lớp:

| Nhãn | Ý nghĩa | Ví dụ đặc trưng |
|------|---------|-----------------|
| **ham** | Bình luận bình thường, chân thực, hợp lệ | Thảo luận môn học, hỏi lịch thi, xin tài liệu, trao đổi đời sống sinh viên... |
| **spam** | Bình luận rác, không liên quan, độc hại | Rao vặt, quảng cáo khóa học/dịch vụ ngoài trường, seeding link lạ, số điện thoại lừa đảo, cày cmt hàng loạt... |

---

## 🗂️ 2. Cấu trúc thư mục dự án

```
.
├── data/
│   ├── comments_labeled_export.csv       # Phase 1: Dữ liệu gán nhãn gốc (31,931 dòng)
│   ├── groups_data_phase2.json           # Phase 2: Dữ liệu crawl mới dạng JSON lồng nhau
│   ├── comments_phase2.csv               # Phase 2: Dữ liệu phẳng hoá sang định dạng chuẩn
│   ├── comments_phase2_cleaned.csv       # Phase 2: Dữ liệu sau khi làm sạch văn bản
│   ├── comments_labeled_phase2.csv       # Phase 2: Dữ liệu sau khi gán nhãn (4,197 dòng)
│   └── comments_labeled_merged.csv       # Dữ liệu cuối gộp Phase 1 + Phase 2 (36,128 dòng)
│
├── convert_phase2_json.py                # Bước 0: Làm phẳng JSON Phase 2 thành CSV chuẩn
├── clean_data_facebook_spamham.py        # Bước 1: Làm sạch HTML, URL, chuẩn hóa text tiếng Việt
├── upload_to_argilla.py                  # Bước 2a: Tải dữ liệu lên Argilla gán nhãn
├── ai_assisted_spam_ham.py               # Bước 2b: Gợi ý nhãn tự động bằng LLM (OpenRouter / GLM)
├── auto_accept_labels.py                 # Bước 2c: Tự động chốt nhãn có độ tin cậy cao (≥ 0.75)
├── download_labeled_data.py              # Bước 2d: Tải dữ liệu đã gán nhãn từ Argilla về CSV
├── merge_labeled_data.py                 # Bước 2e: Gộp Phase 1 + Phase 2, khử trùng, gán nguồn
│
├── spam-ham-comments-v2.ipynb            # Bước 3–5: Trích chọn đặc trưng, Train, Test & Đánh giá (v2)
├── spam-ham-comments.ipynb               # Notebook phiên bản cũ (chỉ Phase 1)
├── BAO_CAO_DU_AN.md                      # Báo cáo tổng quan chi tiết của dự án
└── README.md                             # Tài liệu hướng dẫn dự án
```

---

## 📊 3. Dữ liệu (Dataset)

Dữ liệu được thu thập từ các group Facebook sinh viên PTIT qua 2 giai đoạn (Phase 1 và Phase 2):

| Giai đoạn | Số lượng | Ham | Spam | Tỷ lệ Spam | Mô tả |
|-----------|----------|-----|------|------------|-------|
| **Phase 1** | 31,931 | 28,722 (89.95%) | 3,209 (10.05%) | 10.05% | Dữ liệu nền tảng ban đầu |
| **Phase 2 (mới)** | 4,197 | 3,414 (81.34%) | 783 (18.66%) | 18.66% | Crawl bổ sung, lọc trùng với Phase 1 |
| **TỔNG (Merged)** | **36,128** | **32,136 (88.95%)** | **3,992 (11.05%)** | **11.05%** | Tập dữ liệu hợp nhất huấn luyện v2 |

- Dataset Phase 1 lưu trữ trên Hugging Face: [p-storm/ptit-facebook-comments-spamham](https://huggingface.co/datasets/p-storm/ptit-facebook-comments-spamham).
- Cột dữ liệu chính: `id`, `label`, `cleaned_text`, `text`, `comment_type`, `reaction_count`, `source`.

---

## ⚙️ 4. Quy trình xử lý toàn diện (End-to-End Pipeline)

```
[Dữ liệu thô Facebook Group]
      │
      ├─────────────────────── Phase 1 (CSV)
      │
      └─────────────────────── Phase 2 (JSON)
                                  │
                                  ▼
[Bước 0] convert_phase2_json.py (Làm phẳng post → comments → replies)
      │
      ▼
[Bước 1] clean_data_facebook_spamham.py (Loại bỏ URL, emoji, HTML, chuẩn hóa ký tự)
      │
      ▼
[Bước 2a] upload_to_argilla.py (Đưa dữ liệu lên nền tảng Argilla)
      │
      ▼
[Bước 2b] ai_assisted_spam_ham.py (Gợi ý nhãn tự động với mô hình ngôn ngữ lớn GLM)
      │
      ▼
[Bước 2c] auto_accept_labels.py (Chốt nhãn confidence ≥ 0.75 & rà soát nhãn nghi vấn)
      │
      ▼
[Bước 2d] download_labeled_data.py (Xuất kết quả nhãn Phase 2)
      │
      ▼
[Bước 2e] merge_labeled_data.py (Gộp P1 + P2 → data/comments_labeled_merged.csv)
      │
      ▼
[Bước 3-5] spam-ham-comments-v2.ipynb
      ├── Trích chọn đặc trưng: BoW, TF-IDF, Word2Vec (underthesea)
      ├── Phân chia dữ liệu: 80% Train (28,901 mẫu) / 20% Test (7,227 mẫu - Stratified)
      ├── Huấn luyện mô hình: MultinomialNB, GaussianNB, LinearSVC
      └── Đánh giá, so sánh trước/sau tích hợp & cải tiến xử lý mất cân bằng
```

---

## 🧪 5. Kết quả thực nghiệm (v2 sau khi tích hợp)

Đánh giá trên tập kiểm tra gộp (**Test Set: 7,227 mẫu**, gồm 6,428 ham và 799 spam):

### 5.1. Bảng so sánh 6 tổ hợp mô hình

| Phương pháp biểu diễn | Thuật toán phân loại | Accuracy | Precision (Spam) | Recall (Spam) | **F1-Score (Spam)** |
|----------------------|----------------------|:--------:|:----------------:|:-------------:|:-------------------:|
| **BoW** | Naive Bayes (MultinomialNB) | 92.21% | 68.10% | 55.57% | 61.20% |
| **BoW** | Support Vector Machine (LinearSVC) | 93.33% | 81.51% | 51.31% | 62.98% |
| **TF-IDF** | Naive Bayes (MultinomialNB) | 93.01% | **89.30%** | 41.80% | 56.95% |
| **TF-IDF** | **Support Vector Machine (LinearSVC)** | **93.48%** | 84.60% | 50.19% | **63.00%** |
| **Word2Vec** | Naive Bayes (GaussianNB) | 84.18% | 36.65% | **59.07%** | 45.23% |
| **Word2Vec** | Support Vector Machine (LinearSVC) | 93.22% | 85.52% | 46.56% | 60.29% |

> 🏆 **Mô hình tốt nhất:** **TF-IDF + LinearSVC** đạt hiệu quả cân bằng và tối ưu nhất toàn diện với **Accuracy 93.48%** và **F1-Score (Spam) 63.00%** (Precision 84.60%).

---

### 5.2. So sánh trước vs sau khi tích hợp Phase 2

Đánh giá trên cùng tập kiểm tra gộp (Test gộp):

| Tổ hợp mô hình | F1-Score (Trước - Train P1) | **F1-Score (Sau - Train P1+P2)** | Recall (Trước) | **Recall (Sau)** |
|----------------|:---------------------------:|:--------------------------------:|:--------------:|:----------------:|
| BoW + Naive Bayes | 60.65% | **61.20%** | 52.94% | **55.57%** |
| BoW + SVM | 54.29% | **62.98%** | 41.55% | **51.31%** |
| TF-IDF + Naive Bayes | 39.96% | **56.95%** | 25.91% | **41.80%** |
| **TF-IDF + SVM** | 50.77% | **63.00%** (+12.23%) | 37.30% | **50.19%** (+12.89%) |
| Word2Vec + Naive Bayes | 33.55% | **43.95%** | 48.19% | **57.32%** |
| Word2Vec + SVM | 42.34% | **60.51%** (+18.17%) | 28.54% | **47.18%** (+18.64%) |

> 📈 **Kết luận:** Việc bổ sung 4,197 mẫu dữ liệu mới ở Phase 2 giúp khả năng nhận diện spam (Recall) tăng mạnh từ **~37% lên trên 50%**, F1-score của LinearSVC tăng vượt trội từ **50.77% lên 63.00%**, chứng minh tính hiệu quả rõ rệt của quy trình crawl & tích hợp thêm dữ liệu.

---

### 5.3. Tận dụng Metadata & Xử lý mất cân bằng lớp

- **Kết hợp Metadata:** Kết hợp đặc trưng văn bản TF-IDF với `comment_type` và `reaction_count` giúp tăng nhẹ F1 từ **63.00% lên 63.31%**.
- **Điều chỉnh trọng số & ngưỡng:** Cấu hình `class_weight='balanced'` trên LinearSVC giúp đẩy Recall của Spam lên tới **83.60%** (rất hữu ích trong các hệ thống ưu tiên bắt triệt để spam và chấp nhận kiểm duyệt phụ).

---

## 🚀 6. Hướng dẫn chạy

### 6.1. Chạy trên máy cá nhân (Local)

Môi trường ảo `.venv` trong dự án đã cài đặt sẵn đầy đủ các thư viện (`underthesea`, `gensim`, `scikit-learn`, `pandas`, `matplotlib`, `nbconvert`, `ipykernel`...).

**Cách 1: Chạy trực quan trong VS Code / Cursor / IDE**
1. Mở file [spam-ham-comments-v2.ipynb](file:///Users/cuongvuthanh/Documents/ptit-facebook-spam-ham/spam-ham-comments-v2.ipynb).
2. Góc trên bên phải, chọn Kernel: **Python Environments...** → chọn `.venv` của dự án.
3. Bấm **Run All**.

**Cách 2: Chạy tự động qua dòng lệnh (Headless CLI)**
```bash
# Thực thi toàn bộ notebook và lưu kết quả
./.venv/bin/jupyter nbconvert --to notebook --execute spam-ham-comments-v2.ipynb --output spam-ham-comments-v2-executed.ipynb
```

---

### 6.2. Chạy trên Kaggle (Tùy chọn)

1. Tạo Notebook mới trên Kaggle: **File → Import Notebook** → Chọn `spam-ham-comments-v2.ipynb`.
2. Tải dữ liệu: **Add Input → Upload** file `data/comments_labeled_merged.csv` (hoặc cấu hình `HF_TOKEN` trong *Add-ons → Secrets*).
3. Cấu hình phần cứng: Accelerator = **None (CPU)**, Internet = **On**.
4. Chọn **Run All**.

---

## 📚 7. Tài liệu tham khảo

- Crawler Facebook: [facebook_post_comment_scraper](https://github.com/mohdtalal3/facebook_post_comment_scraper)
- Công cụ tách từ tiếng Việt: [underthesea](https://github.com/undertheseanlp/underthesea)
- Nền tảng quản lý gán nhãn: [Argilla](https://argilla.io/)
- Thư viện Học máy & NLP: [scikit-learn](https://scikit-learn.org/), [gensim Word2Vec](https://radimrehurek.com/gensim/)