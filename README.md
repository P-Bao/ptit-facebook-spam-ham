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
│   ├── comments_labeled_phase2.csv       # Phase 2: Dữ liệu sau khi gán nhãn (6,302 dòng)
│   └── comments_labeled_merged.csv       # Dữ liệu cuối gộp Phase 1 + Phase 2 (38,233 dòng)
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
| **Phase 2 (mới)** | 6,302 | 5,197 (82.47%) | 1,105 (17.53%) | 17.53% | Crawl bổ sung, lọc trùng với Phase 1 |
| **TỔNG (Merged)** | **38,233** | **33,919 (88.72%)** | **4,314 (11.28%)** | **11.28%** | Tập dữ liệu hợp nhất huấn luyện v2 |

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
      ├── Phân chia dữ liệu: 80% Train (30,585 mẫu) / 20% Test (7,648 mẫu - Stratified)
      ├── Huấn luyện mô hình: MultinomialNB, GaussianNB, LinearSVC
      └── Đánh giá, so sánh trước/sau tích hợp & cải tiến xử lý mất cân bằng
```

---

## 🧪 5. Kết quả thực nghiệm (v2 sau khi tích hợp)

Đánh giá trên tập kiểm tra gộp (**Test Set: 7,648 mẫu**, gồm 6,785 ham và 863 spam):

### 5.1. Bảng so sánh 6 tổ hợp mô hình

| Phương pháp biểu diễn | Thuật toán phân loại | Accuracy | Precision (Spam) | Recall (Spam) | **F1-Score (Spam)** |
|----------------------|----------------------|:--------:|:----------------:|:-------------:|:-------------------:|
| **BoW** | Naive Bayes (MultinomialNB) | 92.39% | 69.38% | 58.29% | 63.35% |
| **BoW** | **Support Vector Machine (LinearSVC)** | 93.45% | 82.21% | **53.53%** | **64.84%** |
| **TF-IDF** | Naive Bayes (MultinomialNB) | 93.06% | **89.34%** | 43.68% | 58.68% |
| **TF-IDF** | Support Vector Machine (LinearSVC) | **93.49%** | 84.50% | 51.80% | 64.22% |
| **Word2Vec** | Naive Bayes (GaussianNB) | 84.65% | 38.01% | 57.13% | 45.65% |
| **Word2Vec** | Support Vector Machine (LinearSVC) | 93.34% | 84.98% | 49.83% | 62.82% |

> 🏆 **Mô hình tốt nhất:** **BoW + LinearSVC** đạt F1-Score lớp Spam cao nhất (**64.84%**, Precision 82.21%, Recall 53.53%), sát nút là **TF-IDF + LinearSVC** đạt Accuracy cao nhất (**93.49%**) và F1-Score **64.22%** (Precision 84.50%).

---

### 5.2. So sánh trước vs sau khi tích hợp Phase 2

Đánh giá trên cùng tập kiểm tra gộp (Test gộp: 7,648 mẫu):

| Tổ hợp mô hình | F1-Score (Trước - Train P1) | **F1-Score (Sau - Train P1+P2)** | Recall (Trước) | **Recall (Sau)** |
|----------------|:---------------------------:|:--------------------------------:|:--------------:|:----------------:|
| BoW + Naive Bayes | 61.03% | **63.35%** (+2.32%) | 53.07% | **58.29%** (+5.22%) |
| **BoW + SVM** | 54.23% | **64.84%** (+10.61%) | 41.25% | **53.53%** (+12.28%) |
| TF-IDF + Naive Bayes | 39.35% | **58.68%** (+19.33%) | 25.38% | **43.68%** (+18.30%) |
| **TF-IDF + SVM** | 50.59% | **64.22%** (+13.63%) | 37.20% | **51.80%** (+14.60%) |
| Word2Vec + Naive Bayes | 33.25% | **45.88%** (+12.63%) | 47.51% | **57.36%** (+9.85%) |
| Word2Vec + SVM | 42.60% | **61.75%** (+19.15%) | 28.51% | **48.55%** (+20.04%) |

> 📈 **Kết luận:** Việc bổ sung 6,302 mẫu dữ liệu mới ở Phase 2 giúp khả năng nhận diện spam (Recall) tăng mạnh từ **~37% lên trên 52–58%**, F1-score của LinearSVC tăng vượt trội từ **50.59% lên 64.22%** (TF-IDF) và **54.23% lên 64.84%** (BoW). Đặc biệt, khi kiểm tra trên tập dữ liệu mới (Test P2), F1-Score tăng từ **40.92% lên 87.41%**, chứng minh mô hình đã học thành công các mẫu spam mới xuất hiện.

---

### 5.3. Tận dụng Metadata & Xử lý mất cân bằng lớp

- **Kết hợp Metadata:** Kết hợp đặc trưng văn bản TF-IDF với `comment_type` và `reaction_count` giúp tăng nhẹ Accuracy từ 93.49% lên 93.51% và F1 từ **64.22% lên 64.32%**.
- **Điều chỉnh trọng số (Class Weight):** Cấu hình `class_weight='balanced'` trên LinearSVC giúp đẩy Recall của Spam lên tới **84.59%** (bắt gần 85% spam trong thực tế).
- **Tối ưu ngưỡng (Threshold Tuning):** Tối ưu ngưỡng phân loại giúp **BoW + Naive Bayes (ngưỡng 0.751)** đạt **F1-Score cao nhất toàn bộ thực nghiệm (64.38%)** với Precision 78.96% và Recall 54.35%.

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