# Phát hiện Spam/Ham — Bình luận Facebook PTIT

> **Bài tập lớn** — Môn: *Phân tích & Khai phá Dữ liệu Văn bản*  
> **Yêu cầu thực hiện:** 2b — Phân loại Spam/Ham trên bình luận của sinh viên PTIT tại các Facebook group.

---

## 📌 Mô tả bài toán

Xây dựng hệ thống phân loại tự động bình luận trong các **Facebook group của sinh viên PTIT** thành hai nhóm:

| Nhãn | Ý nghĩa |
|------|---------|
| **ham** | Bình luận bình thường, chân thực (hỏi đáp, chia sẻ kinh nghiệm, thảo luận học tập) |
| **spam** | Bình luận rác (quảng cáo, rao vặt, chào hàng khóa học/sản phẩm, chứa link lạ hoặc số điện thoại đáng ngờ, copy-paste hàng loạt, nội dung lừa đảo/không liên quan) |

---

## 🗂️ Cấu trúc dự án

```
.
├── clean_data_facebook_spamham.py   # Bước 1b: Tiền xử lý & làm sạch dữ liệu thô
├── upload_to_argilla.py             # Bước 2a: Tải dữ liệu đã làm sạch lên Argilla để gán nhãn
├── ai_assisted_spam_ham.py          # Bước 2b: Gợi ý nhãn tự động bằng AI (OpenRouter/GLM)
├── auto_accept_labels.py            # Bước 2c: Tự động chấp nhận nhãn độ tin cậy cao
├── download_labeled_data.py         # Bước 2d: Tải dữ liệu đã gán nhãn từ Argilla về CSV
├── spam-ham-comments.ipynb          # Bước 3–5: Trích chọn đặc trưng, huấn luyện & đánh giá mô hình
└── README.md
```

> **Lưu ý:** Thư mục `data/` **không** được commit vào repo. Xem phần [Dữ liệu](#-dữ-liệu) để biết cách tải về.

---

## 🕷️ Thu thập dữ liệu (Crawl)

Dữ liệu bình luận thô được thu thập từ các **Facebook group của sinh viên PTIT** bằng công cụ:

> **[mohdtalal3/facebook\_post\_comment\_scraper](https://github.com/mohdtalal3/facebook_post_comment_scraper)**

Công cụ này sử dụng Selenium để đăng nhập Facebook, duyệt qua các bài đăng trong group và trích xuất bình luận + reply cùng thông tin đính kèm (loại bình luận, số reaction…).

---

## 📦 Dữ liệu

Dataset đã được gán nhãn (spam/ham) được lưu công khai tại:

> 🤗 **[p-storm/ptit-facebook-comments-spamham](https://huggingface.co/datasets/p-storm/ptit-facebook-comments-spamham)**

| Thông tin | Giá trị |
|-----------|---------|
| Tổng số bình luận | ~31 931 |
| ham | ~28 722 (≈ 90%) |
| spam | ~3 209 (≈ 10%) |
| Cột chính | `id`, `label`, `cleaned_text`, `text`, `comment_type`, `reaction_count` |

> ⚠️ Dataset là **gated** — bạn cần tài khoản Hugging Face và được phê duyệt truy cập trước khi tải về.

### Cách tải dataset (dùng trong notebook)

```python
from huggingface_hub import hf_hub_download

csv_path = hf_hub_download(
    repo_id="p-storm/ptit-facebook-comments-spamham",
    filename="comments_labeled_export.csv",
    repo_type="dataset",
    token="<HF_TOKEN_CỦA_BẠN>",
)
```

---

## 🔄 Quy trình xử lý & gán nhãn

```
Crawl thô (facebook_post_comment_scraper)
        │
        ▼
clean_data_facebook_spamham.py   ← Làm sạch HTML, chuẩn hóa văn bản
        │
        ▼
upload_to_argilla.py             ← Upload lên Argilla (self-hosted, Docker)
        │
        ▼
ai_assisted_spam_ham.py          ← Gợi ý nhãn bằng LLM (OpenRouter)
        │
        ▼
auto_accept_labels.py            ← Tự động chấp nhận nhãn có độ tin cậy cao
        │
        ▼
download_labeled_data.py         ← Export CSV đã gán nhãn
        │
        ▼
spam-ham-comments.ipynb          ← Trích chọn đặc trưng + huấn luyện mô hình
```

---

## 🧪 Phương pháp & Kết quả

### Bước 2 — Tiền xử lý dữ liệu

- Loại bỏ HTML entities, URL, ký tự đặc biệt.
- Chuẩn hóa Unicode (dấu tiếng Việt).
- Tách từ tiếng Việt bằng **[underthesea](https://github.com/undertheseanlp/underthesea)**.

### Bước 3 — Trích chọn đặc trưng

| Phương pháp | Mô tả |
|-------------|-------|
| **BoW** (Bag-of-Words) | `CountVectorizer`, max 5 000 từ |
| **TF-IDF** | `TfidfVectorizer`, max 5 000 từ |
| **Word2Vec** | Huấn luyện trên tập train, biểu diễn câu bằng vector trung bình |

### Bước 4 — Mô hình phân loại

- **Naive Bayes** — `MultinomialNB` (BoW/TF-IDF), `GaussianNB` (Word2Vec)
- **SVM** — `LinearSVC`

Chia tập: **80% train / 20% test** (stratified theo nhãn).

### Bước 5 — Kết quả đánh giá

| Đặc trưng | Thuật toán | Accuracy | Precision (spam) | Recall (spam) | F1 (spam) |
|-----------|-----------|----------|-----------------|---------------|-----------|
| BoW | Naive Bayes | 92.23% | 66.67% | 45.48% | 54.07% |
| BoW | SVM | 92.86% | 77.03% | 41.28% | 53.75% |
| TF-IDF | Naive Bayes | 92.48% | 87.50% | 29.44% | 44.06% |
| **TF-IDF** | **SVM** | **93.06%** | **80.99%** | **40.50%** | **53.99%** |
| Word2Vec | Naive Bayes | 79.74% | 24.73% | 49.69% | 33.02% |
| Word2Vec | SVM | 92.78% | 86.06% | 33.64% | 48.38% |

> 📌 **Nhận xét:** TF-IDF + SVM đạt accuracy cao nhất (93.06%). Do tập dữ liệu mất cân bằng (~90% ham), các mô hình có xu hướng nhận diện tốt ham nhưng còn hạn chế ở recall của spam.

---

## ⚙️ Cài đặt

```bash
pip install argilla pandas tqdm openai gensim underthesea scikit-learn matplotlib huggingface_hub
```

### Chạy Argilla (Docker)

```bash
# Xem thư mục my-argilla/ (không commit vào repo)
docker compose up -d
```

---

## 📚 Tài liệu tham khảo

- Crawler: [facebook_post_comment_scraper](https://github.com/mohdtalal3/facebook_post_comment_scraper)
- Dataset: [p-storm/ptit-facebook-comments-spamham](https://huggingface.co/datasets/p-storm/ptit-facebook-comments-spamham)
- Tách từ tiếng Việt: [underthesea](https://github.com/undertheseanlp/underthesea)
- Gán nhãn: [Argilla](https://argilla.io/)
- scikit-learn: [sklearn.feature_extraction.text](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction)
- Gensim Word2Vec: [gensim.models.Word2Vec](https://radimrehurek.com/gensim/models/word2vec.html)
