# Phát hiện Spam/Ham — Bình luận Facebook PTIT

> **Bài tập lớn** — Môn: *Phân tích & Khai phá Dữ liệu Văn bản*  
> **Yêu cầu thực hiện:** Phân loại Spam/Ham trên bình luận của sinh viên PTIT tại các Facebook group.  
> **Phiên bản:** v3.0 (Học máy thuần — không dùng quy tắc, dữ liệu nhãn v7).

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
├── spam_features.py                      # Bước 3: Transformer FastTextVectorizer (Skip-gram + Subword)
├── spam-ham-comments-v2.ipynb            # Bước 3–5: Trích chọn đặc trưng (BoW, TF-IDF, FastText), Train, Test & Đánh giá
├── spam-ham-comments.ipynb               # Bản đồng bộ notebook chính
└── README.md                             # Tài liệu hướng dẫn dự án
```

---

## 📊 3. Dữ liệu (Dataset)

Dữ liệu được thu thập từ các group Facebook sinh viên PTIT qua 2 giai đoạn (Phase 1 và Phase 2):

| Giai đoạn | Số lượng | Ham | Spam | Tỷ lệ Spam | Mô tả |
|-----------|----------|-----|------|------------|-------|
| **Phase 1** | 31,931 | 28,722 (89.95%) | 3,209 (10.05%) | 10.05% | Dữ liệu nền tảng ban đầu |
| **Phase 2 (mới)** | 6,302 | 5,197 (82.47%) | 1,105 (17.53%) | 17.53% | Crawl bổ sung, lọc trùng với Phase 1 |
| **TỔNG (Merged)** | **38,233** | **33,919 (88.72%)** | **4,314 (11.28%)** | **11.28%** | Tập dữ liệu hợp nhất |

- Dataset lưu trữ trên Hugging Face: [p-storm/ptit-facebook-comments-spamham](https://huggingface.co/datasets/p-storm/ptit-facebook-comments-spamham).
- Notebook huấn luyện dùng file nhãn mới nhất **`labeled_v7.csv`** (38,233 bình luận, nhãn đã rà soát lại): **ham 32,907 (86.1%) — spam 5,326 (13.9%)**.
- Cột dữ liệu chính: `id`, `label`, `cleaned_text`, `text`, `comment_type`, `reaction_count`.
- Dataset là **gated** — cần tài khoản HF được phê duyệt và token `HF_TOKEN` (biến môi trường hoặc Kaggle Secrets).

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
[Bước 3-5] spam-ham-comments-v2.ipynb (và spam-ham-comments.ipynb)
      ├── Tiền xử lý: ẩn danh tên (user_name), chuẩn hóa Unicode, gom emoji (emoji_token), tách từ underthesea
      ├── Trích chọn đặc trưng: BoW, TF-IDF (n-gram 1–2), Word Embeddings (FastText Skip-gram 100 chiều, subword 3–5)
      ├── Phân chia dữ liệu: 80% Train (30,753) / 20% Test (7,480) — theo hash văn bản, chống rò rỉ trùng lặp
      ├── Huấn luyện: MultinomialNB / GaussianNB, LinearSVC (GridSearchCV 5 fold, tối ưu F1 lớp spam)
      └── Đánh giá: Accuracy/Precision/Recall/F1, PR-AUC, bootstrap CI 95%, kiểm chứng OOV/ngữ nghĩa, phân tích lỗi
```

---

## 🧪 5. Kết quả thực nghiệm (nhãn v7 — 3 đặc trưng × 2 thuật toán)

Toàn bộ lựa chọn (tiền xử lý, siêu tham số, mô hình) đều dựa trên **cross-validation trong tập train**; tập test (7,480 mẫu: 6,524 ham, 956 spam) chỉ dùng để đánh giá lần cuối.

### 5.1. Bảng so sánh 6 tổ hợp mô hình (tiền xử lý chính: giữ stop words)

| Đặc trưng | Thuật toán | Accuracy | Precision (Spam) | Recall (Spam) | **F1 (Spam)** | PR-AUC |
|-----------|------------|:--------:|:----------------:|:-------------:|:-------------:|:------:|
| BoW | Naive Bayes (MultinomialNB) | 91.47% | 72.46% | 53.66% | 61.66% | 0.762 |
| BoW | SVM (LinearSVC, balanced) | 90.55% | 59.48% | 81.69% | 68.84% | 0.782 |
| TF-IDF | Naive Bayes (MultinomialNB) | 93.76% | **86.99%** | 60.15% | 71.12% | 0.841 |
| **TF-IDF** | **SVM (LinearSVC, balanced)** | **94.40%** | 74.84% | 84.62% | **79.43%** | **0.863** |
| Word Embeddings (FastText) | Naive Bayes (GaussianNB) | 86.76% | 48.72% | 67.68% | 56.65% | 0.462 |
| Word Embeddings (FastText) | SVM (LinearSVC, balanced) | 89.73% | 56.44% | **86.19%** | 68.21% | 0.812 |

> 🏆 **Mô hình tốt nhất:** **TF-IDF + LinearSVC** (C = 0.1, n-gram 1–2) — F1 79.43%, khoảng tin cậy bootstrap 95%: [77.4%, 81.5%], PR-AUC 0.863. Ba tổ hợp dẫn đầu đại diện cho ba ưu tiên vận hành khác nhau:
> - **TF-IDF + Naive Bayes:** ưu tiên **Precision** (86.99% — rất ít báo nhầm).
> - **TF-IDF + SVM:** cân bằng tốt nhất giữa Precision và Recall (**F1 = 79.43%**, **PR-AUC = 0.863**).
> - **FastText + SVM:** đạt **Recall cao nhất (86.19%**, bắt được 824/956 spam) và **PR-AUC = 0.812** (vượt BoW + SVM) nhờ biểu diễn ngữ nghĩa liên tục và n-gram ký tự (subword) xử lý tốt teencode/từ viết sai (`inboxxx` → `inbox`, `inb0x` → `inb`), bắt thêm được **50 mẫu spam hội thoại** mà TF-IDF bỏ sót.

### 5.2. Thí nghiệm đối chứng loại stop words (Ablation Study)

Chạy song song 2 biến thể tiền xử lý × 6 tổ hợp mô hình (12 thí nghiệm). Cross-validation chọn **giữ stop words** (F1-CV trung bình 6 tổ hợp đạt 0.676 so với 0.652 của biến thể loại):

| Đặc trưng | Thuật toán | F1 (giữ stop words) | F1 (loại stop words) | Δ F1 (loại − giữ) |
|-----------|------------|:-------------------:|:--------------------:|:-----------------:|
| BoW | Naive Bayes | 61.66% | 59.75% | −1.91% |
| BoW | SVM | 68.84% | 68.35% | −0.49% |
| TF-IDF | Naive Bayes | 71.12% | 70.07% | −1.05% |
| TF-IDF | SVM | **79.43%** | 77.67% | −1.76% |
| FastText | Naive Bayes | 56.65% | 54.85% | −1.81% |
| FastText | SVM | 68.21% | 67.19% | −1.02% |

> 💡 Trong bình luận mạng xã hội, các hư từ (`ạ`, `à`, `mà`, `nhé`...) là đặc trưng phong cách của bình luận ham; loại bỏ stop words làm **giảm** F1 ở cả 6 tổ hợp (−0.49 đến −1.91 điểm phần trăm).

### 5.3. Phân tích lỗi & Đối chiếu TF-IDF (Từ khóa) vs FastText (Ngữ nghĩa)

- **False Positives (ham bị báo spam):** phần lớn là bình luận chỉ gồm tên người/username hoặc rất ngắn (vd. `Nguyễn Văn An`) — nhiều spam trong dữ liệu được gán nhãn theo **hình thức** nên mô hình học được tín hiệu "chỉ có tên".
- **False Negatives (spam bị bỏ sót bởi TF-IDF nhưng FastText bắt được — 50 mẫu):** spam viết như hội thoại ngắn gọn hoặc biến thể teencode (`Cách truờng 400m ib`, `Ib tớ ạ, tớ làm 10k1 unit ạ`, `Nguyễn Xuân Bảo Nam 100k mua k b`). Nhờ học vector ngữ nghĩa và subword n-gram, FastText nhận diện được ngữ cảnh chào mời dù không có `urltoken`/`phonetoken`.
- **Hạn chế của FastText pooling:** phép lấy trung bình vector các từ trong câu làm "pha loãng" tín hiệu khi câu dài chỉ chứa 1 từ khóa spam đơn lẻ, khiến Precision của FastText + SVM (56.44%) thấp hơn TF-IDF + SVM (74.84%).

Mô hình tốt nhất được lưu vào `spam_ham_ml_only.joblib` kèm hàm dự đoán ngay trong notebook.

---

## 🛠️ 6. Chi tiết các cải tiến mã nguồn & kỹ thuật vừa thực hiện

Trong đợt rà soát và nâng cấp toàn diện vừa qua, hệ thống đã được tinh chỉnh chuyên sâu trên cả 4 phân hệ chính nhằm đảm bảo tính toàn vẹn dữ liệu, tính tái lập thực nghiệm và đáp ứng xuất sắc các tiêu chí học phần:

### 6.1. Phân hệ Tiền xử lý & Làm sạch dữ liệu (`clean_data_facebook_spamham.py`)
- **Đo lường chính xác URL & Số điện thoại:** Di chuyển kiểm tra Regex `_RE_URL` và `_RE_PHONE_VN` lên trước bước biến đổi văn bản để thống kê trực tiếp trên văn bản gốc. Trước đây, thống kê trên văn bản đã clean dẫn đến việc đếm sai lệch/bỏ sót số lượng bình luận có chứa URL hoặc SĐT.
- **Loại bỏ bình luận rỗng và ký tự nhiễu:** Bổ sung điều kiện kiểm tra `re.search(r"[0-9a-zA-ZÀ-ỹ]", clean_text)` để tự động loại bỏ các bình luận sau khi tiền xử lý chỉ còn lại dấu chấm, dấu câu, icon lỗi hoặc khoảng trắng rỗng, loại trừ hoàn toàn các mẫu nhiễu khỏi tập dữ liệu huấn luyện.
- **Minh bạch hóa nhật ký thống kê:** Tách bạch rõ ràng số lượng dòng thô, số dòng trùng lặp loại bỏ (`dedup`), số dòng rỗng bị loại bỏ (`skipped_empty`) và số lượng mẫu hợp lệ thực tế được lưu vào CSV.

### 6.2. Phân hệ Gán nhãn & Quản trị dữ liệu Argilla
- **Phân tách ID dữ liệu nhiều giai đoạn (`upload_to_argilla.py`):** Bổ sung tham số CLI `--id-prefix` (mặc định: `cmt_`, tùy biến `p2_cmt_` cho Phase 2), ngăn ngừa xung đột đè nhãn bản ghi khi tải dữ liệu bổ sung lên máy chủ Argilla.
- **Cơ chế an toàn khi gán nhãn bằng LLM (`ai_assisted_spam_ham.py`):** Thay đổi giá trị confidence mặc định từ `0.95` thành `0.5` khi mô hình AI (OpenRouter / GLM) không trả về trường độ tin cậy. Điều này ngăn việc tự động duyệt nhãn thiếu căn cứ, ép bản ghi vào luồng rà soát thủ công an toàn (Human-in-the-loop).
- **Đồng bộ hóa tham số CLI (`auto_accept_labels.py`):** Bổ sung alias `--csv` song song với `--csv-in`, giúp lệnh gọi CLI hoàn toàn tương thích và đồng nhất với tài liệu hướng dẫn.
- **Sửa lỗi trích xuất ID & Regex (`download_labeled_data.py`):** Cập nhật Regex ánh xạ sang `r"^(?:p2_)?cmt_"`. Khắc phục triệt để lỗi parse sai ID thành `NaN` khiến dữ liệu Phase 2 bị mất mát khi xuất nhãn từ Argilla và nối metadata với CSV gốc.
- **Đảm bảo tính bất biến khi gộp tập nhãn (`merge_labeled_data.py`):** Áp dụng logic kiểm tra `apply(lambda x: x if x.startswith("p2_") else f"p2_{x}")`, tránh triệt để lỗi nhân đôi tiền tố `p2_p2_` khi script gộp dữ liệu được thực thi lặp lại nhiều lần.

---

## 🚀 7. Hướng dẫn chạy

### 7.1. Chạy trên máy cá nhân (Local)

Môi trường ảo `.venv` trong dự án đã cài đặt sẵn đầy đủ các thư viện (`underthesea`, `huggingface_hub`, `scikit-learn`, `pandas`, `matplotlib`, `joblib`, `nbconvert`, `ipykernel`...).

Notebook tải dataset từ Hugging Face — cần biến môi trường `HF_TOKEN` (hoặc `SPAMHAM_CSV` trỏ tới file CSV đã tải sẵn).

**Cách 1: Chạy trực quan trong VS Code / Cursor / IDE**
1. Mở file `spam-ham-comments.ipynb`.
2. Góc trên bên phải, chọn Kernel: **Python Environments...** → chọn `.venv` của dự án.
3. Bấm **Run All**.

**Cách 2: Chạy tự động qua dòng lệnh (Headless CLI)**
```bash
# Thực thi toàn bộ notebook và lưu kết quả
./.venv/bin/jupyter nbconvert --to notebook --execute spam-ham-comments.ipynb --output spam-ham-comments-executed.ipynb
```

---

### 7.2. Chạy trên Kaggle (Tùy chọn)

1. Tạo Notebook mới trên Kaggle: **File → Import Notebook** → Chọn `spam-ham-comments.ipynb`.
2. Cấu hình `HF_TOKEN` trong *Add-ons → Secrets* — notebook tự tải `labeled_v7.csv` từ Hugging Face.
3. Cấu hình phần cứng: Accelerator = **None (CPU)**, Internet = **On**.
4. Chọn **Run All**.

---

## 📚 8. Tài liệu tham khảo

- Crawler Facebook: [facebook_post_comment_scraper](https://github.com/mohdtalal3/facebook_post_comment_scraper)
- Công cụ tách từ tiếng Việt: [underthesea](https://github.com/undertheseanlp/underthesea)
- Nền tảng quản lý gán nhãn: [Argilla](https://argilla.io/)
- Thư viện Học máy & NLP: [scikit-learn](https://scikit-learn.org/), [joblib](https://joblib.readthedocs.io/)