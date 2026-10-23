# 🚀 KẾ HOẠCH TÍCH HỢP DATA MỚI (PHASE 2) — BÁM THEO PIPELINE CŨ

**Dự án:** Phân loại Spam / Ham bình luận Facebook sinh viên PTIT
**Nguyên tắc:** Data mới đi qua **đúng quy trình cũ** (xem `BAO_CAO_DU_AN.md` mục 3.1). Bước nào làm được bằng Python thì chạy local bằng script. **Chỉ dùng 1 notebook** (giống dự án cũ) cho phần trích đặc trưng + huấn luyện + đánh giá, chạy trên Kaggle.

---

## 1. Hiện trạng

| | Phase 1 (cũ) | Phase 2 (mới) |
|---|---|---|
| File | `data/comments_labeled_export.csv` (đã tải từ HF) | `data/groups_data_phase2.json` |
| Số mẫu | 31,931 (ham 28,722 / spam 3,209) | 13,833 (9,438 comment + 4,395 reply) |
| Định dạng | CSV 6 cột: `id, label, cleaned_text, text, comment_type, reaction_count` | JSON lồng nhau: post → comments → replies |
| Trạng thái | Đã clean + gán nhãn (GLM + Argilla) | Thô, **chưa gán nhãn** |

> [!WARNING]
> **Về file `integrate_phase2_data.py` đã chạy trước đó:** script này gán nhãn Phase 2 bằng một model TF‑IDF học từ chính Phase 1 (pseudo‑label). Cách này **không theo quy trình cũ** và gần như không thêm thông tin mới cho mô hình (nhãn chỉ lặp lại những gì Phase 1 đã biết). Ngoài ra bước khử trùng làm số spam **giảm** từ 3,209 → 3,017.
> 👉 Không dùng `data/merged_comments_labeled.csv` để huấn luyện/báo cáo. Script này sẽ được thay bằng các bước dưới đây.

---

## 2. Pipeline sau khi tích hợp

```
groups_data_phase2.json
        │
        ▼
Bước 0 (MỚI):  convert_phase2_json.py          ← JSON → CSV cùng định dạng crawl cũ
        │
        ▼
Bước 1:        clean_data_facebook_spamham.py  ← (giữ nguyên) làm sạch
        │
        ▼
Bước 2a:       upload_to_argilla.py            ← (giữ nguyên) dataset mới: comments_spamham_phase2
Bước 2b:       ai_assisted_spam_ham.py         ← (giữ nguyên) GLM gợi ý nhãn
Bước 2c:       auto_accept_labels.py           ← (giữ nguyên) chốt nhãn confidence ≥ 0.75
               + rà soát thủ công trên Argilla các bản ghi confidence thấp
Bước 2d:       download_labeled_data.py        ← (giữ nguyên) export CSV Phase 2
        │
        ▼
Bước 2e (MỚI): merge_labeled_data.py           ← gộp Phase 1 + Phase 2 → CSV cuối
        │
        ▼
Bước 3–5:      spam-ham-comments-v2.ipynb      ← (1 notebook, Kaggle) đặc trưng + train + đánh giá
```

**Tổng thay đổi:** 2 script Python mới + 1 notebook v2. Các script cũ **chỉ chạy lại với tham số khác**, không cần sửa (trừ 1 điểm nhỏ ở Bước 2a).

---

## 3. Chi tiết từng bước

### Bước 0 — `convert_phase2_json.py` (MỚI, Python local, ~1 giây)
- Đọc `data/groups_data_phase2.json`, làm phẳng post → comments → replies.
- Xuất `data/comments_phase2.csv` với **cùng các cột như file crawl cũ** (`data/comments.csv`), tối thiểu: `post_id, comment_type, text, reaction_count` (+ `group_name, created_at` nếu file cũ có).
- **Loại trước các bình luận đã có trong Phase 1** (so khớp theo `text` sau chuẩn hóa) → không tốn tiền gọi GLM cho mẫu trùng.
- Bỏ các dòng `text` rỗng.

> [!IMPORTANT]
> Cần xác nhận cấu trúc cột của file crawl cũ `data/comments.csv` (hiện không có trong máy). Nếu không còn file đó, dùng bộ cột tối thiểu ở trên — đủ cho `clean_data_facebook_spamham.py` và `upload_to_argilla.py`.

### Bước 1 — Làm sạch (giữ nguyên script)
```bash
python clean_data_facebook_spamham.py \
  --input data/comments_phase2.csv \
  --output data/comments_phase2_cleaned.csv
```

### Bước 2a–2d — Gán nhãn giống hệt Phase 1
```bash
# 2a. Upload lên Argilla (Docker my-argilla phải đang chạy) — dùng dataset MỚI, không ghi đè Phase 1
python upload_to_argilla.py --csv data/comments_phase2_cleaned.csv --dataset comments_spamham_phase2

# 2b. GLM gợi ý nhãn (OpenRouter), cache riêng cho Phase 2
python ai_assisted_spam_ham.py --csv data/comments_phase2_cleaned.csv \
  --dataset comments_spamham_phase2 --cache data/predictions_glm_cache_phase2.json

# 2c. Chốt nhãn confidence ≥ 0.75, phần còn lại rà soát tay trên giao diện Argilla
python auto_accept_labels.py --threshold 0.75 --dataset comments_spamham_phase2 \
  --cache data/predictions_glm_cache_phase2.json --csv data/comments_phase2_cleaned.csv

# 2d. Export CSV đã gán nhãn
python download_labeled_data.py --dataset comments_spamham_phase2 \
  --output data/comments_labeled_phase2.csv
```
*(Tên tham số chính xác sẽ kiểm tra lại theo `--help` của từng script khi chạy.)*

- **Điểm cần sửa nhỏ ở 2a:** `upload_to_argilla.py` đặt id là `cmt_{idx}` → trùng id với Phase 1. Thêm tham số `--id-prefix` (ví dụ `p2_cmt_`) để id Phase 2 không đè Phase 1.
- **Chi phí/thời gian ước tính:** ~13.8k mẫu (ít hơn sau khi lọc trùng) ÷ 15 mẫu/lần gọi ≈ 900 lần gọi API, 4 luồng → vài chục phút.
- **Phương án không có Argilla (tùy chọn):** `ai_assisted_spam_ham.py` có chế độ chỉ lưu cache, và `auto_accept_labels.py` có chế độ chỉ xuất CSV → vẫn ra được file nhãn, nhưng **mất bước rà soát thủ công** (cần ghi rõ trong báo cáo).

### Bước 2e — `merge_labeled_data.py` (MỚI, Python local, ~1 giây)
- Input: `data/comments_labeled_export.csv` (Phase 1) + `data/comments_labeled_phase2.csv` (Phase 2).
- Giữ **đúng 6 cột cũ** + thêm cột `source` (`phase1` / `phase2`) để notebook so sánh được.
- Khử trùng **chỉ giữa 2 phase** theo `cleaned_text` (ưu tiên giữ bản Phase 1); **không** khử trùng bên trong Phase 1 để giữ nguyên số liệu cũ.
- Output: `data/comments_labeled_merged.csv` + in thống kê ham/spam theo từng phase.
- (Tùy chọn) upload thêm file này lên HF dataset `p-storm/ptit-facebook-comments-spamham` để notebook Kaggle tải về như cũ.

### Bước 3–5 — `spam-ham-comments-v2.ipynb` (1 notebook, chạy Kaggle)
Sao chép từ `spam-ham-comments.ipynb`, **giữ nguyên bố cục cũ**, chỉ thay đổi:

| Mục trong notebook cũ | Thay đổi ở v2 |
|---|---|
| 0. Cài thư viện | Giữ nguyên |
| 1. Tải dataset HF (Kaggle Secrets) | Đổi tên file sang `comments_labeled_merged.csv` (hoặc đọc từ Kaggle Dataset upload tay) |
| 2. Tách từ `underthesea` | Giữ nguyên |
| 3. Chia train/test | Chia stratified 80/20 như cũ; **test set Phase 1 giữ cố định** (cùng `random_state`) để so sánh công bằng |
| 3.1–3.3 BoW / TF‑IDF / Word2Vec | Giữ nguyên |
| 4. NB + SVM (6 tổ hợp) | Giữ nguyên |
| 5.1–5.3 Report / Confusion matrix / Biểu đồ | Giữ nguyên |
| Metadata | Giữ nguyên |
| **6. (MỚI) So sánh trước/sau tích hợp** | Bảng: train Phase 1 vs train Phase 1+2, đánh giá trên cùng test set |
| **7. (MỚI) Cải tiến nhẹ** (theo mục 6.6 báo cáo) | `class_weight='balanced'` cho SVM + tối ưu ngưỡng theo F1‑spam |
| Nhận xét | Cập nhật theo kết quả mới |

Thời gian chạy trên Kaggle: CPU là đủ (không cần GPU), ~5–10 phút.

---

## 4. Cách chạy notebook trên Kaggle
1. **Code → New Notebook → File → Import Notebook** → chọn `spam-ham-comments-v2.ipynb`.
2. **Add‑ons → Secrets**: thêm `HF_TOKEN` (giống notebook cũ). Nếu không dùng HF: **+ Add Input → Upload** file `comments_labeled_merged.csv`.
3. **Settings**: Accelerator = **None (CPU)**, Internet = **On** (để `pip install` và tải HF).
4. **Run All** → kết quả/biểu đồ nằm trong notebook; tải về qua **Save Version** hoặc tab **Output**.

---

## 5. Thứ tự thực hiện & sản phẩm

| # | Việc | Ai làm | Sản phẩm |
|---|---|---|---|
| 1 | Viết `convert_phase2_json.py`, `merge_labeled_data.py`, thêm `--id-prefix` cho `upload_to_argilla.py` | AI | 2 script mới + 1 sửa nhỏ |
| 2 | Chạy Bước 0 → Bước 1 | Local | `comments_phase2_cleaned.csv` |
| 3 | Bật Argilla, chạy Bước 2a–2d, rà soát tay | Bạn (local) | `comments_labeled_phase2.csv` |
| 4 | Chạy Bước 2e | Local | `comments_labeled_merged.csv` |
| 5 | Tạo `spam-ham-comments-v2.ipynb` | AI | Notebook (chưa chạy) |
| 6 | Upload & chạy trên Kaggle | Bạn | Kết quả mới |
| 7 | Cập nhật `BAO_CAO_DU_AN.md`, `README.md` | AI | Báo cáo mục 2.2, 5, 6 |
| 8 | Xóa/để riêng `integrate_phase2_data.py` và `merged_comments_labeled.csv` | AI | Repo gọn |
