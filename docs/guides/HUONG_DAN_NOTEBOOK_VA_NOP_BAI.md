# Cách chạy, trình bày và chuẩn bị báo cáo

Theo thông tin người dùng xác nhận: nộp báo cáo, code dùng khi trình bày. Hai
notebook là giao diện giải thích/thực nghiệm trong repo, không phải file độc lập
nhúng lại mọi module. Chưa viết báo cáo học phần ở lượt tái cấu trúc.

## Hai luồng

1. `notebooks/01_thu_thap_va_gan_nhan.ipynb`: code crawler, JSON thật → bảng xem thử
   → clean → cache/export nhãn đã lưu. Phục vụ ý 1 và một phần ý 2. Run All không
   gọi Facebook/LLM/Argilla và không nhận CSV xem thử là v7.
2. `notebooks/02_thuc_nghiem_va_demo.ipynb`: v7 → tiền xử lý model → split/CV →
   BoW/TF-IDF/FastText, NB/SVM → đánh giá → demo. Không cần dữ liệu tham chiếu của
   notebook 01 hoặc chạy merge. Có heading/cell chờ cho ý 4 bổ sung.

## Môi trường và dữ liệu

Python 3.12, cài `requirements/notebook.txt`, chọn kernel đó và mở notebook trong
repo. Chỉ cần `collection.txt` nếu chạy crawler, `annotation.txt` cho công cụ nhãn;
`dev.txt` gồm các dependency để chạy tests. Không cần PyQt/Selenium cho notebook.

V7: `data/final/labeled_v7.csv`, đổi bằng `SPAMHAM_CSV`; thiếu local thì tải HF
theo revision/hash đã chốt. Không ghi token/cookie vào notebook.

Năm snapshot notebook 01 dùng:

| Folder | File lịch sử |
|---|---|
| `data/raw/` | `groups_data_phase2.json` |
| `data/intermediate/` | `comments_phase2.csv`, `comments_phase2_cleaned.csv`, `predictions_glm_cache_phase2.json` |
| `data/labeled/` | `comments_labeled_phase2.csv` |

File thiếu tải theo `results/source_audit/drive_manifest.json`; offline cần chuẩn
bị sẵn đúng hash. `SPAMHAM_SOURCE_DIR` nhận folder `data/` phân tầng hoặc bundle
flat có các tên file trên. Tên snapshot giữ nguyên để đối chiếu nguồn.

Notebook 01 làm phẳng 3 bài để giải thích; nhãn minh họa dùng đúng ID của bảng
clean/cache/export lịch sử. Không renumber bảng mới rồi áp cache cũ. Tái hiện đầy
đủ CSV lịch sử còn cần dữ liệu tham chiếu; dùng công cụ audit khi cần.

## Chạy và đầu ra

```powershell
.\.venv\Scripts\python.exe scripts/run_notebook.py --notebook 01 --work-dir .work/runs/collection --inplace
.\.venv\Scripts\python.exe scripts/run_notebook.py --notebook 02 --csv data/final/labeled_v7.csv --work-dir .work/runs/training --inplace
```

Runner dùng kernel mới, module repo, lưu hash code/module tại lúc bắt đầu. Chỉ
ghi notebook chính khi thành công; đây không phải kiểm tra notebook tự chứa.

- Notebook 01: `comments_preview.csv`, `cleaned_preview.csv`, `labeled_preview.csv`,
  `collection_demo_manifest.json`.
- Notebook 02: `spam_ham_ml_only.joblib`, `metrics.csv`, `bootstrap_ci.csv`,
  `run_manifest.json`, bảng/hình trong notebook.

Đầu ra ở `SPAMHAM_OUTPUT_DIR`; runner đặt tại `<work-dir>/artifacts/`. Dùng output
thật đã lưu để giải thích, chạy phần dự đoán mới để demo; không cần train mọi tổ
hợp ngay trước lớp. Kịch bản: `docs/presentation/LO_TRINH_TRINH_BAY.md`.

Phần nguồn trình bày một đợt có snapshot; phần thực nghiệm dùng v7 do nhóm cung
cấp, không nói chạy lại công cụ hiện tại tự tạo chính xác v7. Nguồn và giới hạn
phiên bản ở `docs/provenance/`. LLM gán nhãn không thay cho ý 4 bổ sung.
