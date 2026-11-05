# Phân loại Spam/Ham bình luận Facebook PTIT

Nhóm Bảo, Cương, Phi, Phương. Nộp báo cáo; notebook/code dùng để trình bày.
Thực nghiệm BoW/TF-IDF/FastText với NB/SVM trên v7. Ý 4 bổ sung có mục chờ.

## Mở theo thứ tự

1. [01 — Thu thập và gán nhãn](notebooks/01_thu_thap_va_gan_nhan.ipynb): snapshot thật,
   code crawler, JSON → bảng → clean → cache/export nhãn. Không gọi Facebook/LLM/Argilla.
2. [02 — Thực nghiệm và demo](notebooks/02_thuc_nghiem_va_demo.ipynb): v7 → tiền xử lý,
   chia tập → đặc trưng/NB/SVM → đánh giá → dự đoán mới.
3. [Lộ trình mở code](docs/presentation/LO_TRINH_TRINH_BAY.md).
4. [Hướng dẫn chạy và dữ liệu](docs/guides/HUONG_DAN_NOTEBOOK_VA_NOP_BAI.md).
5. [Thay đổi, nguồn file cũ → mới và kiểm chứng](docs/changes/2026-11-05_TAI_CAU_TRUC_VA_DONG_BO.md).

## Cấu trúc

```text
external/facebook_scraper/     Bộ thư viện bàn giao, giữ mã nền
pipeline/
  collect/                    Runner PTIT, JSON → CSV, đọc/tải snapshot
  prepare/                    Clean trước label; preprocess cho model
  label/                      Upload, AI suggestion, chốt/export nhãn
  features/                   FastTextVectorizer
  modeling/                   NB/SVM, CV, sampling/ngưỡng, đánh giá
  inference/                  Dự đoán raw comment bằng model đã lưu
  modern/                     Giữ chỗ ý 4 bổ sung
notebooks/                    Hai notebook import module repo
tools/dataset_versions/        Gộp/đối soát, không cần để train v7
data/{raw,intermediate,labeled,final}/  Dữ liệu local, không đưa vào Git
docs/                         Đề, phân công, hướng dẫn, nguồn, nhật ký, báo cáo
requirements/                 Dependency theo nhu cầu
scripts/run_notebook.py        Chạy notebook trong kernel/output riêng
tests/                        Kiểm tra hồi quy
results/                      Metrics, manifest, nguồn và lịch sử kiểm chứng
```

Tên folder theo chức năng, không phải số ý của đề. Clean trước label chuẩn bị bảng;
preprocess cho model bổ sung tách từ và đối chứng stop words. Giữ `text` gốc.

## Chạy notebook

Python 3.12, từ gốc repo:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements/notebook.txt
```

Mở notebook trong repo, chọn kernel `.venv`, Restart Kernel → Run All. Notebook
import `pipeline/`; không còn là `.ipynb` tự chứa sinh lại mọi module.

- Notebook 01: tìm năm snapshot cần dùng trong `data/`, tải Drive theo manifest/hash
  nếu thiếu. Offline cần chuẩn bị sẵn; `SPAMHAM_SOURCE_DIR` nhận folder phân tầng hoặc flat.
- Notebook 02: dùng `data/final/labeled_v7.csv`, hoặc tải revision/hash đã chốt từ
  [HF của nhóm](https://huggingface.co/datasets/p-storm/ptit-facebook-comments-spamham).
  Có thể cấu hình `SPAMHAM_CSV`. Không cần Phase 1/merge/Argilla.

```powershell
.\.venv\Scripts\python.exe scripts/run_notebook.py --notebook 01 --work-dir .work/runs/collection --inplace
.\.venv\Scripts\python.exe scripts/run_notebook.py --notebook 02 --csv data/final/labeled_v7.csv --work-dir .work/runs/training --inplace
```

Runner dùng đúng interpreter, seed/hash seed đã chốt, module repo và kernel mới.
`--inplace` chỉ ghi output về notebook chính khi mọi cell thành công.

## Công cụ chuẩn bị dữ liệu khi cần

Cài `requirements/collection.txt` nếu chạy crawler; `annotation.txt` nếu dùng công
cụ nhãn. GUI có dependency riêng trong bộ nền. Từ gốc repo dùng `python -m`:

```powershell
python -m pipeline.collect.scrape_ptit_groups --help
python -m pipeline.collect.json_to_csv --help
python -m pipeline.prepare.clean_for_labeling --help
python -m pipeline.label.upload_to_argilla --help
python -m pipeline.label.ai_assisted_spam_ham --help
python -m pipeline.label.auto_accept_labels --help
python -m pipeline.label.download_labeled_data --help
python -m tools.dataset_versions.merge_labeled_data --help
python -m tools.dataset_versions.audit_source --help
```

Luồng mặc định mới: `data/raw/posts.json` → `data/intermediate/comments.csv` →
`data/intermediate/comments_cleaned.csv` → công cụ label → `data/labeled/`.
Với tập mới chưa có tham chiếu, converter có `--keep-phase1-duplicates`. Snapshot/cache
cũ phải giữ đúng ID/thứ tự; cleaner có `--keep-punctuation-only` để tái hiện lịch sử.

## Kiểm tra và nguồn

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements/dev.txt
.\.venv\Scripts\python.exe -m pytest --basetemp .work/pytest -q
```

[Kiểm chứng crawler/Drive](docs/provenance/CRAWLER_VA_DU_LIEU_DRIVE.md),
[đối chiếu Kaggle/HF](docs/provenance/KAGGLE_VA_NGUON_DU_LIEU.md).
V7 có 38.233 mẫu (32.907 ham/5.326 spam). Raw đợt bổ sung có 2.170 bài từ 5 group;
không suy thành toàn bộ dữ liệu do một lượt crawler tạo. Bản gộp Drive/v7 khác
1.366 nhãn, chưa có lịch sử chỉnh sửa đầy đủ; không tự relabel theo điểm test.
