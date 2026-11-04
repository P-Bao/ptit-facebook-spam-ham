# Spam/Ham — Bình luận Facebook PTIT

- [Nhật ký sửa và kiểm chứng](docs/changes/2026-11-04_CAI_TIEN_VA_KIEM_CHUNG.md)
- [Đối chiếu Kaggle và nguồn dữ liệu](docs/references/KAGGLE_VA_NGUON_DU_LIEU.md)
- [Kiểm chứng crawler và dữ liệu Drive](docs/references/CRAWLER_VA_DU_LIEU_DRIVE.md)
- [Cách đọc notebook, phân biệt pipeline, nộp/demo](docs/guides/HUONG_DAN_NOTEBOOK_VA_NOP_BAI.md)
- [Thay đổi bổ sung nguồn thu thập](docs/changes/2026-11-04_BO_SUNG_NGUON_THU_THAP.md)

Notebook chính: **[notebooks/spam-ham-comments.ipynb](notebooks/spam-ham-comments.ipynb)**.

Phân loại `ham` (thảo luận hợp lệ) và `spam` (rác/quảng cáo/chào mời). Thực nghiệm
BoW/TF-IDF/FastText với Naive Bayes/SVM, đối chứng stop words và mất cân bằng.
**Mô hình hiện đại của ý 4 bổ sung đang để mục chờ, chưa thực hiện theo yêu cầu hiện tại.**

## Cấu trúc

```text
notebooks/spam-ham-comments.ipynb   # File code nộp chính, tự sinh module hỗ trợ
facebook_post_comment_scraper_code/
  facebook_post_comment_scraper/   # Mã crawler nguyên bản thành viên bàn giao
pipeline/
  step_01_collect/                 # JSON → CSV và đối soát chuỗi nguồn tới v7
  step_02_preprocess/              # Cleaner và tiền xử lý chung train/dự đoán
  step_03_label/                   # Upload/AI/duyệt/export Argilla
  step_04_merge/                   # Gộp Phase 1 + Phase 2
  step_05_features/                # FastText
  step_06_train_evaluate/          # CV/ngưỡng/under-sampling
docs/
  assignment/de_bai.png            # Ảnh đề
  team/TEAM_ASSIGNMENTS.md         # Phân công
  overview/                       # Kiểm tra và README cũ, lưu lịch sử
  plans/                          # Kế hoạch trước sửa, lưu lịch sử
  references/                     # Đối chiếu Kaggle và nguồn dữ liệu
  guides/                         # Cách đọc/nộp/demo notebook và pipeline
  changes/                        # Cũ → mới, lý do và kết quả kiểm tra
requirements/                     # notebook.txt, annotation.txt, dev.txt
scripts/run_notebook.py            # Chạy notebook trong thư mục riêng
tests/                            # Kiểm tra hồi quy
results/verified_run/              # Metrics/manifest và kiểm tra, không chứa CSV raw
results/source_audit/              # Hash Drive, kiểm kê code và đối soát nguồn
data/                             # CSV/JSON local, không đưa vào Git
.work/                            # Cache và dữ liệu kiểm tra tạm, không đưa vào Git
```

Tên `step_*` là thứ tự công cụ của repo. Các bước theo đề được trình bày trực tiếp
trong notebook; không cần chạy lại Argilla để train từ CSV v7.

## Chạy notebook

Môi trường kiểm tra: Python 3.12. Từ gốc project:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements/notebook.txt
```

Nếu Python 3.12 do `uv` quản lý: `uv venv --python 3.12 .venv`, rồi
`uv pip install --python .venv\Scripts\python.exe -r requirements/notebook.txt`.
Mở notebook, chọn kernel `.venv`, Restart Kernel → Run All. Cell đầu kiểm tra/cài
đúng thư viện; không cài lại khi phiên bản đã khớp.

**CSV sẵn có:** nhập đường dẫn vào `CSV_PATH` ở cell cấu hình, hoặc đặt trước khi
khởi động kernel:

```powershell
$env:SPAMHAM_CSV = (Resolve-Path 'data/labeled_v7.csv').Path
```

**Tự tải HF:** để trống `CSV_PATH`. Notebook tải revision/hash đã chốt của
[dataset nhóm](https://huggingface.co/datasets/p-storm/ptit-facebook-comments-spamham).
API ngày 04/11/2026 trả `gated=false`, CSV tải được không token. Nếu quyền đổi, dùng
`HF_TOKEN` hoặc CSV local. Kaggle cần Internet khi tải; code có đọc Secret nếu cần.
Không ghi token vào notebook.

V7 kiểm tra trực tiếp: 38.233 dòng, ham 32.907, spam 5.326. Có 27 nhóm text giống nhau
mang nhãn khác; không tự sửa khi thiếu ngữ cảnh. Đã có mã crawler và 7 file Drive:
raw Phase 2 gồm 2.170 bài từ 5 group, 13.833 comment/reply. Bản gộp Drive và v7
khác 1.366 nhãn; chưa có lịch sử giải thích thay đổi hoặc raw Phase 1.

**Snapshot nguồn:** notebook tải thêm 7 file Drive để chạy phần đối soát. Nếu chạy
offline, đặt sẵn chúng vào `data/source_drive/` và cấu hình trước khi mở kernel:

```powershell
$env:SPAMHAM_SOURCE_DIR = (Resolve-Path 'data/source_drive').Path
```

Không cần cookie Facebook để chạy notebook. Bộ crawler bàn giao dùng để giải thích
quy trình thu thập và làm căn cứ; notebook không gọi nó để cào live.

**Kiểm tra tự động, thư mục chạy chỉ có notebook:**

```powershell
.\.venv\Scripts\python.exe scripts/run_notebook.py --csv data/labeled_v7.csv --inplace
```

Notebook tự tạo 5 module và artifacts; không phải nộp sẵn các module sinh ra. Khi tách
riêng `.ipynb`, vẫn cần CSV v7 + 7 snapshot hoặc mạng/quyền tải và môi trường Python phù hợp. Runner
dùng chính interpreter gọi nó và `PYTHONHASHSEED=42`, không chọn kernel ngẫu nhiên.

## Công cụ chuẩn bị dữ liệu — chỉ dùng nếu cần làm lại dữ liệu

Từ gốc repo, dùng `python -m` để import hoạt động sau khi chuyển thư mục:

```powershell
python -m pipeline.step_01_collect.convert_phase2_json --help
python -m pipeline.step_01_collect.audit_source --help
python -m pipeline.step_02_preprocess.clean_data_facebook_spamham --help
python -m pipeline.step_03_label.upload_to_argilla --help
python -m pipeline.step_03_label.ai_assisted_spam_ham --help
python -m pipeline.step_03_label.auto_accept_labels --help
python -m pipeline.step_03_label.download_labeled_data --help
python -m pipeline.step_04_merge.merge_labeled_data --help
```

Đường dẫn mặc định vẫn là `data/` ở gốc repo. Script cần Argilla dùng
`requirements/annotation.txt`; server phải được cấu hình riêng. Auto-accept
`--skip-argilla` chỉ xử lý CSV/cache. CSV có `id` sẽ giữ ID; CSV chưa có ID có thể
dùng `--id-prefix p2_cmt_`. Export từ chối nối sai hoặc âm thầm bỏ ID không khớp.
Tái hiện snapshot cleaned cũ phải dùng `--keep-punctuation-only` để không dịch
chỉ số cache cũ. Xem [hướng dẫn bước thu thập](pipeline/step_01_collect/README.md).

## Kiểm tra và kết quả

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements/dev.txt
.\.venv\Scripts\python.exe -m pytest --basetemp .work/pytest -q
```

Metrics/manifest: `results/verified_run/`. Nhật ký cũ → mới: `docs/changes/`.
Không dùng điểm README cũ làm kết quả bản mới vì tiền xử lý, nhóm dấu câu và class
weight đã đổi. AP/bootstrap/OOV là phần thêm ngoài bộ chỉ số bắt buộc.

Module chuẩn được nhúng vào notebook và có test kiểm tra giống nhau. Nếu sửa module,
cập nhật cell `%%writefile` tương ứng, chạy lại phần bị ảnh hưởng. Nguồn Kaggle đã
đối chiếu; không lấy nhận xét/số liệu của nó thay kết quả chạy mới.
