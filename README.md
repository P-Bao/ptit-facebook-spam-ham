# Phân loại Spam/Ham bình luận sinh viên PTIT

Project của Bảo, Cương, Phi, Phương: thu thập và chuẩn bị bình luận Facebook,
thực nghiệm BoW/TF-IDF với NB/SVM, fine-tune viBERT, đánh giá và dự
đoán bình luận mới. S3 và viBERT được so sánh trên cùng 7.474 mẫu test.

- [Tổng quan folder, file và toàn bộ luồng](docs/guides/TONG_QUAN_PROJECT_VA_CACH_THUC_THI.md).
- [Danh mục module pipeline](pipeline/README.md) và [tài liệu project](docs/README.md).
- [Phân công học cho bốn thành viên](docs/team/TEAM_ASSIGNMENTS.md).
- [Hướng dẫn train viBERT trên Colab](docs/guides/HUONG_DAN_VIBERT_COLAB.md).
- [Cách đọc token và chỉ số](docs/guides/DOC_TOKEN_VA_KET_QUA.md).
- [Hướng dẫn tự chạy Terminal, từng cell và cài AI–Argilla](docs/guides/HUONG_DAN_TU_CHAY_VA_GAN_NHAN.md).

## 1. Chọn cách chạy

| Mục đích | Cách thực hiện | Cần kết nối dịch vụ? |
|---|---|---|
| Tạo lại CSV từ dữ liệu/nhãn đã lưu | Notebook 01 hoặc các lệnh offline mục 5 | Không |
| Huấn luyện/đánh giá lại mô hình truyền thống | Notebook 02 hoặc runner mục 3 | Không |
| Kiểm tra toàn bộ cấu hình GridSearchCV | Notebook 04, tùy chọn, sau Notebook 02 | Không |
| Xem kết quả viBERT và dự đoán tại máy | Notebook 03, `MODE="review"`, mục 3–4 | Không khi đủ model/kết quả local |
| Thu thập/gán nhãn cho dữ liệu mới | CLI mục 6 | Facebook/API, Argilla nếu dùng giao diện |
| Fine-tune viBERT lại | Colab GPU, mục 7 | GPU, tải pretrained và lưu Drive |
| Chỉ dự đoán từ mô hình đã học | Gọi hàm Python ở mục 8 | Không |

Notebook điều phối thứ tự, gọi hàm trong `pipeline/` và hiển thị kết quả.
Cell xử lý/fit/ghi file có thực thi; khối trích mã nguồn chỉ hiển thị hàm.
Chỉ mở notebook xem output không tự chạy code.

## 2. Môi trường và đầu vào

### 2.1. Môi trường Notebook 01/02

Mở toàn bộ thư mục project trong VS Code, Terminal PowerShell đứng ở gốc chứa
README này. Dùng Python **3.12.14**. Nếu môi trường đã có thì giữ nguyên.
Khi cần tạo mới bằng `uv`:

```powershell
uv venv --python 3.12.14 --seed .venv
.\.venv\Scripts\python.exe -m pip install -r requirements/notebook.txt
.\.venv\Scripts\python.exe scripts/check_environment.py
.\.venv\Scripts\python.exe -m pip check
```

Tiếp tục khi checker báo `Environment OK`, pip không có dependency lỗi. Có thể
tạo bằng `py -3.12 -m venv .venv` nếu máy đã cài Python 3.12 qua bộ cài thông
thường. Máy hiện tại có Python 3.14 trên PATH; dùng đường dẫn `.venv` trực tiếp
để tránh chọn sai Python. Không cần activate hoặc đổi PowerShell Execution Policy.

### 2.2. Môi trường Notebook 03 tại máy

Giữ riêng `.venv-modern`, không cài torch/transformers vào `.venv` truyền thống.
Nếu chưa có:

```powershell
uv venv --python 3.12.14 --seed .venv-modern
uv pip install --python .venv-modern/Scripts/python.exe torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv-modern/Scripts/python.exe -r requirements/modern.txt
.\.venv-modern\Scripts\python.exe -m pip check
```

Đây là môi trường CPU để review/demo model đã học. Train đầy đủ dùng GPU Colab
và `requirements/modern-colab.txt`, giữ bản PyTorch GPU của Colab.

### 2.3. Đầu vào local cần có

| Bước | File/thư mục |
|---|---|
| 01 | `.cache/collection/posts.json`, `annotation_index.csv`, `annotation_predictions.json` |
| 02 | `data/final/labeled_v7.csv` |
| 03 review | V7 và toàn bộ `artifacts/modern/vibert/`: model/tokenizer, `validation_metrics.csv`, `selected.json`, `run_config.json`, `verification.json`, `comparison_with_svm.csv`, `test_predictions.csv`, lịch sử hai loss trong `selection/`, hình trong `figures/` |
| Tool tổng hợp tùy chọn | `data/labeled/comments_labeled_phase1.csv` và CSV do 01 xuất |
| Train viBERT | V7 và `artifacts/spamham/test_predictions.csv` của Notebook 02 để so sánh S3 |

Các đầu vào đã có trên máy hiện tại. Dữ liệu và trọng số bị Git bỏ qua: thành
viên khác phải nhận thêm các thư mục này, **chỉ pull code chưa đủ để chạy**.
Notebook 02 đọc v7 local, không tải Hugging Face mỗi lần. File tổng hợp do tool
tạo không tự thay thế v7. ZIP chỉ chứa model chưa đủ để hiện toàn bộ lịch sử/
hình của 03; cần chuyển cả các file kết quả nhỏ đi cùng.

## 3. Chạy đầy đủ từ Terminal

Chạy từng lệnh, chờ thành công trước khi chuyển lệnh tiếp theo:

```powershell
.\.venv\Scripts\python.exe scripts/run_notebook.py --notebook 01 --inplace
.\.venv\Scripts\python.exe scripts/run_notebook.py --notebook 02 --inplace
.\.venv-modern\Scripts\python.exe scripts/run_notebook.py --notebook 03 --inplace
.\.venv\Scripts\python.exe scripts/run_notebook.py --notebook 04 --inplace
```

03 mặc định `MODE="review"`: đọc kết quả đã train, xử lý dữ liệu/tokenization,
nạp model và suy luận; không fine-tune lại. Không dùng `.venv` của 01/02 cho 03.
04 là kiểm tra bổ sung các cấu hình CV và ghi bảng vào `artifacts/grid_audit/`;
chạy bằng `.venv` của 01/02. Có thể bỏ qua 04 khi chỉ cần luồng chính 01–03.

| Lệnh | Đầu ra chính |
|---|---|
| 01 | `data/intermediate/`, `data/labeled/comments_labeled.csv`, `data/collection_manifest.json` |
| 02 | `data/processed/v7_preprocessed.csv`; S3 `.joblib`, metrics/CV/test/hình trong `artifacts/spamham/` |
| 03 review | Cập nhật output notebook, hiện lịch sử/test; dự đoán demo bằng checkpoint đã lưu |
| 04 tùy chọn | Bảng toàn bộ cấu hình CV và cấu hình tốt nhất theo nhóm trong `artifacts/grid_audit/` |

Để **học và chạy riêng phần Phi** từ CSV đã có nhãn/token, không cần chạy lại các cell tạo dữ liệu của Notebook 02:

```powershell
.\.venv\Scripts\python.exe scripts/run_phi_pipeline.py --jobs 2
```

Script tự chia train/test, chạy BoW/TF-IDF với Naive Bayes/SVM, khảo sát B4 và S1–S3, chấm test rồi lưu `metrics.csv`, `test_predictions.csv`, `run_summary.json` và `spam_ham_ml_only.joblib` vào `artifacts/spamham/phi_pipeline/`. Thư mục này tách khỏi đầu ra chính của Notebook 02. Đọc [luồng code Phi](docs/team/PHI_DOC_CODE_THEO_LUONG.md) để đối chiếu từng lệnh với module thực hiện; [giải thích bốn lớp scikit-learn](docs/team/PHI_HIEU_4_LOP_SCIKIT_LEARN.md) chỉ đúng phương thức nguồn cần đọc bằng Ctrl+Click.

`--inplace` chỉ ghi lại output notebook gốc nếu tất cả cell thành công. Kernel,
bản notebook thực thi và `execution_status.json` nằm ở `.work/runs/01/`, `02/`,
`03/`, `04/`. Notebook 02 chạy nhiều cấu hình và có thể cần khoảng 10 phút trên máy
hiện tại; không ngắt giữa chừng khi đang fit.

Nếu cần tạo lại file tổng hợp ngoài luồng model:

```powershell
.\.venv\Scripts\python.exe -m tools.dataset_versions.merge_labeled_data
```

Đầu ra là `data/final/comments_merged.csv`; lệnh không ghi đè v7. Không bắt
buộc chạy lại tool mỗi khi huấn luyện 02 hoặc xem 03.

Runner đặt seed, hash seed và số luồng BLAS trước khi mở kernel.
Không mở hai lượt chạy cùng notebook/output đồng thời.

`experiments.py`, `training_utils.py` và
`preprocess_for_model.py` là thư viện cung cấp hàm/class, không có CLI tự chạy
cả thí nghiệm. Muốn chạy đầy đủ phần model từ Terminal, dùng runner 02 ở trên;
không lần lượt mở từng file thư viện rồi bấm Run Python File.

## 4. Chạy từng cell trong VS Code

1. Mở project bằng **File → Open Folder**, bật extension **Python** và **Jupyter**.
2. Mở [Notebook 01](notebooks/01_thu_thap_va_gan_nhan.ipynb), chọn **Select Kernel
   → Python Environments → `.venv\Scripts\python.exe`** của project.
3. Chọn **Restart Kernel → Run All** hoặc chạy cell từ trên xuống. Các bước
   chuyển bảng/làm sạch/tiếp nhận nhãn ghi file vào `data/`.
4. Mở [Notebook 02](notebooks/02_thuc_nghiem_va_demo.ipynb), dùng **cùng `.venv`**.
   Mục 3 ghi CSV token; mục 5/5.1 fit và CV; mục 6 đánh giá; mục 8 ghi S3 `.joblib`.
5. Mở [Notebook 03](notebooks/03_vibert.ipynb), đổi kernel sang
   **`.venv-modern\Scripts\python.exe`**. Giữ `MODE="review"` tại mục 1 rồi Run All.
   Mục 5 đọc lịch sử đã train; mục 8 nạp viBERT và tính dự đoán mới.
6. Nếu muốn rà toàn bộ lưới tham số, mở [Notebook 04](notebooks/04_kiem_tra_grid_search.ipynb),
   chọn lại kernel `.venv` và Run All. Đây là kiểm tra bổ sung, không thay mô hình đã chọn.

Kiểm tra kernel bằng cell tạm nếu cần:

```python
import sys
print(sys.executable)
print(sys.version)
```

Sau khi kiểm tra có thể xóa cell tạm. Chạy cell giữa notebook cần các biến ở
phần trước; `NameError` thường do chưa chạy phần khởi tạo. Sau khi fit lại cần
chạy tiếp đánh giá/lưu/demo để không hiển thị số liệu cũ. Restart Kernel xóa
biến trong RAM nhưng không xóa dữ liệu/model trên đĩa.

Tự chạy cell và chạy runner dùng cùng module/logic; runner thực hiện cả
notebook trong kernel mới và cố định môi trường khởi động. Số thực/thời gian
có thể khác khi đổi máy, phiên bản hoặc chế độ tính toán GPU/CPU.

## 5. Chạy các bước dữ liệu bằng module bên ngoài — offline

Phần này tương ứng các bước chính của Notebook 01 với đầu vào đã lưu, không
gọi Facebook/LLM. Không chạy xen giữa lúc Notebook 01 đang dùng cùng output.

### 5.1. JSON thành CSV rồi làm sạch

```powershell
.\.venv\Scripts\python.exe -m pipeline.collect.json_to_csv --input .cache/collection/posts.json --keep-phase1-duplicates --output data/intermediate/comments.csv
.\.venv\Scripts\python.exe -m pipeline.prepare.clean_for_labeling --input data/intermediate/comments.csv --output data/intermediate/comments_cleaned.csv
```

Flag `--keep-phase1-duplicates` yêu cầu chuyển toàn bộ dữ liệu của JSON, giống
cell Notebook 01. CLI mặc định có nhánh loại trùng với file tham chiếu; bỏ flag
có thể làm số dòng khác notebook. Chỉ nhận JSON dạng mảng bài viết; nếu crawler
lưu JSONL thì cần chuyển định dạng trước, không đưa JSONL vào converter này.

### 5.2. Nối đúng ID với gợi ý nhãn đã lưu

`cached_annotations.py` là thư viện, gọi bằng đoạn Python sau trong PowerShell:

```powershell
@'
import json
from pathlib import Path
import pandas as pd
from pipeline.label.cached_annotations import attach_annotation_ids

folder = Path('data/intermediate')
frame = pd.read_csv(folder / 'comments_cleaned.csv', dtype=str, keep_default_na=False)
index = pd.read_csv('.cache/collection/annotation_index.csv', dtype=str, keep_default_na=False)
cache = json.loads(Path('.cache/collection/annotation_predictions.json').read_text(encoding='utf-8'))
frame, linked = attach_annotation_ids(frame, index, cache)
frame.to_csv(folder / 'comments_cleaned.csv', index=False, encoding='utf-8-sig')
(folder / 'annotation_predictions.json').write_text(json.dumps(linked, ensure_ascii=False, indent=2), encoding='utf-8')
print(len(frame), len(linked))
'@ | .\.venv\Scripts\python.exe -
```

Không lấy vị trí dòng mới làm ID của cache cũ. Hàm kiểm khóa nội dung và
`cleaned_text` trước khi dùng gợi ý, từ chối nếu mâu thuẫn.

### 5.3. Tiếp nhận và xuất CSV có nhãn

```powershell
.\.venv\Scripts\python.exe -m pipeline.label.auto_accept_labels --csv-in data/intermediate/comments_cleaned.csv --cache data/intermediate/annotation_predictions.json --threshold 0.75 --skip-argilla --export-csv data/intermediate/annotation_review.csv
```

File review giữ cả các dòng chưa có nhãn. Xuất riêng mẫu đủ điều kiện:

```powershell
@'
from pathlib import Path
import pandas as pd
review = pd.read_csv('data/intermediate/annotation_review.csv', dtype=str, keep_default_na=False)
labeled = review[review.official_label.isin(['ham', 'spam'])].copy()
labeled = labeled.rename(columns={'official_label': 'label', 'ai_confidence': 'confidence'})
columns = ['id', 'post_id', 'group_id', 'group_name', 'created_at', 'comment_type',
           'text', 'cleaned_text', 'reaction_count', 'label', 'confidence']
output = Path('data/labeled/comments_labeled.csv')
output.parent.mkdir(parents=True, exist_ok=True)
labeled[columns].to_csv(output, index=False, encoding='utf-8-sig')
print(output, len(labeled), labeled.label.value_counts().to_dict())
'@ | .\.venv\Scripts\python.exe -
```

Kết quả hiện tại: 6.289 mẫu, 5.190 ham/1.099 spam. Notebook 01 còn trình bày bảng
và ghi manifest. Để có cả output/manifest đồng bộ, chạy lại Notebook 01.

## 6. Thu thập và gán nhãn mới qua dịch vụ

Đây là nhánh tạo dữ liệu mới, không cần chạy lại để trình bày bộ dữ liệu hiện
tại. API/giao diện Facebook và phiên truy cập có thể thay đổi; code crawler
không bảo đảm lấy đúng số mẫu lịch sử. Xem [README collect](pipeline/collect/README.md)
và [giới hạn crawler](docs/provenance/CRAWLER_VA_DU_LIEU_DRIVE.md).

### 6.1. Thu thập

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements/collection.txt
.\.venv\Scripts\python.exe -m pipeline.collect.scrape_ptit_groups --help
```

Sửa danh sách `PTIT_GROUPS` khi cần. Cấu hình `FB_COOKIES`, `FB_DTSG` trong
`.env` local bằng phiên có quyền truy cập nhóm. Không đưa file `.env` vào Git.
Ví dụ tạo một lần thu thập mới, tránh dùng lại state/output của lần trước:

```powershell
.\.venv\Scripts\python.exe -m pipeline.collect.scrape_ptit_groups --year 2026 --limit 50 --output data/raw/new_run/posts.json --state-file data/raw/new_run/state.json
.\.venv\Scripts\python.exe -m pipeline.collect.json_to_csv --input data/raw/new_run/posts.json --keep-phase1-duplicates --output data/intermediate/new_run/comments.csv
.\.venv\Scripts\python.exe -m pipeline.prepare.clean_for_labeling --input data/intermediate/new_run/comments.csv --output data/intermediate/new_run/comments_cleaned.csv
```

`--limit` là số bài mỗi nhóm; `--year` đặt mốc dưới 01/01, không giới hạn trên.
Writer lưu JSON và cursor/state. Resume/phân trang có giới hạn được ghi trong
biên bản nguồn. Crawler chưa tạo nhãn hoặc v7.

### 6.2. AI và tiếp nhận nhãn không dùng Argilla

Tạo môi trường gán nhãn riêng (chỉ cần làm lần đầu):

```powershell
uv venv --python 3.12.14 --seed .venv-label
.\.venv-label\Scripts\python.exe -m pip install -r requirements/annotation.txt
```

Cấu hình `OPENROUTER_API_KEY` hoặc `OPENAI_API_KEY`, endpoint/model trong `.env`
local. Đặt model API còn được nhà cung cấp hỗ trợ; mặc định code là
`z-ai/glm-4.7-flash`, không cam kết endpoint luôn còn hoạt động. Gọi API có thể
phát sinh chi phí; chỉ chạy khi muốn gán nhãn mới:

```powershell
.\.venv-label\Scripts\python.exe -m pipeline.label.ai_assisted_spam_ham --csv data/intermediate/new_run/comments_cleaned.csv --cache data/intermediate/new_run/predictions.json --no-upload
.\.venv-label\Scripts\python.exe -m pipeline.label.auto_accept_labels --csv-in data/intermediate/new_run/comments_cleaned.csv --cache data/intermediate/new_run/predictions.json --threshold 0.75 --skip-argilla --export-csv data/intermediate/new_run/annotation_review.csv
```

Lọc `official_label` như mục 5.3, đổi đường dẫn theo lần mới. Dùng cache/ID của
chính CSV đó; không tái sử dụng cache lịch sử cho dữ liệu khác. Chưa có ID thì
hai CLI cùng dùng tiền tố mặc định `cmt_`; nếu đổi tiền tố cần đồng bộ.

### 6.3. Nếu nhóm muốn duyệt bằng Argilla

Làm theo [hướng dẫn cài server và cấu hình key](docs/guides/HUONG_DAN_TU_CHAY_VA_GAN_NHAN.md)
trước. Truyền URL/key đúng của server vào mỗi CLI; các mặc định của code có
thể khác Compose vừa cài. Lệnh đầy đủ đã có trong hướng dẫn đó.

Cần Argilla server đang hoạt động và cùng dataset/API URL/key. Cấu hình theo
`--help` của từng CLI, không dùng `--reset` nếu cần giữ nhãn cũ:

```powershell
.\.venv-label\Scripts\python.exe -m pipeline.label.upload_to_argilla --csv data/intermediate/new_run/comments_cleaned.csv --dataset comments_spamham_new
.\.venv-label\Scripts\python.exe -m pipeline.label.ai_assisted_spam_ham --csv data/intermediate/new_run/comments_cleaned.csv --cache data/intermediate/new_run/predictions.json --dataset comments_spamham_new
.\.venv-label\Scripts\python.exe -m pipeline.label.auto_accept_labels --csv-in data/intermediate/new_run/comments_cleaned.csv --cache data/intermediate/new_run/predictions.json --dataset comments_spamham_new --threshold 0.75 --export-csv data/intermediate/new_run/annotation_review.csv
.\.venv-label\Scripts\python.exe -m pipeline.label.download_labeled_data --dataset comments_spamham_new --csv-lookup data/intermediate/new_run/comments_cleaned.csv --output data/labeled/comments_new.csv
```

Sau auto-accept, có thể duyệt phần còn lại trên giao diện trước khi download.
`download_labeled_data` mặc định lấy nhãn đã submit; gợi ý AI chưa duyệt không
tự trở thành nhãn chính thức. Các CLI Argilla dùng `--api-url/--api-key`, riêng
AI dùng `--argilla-url/--argilla-key`; xem help để trỏ đúng server. Nếu không
có server, dùng nhánh `--no-upload`/`--skip-argilla` ở trên.

## 7. Huấn luyện viBERT trên Colab

Từ máy local, tạo gói truyền code/data/S3:

```powershell
.\.venv\Scripts\python.exe scripts/prepare_colab_bundle.py
```

Gói nằm tại `.work/colab_transfer/ptit_vibert_bundle.zip`. Upload, giải nén,
cài dependency GPU và mount Drive theo [hướng dẫn Colab](docs/guides/HUONG_DAN_VIBERT_COLAB.md).
Từ gốc project trên Colab:

```python
RUN_DIR = '/content/drive/MyDrive/PTIT_SpamHam/viBERT/new_run'
!python -m pipeline.modern.train --stage smoke --output "$RUN_DIR"
!python -m pipeline.modern.train --stage select --output "$RUN_DIR"
!python -m pipeline.modern.train --stage finalize --output "$RUN_DIR"
```

Hoặc `--stage all` cho cả ba bước. Lần train khác dùng output mới; không tối ưu
lại theo điểm test đã thấy. Muốn train qua Notebook 03 trên Colab, đặt
`SPAMHAM_VIBERT_OUTPUT` trỏ tới output Drive **trước cell cấu hình**, đổi
`MODE="train"`, chạy từ đầu. CLI và notebook cùng gọi `run_experiment()`.

`select` chọn loss/epoch bằng validation; `finalize` khởi tạo lại pretrained,
fit toàn bộ train, lưu model, đo test và nạp lại kiểm tra. Checkpoint theo epoch
trên Drive cho phép resume khi cùng cấu hình/ID. Khi tải về để review, lấy cả
model và bảng/lịch sử/hình, giữ cấu trúc `artifacts/modern/vibert/`.

## 8. Dự đoán bằng module bên ngoài

Không cần chạy lại train nếu đã có model.

S3, từ PowerShell ở gốc project:

```powershell
@'
import joblib
from pipeline.modeling.predict import predict_spam
bundle = joblib.load('artifacts/spamham/spam_ham_ml_only.joblib')
print(predict_spam(['Cho minh hoi lich thi', 'Lien he https://example.com'], bundle).to_string(index=False))
'@ | .\.venv\Scripts\python.exe -
```

viBERT, dùng môi trường riêng:

```powershell
@'
from pipeline.modern.predict import VibertPredictor
model = VibertPredictor('artifacts/modern/vibert/model', device='cpu')
print(model.predict(['Cho minh hoi lich thi', 'Lien he https://example.com']).to_string(index=False))
'@ | .\.venv-modern\Scripts\python.exe -
```

Ví dụ Terminal dùng chuỗi ASCII để tránh PowerShell cũ đổi dấu khi pipe code;
Notebook 02/03 demo câu có dấu. Nếu viết script `.py` lưu UTF-8 thì dùng tiếng
Việt bình thường. S3 trả decision score/ngưỡng trong bundle; viBERT trả xác
suất spam/ngưỡng 0,5. File `.py` không chứa trọng số đã học.

## 9. Kiểm tra và xử lý lỗi

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements/dev.txt
.\.venv\Scripts\python.exe -m pytest --basetemp .work/pytest -p no:cacheprovider -q
.\.venv-modern\Scripts\python.exe -m pytest tests/test_modern_data.py tests/test_modern_training.py --basetemp .work/pytest-modern -p no:cacheprovider -q
```

Lệnh cuối cần pytest; nếu chưa có trong `.venv-modern`, cài `pytest==9.2.0`
bằng Python môi trường đó. Test BERT nhỏ kiểm chức năng train/save/reload,
không thay thế kết quả fine-tune viBERT thực trên Colab.

| Hiện tượng | Cách xử lý |
|---|---|
| `NameError`/biến chưa tồn tại | Restart Kernel → Run All theo đúng thứ tự |
| `No module named pipeline` | Mở cả project, chạy từ gốc và chọn đúng interpreter |
| Sai phiên bản/thiếu thư viện | Cài đúng requirements cho từng môi trường, restart kernel; không trộn hai venv |
| Thiếu CSV/model/lịch sử | Sao chép đúng file local ở mục 2.3; pull Git không chứa trọng số/data |
| 03 báo cần CUDA khi select/finalize | Giữ review tại máy; train trên GPU Colab |
| 03 không có biểu đồ dù đã tải model ZIP | Tải thêm bảng, lịch sử và `figures/` từ run Colab |
| Token có `##`, `[UNK]` | Đọc chú giải ở mục 3 Notebook 03; không thay font/tokenizer tùy tiện |
| Output sai vị trí | Kiểm các biến `SPAMHAM_CSV`, `SPAMHAM_SOURCE_DIR`, `SPAMHAM_OUTPUT_DIR`, `SPAMHAM_VIBERT_OUTPUT`; bỏ cấu hình cũ và restart kernel |
| Colab bị ngắt | Kết nối lại/mount Drive, chạy cùng stage/config/output để resume; không mở hai lượt train đồng thời |

Windows có thể in `joblib resource_tracker.KeyError` khi đóng kernel sau khi
cell tính toán hoàn tất; đã ghi nhận ở các lượt kiểm tra trước. Đọc trạng thái
runner và kiểm file đã lưu để phân biệt với lỗi trong cell; không coi mọi lỗi
kernel đều là cảnh báo này. Bằng chứng hiện hành: `results/verified_run/`,
`results/modern/`; hồ sơ cũ ở `results/history/`.

Kết quả test đã xác nhận: **S3 F1 0,7552; M1 F1 0,8493**. Đọc cùng Precision,
Recall, FP/FN và giới hạn trong [tổng quan](docs/guides/TONG_QUAN_PROJECT_VA_CACH_THUC_THI.md).
