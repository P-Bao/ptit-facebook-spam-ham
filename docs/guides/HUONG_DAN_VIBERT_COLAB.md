# Huấn luyện và trình bày viBERT trên Colab

**Trạng thái:** đã train trên A100, tải model về và kiểm suy luận local.
Muốn trình bày kết quả hiện có, đến mục “Review tại máy” bên dưới; không cần
lặp lại toàn bộ phần cài/train. Luồng đầy đủ ba notebook và lệnh module nằm
trong [README gốc](../../README.md).

## Môi trường và dữ liệu

Giữ nguyên `.venv` truyền thống. `.venv-modern` kiểm tra/demo trên Python 3.12
local bằng `requirements/modern.txt`. Colab Python 3.13 dùng
`requirements/modern-colab.txt`, giữ PyTorch GPU có sẵn nếu >=2.6. Không cài
PyTorch CPU vào Colab GPU.

Thiết lập môi trường local riêng để xem notebook và nạp model bằng CPU:

```powershell
uv venv --python 3.12.14 --seed .venv-modern
uv pip install --python .venv-modern/Scripts/python.exe torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv-modern/Scripts/python.exe -r requirements/modern.txt
```

Nếu `.venv-modern` đã có thì giữ nguyên, không tạo lại. Chọn interpreter/kernel
`.venv-modern/Scripts/python.exe` khi mở Notebook 03. Đây là môi trường suy luận
CPU; việc huấn luyện đầy đủ vẫn thực hiện trên GPU Colab.

Tạo gói chuyển nội bộ từ gốc project:

```powershell
.\.venv\Scripts\python.exe scripts/prepare_colab_bundle.py
```

`.work/colab_transfer/ptit_vibert_bundle.zip` gồm module, cấu hình, v7 và dự đoán
S3 để so sánh. Không có cookie, token Facebook hay `.env`. ZIP phục vụ truyền
file lên Colab, không phải yêu cầu đóng gói bài nộp.

## Chạy trên Colab

Mở notebook trong đúng tài khoản, chọn GPU, kết nối runtime rồi kiểm tra:

```python
!nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
```

Nếu GPU cao cấp treo ở Connecting, thử notebook bình thường trước khi gắn MCP
để tách lỗi runtime và cầu nối. Không đổi GPU khi đang huấn luyện. Quyền trả phí
không bảo đảm luôn được cấp một GPU cụ thể.

Gắn Drive để checkpoint không phụ thuộc ổ đĩa tạm:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Chuyển gói qua MCP, hoặc tải từ máy bằng `google.colab.files.upload()`, sau đó:

```python
from google.colab import files
files.upload()  # Chọn ptit_vibert_bundle.zip trên máy.
```

```python
from pathlib import Path
import zipfile, sys, subprocess, os
PROJECT_ROOT = Path('/content/ptit-facebook-spam-ham')
PROJECT_ROOT.mkdir(exist_ok=True)
with zipfile.ZipFile('/content/ptit_vibert_bundle.zip') as archive:
    archive.extractall(PROJECT_ROOT)
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements/modern-colab.txt'])
```

Thử checkpoint thật trước, sau đó huấn luyện. Các lệnh dùng cùng module:

```python
RUN_DIR = '/content/drive/MyDrive/PTIT_SpamHam/viBERT/2026-11-05'
!python -m pipeline.modern.train --stage smoke --output "$RUN_DIR"
!python -m pipeline.modern.train --stage select --output "$RUN_DIR"
!python -m pipeline.modern.train --stage finalize --output "$RUN_DIR"
```

`--stage all` thực hiện liên tiếp ba bước. Khi điều khiển qua MCP có thể chạy CLI
bằng `subprocess.Popen`, ghi log vào `training.log`, theo dõi `state.json` và
checkpoint; đây vẫn là huấn luyện thật. Không mở hai tiến trình vào cùng output.

Đường dẫn trên là run đã hoàn tất của nhóm. Với một lần train độc lập, thay
bằng thư mục mới (ví dụ `viBERT/new_run`) trước khi chạy; không dùng điểm test
đã thấy để đổi tham số. Có thể chạy Notebook 03 trên Colab bằng cách đặt biến
`SPAMHAM_VIBERT_OUTPUT` tới `RUN_DIR` trước cell cấu hình, đổi `MODE="train"`;
CLI và notebook đều gọi `pipeline.modern.train.run_experiment()`.

## Lựa chọn và tiếp tục khi gián đoạn

- `smoke`: nạp checkpoint chính thức, kiểm tra forward/backward; không dùng số
  đo ở đây làm kết quả phân loại của bài.
- `select`: fine-tune loss thường/có trọng số trên 24.815 fit, đo 5.944 validation
  mỗi epoch, chọn loss/epoch theo validation F1 spam.
- `finalize`: khởi tạo lại pretrained, học toàn bộ 30.759 train với cấu hình đã
  chọn, đo 7.474 test, lưu model/tokenizer và kiểm tra nạp lại.

Checkpoint lưu theo epoch và tự resume khi chạy lại cùng cấu hình/ID. Nếu đổi
cấu hình, dùng output mới. Sau đánh giá, không đổi tham số để tối ưu trên test.
S3 và viBERT so sánh bằng test Accuracy/Precision/Recall/F1/AP, không so CV F1
với validation F1.

## Đầu ra và trình bày

| Đầu ra trong run | Vai trò |
|---|---|
| `run_config.json`, `split_ids.csv`, `split_summary.csv` | Môi trường, cấu hình, vai trò từng ID. |
| `selection/unweighted/`, `selection/balanced/` | Checkpoint phát triển và `training_history.csv`. |
| `validation_metrics.csv`, `selected.json` | Căn cứ chọn loss và số epoch. |
| `final_training/` | Checkpoint khi fit lại toàn bộ train. |
| `model/` | Trọng số safetensors, cấu hình BERT, tokenizer, cấu hình suy luận. |
| `metrics.csv`, `test_predictions.csv` | Chỉ số và dự đoán thực trên từng ID test. |
| `comparison_with_svm.csv`, `figures/` | So sánh S3, ma trận và PR. |
| `verification.json` | Chỉ có khi test và kiểm tra nạp lại hoàn tất. |

Tải kết quả về `artifacts/modern/vibert/`. Bảng nhẹ/bằng chứng lưu thêm vào
`results/modern/`; trọng số/checkpoint không đưa Git. Cách chia ID còn lưu tại
`data/processed/vibert_split.csv`.

Mở `notebooks/03_vibert.ipynb`, chọn `.venv-modern`, dùng `MODE="review"` để đọc
kết quả/model đã tải. Đặt `SPAMHAM_VIBERT_OUTPUT` nếu output ở vị trí khác; trên
Colab trỏ đến thư mục run trên Drive. `MODE="train"` gọi lại thực nghiệm. Mỗi
chế độ được ghi rõ, không gọi việc đọc CSV là train.

Notebook 02 trình bày bảng so test khi có kết quả hoàn tất; Notebook 03 giải
thích chi tiết tokenization, fine-tune và demo viBERT.


## Kết quả viBERT đã kiểm chứng ngày 05/11/2026

Đã fine-tune checkpoint chính thức trên **NVIDIA A100-SXM4-80GB** của Colab.
Hai cấu hình loss dùng 24.815 fit và 5.944 validation. Loss thường ở epoch 3
đạt validation F1 **0,82048**, loss có trọng số đạt **0,81916**; chọn loss thường
theo tiêu chí đã xác định. Khởi tạo lại pretrained và fit **30.759 train** trong
ba epoch, sau đó mới đo trên cùng **7.474 test** với S3. Không dùng test để chọn
loss/epoch hoặc điều chỉnh ngưỡng; ngưỡng xác suất cố định 0,5.

| Mô hình | Accuracy | Precision spam | Recall spam | F1 spam | AP |
|---|---:|---:|---:|---:|---:|
| S3 — TF-IDF + SVM | 0.9433 | 0.8374 | 0.6877 | 0.7552 | 0.8117 |
| M1 — viBERT | 0.9635 | 0.8942 | 0.8086 | 0.8493 | 0.9108 |

Ma trận M1: **6.432 TN, 91 FP, 182 FN, 769 TP**. So với S3, bắt thêm 115 spam
và giảm 36 ham nhận nhầm. Đây là kết quả trên tập test hiện tại, chưa phải kết
luận cho mọi nguồn dữ liệu hoặc nhiều seed. Không so validation F1 của viBERT
với CV F1 của SVM như các phép đo tương đương.

Đã nạp lại trên GPU và kiểm tra toàn bộ 7.474 dự đoán khớp; model/tokenizer đã
được tải về `artifacts/modern/vibert/model/`. Kiểm tra CPU local trên 64 mẫu đầu
của test: **không khác nhãn**; chênh lệch xác suất tối đa khoảng
**0.00003762** do môi trường/độ chính xác số khác nhau.
Dự đoán lặp lại trên CPU giữ nguyên. Không khẳng định mọi số thực trùng tuyệt đối
giữa GPU FP16 và CPU FP32.

Notebook 03 có hai chế độ rõ ràng: train thực hiện fine-tune; review đọc lịch sử,
chỉ số và nạp checkpoint đã học. Có thể chạy review local từ Terminal:

```powershell
.\.venv-modern\Scripts\python.exe scripts/run_notebook.py --notebook 03 --inplace
```

Chạy cell không chỉ hiển thị tượng trưng: các cell dữ liệu/tokenization xử lý
dữ liệu, các cell review đọc kết quả đã train, cell demo nạp trọng số và tính
dự đoán. Chỉ đổi MODE sang train khi muốn huấn luyện lại bằng GPU.

`results/modern/` giữ báo cáo nhẹ và bằng chứng; model/data được Git bỏ qua.
Các `.bin` checkpoint optimizer trong thư mục huấn luyện dùng để resume;
trọng số triển khai nằm trong `model.safetensors`, không phải `.joblib` của S3.
Phần báo cáo học phần chưa được viết trong đợt triển khai này.

## Chuyển model và kết quả từ Colab về máy

Giữ bản run trên Drive. Để dùng lại chỉ cần model/tokenizer và các file kết
quả phục vụ review; checkpoint optimizer/encoder tạm dùng resume có thể rất
lớn, không cần tải về để demo. Khi muốn resume training, dùng checkpoint đầy
đủ còn trên Drive, không dùng riêng model ZIP như một checkpoint optimizer.

Tạo hai gói chuyển trong Colab nếu cần tải bằng trình duyệt:

```python
from pathlib import Path
from google.colab import files
import shutil, zipfile

run = Path(RUN_DIR)
model_zip = shutil.make_archive('/content/ptit_vibert_model', 'zip', root_dir=run / 'model')
review_zip = Path('/content/ptit_vibert_review.zip')
with zipfile.ZipFile(review_zip, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
    outputs = list(run.glob('*.csv')) + list(run.glob('*.json'))
    outputs += list((run / 'figures').glob('*.png'))
    outputs += list((run / 'selection').glob('*/training_history.csv'))
    outputs += list((run / 'final_training').glob('training_history.csv'))
    for path in outputs:
        archive.write(path, path.relative_to(run))
files.download(str(review_zip))
files.download(model_zip)
```

Nếu trình duyệt chưa tải, mở bảng **Files** của Colab, vào `/content/`, chọn
menu của file ZIP → **Download**; cho phép tải file nếu trình duyệt hỏi. Đây là
gói chuyển nội bộ, không phải yêu cầu ZIP bài nộp.

Tại máy, mở Terminal từ gốc repo. Model ZIP hiện tại của nhóm có trọng số ở
ngay gốc ZIP, nên giải nén **vào `model/`**, không giải nén vào gốc repo:

```powershell
Expand-Archive -LiteralPath "$env:USERPROFILE\Downloads\ptit_vibert_model_20261105.zip" -DestinationPath artifacts/modern/vibert/model -Force
```

Với gói review tạo theo đoạn trên:

```powershell
Expand-Archive -LiteralPath "$env:USERPROFILE\Downloads\ptit_vibert_review.zip" -DestinationPath artifacts/modern/vibert -Force
```

Chỉ giải nén đè nếu đây là kết quả của cùng run muốn sử dụng. Với run khác,
giữ thư mục riêng và đổi `SPAMHAM_VIBERT_OUTPUT` để tránh trộn model/lịch sử.

## Review tại máy

Cấu trúc tối thiểu phục vụ toàn bộ Notebook 03:

```text
artifacts/modern/vibert/
  model/
    model.safetensors
    config.json
    inference_config.json
    tokenizer.json
    tokenizer_config.json
    special_tokens_map.json
    vocab.txt
  selection/
    unweighted/training_history.csv
    balanced/training_history.csv
  figures/
    confusion_vibert.png
    pr_comparison.png
  validation_metrics.csv
  selected.json
  run_config.json
  verification.json
  comparison_with_svm.csv
  test_predictions.csv
```

Các file `metrics.csv`, `split_ids.csv`, `split_summary.csv`, `smoke.json` và
trạng thái cũng nên giữ để kiểm tra kết quả. `training_args.bin` là metadata
huấn luyện; `VibertPredictor` không nạp file này để suy luận. Đầu vào v7 vẫn
nằm tại `data/final/labeled_v7.csv`.

Mở Notebook 03 → chọn kernel `.venv-modern` → giữ `MODE="review"` → Restart
Kernel → Run All. Hoặc dùng runner bằng chính Python `.venv-modern` như lệnh
ở trên. Review xử lý dữ liệu/tokenizer, đọc lịch sử/test, nạp model và tính
dự đoán demo; không gọi lại các epoch train.

Xem [cách đọc token](DOC_TOKEN_VA_KET_QUA.md) nếu thấy `##`/`[UNK]`. Đó không
phải lỗi font. M1 dùng `text` gốc nên giữ cả ký tự nước ngoài có sẵn trong CSV.
Số mẫu quá 128 token và số token `[UNK]` là hai đại lượng khác nhau.

Để dự đoán bằng code bên ngoài notebook, dùng `VibertPredictor` trong
`pipeline/modern/predict.py`; ví dụ Terminal ở [README mục 8](../../README.md#8-dự-đoán-bằng-module-bên-ngoài).
Để chạy 01/02, quay về `.venv`; không áp môi trường modern cho toàn project.
