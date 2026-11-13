# Hướng dẫn tự chạy project và gán nhãn

Chạy các lệnh dưới đây trong **PowerShell tại thư mục gốc project**, nơi chứa
`README.md`, `pipeline/`, `scripts/` và `notebooks/`. Chờ mỗi lệnh hoàn tất rồi
mới chạy lệnh sau. Máy hiện tại đã có `.venv`, `.venv-modern` và dữ liệu local.

## 1. Chọn đúng môi trường

| Công việc | Python dùng để chạy |
|---|---|
| Notebook 01, 02, 04; mô hình truyền thống | `.venv\Scripts\python.exe` |
| Notebook 03 review/demo viBERT | `.venv-modern\Scripts\python.exe` |
| Gọi AI, upload/download Argilla | `.venv-label\Scripts\python.exe`, tạo ở mục 5 |

Không cần activate. Chỉ rõ đường dẫn Python trong mỗi lệnh giúp tránh dùng nhầm
Python 3.14 của hệ thống. Mỗi file `.py` thư viện không tự chạy cả thí nghiệm;
runner/notebook mới điều phối các hàm theo thứ tự.

Kiểm tra môi trường truyền thống:

```powershell
.\.venv\Scripts\python.exe scripts/check_environment.py
.\.venv\Scripts\python.exe -m pip check
.\.venv-modern\Scripts\python.exe -m pip check
```

Nếu cần dựng lại môi trường, làm theo mục 2 của [README](../../README.md).
Dữ liệu và trọng số không nằm trên Git. Trước khi chạy phải có:

- 01: `.cache/collection/posts.json`, `annotation_index.csv`, `annotation_predictions.json`.
- 02: `data/final/labeled_v7.csv`.
- 03 review: toàn bộ `artifacts/modern/vibert/`, gồm `model/`, bảng kết quả,
  lịch sử trong `selection/`, `figures/`; đồng thời cần kết quả S3 của 02.
- 04: `data/processed/v7_preprocessed.csv`, được 02 tạo ra.

## 2. Chạy toàn bộ bằng Terminal

```powershell
.\.venv\Scripts\python.exe scripts/run_notebook.py --notebook 01 --inplace
.\.venv\Scripts\python.exe scripts/run_notebook.py --notebook 02 --inplace
.\.venv-modern\Scripts\python.exe scripts/run_notebook.py --notebook 03 --inplace
```

01 đọc dữ liệu và gợi ý nhãn local, chuyển bảng, làm sạch, tiếp nhận nhãn và
ghi `data/labeled/comments_labeled.csv`. Nó không gọi lại Facebook hoặc API AI.
02 đọc v7, tạo CSV token, chia tập, chạy CV, chọn mô hình, chấm test, lưu/nạp
`.joblib` rồi dự đoán. CSV của 01 không tự thay đầu vào v7 của 02.
03 mặc định `MODE="review"`, đọc kết quả fine-tune đã có và chạy suy luận;
lệnh này không train viBERT lại trên CPU.

Nếu muốn xem điểm của **toàn bộ lưới cấu hình CV**, chạy thêm:

```powershell
.\.venv\Scripts\python.exe scripts/run_notebook.py --notebook 04 --inplace
```

04 là kiểm tra bổ sung. Bảng đầy đủ nằm trong `artifacts/grid_audit/`.
Sau khi chạy đủ 01–04, có thể kiểm trạng thái và đối chiếu lại toàn bộ 7.474
dự đoán test với model `.joblib` vừa lưu:

```powershell
.\.venv\Scripts\python.exe scripts/verify_local_outputs.py
```

Muốn chạy riêng phần Phi từ CSV đã có token/nhãn:

```powershell
.\.venv\Scripts\python.exe scripts/run_phi_pipeline.py --jobs 2
```

Lệnh này ghi vào `artifacts/spamham/phi_pipeline/`; không thay kết quả chính
của 02. Nó thực hiện chia tập → B1–B4 → S1–S3 → chấm test → lưu mô hình.

Runner ghi trạng thái vào `.work/runs/01/` đến `04/execution_status.json`.
`"success": true` nghĩa là tất cả cell đã hoàn tất. `--inplace` cập nhật output
notebook gốc khi lượt chạy thành công. Không chạy hai lượt cùng output đồng thời.

## 3. Chạy bằng từng cell trong VS Code

1. **File → Open Folder** mở cả project; cài extension Python và Jupyter.
2. Mở `notebooks/01_thu_thap_va_gan_nhan.ipynb`. Bấm **Select Kernel → Python
   Environments**, chọn Python trong `.venv` của đúng project.
3. **Restart Kernel → Run All**, hoặc chạy từ cell đầu xuống cuối bằng
   **Shift+Enter**. Chờ cell hết dấu đang chạy trước khi chuyển phần sau.
4. Mở `02_thuc_nghiem_va_demo.ipynb`, giữ kernel `.venv`, chạy từ đầu xuống.
   Cell fit mất vài phút. Chạy tiếp phần đánh giá, lưu và demo sau khi fit xong.
5. Mở `03_vibert.ipynb`, đổi kernel sang `.venv-modern`, giữ `MODE="review"`
   trong cell cấu hình, rồi Run All.
6. Mở `04_kiem_tra_grid_search.ipynb` nếu cần kiểm tra lưới; đổi lại kernel
   `.venv`, chạy từ đầu xuống.

Để kiểm tra interpreter, chạy cell tạm:

```python
import sys
print(sys.executable)
print(sys.version)
```

Cell gọi `fit`, `predict`, `to_csv`, `joblib.dump` có thực thi và ghi đầu ra.
Cell hiển thị trích mã nguồn chỉ dùng để đọc. Mở notebook xem output không tự
chạy lại. Không chạy từ cell giữa nếu chưa có biến do các cell trước tạo.

## 4. Đầu ra và cách dự đoán từ model đã lưu

| Bước | Đầu ra |
|---|---|
| 01 | `data/intermediate/`, `data/labeled/comments_labeled.csv`, `data/collection_manifest.json` |
| 02 | `data/processed/v7_preprocessed.csv`, `artifacts/spamham/metrics.csv`, `test_predictions.csv`, hình và `spam_ham_ml_only.joblib` |
| 03 review | Output notebook và `artifacts/modern/vibert/local_demo.csv`; checkpoint cũ được dùng để dự đoán |
| 04 | `artifacts/grid_audit/all_cv_configs.csv`, `best_by_group.csv` và các bảng đối chiếu |

Ví dụ dùng S3 ngay trong Terminal, không train lại:

```powershell
@'
import joblib
from pipeline.modeling.predict import predict_spam
bundle = joblib.load('artifacts/spamham/spam_ham_ml_only.joblib')
print(predict_spam(['Cho minh hoi lich thi', 'Nhan lam bai thue, lien he https://example.com'], bundle).to_string(index=False))
'@ | .\.venv\Scripts\python.exe -
```

Chuỗi ví dụ trong lệnh pipe dùng không dấu để tránh PowerShell cũ đổi mã chữ.
Trong notebook hoặc file `.py` lưu UTF-8 có thể nhập tiếng Việt có dấu.

## 5. Chuẩn bị môi trường gán nhãn mới

**Argilla gồm hai phần:** SDK Python để code kết nối và server/giao diện web
để chứa, duyệt nhãn. `pip install argilla` chỉ cài SDK, không mở server.
AI tạo gợi ý qua một API khác; Argilla không tự cung cấp mô hình AI.

Tạo môi trường riêng để không đổi thư viện của notebook:

```powershell
uv venv --python 3.12.14 --seed .venv-label
.\.venv-label\Scripts\python.exe -m pip install -r requirements/annotation.txt
.\.venv-label\Scripts\python.exe -m pip check
```

Nếu chưa có `uv` nhưng đã có Python 3.12: thay lệnh đầu bằng
`py -3.12 -m venv .venv-label`. SDK đang dùng là Argilla **2.8.0**;
không dùng hướng dẫn API của Argilla 1.x.

Đầu vào là CSV đã làm sạch, gồm `cleaned_text`, nên giữ `text`, metadata và
`id` nếu có. Nếu chưa có ID, các CLI dùng cùng tiền tố `cmt_` và cùng thứ tự
dòng để xác định ID. Giữ nguyên CSV suốt lượt gán nhãn; dùng cache riêng cho
từng lượt. Các lệnh ví dụ dùng `data/intermediate/new_run/comments_cleaned.csv`.
Nếu dùng CSV hiện tại, thay đường dẫn đó bằng `data/intermediate/comments_cleaned.csv`
và giữ output mới để dễ đối chiếu.

## 6. Cài và khởi động Argilla bằng Docker

Cài [Docker Desktop cho Windows](https://docs.docker.com/desktop/setup/install/windows-install/),
dùng backend WSL 2 theo hướng dẫn của bộ cài; khởi động Docker Desktop và chờ
engine sẵn sàng. Kiểm tra:

```powershell
docker version
docker compose version
```

Từ gốc project, tải Compose của repo Argilla phiên bản 2.8.0:

```powershell
New-Item -ItemType Directory -Force my-argilla | Out-Null
Invoke-WebRequest -Uri 'https://raw.githubusercontent.com/argilla-io/argilla/v2.8.0/examples/deployments/docker/docker-compose.yaml' -OutFile my-argilla/docker-compose.yaml
```

Mở `my-argilla/docker-compose.yaml`: đổi `WORKSPACE: default` thành
`WORKSPACE: argilla` để trùng cấu hình trong các script. Compose upstream tham
chiếu image server `latest`; tên tag của file Compose không khóa phiên bản image.
Giữ file Compose đã dùng cho lần cài để tiện kiểm tra khi nâng cấp.

```powershell
docker compose -f my-argilla/docker-compose.yaml up -d
docker compose -f my-argilla/docker-compose.yaml ps
docker compose -f my-argilla/docker-compose.yaml logs --tail 80 argilla
```

Mở `http://localhost:6900`. Với cấu hình mặc định của file vừa tải, tài khoản là
`argilla`, mật khẩu `12345678`, API key `argilla.apikey`. Đây là thông tin mặc
định cho máy local; nếu đã sửa Compose hoặc dùng server có sẵn, dùng thông tin
của server đó. Workspace cần có tên `argilla` và tài khoản có quyền truy cập.
Nếu server đã khởi tạo trước khi đổi `WORKSPACE`, tạo workspace `argilla`
trên giao diện hoặc bằng SDK, không xóa dữ liệu server cũ để tạo lại.

Thiết lập Terminal hiện tại và kiểm kết nối:

```powershell
$env:ARGILLA_API_URL='http://localhost:6900'
$env:ARGILLA_API_KEY='argilla.apikey'
@'
import os
import argilla as rg
client = rg.Argilla(api_url=os.environ['ARGILLA_API_URL'], api_key=os.environ['ARGILLA_API_KEY'])
print(client.me.username)
print(client.workspaces(name='argilla'))
'@ | .\.venv-label\Scripts\python.exe -
```

Nếu workspace trả `None`, tạo bằng giao diện trước khi upload. Dừng dịch vụ
bằng `docker compose -f my-argilla/docker-compose.yaml stop`; mở lại bằng
`up -d`. Không dùng `down -v` khi cần giữ nhãn đã lưu.

Nguồn triển khai: [hướng dẫn Docker chính thức](https://docs.argilla.io/latest/getting_started/how-to-deploy-argilla-with-docker/),
[Compose của tag v2.8.0](https://github.com/argilla-io/argilla/blob/v2.8.0/examples/deployments/docker/docker-compose.yaml).

## 7. Upload, tạo gợi ý, duyệt và xuất nhãn

Các lệnh sau dùng cùng dataset `comments_spamham_new`, cùng CSV và cùng cache.
Không dùng `--reset`/`--reset-dataset` cho dataset đang có nhãn cần giữ.

### 7.1. Đưa bình luận lên giao diện

```powershell
.\.venv-label\Scripts\python.exe -m pipeline.label.upload_to_argilla --csv data/intermediate/new_run/comments_cleaned.csv --dataset comments_spamham_new --api-url $env:ARGILLA_API_URL --api-key $env:ARGILLA_API_KEY
```

Mở dataset trên web, kiểm tra nội dung và câu hỏi ham/spam. Có thể tự gán nhãn
và Submit ngay mà không dùng AI; sau đó chuyển thẳng đến bước 7.4.

### 7.2. AI tạo gợi ý nhãn — tùy chọn

Chuẩn bị tài khoản API tương thích OpenAI, API key, model ID còn hoạt động và
hạn mức sử dụng. Tạo `.env` ở gốc project bằng VS Code, không đưa vào Git:

```dotenv
OPENROUTER_API_KEY=DIEN_KEY_CUA_BAN
OPENAI_BASE_URL=https://openrouter.ai/api/v1
MODEL_ID=DIEN_MODEL_ID_CON_DUOC_HO_TRO
```

Đây là chỗ điền cấu hình, không chạy nguyên các giá trị `DIEN_...`. API có thể
tính phí và nhận nội dung bình luận được gửi đi. Dùng thử ít mẫu trước:

```powershell
.\.venv-label\Scripts\python.exe -m pipeline.label.ai_assisted_spam_ham --csv data/intermediate/new_run/comments_cleaned.csv --cache data/intermediate/new_run/predictions.json --dataset comments_spamham_new --argilla-url $env:ARGILLA_API_URL --argilla-key $env:ARGILLA_API_KEY --limit 5 --workers 1
```

Kiểm cache và các suggestion trên giao diện. Khi đúng, chạy lại cùng lệnh nhưng
bỏ `--limit 5` để làm toàn bộ. Các ID đã có cache được dùng lại; không đổi nội
dung CSV mà vẫn giữ cache cũ. Gợi ý chưa phải nhãn đã được người duyệt xác nhận.

### 7.3. Duyệt nhãn

Cách ưu tiên để kiểm chất lượng: đọc từng bình luận, chọn ham/spam, **Submit**.
Nếu nhóm áp dụng tiếp nhận tự động ở confidence ≥ 0,75, chạy:

```powershell
.\.venv-label\Scripts\python.exe -m pipeline.label.auto_accept_labels --csv-in data/intermediate/new_run/comments_cleaned.csv --cache data/intermediate/new_run/predictions.json --dataset comments_spamham_new --threshold 0.75 --api-url $env:ARGILLA_API_URL --api-key $env:ARGILLA_API_KEY --export-csv data/intermediate/new_run/annotation_review.csv
```

Sau đó duyệt phần còn lại và kiểm tra mẫu đã tự tiếp nhận trên web. Confidence
là số AI trả về, chưa được hiệu chỉnh thành xác suất nhãn đúng. Mẫu chưa được
duyệt không tự xem là ham. 0,75 này là ngưỡng tiếp nhận nhãn, khác ngưỡng S3.

### 7.4. Tải CSV có nhãn đã submit

```powershell
.\.venv-label\Scripts\python.exe -m pipeline.label.download_labeled_data --dataset comments_spamham_new --csv-lookup data/intermediate/new_run/comments_cleaned.csv --output data/labeled/comments_new.csv --api-url $env:ARGILLA_API_URL --api-key $env:ARGILLA_API_KEY
```

Đầu ra là CSV chứa nhãn đã submit và metadata nối theo ID. Nó không tự trở
thành `labeled_v7.csv`; phải kiểm số dòng, phân bố nhãn và chất lượng trước
khi lựa chọn làm dữ liệu thực nghiệm mới.

## 8. Gán nhãn AI không mở server Argilla

Vẫn cài `requirements/annotation.txt` vì module AI import SDK khi khởi động,
nhưng không cần Docker/server. Sau khi điền `.env`, chạy:

```powershell
.\.venv-label\Scripts\python.exe -m pipeline.label.ai_assisted_spam_ham --csv data/intermediate/new_run/comments_cleaned.csv --cache data/intermediate/new_run/predictions.json --no-upload --limit 5 --workers 1
.\.venv-label\Scripts\python.exe -m pipeline.label.auto_accept_labels --csv-in data/intermediate/new_run/comments_cleaned.csv --cache data/intermediate/new_run/predictions.json --threshold 0.75 --skip-argilla --export-csv data/intermediate/new_run/annotation_review.csv
```

Lần thử chỉ có tối đa 5 gợi ý. Muốn xử lý toàn bộ, chạy lại AI bỏ `--limit 5`,
rồi chạy lại auto-accept. File review giữ cả dòng chưa có nhãn; lọc
`official_label` là `ham`/`spam`, đổi thành `label` như mục 5.3 README trước khi
dùng cho bước sau. Không dùng thẳng toàn bộ bảng review làm dataset đã chốt.

## 9. Lỗi thường gặp

| Hiện tượng | Cách xử lý |
|---|---|
| `No module named argilla` | Chạy bằng `.venv-label` sau khi cài annotation.txt |
| Docker daemon không chạy | Mở Docker Desktop, chờ engine sẵn sàng |
| Không mở được cổng 6900 | Xem `compose ps` và logs; kiểm cổng đang bị ứng dụng khác dùng |
| Argilla 401/403 | Dùng key thực của tài khoản và quyền workspace; mặc định code `admin.apikey` không trùng key Compose ở hướng dẫn này |
| Không tìm thấy dataset | Đồng bộ tên dataset, URL và workspace `argilla` ở mọi bước |
| API AI 401/402/429 hoặc model không tồn tại | Kiểm key, hạn mức, rate limit và model ID của nhà cung cấp |
| Có suggestion nhưng CSV xuất thiếu dòng | Chỉ response đã Submit mới được xuất làm nhãn chính thức |
| `NameError` trong notebook | Chọn đúng kernel và Restart Kernel → Run All |
| Thiếu dữ liệu/checkpoint | Nhận thêm các đầu vào local ở mục 1; Git không chứa data/model |

Hướng dẫn này không có nghĩa server Argilla hoặc API AI đã được gọi lại trong
lượt kiểm tra offline. Các bước cấp quyền, API key và khởi động Docker do bạn
thực hiện khi muốn tạo nhãn mới.
