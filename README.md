# Phân loại Spam/Ham bình luận sinh viên PTIT

Project của nhóm Bảo, Cương, Phi, Phương: chuẩn bị dữ liệu bình luận Facebook,
thực nghiệm BoW/TF-IDF/FastText với Naive Bayes/SVM, đánh giá và dự đoán bình luận mới.
Phần mô hình hiện đại được dành để triển khai tiếp theo.

- [Tổng quan cấu trúc, module và luồng xử lý](docs/guides/TONG_QUAN_PROJECT_VA_CACH_THUC_THI.md).
- [Notebook 01 — Thu thập và chuẩn bị dữ liệu](notebooks/01_thu_thap_va_gan_nhan.ipynb).
- [Notebook 02 — Thực nghiệm và ứng dụng](notebooks/02_thuc_nghiem_va_demo.ipynb).
- [Lộ trình mở mã nguồn khi trình bày](docs/presentation/LO_TRINH_TRINH_BAY.md).

## 1. Chuẩn bị môi trường

Mở **toàn bộ thư mục project** bằng VS Code. Terminal PowerShell phải đứng tại
thư mục chứa README này, `pipeline/`, `notebooks/` và `requirements/`.
Sử dụng Python **3.12**; môi trường đã kiểm tra dùng Python **3.12.14**.

Kiểm tra các Python đang có:

```powershell
py -0p
```

Nếu đã có `.venv`, giữ môi trường đó và chuyển đến bước kiểm tra bên dưới.
Nếu cần tạo môi trường mới, cách phù hợp với máy hiện tại (đã có `uv`) là:

```powershell
uv venv --python 3.12.14 --seed .venv
```

Trên máy đã cài Python 3.12 qua bộ cài thông thường, cũng có thể dùng
`py -3.12 -m venv .venv`. Máy hiện tại có Python 3.14 trên PATH và Python 3.12
do uv quản lý; gọi `python` trần hoặc `py -3.12` có thể chọn sai/không tìm được
runtime. Các lệnh sau dùng trực tiếp Python trong `.venv`.

Cài và kiểm tra thư viện bằng chính Python của project:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements/notebook.txt
.\.venv\Scripts\python.exe scripts/check_environment.py
.\.venv\Scripts\python.exe -m pip check
```

Tiếp tục khi checker báo **Environment OK** và pip báo **No broken requirements found**.
Các lệnh dưới đây dùng đường dẫn `.venv` trực tiếp nên không cần kích hoạt môi trường
hoặc đổi Execution Policy. Giữ các phiên bản đã pin trong requirements.

## 2. Chuẩn bị đầu vào

```powershell
.\.venv\Scripts\python.exe -m pipeline.collect.snapshots --include-phase1
```

Lệnh kiểm tra/tải ba đầu vào tái lập vào `.cache/collection/` và chuẩn bị CSV có
nhãn có sẵn tại `data/labeled/comments_labeled_phase1.csv`, có đối chiếu SHA-256.
Lần đầu cần mạng nếu chưa có file; các lần sau dùng lại file đúng hash.
Notebook 02 dùng `data/final/labeled_v7.csv` nếu có; nếu thiếu sẽ tải revision/hash
đã xác định từ Hugging Face. Notebook 01 tái lập từ JSON và gợi ý nhãn đã lưu,
không yêu cầu đăng nhập Facebook hoặc gọi API gán nhãn.

## 3. Chạy project từ Terminal

Thực hiện lần lượt:

```powershell
.\.venv\Scripts\python.exe scripts/run_notebook.py --notebook 01 --inplace
.\.venv\Scripts\python.exe -m tools.dataset_versions.merge_labeled_data
.\.venv\Scripts\python.exe scripts/run_notebook.py --notebook 02 --inplace
```

| Lệnh | Kết quả |
|---|---|
| Notebook 01 | Các bảng trung gian trong `data/intermediate/`; CSV cuối tại `data/labeled/comments_labeled.csv`. |
| Tool gộp | Đọc CSV có sẵn và CSV notebook 01 vừa tạo; ghi `data/final/comments_merged.csv`. |
| Notebook 02 | Huấn luyện/đánh giá trên v7; lưu model và chỉ số vào `artifacts/spamham/`. |

`--inplace` chỉ cập nhật output notebook gốc khi tất cả cell thành công. Kernel,
bản sao notebook và trạng thái tạm nằm trong `.work/runs/01/` hoặc `02/`;
đầu ra dữ liệu/model nằm ở các thư mục trong bảng. Huấn luyện nhiều cấu hình
cần thời gian; lần kiểm tra trên máy hiện tại khoảng 24 phút.
Runner cố định hash seed và số luồng BLAS trước khi mở kernel; dùng runner khi
cần đối chiếu kết quả tái lập giữa các lần chạy.

V7 là đầu vào mặc định của thực nghiệm. Tool gộp tạo file riêng và giữ nguyên v7.
Các module trong `pipeline/` được notebook gọi theo từng bước.

## 4. Chạy notebook bằng VS Code

1. Cài/bật extension **Python** và **Jupyter**, mở project bằng **File → Open Folder**.
2. Chọn **Python: Select Interpreter → Enter interpreter path** và chỉ tới
   `<project>\.venv\Scripts\python.exe`.
3. Mở notebook, bấm **Select Kernel → Python Environments**, chọn đúng interpreter
   `.venv` của project này. Kiểm tra đường dẫn, không chỉ dựa vào tên hiển thị.
4. Chọn **Restart Kernel → Run All**. Chờ từng notebook hoàn tất.
5. Sau notebook 01, chạy lệnh tool gộp ở mục 3 nếu cần tạo lại CSV gộp.

Có thể kiểm tra kernel trong một cell tạm, rồi xóa cell này sau khi kiểm tra:

```python
import sys
print(sys.executable)
print(sys.version)
```

Đường dẫn phải thuộc `.venv` của project, phiên bản Python là 3.12.
Notebook dùng chung module trong repo; để chạy lại cần giữ cấu trúc thư mục đi kèm.

## 5. Dữ liệu và kết quả

```text
data/
  intermediate/                  Bảng chuyển đổi, làm sạch, gợi ý và kết quả tiếp nhận nhãn
  labeled/
    comments_labeled.csv         CSV cuối của notebook 01
    comments_labeled_phase1.csv   CSV có nhãn có sẵn
  final/
    comments_merged.csv           CSV do tool gộp tạo
    labeled_v7.csv               Đầu vào thực nghiệm nếu đã lưu local
  collection_manifest.json       Số lượng và hash của lượt xử lý notebook 01
artifacts/spamham/                Model .joblib, metrics, bootstrap CI, run manifest
.cache/collection/                Ba đầu vào cần để tái lập notebook 01
results/verified_run/             Bằng chứng kiểm tra hiện hành
```

`data/`, `artifacts/`, `.cache/`, `.work/` được Git bỏ qua. Khi chia sẻ để chạy
offline, gửi thêm dữ liệu cần thiết; chỉ pull code chưa đồng nghĩa có dữ liệu/model.
Tài liệu cũ ở `docs/archive/`, bằng chứng cũ ở `results/history/`.

## 6. Kiểm thử và xử lý lỗi thường gặp

Cài thêm dependency kiểm thử khi cần:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements/dev.txt
.\.venv\Scripts\python.exe -m pytest --basetemp .work/pytest -p no:cacheprovider -q
```

| Hiện tượng | Cách xử lý |
|---|---|
| Thiếu thư viện/sai phiên bản | Cài lại `requirements/notebook.txt` bằng `.venv`, chạy checker và restart kernel. |
| `No module named pipeline` | Mở notebook trong toàn bộ project và chọn đúng kernel. |
| `NameError` sau khi chạy một cell giữa notebook | Restart Kernel → Run All theo thứ tự. |
| Không tải được dữ liệu | Chuẩn bị đầu vào khi có mạng; giữ `.cache/collection/` nếu cần chạy offline. |
| SHA-256 không khớp | Kiểm tra đúng nguồn/file, giữ bước kiểm hash để tránh dùng nhầm dữ liệu. |
| Output lưu sai vị trí | Kiểm tra các biến tùy chỉnh `SPAMHAM_OUTPUT_DIR`, `SPAMHAM_SOURCE_DIR`, `SPAMHAM_CSV`; restart kernel sau khi đổi cấu hình. |

Kiểm thử đóng kernel trên Windows có ghi nhận thông báo `resource_tracker.KeyError`
của joblib ở bước dọn bộ nhớ tạm. Trong kiểm thử này, worker vẫn thoát và thư mục
tạm được giải phóng; không có cell tính toán thất bại. Bằng chứng được lưu tại
`results/verified_run/kernel_shutdown_check.json`. Các phiên bản thư viện được giữ
nguyên để bảo toàn cấu hình thực nghiệm đã kiểm chứng.

Để thu thập/gán nhãn qua dịch vụ, cài thêm `requirements/collection.txt` hoặc
`requirements/annotation.txt`, cấu hình phiên/API trong môi trường và xem
`--help` của module tương ứng. Các thao tác này có kết nối dịch vụ; hai notebook
không tự gọi lại crawler hoặc LLM. `.env` thật và token/cookie được giữ ngoài Git.
