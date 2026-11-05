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

## 2. Đặt dữ liệu local

Notebook đọc trực tiếp các file dưới đây; không tải dữ liệu hoặc kiểm checksum.

| Bước | File cần có |
|---|---|
| Notebook 02 | `data/final/labeled_v7.csv` |
| Notebook 01 | `.cache/collection/posts.json`, `annotation_index.csv`, `annotation_predictions.json` |
| Tool gộp | `data/labeled/comments_labeled_phase1.csv` và CSV notebook 01 xuất ra |

Các file này đã có trên máy hiện tại. Khi chuyển sang máy khác, sao chép dữ liệu
cùng đúng cấu trúc thư mục. Notebook 01 dùng JSON thu thập và gợi ý nhãn đã lưu;
không đăng nhập Facebook hoặc gọi API gán nhãn trong chế độ trình bày này.

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
| Notebook 02 | Đọc v7, lưu văn bản đã tiền xử lý tại `data/processed/v7_preprocessed.csv`, huấn luyện/đánh giá và lưu model vào `artifacts/spamham/`. |

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
  processed/
    v7_preprocessed.csv          Toàn bộ v7 kèm hai cột token dùng cho huấn luyện
  final/
    comments_merged.csv           CSV do tool gộp tạo
    labeled_v7.csv               Đầu vào thực nghiệm local
  collection_manifest.json       Số lượng, nhãn và đường dẫn đầu ra notebook 01
artifacts/spamham/                Model .joblib, metrics, bootstrap CI, run manifest
.cache/collection/                Ba đầu vào local của notebook 01
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
| Không tìm thấy CSV/JSON | Đặt đúng file local theo mục 2, kiểm tra đường dẫn rồi chạy lại. |
| Output lưu sai vị trí | Kiểm tra các biến tùy chỉnh `SPAMHAM_OUTPUT_DIR`, `SPAMHAM_SOURCE_DIR`, `SPAMHAM_CSV`; restart kernel sau khi đổi cấu hình. |

Khi đóng kernel trên Windows, joblib có ghi thông báo `resource_tracker.KeyError`
ở bước dọn bộ nhớ tạm. Thông báo cũng xuất hiện sau lượt chạy đầy đủ hiện tại;
các cell tính toán hoàn tất, worker thoát và kết quả lưu được kiểm tra lại.
Trạng thái chạy nằm trong `results/verified_run/verification.json`. Các phiên bản
thư viện được giữ nguyên để bảo toàn cấu hình thực nghiệm đã kiểm chứng.

Để thu thập/gán nhãn qua dịch vụ, cài thêm `requirements/collection.txt` hoặc
`requirements/annotation.txt`, cấu hình phiên/API trong môi trường và xem
`--help` của module tương ứng. Các thao tác này có kết nối dịch vụ; hai notebook
không tự gọi lại crawler hoặc LLM. `.env` thật và token/cookie được giữ ngoài Git.

## 7. Luồng notebook, module và các thư mục hỗ trợ

Notebook tổ chức thứ tự xử lý, gọi hàm trong `pipeline/` và trình bày dữ liệu/bảng/hình.
Cell xử lý và huấn luyện có thực thi; khối hiển thị mã nguồn chỉ đọc hàm để giải thích.
Lệnh `scripts/run_notebook.py` thực hiện toàn bộ các cell bằng Python của `.venv`.

Các module có CLI như thu thập, chuyển CSV, làm sạch, gán nhãn và tool gộp có thể
chạy bằng `python -m <module>` trong đúng môi trường. `experiments.py`,
`training_utils.py` và `preprocess_for_model.py` chủ yếu cung cấp hàm: chạy riêng
file không tự dựng đủ dữ liệu/split rồi huấn luyện tất cả mô hình. Notebook 02
hoặc runner là nơi điều phối thí nghiệm đầy đủ.

| Thư mục | Vai trò | Có cần cho huấn luyện/dự đoán? |
|---|---|---|
| `tests/` | Mã kiểm thử và các tình huống giả lập nhỏ để kiểm tra dữ liệu, tiền xử lý, CV/sampling, lưu/tải model, hiển thị và runner. | Chạy riêng để kiểm tra chức năng; không thay thế đánh giá trên v7. |
| `results/verified_run/` | Bảng chỉ số và biên bản kiểm tra lần chạy hiện hành. | Dùng đối chiếu/giải thích kết quả. |
| `results/source_audit/` | Hồ sơ đối chiếu nguồn trước đây. | Không được notebook đọc để nạp dữ liệu. |
| `results/history/` | Lưu bằng chứng các phiên bản cũ. | Dùng tra cứu lịch sử. |
| `artifacts/spamham/` | Mô hình đã huấn luyện và báo cáo được notebook 02 sinh ra. | Bundle `.joblib` dùng để dự đoán lại. |

Mất cân bằng lớp được khảo sát ở mục 5.1 notebook 02: SVM thử trọng số lớp,
Naive Bayes thử ngưỡng và có nhánh under-sampling. Chọn mô hình bằng F1 spam trên
CV; validation/test giữ tỷ lệ nhãn tự nhiên. Chi tiết trong tài liệu tổng quan.
