# Phân loại Spam/Ham bình luận sinh viên PTIT

Project của nhóm Bảo, Cương, Phi, Phương: chuẩn bị dữ liệu bình luận Facebook,
thực nghiệm BoW/TF-IDF/FastText với Naive Bayes/SVM, đánh giá và dự đoán bình luận mới.
Nhánh mô hình hiện đại fine-tune viBERT và đối chiếu với S3 trên cùng test.

- [Tổng quan cấu trúc, module và luồng xử lý](docs/guides/TONG_QUAN_PROJECT_VA_CACH_THUC_THI.md).
- [Notebook 01 — Thu thập và chuẩn bị dữ liệu](notebooks/01_thu_thap_va_gan_nhan.ipynb).
- [Notebook 02 — Thực nghiệm và ứng dụng](notebooks/02_thuc_nghiem_va_demo.ipynb).
- [Notebook 03 — Fine-tune và ứng dụng viBERT](notebooks/03_vibert.ipynb).
- [Huấn luyện viBERT trên Colab, môi trường và đầu ra](docs/guides/HUONG_DAN_VIBERT_COLAB.md).
- [Lộ trình mở mã nguồn khi trình bày](docs/presentation/LO_TRINH_TRINH_BAY.md).
- [Các thay đổi của notebook và ngưỡng SVM](docs/guides/CAP_NHAT_NOTEBOOK_VA_NGUONG_SVM.md).

## 1. Chuẩn bị môi trường

Các lệnh `.venv` ở mục 1–4 dành cho Notebook 01/02. Notebook 03 dùng
`.venv-modern` để review/demo local hoặc môi trường GPU Colab riêng theo hướng
dẫn viBERT ở trên; không cài PyTorch/Transformers vào `.venv` truyền thống.

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
cần thời gian; lần kiểm tra sau cập nhật trên máy hiện tại khoảng 10 phút.
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

Để kiểm tra lại toàn bộ kết quả, dùng **Restart Kernel → Run All** hoặc runner
ở mục 3. Chạy riêng một cell giữa notebook yêu cầu các biến của phần trước đã
được tạo; sau khi huấn luyện lại cần chạy tiếp đánh giá và lưu mô hình để tránh
hiển thị kết quả cũ. Cùng dữ liệu, phiên bản và cấu hình seed có thể tái lập các
chỉ số; thời gian chạy thay đổi. Runner còn cố định hash seed và số luồng trước
khi mở Python, nên dùng cách này khi cần đối chiếu chính xác cả FastText.

Cell lưu mô hình ở mục 8 notebook 02 có gọi `joblib.dump()`: chạy cell sẽ ghi
lại `artifacts/spamham/spam_ham_ml_only.joblib`, nạp lại và kiểm tra dự đoán.
Chỉ mở notebook hoặc xem output cũ không huấn luyện và không ghi lại model.

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
artifacts/modern/vibert/          Model/tokenizer viBERT, lịch sử, metrics và dự đoán test
.cache/collection/                Ba đầu vào local của notebook 01
results/verified_run/             Bằng chứng kiểm tra hiện hành
results/modern/                   Bảng so sánh viBERT–S3 và bằng chứng kiểm tra viBERT
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
`--help` của module tương ứng. Các thao tác này có kết nối dịch vụ; Notebook 01/02
không tự gọi lại crawler hoặc LLM. `.env` thật và token/cookie được giữ ngoài Git.

## 7. Luồng notebook, module và các thư mục hỗ trợ

Notebook tổ chức thứ tự xử lý, gọi hàm trong `pipeline/` và trình bày dữ liệu/bảng/hình.
Cell xử lý và huấn luyện có thực thi; khối hiển thị mã nguồn chỉ đọc hàm để giải thích.
Lệnh `scripts/run_notebook.py` thực hiện toàn bộ cell bằng Python dùng để gọi
lệnh: `.venv` cho 01/02, `.venv-modern` cho 03.

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

Mục 4 notebook 02 giải thích train, validation, CV và test. Mục 5 đối chiếu từ
dừng bằng CV, rồi trình bày sáu tổ hợp BoW/TF-IDF/FastText + NB/SVM ở cấu hình
tiền xử lý được chọn. SVM cơ sở không dùng trọng số lớp; không lấy mẫu lại.

Mục 5.1 cố định **TF-IDF + SVM**, so sánh bốn cấu hình: không điều chỉnh,
trọng số balanced, under-sampling 2:1 và điều chỉnh ngưỡng decision score bằng
OOF trên train. CV Precision/Recall/F1 phục vụ lựa chọn; test không dùng để chọn
tham số hoặc ngưỡng. Mục 6 báo cáo sáu tổ hợp cơ sở; 6.1 đối chiếu bốn chiến lược
trên test cùng FP/FN. Mục 6.2–6.3 có ma trận nhầm lẫn và Precision–Recall cho cả
hai phạm vi. Cấu hình chỉ đổi ngưỡng dùng chung đường PR với mô hình đối chứng,
nhưng có điểm Precision/Recall khác.

Không có phụ lục từ dừng hoặc nhánh chỉnh ngưỡng NB trong notebook hiện tại.
Mô hình được chọn theo CV trong phạm vi sáu cấu hình cơ sở và ba cấu hình bổ sung,
không khẳng định tối ưu trong mọi kết hợp chiến lược. Bundle dùng đúng ngưỡng
trong đánh giá, nạp lại và demo.

Trong `artifacts/spamham/`, ngoài model và metrics còn có `stopword_cv.csv`,
`imbalance_cv.csv`, `imbalance_comparison.csv`, `test_predictions.csv`;
`imbalance_training.png`, `imbalance_results.png`, `confusion_baselines.png`,
`confusion_strategies.png`, `pr_baselines.png`, `pr_strategies.png`.


## 8. viBERT và kết quả mô hình hiện đại

Đã fine-tune viBERT trên Colab A100 80 GB, chọn loss thường/3 epoch bằng
validation rồi fit toàn bộ train. Trên cùng 7.474 test: Accuracy **96,35%**,
Precision **89,42%**, Recall **80,86%**, F1 **0,8493**, AP **0,9108**;
S3 có F1 **0,7552**. Kết quả và giới hạn được trình bày trong Notebook 03.

Notebook 02 mục 5.2 hiển thị mã viBERT, mục 6.4 đọc bảng so sánh test. Notebook
03 giải thích toàn bộ fine-tune và demo; dùng `.venv-modern`, không dùng `.venv`
truyền thống. Runner hỗ trợ `--notebook 03`, mặc định review model đã tải.
Model/tokenizer local: `artifacts/modern/vibert/model/`; bảng/evidence Git:
`results/modern/`. Chỉ pull Git chưa có trọng số và dữ liệu để chạy demo.
