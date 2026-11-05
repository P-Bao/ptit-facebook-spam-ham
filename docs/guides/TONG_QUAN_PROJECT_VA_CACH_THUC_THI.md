# Tổng quan project và luồng thực hiện

## Mục tiêu

Phân loại bình luận Facebook của sinh viên PTIT thành **ham** (bình luận thông
thường) và **spam** (nội dung rác hoặc chào mời). Project trình bày quá trình chuẩn
bị dữ liệu, xử lý tiếng Việt, biểu diễn văn bản, huấn luyện, đánh giá và ứng dụng
mô hình cho bình luận mới. Nhóm gồm Bảo, Cương, Phi, Phương.

## Hai notebook

| Notebook | Nội dung | Kết quả chính |
|---|---|---|
| [01_thu_thap_va_gan_nhan.ipynb](../../notebooks/01_thu_thap_va_gan_nhan.ipynb) | Nguồn thu thập → bài/comment/reply → JSON thành bảng → làm sạch → gán nhãn → xuất CSV | `comments_labeled.csv`, thống kê và ví dụ dữ liệu |
| [02_thuc_nghiem_va_demo.ipynb](../../notebooks/02_thuc_nghiem_va_demo.ipynb) | Khảo sát v7 → tiền xử lý → chia tập/CV → đặc trưng và mô hình → đánh giá → dự đoán | Model `.joblib`, bảng chỉ số, biểu đồ, phân tích lỗi và demo |

Notebook 01 tái lập xử lý từ kết quả thu thập và phản hồi gán nhãn đã lưu, không
gọi Facebook/LLM/Argilla. Notebook 02 thực hiện huấn luyện khi chạy toàn bộ cell.
Các khối “Mã nguồn” chỉ đọc và hiển thị hàm; các cell xử lý dữ liệu, gọi CLI và
huấn luyện mới thực thi công việc tương ứng.

## Cấu trúc thư mục và vai trò các file

| Thư mục | File chính và nhiệm vụ |
|---|---|
| `notebooks/` | Hai notebook tổ chức quá trình trình bày, gọi module và hiển thị kết quả. |
| `external/facebook_scraper/` | Bộ mã nền thu thập: `group_post_scraper_v2.py` lấy bài nhóm; `main.py` điều phối lấy comment/reply; `comment_scraper.py` xử lý bình luận. `post_scraper.py`, `single_post_image.py` hỗ trợ các nhánh lấy bài/ảnh; `proxy_utils.py` hỗ trợ kết nối; `facebook_ui.py` là GUI tùy chọn. README, requirements và `.env.example` hướng dẫn bộ công cụ. |
| `pipeline/collect/` | `scrape_ptit_groups.py`: runner PTIT, cấu hình nhóm, gọi bộ nền và ghi JSON/state. `json_to_csv.py`: làm phẳng comment/reply thành bảng. `snapshots.py`: tìm/tải đầu vào và kiểm SHA-256. |
| `pipeline/prepare/` | `clean_for_labeling.py`: làm sạch bảng, khử trùng/lọc và giữ text gốc. `preprocess_for_model.py`: chuẩn hóa đầu vào mô hình, xử lý tên/URL/điện thoại/emoji, tách từ và stop words. |
| `pipeline/label/` | `upload_to_argilla.py`: đưa record lên giao diện nhãn. `ai_assisted_spam_ham.py`: tạo gợi ý label/confidence. `auto_accept_labels.py`: tiếp nhận gợi ý theo ngưỡng. `download_labeled_data.py`: xuất nhãn đã duyệt. `label_utils.py`: kiểm ID/nhãn và metadata. `cached_annotations.py`: nối gợi ý với đúng bình luận theo nội dung, không dựa vào vị trí dòng. |
| `pipeline/features/` | `spam_features.py`: FastTextVectorizer, học embedding và tổng hợp vector câu. BoW/TF-IDF sử dụng vectorizer của scikit-learn trong `modeling/`. |
| `pipeline/modeling/` | `experiments.py`: tạo pipeline BoW/TF-IDF/FastText + NB/SVM, tìm tham số và đánh giá. `training_utils.py`: chia nhóm/fold, sampling trong train, chỉnh ngưỡng và hỗ trợ dự đoán. |
| `pipeline/inference/` | `predict.py`: nhận bình luận mới và bundle đã huấn luyện, tiền xử lý nhất quán rồi trả nhãn/score. |
| `pipeline/presentation/` | `notebook_display.py`: bảng, mã nguồn và thông tin file có màu tương phản; không thay đổi dữ liệu hoặc dự đoán. |
| `pipeline/modern/` | Vị trí dành cho mã mô hình hiện đại; hiện chưa triển khai. |
| `tools/dataset_versions/` | Công cụ quản lý/kiểm tra dữ liệu nội bộ; không thuộc luồng trình bày hai notebook. |
| `scripts/` | `run_notebook.py`: chạy notebook bằng kernel mới, lưu output, thời gian và hash mã đã thực thi. |
| `requirements/` | `notebook.txt`: notebook; `collection.txt`: crawler; `annotation.txt`: công cụ nhãn; `dev.txt`: kiểm thử. |
| `tests/` | Các file `test_*.py` kiểm tra xử lý dữ liệu, ID/nhãn, liên kết cache, tiền xử lý, chia tập, huấn luyện–lưu–tải–dự đoán và hiển thị notebook. |
| `data/` | Đầu vào local: `raw/` chứa JSON; `intermediate/` chứa bảng/cache; `labeled/` chứa dữ liệu nhãn; `final/labeled_v7.csv` dùng thực nghiệm. Không đưa dữ liệu vào Git. |
| `artifacts/`, `.work/` | CSV, mô hình, cache và output sinh khi chạy; không đưa vào Git. `.venv/` chứa môi trường Python. |
| `results/` | `verified_run/`: bằng chứng lần chạy hiện tại. `final_editorial/`: kiểm tra lần hoàn thiện notebook. Các thư mục còn lại lưu kiểm chứng nguồn, hiển thị và lịch sử. |
| `docs/` | `assignment/`: ảnh đề; `team/`: phân công; `guides/`: hướng dẫn; `presentation/`: lộ trình mở code; `provenance/`: hồ sơ nguồn; `plans/`, `overview/`, `changes/`: kế hoạch và lịch sử; `report/`: dành cho báo cáo. |

Các file `__init__.py` tạo package để import. `.gitignore` loại dữ liệu, môi trường,
model và cache khỏi Git; `.env.example` là mẫu cấu hình, không chứa phiên đăng nhập.

## Luồng thực hiện

1. **Thu thập:** `scrape_ptit_groups.py` nhận cấu hình nhóm/phiên, gọi
   `group_post_scraper_v2.py` lấy bài, `main.py` và `comment_scraper.py` lấy comment/reply.
   Writer lưu metadata và nội dung vào `data/raw/posts.json` khi chạy crawler.
2. **Tạo bảng:** `json_to_csv.py` chuyển cấu trúc JSON thành từng dòng bình luận,
   giữ post/group/thời gian/loại nội dung → `comments.csv`.
3. **Làm sạch:** `clean_for_labeling.py` chuẩn hóa và lọc dữ liệu, giữ `text`, thêm
   `cleaned_text` → `comments_cleaned.csv`. Đây là chuẩn bị dữ liệu để gán nhãn.
4. **Gán nhãn:** công cụ AI tạo gợi ý; `auto_accept_labels.py` áp ngưỡng hoặc
   người duyệt trên Argilla. Xuất các mẫu đủ điều kiện → `comments_labeled.csv`.
   Notebook sử dụng gợi ý đã lưu, liên kết bằng `cached_annotations.py` trước khi tiếp nhận.
5. **Chuẩn bị thực nghiệm:** notebook 02 đọc `data/final/labeled_v7.csv`, kiểm
   cấu trúc/nhãn, gọi `preprocess_for_model.py` để tách từ và tạo hai cấu hình
   giữ/bỏ stop words. Bước này chuẩn bị đầu vào cho mô hình, không gán lại nhãn.
6. **Huấn luyện:** chia train/test theo khóa văn bản, CV 5 fold trên train;
   `experiments.py` huấn luyện 12 tổ hợp chính và các đối chứng cân bằng lớp/ngưỡng.
   Vectorizer, IDF và FastText chỉ được học trong phần train tương ứng.
7. **Đánh giá:** chọn cấu hình theo CV, đo trên test; báo cáo Accuracy,
   Precision/Recall/F1 spam, AP, confusion matrix, so sánh stop words và phân tích FP/FN.
8. **Lưu và ứng dụng:** notebook lưu `spam_ham_ml_only.joblib`; `predict.py` tải
   bundle, áp dụng cùng tiền xử lý và dự đoán bình luận mới. Kiểm tra tải lại không đổi kết quả.

## Cách chạy và đầu ra

Từ thư mục gốc project, tạo môi trường Python 3.12 và cài thư viện:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements/notebook.txt
```

Mở notebook trong repo, chọn kernel `.venv`, thực hiện **Restart Kernel → Run All**.
Notebook cần các module đi kèm trong repo. Dữ liệu thiếu được tải theo manifest
hoặc revision/hash; nếu chạy offline phải chuẩn bị dữ liệu trước.

```powershell
.\.venv\Scripts\python.exe scripts/run_notebook.py --notebook 01 --work-dir .work/runs/collection --inplace
.\.venv\Scripts\python.exe scripts/run_notebook.py --notebook 02 --work-dir .work/runs/training --inplace
```

Runner ghi vào `<work-dir>/artifacts/`, chỉ cập nhật output notebook gốc khi chạy
thành công. Mở trực tiếp notebook dùng mặc định `artifacts/collection/` cho 01 và
`artifacts/spamham/` cho 02; có thể cấu hình `SPAMHAM_OUTPUT_DIR`.

| Notebook | File sinh ra |
|---|---|
| 01 | `comments.csv`, `comments_cleaned.csv`, `annotation_predictions.json`, `annotation_review.csv` là trung gian; `comments_labeled.csv` là tập có nhãn cuối; `collection_manifest.json` ghi số liệu/hash. |
| 02 | `spam_ham_ml_only.joblib`: mô hình; `metrics.csv`: bảng chỉ số; `bootstrap_ci.csv`: khoảng tin cậy; `run_manifest.json`: cấu hình, dữ liệu và kết quả lựa chọn. |

Chạy crawler cần thêm `requirements/collection.txt` và phiên Facebook hợp lệ;
gọi công cụ nhãn qua dịch vụ cần `requirements/annotation.txt` cùng cấu hình tương ứng.
Không lưu cookie/token vào notebook hoặc Git. Hai notebook không tự chạy lại mọi
file `.py`; mỗi cell chỉ gọi những hàm/CLI được viết trong cell đó.

## Kết quả và giới hạn

- Notebook 01: 5 nhóm, 2.170 bài, 13.833 comment/reply; 13.704 nội dung không rỗng,
  10.974 mẫu sau làm sạch. Có 6.571 gợi ý khớp; ngưỡng 0,75 tiếp nhận 6.289 mẫu
  (5.190 ham, 1.099 spam). Mẫu thiếu gợi ý hoặc chưa đạt ngưỡng không nằm trong CSV có nhãn.
- Notebook 02: v7 có 38.233 mẫu; train 30.759, test 7.474. TF-IDF + SVM giữ stop words
  được chọn theo CV. Test: Accuracy **94,19%**, Precision spam **88,18%**,
  Recall spam **62,78%**, F1 spam **0,7334**, AP **0,8117**.
- Độ chính xác tổng thể cần đọc cùng Recall/F1 vì dữ liệu lệch lớp. Mô hình còn
  bỏ sót spam và có lỗi ở bình luận ngắn/phụ thuộc ngữ cảnh. Demo là ví dụ ứng dụng,
  không thay thế tập test. Phần mô hình hiện đại vẫn đang chờ triển khai.

## Mã mô hình và mô hình đã huấn luyện

| Thành phần | Ý nghĩa |
|---|---|
| `.py` trong `pipeline/modeling/` | Mã định nghĩa, cấu hình và hàm huấn luyện; bản thân file không chứa trọng số học được. |
| Đối tượng sau `.fit()` | Mô hình đã học nằm trong bộ nhớ, gồm vectorizer và tham số/trọng số. |
| `spam_ham_ml_only.joblib` trong thư mục đầu ra | Bản lưu của mô hình đã học, kèm cấu hình để tải lại và dự đoán. |
| `pipeline/modern/` | Mã cho phương pháp hiện đại sẽ viết sau; checkpoint được lưu trong thư mục đầu ra khi triển khai. |

Yêu cầu xây dựng và huấn luyện được thể hiện bằng mã cùng kết quả thực thi,
không phụ thuộc tên thư mục hay đuôi file. Cả mô hình cơ sở và mô hình hiện đại
đều có mã xây dựng, bước học và trạng thái đã học cần lưu nếu muốn tái sử dụng.

Theo phạm vi đã thống nhất, báo cáo là tài liệu nộp; notebook và các module dùng
để trình bày/demo. Để người khác chạy lại, cần repo, môi trường và dữ liệu tương ứng.
