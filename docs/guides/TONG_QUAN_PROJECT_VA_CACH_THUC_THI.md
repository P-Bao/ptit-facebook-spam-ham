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
| `pipeline/collect/` | `scrape_ptit_groups.py`: runner PTIT, cấu hình nhóm, gọi bộ nền và ghi JSON/state. `json_to_csv.py`: làm phẳng comment/reply thành bảng. Notebook 01 đọc trực tiếp các file local trong `.cache/collection/`. |
| `pipeline/prepare/` | `clean_for_labeling.py`: làm sạch bảng, khử trùng/lọc và giữ text gốc. `preprocess_for_model.py`: chuẩn hóa đầu vào mô hình, xử lý tên/URL/điện thoại/emoji, tách từ và stop words. |
| `pipeline/label/` | `upload_to_argilla.py`: đưa record lên giao diện nhãn. `ai_assisted_spam_ham.py`: tạo gợi ý label/confidence. `auto_accept_labels.py`: tiếp nhận gợi ý theo ngưỡng. `download_labeled_data.py`: xuất nhãn đã duyệt. `label_utils.py`: kiểm ID/nhãn và metadata. `cached_annotations.py`: nối gợi ý với đúng bình luận theo nội dung, không dựa vào vị trí dòng. |
| `pipeline/features/` | `spam_features.py`: FastTextVectorizer, học embedding và tổng hợp vector câu. BoW/TF-IDF sử dụng vectorizer của scikit-learn trong `modeling/`. |
| `pipeline/modeling/` | `experiments.py`: tạo pipeline BoW/TF-IDF/FastText + NB/SVM, tìm tham số và đánh giá. `training_utils.py`: chia nhóm/fold, sampling trong train, chỉnh ngưỡng và hỗ trợ dự đoán. |
| `pipeline/inference/` | `predict.py`: nhận bình luận mới và bundle đã huấn luyện, tiền xử lý nhất quán rồi trả nhãn/score. |
| `pipeline/presentation/` | `notebook_display.py`: bảng, mã nguồn và thông tin file có màu tương phản; không thay đổi dữ liệu hoặc dự đoán. |
| `pipeline/modern/` | Vị trí dành cho mã mô hình hiện đại; hiện chưa triển khai. |
| `tools/dataset_versions/` | `merge_labeled_data.py`: gộp CSV có sẵn với đầu ra notebook 01 → `data/final/comments_merged.csv`. `audit_source.py`: đối chiếu bundle nguồn khi cần kiểm tra nội bộ. |
| `scripts/` | `run_notebook.py`: chạy notebook bằng kernel mới, tách đầu ra khỏi runtime tạm. `check_environment.py`: kiểm Python, đúng `.venv` và toàn bộ phiên bản notebook đã pin. |
| `requirements/` | `notebook.txt`: notebook; `collection.txt`: crawler; `annotation.txt`: công cụ nhãn; `dev.txt`: kiểm thử. |
| `tests/` | Các file `test_*.py` kiểm tra dữ liệu, ID/nhãn, liên kết cache, tiền xử lý, CV, lưu/tải/dự đoán và hiển thị. `test_notebook_runner.py` kiểm tra đường dẫn CSV local, đầu ra và đóng kernel. |
| `data/` | `intermediate/`: đầu ra trung gian notebook 01; `processed/v7_preprocessed.csv`: v7 và các cột token cho mô hình; `labeled/comments_labeled.csv`: CSV mới; `labeled/comments_labeled_phase1.csv`: CSV có sẵn; `final/comments_merged.csv`: bản tool tạo; `final/labeled_v7.csv`: đầu vào thực nghiệm local. |
| `artifacts/`, `.cache/`, `.work/` | `artifacts/spamham/`: model và kết quả huấn luyện; `.cache/collection/`: đầu vào tái lập; `.work/`: runtime/test tạm. Không đưa vào Git. `.venv/` chứa môi trường Python. |
| `results/` | `verified_run/`: bằng chứng lần chạy hiện tại; `source_audit/`: hồ sơ kiểm chứng nguồn; `history/`: bằng chứng các phiên bản trước. |
| `docs/` | `assignment/`: ảnh đề; `team/`: phân công; `guides/`: hướng dẫn; `presentation/`: lộ trình mở code; `provenance/`: hồ sơ nguồn; `archive/`: tài liệu cũ đã gom lại; `report/`: dành cho báo cáo. |

Các file `__init__.py` tạo package để import. `.gitignore` loại dữ liệu, môi trường,
model và cache khỏi Git; `.env.example` là mẫu cấu hình, không chứa phiên đăng nhập.
`pytest.ini` cấu hình thư mục kiểm thử và đường dẫn import từ gốc project.

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

**Tổng hợp dữ liệu:** sau notebook 01, chạy
   `.\.venv\Scripts\python.exe -m tools.dataset_versions.merge_labeled_data`
   để gộp `data/labeled/comments_labeled_phase1.csv` với `data/labeled/comments_labeled.csv`,
   xuất `data/final/comments_merged.csv`. Đây là thao tác riêng bên ngoài notebook.

5. **Chuẩn bị thực nghiệm:** notebook 02 đọc `data/final/labeled_v7.csv`, kiểm
   cấu trúc/nhãn, gọi `preprocess_for_model.py` để tách từ và tạo hai cấu hình
   giữ/bỏ stop words. Ghi `data/processed/v7_preprocessed.csv` rồi đọc lại các cột token
   để huấn luyện. Giữ nguyên ID, nội dung nguồn và nhãn.
6. **Huấn luyện:** chia train/test theo khóa văn bản, CV 5 fold trên train;
   `experiments.py` đối chiếu hai cấu hình từ dừng trên sáu tổ hợp, chọn tiền xử lý
   bằng CV rồi khảo sát ba chiến lược bổ sung trên TF-IDF + SVM. Mỗi ứng viên
   có CV Precision/Recall/F1; F1 spam là tiêu chí lựa chọn.
   Vectorizer, IDF và FastText chỉ được học trong phần train tương ứng.
7. **Đánh giá:** chọn cấu hình theo CV, đo trên test; báo cáo Accuracy,
   Precision/Recall/F1 spam, AP, ma trận nhầm lẫn, PR và phân tích FP/FN.
   Bảng test gồm sáu cấu hình cơ sở và ba cấu hình bổ sung; cấu hình TF-IDF + SVM
   không điều chỉnh được dùng lại làm đối chứng, không huấn luyện trùng thành ứng viên mới.
8. **Lưu và ứng dụng:** notebook lưu `spam_ham_ml_only.joblib`; `predict.py` tải
   bundle, áp dụng cùng tiền xử lý và dự đoán bình luận mới. Kiểm tra tải lại không đổi kết quả.

## Cách chạy và đầu ra

Các bước cài môi trường, chọn kernel VS Code và lệnh chạy lần lượt nằm trong
[README ở gốc project](../../README.md). Chạy notebook 01 trước, chạy tool gộp,
rồi chạy notebook 02 khi cần huấn luyện/đánh giá lại.

| Bước | Đầu ra |
|---|---|
| Notebook 01 | `data/intermediate/comments.csv`, `comments_cleaned.csv`, `annotation_predictions.json`, `annotation_review.csv`; CSV cuối `data/labeled/comments_labeled.csv`; `data/collection_manifest.json`. |
| Tool gộp | `data/final/comments_merged.csv`, tạo từ CSV có sẵn và CSV vừa xuất. |
| Notebook 02 | `data/processed/v7_preprocessed.csv`; trong `artifacts/spamham/` có model `.joblib`, `metrics.csv`, `bootstrap_ci.csv`, `run_manifest.json`, `stopword_cv.csv`, `imbalance_cv.csv`, `imbalance_comparison.csv`, `test_predictions.csv` và sáu hình số mẫu/trọng số, kết quả chiến lược, ma trận và PR. Công thức tên hình xem README. Notebook kiểm tra dự đoán từ raw text và sau nạp bundle trên toàn bộ test. |

Notebook 01 không cần cookie/API. Crawler và công cụ nhãn qua dịch vụ có bộ
requirements riêng. `.work/` chỉ giữ runtime/test tạm; đầu ra chính không đặt ở đó.

## Kết quả và giới hạn

- Notebook 01: 5 nhóm, 2.170 bài, 13.833 comment/reply; 13.704 nội dung không rỗng,
  10.974 mẫu sau làm sạch. Có 6.571 gợi ý khớp; ngưỡng 0,75 tiếp nhận 6.289 mẫu
  (5.190 ham, 1.099 spam). Mẫu thiếu gợi ý hoặc chưa đạt ngưỡng không nằm trong CSV có nhãn.
- CSV có sẵn có 31.931 mẫu; tool gộp với 6.289 mẫu mới tạo thành **38.220 mẫu**.
- Notebook 02: v7 có 38.233 mẫu; train 30.759, test 7.474. Cấu hình **S3 — TF-IDF
  + SVM, giữ từ dừng, điều chỉnh ngưỡng** được chọn theo CV F1 **0,7876**. Ngưỡng
  decision score **−0,1776**; không dùng trọng số lớp hoặc lấy mẫu lại. Test:
  Accuracy **94,33%**, Precision spam **83,74%**, Recall spam **68,77%**, F1 spam
  **0,7552**, AP **0,8117**. So với ngưỡng 0, spam bỏ sót giảm từ 354 xuống 297,
  ham nhận nhầm tăng từ 80 lên 127. Đây là đánh đổi cần nêu khi trình bày.
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

## Dữ liệu trước và sau tiền xử lý

Mục 2 notebook 02 hiển thị bình luận từ v7, với ID, nhãn, text gốc, text đã làm
sạch và loại bình luận. Mục 3 dùng cùng các ID để so sánh thêm hai cột token:
`tokens_keep_stopwords` và `tokens_remove_stopwords`.

Toàn bộ 38.233 mẫu được ghi vào `data/processed/v7_preprocessed.csv`, không chỉ
các dòng xem trước. Notebook đọc lại file này, kiểm tra ID/nhãn giữ nguyên rồi
đưa các cột token vào `TEXT`, là đầu vào các vòng huấn luyện. Từ vựng, IDF và
embedding vẫn được học trong phần train của từng fold.

Trong dữ liệu hiện tại, có 2 dòng token rỗng khi giữ stop words và 103 dòng khi
loại stop words. Các dòng này vẫn giữ ID/nhãn; vectorizer biểu diễn bằng vector
không có đặc trưng thay vì âm thầm bỏ mẫu.

## Thiết kế CV và test

Train gồm 30.759 mẫu; test giữ riêng 7.474 mẫu. Train được chia thành năm fold
theo nhóm văn bản. Trong mỗi lượt CV, mô hình học từ bốn fold và dự đoán fold
validation còn lại. Cột “Số mẫu spam trong validation” là số spam trong phần
validation, không phải một tập dữ liệu riêng.

Trung bình F1 spam trên các fold là **CV F1**, dùng để chọn cấu hình. Precision
và Recall CV cũng tính trên cùng fold. Sau đó mô hình fit toàn bộ train và đo
Accuracy/Precision/Recall/F1/AP trên test. CV có thể lạc quan do tìm nhiều cấu
hình; đây không phải nested CV. Không chọn lại mô hình hoặc ngưỡng bằng test.

## Mất cân bằng nhãn

V7 có 32.907 ham (86,07%) và 5.326 spam (13,93%), xấp xỉ 6,18 ham trên mỗi spam.
Sáu tổ hợp cơ sở dùng train nguyên bản; SVM không trọng số lớp, NB dùng ngưỡng
0,5 và SVM dùng ngưỡng decision score 0. TF-IDF + SVM dẫn đầu bước CV cơ sở,
nên được chọn để khảo sát sâu ba chiến lược. Đây là thiết kế theo từng bước,
chưa khảo sát mọi chiến lược trên cả sáu tổ hợp.

| Cấu hình TF-IDF + SVM | Cách thực hiện |
|---|---|
| Không điều chỉnh | 26.384 ham + 4.375 spam, `class_weight=None`, ngưỡng 0; dùng lại ứng viên cơ sở B4. |
| Trọng số balanced | `experiments.fit_experiment(class_weight="balanced")`; số mẫu giữ nguyên. Trọng số toàn bộ train khoảng 0,583 cho ham và 3,515 cho spam; mỗi fold tính lại từ phần train. |
| Under-sampling 2:1 | `training_utils.undersampled_search()`; fit cuối dùng 8.750 ham + 4.375 spam. Lấy mẫu riêng trong train của mỗi fold, không dùng balanced. |
| Điều chỉnh ngưỡng | `training_utils.tune_svm_threshold()`; giữ siêu tham số của SVM không điều chỉnh, tạo decision score OOF trên train và chọn ngưỡng bằng trung bình F1 validation. |

Nhánh ngưỡng khảo sát 199 phân vị 1–99% của điểm OOF, ngưỡng 0 và hai biên;
nếu đồng điểm F1 thì ưu tiên ngưỡng gần 0. Số fold và khóa nhóm không thay đổi.
Trọng số và under-sampling tìm lại C/ngram bằng cùng lưới cơ sở. Các chiến lược
được thử riêng; không kết hợp ba cách. Tỷ lệ lấy mẫu 2:1 là cấu hình khảo sát,
không được lựa chọn từ test. Điều chỉnh ngưỡng thay đổi quyết định, không thay
đổi số mẫu hay cân bằng lại phân bố dữ liệu.

Mục 5.1 trình bày hình số mẫu/trọng số và bảng CV bốn cấu hình. Mục 6.1 báo cáo
test Precision/Recall/F1, Accuracy/AP và FP/FN. Mục 6.2 có hai nhóm ma trận:
sáu tổ hợp cơ sở và bốn cấu hình TF-IDF + SVM. Mục 6.3 có PR cho hai phạm vi;
nhánh chỉ đổi ngưỡng dùng chung đường PR/AP với đối chứng, hai điểm đánh dấu
cho thấy vị trí quyết định khác nhau. Bootstrap F1 gồm cả chín ứng viên.

`predict_record()` và `predict_spam()` dùng cùng ngưỡng SVM; score lớn hơn ngưỡng
được dự đoán spam. Ngưỡng mặc định là 0 với SVM và 0,5 với NB. Ngưỡng lưu trong
bundle được dùng lại khi demo. Mục phân tích lỗi ghi rõ cấu hình được chọn và
hiển thị ID, text gốc, văn bản xử lý, nhãn thực tế, dự đoán và score.

## Đối chiếu từ dừng

So sánh giữ/loại từ dừng là khảo sát bổ trợ cho lựa chọn tiền xử lý, trình bày
bằng bảng CV F1 tại mục 5, không có phụ lục. Cấu hình giữ từ dừng được CV chọn;
sáu tổ hợp cơ sở và các chiến lược tiếp theo cùng dùng cấu hình này.

Không chấm test cho nhánh từ dừng đã loại khỏi luồng chính. Cách tiền xử lý,
hai cột token và dữ liệu nguồn vẫn được giữ nguyên; từ vựng/embedding vẫn chỉ
học trong phần train. Khảo sát OOV riêng được bỏ để tập trung vào phân loại;
FastText vẫn là một phương pháp biểu diễn trong sáu tổ hợp.

## Notebook và module có cùng luồng xử lý

Notebook vừa điều phối dữ liệu/split/cấu hình, vừa gọi hàm trong `pipeline/` và
hiển thị kết quả. Chạy cell xử lý là thực thi công việc đó. Chạy runner từ Terminal
là một cách thực thi toàn bộ notebook; mở khối mã nguồn chỉ để xem cách viết hàm.

Các module có `main()`/CLI có thể chạy rời bằng `-m` với đúng đầu vào. Các module
thư viện như `experiments.py` không tự huấn luyện cả bài khi gọi trực tiếp file;
cần notebook hoặc chương trình điều phối cung cấp dữ liệu và cấu hình.

## Các thư mục kiểm tra và kết quả

`tests/` chứa chương trình kiểm thử và các tình huống giả lập nhỏ để kiểm tra
hành vi hàm, không phải dataset thực nghiệm v7. Các nhóm gồm:
`test_data_pipeline.py` và `test_collection_provenance.py` kiểm luồng dữ liệu;
`test_cached_annotations.py` kiểm nối nhãn đúng ID; `test_preprocessing_and_training.py`
kiểm tiền xử lý, CV, sampling và ngưỡng SVM; `test_restructured_modules.py` kiểm module/CLI và
lưu–tải–dự đoán; `test_notebook_display.py` kiểm hiển thị; `test_notebook_runner.py`
kiểm cấu hình runner và đường dẫn local.

`results/verified_run/` giữ chỉ số và biên bản kiểm tra mới nhất;
`results/source_audit/` là hồ sơ đối chiếu nguồn; `results/history/` giữ kết quả
trước đây. Hai notebook không dùng các thư mục này để nạp dữ liệu hay dự đoán.
Mô hình để tái sử dụng nằm trong `artifacts/spamham/`, dữ liệu trong `data/`.
