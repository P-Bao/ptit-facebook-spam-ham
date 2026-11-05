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
   `experiments.py` huấn luyện 12 tổ hợp chính và các đối chứng cân bằng lớp/ngưỡng.
   Vectorizer, IDF và FastText chỉ được học trong phần train tương ứng.
7. **Đánh giá:** chọn cấu hình theo CV, đo trên test; báo cáo Accuracy,
   Precision/Recall/F1 spam, AP, confusion matrix, so sánh stop words và phân tích FP/FN.
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
| Notebook 02 | `data/processed/v7_preprocessed.csv`; trong `artifacts/spamham/` có model `.joblib`, `metrics.csv`, `bootstrap_ci.csv`, `run_manifest.json`, `imbalance_comparison.csv` và hai hình `imbalance_training.png`, `imbalance_results.png`. |

Notebook 01 không cần cookie/API. Crawler và công cụ nhãn qua dịch vụ có bộ
requirements riêng. `.work/` chỉ giữ runtime/test tạm; đầu ra chính không đặt ở đó.

## Kết quả và giới hạn

- Notebook 01: 5 nhóm, 2.170 bài, 13.833 comment/reply; 13.704 nội dung không rỗng,
  10.974 mẫu sau làm sạch. Có 6.571 gợi ý khớp; ngưỡng 0,75 tiếp nhận 6.289 mẫu
  (5.190 ham, 1.099 spam). Mẫu thiếu gợi ý hoặc chưa đạt ngưỡng không nằm trong CSV có nhãn.
- CSV có sẵn có 31.931 mẫu; tool gộp với 6.289 mẫu mới tạo thành **38.220 mẫu**.
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

## Mất cân bằng nhãn

V7 có 32.907 ham (86,07%) và 5.326 spam (13,93%), xấp xỉ 6,18 ham trên mỗi spam.
Project đã thực nghiệm các cách sau:

| Cách xử lý | Thực hiện ở đâu và tác dụng |
|---|---|
| SVM class weight | `experiments.py` thử `None` và `balanced`; balanced tăng mức phạt lỗi ở lớp ít mẫu theo thống kê của phần train đang fit. |
| Ngưỡng Naive Bayes | `training_utils.tune_nb_threshold()` dùng dự đoán ngoài fold trên train để chọn ngưỡng theo F1 spam. |
| Under-sampling | `training_utils.undersampled_search()` giảm số ham trong phần train đến tối đa 2 ham/1 spam; không lấy mẫu validation/test. |
| Tiêu chí đánh giá | Tối ưu F1 spam bằng CV; đọc cùng Precision, Recall và confusion matrix để thấy spam bị bỏ sót. |

Mô hình cuối hiện là TF-IDF + SVM, `class_weight=None`, vì CV F1 cao hơn cấu hình
balanced. Trên test, balanced tăng Recall spam từ 62,78% lên 77,92%, đồng thời
Precision giảm từ 88,18% xuống 71,39%. Cấu hình cuối được chọn bằng CV trước khi
đánh giá test. Các biện pháp này khảo sát ảnh hưởng mất cân bằng; mô hình vẫn có
sai số và chưa loại bỏ hoàn toàn việc bỏ sót spam.

Notebook 02 trình bày rõ phạm vi thực nghiệm:

| Nhánh | Dữ liệu train khi fit cuối | Điều chỉnh |
|---|---|---|
| TF-IDF + SVM gốc | 26.384 ham + 4.375 spam | `class_weight=None` |
| TF-IDF + SVM balanced | Giữ nguyên số mẫu | Trọng số toàn bộ train khoảng 0,583 cho ham và 3,515 cho spam; mỗi fold tính lại từ train của fold. |
| TF-IDF + SVM under-sampling | 8.750 ham + 4.375 spam | Giữ spam, chọn ngẫu nhiên ham theo tỷ lệ tối đa 2:1; không kết hợp trọng số balanced. |
| TF-IDF + Naive Bayes gốc | Giữ nguyên số mẫu | Ngưỡng mặc định 0,5 |
| TF-IDF + Naive Bayes chỉnh ngưỡng | Giữ nguyên số mẫu | Ngưỡng khoảng 0,1721 từ dự đoán ngoài fold trên train. |

Ba biện pháp được thử thành các nhánh riêng. Under-sampling có thể áp dụng cho
các mô hình khác, nhưng thí nghiệm này chọn SVM để đối chiếu với cách dùng trọng
số trên cùng họ thuật toán. Không có yêu cầu áp dụng đồng thời cả ba cách cho
mọi mô hình; phối hợp chúng sẽ cần một thí nghiệm riêng để kiểm tra tác động.
Tỷ lệ lấy mẫu 2:1 là cấu hình đối chứng cố định, không được chọn bằng kết quả test.

Mục 5.1 có bảng phạm vi, bảng số mẫu và hình trọng số/lấy mẫu. Mục 6.1 có bảng
chỉ số, số FP/FN và biểu đồ Precision–Recall–F1 cùng lỗi phân loại. Các giá trị được
lấy từ mô hình đã fit và dự đoán trên test, không phải số minh họa tự đặt.

## Đối chứng stop words

Đề yêu cầu áp dụng kỹ thuật tiền xử lý và so sánh đặc trưng/thuật toán; so sánh
giữ/bỏ stop words là thực nghiệm bổ sung. Bảng này được chuyển xuống phụ lục,
giữ phần chính gọn và vẫn trình bày được nhánh loại từ dừng.

Cấu hình giữ stop words hiện được CV lựa chọn. Với TF-IDF + SVM chính, CV F1
khi giữ là 0,7736 và khi loại là 0,7562; test F1 tương ứng là 0,7334 và 0,7194.
Vì vậy, bỏ toàn bộ nhánh giữ stop words có thể thay đổi mô hình được chọn và
chỉ số cuối. Việc đưa đối chứng xuống phụ lục giữ nguyên dữ liệu, thí nghiệm và
nguyên tắc lựa chọn theo CV; không chọn lại cấu hình bằng kết quả test.

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
kiểm tiền xử lý, CV và sampling; `test_restructured_modules.py` kiểm module/CLI và
lưu–tải–dự đoán; `test_notebook_display.py` kiểm hiển thị; `test_notebook_runner.py`
kiểm cấu hình runner và đường dẫn local.

`results/verified_run/` giữ chỉ số và biên bản kiểm tra mới nhất;
`results/source_audit/` là hồ sơ đối chiếu nguồn; `results/history/` giữ kết quả
trước đây. Hai notebook không dùng các thư mục này để nạp dữ liệu hay dự đoán.
Mô hình để tái sử dụng nằm trong `artifacts/spamham/`, dữ liệu trong `data/`.
