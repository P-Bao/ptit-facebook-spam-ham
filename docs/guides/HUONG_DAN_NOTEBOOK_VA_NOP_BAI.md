# Notebook, pipeline và cách nộp/demo

Tài liệu này giải thích bản notebook sau khi bổ sung nguồn thu thập. Phạm vi là
các bước cốt lõi trong ảnh đề; chưa viết báo cáo học phần.

File thực nghiệm chính: `notebooks/spam-ham-comments.ipynb`.
Nguồn kiểm chứng chi tiết: `docs/references/CRAWLER_VA_DU_LIEU_DRIVE.md`.

## 1. Notebook trình bày lần lượt thế nào?

| Mục | Nội dung để giải thích khi trình bày | Tác dụng |
|---|---|---|
| 0. Môi trường/cấu hình | Dependency, seed, đường dẫn CSV/artifacts; các cell ghi module hỗ trợ | Tách `.ipynb` khỏi repo vẫn có mã cần thiết. |
| 1. Thu thập và truy xuất nguồn | Crawler được bàn giao, đoạn code lấy bài/comment, 5 group thật, timestamp, hash; đối soát 7 file Drive tới v7 | Chứng minh dữ liệu có nguồn và hiểu quá trình làm, thay việc chỉ tải CSV không giải thích. |
| 2. Tiền xử lý | Chuẩn hóa, URL/emoji/tên, tách từ; tạo hai biến thể giữ/bỏ stop words | Đầu vào thống nhất cho train và dự đoán; kiểm tra tác động stop words. |
| 3. Chia tập | Train/test và CV theo nhóm text trùng | Hạn chế cùng nội dung rơi vào cả phía học và kiểm tra; lưu giới hạn nhãn mâu thuẫn. |
| 4. Trích chọn đặc trưng, huấn luyện | BoW, TF-IDF, FastText; NB/SVM; chọn tham số theo CV | So sánh 12 tổ hợp chính trên quy trình chung. FastText được học trong phần train, không fit trên test. |
| 5. Đánh giá | Accuracy, Precision, Recall, F1, Confusion Matrix; so sánh phương pháp | Đáp ứng bộ chỉ số trong đề; chọn mô hình bằng CV rồi mới đo test. AP/CI là phân tích thêm. |
| 6. Phân tích lỗi | FP/FN, đặc trưng, FastText/OOV và giới hạn nhãn/ngữ cảnh | Giải thích vì sao sai, tránh chỉ đọc điểm số. |
| 7. Dự đoán và lưu | Nhập raw text, chạy lại đúng tiền xử lý, áp ngưỡng; lưu/tải mô hình kiểm tra | Demo cách dùng mô hình và tái sử dụng kết quả. |
| 8. Nhận xét | Sinh kết luận từ output của lần chạy | Không dùng số cũ hoặc khẳng định class weight luôn tốt. |
| 9. Hướng dẫn đọc/nộp | Nhắc thứ tự và phân biệt notebook với công cụ ngoài | Người chấm hiểu cách chạy và dữ liệu cần kèm. |

Phần thu thập hiển thị đúng luồng: **13.833 raw → 9.569 CSV → 9.398 cleaned lịch
sử → 6.303 nhãn Phase 2 → bỏ một trùng → gộp 38.233**. Notebook còn chỉ ra rằng
v7 thay 1.366 nhãn so với bản gộp Drive; không âm thầm trộn hai phiên bản nhãn.

Các trích đoạn HTTP thu thập được đặt trong Markdown để giải thích code gốc.
Run All chạy lại phần đọc/làm phẳng/đối soát snapshot, không sử dụng phiên Facebook.
Đây là tái hiện từ dữ liệu đã thu, không phải một lần cào live mới.

## 2. Đối chiếu các yêu cầu cốt lõi trong ảnh đề

| Yêu cầu | Bản hiện tại | Đánh giá và giới hạn |
|---|---|---|
| Bình luận sinh viên PTIT trên Facebook group; phân loại spam/ham | Raw Phase 2 có group ID/tên, bài/comments/replies; CSV v7 có hai nhãn | Đúng chủ đề được chọn. Chỉ chứng minh nguồn raw Phase 2; Phase 1 mới có CSV nhãn. |
| 1. Thu thập dữ liệu | Có mã bàn giao, snapshot, đoạn code và đối soát | Đã có phần trình bày/kiểm chứng; không khẳng định crawler lấy đủ toàn bộ hoặc mọi nhãn đều đã duyệt tay. |
| 2. Tiền xử lý: tách từ, stop words… | Có chuẩn hóa/tách từ, giữ/bỏ stop words | Đáp ứng và có đối chứng. |
| 3. Ít nhất hai phương pháp đặc trưng | BoW, TF-IDF, FastText | Đáp ứng với ba phương pháp. |
| 4. NB/SVM, chia train/test | Có cả hai họ thuật toán, grouped split và CV | Đáp ứng; GaussianNB dùng cho embedding, MultinomialNB cho đếm/TF-IDF. |
| 5. Accuracy/Precision/Recall/F1/Confusion Matrix, so sánh | Có bảng, hình và phân tích kết quả | Đáp ứng về code/thực nghiệm; chất lượng nhãn vẫn là giới hạn của kết luận. |

Về **phần code cốt lõi**, notebook đã có các bước yêu cầu. Không nên phát biểu
“nguồn hoàn toàn chuẩn, nhãn đúng tuyệt đối”: thiếu raw/log Phase 1 và lịch sử sửa
v7; còn 27 nhóm text giống nhau có nhãn khác. Không cần tự cào lại hoặc tự sửa các
nhãn này khi chưa có ngữ cảnh/xác nhận. Việc thu hồi lịch sử nhãn và rà các nhóm
mâu thuẫn là ưu tiên cải thiện chất lượng dữ liệu, không phải lỗi chạy code.

## 3. Các file ngoài và notebook khác nhau ra sao?

| Thành phần | Đầu vào | Việc làm | Đầu ra |
|---|---|---|---|
| Bộ crawler bàn giao | Group ID, cấu hình, phiên truy cập | Lấy bài, comments/replies và metadata | JSON raw, trạng thái/cursor; có thể thêm file text/media tùy nhánh chạy. |
| `step_01_collect` | JSON raw, dữ liệu Phase 1 | Làm phẳng, loại rỗng/trùng Phase 1; audit đọc thêm các snapshot | CSV Phase 2; JSON đối soát. |
| `step_02_preprocess` | CSV | Clean, khử trùng; module tiền xử lý dùng chung | CSV cleaned; token/chuỗi đầu vào mô hình. |
| `step_03_label` | CSV cleaned, Argilla/cache | Upload, gợi ý LLM, duyệt/accept và export có nối metadata | Cache gợi ý và CSV có nhãn; cần dịch vụ/người duyệt ở các bước tương ứng. |
| `step_04_merge` | CSV nhãn hai phase | Giữ nguồn, thêm prefix ID Phase 2, loại trùng chéo | CSV gộp; không tự tạo lịch sử chỉnh nhãn v7. |
| `step_05_features`, `step_06_train_evaluate` | Văn bản train/tham số | Định nghĩa FastText, CV, cân bằng/ngưỡng | Module chuẩn để notebook nhúng và sử dụng; không tự huấn luyện khi chỉ mở file. |
| Notebook | V7 + 7 snapshot nguồn | Đối soát, tiền xử lý, chia tập, fit/CV, đánh giá, phân tích, demo và lưu | Bảng/hình/output trong `.ipynb`, model joblib, metrics, CI, manifest và source audit trong artifacts. |
| `scripts/run_notebook.py` | Notebook và CSV | Chạy notebook trong thư mục riêng; chỉ ghi lại notebook chính nếu thành công | Bản executed, artifacts và execution status; không thay vai trò thu thập/gán nhãn. |

`pipeline/` là bộ công cụ theo bước, **không có một lệnh tự động chạy tất cả từ
Facebook tới mô hình**. Với dữ liệu hiện có, có thể tái hiện tới bản gộp Drive.
Không thể khẳng định tái tạo chính xác v7 nếu chạy lại LLM hoặc thiếu log 1.366 lần
đổi nhãn. Notebook bắt đầu thực nghiệm từ v7 đã chốt nên không cần chạy lại Argilla.

Sau Run All, bên cạnh output bảng/hình trong notebook, thư mục `ARTIFACT_DIR` có:

| File | Dùng để làm gì? |
|---|---|
| `spam_ham_ml_only.joblib` | Mô hình được chọn, ngưỡng, cấu hình tiền xử lý và manifest; notebook kiểm tra tải lại dự đoán khớp. |
| `metrics.csv` | Bảng so sánh các tổ hợp/thí nghiệm đã đánh giá. |
| `bootstrap_ci.csv` | Khoảng tin cậy bootstrap F1, phần phân tích thêm. |
| `run_manifest.json` | Hash dữ liệu, phiên bản môi trường, split/seed và mô hình được chọn. |
| `source_audit.json` | Kết quả đối soát nguồn Drive và chênh lệch tới v7. |

Thư mục mặc định là `artifacts/` trong nơi chạy notebook; có thể đổi bằng
`SPAMHAM_OUTPUT_DIR`. Lần kiểm chứng hiện tại lưu artifacts đầy đủ ở
`.work/runs/source_verified/artifacts/`; các bảng/JSON cần theo dõi được chép sang
`results/verified_run/` và `results/source_audit/`. Dataset/model lớn không đưa vào Git.

## 4. Nộp mỗi notebook có chạy được không?

**Về mã Python: có.** Notebook tự ghi năm module hỗ trợ từ các cell `%%writefile`;
không import `pipeline/` hay thư mục crawler bên ngoài. Không cần nộp các `.py`
ngoài để các cell hoạt động. Lần kiểm chứng chạy trong thư mục riêng với notebook
và đường dẫn dữ liệu, không dựa vào việc import module của repo.

**Về dữ liệu/môi trường: vẫn cần chuẩn bị.** Có hai cách:

1. Có Internet: notebook tải v7 từ HF và 7 snapshot từ Drive, kiểm tra revision/hash.
   Cần Python/dependency phù hợp và thư mục chạy cho phép ghi file. Quyền các link
   có thể đổi sau này; không chỉ dựa vào việc output cũ đang hiển thị.
2. Offline: kèm `labeled_v7.csv` và một folder chứa đúng 7 file Drive. Đặt `CSV_PATH`
   hoặc `SPAMHAM_CSV`, và `SPAMHAM_SOURCE_DIR` tới folder đó trước Run All. Tên file
   và hash phải khớp; không dùng bản gộp Drive đổi tên thành v7.

Để nộp code và demo ổn định: dùng **notebook đã có output** làm file chính, giữ một
bản dữ liệu offline và môi trường đã cài để dự phòng. Nộp thêm mã crawler như tài
liệu nguồn nếu giảng viên yêu cầu minh chứng cào dữ liệu hoặc nộp toàn bộ source;
nó không phải dependency thực thi của notebook. Ảnh đề có yêu cầu báo cáo; hình
thức nộp cuối phải theo thông báo giảng viên, chưa có căn cứ nói `.ipynb` thay thế
mọi tài liệu bắt buộc. Theo yêu cầu hiện tại, báo cáo sẽ làm sau khi chốt code.

Khi trình bày, đi theo mục 1 → 8 và dùng output đã chạy để giải thích. Có thể chạy
cell dự đoán để demo; không cần chờ train lại mọi tổ hợp ngay trước lớp hoặc cào
Facebook live. Nếu demo chính việc cào mới thì cần bộ crawler ngoài và sửa các
giới hạn phân trang/checkpoint nêu trong tài liệu kiểm chứng nguồn trước.
