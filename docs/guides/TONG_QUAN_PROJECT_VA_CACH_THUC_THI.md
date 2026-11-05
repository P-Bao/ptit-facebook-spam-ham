# Tổng quan project, luồng thực thi và cách đọc kết quả

Cập nhật 05/11/2026. Bài toán: phân loại bình luận sinh viên PTIT thành ham/spam.
Nhóm dùng mã/snapshot thu thập để giải thích quá trình chuẩn bị dữ liệu và dùng
dataset v7 đã chốt cho thực nghiệm. Hai phần có đầu vào và phạm vi riêng.

## 1. Những gì đã sửa ở notebook

- Code đọc từ module được hiển thị chữ sáng trên nền xanh đậm cố định. Bảng dùng
  chữ tối trên nền trắng, đặt màu inline cho từng ô để tránh theme IDE đổi màu.
- Khối source nằm trong “Xem code” có thể mở/đóng, đặt sau dữ liệu đầu ra.
- Giữ tên chỉ mục của Series/DataFrame, nên bảng môi trường/thống kê không mất
  ý nghĩa của từng dòng. Escape nội dung comment/source trước khi render HTML.
- Notebook 01 hiển thị group/bài thật, comment và reply, text trước/sau clean,
  ví dụ ham/spam, trạng thái xử lý và tên/kích thước/số dòng/nội dung từng file.
- Có chuyển đổi toàn bộ snapshot thành CSV, chạy cleaner và CLI chấp nhận nhãn
  từ cache thật, không chỉ gọi hàm show source. Sau khi ghi CSV có đọc lại để kiểm tra.

Mở lại notebook nếu IDE còn giữ output cũ. Run All tạo output mới; không cần đổi
theme. Bộ renderer ở `pipeline/presentation/notebook_display.py`, dùng cho cả hai
notebook. Kiểm tra trình duyệt: `results/notebook_display/contrast_verification.json`.

## 2. Các folder và từng nhóm file

| Folder | File/chức năng |
|---|---|
| `notebooks/` | `01_thu_thap_va_gan_nhan.ipynb` trình bày xử lý offline từ JSON/cache; `02_thuc_nghiem_va_demo.ipynb` thực nghiệm v7 và demo. |
| `external/facebook_scraper/` | Snapshot thư viện nền tham khảo. Các file Python giữ nguyên từ bộ bàn giao; không tự gọi GUI khi mở notebook. |
| `pipeline/collect/` | Runner PTIT, chuyển JSON thành bảng, đọc/tải snapshot. |
| `pipeline/prepare/` | Hai lớp xử lý: clean bảng trước label và preprocess đầu vào mô hình. |
| `pipeline/label/` | Upload, AI suggestion, chấp nhận/duyệt nhãn, export và kiểm tra ID. |
| `pipeline/features/` | Vectorizer FastText, tương thích scikit-learn. |
| `pipeline/modeling/` | Tạo pipeline, grid/CV, cân bằng/ngưỡng, đánh giá. |
| `pipeline/inference/` | Dự đoán raw comment bằng bundle đã train. |
| `pipeline/presentation/` | Chỉ định dạng bảng/source/file info để dễ đọc; không sửa dữ liệu hoặc mô hình. |
| `pipeline/modern/` | Giữ chỗ ý 4 bổ sung, chưa có mô hình/kết quả. |
| `tools/dataset_versions/` | Gộp và đối soát phiên bản; không bắt buộc để train v7. |
| `data/` | Dữ liệu local, không commit: `raw` JSON; `intermediate` CSV/cache; `labeled` export nhãn; `final/labeled_v7.csv` cho thực nghiệm. |
| `requirements/` | `notebook.txt` cho notebook; `collection.txt` cho crawler; `annotation.txt` cho Argilla/AI; `dev.txt` cho tests. |
| `scripts/` | `run_notebook.py` chọn notebook 01/02, mở kernel mới, đặt seed/path, lưu output và bằng chứng chạy. |
| `tests/` | Kiểm tra dữ liệu/ID, preprocess, CV, model–save/load–predict, đường dẫn module và renderer. |
| `results/` | `verified_run` kết quả chạy hiện tại; `notebook_display` kiểm chứng lượt sửa này; `source_audit` nguồn; `history` bằng chứng phiên bản trước; `restructure` hồ sơ lần chuyển cấu trúc. |
| `docs/` | `assignment` ảnh đề, `team` phân công, `plans` kế hoạch, `overview` tài liệu cũ, `provenance` nguồn/giới hạn, `guides` hướng dẫn, `presentation` kịch bản mở code, `changes` nhật ký, `report` chỗ báo cáo làm sau. |
| `.work/`, `artifacts/` | Cache, model/CSV và output local khi chạy; không commit. `.venv/` là môi trường Python. |

### Các module thực hiện công việc

| File | Nhận gì → làm gì → trả/ghi gì? |
|---|---|
| `pipeline/collect/scrape_ptit_groups.py` | Group/cấu hình phiên → điều phối lấy bài/comment → JSON/JSONL, state, text/media tùy chọn. |
| `external/facebook_scraper/group_post_scraper_v2.py` | Group/cursor → lấy trang bài và metadata, media → list bài và callback. |
| `external/facebook_scraper/main.py` | Post ID → `fetch_comments_for_post()` ghép comment/reply → danh sách bình luận của bài; còn các nhánh menu khác. |
| `external/facebook_scraper/comment_scraper.py` | Feedback/comment ID → HTTP và phân tích response → comment/reply. |
| `external/facebook_scraper/post_scraper.py` | Nhánh page/profile; runner PTIT không gọi chức năng lấy bài page nhưng `main.py` vẫn import module. |
| `external/facebook_scraper/single_post_image.py` | Nhánh ảnh bài đơn, cũng được import gián tiếp. |
| `external/facebook_scraper/proxy_utils.py` | Tiện ích kết nối/proxy của bộ nền; không phải bước NLP. |
| `external/facebook_scraper/facebook_ui.py` | GUI PyQt6; không cần mở để chạy hai notebook. |
| `pipeline/collect/json_to_csv.py` | JSON → `flatten_posts()` tách từng comment/reply; CLI còn bỏ rỗng/lọc trùng tham chiếu → CSV chưa nhãn. |
| `pipeline/collect/snapshots.py` | Tên file + manifest → tìm/tải, kiểm SHA-256 → path snapshot; không cào Facebook. |
| `pipeline/prepare/clean_for_labeling.py` | CSV text → chuẩn hóa/khử trùng/lọc, giữ text gốc → CSV thêm cleaned_text. |
| `pipeline/prepare/preprocess_for_model.py` | Text/cleaned_text → che tên bảo thủ, chuẩn hóa, tách từ, stop words/emoji → chuỗi token cho model. |
| `pipeline/label/upload_to_argilla.py` | CSV → record/câu hỏi trên Argilla; chưa đồng nghĩa có nhãn chốt. |
| `pipeline/label/ai_assisted_spam_ham.py` | CSV/Argilla → gọi LLM nếu thiếu dự đoán → cache label/confidence và suggestion; có tùy chọn tự chốt. |
| `pipeline/label/auto_accept_labels.py` | Cache + CSV đúng ID → áp ngưỡng → response hoặc CSV official_label; `--skip-argilla` chỉ xử lý local. |
| `pipeline/label/download_labeled_data.py` | Response submitted + lookup → kiểm tra và nối metadata → CSV có nhãn. |
| `pipeline/label/label_utils.py` | Kiểm ID/prediction/text → chuẩn hóa và nối đúng dòng, từ chối ghép sai. |
| `pipeline/features/spam_features.py` | Token train → học FastText/IDF → vector câu; transform test không fit lại. |
| `pipeline/modeling/experiments.py` | X/y/folds/cấu hình → BoW/TF-IDF/FastText + NB/SVM, GridSearchCV, đánh giá/bootstrap → model/params/metrics. |
| `pipeline/modeling/training_utils.py` | Key/indices/model → nhóm text, fold, sampling train, OOF threshold và predict → chỉ số/model/score. |
| `pipeline/inference/predict.py` | Raw text + bundle → cùng preprocess, vectorizer/model đã fit → text/score/score_type/du_doan. |
| `pipeline/presentation/notebook_display.py` | Bảng/source/path → HTML tương phản và file info; `code_excerpt_html` không gọi hàm được show. |
| `tools/dataset_versions/merge_labeled_data.py` | Hai CSV đã nhãn → nối, xử lý ID/trùng → CSV gộp; không tạo nhãn mới hoặc tự ra v7. |
| `tools/dataset_versions/audit_source.py` | Snapshot/corpus/runner → tái hiện và so sánh → JSON kiểm chứng; không đổi nhãn. |

Các `__init__.py` tạo package import; README hướng dẫn; `.env.example` chỉ là mẫu.
Không đưa `.env` thật hoặc cookie/token vào notebook/Git.

Các file kiểm tra hỗ trợ:

- `tests/test_data_pipeline.py`: flatten, ID/nhãn/metadata và CLI dữ liệu offline.
- `tests/test_collection_provenance.py`: legacy clean, nguồn snapshot và chênh phiên bản nhãn.
- `tests/test_preprocessing_and_training.py`: che tên, raw/clean tokens, nhóm fold, sampling và FastText.
- `tests/test_restructured_modules.py`: tải/kiểm hash, CLI runner, default path, train–save/load–predict.
- `tests/test_notebook_display.py`: màu, escape HTML, giữ index/DataFrame và thông tin file.

`scripts/run_notebook.py --notebook 01/02` điều phối một notebook, không tự chạy
mọi module của project. Tập tin test/requirements không tham gia dự đoán cuối.

## 3. Luồng đầy đủ của project

### A. Khi thực sự chuẩn bị dữ liệu từ đầu

1. Cấu hình nguồn trong runner PTIT; runner gọi module lấy bài, rồi lấy comment/reply.
2. Callback tạo record, writer ghi `data/raw/posts.json`; state lưu cursor.
3. Converter đọc JSON, làm phẳng và lọc theo cấu hình → `data/intermediate/comments.csv`.
4. Cleaner → `data/intermediate/comments_cleaned.csv`, giữ raw text.
5. Upload/LLM gợi ý → cache; tự chấp nhận theo ngưỡng hoặc người duyệt → export nhãn
   tại `data/labeled/comments_labeled.csv`.
6. Tổng hợp/đối soát khi có nhiều bảng; chọn bản phát hành dữ liệu dùng thực nghiệm.
   Hiện nhóm chọn v7, không có đầy đủ lịch sử chỉnh nhãn để nhận merge tự tạo v7.
7. Nạp v7, kiểm tra, preprocess, chia nhóm train/test/CV.
8. Fit đặc trưng và NB/SVM trên train/fold, chọn theo CV rồi đánh giá test.
9. Lưu bundle; comment mới đi qua cùng preprocess và bộ vectorizer/model đã học.

Ý 1: thu thập; ý 2: xử lý văn bản; ý 3: ít nhất hai đặc trưng; ý 4: NB/SVM và
chia tập; ý 5: chỉ số/Confusion Matrix/so sánh. Ý 4 bổ sung giữ chỗ làm sau.

### B. Run All notebook 01 hiện tại thực sự làm gì?

| Bước | Thực thi | Kết quả của lần kiểm tra |
|---|---|---|
| Nạp nguồn | Đọc file thật/kiểm hash; tải Drive nếu thiếu | 2.170 bài, 5 group; hiển thị metadata, comment và reply |
| Show source | Đọc AST và hiển thị code trong khối mở/đóng | Không gọi hàm crawler/LLM chỉ vì show source |
| Chuyển bảng | Chạy `flatten_posts()` toàn bộ JSON, bỏ text rỗng, ghi và đọc lại CSV | 13.833 comment/reply → 13.704 dòng không rỗng |
| Clean | Chạy `clean_facebook_comments()` thật, có khử trùng/lọc | 10.974 dòng trong CSV mới |
| Áp ngưỡng cache | Chạy subprocess CLI `auto_accept_labels --skip-argilla` | Nguồn cleaned lịch sử 9.398 dòng; 6.302 dòng có nhãn đủ ngưỡng |
| Nhãn chốt | Đọc export đã lưu, kiểm ID/text với cleaned lịch sử, xuất lại | 6.303 nhãn; hiển thị ví dụ cả ham và spam |
| Tổng hợp | Tạo manifest và bảng file info | Biết rõ file, số dòng, kích thước và cột/nội dung |

Nhánh CSV mới xử lý toàn bộ snapshot, không lọc theo tập tham chiếu như lịch sử.
Vì thế 10.974 dòng mới khác 9.398 dòng nguồn cache. Bước cache dùng đúng file cũ,
không áp chỉ số cmt_* lên CSV vừa đổi dòng. Export chốt có một ID ngoài cache;
chưa biết lịch sử ai thêm. Không coi việc đọc nhãn sẵn là vừa gọi AI mới.

Các file thật notebook 01 ghi trong ARTIFACT_DIR:

| File | Chứa gì? |
|---|---|
| `comments_from_snapshot.csv` | 13.704 dòng raw comment/reply không rỗng và metadata |
| `comments_cleaned_from_snapshot.csv` | 10.974 dòng sau cleaner, thêm cleaned_text |
| `accepted_from_saved_cache.csv` | Đủ 9.398 dòng nguồn cache; dòng chưa được nhận có official_label rỗng |
| `labels_auto_accepted.csv` | 6.302 dòng vừa được áp ngưỡng từ cache |
| `labels_saved_official_export.csv` | 6.303 nhãn chốt đã lưu, được đối soát/xuất lại |
| `collection_demo_manifest.json` | Số lượng, hash, chế độ chạy và danh sách output |

Lần kiểm chứng nằm ở `.work/runs/readable_collection/artifacts/`. Mở notebook trong
repo mặc định dùng `artifacts/collection/`; có thể đổi `SPAMHAM_OUTPUT_DIR`.

### C. Run All notebook 02 hiện tại thực sự làm gì?

Đọc v7, chạy preprocess, tạo split/folds, fit/CV 12 tổ hợp chính và 4 thí nghiệm
bổ sung, đánh giá test, vẽ bảng/hình, lưu model, tải lại và dự đoán mới. Có tính
toán và huấn luyện thật; không chỉ hiển thị bảng metrics cũ từ file.

Output: `spam_ham_ml_only.joblib`, `metrics.csv`, `bootstrap_ci.csv`,
`run_manifest.json` và output trong notebook. Lần kiểm chứng mới ở
`.work/runs/readable_training_verified/artifacts/`.

## 4. Chạy cell khác chạy file riêng thế nào?

| Cách chạy | Ý nghĩa |
|---|---|
| `import module` | Nạp định nghĩa/hằng số của module lần đầu trong kernel; không tự chạy khối `if __name__ == '__main__'`. |
| `def ...` trong cell | Tạo hàm; thân hàm chưa chạy cho tới khi gọi. |
| `show_code(...)` | Chỉ đọc và hiển thị source; không thực hiện công việc trong source. |
| `flatten_posts(...)`, `clean_facebook_comments(...)`, `fit_experiment(...)` | Gọi hàm thật từ module, có tính toán/I/O theo hàm đó. |
| Cell gọi `subprocess ... python -m ...` | Tạo process Python mới và chạy CLI thật với tham số được ghi trong cell. |
| `python -m pipeline.collect.scrape_ptit_groups ...` | Chạy runner live thật; cần phiên/quyền/endpoint hoạt động, có thể ghi dữ liệu mới. |

Notebook giữ biến trong kernel giữa các cell nên có thể sai trạng thái nếu chạy
lộn thứ tự. Restart Kernel → Run All giúp kiểm tra một lượt sạch. Script CLI bắt
đầu bằng process mới, nhận tham số, không tự có bảng/hình trong notebook.

Nếu cùng hàm, dữ liệu, tham số và môi trường thì phép xử lý là cùng code; khác
chủ yếu ở điều phối, trạng thái kernel và cách hiển thị. Run All **chỉ chạy những
nhánh được gọi trong notebook**, không mặc nhiên chạy mọi file trong project.
Do đó notebook 01 là lượt offline từ snapshot/cache, chưa phải lượt Facebook →
LLM mới; notebook 02 là lượt huấn luyện/đánh giá thật từ v7.

## 5. Đánh giá hiện tại và giới hạn

Phần code cốt lõi có đủ các ý xử lý/đặc trưng/NB/SVM/đánh giá; test đã kiểm tra
ID, rò rỉ theo nhóm, sampling train, save/load và đầu vào demo. Không có căn cứ
khẳng định mọi nguồn/nhãn đúng tuyệt đối hoặc đã hoàn tất ý 4 bổ sung.

Mô hình chọn theo CV: TF-IDF + SVM, giữ stop words. Trên 7.474 mẫu test, theo nhãn v7:

| Chỉ số | Kết quả |
|---|---:|
| Accuracy | 94,19% |
| Precision spam | 88,18% |
| Recall spam | 62,78% |
| F1 spam | 0,7334 |
| Ham đúng / ham nhầm spam | 6.443 / 80 |
| Spam bắt được / spam bỏ sót | 597 / 354 |

Đây là baseline có precision khá, nhưng còn bỏ sót khoảng 37,2% spam. Không thể
dựa vào Accuracy cao để nói hệ thống đã hoàn hảo, do dữ liệu lệch ham/spam.
Các ví dụ vài câu demo không phải bằng chứng độ chính xác chung.

Kiểm tra model đã lưu: API nhận raw text cho kết quả trùng luồng input thực nghiệm
trên toàn bộ 7.474 test IDs; save/load–predict cũng có kiểm chứng. Điều này xác
nhận tính nhất quán triển khai, không đảm bảo mọi comment ngoài thực tế đều đúng.
`score` của SVM là decision score, không phải xác suất phần trăm.

Trong bốn câu demo đang có, model gán tên riêng “Nguyễn Văn An” và dấu “.” thành
spam. Không nên diễn giải hai dự đoán đó là đúng tuyệt đối: bình luận ngắn/tag
tên cần ngữ cảnh và có thể là lỗi. Các ví dụ này giúp thấy giới hạn thật, không
được sửa bằng quy tắc đặc biệt chỉ để làm đẹp demo.

Còn giới hạn nguồn Phase 1/lịch sử 1.366 thay đổi nhãn v7, 27 nhóm text giống nhau
khác nhãn và thiếu ngữ cảnh khi phân loại. Chưa chạy lại crawl/AI live trong lượt
sửa UI. Không sửa nhãn hoặc chọn dataset theo điểm test. Phần hiện đại để sau.

Split hiện tại nhóm theo text đã chuẩn hóa, chưa phải đánh giá trên group Facebook
hoặc thời gian hoàn toàn khác. Nó hạn chế rò rỉ text trùng, không chứng minh mọi
bình luận liên quan cùng một bài/nhóm đều độc lập giữa train và test.

Bằng chứng: `results/notebook_display/`, `results/verified_run/`, hồ sơ nguồn ở
`docs/provenance/`; các bản trước lưu trong `results/history/`.
