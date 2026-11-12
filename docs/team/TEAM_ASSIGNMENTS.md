# Phân công học và trình bày project hiện tại

**Thành viên: Cương – Bảo – Phi – Phương.** Phân công này thay bản cũ; bao phủ
từ bình luận Facebook đến đánh giá và demo viBERT. Đây là phân công **học/giải
thích**, không phải khẳng định ai là tác giả mọi file được giao.

## 1. Bốn phần nối tiếp

| Thành viên | Phần chính | Notebook phụ trách | Bàn giao |
|---|---|---|---|
| **Cương** | Dữ liệu, crawler, hai lớp tiền xử lý và thiết kế chia tập | 01 mục 1–4; 02 mục 1–4; 03 mục 2 | Bảng sạch/ID; CSV token; train/test/fold và split viBERT |
| **Bảo** | Gán nhãn, kiểm chất lượng nhãn và đánh giá/phân tích lỗi truyền thống | 01 mục 5–6 và nhận xét; 02 mục 6–6.3, 7 | CSV có nhãn; cách đọc test, FP/FN, PR/AP, bootstrap |
| **Phi** | Hai đặc trưng, NB/SVM, lựa chọn theo CV và ba chiến lược trên SVM; lưu/demo S3 | 02 mục 5, 5.1, 5.2 ở mức dẫn nhập, 8–9 | B1–B4/S1–S3; cấu hình S3 và bundle `.joblib` |
| **Phương** | viBERT: tokenization, fine-tune, chọn loss/epoch, test chung và ứng dụng | 03 mục 1, 3–9; 02 mục 6.4 | M1 safetensors/tokenizer, bảng S3–M1, giới hạn và demo |

Mỗi phần có một khối lý thuyết, một khối code và một thực hành. Cương có nhiều
bước dữ liệu/hạ tầng; Bảo có nhiều module nhãn nhưng tập trung vào luồng ID và
đánh giá; Phi chịu phần CV/thuật toán; Phương chịu Transformer/GPU. Không chia
theo số dòng vì độ khó giữa code API, thống kê và mô hình khác nhau. Mỗi người
dự kiến 6–8 phút trình bày chính, cùng tham gia hỏi đáp và rà chéo.

## 2. Cương — Dữ liệu và thiết kế thực nghiệm

### Kiến thức cần học

- Post, comment, reply; JSON lồng nhau và CSV phẳng; metadata group/post/thời
  gian; crawler chỉ thu nội dung, chưa tạo nhãn spam/ham.
- Unicode NFC, HTML entity, regex URL/điện thoại, khử trùng theo khóa nội dung,
  giữ `text` gốc và `cleaned_text`; khác nhau giữa làm sạch cho nhãn và token
  hóa cho bộ phân loại. Tách từ tiếng Việt, từ dừng và token đặc biệt.
- Data leakage, train/validation/test, nhóm văn bản, khóa chuẩn hóa, CV năm
  fold. Tập validation một fold trong viBERT khác CV năm fold của SVM.
- Giới hạn chia theo văn bản: không đồng nghĩa chia theo tác giả/nhóm/thời
  gian; cách tiền xử lý theo quy tắc khác học vocabulary/IDF từ train.
- Venv/kernel và đầu vào/đầu ra, phân biệt đọc dữ liệu đã lưu với gọi Facebook.

### Code cần hiểu, theo thứ tự

| File | Đoạn cần đọc và câu cần giải thích |
|---|---|
| `pipeline/collect/scrape_ptit_groups.py` | `PTIT_GROUPS`, `run_scraper`, `on_batch`, `on_page`, `SingleFileWriter.save_records`: nhận nguồn, lấy bài/comment, lưu JSON/state |
| `external/facebook_scraper/group_post_scraper_v2.py`, `main.py`, `comment_scraper.py` | Vai trò `fetch_posts` và `fetch_comments_for_post`; hiểu điểm gọi/cấu trúc trả về, không cần học thuộc mọi payload Facebook |
| `pipeline/collect/json_to_csv.py` | `flatten_posts`: comment/reply thành dòng; CLI có tùy chọn lọc tham chiếu, notebook dùng trực tiếp hàm |
| `pipeline/prepare/clean_for_labeling.py` | `preprocess`, `clean_facebook_comments`: làm sạch, khử trùng/lọc, giữ text gốc |
| `pipeline/prepare/preprocess_for_model.py` | `clean_text`, `mask_names`, `preprocess`: xử lý tên/emoji/URL/SĐT, underthesea, hai nhánh từ dừng |
| `pipeline/modeling/training_utils.py` | `norm_key`, `bucket`, `grouped_folds`: giữ cùng khóa trong cùng tập/fold; phần chiến lược do Phi giải thích |
| `pipeline/modern/data.py` | `normalize_text`, `prepare_data`, `split_summary`: giữ cùng test, chọn fit/validation từ train |
| `tools/dataset_versions/merge_labeled_data.py`, `audit_source.py` | Biết mục đích tiện ích và giới hạn dữ liệu, không coi tool là một bước train hoặc tự tạo v7 |
| `scripts/check_environment.py`, `scripts/run_notebook.py` | Chọn Python, kernel mới, đầu ra và thời điểm ghi notebook; `presentation/notebook_display.py` chỉ hiển thị |

### Nội dung notebook/output phải trình bày được

- **01 mục 1–4:** nhóm/link và comment thật; 2.170 bài, 13.833 nội dung, 13.704
  không rỗng, 10.974 sau làm sạch. Bàn giao cho Bảo bảng có text/metadata; Bảo
  giải thích nối ID và gợi ý nhãn ở chặng tiếp theo.
- **02 mục 1–3:** v7 38.233 mẫu, 32.907 ham/5.326 spam; so cùng ID trước/sau
  xử lý; toàn bộ file `data/processed/v7_preprocessed.csv` được dùng tiếp, không
  chỉ vài dòng ví dụ. Phần giữ/bỏ từ dừng chọn bằng CV do Phi trình bày.
- **02 mục 4 / 03 mục 2:** 30.759 train/7.474 test; viBERT phát triển trên
  24.815 fit/5.944 validation, sau lựa chọn mới fit toàn train.

### Thực hành và tự kiểm tra

Chạy 01 bằng `.venv`, kiểm CSV và số dòng; mở 02 đến mục 4, chỉ ra một bình luận
trước/sau xử lý và số mẫu/fold. Đọc `tests/test_data_pipeline.py`,
`test_collection_provenance.py`, `test_notebook_runner.py`, `test_modern_data.py`.
Phải trả lời: “Tại sao xử lý văn bản ở cả 01 và 02?”, “Làm sao tránh mẫu cùng
khóa nằm ở hai tập?”, “Run All 01 có đăng nhập Facebook không?”.

## 3. Bảo — Nhãn và đánh giá mô hình truyền thống

### Kiến thức cần học

- Tiêu chí ham/spam trong guideline, bình luận chào mời so với trao đổi thông
  thường; thiếu ngữ cảnh có thể gây nhãn sai. Không mặc định tên riêng là spam.
- AI-assisted labeling: prompt, batch, workers, retry, cache; gợi ý khác nhãn
  đã tiếp nhận; confidence do AI trả về chưa phải xác suất đã calibration.
- ID ổn định, nối theo nội dung thay vì vị trí dòng; Argilla suggestion/response,
  trạng thái submitted, auto-accept 0,75 và phần cần người duyệt.
- TP/TN/FP/FN, Accuracy, Precision/Recall/F1 lớp spam, PR/AP; vì sao Accuracy
  riêng lẻ gây hiểu nhầm khi lệch lớp. Bootstrap CI 95% không tự chứng minh
  mô hình này vượt mô hình kia có ý nghĩa thống kê.
- Phân biệt lỗi nhãn, lỗi tiền xử lý và lỗi mô hình; ví dụ sai không đủ chứng
  minh phải đổi nhãn. CV chọn cấu hình, test báo cáo cấu hình đã chốt.

### Code cần hiểu

| File | Đoạn trọng tâm |
|---|---|
| `pipeline/label/ai_assisted_spam_ham.py` | Prompt/guideline; `predict_batch_with_glm`, `load_cache`, `save_cache`, `main`: request/parse, batch, lưu tiến độ, upload tùy chọn |
| `pipeline/label/upload_to_argilla.py` | `DEFAULT_GUIDELINES`, field/question, record ID, log theo batch |
| `pipeline/label/cached_annotations.py` | `attach_annotation_ids`: khóa nội dung và kiểm `cleaned_text` trước khi nhận cache |
| `pipeline/label/label_utils.py` | `record_ids`, `normalize_prediction`, `join_metadata`: kiểm nhãn/confidence/ID và bảo toàn nội dung |
| `pipeline/label/auto_accept_labels.py` | Tách đủ/chưa đủ ngưỡng; `--skip-argilla`; các cột `official_label`, `ai_confidence`, `is_auto_accepted` |
| `pipeline/label/download_labeled_data.py` | `extract_submitted_response`, nối metadata; nhãn submit khác suggestion |
| `pipeline/modeling/experiments.py` | `evaluate`, `boot_ci`; phần fit/grid do Phi phụ trách |
| Notebook 02 cell đánh giá/lỗi | Cách lấy prediction/score của đúng record, tính FP/FN, vẽ ma trận/PR, lấy ví dụ của `best` |

### Nội dung notebook/output phải trình bày được

- **01 mục 5–6:** 6.571 gợi ý khớp và 6.289 mẫu đủ điều kiện (5.190 ham, 1.099
  spam); chỉ ra ID, text, cleaned_text, label, confidence trong CSV. Notebook
  dùng phản hồi đã lưu, không gọi lại GLM. Mẫu chưa đủ điều kiện không vào CSV.
- **02 mục 6–6.3:** bảng B1–B4, bảng chiến lược B4/S1/S2/S3, FP/FN, hai nhóm
  ma trận và hai nhóm PR. B4 được dùng lại; đổi ngưỡng không đổi thứ tự score/AP.
- **02 mục 7:** giải thích một FP và một FN của S3, liên hệ dữ liệu/nhãn/ngữ
  cảnh. Không gán nguyên nhân chắc chắn khi chưa đủ căn cứ.
- S3 trên test: Precision 0,8374, Recall 0,6877, F1 0,7552; FP=127, FN=297.
  Đối chứng luôn ham có Accuracy 87,28% nhưng F1 spam=0; đây không phải lỗi tính.

### Thực hành và tự kiểm tra

Chạy offline các bước nối/tiếp nhận cache theo README; dùng model/kết quả đã
lưu để kiểm lại ma trận từ `artifacts/spamham/test_predictions.csv`. Đọc
`tests/test_cached_annotations.py` và phần label trong `test_restructured_modules.py`.
Phải trả lời: “0,75 khác ngưỡng SVM/viBERT ở đâu?”, “FN cao ảnh hưởng gì?”,
“Vì sao hai đường PR có thể trùng nhưng F1 tại ngưỡng lại khác?”.

## 4. Phi — Đặc trưng, huấn luyện và lựa chọn S3

### Kiến thức cần học

- BoW, TF-IDF (TF/IDF, sublinear TF), n-gram; biểu diễn ma trận thưa.
- MultinomialNB cho đặc trưng đếm/TF-IDF không âm, làm trơn alpha. SVM tuyến tính (`LinearSVC`), margin, siêu tham số C.
- scikit-learn Pipeline/fit/transform, GridSearchCV năm fold, multi-metric và
  refit F1; học đặc trưng trong fold, không fit trước trên cả train+test.
- Balanced: `n/(2*n_lớp)`; under-sampling 2:1 chỉ train; OOF decision score và
  chọn ngưỡng. Ba chiến lược riêng, không cộng dồn. Chỉnh ngưỡng không cân
  bằng lại số lượng mẫu, cũng không phải xác suất.
- `.py` định nghĩa cách làm, đối tượng đã fit chứa tri thức, `.joblib` lưu
  pipeline/ngưỡng/cấu hình. CV chọn S3, test không dùng chọn lại.

### Code cần hiểu

| File | Đoạn trọng tâm |
|---|---|
| `pipeline/modeling/experiments.py` | `make_pipe`, các lưới ngram/alpha/C; `fit_experiment`, refit F1 và các trường record |
| `pipeline/modeling/training_utils.py` | `sample_training_indices`, `undersampled_search`, `mean_cv_scores`, `tune_svm_threshold`, `predict_record` |
| `pipeline/inference/predict.py` | `predict_spam`: tiền xử lý theo bundle, score_type, áp ngưỡng đã lưu |
| Notebook 02 mục 5/5.1/8 | Đối chiếu từ dừng, đánh mã B/S, `CANDIDATES`, chọn `best`, `joblib.dump/load`, manifest và kiểm dự đoán |

### Nội dung notebook/output phải trình bày được

- **02 mục 5:** bốn tổ hợp × hai cách từ dừng được xem xét ở CV, không phải
  tám mô hình cuối; nhánh giữ từ dừng đi tiếp. Bảng có CV Precision/Recall/F1.
- **02 mục 5.1:** TF-IDF+SVM dẫn đầu cơ sở theo CV nên khảo sát sâu trên cùng
  mô hình. B4 không điều chỉnh; S1 balanced; S2 under-sampling; S3 ngưỡng OOF.
  Chưa khảo sát mọi chiến lược trên cả bốn tổ hợp.
- S3: C=0,3, ngram=(1,2), ngưỡng −0,1776279252, CV F1 0,7876. Score > ngưỡng
  → spam; ngưỡng âm không phải số xác suất lỗi. S3 chỉ nhỉnh hơn S2 trên test.
- **02 mục 5.2:** dẫn nhập sang viBERT, không giải thích M1 như thuật toán thứ
  ba ghép vào BoW/TF-IDF. Phương trình bày chi tiết nhánh đó.
- **02 mục 8–9:** bundle là S3 trong phạm vi truyền thống, không bị M1 ghi đè;
  demo dùng cùng ngưỡng và tiền xử lý. Nêu rõ đánh đổi Precision/Recall.

### Thực hành và tự kiểm tra

Chạy 02 bằng runner, tìm `stopword_cv.csv`, `imbalance_cv.csv`, `metrics.csv`
và `.joblib`; nạp model, dự đoán câu mới. Đọc
`tests/test_preprocessing_and_training.py` và phần model trong
`test_restructured_modules.py`. Phải trả lời: “4+3 vì sao không phải 8?”,
“OOF là gì?”, “Vì sao không chọn bằng test?”, “Tại sao score âm vẫn có thể là spam?”.

## 5. Phương — Fine-tune, đánh giá và ứng dụng viBERT

### Kiến thức cần học

- BERT encoder/ngữ cảnh hai chiều ở mức kiến trúc, pretrained khác fine-tuned;
  thêm classification head hai lớp, fine-tune encoder và head cùng nhau.
- Tokenizer cased, WordPiece, token ID, `[CLS]/[SEP]/[PAD]/[UNK]`, `##`,
  attention mask, truncation/padding. Giữ hoa/thường/từ dừng; không đưa token
  underthesea vào viBERT. `[UNK]` mất thông tin, không phải lỗi mã hóa CSV.
- Logits, softmax, cross-entropy thường/có trọng số; learning rate, epoch,
  batch/gradient accumulation, warmup, weight decay, FP16 ở mức giải thích
  tác dụng. Không cần học thuộc phép đạo hàm mọi lớp Transformer.
- Chọn loss/epoch bằng validation, khởi tạo lại pretrained và fit toàn train;
  cùng test S3 để so sánh công bằng. Validation một fold khác CV năm fold.
- Colab/Drive, checkpoint/resume, `save_pretrained`, safetensors/tokenizer,
  `eval()`/inference; review không phải train. Score SVM khác xác suất viBERT.

### Code cần hiểu

| File | Đoạn trọng tâm |
|---|---|
| `pipeline/modern/vibert_config.json` | Checkpoint/revision, seed, max_length 128, LR 2e-5, batch 8×accumulation 2, ba epoch, ngưỡng 0,5 |
| `pipeline/modern/data.py` | `normalize_text`, `class_weights`; `prepare_data` do Cương giải thích chính, Phương cần hiểu đầu vào/bàn giao |
| `pipeline/modern/train.py` | `CommentDataset`, `ClassificationTrainer.compute_loss`, `load_pretrained`, `make_arguments`, `fit_variant`, `select`, `finalize`, `compare_with_svm`, `run_experiment` |
| `pipeline/modern/predict.py` | `VibertPredictor`: nạp local, tokenizer/batch, logits→softmax→ngưỡng; không train lại |
| `scripts/prepare_colab_bundle.py` | Chọn file chuyển, không đưa phiên/API; quy trình trên Colab theo hướng dẫn |
| Notebook 03 | `MODE`, đường dẫn output, đọc lịch sử/bảng, ma trận/PR, lỗi, nạp model và demo |

### Nội dung notebook/output phải trình bày được

- **03 mục 1, 3–5:** giải thích bảng token, ký tự Nhật có sẵn trong `text`,
  0,77% train vượt 128 token; phân biệt số `[UNK]` và số mẫu bị cắt.
- Hai loss: thường validation F1 0,82048, balanced 0,81916; chọn thường/epoch 3,
  rồi fit lại 30.759 train từ pretrained. Test chưa được dùng để chọn loss/epoch.
- **03 mục 6–7 / 02 mục 6.4:** M1 test Accuracy 0,9635, Precision 0,8942,
  Recall 0,8086, F1 0,8493, AP 0,9108; FP=91, FN=182. So S3: bắt thêm 115 spam,
  giảm 36 ham nhận nhầm; chưa kết luận ưu thế mọi dataset hoặc nhiều seed.
- **03 mục 8–9:** `.safetensors` khác `.joblib`; cả hai là trọng số/trạng thái
  đã học. Xác suất cao chưa bảo đảm đúng; nêu hạn chế câu ngắn/chỉ dấu câu.

### Thực hành và tự kiểm tra

Chạy 03 bằng `.venv-modern`, `MODE="review"`; nạp model dự đoán, chỉ ra file
trọng số, tokenizer, lịch sử và `verification.json`. Biết cách train/resume
qua Colab nhưng không cần train lại chỉ để ôn. Đọc `tests/test_modern_data.py`,
`test_modern_training.py`: test BERT nhỏ kiểm chức năng, không phải kết quả M1.
Phải trả lời: “Vì sao không ghép TF-IDF vào viBERT?”, “Tại sao fit lại sau chọn
epoch?”, “Run All review có train không?”, “Model ZIP khác toàn bộ run ở đâu?”.

## 6. Kiến thức chung và cách phối hợp

Mọi thành viên cần thuộc mục tiêu ham/spam, luồng ba notebook, 38.233 mẫu v7,
vai trò train/CV/validation/test, S3 so với M1, các đầu ra và giới hạn dữ liệu.
Không cần thuộc mọi dòng vendor/API hoặc khẳng định model đúng mọi bình luận.

| Bàn giao | Người giải thích chính | Người kiểm tra chéo |
|---|---|---|
| Bảng sạch, ID và nhãn → CSV | Cương → Bảo | Bảo kiểm ID/nhãn, Cương kiểm text/metadata |
| CSV token/split → huấn luyện | Cương → Phi | Phi kiểm fit trong train/fold |
| S3 đã chọn → test/lỗi | Phi → Bảo | Bảo đối chiếu prediction với nhãn và ngưỡng |
| V7/test chung/S3 predictions → viBERT | Cương, Phi → Phương | Phương đối chiếu ID/nhãn trước bảng S3–M1 |
| Kết quả cuối/hạn chế → trình bày | Bảo, Phương | Cả nhóm thống nhất cách đọc chỉ số |

Mỗi người chuẩn bị: một sơ đồ đầu vào→hàm→đầu ra; một ví dụ thật xuyên suốt
phần được giao; một lượt thực hành trong đúng venv; ba câu hỏi dễ nhầm và cách
trả lời. Bảo/Phương thống nhất cách dùng FP/FN và thang score; Cương/Phi thống
nhất cách giải thích split/CV và tiền xử lý.

Thứ tự thuyết trình gợi ý: **Cương (nguồn/làm sạch) → Bảo (nhãn) → Cương
(khảo sát/split) → Phi (đặc trưng/CV/chiến lược) → Bảo (test/lỗi) → Phi
(S3/demo) → Phương (viBERT/test/demo/kết luận)**. Có thể gộp lượt khi làm slide,
nhưng giữ thứ tự dữ liệu→học→đánh giá; không cần mở mọi file cùng lúc.

Phân công này không tạo báo cáo thay nhóm. Khi viết báo cáo sau, dùng số liệu
đã xác nhận và [ảnh đề](../assignment/de_bai.png), không dùng chỉ số/đường dẫn
của bản phân công cũ.
