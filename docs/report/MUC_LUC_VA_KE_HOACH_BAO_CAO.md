# Mục lục và kế hoạch viết báo cáo

**Ngày lập:** 06/11/2026. **Trạng thái:** đề cương để duyệt, chưa phải báo cáo.
Đây là file làm việc tạm; chỉ xóa sau khi báo cáo hoàn chỉnh được nghiệm thu.

**Tên đề tài đề xuất:**

> XÂY DỰNG HỆ THỐNG PHÂN LOẠI SPAM/HAM CHO BÌNH LUẬN SINH VIÊN PTIT TRÊN FACEBOOK

Căn cứ [ảnh đề](../assignment/de_bai.png), bố cục giữ đúng: **Giới thiệu bài
toán → Cơ sở lý thuyết → Thực nghiệm và kết quả → Tài liệu tham khảo**. Theo
cách tổ chức đã yêu cầu, phần giới thiệu đặt tên **Mở đầu**, cơ sở lý thuyết là
**Chương 1**, thực nghiệm/kết quả là **Chương 2**; thêm **Kết luận** ngắn trước
tài liệu tham khảo. Không cần thêm một chương lý thuyết hoặc thực nghiệm khác.

Thu thập thuộc **Notebook 01**, mô hình truyền thống thuộc **02**, viBERT thuộc
**03**. Ba chặng được tách thành ba mục trong Chương 2, sau đó đánh giá chung.
Các ký hiệu `[Rxx]` dưới đây liên kết tới danh mục nguồn ở phần D; khi viết báo
cáo sẽ đổi thành số tài liệu theo thứ tự trích dẫn, không giữ chữ R.

## A. Mục lục báo cáo và nội dung dự kiến

### Phần đầu báo cáo

- **Trang bìa:** trường/khoa, học phần, tên đề tài, giảng viên, lớp/nhóm, họ tên
  và MSSV của bốn thành viên, địa điểm và thời gian.
- **Mục lục:** tạo tự động từ các Heading sau khi dàn trang.
- **Danh mục bảng, danh mục hình:** tên và số trang tự động.
- **Danh mục từ viết tắt:** NLP, BoW, TF-IDF, NB, SVM, BERT, CV, OOF, TP/TN/FP/FN,
  AP, GPU; giải nghĩa những từ thực sự xuất hiện trong bài.

Ảnh đề không yêu cầu lời cảm ơn/cam đoan; chỉ bổ sung nếu mẫu học phần yêu cầu.

### MỞ ĐẦU

| Tiểu mục | Nội dung chính |
|---|---|
| Bối cảnh và lý do chọn đề tài | Bình luận trong cộng đồng sinh viên; nội dung chào mời/rác ảnh hưởng việc trao đổi; nhu cầu hỗ trợ phân loại |
| Bài toán và mục tiêu | Đầu vào là văn bản bình luận, đầu ra ham/spam; xây dựng quy trình dữ liệu, so sánh biểu diễn và bộ phân loại, bổ sung viBERT |
| Đối tượng, dữ liệu và phạm vi | Bình luận/comment-reply PTIT trên Facebook; dataset v7 dùng thực nghiệm; phân loại dựa trên text, giới hạn về ngữ cảnh và nhãn |
| Phương pháp và sản phẩm | Làm sạch/tách từ; BoW/TF-IDF/FastText với NB/SVM; fine-tune viBERT; đánh giá test, lưu mô hình và demo bình luận mới |
| Bố cục báo cáo | Vai trò Chương 1, Chương 2 và Kết luận; nêu ngắn, không diễn giải trước mọi kết quả |

Mở đầu dùng văn xuôi, khoảng 1–2 trang. Không biến thành phần hướng dẫn cài đặt
hoặc liệt kê toàn bộ file của repo.

### CHƯƠNG 1. CƠ SỞ LÝ THUYẾT

#### 1.1. Bài toán phân loại văn bản

- **1.1.1. Học có giám sát và phân loại nhị phân:** mẫu, nhãn, đặc trưng, mô hình,
  học từ dữ liệu và suy luận; quy ước ham/spam, spam là lớp dương. `[R01, R06]`
- **1.1.2. Đặc điểm văn bản bình luận tiếng Việt:** câu ngắn, tên riêng,
  teencode, emoji, URL, số điện thoại, thiếu ngữ cảnh; những đặc điểm quan sát
  được trong dataset. Phần quan sát dữ liệu dẫn sang Chương 2, không gán thành
  một kết luận phổ quát từ tài liệu.
- **1.1.3. Gán nhãn có hỗ trợ AI:** hướng dẫn nhãn, gợi ý từ mô hình ngôn ngữ,
  confidence, nhãn được tiếp nhận và phần cần thẩm định; chất lượng nhãn và
  nhiễu nhãn. Confidence do AI trả về chưa phải độ tin cậy đã hiệu chỉnh. `[R24]`

#### 1.2. Làm sạch và tiền xử lý văn bản

- **1.2.1. Chuẩn hóa và bảo toàn dữ liệu:** Unicode NFC, HTML, khoảng trắng,
  chữ hoa/thường; văn bản gốc và văn bản xử lý, bản ghi trùng/rỗng, ID và
  metadata. Giới thiệu khái niệm; quy tắc cụ thể của nhóm đặt ở Chương 2. `[R01, R02]`
- **1.2.2. Tách từ tiếng Việt và từ dừng:** từ/âm tiết/token, từ nhiều âm tiết,
  underthesea, stop words và ảnh hưởng của việc giữ/loại. `[R01, R03]`
- **1.2.3. Xử lý tín hiệu đặc biệt:** thay URL/điện thoại bằng token, tín hiệu
  emoji, xử lý tên theo quy tắc; khác nhau giữa làm sạch để gán nhãn và chuẩn
  bị đầu vào từng mô hình. Quy tắc che tên không bảo đảm ẩn danh hoàn toàn.

#### 1.3. Các phương pháp biểu diễn văn bản

- **1.3.1. Bag-of-Words:** vocabulary, vector đếm, ma trận thưa, unigram/bigram,
  ưu điểm và hạn chế về thứ tự/ngữ cảnh. `[R01, R04]`
- **1.3.2. TF-IDF:** TF, DF, IDF, sublinear TF, làm trơn IDF, chuẩn hóa L2;
  phân biệt công thức tổng quát với cấu hình scikit-learn được dùng. `[R01, R04]`
- **1.3.3. FastText:** word embedding, skip-gram, character n-gram/subword,
  từ ngoài từ vựng (OOV); từ vector từ đến vector câu bằng mean hoặc trung
  bình có trọng số TF-IDF. Cách pooling là thiết kế của project, không mặc
  nhiên là một bộ phân loại do bài báo FastText đề xuất. `[R05]`
- **1.3.4. Đối chiếu ba biểu diễn:** thưa/dày, số chiều, khả năng biểu diễn từ
  hiếm và ngữ cảnh, chi phí; bảng khái niệm, chưa đưa điểm test vào đây.

#### 1.4. Các mô hình phân loại truyền thống

- **1.4.1. Naive Bayes:** định lý Bayes, prior/likelihood/posterior, giả định
  độc lập có điều kiện, quyết định theo hậu nghiệm. `[R01, R06]`
- **1.4.2. MultinomialNB và GaussianNB:** mô hình đếm và làm trơn cộng alpha
  (Laplace khi alpha = 1); phân phối Gaussian theo từng đặc trưng và tham số
  làm trơn phương sai. Giải thích lựa chọn cho BoW/TF-IDF và embedding, không
  coi giả định Gaussian tự động đúng chỉ vì vector có giá trị âm. `[R06]`
- **1.4.3. SVM tuyến tính:** siêu phẳng, margin, regularization, C, hinge và
  squared hinge, decision score, ngưỡng quyết định; StandardScaler cho nhánh
  FastText. Công thức khi viết phải khớp `LinearSVC` với loss mặc định
  `squared_hinge`, không trình bày như nhóm dùng kernel RBF. `[R07–R09]`

#### 1.5. Mô hình ngôn ngữ tiền huấn luyện và viBERT

- **1.5.1. Transformer encoder:** embedding đầu vào/vị trí, self-attention,
  Q/K/V, multi-head attention, feed-forward, residual/layer normalization;
  sơ đồ ở mức cần để hiểu encoder, không chứng minh toàn bộ kiến trúc. `[R10, R11]`
- **1.5.2. BERT và học chuyển giao:** biểu diễn ngữ cảnh hai chiều, tiền huấn
  luyện MLM/NSP trong BERT gốc, pretrained và fine-tuning. Nhóm fine-tune mô
  hình có sẵn, không tiền huấn luyện BERT lại từ đầu. `[R11]`
- **1.5.3. viBERT của FPT.AI:** nguồn mô hình, checkpoint
  `FPTAI/vibert-base-cased`, khác nhiệm vụ tiền huấn luyện với phân loại spam
  PTIT. Bài báo nguồn do repo viBERT dẫn là nghiên cứu sequence tagging;
  không lấy điểm NER/POS trong đó làm bằng chứng cho bài toán spam. `[R12, R13]`
- **1.5.4. WordPiece và đầu vào BERT:** subword, `##`, `[CLS]/[SEP]/[PAD]/[UNK]`,
  token ID, attention mask, padding/truncation, tokenizer cased. `[R11, R14]`
- **1.5.5. Fine-tune cho phân loại hai lớp:** encoder/head, logits, softmax,
  cross-entropy thường/có trọng số, AdamW, learning rate, weight decay,
  warmup, epoch, batch, gradient accumulation và FP16. Giải thích tác dụng
  ngắn gọn, không mở rộng thành một chương tối ưu số. `[R15, R16, R23]`

#### 1.6. Thiết kế thực nghiệm và xử lý mất cân bằng

- **1.6.1. Train, validation, test và rò rỉ dữ liệu:** phân chia theo nhóm văn
  bản, fit/transform trong Pipeline; vocabulary, IDF, scaler và embedding chỉ
  học trên phần train được phép. `[R17]`
- **1.6.2. CV, tìm siêu tham số và OOF:** k-fold CV, GridSearchCV, trung bình
  F1 fold, dự đoán out-of-fold, lựa chọn loss/epoch bằng validation, seed và
  khả năng tái lập. Phân biệt điểm dùng lựa chọn với test cuối. `[R17, R23]`
- **1.6.3. Mất cân bằng và trọng số lớp:** lớp đa số/thiểu số, chi phí lỗi,
  trọng số balanced và weighted loss; không đồng nhất số mẫu với trọng số. `[R18, R16]`
- **1.6.4. Giảm mẫu và điều chỉnh ngưỡng:** random under-sampling chỉ train,
  ngưỡng score/xác suất, đánh đổi Precision–Recall; chỉnh ngưỡng không làm
  thay đổi phân bố số lượng mẫu. `[R19, R20]`

#### 1.7. Đánh giá và phân tích mô hình

- **1.7.1. Ma trận nhầm lẫn và các chỉ số:** TP/TN/FP/FN, Accuracy, Precision,
  Recall, F1 lớp spam; trường hợp không dự đoán spam và quy ước chia cho 0. `[R21]`
- **1.7.2. Đường Precision–Recall và AP:** đường cong theo score, điểm hoạt
  động tại ngưỡng; Average Precision khác tích phân hình thang PR-AUC và khác
  Accuracy. Không gọi decision score SVM là xác suất. `[R21]`
- **1.7.3. Khoảng tin cậy bootstrap và phân tích lỗi:** lấy mẫu lại, percentile
  CI, giới hạn khi resample từng dòng/các mẫu phụ thuộc; FP/FN và kiểm tra ngữ
  cảnh. CI của từng mô hình chưa chứng minh chênh lệch có ý nghĩa thống kê. `[R22]`

**Mức độ trình bày:** với phương pháp chính, viết khái niệm → công thức/biến →
cách hoạt động → ưu/nhược điểm → liên hệ ngắn với bài toán. Những khái niệm hỗ
trợ như seed, FP16 hoặc checkpoint chỉ cần đoạn giải thích, không cần mục lý
thuyết dài bằng NB/SVM/BERT. Không thêm thuật toán chưa sử dụng.

### CHƯƠNG 2. THỰC NGHIỆM VÀ KẾT QUẢ

#### 2.1. Thu thập và chuẩn bị dữ liệu

- **2.1.1. Quy trình tổng thể và môi trường:** sơ đồ dữ liệu → nhãn → hai nhánh
  mô hình → test → demo; Python/thư viện chính, CPU local và GPU Colab, vai
  trò của ba notebook. Không chép hướng dẫn cài đặt dài vào thân báo cáo.
- **2.1.2. Nguồn và phương pháp thu thập:** nhóm/link có trong JSON thực tế,
  cấu trúc post → comment → reply; runner PTIT, mã nền tham khảo, lưu JSON và
  state. Tách rõ mã được kế thừa và phần điều phối của nhóm. `[R25]`
- **2.1.3. Chuyển đổi và làm sạch:** JSON → CSV, lọc rỗng/trùng, bảo toàn
  metadata và text; ví dụ trước/sau cùng ID và bảng số lượng từng bước.
- **2.1.4. Gán nhãn Spam/Ham:** guideline, gợi ý AI, label/confidence,
  Argilla/tiếp nhận nhãn, ngưỡng 0,75, nối đúng ID, xuất CSV; phạm vi bước nào
  dùng phản hồi đã lưu trong notebook. Không khẳng định toàn bộ đã duyệt tay. `[R24]`
- **2.1.5. Tập dữ liệu thực nghiệm và khảo sát:** v7, schema, số dòng/nhãn,
  comment-reply, độ dài, bình luận thật, phân bố lệch lớp. CSV đầu ra của quy
  trình minh họa thu thập và v7 được ghi đúng phạm vi; không viết rằng một lần
  Run All 01 tự sinh v7. `[R26]`
- **2.1.6. Tiền xử lý cho hai nhánh:** văn bản/token underthesea và file
  `v7_preprocessed.csv` cho truyền thống; chuẩn hóa nhẹ text gốc cho viBERT;
  bảng minh họa cùng ID, giải thích vì sao không dùng chung một chuỗi token.
- **2.1.7. Phân chia dữ liệu và giao thức lựa chọn:** nhóm văn bản, train/test
  chung, năm fold cho truyền thống; fit/validation trong train của viBERT;
  các bước kiểm soát rò rỉ, giới hạn của chia theo văn bản.

Tương ứng **Notebook 01**, mục 1–4 của **Notebook 02** và phần chia tập của **03**.

#### 2.2. Xây dựng và huấn luyện mô hình truyền thống

- **2.2.1. Cấu hình BoW và TF-IDF:** vocabulary trong fold, min_df, n-gram,
  sublinear TF; bảng lưới tham số sử dụng.
- **2.2.2. Cấu hình FastText:** skip-gram 100 chiều, subword 3–5 ký tự, học
  trong fold, mean/TF-IDF pooling; scaler khi ghép SVM.
- **2.2.3. Sáu tổ hợp cơ sở B1–B6:** ba biểu diễn × hai họ mô hình; chỉ rõ
  MultinomialNB/GaussianNB/LinearSVC, ngưỡng mặc định và SVM không trọng số.
- **2.2.4. Tìm tham số và chọn tiền xử lý:** bảng CV của giữ/bỏ từ dừng,
  GridSearchCV và refit F1 spam; giữ từ dừng cho các bước tiếp, không tạo phụ
  lục đối chiếu lặp lại hoặc chấm test cho cả hai nhánh.
- **2.2.5. Khảo sát chiến lược trên TF-IDF+SVM:** B4 đối chứng, S1 balanced,
  S2 under-sampling 2:1, S3 ngưỡng OOF; hình số mẫu/trọng số, bảng CV và căn cứ
  chọn S3. Ba chiến lược thử riêng, chưa khảo sát mọi kết hợp trên cả sáu tổ hợp.

Tương ứng **Notebook 02 mục 5 và 5.1**. Bảng ở đây dùng **CV**, chưa phải bảng
test của mục 2.4. Tổng số ứng viên ở tiền xử lý đã chọn là **6 + 3 = 9**;
B4 dùng lại làm đối chứng, không tính thành một mô hình thứ mười.

#### 2.3. Xây dựng và fine-tune viBERT

- **2.3.1. Đầu vào và kiến trúc phân loại:** checkpoint/tokenizer cased, text
  gốc, head hai lớp, fine-tune encoder và head; bảng token và ví dụ `[UNK]`.
- **2.3.2. Cấu hình huấn luyện:** max length 128, LR 2e-5, batch 8 và gradient
  accumulation 2, tối đa 3 epoch, warmup 0,1, weight decay 0,01, FP16; GPU A100.
- **2.3.3. So sánh hai loss trên validation:** thường/balanced, cách tính
  trọng số từ train, lịch sử loss/validation F1; chọn loss thường, epoch 3.
- **2.3.4. Huấn luyện cuối và lưu checkpoint:** khởi tạo lại pretrained, fit
  toàn bộ train với số epoch đã chọn; lưu model/tokenizer, ngưỡng xác suất
  0,5, kiểm tra nạp lại. Chưa phân tích bảng test tại đây.
- **2.3.5. Cách tái sử dụng thực nghiệm:** train trên Colab, checkpoint/resume,
  review/demo local; nêu ngắn để phân biệt lịch sử train với cell đọc kết quả.

Tương ứng **Notebook 03 mục 1–5**. Phần xây dựng mô hình hiện đại bổ sung được
trình bày như một nhánh phân loại hoàn chỉnh, không phải viBERT ghép lần lượt
với BoW/TF-IDF/FastText.

#### 2.4. Đánh giá và so sánh kết quả

- **2.4.1. Giao thức đánh giá:** cùng 7.474 ID/nhãn test; spam là lớp dương;
  Accuracy/Precision/Recall/F1/AP và FP/FN; cấu hình đã chốt trước đánh giá test.
- **2.4.2. Sáu tổ hợp truyền thống:** bảng B1–B6, đối chiếu biểu diễn/bộ phân
  loại, ma trận sáu tổ hợp; kết luận theo từng chỉ số.
- **2.4.3. Tác động của các chiến lược SVM:** bảng B4/S1/S2/S3 trên test, FP/FN,
  ma trận/đường PR theo phạm vi phù hợp; đối chiếu với điểm CV đã dùng chọn.
- **2.4.4. Kết quả viBERT và đối chiếu S3:** bảng chung S3/M1, ma trận M1,
  đường PR chung, chênh lệch số lỗi; không so CV F1 của SVM với validation F1
  của viBERT như hai phép đo tương đương.
- **2.4.5. Phân tích dự đoán và độ bất định:** ví dụ FP/FN có ID/text/nhãn/score
  của S3 và M1, đặc trưng hệ số SVM; bootstrap F1 truyền thống và giới hạn.
  Không gán khoảng bootstrap chưa tính cho viBERT.

Tương ứng **Notebook 02 mục 6–7** và **Notebook 03 mục 6–7**. Đây là phần báo
cáo hiệu quả test, khác mục 2.2/2.3 dùng điểm phát triển để chọn cấu hình.

#### 2.5. Lưu mô hình và ứng dụng phân loại bình luận mới

- **2.5.1. Đầu ra và cách nạp lại:** bundle S3 `.joblib` gồm pipeline/ngưỡng/
  cấu hình; thư mục M1 có safetensors, tokenizer và cấu hình suy luận.
- **2.5.2. Luồng dự đoán:** bình luận mới → chuẩn hóa phù hợp → transform/
  tokenizer → score/xác suất → ngưỡng → ham/spam. Nêu tên hàm/module chính.
- **2.5.3. Minh họa ứng dụng:** bảng vài câu mới và kết quả hai mô hình, có ví
  dụ dễ và trường hợp khó; dữ liệu demo không thay cho test. Sản phẩm hiện
  tại là demo trong notebook/hàm Python.

Tương ứng **mục 8 của Notebook 02/03**. Không đồng nhất file `.py` với trọng
số đã học, không gọi việc nạp model để demo là huấn luyện lại.

#### 2.6. Thảo luận và nhận xét thực nghiệm

- **2.6.1. Những kết quả đạt được:** hiệu quả từng biểu diễn, lựa chọn S3,
  tác động ngưỡng và lợi ích quan sát được của M1 trên test hiện tại.
- **2.6.2. Đánh đổi và chi phí:** Precision/Recall, số FP/FN; yêu cầu GPU khi
  fine-tune và khả năng demo CPU. Chỉ nhận xét chi phí định tính nếu chưa có
  benchmark công bằng, không tự lập bảng tốc độ/VRAM.
- **2.6.3. Giới hạn:** nhãn/ngữ cảnh, bình luận gần trùng, teencode/tên riêng,
  `[UNK]`, câu quá ngắn; một split/seed, CV dùng lựa chọn chưa phải nested CV,
  test chưa độc lập theo thời gian/nhóm, xác suất chưa calibration.

Phần này diễn giải ý nghĩa của kết quả; không chép lại mọi bảng chỉ số từ 2.4.

### KẾT LUẬN

- **Kết quả tổng quát:** đối chiếu mục tiêu mở đầu, những bước và sản phẩm đã
  hoàn thành; nêu một vài kết quả chính trong phạm vi thực nghiệm.
- **Hạn chế còn lại:** tổng hợp ngắn, không viết lại toàn bộ mục 2.6.
- **Hướng phát triển:** rà nhãn có người thẩm định, thêm dữ liệu/ngữ cảnh,
  test theo thời gian/nhóm, nhiều seed, calibration và triển khai giao diện/API
  nếu cần. Ghi là định hướng, không mô tả như chức năng đã có.

Khoảng một trang, không thêm lý thuyết hoặc kết quả mới chưa trình bày trước đó.

### TÀI LIỆU THAM KHẢO

Bài báo/sách cho nguyên lý; tài liệu chính thức cho công thức/triển khai;
nguồn dataset/checkpoint/crawler cho xuất xứ. Danh mục sơ bộ ở phần D.

### PHỤ LỤC — chỉ thêm khi cần

Lưới tham số dài, bảng bootstrap đầy đủ, danh mục file/model hoặc trích đoạn
code quan trọng. Phần bắt buộc của đề và kết quả chính phải ở thân báo cáo;
không cần phụ lục đối chiếu từ dừng đã được trình bày tại 2.2.4.

## B. Kế hoạch viết và bằng chứng sử dụng

### B1. Đối chiếu yêu cầu đề bài

| Yêu cầu trong ảnh đề | Vị trí báo cáo |
|---|---|
| Nguồn bình luận PTIT và phân loại spam/ham | Mở đầu, 1.1, 2.1 |
| Thu thập dữ liệu | 2.1.2–2.1.5 |
| Tiền xử lý, tách từ, từ dừng | 1.2, 2.1.6, 2.2.4 |
| Ít nhất hai phương pháp biểu diễn | 1.3, 2.2.1–2.2.3: thực hiện ba phương pháp |
| NB/SVM, chia huấn luyện và kiểm tra | 1.4, 1.6, 2.1.7, 2.2 |
| Accuracy, Precision, Recall, F1, Confusion Matrix; so sánh | 1.7, 2.4 |
| Mô hình hiện đại bổ sung | 1.5, 2.3, 2.4.4, 2.5 |
| Tiêu đề, giới thiệu, lý thuyết, thực nghiệm/kết quả, tham khảo | Toàn bộ bố cục ở phần A |

CV/OOF, PR/AP, bootstrap, đối chiếu từ dừng và các chiến lược cân bằng/ngưỡng
là phần bổ trợ đã triển khai; ảnh đề không bắt buộc riêng từng kỹ thuật đó.
Lý thuyết của các phần này vẫn được giải thích, nhưng dung lượng vừa đủ.

### B2. Thứ tự biên soạn

| Lượt | Công việc | Kết quả cần có trước khi chuyển bước |
|---|---|---|
| 1 | Duyệt mục lục, tên đề tài, phạm vi và quy cách | Chốt phần chính/bổ trợ; nhận thông tin bìa và mẫu trường nếu có |
| 2 | Lập bảng nguồn và ký hiệu | Mỗi nhóm khái niệm có nguồn gốc; thống nhất từ vựng và biến trong công thức |
| 3 | Viết Chương 1 | Định nghĩa/công thức đúng nguồn, đúng biến thể đang dùng; trích dẫn ngay tại đoạn tương ứng |
| 4 | Viết 2.1–2.3 từ notebook/code | Luồng dữ liệu, bảng split, cấu hình và tiêu chí lựa chọn; phân biệt CV/validation/test |
| 5 | Viết 2.4–2.6 từ CSV/hình đã xác nhận | Bảng test, phân tích lỗi, demo và giới hạn; không đổi mô hình theo test |
| 6 | Viết Mở đầu và Kết luận | Khớp phương pháp/sản phẩm/kết quả thực tế, tránh hứa vượt phạm vi |
| 7 | Dàn trang bản chỉnh sửa được và PDF dự kiến nộp | Heading/mục lục/caption/trích dẫn tự động; công thức và hình đọc rõ |
| 8 | Rà nội dung và xem từng trang xuất PDF | Không lệch số liệu, thiếu nguồn, cắt bảng/công thức, trang trắng hoặc tham chiếu lỗi |

Đề xuất quy mô khoảng **25–35 trang kể cả bảng/hình**, điều chỉnh theo mẫu học
phần và độ rõ ràng; không kéo dài để đạt số trang. Chương 1 khoảng 8–10 trang,
Chương 2 khoảng 12–16 trang; Mở đầu/Kết luận ngắn. Đây không phải hạn mức của đề.

### B3. Nguồn số liệu nội bộ và ánh xạ code

| Phần viết | Bằng chứng/module dùng để đối chiếu |
|---|---|
| Thu thập/chuyển bảng | `notebooks/01_thu_thap_va_gan_nhan.ipynb`, `pipeline/collect/`, `external/facebook_scraper/`, `data/collection_manifest.json` |
| Làm sạch/nhãn | `pipeline/prepare/clean_for_labeling.py`, `pipeline/label/`, `data/intermediate/`, `data/labeled/comments_labeled.csv` |
| Dataset/tiền xử lý/split | `data/final/labeled_v7.csv`, `data/processed/v7_preprocessed.csv`, `pipeline/prepare/preprocess_for_model.py`, `pipeline/modeling/training_utils.py` |
| Truyền thống và CV | Notebook 02, `pipeline/features/spam_features.py`, `pipeline/modeling/experiments.py`, `artifacts/spamham/stopword_cv.csv`, `imbalance_cv.csv` |
| Test/CI/lỗi S3 | `artifacts/spamham/metrics.csv`, `imbalance_comparison.csv`, `bootstrap_ci.csv`, `test_predictions.csv`, các hình ma trận/PR |
| viBERT phát triển và test | Notebook 03, `pipeline/modern/`, `artifacts/modern/vibert/run_config.json`, `validation_metrics.csv`, `selected.json`, lịch sử và `test_predictions.csv` |
| So sánh và kiểm chứng | `results/modern/comparison_with_svm.csv`, `results/verified_run/final_review.json`, `results/modern/local_validation.json` |
| Lưu và dự đoán mới | `pipeline/inference/predict.py`, `pipeline/modern/predict.py`, hai thư mục model trong `artifacts/` |

Trong báo cáo chỉ trích code ngắn nếu giúp giải thích thuật toán/luồng; không
chép toàn bộ notebook hoặc README. Kết quả là số liệu thực nghiệm của nhóm,
không phải dữ kiện lấy từ các bài báo lý thuyết.

### B4. Những số liệu cần thống nhất

| Đối tượng | Số liệu hiện có cần đối chiếu khi viết |
|---|---|
| Quy trình Notebook 01 | 5 nhóm, 2.170 bài; 13.833 comment/reply; 10.974 sau làm sạch; 6.289 mẫu được tiếp nhận, 5.190 ham/1.099 spam |
| V7 thực nghiệm | 38.233 mẫu; 32.907 ham/5.326 spam; train 30.759, test 7.474 |
| Phát triển viBERT | 24.815 fit, 5.944 validation trong train; giữ cùng 7.474 test với S3 |
| Lựa chọn S3 | CV F1 0,7876; C=0,3, ngram=(1,2), giữ từ dừng; ngưỡng decision score −0,1776279252 |
| Lựa chọn M1 | Loss thường, epoch 3; validation F1 0,82048 so với 0,81916 của balanced; ngưỡng xác suất 0,5 |
| Test S3 | Accuracy 0,9433; Precision 0,8374; Recall 0,6877; F1 0,7552; AP 0,8117; FP 127, FN 297 |
| Test M1 | Accuracy 0,9635; Precision 0,8942; Recall 0,8086; F1 0,8493; AP 0,9108; FP 91, FN 182 |

Giữ số đầy đủ trong file nguồn, hiển thị chỉ số bốn chữ số thập phân hoặc tỷ
lệ phần trăm hai chữ số; không trộn hai cách trong cùng một cột.

### B5. Bảng, hình và công thức dự kiến

**Bảng chính:** quy ước nhãn/ví dụ; schema và số lượng xử lý; thống kê v7/split;
ví dụ tiền xử lý; lưới tham số/ma trận B1–B6; đối chiếu từ dừng bằng CV;
B4/S1/S2/S3 bằng CV; cấu hình viBERT và hai loss validation; test B1–B6;
test các chiến lược SVM; test S3/M1; FP/FN và demo mới. Gộp bảng khi giúp giảm
lặp; lưới dài/CI đầy đủ có thể chuyển phụ lục.

**Hình chính:** sơ đồ hệ thống; phân bố nhãn; số mẫu/trọng số chiến lược;
loss/validation F1 viBERT; ma trận sáu cơ sở, bốn chiến lược và M1; PR S3–M1.
Các đường PR truyền thống còn lại chọn lọc hoặc để phụ lục, không thu nhỏ
nhiều biểu đồ đến mức không đọc được. Nguồn hình lấy từ artifacts đã kiểm tra;
sơ đồ nguyên lý tự vẽ lại và dẫn tài liệu gốc nếu dựa theo bài báo.

**Công thức cần chuẩn bị:** TF/IDF và chuẩn hóa; pooling FastText; Bayes/MNB/
Gaussian; decision function và mục tiêu LinearSVC phù hợp; attention, softmax,
cross-entropy; trọng số balanced; Accuracy/Precision/Recall/F1/AP và CV mean.
Mọi biến được giải thích ngay sau công thức; không cần khai triển mọi phép
cập nhật AdamW hoặc chứng minh tối ưu SVM.

## C. Quy cách trình bày đề xuất

Ảnh đề **không nêu mẫu trình bày hay thông số font/căn lề**. Các giá trị sau là
quy ước thống nhất đề xuất cho nhóm; nếu giảng viên/khoa có mẫu riêng thì ưu
tiên mẫu đó. Markdown này lưu kế hoạch; cỡ chữ/căn lề sẽ được áp khi dàn Word/PDF.

| Thành phần | Quy cách dự kiến |
|---|---|
| Khổ giấy/căn lề | A4 dọc; trái 3 cm, phải 2 cm, trên/dưới 2,5 cm |
| Nội dung | Times New Roman 13 pt, màu đen, căn đều hai lề, giãn dòng 1,5 |
| Đoạn văn | Thụt đầu dòng 1 cm; trước 0 pt/sau 6 pt; không chèn dòng trống thủ công để tạo khoảng cách |
| Chương và phần lớn | 16 pt, đậm, tên chương viết hoa; căn giữa; bắt đầu trang mới |
| Mục cấp 2 | 14 pt, đậm, căn trái; đánh số `1.1`, `2.1` tự động |
| Tiểu mục cấp 3 | 13 pt, đậm, căn trái; đánh số `1.1.1`; tối đa ba cấp số trong nội dung |
| Mở đầu/Kết luận/Tham khảo | Tiêu đề không gán thành Chương 0/3/4; vẫn xuất hiện trong mục lục |
| Bảng | Chữ 11–12 pt, tiêu đề trên bảng; dòng tiêu đề lặp khi qua trang, tránh tách một hàng dữ liệu |
| Hình | Chú thích dưới hình, chữ khoảng 12 pt; ưu tiên sơ đồ vector hoặc PNG đủ nét, không chụp cả màn hình IDE |
| Đánh số bảng/hình | Theo chương: Bảng 2.1, Hình 2.1; caption/đối chiếu tạo tự động |
| Công thức | Equation, căn giữa, số dạng (1.1); ký hiệu nhất quán và có giải thích |
| Code cần trích | Consolas 10–11 pt, ngắn, có tên hàm/file; đoạn dài chuyển phụ lục |
| Liệt kê | Một kiểu bullet thống nhất; chỉ đánh số khi có thứ tự; hạn chế liệt kê nhiều tầng |
| Số trang | Bìa không hiện số; phần đầu có thể dùng La Mã; từ Mở đầu đánh số Ả Rập liên tục |
| Mục lục | Heading/TOC tự động; không tự điền số trang khi nội dung chưa dàn xong |
| Trích dẫn | Dạng số `[1]`, `[2]` theo thứ tự xuất hiện; tài liệu có tác giả/tổ chức, năm nếu xác định, tên, nơi công bố và URL/DOI; nguồn web có ngày truy cập |

**Văn phong:** dùng “nhóm”, “nghiên cứu” hoặc “thực nghiệm” nhất quán; viết câu
trọn ý, thuật ngữ tiếng Việt kèm tiếng Anh ở lần đầu. Chương 1 trình bày nguyên
lý; Chương 2 nêu lựa chọn, cách làm và kết quả. Mỗi nhận xét đi sau bảng/hình
cần chỉ rõ nó dựa vào số nào. Giả thuyết về nguyên nhân lỗi phải ghi là khả
năng, không biến thành kết luận đã được chứng minh.

## D. Danh mục nguồn để viết và trích dẫn

Ngày tra cứu của kế hoạch: **06/11/2026**. Mã R phục vụ biên soạn, sẽ đánh số
lại khi viết. Đây là danh mục nguồn đã xác định, không phải tuyên bố đã đọc
toàn bộ mọi trang/bài. Trước khi đưa công thức hoặc diễn giải chi tiết vào
bản thảo, đọc đúng phần tương ứng; ưu tiên bài báo/sách và tài liệu chính thức.

### D1. Sách, bài báo và tài liệu nguyên lý

- **[R01]** C. D. Manning, P. Raghavan, H. Schütze (2008), *Introduction to
  Information Retrieval*, Cambridge University Press. [Trang sách của tác giả](https://www-nlp.stanford.edu/IR-book/),
  [bản PDF](https://nlp.stanford.edu/IR-book/pdf/irbookprint.pdf).
  Đọc các chương 2, 6, 13, 15 cho tiền xử lý, trọng số từ, NB và SVM.
  Metadata xác minh qua trang Stanford được lập chỉ mục; truy cập trực tiếp
  trang sách bị timeout trong lượt tra cứu, cần mở PDF/chương khi viết chi tiết.
- **[R02]** Unicode Consortium, *Unicode Standard Annex #15: Unicode
  Normalization Forms*. [Tài liệu chuẩn hóa Unicode](https://www.unicode.org/reports/tr15/).
  Dùng cho NFC; không suy ra phiên bản Unicode runtime từ phiên bản trang web.
- **[R03]** Underthesea, *word_tokenize*. [Tài liệu chính thức](https://undertheseanlp.github.io/underthesea/docs/next/api/word-tokenize).
  Dùng cho tách từ tiếng Việt; trang đang ở nhánh Next, chỉ đối chiếu khái niệm/
  ví dụ chung. Chữ ký và hành vi thực tế kiểm theo underthesea 8.3.0 của project.
- **[R04]** scikit-learn 1.6.1, *Feature extraction*, mục text feature extraction.
  [Tài liệu](https://scikit-learn.org/1.6/modules/feature_extraction.html).
  Dùng cho CountVectorizer/TF-IDF, n-gram và công thức triển khai.
- **[R05]** P. Bojanowski, E. Grave, A. Joulin, T. Mikolov (2017), *Enriching
  Word Vectors with Subword Information*, TACL, 5, 135–146.
  [Bài báo/PDF và BibTeX](https://aclanthology.org/Q17-1010/), DOI: 10.1162/tacl_a_00051.
- **[R06]** scikit-learn 1.6.1, *Naive Bayes*.
  [Tài liệu](https://scikit-learn.org/1.6/modules/naive_bayes.html).
  Dùng cho giả định độc lập, GaussianNB/MultinomialNB, additive smoothing.
- **[R07]** C. Cortes, V. Vapnik (1995), *Support-vector networks*, Machine
  Learning, 20, 273–297. [Trang bài báo](https://link.springer.com/article/10.1007/BF00994018),
  DOI: 10.1007/BF00994018. Trang công khai có metadata/abstract; bản toàn văn
  trên nhà xuất bản có thể yêu cầu quyền. Đối chiếu biến thể triển khai bằng R08/R09.
- **[R08]** R.-E. Fan, K.-W. Chang, C.-J. Hsieh, X.-R. Wang, C.-J. Lin (2008),
  *LIBLINEAR: A Library for Large Linear Classification*, JMLR, 9, 1871–1874.
  [Bài báo và PDF](https://jmlr.org/papers/v9/fan08a.html).
- **[R09]** scikit-learn 1.6.1, *LinearSVC*.
  [API chính thức](https://scikit-learn.org/1.6/modules/generated/sklearn.svm.LinearSVC.html).
  Dùng để đối chiếu squared hinge, C, class_weight và decision score.
- **[R10]** A. Vaswani và cộng sự (2017), *Attention Is All You Need*, NeurIPS.
  [Bài báo](https://arxiv.org/abs/1706.03762). Dùng cho Transformer/attention;
  viBERT sử dụng encoder, không mô tả nhóm xây dựng mô hình dịch encoder-decoder.
- **[R11]** J. Devlin, M.-W. Chang, K. Lee, K. Toutanova (2019), *BERT:
  Pre-training of Deep Bidirectional Transformers for Language Understanding*,
  NAACL-HLT, 4171–4186. [Bài báo/PDF](https://aclanthology.org/N19-1423/),
  DOI: 10.18653/v1/N19-1423.
- **[R12]** The Viet Bui, Thi Oanh Tran, Phuong Le-Hong (2020), *Improving
  Sequence Tagging for Vietnamese Text using Transformer-based Neural Models*,
  PACLIC 34, 13–20. [Bài báo/PDF](https://aclanthology.org/2020.paclic-1.2/),
  [arXiv](https://arxiv.org/abs/2006.15994). Đây là bài repo viBERT yêu cầu dẫn;
  nhiệm vụ gốc là sequence tagging, không phải phân loại spam PTIT.
- **[R13]** FPT.AI, *viBERT* và *FPTAI/vibert-base-cased*.
  [Repo chính thức](https://github.com/fpt-corp/viBERT),
  [checkpoint đã dùng](https://huggingface.co/FPTAI/vibert-base-cased).
  Trang checkpoint hiện không có model card mô tả đầy đủ; không tự thêm số
  liệu corpus/giấy phép/hiệu năng chưa có căn cứ. Cấu hình sử dụng lấy từ code/run.
- **[R14]** Hugging Face Transformers, *BERT / BertTokenizer*.
  [Tài liệu](https://huggingface.co/docs/transformers/model_doc/bert#berttokenizer).
  Dùng cho WordPiece, token đặc biệt, cấu trúc đầu vào.
- **[R15]** I. Loshchilov, F. Hutter (2019), *Decoupled Weight Decay
  Regularization*, ICLR 2019. [Bài báo](https://arxiv.org/abs/1711.05101).
  Bản arXiv đầu năm 2017, bản công bố hội nghị năm 2019; thống nhất năm theo
  dạng trích dẫn được chọn. Dùng cho AdamW, không đánh đồng weight decay với L2 trong Adam.
- **[R16]** PyTorch, *CrossEntropyLoss*.
  [Tài liệu chính thức](https://docs.pytorch.org/docs/2.15/generated/torch.nn.CrossEntropyLoss.html).
  Dùng cho logits, loss hai lớp và trọng số. Đây là phiên bản trang tra cứu;
  môi trường train của nhóm lấy từ `run_config.json`, không gọi là PyTorch 2.15.
- **[R17]** scikit-learn 1.6.1, *Cross-validation: evaluating estimator
  performance*. [Tài liệu](https://scikit-learn.org/1.6/modules/cross_validation.html).
  Dùng cho train/validation/test, CV và fit biến đổi trong Pipeline.
- **[R18]** scikit-learn 1.6.1, *compute_class_weight*.
  [Tài liệu](https://scikit-learn.org/1.6/modules/generated/sklearn.utils.class_weight.compute_class_weight.html).
  Dùng cho công thức balanced; tỷ lệ/chính sách khảo sát vẫn do nhóm xác định.
- **[R19]** imbalanced-learn, *Under-sampling*.
  [Tài liệu chính thức](https://imbalanced-learn.org/stable/under_sampling.html).
  Dùng cho nguyên lý random under-sampling; project tự thực hiện bằng NumPy,
  không viết rằng đã gọi RandomUnderSampler hoặc sử dụng SMOTE.
- **[R20]** scikit-learn 1.5.2, *Tuning the decision threshold for class
  prediction*. [Tài liệu](https://scikit-learn.org/1.5/modules/classification_threshold.html).
  Dùng cho nguyên lý score/ngưỡng. Project dùng scikit-learn 1.6.1 và hàm OOF
  tự viết, không phải triển khai nguyên xi TunedThresholdClassifierCV.
- **[R21]** scikit-learn 1.6.1, *Metrics and scoring: quantifying the quality
  of predictions*. [Tài liệu](https://scikit-learn.org/1.6/modules/model_evaluation.html).
  Dùng cho chỉ số nhị phân, ma trận, PR và AP.
- **[R22]** B. Efron (1979), *Bootstrap Methods: Another Look at the
  Jackknife*, The Annals of Statistics, 7(1), 1–26.
  [DOI](https://doi.org/10.1214/aos/1176344552),
  [bản bài báo trong tài liệu giảng dạy NTU](https://www.math.ntu.edu.tw/~hchen/teaching/LargeSample/references/Efron79.pdf).
  Bản PDF là scan; cần đọc phần phương pháp khi diễn giải, không lấy CI của
  project từ kết quả minh họa trong bài.

### D2. Công cụ triển khai và nguồn dữ liệu

- **[R23]** Hugging Face Transformers, *Trainer / TrainingArguments*, nhánh
  tài liệu 4.57.1. [Tài liệu](https://huggingface.co/docs/transformers/v4.57.1/en/main_classes/trainer).
  Dùng cho training loop, epoch/checkpoint, batch/accumulation, mixed precision.
  Phiên bản cài thực nghiệm của nhóm là 4.57.6; không lấy phiên bản website làm môi trường đã chạy.
- **[R24]** Argilla, *Annotate datasets*.
  [Tài liệu](https://docs.argilla.io/latest/how_to_guides/annotate/).
  Dùng cho suggestion/response và quy trình nhãn; không dùng tài liệu này để
  khẳng định ngưỡng confidence 0,75 đã được chứng minh tối ưu.
- **[R25]** mohdtalal3, *facebook_post_comment_scraper*.
  [Repo tham khảo](https://github.com/mohdtalal3/facebook_post_comment_scraper).
  Dẫn nguồn công cụ; phạm vi mã bàn giao và chỉnh sửa nhóm kiểm theo
  `external/facebook_scraper/`, runner PTIT và hồ sơ nguồn của project.
- **[R26]** p-storm, *ptit-facebook-comments-spamham*.
  [Địa chỉ dataset trong project](https://huggingface.co/datasets/p-storm/ptit-facebook-comments-spamham).
  Trang dataset/card không tải được trực tiếp trong lượt này; không khẳng định
  đã đọc card hiện hành. Số lượng/nhãn dùng viết báo cáo lấy từ v7 local và
  bằng chứng đã kiểm tra; không lấy số làm tròn trên danh mục web.

Không đưa tất cả nguồn vào danh mục cuối chỉ để tăng số lượng. Nguồn nào
thực sự dùng trong bản thảo mới giữ và phải có trích dẫn trong nội dung.
Không dùng blog tổng hợp thay bài gốc khi đã có nguồn chính thức.

## E. Kiểm tra trước khi chốt bản báo cáo

- Nội dung bắt buộc trong ảnh đề đều có vị trí; phần viBERT bổ sung được viết
  đầy đủ cả xây dựng/huấn luyện/đánh giá, không chỉ giới thiệu tên mô hình.
- Lý thuyết đúng biến thể code; công thức TF-IDF, loss SVM, prior/smoothing NB,
  weighted loss và pooling phải đối chiếu implementation.
- Mọi bảng có nguồn CSV cụ thể, ghi đúng CV/validation/test và lớp dương;
  B4 không bị đếm hai lần, S3 được chọn theo CV trước test.
- Thống kê thu thập/CSV minh họa/v7 không bị dùng thay thế nhau; không bổ sung
  ngày crawl, người duyệt hoặc lịch sử dữ liệu chưa xác nhận.
- Không viết có web app, benchmark latency/VRAM, bootstrap M1 hoặc khảo sát
  nhiều seed khi project chưa thực hiện. Định hướng tương lai được ghi riêng.
- Bìa cần thông tin thật: tên học phần, lớp, giảng viên, họ tên đầy đủ và MSSV
  của bốn thành viên. Hiện mới có tên gọi Bảo/Cương/Phi/Phương; không tự điền.
- Văn phong, thuật ngữ, caption, ký hiệu và số thập phân thống nhất; không dùng
  câu hỏi làm tiêu đề mục khoa học hoặc lời dẫn kiểu trao đổi chat.
- Trích dẫn khớp danh mục, DOI/link/tác giả/năm được rà lại; hình mượn có nguồn.
- Khi dàn trang xong, kiểm mục lục, số trang, bảng/hình/công thức trên từng
  trang PDF. Chỉ sau khi bản hoàn chỉnh được duyệt mới xóa file kế hoạch này.

**Phạm vi hiện tại kết thúc ở đề cương/kế hoạch.** Chưa viết các chương báo cáo,
chưa tạo bản Word/PDF, chưa cần chạy lại huấn luyện để lập mục lục.
