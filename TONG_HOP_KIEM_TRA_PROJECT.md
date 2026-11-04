# Tổng hợp và kiểm tra project phân loại Spam/Ham bình luận Facebook PTIT

**Ngày kiểm tra:** 03/11/2026.  
**Project:** `ptit-facebook-spam-ham`.  
**Phiên bản mã nguồn tại thời điểm kiểm tra:** Git HEAD `efc4fba`.  
**Căn cứ:** toàn bộ 13 file ở thư mục gốc trước khi tạo tài liệu này, gồm 8 file Python, notebook, README, phân công nhóm, `.gitignore` và ảnh đề bài.

> **Kết luận chính:** Project đã có phần thực nghiệm học máy khá đầy đủ cho bài toán Spam/Ham và có kết quả được lưu trong notebook. Tuy nhiên, chưa nên kết luận đã hoàn thành toàn bộ đề bài: chưa có thí nghiệm riêng chứng minh đáp ứng yêu cầu bổ sung về mô hình hiện đại, tài liệu kết luận chưa phản ánh đầy đủ kết quả, và còn lỗi trong luồng xử lý nhãn Phase 2. Bản thư mục hiện tại cũng chưa đủ dữ liệu và môi trường để xác nhận chạy lại toàn bộ quy trình.

## Mục lục

1. [Project làm gì và tại sao cần?](#1-project-làm-gì-và-tại-sao-cần)
2. [Đề bài thực sự yêu cầu những gì?](#2-đề-bài-thực-sự-yêu-cầu-những-gì)
3. [Cách các thành phần kết nối với nhau](#3-cách-các-thành-phần-kết-nối-với-nhau)
4. [Giải thích từng file](#4-giải-thích-từng-file)
5. [Đọc chi tiết notebook](#5-đọc-chi-tiết-notebook)
6. [Dữ liệu và kết quả hiện có](#6-dữ-liệu-và-kết-quả-hiện-có)
7. [Đối chiếu mức độ đáp ứng đề bài](#7-đối-chiếu-mức-độ-đáp-ứng-đề-bài)
8. [Các lỗi và vấn đề cần xử lý](#8-các-lỗi-và-vấn-đề-cần-xử-lý)
9. [Project đã xong chưa? Đánh giá tổng thể](#9-project-đã-xong-chưa-đánh-giá-tổng-thể)
10. [Thứ tự hoàn thiện và bàn giao](#10-thứ-tự-hoàn-thiện-và-bàn-giao)
11. [Những điều cần hiểu khi trình bày project](#11-những-điều-cần-hiểu-khi-trình-bày-project)
12. [Phạm vi kiểm chứng và nguồn tham chiếu](#12-phạm-vi-kiểm-chứng-và-nguồn-tham-chiếu)

## 1. Project làm gì và tại sao cần?

### 1.1. Bài toán

Project xây dựng bộ phân loại tự động cho **bình luận tiếng Việt trong các Facebook group của sinh viên PTIT**, với hai nhãn:

| Nhãn | Ý nghĩa trong project | Ví dụ minh họa |
|---|---|---|
| `ham` | Bình luận bình thường, hợp lệ, có ích cho thảo luận | Hỏi lịch thi, xin tài liệu, trao đổi môn học |
| `spam` | Bình luận rác, chào mời, quảng cáo hoặc không phù hợp theo quy ước gán nhãn | Mời mua dịch vụ, seeding, chèn liên hệ quảng cáo |

Đầu vào trực tiếp của bộ phân loại là văn bản bình luận. Đầu ra là nhãn `spam` hoặc `ham`, kèm điểm mô hình khi sử dụng hàm dự đoán trong notebook.

**Spam/Ham không phải phân loại cảm xúc.** Bình luận tích cực vẫn có thể là quảng cáo; bình luận tiêu cực về môn học vẫn có thể là thảo luận hợp lệ. Project cũng không thực hiện bài toán toxic/non-toxic.

### 1.2. Mục tiêu học phần

Mục tiêu là thực hành một quy trình khai phá dữ liệu văn bản: chuẩn bị dữ liệu → gán nhãn → tiền xử lý tiếng Việt → biểu diễn văn bản bằng số → huấn luyện bộ phân loại → đo và so sánh kết quả.

Project đặt ba cách biểu diễn văn bản — BoW, TF-IDF, FastText — cạnh hai họ thuật toán Naive Bayes và SVM để trả lời: **cách nào phân biệt spam tốt hơn trên bộ dữ liệu này, và phải đánh đổi gì giữa bắt được spam với báo nhầm ham?**

### 1.3. Tại sao cần và có thể dùng vào việc gì?

Bình luận rác làm loãng thảo luận, khiến sinh viên khó tìm thông tin và làm quản trị viên mất thời gian kiểm duyệt. Bộ phân loại có thể hỗ trợ lọc danh sách nghi vấn, ưu tiên bình luận cần xem hoặc thống kê tỷ lệ spam.

Hiện tại đây là **project thực nghiệm chạy bằng script và notebook**. Chưa có mã tích hợp quản trị Facebook, hệ thống theo dõi bình luận mới, giao diện sử dụng hay dịch vụ triển khai. Vì vậy, có thể nói project xây dựng và đánh giá bộ phân loại; chưa thể nói nó đang tự động kiểm duyệt Facebook thực tế.

### 1.4. Hai loại AI có vai trò khác nhau

| Thành phần | Vai trò thực tế |
|---|---|
| GLM được gọi qua OpenRouter trong `ai_assisted_spam_ham.py` | Hỗ trợ tạo/gợi ý nhãn dữ liệu trước huấn luyện |
| Naive Bayes và SVM trong notebook | Các bộ phân loại được huấn luyện và đánh giá |
| FastText | Học vector từ để cung cấp đặc trưng cho Naive Bayes/SVM |

Không nên gọi việc dùng GLM gán nhãn là đã có một thí nghiệm phân loại hiện đại: hiện chưa có bảng đánh giá GLM như một mô hình dự đoán độc lập trên tập kiểm tra.

## 2. Đề bài thực sự yêu cầu những gì?

Đã đọc trực tiếp [ảnh đề bài](<ảnh đề bài.png>). Nội dung liên quan đến nhóm này như sau.

### 2.1. Chọn nguồn dữ liệu và dạng phân loại

Ảnh nêu các lựa chọn nguồn dữ liệu: bình luận sinh viên PTIT trên Facebook group; review sản phẩm Điện Máy Xanh; hoặc bình luận phim Mưa đỏ trên fanpage/group. Các dạng phân loại gồm cảm xúc, spam/ham và toxic/non-toxic.

Project đang theo lựa chọn **bình luận sinh viên PTIT + spam/ham**, phù hợp với một hướng được đề bài cho phép. Không có căn cứ để yêu cầu project phải thực hiện đồng thời cả ba nguồn hoặc cả ba bài toán.

### 2.2. Các bước bắt buộc

| Bước | Yêu cầu trong ảnh |
|---|---|
| 1. Thu thập dữ liệu | Tìm kiếm, thu thập dataset bình luận/sản phẩm |
| 2. Tiền xử lý | Áp dụng kỹ thuật đã học như tách từ, loại stop words… |
| 3. Trích chọn đặc trưng | Thử nghiệm **ít nhất 2 phương pháp**, gợi ý BoW, TF-IDF, Word Embeddings |
| 4. Xây dựng và huấn luyện | Sử dụng **2 thuật toán phân loại**, gợi ý Naive Bayes và SVM; chia train/test |
| Bổ sung bước 4 | Dòng đỏ phía trên bên phải: **“Xây dựng một mô hình hiện đại giải quyết bài toán của bạn theo tuỳ chọn”** |
| 5. Đánh giá | Accuracy, Precision, Recall, F1-Score, Confusion Matrix; so sánh đặc trưng và thuật toán |
| Viết báo cáo | Tiêu đề; Giới thiệu bài toán; Cơ sở lý thuyết; Thực nghiệm và kết quả; Tài liệu tham khảo |

### 2.3. Các lưu ý khác trong ảnh

Các nhóm trùng đề tài có thể khai thác chung dữ liệu nhưng phải đảm bảo chất lượng, số lượng và khác nhau về cách làm/cách giải quyết. Chỉ đọc project này chưa đủ để xác nhận mức độ khác biệt với các nhóm khác.

Phần hướng dẫn về tạo nhãn cảm xúc ở cột phải không phải yêu cầu phải thêm nhãn cảm xúc vào bài Spam/Ham.

Ảnh **không nêu ngưỡng Accuracy/F1 tối thiểu**, cũng không yêu cầu bắt buộc làm website hoặc triển khai dịch vụ. Vì thế không thể dựa vào một mốc điểm tự đặt hay việc chưa có giao diện để kết luận project không đạt.

## 3. Cách các thành phần kết nối với nhau

Project có hai luồng: chuẩn bị nhãn và thực nghiệm học máy.

```text
Dữ liệu crawl Phase 2 dạng JSON
  → convert_phase2_json.py: trải bình luận và trả lời thành CSV
  → clean_data_facebook_spamham.py: làm sạch, tạo cleaned_text
  → upload_to_argilla.py: đưa lên công cụ gán nhãn
  → ai_assisted_spam_ham.py: GLM gợi ý spam/ham
  → auto_accept_labels.py: duyệt tự động phần đủ ngưỡng
  → Người gán nhãn kiểm tra/sửa trên Argilla
  → download_labeled_data.py: xuất nhãn chính thức về CSV
  → merge_labeled_data.py: gộp Phase 1 và Phase 2

Dataset đã rà soát labeled_v7.csv (Hugging Face hoặc SPAMHAM_CSV)
  → spam-ham-comments.ipynb
      → khám phá dữ liệu và chia train/test theo nhóm văn bản
      → tiền xử lý, tách từ, thử giữ/bỏ stop words
      → BoW / TF-IDF / FastText
      → Naive Bayes / SVM, tìm siêu tham số bằng CV
      → bảng điểm, biểu đồ, phân tích lỗi
      → lưu spam_ham_ml_only.joblib và minh họa dự đoán
```

**Chỗ chưa nối liền:** script gộp xuất `comments_labeled_merged.csv`, còn notebook mặc định đọc `labeled_v7.csv`. Repo chưa có script/lịch sử biến đổi đầy đủ giải thích từ bản gộp sang nhãn v7. Do đó sơ đồ mô tả các thành phần đã có, không chứng minh một lệnh có thể tái tạo toàn bộ dữ liệu cuối cùng từ đầu.

Nếu đã có CSV nhãn v7 hợp lệ thì có thể đi thẳng vào notebook, không cần gọi lại dịch vụ gán nhãn. Argilla và GLM phục vụ khâu chuẩn bị dữ liệu, không phải phụ thuộc bắt buộc của mô hình SVM khi dự đoán.

## 4. Giải thích từng file

### 4.1. `README.md` — tài liệu giới thiệu và hướng dẫn

**Dùng để:** giới thiệu bài toán, dữ liệu hai phase, quy trình, bảng kết quả, cải tiến và cách chạy local/Kaggle.

**Tại sao cần:** giúp người nhận repo hiểu phải chạy gì, dùng dữ liệu nào và kết quả kỳ vọng là gì.

**Nhận xét:** nội dung khá nhiều nhưng một số chỗ chưa khớp hiện trạng. README nói đã có `.venv`, trong thư mục kiểm tra không có môi trường này; ví dụ lệnh chạy dùng đường dẫn kiểu Unix; phần kết quả chưa trình bày thí nghiệm mất cân bằng có F1 cao hơn mô hình được giới thiệu. Chi tiết tại mục 8.

### 4.2. `TEAM_ASSIGNMENTS.md` — phân công nhóm

**Dùng để:** phân trách nhiệm Bảo, Cương, Phi, Phương và mô tả đầu việc theo các bước của project.

| Thành viên | Công việc được ghi trong file |
|---|---|
| Bảo | Thu thập, chuyển JSON → CSV; huấn luyện các thí nghiệm |
| Cương | Tiền xử lý; BoW và TF-IDF |
| Phi | FastText, phân tích embedding/OOV, chia train/test |
| Phương | Đánh giá, so sánh và xử lý mất cân bằng |

**Tại sao cần:** xác định phần bàn giao và phạm vi giải thích khi làm báo cáo/thuyết trình.

**Nhận xét:** đây là kế hoạch, không phải bằng chứng từng thành viên đã hoàn tất đầu việc. File cũng chưa giao riêng phần mô hình hiện đại trong yêu cầu bổ sung. Hiện file bị `.gitignore` bỏ qua nên có trên máy này nhưng không nằm trong danh sách file Git đang theo dõi.

### 4.3. `ảnh đề bài.png` — căn cứ nghiệm thu

**Dùng để:** xác định đúng yêu cầu giảng viên, đặc biệt phần bổ sung mô hình hiện đại và cấu trúc báo cáo.

**Tại sao cần:** tránh chỉ đối chiếu với README hoặc phân công nội bộ rồi bỏ sót yêu cầu gốc.

**Hiện trạng:** có ở máy local, đang là file chưa được Git theo dõi. Người chỉ nhận các file đã commit có thể không nhận được ảnh này.

### 4.4. `.gitignore` — quy định file không đưa vào Git

**Dùng để:** bỏ qua dữ liệu lớn, môi trường ảo, cache Python, `.env`, mô hình `.joblib`, thư mục Argilla local và một số tài liệu nội bộ.

**Tại sao cần:** giữ repo gọn, tránh đưa cấu hình riêng và dữ liệu lớn lên kho mã nguồn.

**Điểm cần biết:** `data/`, `.venv/`, `namedb_cache/`, `*.joblib`, `TEAM_ASSIGNMENTS.md`, một số tên báo cáo và toàn bộ `*.pdf`/`*.html` đều bị bỏ qua. Vì vậy việc một file không có trong Git chưa chứng minh nó chưa từng được tạo; ngược lại, README nói có file cũng chưa chứng minh file có trong checkout đang xét.

### 4.5. `convert_phase2_json.py` — làm phẳng dữ liệu Phase 2

**Đầu vào mặc định:** `data/groups_data_phase2.json`; tham chiếu dữ liệu Phase 1 ở `data/comments_labeled_export.csv`.

**Đầu ra mặc định:** `data/comments_phase2.csv` với các cột `post_id`, `group_id`, `group_name`, `created_at`, `comment_type`, `text`, `reaction_count`.

**Cách hoạt động:**

1. `flatten_posts()` duyệt bài đăng → bình luận → các trả lời trực tiếp, tạo một dòng cho mỗi bình luận/trả lời.
2. `to_int()` chuyển số reaction thành số nguyên, dùng 0 nếu không chuyển được.
3. Bỏ văn bản rỗng.
4. `load_phase1_texts()` lấy tập văn bản Phase 1; dùng lại hàm làm sạch của file cleaner để tìm các mẫu đã có.
5. Xuất CSV UTF-8 có BOM.

**Tại sao cần:** dữ liệu JSON lồng nhau khó đưa trực tiếp vào pandas hoặc công cụ gán nhãn; trải phẳng giúp các bước sau xử lý đồng nhất. Lọc mẫu đã có còn giảm công gán nhãn lại.

**Giới hạn:** file này không crawl Facebook. Nó giả định JSON là danh sách bài đăng có schema phù hợp; chỉ duyệt một tầng replies. `created_at` lấy từ bài đăng, không lấy thời gian riêng của từng bình luận. Không có `comment_id` Facebook gốc hoặc quan hệ cha–con đầy đủ trong đầu ra.

### 4.6. `clean_data_facebook_spamham.py` — làm sạch trước gán nhãn

**Đầu vào mặc định:** `data/comments.csv`, `data/comments (1).csv`. Có thể truyền file khác bằng `--input`.

**Đầu ra mặc định:** `data/comments_cleaned.csv`; giữ các trường gốc và thêm `cleaned_text`.

**Cách hoạt động:**

1. Đọc CSV và bỏ dòng trùng theo bộ `(post_id, comment_type, text)`.
2. Chuẩn hóa Unicode NFC, giải mã HTML entities, xóa thẻ HTML.
3. Thay URL bằng `urltoken`, số điện thoại khớp mẫu bằng `phonetoken`.
4. Xóa emoji/ký tự không cần thiết, rút gọn dấu câu lặp và khoảng trắng.
5. Bỏ dòng sau làm sạch không còn chữ hoặc số theo điều kiện regex của script.
6. Lưu CSV và in thống kê.

**Tại sao cần:** giảm nhiễu nhưng vẫn giữ tín hiệu liên hệ quảng cáo. Thay URL/SĐT bằng token giúp mô hình học rằng bình luận chứa đường dẫn/liên hệ, thay vì học một URL/số cụ thể.

**Lưu ý:** đây chưa phải toàn bộ tiền xử lý phục vụ mô hình. Tách từ, chuyển chữ thường, che tên và đối chứng stop words nằm trong notebook. `preprocess()` ở cleaner và `preprocess()` trong notebook là hai hàm khác nhau.

**Đã kiểm tra nhỏ:** thay URL/SĐT hoạt động trên mẫu minh họa; luồng cleaner giữ một dòng `Hello` khi đầu vào có hai dòng giống nhau và một dòng chỉ có `!!!`.

### 4.7. `upload_to_argilla.py` — đưa dữ liệu lên giao diện gán nhãn

**Đầu vào:** CSV sạch; địa chỉ/API key Argilla; tên dataset.

**Đầu ra:** dataset và các records trên Argilla, không phải mô hình phân loại.

**Cách hoạt động:** tạo hoặc mở dataset trong workspace `argilla`; cấu hình trường `cleaned_text`, câu hỏi nhãn `spam/ham` và hướng dẫn gán nhãn; đọc CSV rồi upload theo batch.

**ID:** mặc định `cmt_{chỉ_số_dòng}`; `--id-prefix p2_cmt_` cho phép tách ID Phase 2. Metadata có loại bình luận và reaction.

**Tại sao cần:** tạo nơi con người kiểm tra, sửa và xác nhận nhãn có hệ thống.

**Giới hạn:** ID dựa vào thứ tự dòng CSV nên cần giữ đúng bản CSV để đối chiếu. Cờ `--reset` xóa dataset cũ trước khi tạo lại. Repo hiện không kèm cấu hình Docker Compose để dựng Argilla như tài liệu gợi ý.

### 4.8. `ai_assisted_spam_ham.py` — hỗ trợ gán nhãn bằng GLM

**Đầu vào:** bình luận từ dataset Argilla; nếu dataset trống thì lấy CSV. Cấu hình API có thể đến từ biến môi trường hoặc `.env`.

**Đầu ra:** `data/predictions_glm_cache.json`; tùy chọn cập nhật suggestions và submitted responses trên Argilla.

**Các phần chính:**

| Hàm/phần | Tác dụng |
|---|---|
| `load_env_file()`, `parse_args()` | Đọc cấu hình và tham số chạy |
| `load_cache()`, `save_cache()` | Lưu/đọc dự đoán để tiếp tục sau khi dừng |
| `clean_json_response()` | Gỡ phần bao Markdown hoặc văn bản thừa để đọc JSON |
| `predict_batch_with_glm()` | Gửi nhiều bình luận trong một yêu cầu HTTP, đọc nhãn/confidence, thử lại khi lỗi |
| `parse_dataset_identifier()` | Nhận tên, UUID hoặc URL dataset |
| `setup_argilla_dataset()` | Tìm/tạo dataset; có tùy chọn reset |
| `main()` | Điều phối lấy dữ liệu, dự đoán đa luồng, lưu cache và upload |

**Tại sao cần:** giảm công gán nhãn ban đầu khi dữ liệu lớn. Tuy nhiên, nhãn do AI sinh ra vẫn cần kiểm tra chất lượng.

**Điểm cần hiểu:** tên biến/cờ có chữ `OPENAI` ở đây phục vụ endpoint tương thích; cấu hình mặc định trong mã là OpenRouter. Đây là gọi mô hình có sẵn qua API, không phải huấn luyện GLM trong repo. Không kiểm tra khả dụng hay chi phí của model ID bên ngoài trong lần rà soát này.

**Vấn đề chính:** `--no-upload` vẫn kết nối Argilla; một nhánh phục hồi suggestion dùng confidence mặc định 1.0; cấu hình Argilla từ môi trường được đọc nhưng chưa dùng làm mặc định CLI. Chi tiết ở mục 8.

### 4.9. `auto_accept_labels.py` — chuyển gợi ý thành nhãn chính thức

**Đầu vào:** cache JSON chứa `label` và `confidence`; CSV sạch để nối lại văn bản.

**Đầu ra:** submitted responses trên Argilla nếu bật cập nhật; CSV mặc định `data/comments_auto_labeled.csv`.

**Cách hoạt động:** chia cache thành nhóm confidence đạt ngưỡng và nhóm cần xem thủ công. Mặc định ngưỡng là 0.75. CSV xuất có `official_label`, `ai_confidence`, `is_auto_accepted`; nhãn chưa được tự duyệt để trống.

**Tại sao cần:** tăng tốc xác nhận những nhãn được đánh giá là ít nghi vấn, tập trung công sức vào phần còn lại.

**Phân biệt:** “official” ở đây là trạng thái do chương trình chốt; không đồng nghĩa đã được con người xác minh. Confidence do LLM tự trả về cũng không tự động có nghĩa là xác suất đúng đã được hiệu chuẩn.

**Lỗi đã tái hiện:** xuất CSV vẫn tìm ID `cmt_0`, `cmt_1`… nên cache mang tiền tố `p2_cmt_` không được nối đúng. Cache rỗng cũng gây chia cho 0 khi in thống kê.

### 4.10. `download_labeled_data.py` — tải nhãn chính thức từ Argilla

**Đầu vào:** dataset trên Argilla; tùy chọn CSV gốc qua `--csv-lookup`.

**Đầu ra mặc định:** `data/comments_labeled_export.csv`, có `id`, `label`, `cleaned_text` và các trường gốc nối thêm nếu lookup được.

**Cách hoạt động:** `build_records_iter()` lấy records đã submitted; `extract_submitted_response()` lấy phần tử submitted cuối trong danh sách; lọc nhãn hợp lệ; nối `text`, `comment_type`, `reaction_count` bằng chỉ số tách từ ID.

**Tại sao cần:** đưa dữ liệu từ công cụ gán nhãn về dạng CSV để huấn luyện và lưu trữ.

**Giới hạn:** regex nhận `cmt_`/`p2_cmt_`, nhưng chưa bảo đảm đang lookup đúng CSV của từng phase. ID không chuyển thành số bị loại bằng `dropna`, có thể khiến mất dòng mà chưa có thống kê riêng. Nếu lọc ra 0 nhãn hợp lệ, `DataFrame` không có cột `label` và đoạn thống kê có thể lỗi. “Phần tử cuối” cũng chưa chắc là response mới nhất theo thời gian nếu SDK không đảm bảo thứ tự.

Theo [tài liệu Argilla về Record](https://docs.argilla.io/latest/reference/argilla/records/records/), thuộc tính `Record.id` trong SDK được mô tả trỏ tới external ID, còn `_server_id` là UUID máy chủ. Vì chưa khóa phiên bản SDK, cần kiểm tra bằng phiên bản thực dùng; không nên chỉ thấy code đọc `rec.id` rồi khẳng định nó chắc chắn trả UUID và chắc chắn làm mất dữ liệu.

### 4.11. `merge_labeled_data.py` — gộp dữ liệu hai giai đoạn

**Đầu vào mặc định:** `comments_labeled_export.csv` và `comments_labeled_phase2.csv` trong `data/`.

**Đầu ra mặc định:** `data/comments_labeled_merged.csv`.

**Cách hoạt động:** giữ sáu cột dữ liệu chính; chuẩn hóa/lọc nhãn Phase 2; chấp nhận cả CSV có cột `official_label`; thêm tiền tố `p2_` nếu chưa có; bỏ các dòng Phase 2 có `cleaned_text` trùng Phase 1; thêm cột `source` rồi ghép.

**Tại sao cần:** tích hợp dữ liệu bổ sung mà vẫn phân biệt nguồn và tránh ID hai phase trùng nhau.

**Giới hạn:** chủ ý giữ Phase 1; không kiểm tra/làm sạch nhãn Phase 1 tương đương Phase 2; không khử trùng toàn bộ bên trong từng phase; so trùng chỉ strip khoảng trắng hai đầu, chưa chuẩn hóa chữ hoa/thường hoặc Unicode. File không sinh ra nhãn v7.

**Đã kiểm tra nhỏ:** nhãn Phase 2 không hợp lệ bị lọc; ID đã có `p2_` không bị thêm lần nữa; hai văn bản giống nhau bên trong Phase 2 vẫn được giữ. Docstring còn nhắc notebook `spam-ham-comments-v2.ipynb` đã không còn trong checkout.

### 4.12. `spam_features.py` — biến văn bản thành vector FastText

**Đầu vào:** dãy câu đã tách từ, các token ngăn nhau bằng khoảng trắng.

**Đầu ra:** mỗi câu thành vector 100 chiều theo cấu hình đang dùng; cả tập thành ma trận `(số_câu, 100)`.

**Thành phần:** lớp `FastTextVectorizer` kế thừa giao diện transformer của scikit-learn.

| Phương thức | Việc thực hiện |
|---|---|
| `__init__()` | Lưu cấu hình vector, cửa sổ ngữ cảnh, epochs, subword, pooling, seed, workers |
| `fit()` | Học FastText Skip-gram trên dữ liệu nhận được; giữ `model.wv`; tính IDF từ chính tập đó |
| `transform()` | Lấy vector token, rồi trung bình theo tần suất hoặc trọng số TF-IDF; câu rỗng nhận vector 0 |

**Tại sao cần:** BoW/TF-IDF dựa vào từ vựng rời rạc; FastText bổ sung biểu diễn đặc, tận dụng n-gram ký tự để tạo vector cho từ chưa gặp. Đặt transformer trong Pipeline giúp embedding và IDF chỉ học trên train của từng fold.

**Điểm cần lưu ý:** đây là bộ trích đặc trưng, không tự quyết định nhãn. Cơ chế lấy trung bình không giữ đầy đủ thứ tự từ và ngữ cảnh câu. `workers=4` khiến seed đơn lẻ chưa đảm bảo tái lập chính xác tuyệt đối; [Gensim nêu thêm yêu cầu về workers và PYTHONHASHSEED](https://radimrehurek.com/gensim/models/fasttext.html).

### 4.13. `spam-ham-comments.ipynb` — trung tâm thực nghiệm

**Dùng để:** đọc dataset nhãn v7, khám phá dữ liệu, chia tập, tiền xử lý, huấn luyện, đánh giá, phân tích và lưu mô hình.

**Tại sao cần:** đây là nơi trả lời phần lớn yêu cầu học phần bằng thí nghiệm và kết quả trực quan. Tám script còn lại chủ yếu phục vụ chuẩn bị dữ liệu hoặc cung cấp transformer.

Notebook có **38 cell: 20 code và 18 Markdown**, trong đó cell Markdown cuối rỗng. Nội dung chi tiết được giải thích ngay dưới đây.

### 4.14. Thư mục `.git/`

Đây là dữ liệu nội bộ của Git để lưu lịch sử, chỉ mục và trạng thái phiên bản; không phải một phân hệ Spam/Ham. Đã kiểm tra danh sách file theo dõi, trạng thái và một số commit gần nhất; không cần đọc từng object nhị phân của Git để đánh giá thuật toán.

## 5. Đọc chi tiết notebook

**Quy ước:** số cell dưới đây đếm từ 1 theo thứ tự trong file, không phải số `In [n]` của Jupyter.

| Cell | Nội dung và ý nghĩa |
|---|---|
| 1 | Tiêu đề và phạm vi “học máy thuần” |
| 2 | Cài thư viện; không khóa phiên bản |
| 3 | Dùng `%%writefile` ghi lại `spam_features.py`; nội dung hiện trùng file Python đi kèm |
| 4 | Import, đặt seed 42, cấu hình hiển thị |
| 5–6 | Giới thiệu nguồn và đọc `labeled_v7.csv` từ `SPAMHAM_CSV`, cache HF hoặc tải HF; lọc nhãn hợp lệ và `cleaned_text` thiếu |
| 7–8 | Phân bố nhãn, độ dài, loại comment/reply, tỷ lệ văn bản lặp và biểu đồ khám phá |
| 9–10 | Chuẩn hóa khóa văn bản, chia test bằng hash; tạo năm fold CV theo hash khác |
| 11–12 | Giải thích che tên; chuẩn bị từ điển tên và tải dữ liệu tên từ GitHub; định nghĩa hàm che tên/nhận emoji |
| 13 | Stop words, chuẩn hóa, tách từ underthesea, thêm dấu hiệu emoji |
| 14 | Tạo hai bộ văn bản: giữ và loại stop words |
| 15–16 | Giải thích và chạy 12 cấu hình thí nghiệm; GridSearchCV tìm siêu tham số |
| 17–18 | Chọn kiểu tiền xử lý theo trung bình F1-CV của sáu tổ hợp |
| 19–20 | Bảng Accuracy, Precision, Recall, F1, AP và bảng so sánh đặc trưng × thuật toán |
| 21–22 | Sáu confusion matrix, bố trí thành lưới 3×2 |
| 23–24 | Bootstrap 500 lần để ước lượng khoảng tin cậy F1; vẽ đường Precision–Recall |
| 25–26 | Đối chứng ảnh hưởng của việc loại stop words |
| 27–28 | So sánh SVM không cân bằng, SVM balanced, NB chỉnh ngưỡng và SVM under-sampling |
| 29–30 | Chọn mô hình tốt nhất trong sáu tổ hợp chính theo F1-CV; xem hệ số từ vựng và các lỗi FP/FN |
| 31–32 | Kiểm tra từ gần nhau trong FastText và các biến thể OOV |
| 33–34 | Đối chiếu những mẫu TF-IDF/FastText dự đoán đúng khác nhau |
| 35–36 | Lưu mô hình, thông tin stop words và định nghĩa `predict_spam()` |
| 37 | Nhận xét tổng kết; có vài kết luận cần sửa theo kết quả thực tế |
| 38 | Markdown rỗng, không tác động xử lý |

### 5.1. Chia dữ liệu và chống rò rỉ

`norm_key()` chuyển chữ thường, gom dãy số thành `0`, bỏ dấu câu và gộp khoảng trắng. MD5 của khóa với salt `test` quyết định khoảng 20% nhóm làm test; salt `cv` chia phần train thành năm fold. Có assert kiểm tra nhóm train/test không giao nhau.

Đây là điểm tốt hơn chia ngẫu nhiên từng dòng khi dữ liệu có bình luận lặp. Các bộ vector hóa nằm trong Pipeline nên được fit trên train của từng fold.

Tuy nhiên, tỷ lệ là **xấp xỉ theo số dòng**, không ép đúng 80/20; nhãn không được stratify. Hash chỉ bảo vệ các nhóm trùng theo khóa định nghĩa, không loại mọi dạng gần trùng. Khóa còn được tạo trước bước che tên, nên một số câu khác tên có thể thành cùng biểu diễn sau tiền xử lý nhưng vẫn nằm khác tập.

### 5.2. Vì sao có 12 thí nghiệm?

`3 đặc trưng × 2 họ thuật toán × 2 kiểu stop words = 12 tổ hợp`. Đây không phải chỉ có 12 lần fit: GridSearchCV thử nhiều bộ siêu tham số ở mỗi tổ hợp, mỗi bộ qua năm fold rồi refit.

| Đặc trưng | Nhánh Naive Bayes | Nhánh SVM |
|---|---|---|
| BoW | MultinomialNB, tìm `alpha` | LinearSVC balanced, tìm `C` |
| TF-IDF | MultinomialNB, tìm `alpha` | LinearSVC balanced, tìm `C` |
| FastText | GaussianNB, tìm `var_smoothing` | StandardScaler + LinearSVC balanced, tìm `C` |

BoW/TF-IDF thử unigram và unigram + bigram; FastText thử pooling mean/TF-IDF. GaussianNB được dùng cho embedding vì các chiều vector có thể âm, không phù hợp với đầu vào không âm của MultinomialNB.

### 5.3. “Học máy thuần, không quy tắc” nghĩa là gì?

Quyết định nhãn cuối do bộ phân loại đã học thực hiện. Project vẫn dùng regex để xử lý URL, điện thoại, tên và emoji. Cách mô tả chính xác là **không dùng bộ luật quyết định nhãn ở đầu ra**; không phải hoàn toàn không có quy tắc tiền xử lý.

### 5.4. Điểm mô hình và mô hình lưu

SVM sử dụng `decision_function()`: số dương/âm biểu diễn phía và độ xa tương đối với ranh giới phân loại, **không phải xác suất**. `predict_spam()` trả trường `score` chung cho cả xác suất NB và điểm SVM, nên khi giải thích phải nêu đúng mô hình.

Bundle `.joblib` chứa pipeline đã fit, stop words, cờ giữ/bỏ stop words và tên biến thể. Nó chưa đóng gói đầy đủ các hàm che tên, tách từ, từ điển tên và bước chuẩn hóa dữ liệu thô để một ứng dụng độc lập có thể nạp rồi dùng ngay.

## 6. Dữ liệu và kết quả hiện có

### 6.1. Nguồn và các phiên bản dữ liệu

| Phiên bản | Số mẫu | Ham | Spam | Căn cứ trong checkout |
|---|---:|---:|---:|---|
| Phase 1 | 31.931 | 28.722 | 3.209 | README; không có CSV local để đếm lại |
| Phase 2 | 6.302 | 5.197 | 1.105 | README; không có JSON/CSV local để đếm lại |
| Bản gộp được README mô tả | 38.233 | 33.919 | 4.314 | README |
| Nhãn v7 dùng trong notebook | 38.233 | 32.907 | 5.326 | Output đã lưu ở cell 6 |

Hai bản cùng 38.233 dòng có phân bố nhãn khác nhau. README giải thích v7 đã rà soát nhãn, nhưng repo chưa có changelog/biên bản đủ để đối chiếu những dòng đổi nhãn. Không nên lấy số lượng spam của bản gộp cũ để trình bày thí nghiệm v7.

Tập train được output ghi là **30.753 mẫu**, trong đó **4.370 spam**. Test có **7.480 mẫu: 6.524 ham, 956 spam**. Tỷ lệ spam test khoảng 12,8%, khác tỷ lệ toàn bộ dataset vì chia theo nhóm hash.

### 6.2. Sáu tổ hợp chính — giữ stop words

**Các số dưới đây lấy từ output lưu trong notebook, chưa phải kết quả huấn luyện lại trong lần kiểm tra này.**

| Đặc trưng | Thuật toán | Accuracy | Precision spam | Recall spam | F1 spam | AP, được notebook gọi PR-AUC |
|---|---|---:|---:|---:|---:|---:|
| BoW | MultinomialNB | 91,47% | 72,46% | 53,66% | 61,66% | 0,7620 |
| BoW | SVM balanced | 90,55% | 59,48% | 81,69% | 68,84% | 0,7823 |
| TF-IDF | MultinomialNB | 93,76% | 86,99% | 60,15% | 71,12% | 0,8413 |
| TF-IDF | SVM balanced | 94,40% | 74,84% | 84,62% | 79,43% | 0,8629 |
| FastText | GaussianNB | 86,76% | 48,72% | 67,68% | 56,65% | 0,4623 |
| FastText | SVM balanced | 89,73% | 56,44% | 86,19% | 68,21% | 0,8118 |

Trong **sáu tổ hợp chính**, TF-IDF + SVM balanced được chọn theo CV, có F1-CV 0,7878; F1 test 0,7943 và CI bootstrap được lưu `[0,7738; 0,8145]`.

FastText + SVM có Recall cao nhất trong bảng, nhưng Precision thấp hơn nhiều: nó bắt nhiều spam hơn và cũng báo nhầm nhiều ham hơn. Không thể chỉ lấy Recall để nói FastText tốt nhất toàn diện.

### 6.3. Đối chứng stop words

| Đặc trưng + thuật toán | F1 giữ stop words | F1 loại stop words |
|---|---:|---:|
| BoW + NB | 0,6166 | 0,5975 |
| BoW + SVM | 0,6884 | 0,6835 |
| TF-IDF + NB | 0,7112 | 0,7007 |
| TF-IDF + SVM | 0,7943 | 0,7767 |
| FastText + NB | 0,5665 | 0,5485 |
| FastText + SVM | 0,6821 | 0,6719 |

Trung bình F1-CV của sáu tổ hợp: giữ stop words **0,6757**, loại stop words **0,6523**. Mã chọn giữ theo CV; kết quả test lưu cũng giảm khi loại ở cả sáu tổ hợp.

Đề yêu cầu áp dụng/thử kỹ thuật tiền xử lý, không có nghĩa phải giữ một biến thể làm kết quả xấu hơn. Việc thử loại stop words rồi dùng đối chứng để chọn giữ là có cơ sở.

### 6.4. Kết quả xử lý mất cân bằng — cần đọc cùng bảng chính

Output ở cell 28 ghi:

| Phương pháp trên TF-IDF | Accuracy | Precision spam | Recall spam | F1 spam | AP |
|---|---:|---:|---:|---:|---:|
| SVM không dùng class weight | **95,59%** | **85,89%** | 78,35% | **81,95%** | **0,8690** |
| SVM `class_weight='balanced'` | 94,40% | 74,84% | **84,62%** | 79,43% | 0,8629 |
| NB chỉnh ngưỡng khoảng 0,385 | 95,03% | 81,33% | 79,29% | 80,30% | 0,8413 |
| SVM under-sampling Ham:Spam = 2:1 | 94,55% | 76,40% | 82,95% | 79,54% | 0,8551 |

**Nhận định đúng:** balanced tăng Recall nhưng giảm Precision và giảm F1 so với baseline SVM trong lần chạy được lưu. Mất cân bằng không tự động đồng nghĩa dùng class weight sẽ tốt hơn ở mọi chỉ số.

Câu in trong notebook “Recall chỉ 78,3% — bỏ sót phần lớn spam” sai về lượng: Recall 78,35% nghĩa là bắt khoảng 78% và bỏ sót khoảng 22%, không phải bỏ sót phần lớn.

**Phạm vi lựa chọn mô hình:** SVM balanced là tốt nhất theo CV trong sáu tổ hợp chính; SVM không cân bằng có F1 test cao nhất trong các bảng kết quả được lưu ở đây. Hai phát biểu này khác nhau. Chưa nên đổi mô hình cuối chỉ bằng cách nhìn bảng test; cần đưa các phương án bổ sung vào quy trình chọn bằng CV thống nhất.

### 6.5. Phân tích lỗi và embedding

Notebook có cả FP, FN và đối chiếu hai mô hình:

- FastText + SVM bắt được 50 mẫu spam mà TF-IDF + SVM bỏ sót; chiều ngược lại có 35 mẫu.
- Có trường hợp hợp lệ chứa link, tên hoặc lời nhắc liên hệ bị báo spam; ví dụ minh họa cuối notebook báo câu tuyển thành viên CLB là spam.
- OOV như `inboxxx` có vector gần `inbox`, nhưng `z.a.l.o` lại gần các từ không rõ liên quan trong output. Khả năng tạo vector cho từ lạ không bảo đảm hiểu đúng mọi biến thể.
- Những từ có vector gần nhau có thể gần về hình thức hoặc cùng xuất hiện trong ngữ cảnh, chưa chắc là từ đồng nghĩa.

Vì vậy, phần giải thích về “ngữ nghĩa chuẩn xác” cần tiết chế và gắn với ví dụ thực tế. Với câu rất ngắn hoặc nhãn phụ thuộc bài đăng cha, chỉ dùng nội dung bình luận đơn lẻ có giới hạn tự nhiên.

## 7. Đối chiếu mức độ đáp ứng đề bài

| Yêu cầu | Bằng chứng | Đánh giá |
|---|---|---|
| Đúng nguồn và bài toán được phép | README, notebook và scripts mô tả bình luận Facebook PTIT, Spam/Ham | **Phù hợp về lựa chọn**; chưa kiểm chứng lại nguồn dữ liệu từng dòng |
| Thu thập dữ liệu | Có luồng xử lý dữ liệu hai phase và dataset được tham chiếu | **Có mô tả và dữ liệu đã dùng trong output**, nhưng thiếu raw data/crawler/log để tái kiểm chứng thu thập |
| Tiền xử lý | Unicode, URL/SĐT, làm sạch, tách từ, stop words, che tên, emoji | **Có thực hiện**, còn vấn đề che tên/emoji cần sửa |
| Ít nhất 2 cách trích đặc trưng | BoW, TF-IDF, FastText | **Đạt về mã và output**, có 3 cách |
| 2 thuật toán phân loại | Naive Bayes và SVM | **Đạt về mã và output** |
| Chia train/test | Hash nhóm, assert không giao nhau | **Có**, thiết kế tốt ở luồng chính |
| Mô hình hiện đại bổ sung | FastText dùng để trích đặc trưng; GLM dùng gán nhãn; chưa có thực nghiệm classifier riêng theo bổ sung | **Chưa có bằng chứng đáp ứng rõ ràng** |
| Accuracy, Precision, Recall, F1 | Bảng kết quả chính và bảng bổ sung | **Có đầy đủ** |
| Confusion Matrix | Sáu ma trận trong notebook | **Có** |
| So sánh đặc trưng và thuật toán | Bảng, biểu đồ, ablation, FP/FN | **Có, khá đầy đủ** |
| Báo cáo đúng cấu trúc | README và notebook chứa nhiều nội dung; chưa có báo cáo học phần hoàn chỉnh riêng trước lần rà soát này | **Chưa đủ căn cứ xác nhận hoàn tất bản nộp** |
| Dữ liệu đủ chất lượng/số lượng | Output v7 có 38.233 mẫu; chưa có audit nhãn độc lập | **Có quy mô thực nghiệm**, chất lượng nhãn cần bằng chứng bổ sung |
| Khác cách làm với nhóm cùng đề tài | Không có project/báo cáo nhóm khác để đối chiếu | **Chưa thể xác minh** |

### Về cách hiểu “mô hình hiện đại”

Ảnh không chỉ định bắt buộc phải là BERT/PhoBERT hay một kiến trúc cụ thể. Vì vậy không thể tự gán thêm tiêu chí của giảng viên. Tuy nhiên, project hiện chỉ đánh giá các classifier NB/SVM; đưa FastText vào bước đặc trưng hoặc dùng LLM hỗ trợ nhãn chưa chứng minh đã thực hiện một mô hình phân loại hiện đại riêng ở bước 4.

Một phương án hoàn thiện rõ ràng là fine-tune **PhoBERT** hoặc mô hình tiếng Việt tương tự cho hai nhãn rồi đánh giá trên cùng cách chia dữ liệu, so sánh với baseline. PhoBERT là mô hình ngôn ngữ tiền huấn luyện cho tiếng Việt theo [bài báo gốc](https://aclanthology.org/2020.findings-emnlp.92/). Đây là đề xuất cho phần còn thiếu, **chưa phải thí nghiệm đã có và chưa biết trước điểm số**.

## 8. Các lỗi và vấn đề cần xử lý

Quy ước bằng chứng: **đã tái hiện** = kiểm tra nhỏ chạy trực tiếp; **đọc mã/output** = thấy trong file; **cần tích hợp** = cần dữ liệu/dịch vụ thật để xác nhận toàn bộ ảnh hưởng.

### 8.1. Ưu tiên cao: thiếu thí nghiệm cho yêu cầu mô hình hiện đại

**Bằng chứng:** toàn bộ mô hình trong notebook là NB/SVM; cuối notebook chỉ nêu fine-tune PhoBERT như hướng phát triển.

**Ảnh hưởng:** chưa thể kết luận đáp ứng đầy đủ đề có dòng bổ sung.

**Cần làm:** thêm một mô hình phân loại hiện đại, ghi cấu hình huấn luyện, cách chọn checkpoint/siêu tham số bằng validation hoặc CV trong train, kết quả test và so sánh. Không dùng chính test để lựa chọn cấu hình. Chưa cần làm giao diện vì đề không yêu cầu.

### 8.2. Ưu tiên cao: kết luận “mô hình tốt nhất” chưa xét đầy đủ kết quả

**Bằng chứng:** cell 28 lưu baseline SVM F1 0,8195, nhưng cell 30 chỉ chọn `best` từ sáu mô hình `sel`, còn cell 36 lưu chính `best` đó. README nổi bật F1 0,7943 mà chưa giải thích phạm vi này.

**Ảnh hưởng:** người đọc dễ hiểu sai rằng 0,7943 là F1 cao nhất của tất cả thí nghiệm; mô hình được lưu không đại diện cho việc so sánh toàn bộ phương án.

**Cần làm:** thống nhất tiêu chí chọn; cho `class_weight` tham gia so sánh CV; xem cả cấu hình điều chỉnh ngưỡng/under-sampling theo quy trình hợp lệ; cập nhật kết luận và artifact tương ứng. Nếu chủ ý ưu tiên Recall, ghi rõ lý do chọn balanced dù F1 thấp hơn.

### 8.3. Ưu tiên cao: `auto_accept_labels.py` chưa đồng bộ ID Phase 2

**Vị trí:** dòng 164 và 221 sinh `rid = f"cmt_{idx}"`, trong khi uploader đã cho phép `p2_cmt_`.

**Đã tái hiện:** với một dòng CSV và cache `p2_cmt_0: {label: spam, confidence: 0.9}`, file xuất nhận `official_label=None`, `is_auto_accepted=False`, dù cache đủ ngưỡng.

**Ảnh hưởng:** nhãn có thể bị bỏ sót trong nhánh xuất CSV tự động; lookup văn bản để cập nhật Argilla cũng không khớp tiền tố. Điều này không chứng minh mọi luồng gán nhãn Phase 2 đều hỏng — nhánh tải response trực tiếp có đường đi khác — nhưng là lỗi thật của nhánh trên.

**Cần làm:** dùng một cơ chế ID chung cho uploader/AI/auto-accept/download; ưu tiên ID ổn định được lưu trong CSV, thay vì chỉ số dòng. Thêm kiểm tra số dòng nối thành công và thử lại bằng ví dụ Phase 2.

### 8.4. Ưu tiên cao: nhánh under-sampling bỏ quy tắc CV theo nhóm

**Bằng chứng:** ở cell 28, `gs_us = GridSearchCV(..., cv=5, ...)`, trong khi thí nghiệm chính dùng `CV_FOLDS` theo hash. Với classifier hiện tại, `cv=5` không mang thông tin nhóm văn bản.

**Ảnh hưởng:** văn bản lặp còn lại trong tập sau lấy mẫu có thể nằm ở cả train/validation, làm kết quả chọn tham số không so sánh công bằng với luồng chính. Tập test bên ngoài vẫn dùng phân chia cũ; đây là vấn đề của **CV bên trong nhánh under-sampling**, không phải bằng chứng test đã bị nhập vào train.

**Cần làm:** giữ khóa nhóm cho tập sau lấy mẫu, tạo CV theo nhóm tương ứng. Thiết kế chặt hơn là chỉ under-sample phần train của mỗi fold, để validation giữ phân bố gốc.

### 8.5. Ưu tiên cao: che tên đang quá rộng và chưa đúng mô tả

**Bằng chứng mã:** trong cell 12, `mask_names()` dùng số token đầu có dạng viết hoa để quyết định che. Hàm `name_like()` có kiểm tra từ điển nhưng không được gọi tại đây; khi từ đầu không thuộc từ điển, số token viết hoa `k` vẫn có thể được dùng.

**Đã tái hiện bằng hàm trích từ notebook:**

```text
Máy Tính học tốt      → user_name học tốt
Nguyễn Văn An ơi      → user_name ơi
bạn Nguyễn Văn An ơi  → bạn Nguyễn Văn An ơi
```

**Ảnh hưởng:** cụm viết hoa không phải tên người có thể bị xóa thông tin; tên nằm giữa câu lại không được che. Mô tả “ẩn danh tên” chưa phản ánh phạm vi thực tế và chưa bảo đảm ẩn danh hoàn toàn.

**Cần làm:** xác định rõ chỉ che tên đầu câu hay mọi vị trí; dùng điều kiện nhận diện thật trước khi thay thế; thử các câu có tên người, tên môn, cụm viết hoa và cụm người tham gia ẩn danh. Khi thay đổi, phải chạy lại thí nghiệm vì đặc trưng đầu vào đổi.

### 8.6. Ưu tiên vừa: đường đi dự đoán dữ liệu mới thiếu bước làm sạch ban đầu

**Bằng chứng:** `predict_spam()` ở cell 36 chỉ gọi hàm tiền xử lý notebook; hàm này không thay URL/SĐT bằng `urltoken`/`phonetoken` như cleaner. Các ví dụ minh họa đã nhập sẵn `urltoken`.

**Ảnh hưởng:** nếu người dùng nhập nguyên văn có URL/SĐT, dữ liệu dự đoán khác dạng mà mô hình đã được huấn luyện. Nạp `.joblib` riêng cũng thiếu phần tiền xử lý ngoài pipeline.

**Cần làm:** tạo luồng dự đoán nhận raw text rõ ràng, gọi cùng bước làm sạch và tách từ như huấn luyện; giữ raw text riêng nếu cần phát hiện emoji. Đóng gói hoặc cung cấp đầy đủ tiền xử lý kèm mô hình và ví dụ load–predict độc lập.

### 8.7. Ưu tiên vừa: confidence thiếu vẫn có nhánh bị nâng thành 1.0

**Bằng chứng:** `predict_batch_with_glm()` dùng mặc định 0.5 nếu thiếu confidence, nhưng dòng 480 khi đọc suggestion sẵn có lại dùng `s.score or 1.0`.

**Ảnh hưởng:** suggestion có score `None` hoặc `0` bị đưa vào cache thành 1.0, có thể đi vào nhánh duyệt tự động. README mô tả đã xử lý an toàn confidence thiếu nhưng chưa bao quát đường đi này.

**Cần làm:** xử lý thiếu score nhất quán, phân biệt `None` với 0; kiểm tra miền giá trị hữu hạn và nhãn hợp lệ. Dùng một tập đã gán nhãn thủ công để đo chất lượng nhóm auto-accept thay vì coi confidence là bảo đảm đúng.

### 8.8. Ưu tiên vừa: cache và lookup chưa gắn chặt với nguồn dữ liệu

**Bằng chứng:** cache dùng ID làm khóa, chưa lưu hash văn bản/dataset/prompt để xác nhận tương thích khi chạy tiếp. Download lookup dựa vào vị trí dòng CSV và bỏ ID không parse được.

**Ảnh hưởng:** đổi CSV nhưng tái sử dụng ID/cache cũ có thể lấy nhầm nhãn; dùng chung dataset nhiều phase nhưng lookup một CSV có thể nối sai metadata. Đây là rủi ro từ thiết kế, chưa xác nhận đã xảy ra trên nhãn v7.

**Cần làm:** lưu ID gốc và nguồn ổn định, gắn cache với phiên bản dữ liệu/model/prompt, báo số ID thiếu/trùng/không nối được; đối soát tổng số trước và sau export.

### 8.9. Ưu tiên vừa: cấu hình và các chế độ CLI chưa đúng như gợi ý

| Vấn đề | Bằng chứng và cách xử lý |
|---|---|
| `--no-upload` vẫn cần Argilla | `main()` kết nối và thiết lập dataset trước khi xét upload. Nếu muốn chạy CSV ngoại tuyến với Argilla, cần tách luồng lấy CSV khỏi khởi tạo client |
| Biến môi trường Argilla chưa được dùng đúng | Dòng 98–99 đọc `env_argilla_url`, `env_argilla_key`, nhưng `--argilla-url`/`--argilla-key` vẫn default từ hằng. Cần dùng giá trị môi trường đã đọc hoặc sửa tài liệu |
| Cache rỗng gây lỗi | Đã tái hiện `ZeroDivisionError` trong auto-accept; cần xử lý tổng bằng 0 trước tính phần trăm |
| Export không còn nhãn hợp lệ | Download tạo DataFrame từ `rows=[]` rồi truy cập `df['label']`; cần tạo schema cố định và xử lý xuất rỗng |
| Merge thiếu xác thực đầu vào | Nên kiểm tra cột bắt buộc, nhãn, ID trùng/rỗng và thống kê xung đột nhãn; hiện chủ yếu bảo toàn Phase 1 |

### 8.10. Ưu tiên vừa: môi trường và dữ liệu chưa đủ để tái lập

**Hiện trạng local:** không có `data/`, `.venv/`, `namedb_cache/` hay artifact `.joblib`. Python hiện được gọi là 3.14.5, trong khi metadata notebook ghi 3.13.5. Kiểm tra phát hiện module cho thấy máy có một số thư viện, nhưng thiếu `gensim` và `argilla` trong interpreter này.

**Thiếu tài liệu/cấu hình:** chưa có `requirements.txt`/lock file, cấu hình Argilla local, phiên bản dataset được pin theo revision/hash và hướng dẫn hoàn chỉnh để tái tạo v7.

**Cần làm:** ghi phiên bản môi trường đã chạy thành công; khóa dependencies phù hợp; ghi revision và SHA-256 dataset; cung cấp cách lấy dữ liệu hợp lệ hoặc dữ liệu mẫu; sửa README theo Windows và môi trường thực tế. Không kết luận Python hiện tại chắc chắn không tương thích chỉ từ khác số phiên bản.

**Kaggle:** notebook chưa có mã đọc `UserSecretsClient().get_secret('HF_TOKEN')`, dù README hướng dẫn dùng Kaggle Secrets. Tài liệu chính thức của [Kaggle về Secrets](https://www.kaggle.com/product-feedback/114053) cho thấy cách lấy secret bằng client; [Hugging Face](https://huggingface.co/docs/huggingface_hub/main/en/package_reference/environment_variables) hỗ trợ token qua `HF_TOKEN`. Cần nối hai cơ chế trong notebook hoặc hướng dẫn rõ cách đưa token vào môi trường trước khi tải, không chỉ giả định thêm secret là tự đọc được.

### 8.11. Các điểm nên sửa thêm trong diễn giải và bảo trì

- **Emoji chưa được gom hoàn toàn:** output cell 13 có `😂 😂 emoji_token`; code thêm token nhưng chưa xóa emoji còn trong chuỗi đưa vào tách từ. Có đúng một token bổ sung không đồng nghĩa tất cả emoji gốc đã bị thay thế. Cần thống nhất cách biểu diễn và tài liệu.
- **Tái lập FastText:** `workers=4`, dữ liệu/từ điển tải từ nhánh thay đổi và pip không khóa phiên bản không hỗ trợ lời hứa “chạy lại sẽ cho kết quả như vậy” tuyệt đối. Ghi sai số cho phép hoặc cấu hình tái lập chặt hơn.
- **Hai nơi chứa cùng lớp:** cell 3 ghi đè `spam_features.py`. Hiện hai bản giống nhau; về sau sửa riêng file có thể bị notebook ghi đè. Nên xác định nguồn chuẩn và cách dùng trên Kaggle/local.
- **Tên chỉ số:** code tính `average_precision_score`; nên đặt cột `Average Precision (AP)` hoặc giải thích rõ, tránh mặc định hiểu là diện tích PR tính bằng quy tắc hình thang.
- **Khoảng tin cậy:** bootstrap đang lấy lại từng dòng; khi test còn nhiều bình luận thuộc cùng nhóm, có thể xem xét bootstrap theo nhóm. Khoảng F1 từng mô hình chồng lấp chưa đủ để kết luận hai mô hình tương đương.
- **Lưu kết quả độc lập:** nên xuất bảng metrics, dự đoán test và cấu hình ra CSV/JSON cùng mã phiên bản, thay vì chỉ dựa vào output notebook. Không có output error không bảo đảm notebook vừa được chạy lại toàn bộ ở phiên bản mã hiện tại.
- **Chất lượng nhãn:** lưu hướng dẫn, ai/AI nào gán nhãn, tỷ lệ được kiểm tra thủ công, cách giải quyết bất đồng và các thay đổi tới v7. Dữ liệu đánh giá phụ thuộc nhãn AI cần có một phần đối chiếu thủ công đáng tin cậy.
- **Ngữ cảnh:** nếu định nghĩa spam dựa vào bài đăng, lặp hàng loạt hoặc tác giả, cần lưu và khai thác thông tin đó; classifier hiện chủ yếu chỉ nhìn văn bản. `comment_type`/`reaction_count` đang dùng khám phá/metadata, không được đưa vào các pipeline phân loại chính.

## 9. Project đã xong chưa? Đánh giá tổng thể

### 9.1. Những phần đã có nền tảng tốt

1. Bài toán rõ ràng, bám nguồn dữ liệu được phép trong đề.
2. Có luồng chuẩn bị dữ liệu và công cụ gán nhãn, không chỉ có cell train mô hình.
3. Thực hiện đủ ba cách biểu diễn và hai họ thuật toán, có đối chứng stop words.
4. Chọn siêu tham số bằng CV và có cơ chế nhóm văn bản chống lặp ở luồng chính.
5. Đánh giá chú ý lớp spam, có confusion matrix, AP/PR curve, CI và phân tích lỗi.
6. Có thử nghiệm class weight, chỉnh ngưỡng, under-sampling và phân tích OOV.

### 9.2. Đánh giá theo ba mục đích khác nhau

| Mục đích | Nhận định |
|---|---|
| Hiểu và trình bày một bài thực nghiệm học máy cổ điển | **Đã có đủ nội dung chính**, nhưng phải sửa các kết luận chưa khớp kết quả |
| Nộp bài theo toàn bộ ảnh đề, kể cả bổ sung | **Chưa thể coi hoàn tất**, cần giải quyết mô hình hiện đại, báo cáo và xác nhận tái lập |
| Dùng kiểm duyệt Facebook thực tế | **Chưa sẵn sàng**, còn thiếu luồng raw-text inference độc lập, tích hợp và đánh giá vận hành |

Không chấm một điểm số hay phần trăm hoàn thành tùy ý vì ảnh không cung cấp thang điểm và bản local chưa có đủ đầu ra bàn giao. Đánh giá hợp lý là: **phần thực nghiệm cốt lõi khá đầy đủ; độ nhất quán tài liệu, độ tin cậy quy trình dữ liệu và tính tái lập cần hoàn thiện**.

Việc chưa có giao diện/web không phải thiếu bắt buộc của đề. Với đầu ra hiện có, phù hợp nhất là định vị hệ thống hỗ trợ đưa bình luận vào hàng chờ kiểm tra, không khẳng định đã đủ bằng chứng để tự động xóa bình luận hợp lệ mà không cần giám sát.

## 10. Thứ tự hoàn thiện và bàn giao

### 10.1. Nên làm trước khi viết kết luận cuối và nộp

1. **Chuẩn hóa căn cứ dữ liệu:** xác nhận đúng `labeled_v7.csv`, revision/hash, nhãn và nguồn; làm rõ quá trình từ bản gộp sang v7.
2. **Sửa các lỗi ảnh hưởng dữ liệu/thực nghiệm:** ID Phase 2, che tên, confidence thiếu, CV under-sampling và luồng dự đoán raw text.
3. **Thống nhất cách chọn mô hình:** đưa các lựa chọn cân bằng lớp vào đánh giá bằng CV; ghi rõ mục tiêu F1 hay ưu tiên Recall.
4. **Thực hiện mô hình hiện đại:** chạy thí nghiệm có ghi nhận cấu hình và kết quả thật; chưa có kết quả thì trình bày là chưa làm.
5. **Chạy lại từ đầu trong môi trường được ghi phiên bản:** kiểm tra không phụ thuộc trạng thái cell cũ, lưu notebook output và artifacts cần thiết. Nếu quá trình phát triển đã dùng test nhiều lần để điều chỉnh phương án, cần cân nhắc thêm tập kiểm tra độc lập cho đánh giá cuối.
6. **Viết báo cáo và cập nhật README:** dùng số liệu từ lần chạy đã chốt, chỉ rõ mô hình cuối cùng và giới hạn.

### 10.2. Khung báo cáo học phần còn cần hoàn thiện

| Phần theo đề | Nội dung nên có từ project |
|---|---|
| Tiêu đề | Tên bài toán, môn học, nhóm/thành viên |
| Giới thiệu bài toán | Bối cảnh group PTIT, định nghĩa spam/ham, mục tiêu và phạm vi |
| Cơ sở lý thuyết | Tiền xử lý tiếng Việt, BoW, TF-IDF, FastText, NB, SVM và mô hình hiện đại được chọn |
| Thực nghiệm và kết quả | Nguồn/phiên bản dữ liệu, nhãn, split, CV, tham số, bảng điểm, confusion matrix, đối chứng, phân tích lỗi |
| Tài liệu tham khảo | Nguồn dữ liệu/crawler, thư viện và bài báo mô hình; thống nhất cách trích dẫn |

Tài liệu đang đọc là **bản giải thích và kiểm tra project**, không tự động thay thế báo cáo học phần hoàn chỉnh hoặc các thí nghiệm chưa thực hiện.

### 10.3. Gói bàn giao nên có

| Thành phần | Hiện trạng trước bản kiểm tra này |
|---|---|
| Mã nguồn và notebook | Có |
| Ảnh đề bài và phân công | Có local; ảnh chưa được theo dõi, phân công đang bị ignore |
| Dữ liệu hoặc hướng dẫn truy cập đúng phiên bản | Có mô tả nguồn, chưa có dữ liệu local và revision/hash được chốt trong repo |
| Môi trường/requirements có phiên bản | Chưa có |
| Kết quả chạy cuối, metrics/config độc lập | Notebook có output; chưa có bộ kết quả độc lập |
| Mô hình và hướng dẫn load/predict đầy đủ | Có code lưu; artifact local chưa có; còn thiếu đóng gói tiền xử lý |
| Thí nghiệm mô hình hiện đại | Chưa có bằng chứng |
| Báo cáo theo cấu trúc đề | Chưa thấy file hoàn chỉnh riêng trong checkout |

Nếu muốn đồng đội nhận được tài liệu phân công, báo cáo hoặc artifacts, cần quyết định rõ cách chia sẻ vì `.gitignore` đang loại nhiều loại file. Không cần tạo ZIP nếu hình thức nộp không yêu cầu.

## 11. Những điều cần hiểu khi trình bày project

### “Số lượng lớn và Accuracy cao có đủ chứng minh tốt không?”

Không. Riêng test hiện có 6.524 ham/7.480 mẫu; đoán toàn ham cũng đạt khoảng **87,22% Accuracy** nhưng Recall spam bằng 0. Cần đọc Precision, Recall, F1 của spam cùng confusion matrix và lỗi thực tế.

### “BoW, TF-IDF và FastText khác nhau ở đâu?”

BoW đếm token; TF-IDF giảm trọng số token quá phổ biến và nhấn mạnh token đặc trưng; FastText học vector đặc từ ngữ cảnh và các mảnh ký tự. FastText có thể xử lý từ lạ nhưng việc trung bình vector làm mất thông tin thứ tự và có thể làm loãng tín hiệu. Kết quả nào tốt hơn phải dựa vào thí nghiệm trên bộ dữ liệu cụ thể.

### “SVM tốt hơn Naive Bayes trong mọi trường hợp không?”

Trong sáu tổ hợp chính đã lưu, SVM có F1/Recall cao hơn nhánh NB tương ứng. Tuy nhiên NB TF-IDF có Precision cao, và chỉnh ngưỡng làm thay đổi đánh đổi. Không suy rộng thành SVM luôn tốt nhất mọi dữ liệu hoặc mọi mục tiêu.

### “Tại sao nhãn của bình luận chỉ có tên người hoặc dấu chấm lại khó?”

Một câu ngắn như `Có.` có thể là trả lời hợp lệ hoặc seeding tùy bài đăng và hành vi người dùng. Nếu người gán nhãn dựa vào thông tin ngoài câu nhưng mô hình chỉ nhận câu, có thể không đủ dữ kiện để dự đoán nhất quán. Cần giải thích giới hạn của định nghĩa nhãn và thông tin đầu vào.

### “Có thể nói dữ liệu đã được ẩn danh hết chưa?”

Chưa. Raw text vẫn được giữ trong dữ liệu, che tên hiện mới ở một số trường hợp, và một số ví dụ output vẫn có tên. Nếu chia sẻ dữ liệu hoặc gửi bình luận sang dịch vụ gán nhãn bên ngoài, cần xác định rõ phần nào đã được thay thế và phạm vi chia sẻ phù hợp; không mô tả đây là bộ dữ liệu đã ẩn danh hoàn toàn.

### “Có thể nói đã chạy thành công trên máy hiện tại chưa?”

Chưa thể nói toàn bộ như vậy. Notebook có chuỗi execution count và output không có lỗi được lưu; lần rà soát này xác nhận cấu trúc, cú pháp và một số hành vi nhỏ. Nó không thay thế việc train lại đầy đủ, kết nối Argilla hoặc tải dataset thật.

## 12. Phạm vi kiểm chứng và nguồn tham chiếu

### 12.1. Đã thực hiện trong lần kiểm tra này

- Đọc toàn bộ 8 script Python, README, phân công nhóm và `.gitignore`; đọc trực tiếp ảnh đề bài.
- Đọc toàn bộ source của 38 cell notebook và các output dạng văn bản/bảng HTML; kiểm tra các phần tạo biểu đồ và đánh giá.
- Đối chiếu danh sách file local, file Git theo dõi, ignore và commit hiện tại.
- Kiểm tra AST/cú pháp của cả 8 script: đều phân tích được.
- Kiểm tra biên dịch 18 cell Python thông thường: đều được; hai cell lệnh cài đặt/`%%writefile` được đọc riêng vì là cú pháp IPython.
- Kiểm tra notebook bằng `nbformat.validate`: hợp lệ. Không có output loại `error` được lưu trong notebook.
- Đối chiếu lớp FastText trong notebook với `spam_features.py`: hiện trùng nhau.
- Chạy kiểm tra nhỏ cho cleaner, converter và logic nạp Phase 2; tái hiện lỗi che tên, lỗi export ID `p2_cmt_` và lỗi cache rỗng. Các kiểm tra logic auto-accept dùng dữ liệu trong bộ nhớ và Argilla giả lập, không phải máy chủ thật.
- Kiểm tra khả năng tìm thấy các module trong Python hiện tại; thiếu `gensim` và `argilla`. Việc tìm thấy một module không đồng nghĩa toàn bộ chức năng thư viện đó đã được kiểm thử.

### 12.2. Chưa thực hiện và không suy diễn thành kết quả

- Chưa tải hoặc đọc toàn bộ dataset gated; chưa kiểm chứng lại số dòng, nguồn Facebook và nhãn v7 trên dữ liệu thật.
- Chưa chạy lại toàn notebook, GridSearchCV, FastText hay bootstrap trên dataset thật.
- Chưa kết nối Argilla, gọi API GLM/OpenRouter, thay đổi nhãn trên dịch vụ hoặc kiểm thử cập nhật records thực tế.
- Chưa xác nhận có thể phục hồi toàn bộ dữ liệu v7 từ dữ liệu thô bằng những script hiện có.
- Chưa đánh giá mô hình hiện đại hoặc khả năng hoạt động trên bình luận tương lai/nhóm khác.
- Không chỉnh sửa các file mã nguồn hoặc notebook trong lần kiểm tra này; tài liệu này ghi nhận hiện trạng và đề xuất sửa.

### 12.3. Cách hiểu mức tin cậy của các kết luận

Các nhận xét về thiếu file, đường đi mã, bảng số liệu lưu sẵn và các lỗi đã tái hiện có bằng chứng trực tiếp tại checkout. Các số liệu thực nghiệm là **kết quả có sẵn trong notebook**, không được trình bày như số vừa đo lại. Các rủi ro liên quan dữ liệu thật/SDK/dịch vụ được ghi rõ là cần kiểm chứng tích hợp.

Những tham chiếu ngoài repo chỉ dùng để kiểm tra cơ chế SDK, secret, tái lập hoặc gợi ý mô hình, đã gắn link tại đoạn liên quan. Chúng không được dùng thay cho bằng chứng rằng project hiện đã triển khai các phần đó.
