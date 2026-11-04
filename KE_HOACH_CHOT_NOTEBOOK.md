# Kế hoạch chốt notebook nộp bài Spam/Ham PTIT

Ngày: **04/11/2026**. Căn cứ mã nguồn hiện tại: Git HEAD `efc4fba`.

## Phạm vi đã chốt

- Notebook là file mã nguồn nộp chính và là nội dung dùng để giải thích quá trình thực hiện.
- Dùng tập bình luận PTIT đã thu thập và gán nhãn sẵn. Không thiết kế lại hệ thống crawl/gán nhãn định kỳ.
- Hoàn thiện code, chạy và chốt notebook trước; viết báo cáo học phần sau.
- Tài liệu này là danh sách công việc và phương án thực hiện, chưa phải xác nhận đã sửa code hoặc đã chạy các thí nghiệm mới.

**Kết luận:** giữ nền tảng BoW/TF-IDF/FastText + Naive Bayes/SVM hiện có. Bổ sung một mô hình phân loại hiện đại; sửa các lỗi ảnh hưởng tiền xử lý, đánh giá, dự đoán và khả năng chạy notebook. Không mặc định yêu cầu sửa hoặc chạy lại tất cả script chuẩn bị dữ liệu.

## Cách đọc mức độ quan trọng

| Nhãn | Ý nghĩa | Cách xử lý |
|---|---|---|
| **ĐỀ — rất cao** | Nội dung đề trực tiếp yêu cầu, hiện còn thiếu hoặc cần thể hiện rõ | Phải hoàn thiện trước khi kết luận đủ phần code |
| **ĐÚNG/CHẠY — cao** | Không phải tính năng được đề nêu riêng, nhưng cần để phần đang trình bày đúng và chạy được | Sửa trước bản nộp; nếu là phần nâng cao thì có thể bỏ phần đó thay vì giữ sai |
| **NÊN — vừa** | Giúp bài dễ giải thích, kiểm chứng hoặc tái chạy hơn; đề không bắt buộc riêng | Làm nếu phù hợp, không biến thành điều kiện đạt đề |
| **TÙY CHỌN — thấp** | Mở rộng thêm; đề không yêu cầu | Có thể bỏ để tập trung hoàn thiện |
| **ĐIỀU KIỆN** | Chỉ cần sửa/chạy lại khi xác nhận lỗi có ảnh hưởng dataset hoặc đầu ra đang dùng | Kiểm tra căn cứ trước, không chạy lại hàng loạt |

Lỗi cần sửa cho kết quả đúng khác với một kỹ thuật nâng cao cần bổ sung. Ví dụ: under-sampling không bắt buộc; nhưng nếu giữ bảng kết quả under-sampling thì phải sửa cách CV của nhánh đó.

## 1. Các phần phải sửa hoặc nên cải thiện trong notebook

Số cell dưới đây đếm từ **1**, theo vị trí trong `spam-ham-comments.ipynb`, không phải số `In [n]`. Số cell sẽ thay đổi khi thêm/bớt nội dung.

### 1.1. Các phần gắn trực tiếp với đề bài

| Mã | Mức | Làm gì, sửa ở đâu? | Vì sao và tác dụng | Dấu hiệu hoàn thành |
|---|---|---|---|---|
| N01 | ĐỀ — rất cao | Thêm một nhánh phân loại hiện đại ở bước 4; phương án tại mục 3 | Đáp ứng dòng bổ sung trong ảnh đề; hiện notebook mới đánh giá NB/SVM | Có mã huấn luyện, cấu hình, kết quả thật và bảng so sánh |
| N02 | ĐỀ — cao | Bổ sung Markdown và thống kê tại mục dữ liệu, cell 5–8: nguồn PTIT, quá trình thu thập, tiêu chí spam/ham, vai trò AI và con người, phiên bản v7 | Khi chỉ trình bày notebook, người đọc vẫn hiểu bước thu thập/gán nhãn thay vì chỉ thấy tải một CSV | Có mô tả dựa trên thông tin thật; các số đếm lấy từ CSV đang dùng; phần chưa biết ghi rõ |
| N03 | ĐỀ — cao | Đưa mô hình hiện đại vào phần đánh giá với Accuracy, Precision, Recall, F1, Confusion Matrix; so sánh cùng dữ liệu test | Đề yêu cầu đánh giá và so sánh các thuật toán | Bảng tổng hợp và confusion matrix có cả nhánh mới |
| N04 | ĐỀ — duy trì | Giữ ít nhất hai cách đặc trưng, hai thuật toán, train/test, tiền xử lý và so sánh | Những nội dung này đã có, không cần viết lại toàn bộ | Các cell liên quan vẫn chạy và giải thích được |

Khuyến nghị giữ cả ba đặc trưng hiện có vì đã có cấu trúc so sánh tốt. Đề chỉ yêu cầu ít nhất hai. Thí nghiệm giữ/bỏ stop words cũng đã có, không cần ép chọn bỏ stop words khi kết quả không tốt.

Không cần crawl lại Facebook trong mỗi lần Run All. Mục thu thập có thể mô tả quá trình đã thực hiện và nạp dataset cố định. Không tự bịa tên group, thời gian crawl, số người rà nhãn hay tỷ lệ duyệt thủ công khi chưa có căn cứ.

### 1.2. Các lỗi cần sửa để notebook hiện có đúng và chạy ổn

| Mã | Mức | Hiện trạng | Việc sửa cụ thể | Tác dụng |
|---|---|---|---|---|
| N05 | ĐÚNG/CHẠY — cao | Cell 2 dùng `!pip`, không chốt môi trường đã thử | Dùng `%pip` hoặc cách cài gắn đúng kernel; ghi phiên bản Python/thư viện và kiểm tra import sau cài | Giảm lỗi cài vào nhầm môi trường, xác định được môi trường bản nộp |
| N06 | ĐÚNG/CHẠY — cao | Cell 6 mặc định tải dataset có hạn chế truy cập | Có cấu hình CSV rõ ràng, hỗ trợ CSV đính kèm hoặc nguồn HF được cấp quyền, thông báo thiếu dữ liệu cụ thể | Người nhận biết cách chạy và không phụ thuộc đường dẫn riêng trên máy nhóm |
| N07 | ĐÚNG/CHẠY — cao | Đọc CSV xong lọc nhãn/thiếu text nhưng chưa đối soát đầy đủ | Kiểm tra cột, nhãn, văn bản rỗng, ID thiếu/trùng; in số dòng trước/sau lọc và thống kê xung đột | Phát hiện dữ liệu bị hụt hoặc không hợp lệ trước huấn luyện |
| N08 | ĐÚNG/CHẠY — cao | `mask_names()` ở cell 12 có thể che cụm viết hoa không phải tên, như `Máy Tính`; không che mọi vị trí trong câu | Sửa điều kiện nhận diện; chốt rõ phạm vi che tên; kiểm tra tên người, tên môn/cụm viết hoa và tên giữa câu | Không xóa nhầm thông tin, mô tả khớp hành vi. Che tên không bắt buộc theo đề; nếu không giữ thì bỏ sạch logic và chạy lại |
| N09 | ĐÚNG/CHẠY — vừa | Cell 13 thêm `emoji_token` nhưng output vẫn có emoji gốc | Chốt một cách xử lý: nhận diện từ raw text, bỏ emoji gốc nếu chủ đích là thay bằng một token; dùng một regex thống nhất | Đúng mô tả, tránh giữ đồng thời hai biểu diễn ngoài ý muốn. Xử lý emoji là lựa chọn kỹ thuật, đề không ép dùng token này |
| N10 | ĐÚNG/CHẠY — cao | Nhánh under-sampling ở cell 28 dùng `cv=5`, không theo nhóm văn bản | Nếu giữ: tạo CV theo nhóm trên train gốc, under-sample riêng phần train của từng fold; validation giữ nguyên. Nếu bỏ: bỏ cả code, bảng và nhận xét liên quan | So sánh hợp lệ, giảm lặp văn bản giữa train/validation. Under-sampling không phải yêu cầu bắt buộc của đề |
| N11 | ĐÚNG/CHẠY — cao | Cell 30 chọn `best` trong sáu tổ hợp chính, còn các phương án bổ sung ở cell 28 đứng ngoài; cell 36 lưu `best` | Tách rõ bảng thí nghiệm chính và bổ sung; nếu chọn mô hình cuối trên toàn bộ phương án thì dùng CV/validation thống nhất. Ghi phạm vi “tốt nhất”, lưu đúng mô hình đã chọn | Bảng kết quả, kết luận và mô hình lưu nhất quán; không chọn mô hình chỉ vì điểm test cao |
| N12 | ĐÚNG/CHẠY — vừa | `predict_spam()` ở cell 36 chưa thay URL/SĐT như bước chuẩn bị dữ liệu | Nhúng các hàm làm sạch cần thiết vào notebook; dùng cùng tiền xử lý cho huấn luyện/dự đoán; minh họa bằng bình luận raw có URL và SĐT | Phần demo nhận bình luận thật, không cần người dùng viết sẵn `urltoken`. Demo không bắt buộc riêng trong đề; nếu giữ phải đúng |
| N13 | ĐÚNG/CHẠY — cao | Cell 37 và một số câu in có kết luận quá mạnh hoặc sai | Viết lại theo lần chạy cuối; sửa “Recall 78,3% bỏ sót phần lớn”; nêu đúng đánh đổi Precision/Recall; không coi mọi vector gần là đồng nghĩa | Trình bày và bảo vệ kết quả đúng |
| N14 | ĐÚNG/CHẠY — cao | Output đã lưu không chứng minh mã sau sửa đã chạy hết | Sau khi chốt dữ liệu/code: Restart kernel, Run All trong thư mục chỉ có notebook và dữ liệu được cấu hình; lưu output cuối | Xác nhận bản nộp không dựa vào trạng thái cell cũ hoặc file Python chuẩn bị sẵn |

**Lưu ý N11:** output cũ lưu TF-IDF + SVM balanced F1 0,7943, trong khi SVM không class weight đạt F1 test 0,8195 ở bảng bổ sung. Đây là căn cứ để sửa cách trình bày/lựa chọn, không phải lý do dùng test để mặc định đổi mô hình thắng cuộc. Những số này chưa được đo lại trong lượt lập kế hoạch.

**Lưu ý về chia tập:** giữ ý tưởng nhóm hash đang có. Cần kiểm tra nhóm trùng thật và trường hợp chỉ dấu câu đang được gán khóa riêng theo dòng. Không mặc định coi mọi câu có cùng vector hoặc cùng token `user_name` là cùng một bình luận; gom quá rộng có thể ghép các câu khác ngữ nghĩa. Hash trước bước che tên không tự nó chứng minh có rò rỉ dữ liệu.

**Lưu ý N12:** lưu `.joblib` không phải yêu cầu riêng của đề. Nếu giữ phần này thì bổ sung đủ thông tin tiền xử lý và thử load–predict. Không cần mở rộng thành một dịch vụ triển khai.

### 1.3. Những cải thiện nên có, đề không yêu cầu riêng

| Mã | Mức | Cải thiện | Lý do và tác dụng |
|---|---|---|---|
| N15 | NÊN — vừa | Chốt mã phiên bản hoặc SHA-256 dataset; giữ bản CSV cuối cố định | Biết các lần chạy và bảng điểm dùng cùng dữ liệu; không biến thành một hệ thống quản lý dữ liệu phức tạp |
| N16 | NÊN — vừa | Nhúng từ điển tên cố định cần thiết thay vì tải nhiều file từ GitHub ở cell 12 | Giảm một phụ thuộc mạng; nếu dùng nguồn ngoài thì giữ thông tin nguồn phù hợp. Đây là đề xuất ưu tiên cho notebook dễ chuyển máy |
| N17 | NÊN — vừa | Đổi nhãn chỉ số tính bằng `average_precision_score` thành `AP`; giữ PR curve nếu hữu ích | Tránh nhầm AP với cách tính diện tích PR khác. AP và PR curve là phần thêm ngoài bộ chỉ số bắt buộc |
| N18 | NÊN — vừa | Có baseline dự đoán toàn ham | Giải thích dễ hiểu tác động mất cân bằng và vì sao Accuracy chưa đủ |
| N19 | NÊN — vừa | Hiện cảnh báo quan trọng, ghi thời gian từng nhánh; không tắt toàn bộ warnings | Phát hiện lỗi hội tụ, biết tài nguyên thực tế và thuận tiện giải thích quá trình |
| N20 | NÊN — vừa | Xuất metrics/config hoặc dự đoán test ra CSV/JSON; thử tải lại mô hình nếu lưu | Đối chiếu và tái sử dụng dễ hơn; file này là đầu ra, không nhất thiết phải nộp kèm nếu không yêu cầu |
| N21 | TÙY CHỌN — thấp | Giữ bootstrap CI, OOV, phân tích riêng FastText nếu còn hữu ích | Làm bài sâu hơn nhưng không thay thế yêu cầu mô hình hiện đại. Nếu giữ CI cần diễn giải đúng, không kết luận tương đương chỉ từ CI chồng lấp |
| N22 | TÙY CHỌN — thấp | Chạy nhiều seed, thử thêm mô hình, ensemble TF-IDF + FastText, khảo sát nhiều ngưỡng | Có thể đánh giá sâu hơn; chưa cần cho mục tiêu chốt bài này |

Không cần thêm website, API, Docker triển khai, crawler định kỳ, hệ thống tài khoản hay huấn luyện lại LLM gán nhãn để đáp ứng đề đang xét.

### 1.4. Chốt cách nộp một notebook

Notebook đang tự tạo `spam_features.py` bằng cell 3 trước khi import ở cell 4. Hai bản lớp FastText hiện giống nhau. Vì vậy không có căn cứ nói tách notebook chắc chắn lỗi thiếu file Python này.

Phương án ít thay đổi nhất: giữ cell sinh module để dùng Pipeline/GridSearchCV, bảo đảm nó chạy trước import và thư mục cho phép ghi. Không buộc phải chuyển lớp vào cell thông thường vì còn liên quan cách chạy đa tiến trình/cache và lưu mô hình.

Với dữ liệu, có hai lựa chọn rõ ràng:

| Lựa chọn | Cách dùng | Giới hạn cần ghi |
|---|---|---|
| Chạy qua HF | Notebook tự tải đúng dataset sau khi đã có quyền/token | Chỉ nhận một `.ipynb` nhưng vẫn cần mạng và quyền dataset; không hardcode token vào file |
| Chạy qua CSV sẵn có | Cấu hình đường dẫn CSV trên máy hoặc gắn dataset trong Kaggle | Không cần gọi lại Argilla; người chạy vẫn phải có CSV đúng phiên bản |

Nếu chạy Kaggle, cần mã đọc secret vào cấu hình tải dữ liệu; chỉ tạo secret chưa thay thế mã lấy secret. Cơ chế được thể hiện trong [mã Kaggle chính thức](https://github.com/Kaggle/docker-python/blob/main/patches/kaggle_secrets.py).

Không thể đồng thời hứa “chỉ một file, không dữ liệu bên ngoài, không mạng” với cách tải HF hiện tại. Ảnh đề không quy định phải nhúng toàn bộ dataset vào notebook; không cần tự thêm yêu cầu này. Hình thức nộp cuối vẫn theo hướng dẫn nộp của giảng viên; kế hoạch này xử lý khả năng chạy của file code.

## 2. Những file ngoài notebook nào thật sự cần sửa?

### 2.1. Nguyên tắc quyết định

**Không chạy lại một script chỉ vì phát hiện nó có thể lỗi ở tình huống khác.** Với dataset cố định, cần kiểm tra chính dữ liệu đã tạo và đường đi đã sử dụng. Nhưng việc chỉ chạy script một lần cũng không bảo đảm dữ liệu không bị lỗi: lỗi ở lần chạy đó vẫn có thể ảnh hưởng toàn bộ kết quả.

Thực hiện một lần kiểm tra CSV cuối trong notebook: cột/nhãn hợp lệ; text/ID rỗng; số dòng trước/sau lọc; ID trùng; cùng văn bản khác nhãn; phân bố nhãn; đối chiếu đúng phiên bản và, khi có căn cứ nguồn, đúng số bản ghi dự kiến. Số dòng bằng số README không tự chứng minh nhãn đúng; cần xem mẫu nhãn và đối chiếu những nhóm nghi vấn.

Nếu CSV đạt kiểm tra phù hợp và không có bằng chứng lỗi ở luồng đã chạy, có thể giữ các script một lần nguyên trạng cho bản nộp. Đây là quyết định giới hạn phạm vi, không phải xác nhận mọi nhánh script đều đã đúng.

### 2.2. Quyết định cho từng file

| File | Mức và quyết định cho bản nộp | Khi nào phải sửa/chạy lại? | Tác dụng của việc sửa nếu cần |
|---|---|---|---|
| `spam_features.py` | Liên quan trực tiếp; hiện **không có lỗi bắt buộc sửa đã xác nhận**. Notebook có bản nhúng giống file | Nếu thay đổi lớp FastText thì đồng bộ bản nhúng hoặc xác định một nguồn chuẩn; chạy lại nhánh bị thay đổi | Tránh notebook và file ngoài dùng hai transformer khác nhau |
| `clean_data_facebook_spamham.py` | Giữ nếu CSV cuối đúng; đưa các hàm cần cho demo vào notebook mà không bắt buộc import file này | Nếu phát hiện chính bước clean đã xóa/méo văn bản cần thiết; khi đó sửa và tái tạo cleaned text từ raw text phù hợp | Khôi phục nội dung đầu vào đúng; không nhất thiết gán nhãn lại tất cả |
| `convert_phase2_json.py` | Chưa thấy lý do bắt buộc sửa cho tập đã có | Nếu đối chiếu JSON thật phát hiện mất comments/replies hoặc sai metadata cần dùng. File hiện chỉ duyệt một tầng replies | Khôi phục phần dữ liệu bị bỏ sót; xác định phần nào cần gán nhãn bổ sung |
| `upload_to_argilla.py` | Không cần sửa nếu không upload lại và ID dữ liệu đã đúng | Nếu phải upload bổ sung mà ID có nguy cơ đụng nhau hoặc không nối được nguồn | Giữ ánh xạ dữ liệu, tránh cập nhật nhầm record |
| `auto_accept_labels.py` | **ĐIỀU KIỆN — cao:** có lỗi thật với export cache ID `p2_cmt_`; không mặc định buộc chạy lại | Nếu dataset cuối được tạo qua nhánh lỗi và bị thiếu `official_label`/mất mẫu khi merge: sửa ID rồi export/merge lại phần đó. Có thể tận dụng cache/nhãn đã có | Khôi phục nhãn đã tồn tại, không tốn công gọi LLM lại vô ích |
| `download_labeled_data.py` | **ĐIỀU KIỆN — cao:** kiểm tra lookup và số dòng | Nếu có record bị loại do ID không parse được hoặc text/metadata nối sai CSV: sửa export và tải/ghép lại phần cần thiết | Đảm bảo nhãn đúng văn bản và đủ record |
| `merge_labeled_data.py` | **ĐIỀU KIỆN — cao:** kiểm tra dữ liệu cuối, không tái gộp vô điều kiện | Nếu xác nhận ID trùng không hợp lệ, mất phase hoặc xung đột/lặp dữ liệu ngoài chủ đích: sửa quy tắc và tạo lại CSV. Không xóa mọi câu trùng một cách mù quáng | Tránh đếm sai hoặc dùng dữ liệu không đúng chủ đích trong đánh giá |
| `ai_assisted_spam_ham.py` | Không cần gọi lại API khi nhãn cuối đã được kiểm tra. `--no-upload`, env và resume là sửa bảo trì nếu dùng lại | Nếu nhãn v7 thực sự chứa nhóm bị auto-accept do confidence thiếu bị nâng thành 1.0: rà soát nhóm đó. Sửa mã trước nếu phải dùng nhánh này lại | Sửa phần nhãn có căn cứ nghi vấn; không đồng nghĩa phải gán nhãn lại toàn bộ |
| `README.md` | **NÊN — vừa:** cập nhật ngắn sau khi notebook chốt | Cập nhật môi trường, cách lấy CSV, mô hình cuối và kết quả thực tế nếu tiếp tục chia sẻ repo | Không để hướng dẫn mâu thuẫn với bản code nộp; không phải điều kiện riêng để nộp notebook |
| `TEAM_ASSIGNMENTS.md` | **NÊN — vừa:** bổ sung trách nhiệm mô hình hiện đại và kiểm tra Run All | Cập nhật khi chốt việc nhóm | Tránh phần bổ sung không có người phụ trách; không ảnh hưởng thuật toán |
| `.gitignore` | Không phải sửa để đạt đề | Chỉ sửa khi cần chia sẻ file đang bị ignore hoặc theo dõi một artifact cụ thể | Bảo đảm bàn giao đủ file; không thay đổi kết quả mô hình |

**Không có một file Python ngoài notebook nào mà chỉ việc chưa sửa nó đã làm nhóm tự động sai đề.** Điều cần bảo vệ là dataset đầu vào đúng và toàn bộ code được trình bày trong notebook chạy, đánh giá đúng.

### 2.3. Nếu phát hiện dữ liệu bị ảnh hưởng, phải chạy lại đến đâu?

| Loại thay đổi | Cần chạy lại |
|---|---|
| Chỉ sửa README, Markdown hoặc tên hiển thị chỉ số; không đổi tính toán | Phần hiển thị liên quan; không cần train lại |
| Sửa script một lần nhưng CSV dùng train không đổi | Không cần train lại chỉ vì script thay đổi |
| Khôi phục nhãn export, thêm/bớt mẫu, sửa nhãn hoặc văn bản đầu vào | Chốt lại dữ liệu/split rồi chạy lại các mô hình để so sánh trên cùng tập |
| Sửa che tên/emoji/tiền xử lý | Chạy lại các nhánh bị đổi đầu vào; cập nhật mọi bảng so sánh liên quan |
| Chỉ sửa CV under-sampling | Chạy lại nhánh đó và phần tổng hợp/chọn mô hình nếu liên quan; không cần crawl/gán nhãn lại |
| Bổ sung mô hình hiện đại trên dataset/split đã chốt | Chạy nhánh mới; cập nhật bảng/đồ thị chung. Nếu thay đổi dataset/split/tiền xử lý nhánh cũ thì cũng chạy lại nhánh cũ |

Do kế hoạch hiện có cả sửa tiền xử lý và thêm mô hình mới, vẫn cần một lần chạy notebook đầy đủ ở cấu hình cuối. Chỉ cần tái tạo dữ liệu thô/nhãn nếu kiểm tra dữ liệu xác nhận phải làm.

## 3. Ý tưởng cho ý 4 bổ sung: mô hình hiện đại

### 3.1. Phương án chính: fine-tune PhoBERT cho Spam/Ham

**Mức:** ĐỀ — rất cao đối với việc phải bổ sung mô hình hiện đại. **PhoBERT là lựa chọn đề xuất, không phải tên mô hình giảng viên bắt buộc.**

Sơ đồ:

```text
Bình luận PTIT + nhãn spam/ham đã có
  → chuẩn hóa nhẹ và tách từ phù hợp PhoBERT
  → tokenizer PhoBERT
  → encoder PhoBERT tiền huấn luyện
  → tầng phân loại hai nhãn
  → fine-tune trên train
  → chọn checkpoint theo F1 spam trên validation
  → đo các chỉ số trên test đã chốt
  → so sánh với BoW/TF-IDF/FastText + NB/SVM
```

**Vì sao phù hợp:** mô hình ngôn ngữ tiếng Việt có biểu diễn theo ngữ cảnh, bổ sung một hướng khác với vector từ trung bình và đặc trưng tần suất. Có thể kiểm tra nó có giảm lỗi ở các bình luận chào mời ngắn hay không; đây là giả thuyết cần đo, chưa phải kết quả đã biết.

Checkpoint đề xuất là `vinai/phobert-base-v2`; cần khóa revision khi thực thi. [Model card VinAI](https://huggingface.co/vinai/phobert-base-v2) nêu đầu vào cần tách từ và khuyến nghị RDRSegmenter/VnCoreNLP. Vì vậy không truyền raw text vào tokenizer rồi coi như đã làm đúng toàn bộ, cũng không tự giả định tách từ underthesea hiện có tương đương cách được khuyến nghị.

### 3.2. Cách triển khai vừa đủ cho bài

| Hạng mục | Làm gì? | Tại sao/tác dụng |
|---|---|---|
| Chia dữ liệu | Dùng cùng outer train/test với phần cổ điển; lấy validation theo nhóm bên trong train | Kết quả test so sánh được; tránh dùng test chọn checkpoint |
| Đầu vào | Dùng cùng tập bình luận/nhãn đã chốt; chuẩn hóa nhẹ, giữ ngữ cảnh; không mặc định bỏ stop words hoặc chuyển chữ thường theo nhánh cổ điển | Phù hợp cách dùng encoder; không làm mất thông tin mà mô hình có thể tận dụng |
| Tokenization | Tách từ theo phương án được ghi rõ; tokenizer của checkpoint; padding/truncation theo cấu hình | Tương thích mô hình và chạy được theo batch |
| Mô hình | Dùng kiến trúc phân loại hai nhãn, ví dụ `AutoModelForSequenceClassification(num_labels=2)`, rồi fine-tune | Thực sự học bài toán Spam/Ham; chỉ tải model hoặc chạy fill-mask chưa giải quyết ý bổ sung |
| Cấu hình ban đầu | Điểm xuất phát có thể là learning rate `2e-5`, tối đa 3 epoch, batch 8 hoặc 16 tùy VRAM; độ dài 128/256 tùy thống kê và giới hạn checkpoint | Có phương án thực thi gọn. Đây là cấu hình dự kiến, chưa được kiểm chứng là tối ưu hoặc đủ tài nguyên |
| Lựa chọn | Chọn checkpoint theo F1 spam validation; nếu thử cấu hình khác cũng dựa vào train/validation | Không tối ưu trên test. Không cần GridSearchCV 5-fold cho Transformer để chứng minh đủ đề |
| Mất cân bằng | Bắt đầu với thiết lập rõ ràng; thử weighted loss nếu cần và validation cho thấy có ích | Weighted loss là tùy chọn, không phải yêu cầu bắt buộc cho ý bổ sung |
| Đánh giá | Accuracy, Precision, Recall, F1 spam và Confusion Matrix; AP/thời gian là phần nên có | Đáp ứng bộ chỉ số của đề và giải thích đánh đổi |
| Tài nguyên | Chạy GPU trên nền tảng có quyền truy cập; ghi thiết bị, thời gian, cấu hình thực tế | Fine-tune khả thi hơn; chưa có GPU được xác nhận trong lần lập kế hoạch này |
| Hoàn tất | Lưu output thật trong notebook; kiểm tra lại phần dự đoán và bảng chung | Người chấm theo dõi được mã và kết quả của nhánh bổ sung |

Nếu kiểm tra tiền xử lý/tập dữ liệu dẫn đến đổi split, mọi mô hình so sánh phải theo split đã chốt mới; không ghép điểm cũ và điểm mới thành cùng một bảng so sánh như thể chúng cùng điều kiện.

Một mô hình hiện đại **không bắt buộc phải thắng** SVM. Nếu thấp hơn thì vẫn trình bày trung thực cách xây dựng, đánh giá và hạn chế. Nếu chưa chạy nhánh này, việc thêm code hoặc để `RUN_MODERN=False` chưa đủ để tuyên bố đã hoàn thành thí nghiệm.

### 3.3. Phương án thay thế phù hợp miền dữ liệu: ViSoBERT

Nếu muốn ưu tiên mô hình được thiết kế cho văn bản mạng xã hội tiếng Việt, có thể chọn `uitnlp/visobert` thay cho PhoBERT. [Bài báo gốc](https://aclanthology.org/2023.emnlp-main.315/) mô tả tiền huấn luyện trên văn bản mạng xã hội và có đánh giá spam reviews; [model card](https://huggingface.co/uitnlp/visobert) hướng dẫn tokenizer và thư viện.

Đó là lý do hợp lý để thử trên bình luận PTIT, không chứng minh sẵn rằng mô hình sẽ tốt hơn PhoBERT hoặc SVM trên dataset này. Quy trình fine-tune/validation/test vẫn như trên, nhưng tiền xử lý theo chính model card và tokenizer được chọn, không sao chép nguyên bước tách từ PhoBERT.

**Chỉ chọn một mô hình cho lần hoàn thiện chính.** Giữ PhoBERT là phương án chính để bám kế hoạch trước; ViSoBERT là thay thế có căn cứ nếu muốn tập trung miền mạng xã hội. Không cần chạy cả hai để đáp ứng một ý bổ sung.

## 4. Thứ tự thực hiện để tránh làm lại nhiều lần

1. Kiểm tra CSV cuối và thông tin nguồn. Chỉ sửa/chạy lại khâu ngoài notebook khi xác nhận dữ liệu bị ảnh hưởng.
2. Chốt nguồn dữ liệu, môi trường, cách tải và các phần mã cần nhúng để notebook có thể tách khỏi repo.
3. Sửa tiền xử lý, nhóm dữ liệu/CV, lựa chọn mô hình và hàm dự đoán; chốt giữ/bỏ các phần nâng cao.
4. Bổ sung một nhánh hiện đại, chạy thử cấu trúc và cấu hình tài nguyên; số thử nhanh không dùng thay cho kết quả đầy đủ.
5. Chạy đầy đủ trên dữ liệu/split đã chốt; lưu kết quả thật; sửa nhận xét và bảng tổng hợp theo output.
6. Xác nhận notebook Run All từ kernel mới, không cần chuẩn bị sẵn `spam_features.py` hay chạy lại Argilla.
7. Chốt notebook nộp chính. Sau đó mới dùng code/kết quả đã chốt để viết báo cáo học phần.

## 5. Điều kiện để gọi là đã chốt phần code

- [ ] Notebook giải thích đúng nguồn dữ liệu và quá trình thực hiện, không tự thêm thông tin chưa xác nhận.
- [ ] CSV cuối đã được kiểm tra; không còn nghi vấn dữ liệu quan trọng chưa xử lý.
- [ ] Có đủ tiền xử lý, ít nhất hai đặc trưng, hai thuật toán và train/test hợp lệ.
- [ ] Có một mô hình hiện đại đã được huấn luyện/đánh giá thật, không chỉ có code dự kiến.
- [ ] Đủ Accuracy, Precision, Recall, F1, Confusion Matrix và bảng so sánh cùng điều kiện.
- [ ] Các phần nâng cao được giữ lại đều tính/diễn giải đúng; không coi nâng cao là yêu cầu bắt buộc của đề.
- [ ] Cách chọn mô hình không dựa vào test; kết luận nêu rõ phạm vi so sánh.
- [ ] Demo và lưu mô hình, nếu giữ, dùng đúng tiền xử lý và đúng mô hình.
- [ ] Cách cung cấp dữ liệu/thư viện/tài nguyên rõ ràng; không chứa token riêng trong notebook.
- [ ] Restart kernel + Run All thành công ở cấu hình cuối, đã lưu output để nộp.

Tại thời điểm lập kế hoạch, chưa chạy lại dataset thật hoặc thí nghiệm hiện đại; chưa kết luận được CSV v7 có bị các lỗi script một lần ảnh hưởng hay không. Vì vậy phần sửa file ngoài được ghi có điều kiện thay vì coi là bắt buộc toàn bộ.
