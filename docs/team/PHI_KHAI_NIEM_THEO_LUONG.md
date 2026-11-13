# Phi học khái niệm theo luồng: từ dữ liệu đã chuẩn bị đến dự đoán

Tài liệu này chỉ nói về **phần Phi phụ trách**. Điểm bắt đầu là một bảng bình luận đã có nhãn **ham/spam**, văn bản đã làm sạch và hai cách biểu diễn văn bản đã chuẩn bị sẵn: **giữ từ dừng** và **bỏ từ dừng**. Phi không cần thu thập, gán nhãn hay làm sạch lại bảng này. Từ đây, Phi tự chia tập, biến văn bản thành số, huấn luyện, chọn mô hình, đánh giá và đưa mô hình vào sử dụng.

Luồng cần nhớ: **dữ liệu đã chuẩn bị → chia train/test → chia train thành các lượt kiểm định → học đặc trưng và mô hình → khảo sát lệch lớp → chọn bằng kiểm định → đánh giá test → lưu mô hình → dự đoán bình luận mới**.

## 1. Hiểu đầu vào và xác định bài toán

**Phân loại nhị phân** là việc chọn một trong hai nhãn cho mỗi bình luận. Ở đây, **ham** là bình luận thông thường, **spam** là bình luận rác theo quy ước gán nhãn của nhóm. Trong phép đánh giá, spam được xem là **lớp dương**: phát hiện đúng spam là *true positive*; bỏ sót spam là *false negative*.

Mỗi dòng đầu vào có một mã nhận diện, văn bản và nhãn đã biết. Văn bản dùng để học đã được tách thành các đơn vị từ (*token*). Nhóm đã chuẩn bị hai phiên bản: một phiên bản giữ các từ thường gặp như “là”, “và”, “của”, và một phiên bản bỏ chúng. Phi chưa biết phiên bản nào tốt hơn, nên sẽ so bằng dữ liệu huấn luyện ở các bước sau.

**Đầu ra của bước này:** xác định rõ mỗi bình luận là một mẫu, nhãn cần dự đoán là ham hoặc spam, và hai phiên bản văn bản là hai lựa chọn để thử. Không được dùng nhãn của một bình luận mới khi dự đoán bình luận đó.

## 2. Chia dữ liệu: train, validation và test

### 2.1. Vì sao phải chia trước khi học?

Nếu học và chấm điểm trên cùng những bình luận, điểm thường quá lạc quan vì mô hình đã gặp nội dung ấy. Vì vậy, trước khi học từ vựng hay trọng số, Phi dành riêng một **tập test**. Tập còn lại gọi là **train**. Test chỉ được mở để đánh giá sau khi đã chọn xong cách làm bằng train.

Tập **validation** không phải một tập thứ ba cố định. Trong bài này, validation lần lượt là một phần của train ở mỗi lượt kiểm định chéo. Cách chia hiện tại tạo khoảng **30.759 bình luận train** và **7.474 bình luận test**. Tỷ lệ gần 80/20, không bắt buộc đúng 80/20 từng dòng vì các bình luận cùng nhóm nội dung phải đi cùng nhau. Cách chia này cũng **không ép tỷ lệ ham/spam bằng nhau** giữa các tập; sau khi chia cần kiểm tra cả hai tập đều có đủ hai nhãn.

### 2.2. Giữ bình luận giống nhau ở cùng phía

Một câu có thể xuất hiện nhiều lần với khác biệt nhỏ về chữ hoa, khoảng trắng, số điện thoại hoặc dấu câu. Nếu một bản ở train và bản gần giống ở test, mô hình có thể nhớ câu đó thay vì thực sự nhận ra spam. Đó là một dạng **rò rỉ dữ liệu**.

Quy trình chia theo nội dung diễn ra bằng lời như sau:

1. Lấy văn bản đã làm sạch của mỗi bình luận để tạo **khóa nhóm**. Chuẩn hóa cách viết Unicode, chuyển thành chữ thường, gom khoảng trắng, thay các dãy chữ số bằng cùng một ký hiệu và bỏ dấu câu trong khóa. Khóa này **chỉ dùng để chia tập**, không phải văn bản đưa vào mô hình.
2. Hai bình luận cho ra cùng khóa được xem là một nhóm khi chia.
3. Biến mỗi khóa thành một giá trị băm ổn định rồi xếp vào một trong năm nhóm. Một nhóm làm test, bốn nhóm làm train. Cùng khóa luôn rơi vào cùng nhóm; chạy lại với cùng dữ liệu vẫn cho cùng cách chia.
4. Kiểm tra không có khóa nào đồng thời xuất hiện ở train và test.

Điều này xử lý các câu trùng theo **quy tắc tạo khóa**, nhưng không bảo đảm phát hiện mọi câu đồng nghĩa, mọi bài đăng liên quan hay cùng tác giả. Giá trị băm chỉ giúp chia ổn định; nó không phải đặc trưng để mô hình học.

### 2.3. Kiểm định chéo năm lượt là gì?

Chỉ **train** được chia tiếp thành năm phần. Mỗi lượt lấy bốn phần để học và một phần để kiểm tra. Lặp năm lượt để mỗi phần được làm validation đúng một lần. Các bình luận cùng khóa vẫn nằm trong cùng một phần. Mỗi phía của từng lượt phải có cả ham và spam.

Ví dụ bằng lời: ở lượt thứ nhất, học trên phần 2–5 và chấm trên phần 1; ở lượt thứ hai, học trên phần 1, 3–5 và chấm trên phần 2; tiếp tục đến lượt thứ năm. Điểm **CV** (*cross-validation*, kiểm định chéo) là trung bình điểm validation của năm lượt, **không phải điểm test**.

**Đầu ra của giai đoạn chia:** hai tập train/test và năm cặp train–validation nằm hoàn toàn bên trong train. Từ đây cho đến khi chốt mô hình, Phi chỉ dùng train và các lượt validation.

## 3. Biến bình luận thành vector số

Máy học truyền thống không nhận trực tiếp một câu chữ. Cần biến mỗi bình luận thành một **vector**: một dãy số có các vị trí tương ứng với từ hoặc cụm từ. Tập các vị trí được gọi là **từ vựng**. Với nhiều từ nhưng một câu chỉ chứa vài từ, phần lớn vị trí bằng 0; đó là **ma trận thưa**.

### 3.1. Từ đơn, cặp từ và từ vựng

**Unigram** là một token riêng lẻ. **Bigram** là hai token liên tiếp, chẳng hạn “lịch thi”. Bài này thử hai lựa chọn: chỉ unigram; hoặc unigram cùng bigram. Một từ/cặp từ chỉ được đưa vào từ vựng nếu xuất hiện trong ít nhất **hai bình luận thuộc phần đang học**. Quy tắc này giảm các đặc trưng quá hiếm.

Điểm mấu chốt là **học từ vựng bên trong từng lượt**: chỉ nhìn bốn phần dùng để học, rồi dùng đúng từ vựng đó để biến phần validation thành số. Test cũng chỉ được biến đổi bằng từ vựng đã học từ train. Nếu tạo từ vựng trước trên toàn bộ dữ liệu, validation/test đã góp thông tin vào quá trình học, gây rò rỉ.

### 3.2. Bag of Words: đếm từ

**Bag of Words (BoW)** biểu diễn câu bằng số lần xuất hiện của từng từ/cặp từ trong từ vựng. Ví dụ, với từ vựng gồm “lịch”, “thi”, “mua”, câu “lịch thi lịch” có giá trị tương ứng là 2, 1, 0. BoW biết một từ xuất hiện bao nhiêu lần nhưng không hiểu nghĩa sâu của câu. Bigram bổ sung chút thông tin về thứ tự hai từ cạnh nhau.

Các bước: học từ vựng trên phần train của lượt hiện tại → đếm mỗi đặc trưng trong từng bình luận → tạo ma trận số cho mô hình. Khi gặp bình luận mới, chỉ đếm theo **từ vựng đã học**, không tự thêm cột mới.

### 3.3. TF-IDF: giảm ảnh hưởng của từ quá phổ biến

**TF-IDF** cũng tạo một cột cho mỗi từ/cặp từ, nhưng dùng **trọng số** thay cho số đếm thô. Nó kết hợp hai ý:

- **TF** (*term frequency*): đặc trưng xuất hiện trong bình luận này bao nhiêu lần. Bài này giảm tác động của việc lặp quá nhiều bằng quy tắc **1 + log(số lần xuất hiện)** khi số lần xuất hiện lớn hơn 0.
- **IDF** (*inverse document frequency*): đặc trưng hiếm trong tập đang học được trọng số cao hơn đặc trưng gặp trong hầu hết bình luận. Với cách làm trơn đang dùng, IDF bằng **1 + log[(1 + N) / (1 + số bình luận chứa đặc trưng)]**, trong đó N là số bình luận của phần dùng để học.

Nhân TF với IDF rồi chuẩn hóa độ dài vector. Nhờ đó, một từ lặp nhiều không lấn át toàn bộ câu, và từ rất phổ biến có ít sức phân biệt hơn. **IDF cũng phải học chỉ từ phần train của từng lượt**, giống từ vựng.

**Đầu ra của giai đoạn biểu diễn:** mỗi bình luận đã trở thành vector BoW hoặc TF-IDF. Đây là đầu vào cho bộ phân loại; vector chưa phải nhãn dự đoán.

## 4. Học hai bộ phân loại từ vector và nhãn

### 4.1. Naive Bayes đa thức

**Naive Bayes (NB)** học từ các bình luận đã có nhãn để ước lượng mức liên quan giữa đặc trưng và từng lớp. “Naive” nói đến giả định đơn giản hóa: khi đã biết lớp, các đặc trưng được xử lý như gần độc lập. Giả định này không hoàn toàn đúng với ngôn ngữ, nhưng làm mô hình nhanh và tạo mốc so sánh hữu ích.

Các bước tự nhiên: ước lượng **mức phổ biến ban đầu của mỗi lớp** từ số ham/spam trong phần train → đếm mức xuất hiện của đặc trưng theo từng lớp → cộng một lượng **làm trơn** để từ chưa thấy ở một lớp không khiến phép tính về 0 → với câu mới, cộng điểm ban đầu của lớp và đóng góp của các đặc trưng → chọn lớp có điểm cao hơn. Mức làm trơn có tên **alpha**; bài thử 0,1; 0,3; và 1,0. BoW và TF-IDF đều không âm nên dùng được trong thí nghiệm này, dù TF-IDF là trọng số chứ không phải số lần đếm nguyên.

### 4.2. Máy vectơ hỗ trợ tuyến tính

**SVM tuyến tính** học một ranh giới tách vector ham và spam. Có thể hình dung mỗi đặc trưng có một trọng số; cộng các trọng số của bình luận rồi so với một mốc để quyết định lớp. Mô hình cố tìm ranh giới có **khoảng cách an toàn** tốt, đồng thời chịu một mức phạt cho các điểm bị phân loại sai hoặc nằm quá gần ranh giới.

Mức phạt được điều khiển bởi **C**: C lớn thường ép mô hình sửa lỗi trên train mạnh hơn; C nhỏ cho phép nhiều sai sót hơn để ranh giới bớt phụ thuộc dữ liệu train. Bài thử C bằng 0,1; 0,3; 1,0; 3,0 rồi chọn bằng validation, không đoán trước giá trị nào tốt nhất.

SVM cho một **điểm quyết định có dấu**. Theo quy tắc mặc định của bài, điểm lớn hơn 0 được gán spam, điểm không vượt 0 được gán ham. Điểm này **không phải xác suất**; giá trị âm không có nghĩa là “xác suất âm”.

### 4.3. Thử cấu hình và chọn bằng CV

Hai cách biểu diễn kết hợp với hai bộ phân loại thành bốn cặp:

| Mã | Biểu diễn | Bộ phân loại |
|---|---|---|
| B1 | BoW | Naive Bayes |
| B2 | BoW | SVM tuyến tính |
| B3 | TF-IDF | Naive Bayes |
| B4 | TF-IDF | SVM tuyến tính |

Mỗi cặp còn được thử trên **hai phiên bản giữ/bỏ từ dừng**, thành tám nhánh khảo sát. Trong từng nhánh, thử các lựa chọn unigram/bigram và alpha hoặc C. Những lựa chọn đặt trước khi học như alpha, C và độ dài cụm từ là **siêu tham số**; từ vựng, IDF và trọng số phân loại là những thứ **được học** từ dữ liệu. Với mỗi tổ hợp siêu tham số, học và chấm qua năm lượt validation, rồi lấy **trung bình F1 của lớp spam**. Tổ hợp có CV F1 cao nhất được chọn và **học lại trên toàn bộ train**. Sau đó, so tám nhánh để chọn phiên bản từ dừng; dữ liệu hiện tại chọn **giữ từ dừng**. B1–B4 là bốn cặp thuộc phiên bản được chọn, và **B4 dẫn đầu** ở bước này.

**Vì sao chọn F1 spam?** Spam ít hơn ham. Một mô hình luôn nói “ham” vẫn có thể đạt Accuracy cao nhưng không phát hiện được spam. F1 kết hợp **Precision** (khi nói spam thì đúng bao nhiêu) và **Recall** (trong spam thật thì tìm được bao nhiêu). Việc chọn theo CV diễn ra trước khi xem kết quả test.

## 5. Khảo sát mất cân bằng lớp trên cấu hình B4

Trong toàn bộ dữ liệu có **32.907 ham** và **5.326 spam**. Ở train hiện tại có **26.384 ham** và **4.375 spam**. Lớp ham nhiều hơn nên cần kiểm tra xem mô hình có thiên về ham và bỏ sót spam không. Phi giữ B4 làm **mốc không điều chỉnh**, rồi thử ba thay đổi **riêng biệt** trên cùng cặp TF-IDF + SVM; không cộng cả ba thay đổi vào một mô hình.

### 5.1. B4: mốc không điều chỉnh

B4 học trên toàn bộ train, không đặt trọng số lớp đặc biệt, không giảm mẫu và dùng ngưỡng SVM mặc định là 0. Nó là đối chứng để biết mỗi thay đổi sau đây tác động ra sao.

### 5.2. S1: tăng trọng số lỗi của lớp ít mẫu

**Trọng số cân bằng** không tạo thêm hay xóa bình luận. Khi học, lỗi ở lớp ít mẫu được tính nặng hơn. Quy tắc trọng số cho một lớp là: **tổng số mẫu ÷ (số lớp × số mẫu của lớp đó)**. Với train hiện tại, trọng số ham xấp xỉ **0,583**, còn spam xấp xỉ **3,515**. Vì vậy, lỗi trên spam có sức nặng lớn hơn lỗi trên ham trong quá trình tìm ranh giới. Validation và test vẫn giữ nguyên phân bố ban đầu.

### 5.3. S2: giảm bớt ham trong phần dùng để học

**Under-sampling 2:1** giữ tất cả spam, chọn ngẫu nhiên nhưng tái lập được tối đa hai ham cho mỗi spam. Nếu làm trên toàn train ở lượt học cuối, số mẫu dùng để học là **4.375 spam + 8.750 ham = 13.125**. Trong mỗi lượt CV, việc chọn ham chỉ diễn ra ở **bốn phần dùng để học**; phần validation vẫn giữ tỷ lệ thật. Nếu giảm mẫu trước rồi mới chia, validation sẽ bị thay đổi và phép đo thiếu trung thực.

Cách này giúp spam bớt bị số đông ham lấn át, nhưng cũng bỏ đi nhiều ví dụ ham có thể hữu ích. Vì thế phải so điểm thay vì mặc định coi giảm mẫu là tốt hơn.

### 5.4. S3: đổi ngưỡng ra quyết định của SVM

S3 **không đổi số mẫu và không đặt trọng số lớp**. Nó giữ cặp đặc trưng–bộ phân loại và tham số đã chọn cho B4, rồi tìm ngưỡng score phù hợp hơn mốc 0. Quy trình chọn ngưỡng:

1. Qua năm lượt, mỗi bình luận train được một mô hình **chưa từng học trên chính bình luận đó** cho điểm. Các điểm này gọi là điểm **ngoài fold** (*out-of-fold*, OOF).
2. Thử nhiều ngưỡng dựa trên phân bố điểm OOF, gồm mốc 0, các mức từ thấp đến cao và hai mức biên.
3. Với mỗi ngưỡng, đổi điểm thành nhãn, tính Precision/Recall/F1 spam trên từng phần validation rồi lấy trung bình. Chọn ngưỡng có F1 trung bình lớn nhất; nếu bằng nhau, ưu tiên ngưỡng gần 0 hơn.
4. Học lại mô hình trên toàn bộ train và giữ ngưỡng vừa chọn cho lúc dự đoán.

Ngưỡng hiện tại khoảng **−0,1776**. Hạ ngưỡng xuống dưới 0 nghĩa là một số bình luận có điểm hơi âm nay được gọi là spam. Điều này có thể tăng Recall nhưng cũng có thể tăng báo động nhầm. Ngưỡng âm hoàn toàn hợp lệ vì đây là **điểm quyết định**, không phải xác suất.

**Giới hạn cần nhớ:** S3 dùng chính các điểm OOF để **tìm** ngưỡng rồi **báo** điểm CV cho ngưỡng ấy. Điểm CV đó có thể hơi lạc quan; đây không phải một lớp kiểm định chéo lồng bên ngoài hoàn toàn độc lập. Test vẫn được giữ riêng để đánh giá sau khi chốt lựa chọn.

### 5.5. Chọn cấu hình nào?

So CV F1 spam của **B1–B4 và S1–S3**; ứng viên cao nhất được chốt **trước khi chấm test**. Lượt hiện tại chọn **S3** với CV F1 khoảng **0,7876**. “S3 tốt nhất” chỉ có nghĩa là tốt nhất **trong bảy ứng viên đã thử**, không khẳng định mọi cách xử lý khác đều kém hơn. Chỉ khảo sát S1–S3 trên B4 giúp so tác động từng thay đổi trên cùng một mốc; bài này chưa thử mọi chiến lược cho mọi cặp mô hình.

## 6. Đánh giá trên tập test đã giữ riêng

Sau khi chốt S3 theo CV, Phi dùng tập test để **báo cáo hiệu năng**, không quay lại đổi mô hình theo điểm test. Có thể chấm cả bảy ứng viên trên cùng test để nhìn toàn cảnh, nhưng quyết định chọn vẫn phải dựa vào CV đã làm trước đó.

### 6.1. Ma trận nhầm lẫn và bốn loại kết quả

Với spam là lớp dương:

| Loại | Ý nghĩa | S3 trên test hiện tại |
|---|---|---:|
| Đúng ham (*TN*) | Ham thật, dự đoán ham | 6.396 |
| Báo nhầm spam (*FP*) | Ham thật, dự đoán spam | 127 |
| Bỏ sót spam (*FN*) | Spam thật, dự đoán ham | 297 |
| Đúng spam (*TP*) | Spam thật, dự đoán spam | 654 |

Từ bốn số này có thể tự tính các chỉ số:

- **Accuracy** = (TP + TN) / tổng số bình luận. Nó tính đúng chung nhưng dễ gây hiểu lầm khi ham chiếm đa số.
- **Precision của spam** = TP / (TP + FP): trong các bình luận bị báo spam, bao nhiêu là spam thật. S3 khoảng **0,8374**.
- **Recall của spam** = TP / (TP + FN): trong spam thật, tìm được bao nhiêu. S3 khoảng **0,6877**.
- **F1 của spam** = 2 × Precision × Recall / (Precision + Recall): cân bằng hai mục tiêu trên. S3 khoảng **0,7552**.

Một mô hình luôn đoán ham có thể đạt Accuracy khoảng 87% vì ham nhiều, nhưng **Recall và F1 spam bằng 0**. Vì vậy, khi trình bày phải nhìn các chỉ số của lớp spam cùng ma trận nhầm lẫn.

### 6.2. Đường Precision–Recall và AP

Nếu thay ngưỡng từ rất cao xuống rất thấp, số bình luận được gọi là spam sẽ thay đổi. Ở mỗi ngưỡng có một cặp Precision–Recall; nối các cặp tạo thành **đường Precision–Recall**. **AP** (*Average Precision*) tóm tắt chất lượng xếp hạng các bình luận theo score trên toàn dải Recall. Đổi **chỉ** ngưỡng của cùng một mô hình có thể đổi nhãn, Precision, Recall và F1 tại ngưỡng đang dùng, nhưng không đổi thứ tự score nên AP thường không đổi. AP không phải Accuracy và không phải F1.

### 6.3. Đọc kết quả một cách thận trọng

Trên test, F1 spam của B4 khoảng **0,7334**, S1 **0,7451**, S2 **0,7542**, S3 **0,7552**. S3 chỉ hơn S2 khoảng **0,0010** ở test; không nên nói S3 áp đảo. S3 được chọn vì **CV F1 cao nhất**, còn test cho thấy lựa chọn đó vẫn hoạt động tốt trên phần dữ liệu giữ riêng. Đọc các ví dụ FP/FN giúp nhận ra kiểu bình luận dễ báo nhầm hoặc bị bỏ sót, nhưng một ví dụ riêng lẻ không chứng minh nguyên nhân chắc chắn.

Nếu ước lượng độ dao động của F1 bằng **bootstrap**, bài này rút **500** tập có hoàn lại từ test, tính lại F1 mỗi lần rồi lấy hai phân vị **2,5% và 97,5%** làm khoảng tham khảo 95%. Một dòng có thể được rút nhiều lần, dòng khác có thể không xuất hiện trong một lượt. Khoảng này cho biết độ nhạy của điểm đo với việc lấy mẫu lại trên test hiện có; nó không chứng minh mô hình sẽ giữ nguyên hiệu năng trên mọi nhóm Facebook hoặc thời điểm khác.

## 7. Lưu mô hình và dự đoán bình luận mới

Sau lượt học cuối, mô hình cần giữ được **từ vựng đã học, trọng số IDF nếu dùng TF-IDF, trọng số của bộ phân loại, ngưỡng phân loại và lựa chọn giữ/bỏ từ dừng**. Đó là những thứ cần để dự đoán lại mà không huấn luyện từ đầu. Lưu xong phải nạp lại và kiểm một số bình luận đã biết: trước và sau lưu cần cho cùng nhãn.

Đường đi của **một bình luận mới** là: nhận văn bản thô → áp đúng quy tắc chuẩn bị văn bản và lựa chọn từ dừng đã chốt → biến thành vector theo **từ vựng/IDF đã học**, không học lại → SVM tính điểm quyết định → so điểm với ngưỡng S3 → trả ham hoặc spam. Nếu câu mới có nhiều từ chưa từng gặp, vector có thể thiếu thông tin; mô hình có thể sai. Một ví dụ demo chỉ chứng minh quá trình dự đoán hoạt động, không chứng minh độ đúng chung.

**Kết quả cuối của phần Phi:** một quy trình phân loại truyền thống đã được chọn bằng CV, có chỉ số test và lỗi được giải thích, cùng một mô hình đã học có thể nạp lại để dự đoán bình luận mới. Mọi kết luận về chất lượng phải gắn với tập dữ liệu, cách chia và bảy ứng viên đã khảo sát trong bài.
