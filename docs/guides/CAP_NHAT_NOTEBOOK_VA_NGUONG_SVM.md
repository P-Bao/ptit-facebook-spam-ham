# Cập nhật thực nghiệm và ngưỡng SVM

Tài liệu ghi lại đợt hoàn thiện SVM trước bước viBERT. Nhánh viBERT được mô tả
riêng trong [hướng dẫn Colab](HUONG_DAN_VIBERT_COLAB.md) và Notebook 03.

## Phạm vi thay đổi

Chỉnh Notebook 02 và các hàm huấn luyện/dự đoán liên quan. Giữ nguyên Notebook
01, dataset v7, hàm tiền xử lý và quy tắc chia nhóm train/test/CV. Phần mô hình
hiện đại vẫn dành cho bước tiếp theo, chưa triển khai trong lần cập nhật này.

## Trước và sau

| Nội dung | Trước | Sau | Mục đích |
|---|---|---|---|
| Thiết kế thực nghiệm | Có train/test/CV nhưng giải thích ngắn, trọng số đặt sớm. | Mục 4 định nghĩa train, validation, CV, test; trọng số chuyển sang 5.1. | Phân biệt lựa chọn cấu hình và đánh giá cuối. |
| Từ dừng | In kết quả khi huấn luyện; bảng đối chiếu ở phụ lục. | Bảng CV ở mục 5, chọn tiền xử lý rồi trình bày sáu tổ hợp; bỏ phụ lục. | Luồng lựa chọn liền mạch, không lặp phần cuối. |
| Sáu tổ hợp cơ sở | SVM tìm cả None/balanced; tên “chính” không thể hiện cách xử lý. | SVM cơ sở không trọng số, không lấy mẫu lại; tên “Không điều chỉnh”, mã B1–B6. | Có đối chứng rõ ràng trước khi khảo sát chiến lược. |
| Khảo sát mất cân bằng | SVM dùng trọng số/lấy mẫu, NB chỉnh ngưỡng; bảng năm nhánh. | Cùng TF-IDF + SVM: không điều chỉnh, balanced, lấy mẫu 2:1, chỉnh ngưỡng. | Cố định biểu diễn và bộ phân loại để dễ đối chiếu. |
| Chỉnh ngưỡng | Chỉ NB sử dụng ngưỡng tùy chọn; SVM gọi predict mặc định. | SVM chọn ngưỡng decision score bằng OOF trên train; cùng ngưỡng trong đánh giá và demo. | Ngưỡng được thực thi nhất quán, không chỉ hiển thị trong bảng. |
| Đánh giá | Có bảng rộng và tên cấu hình lặp; hình tập trung sáu tổ hợp. | Tách sáu tổ hợp và bốn chiến lược, có ma trận và PR cho hai phạm vi. | So sánh theo câu hỏi thực nghiệm, giảm bảng quá nhiều cột. |
| Phân tích lỗi | Bảng lỗi dùng ứng viên cuối nhưng đặc trưng lấy từ SVM cơ sở. | Ghi rõ ứng viên được chọn; cùng mô hình cho đặc trưng, lỗi và demo; giữ ID/text gốc. | Đối chiếu được từng dự đoán với đúng cấu hình. |
| Khảo sát OOV | Có mục riêng sau phân tích lỗi. | Bỏ mục riêng, giữ FastText trong sáu tổ hợp và giải thích subword ngắn. | Tập trung vào phân loại và kết quả liên quan. |

## Mã nguồn thay đổi

- `notebooks/02_thuc_nghiem_va_demo.ipynb`: điều phối thực nghiệm, bảng CV/test,
  hình, phân tích lỗi, lưu mô hình và nhận xét.
- `pipeline/modeling/experiments.py`: cấu hình trọng số tường minh, tìm tham số
  với CV Precision/Recall/F1 và chọn theo F1; ghi đúng ngưỡng mặc định cho NB/SVM.
- `pipeline/modeling/training_utils.py`: under-sampling trả đủ chỉ số CV;
  `tune_svm_threshold()` tạo score OOF và chọn ngưỡng; `predict_record()` áp dụng
  ngưỡng SVM. Hàm ngưỡng NB không còn được sử dụng nên được thay thế.
- `pipeline/inference/predict.py`: chuyển ngưỡng bundle cho hàm dự đoán, dùng
  mặc định theo loại mô hình khi bundle không khai báo ngưỡng.
- Hai file kiểm thử về huấn luyện/module: kiểm OOF không fit lên validation,
  ngưỡng SVM, cấu hình cơ sở và ngưỡng được giữ khi lưu–nạp–dự đoán.
- `README.md`, hướng dẫn tổng quan và lộ trình trình bày: cập nhật luồng và đầu ra.

Không tạo mô-đun Python mới cho chức năng huấn luyện. Sáu tổ hợp cơ sở và ba
cấu hình bổ sung tương ứng chín ứng viên ở cấu hình từ dừng được chọn. Cấu hình
TF-IDF + SVM cơ sở được dùng lại trong bảng bốn chiến lược, không tính thành một
ứng viên mới. Các lần fit tìm tham số/fold không phải các thuật toán mới.

## Cách đọc kết quả

Mục 5 và 5.1 báo cáo điểm CV dùng để lựa chọn. Mục 6 dùng test để báo cáo hiệu
quả của các cấu hình đã xác định. CV có thể lạc quan do tìm cấu hình và ngưỡng;
không diễn giải như nested CV hoặc bảo đảm kết quả trên dữ liệu mới.

Nhánh ngưỡng giữ siêu tham số của SVM không điều chỉnh. Ngưỡng được khảo sát từ
199 phân vị 1–99% của score OOF, ngưỡng 0 và hai biên; đồng điểm ưu tiên gần 0.
Ngưỡng chỉ thay đổi quyết định, không thay đổi số mẫu hoặc cân bằng lại phân bố.
PR/AP của nhánh ngưỡng và đối chứng giống nhau; hai điểm hoạt động khác nhau
được đánh dấu trên cùng đường. Balanced và under-sampling tìm lại C/ngram theo
cùng lưới, nên bảng so sánh chiến lược đã tinh chỉnh, không phải chỉ đổi một tham
số trong khi giữ tất cả tham số còn lại cố định.

Kết luận lựa chọn giới hạn trong các cấu hình đã thử. Chưa thử mọi chiến lược
trên cả sáu tổ hợp. Không chọn lại mô hình chỉ vì thấy một dòng có Test F1 cao
hơn. Cấu hình có CV F1 lớn nhất được lưu trong bundle `.joblib`, bao gồm ngưỡng
để dùng nhất quán trong demo.

## Kết quả và kiểm tra

Kết quả lần chạy sau cập nhật được ghi tại `results/verified_run/`; model,
score từng mẫu test và hình nằm trong `artifacts/spamham/`. Bằng chứng trước
cập nhật được giữ tại `results/history/d938b0a/verified_run/`.


Lượt chạy đầy đủ ngày 05/11/2026 hoàn tất trong **603,77 giây**, 34 cell gồm 18
cell code, không có lỗi trong cell. **55 kiểm thử đạt**. Tám cấu hình giữ lại
(sáu cơ sở, balanced, under-sampling) có CV F1, chỉ số test và tham số khớp bản
trước; không so sánh thời gian chạy. Nhánh từ dừng bị loại khỏi luồng test và
nhánh ngưỡng NB cũ không nằm trong chín ứng viên hiện tại.

| TF-IDF + SVM | CV F1 | Test Precision | Test Recall | Test F1 | FP | FN |
|---|---:|---:|---:|---:|---:|---:|
| Không điều chỉnh | 0,7736 | 0,8818 | 0,6278 | 0,7334 | 80 | 354 |
| Trọng số balanced | 0,7580 | 0,7139 | 0,7792 | 0,7451 | 297 | 210 |
| Under-sampling 2:1 | 0,7629 | 0,7791 | 0,7308 | 0,7542 | 197 | 256 |
| **Điều chỉnh ngưỡng** | **0,7876** | **0,8374** | **0,6877** | **0,7552** | **127** | **297** |

Ứng viên được chọn là **S3**, giữ từ dừng, C=0,3, n-gram=(1,2), không trọng số
lớp, không lấy mẫu lại, ngưỡng **−0,1776279252**. Test Accuracy **94,33%**,
AP **0,8117**. F1 và Recall tăng so với ngưỡng 0; Precision giảm. Kết quả này
không có nghĩa cấu hình đạt Recall cao nhất hoặc giải quyết hết mất cân bằng.

Đã kiểm tra nạp bundle và dự đoán toàn bộ 7.474 mẫu test. Nhãn từ bình luận gốc
khớp với nhãn khi dùng cột token đã lưu. Có **9/7.474 mẫu khác score** do `text`
và `cleaned_text` sẵn trong dataset tạo token khác nhau; không mẫu nào đổi nhãn.
Điểm test báo cáo dùng đường vào đã tiền xử lý cho thực nghiệm. Giữ nguyên hàm
tiền xử lý/dataset theo phạm vi đã thống nhất, ghi nhận chênh lệch này để tránh
khẳng định mọi score của hai đường vào đều giống nhau.

Khi đóng kernel trên Windows vẫn có thông báo joblib `resource_tracker.KeyError`
ở bước dọn bộ nhớ tạm, như phiên bản trước. Runner trả mã 0, trạng thái thành
công; các cell, file đầu ra và phép kiểm tra mô hình đã hoàn tất. Bộ kiểm thử
có một cảnh báo deprecation từ Argilla/Pydantic, không có kiểm thử thất bại.


## Kiểm tra tái lập toàn bộ luồng

Đã chạy lại tuần tự Notebook 01 → tool gộp → Notebook 02 bằng môi trường
Python 3.12.14 của project. Notebook 01 hoàn tất 4.34
giây, Notebook 02 hoàn tất 575.52 giây; mọi cell
code đều hoàn tất, không có lỗi trong cell. Đây là luồng local: xử lý JSON và
phản hồi nhãn đã lưu, không gọi lại Facebook/LLM/Argilla.

CSV 6.289 mẫu của Notebook 01, CSV gộp 38.220 mẫu, v7 và CSV tiền xử lý giữ nguyên
nội dung. **Cả chín dòng chỉ số, tham số và ngưỡng tái lập đúng lần trước**, chỉ
thời gian chạy thay đổi. Các bảng CV, bootstrap và dự đoán từng mẫu test cũng
khớp. 55 kiểm thử đạt. Bằng chứng: `results/verified_run/stability_check.json`.

S3 có CV F1 và Test F1 cao nhất trong chín cấu hình hiện tại. Test F1 S3 là
0,7552, S2 là 0,7542: chênh lệch nhỏ, chưa đủ kết luận S3 vượt trội về thống kê.
B4 có Precision cao nhất; S1 có Recall và AP cao nhất. Không có một cấu hình
đứng đầu mọi chỉ số.

Ngưỡng −0,1776 là ngưỡng **decision score**, không phải xác suất âm. SVM mặc
định dùng ngưỡng 0. Nhánh S3 giữ nguyên mô hình, chọn ngưỡng theo F1 validation
của điểm OOF trên train. Trên test, vùng điểm (−0,1776; 0] có 104 bình luận
đổi từ dự đoán ham sang spam: bắt thêm 57 spam và nhận nhầm thêm 47 ham. Đó là
lý do Recall/F1 tăng còn Precision giảm; không phải mô hình được huấn luyện
lại bằng nhãn test.

Đối chứng luôn dự đoán ham **không bị lỗi**: tập test có 6.523 ham/7.474 mẫu,
Accuracy = 87,28%, Recall/F1 spam = 0. Precision spam khi không dự đoán mẫu spam
nào là không xác định về phép chia; bảng cũ dùng quy ước hiển thị 0. Để tập trung
vào B1–B6, bảng riêng được rút thành ghi chú. Ma trận nhầm lẫn dùng chữ sáng
trên ô tối và chữ tối trên ô sáng.

Cell lưu đã ghi lại `.joblib` trong lượt chạy mới. Bundle giữ đúng pipeline,
ngưỡng, cấu hình từ dừng; dự đoán khớp trên toàn bộ 7.474 mẫu test sau khi tải
lại. Đã thử thêm hai chu kỳ lưu–nạp trên file tạm: nhãn và score giữ nguyên.
Khác biệt score ở chín đầu vào raw/cleaned đã ghi ở trên vẫn giữ nguyên, không
đổi nhãn. Hai notebook sẵn sàng làm cơ sở cho bước viBERT; chưa triển khai
viBERT trong lượt kiểm tra này.

Khi chạy tay, dùng Restart Kernel → Run All hoặc runner; cùng dữ liệu, môi
trường, seed và số luồng. Cell chạy riêng phụ thuộc biến của phần trước; sau
khi huấn luyện lại cần chạy tiếp đánh giá/lưu. Không bảo đảm trùng mọi chữ số
trên mọi máy hoặc khi đổi phiên bản/tham số. Model được ghi khi chạy cell có
`joblib.dump()`, không phải khi mở notebook để xem output.
