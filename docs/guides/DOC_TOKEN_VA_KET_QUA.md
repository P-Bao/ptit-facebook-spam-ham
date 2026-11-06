# Đọc token và kết quả dự đoán

## Bảng token trong Notebook 03

Cột bình luận là `text` gốc trong v7 sau chuẩn hóa nhẹ. Cột WordPiece là danh
sách token dùng cho viBERT, không phải câu tiếng Việt để đọc thông thường.

| Ký hiệu | Ý nghĩa | Có phải lỗi mã hóa không? |
|---|---|---|
| `##` | Phần tiếp nối của một từ; ví dụ `Ph` + `##am` | Không; đây là subword |
| `[CLS]` | Token đầu chuỗi, dùng cho biểu diễn phân loại của BERT | Không |
| `[SEP]` | Kết thúc/phân cách chuỗi | Không |
| `[PAD]` | Đệm chuỗi trong batch cho cùng chiều dài | Không; bảng một câu chưa padding có thể không hiện |
| `[UNK]` | Phần nội dung không biểu diễn được bằng từ vựng | Không phải lỗi font; có mất thông tin biểu diễn |

Đối chiếu trực tiếp v7: `cmt_10948` có `text` là
`グエン・ヴィエ・ フイ cuối kỳ gánh hết`. Tên tiếng Nhật đã có trong dữ liệu nguồn;
không phát sinh do bảng HTML hay chuyển Unicode. `cleaned_text` của dòng này
là `cuối kỳ gánh hết`; viBERT dùng `text`, giữ case và nội dung gốc, nên tokenizer
gặp cả tên nước ngoài. Không tự dịch tên hoặc sửa dữ liệu để làm mất ví dụ khó.

Trong Notebook 03 mục 3, “Số token sau cắt” tính cả token đặc biệt, “Số [UNK]”
đếm trên chuỗi sau cắt, cột cuối chỉ hiện tối đa 24 token đầu. Max length 128
tính cả `[CLS]`/`[SEP]`; padding thực hiện khi ghép batch. Trên train hiện tại
0,77% bình luận vượt giới hạn. Đây không phải tỷ lệ `[UNK]` hoặc tỷ lệ lỗi dự đoán.

Xem [tài liệu tokenizer BERT/WordPiece](https://huggingface.co/docs/transformers/model_doc/bert#berttokenizer).
Nếu một tên bị chia thành nhiều mảnh nhưng không có `[UNK]`, không thể kết
luận tokenizer lỗi chỉ dựa vào hình thức. Đổi tokenizer/vocab cần thực nghiệm
khác và có thể phải train lại, không sửa trong lượt hoàn thiện này.

## Phân biệt ba ngưỡng

| Bước | Ngưỡng | Tác dụng |
|---|---|---|
| Tiếp nhận nhãn AI | Confidence ≥ 0,75 | Quyết định mẫu nào được đưa vào CSV có nhãn của 01 |
| S3 | Decision score > −0,1776 | Quyết định spam khi dùng SVM đã huấn luyện |
| M1 | Xác suất spam ≥ 0,5 | Quyết định spam khi dùng viBERT đã fine-tune |

Confidence gán nhãn và xác suất/score phân loại là các đại lượng khác nhau.
Decision score SVM có thể âm; ngưỡng âm không phải xác suất âm. Xác suất viBERT
trong [0,1] cũng chưa bảo đảm độ tin cậy đã hiệu chỉnh.

## Đọc điểm số đúng phạm vi

- **S3 CV F1 = 0,7876:** trung bình F1 năm fold train, dùng chọn cấu hình/ngưỡng
  trong phạm vi khảo sát; chưa phải nested CV độc lập.
- **M1 validation F1 = 0,82048:** một fold validation trong train, dùng chọn
  loss và epoch; không đối chiếu trực tiếp với CV F1 như cùng phép đo.
- **Test:** cùng 7.474 ID/nhãn; S3 F1 = 0,7552, M1 F1 = 0,8493. Đây là bảng dùng
  để báo cáo so sánh sau khi chốt lựa chọn.

M1 có 91 FP và 182 FN. Bình luận ngắn/ngoài ngữ cảnh có thể vẫn sai. Demo một
vài câu dùng giải thích cách gọi mô hình, không thay thế đánh giá trên test.
