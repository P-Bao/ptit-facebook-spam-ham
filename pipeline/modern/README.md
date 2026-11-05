# Phân loại Spam/Ham bằng viBERT

| File | Nhiệm vụ |
|---|---|
| `data.py` | Đọc v7, giữ test của Notebook 02, tách validation theo nhóm và chuẩn hóa nhẹ. |
| `train.py` | Kiểm tra checkpoint thật, fine-tune hai loss, chọn bằng validation F1, fit cuối và đánh giá test. |
| `predict.py` | Nạp model/tokenizer đã lưu, phân loại bình luận mới. |
| `vibert_config.json` | Checkpoint/revision, seed, độ dài, batch, learning rate, số epoch và loss. |

Checkpoint: [FPTAI/vibert-base-cased](https://huggingface.co/FPTAI/vibert-base-cased),
nguồn [FPT viBERT](https://github.com/fpt-corp/viBERT). Tokenizer cased giữ chữ
hoa/thường và từ dừng; không dùng chuỗi token underthesea của mô hình truyền thống.

Phát triển: 24.815 fit + 5.944 validation, chọn loss/epoch bằng validation F1.
Khởi tạo lại từ checkpoint gốc, fit toàn bộ 30.759 train rồi đo trên 7.474 test.
Validation F1 không phải CV F1 năm fold. Ngưỡng xác suất spam cố định 0,5.

Model/tokenizer lưu trong thư mục output, tách khỏi mã nguồn. `verification.json`
chỉ ghi sau khi hoàn tất test và kiểm tra nạp lại; `smoke.json` không phải kết
quả thực nghiệm đầy đủ. Checkpoint tạm lưu theo epoch để tiếp tục khi gián đoạn.

Hướng dẫn: [chạy Colab và thu kết quả](../../docs/guides/HUONG_DAN_VIBERT_COLAB.md).
