# Phân loại Spam/Ham bằng viBERT

| File | Nhiệm vụ |
|---|---|
| `data.py` | Đọc v7, giữ test của Notebook 02, tách validation theo nhóm và chuẩn hóa nhẹ. |
| `train.py` | Kiểm tra checkpoint thật, fine-tune hai loss, chọn bằng validation F1, fit cuối và đánh giá test. |
| `predict.py` | Nạp model/tokenizer đã lưu, phân loại bình luận mới. |
| `vibert_config.json` | Checkpoint/revision, seed, độ dài, batch, learning rate, số epoch và loss. |
| `__init__.py` | Khai báo package. |

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

## Thứ tự thực hiện

1. `data.prepare_data()` đọc `text` gốc, chuẩn hóa nhẹ, giữ test và lấy fold 0
   trong train làm validation. Không dùng chuỗi token underthesea của Notebook 02.
2. `train.load_pretrained()` nạp encoder và tokenizer cased; đầu phân loại hai
   lớp được khởi tạo để học. `smoke()` kiểm forward/backward trước khi train.
3. `train.select()` gọi `fit_variant()` cho `unweighted`/`balanced`;
   `ClassificationTrainer.compute_loss()` thực hiện loss tương ứng. Chọn loss
   và epoch bằng validation F1, không dùng test.
4. `train.finalize()` khởi tạo lại pretrained, fit toàn bộ train; lưu
   `model/`, đánh giá test, dùng `compare_with_svm()` đối chiếu đúng ID/nhãn S3,
   vẽ hình và nạp model kiểm tra lại. Sau khi hoàn tất mới ghi trạng thái xác nhận.
5. `predict.VibertPredictor` nạp thư mục `model/` và dự đoán bình luận mới;
   không chạy lại fine-tune. Notebook 03 ở `MODE="review"` thực hiện bước này
   sau khi đọc các bảng/lịch sử đã lưu.

CLI huấn luyện trên GPU Colab, từ gốc project:

```text
python -m pipeline.modern.train --stage all --output /content/drive/MyDrive/PTIT_SpamHam/viBERT/new_run
```

Không chạy hai tiến trình vào cùng output. Đổi cấu hình thì dùng thư mục mới.
Local `.venv-modern` dùng để review/demo CPU; `train.py` yêu cầu CUDA cho
select/finalize, chỉ smoke có thể chạy CPU.

Kết quả M1: loss thường, 3 epoch; test F1 **0,8493**, Precision **0,8942**,
Recall **0,8086**, AP **0,9108**. Trọng số thực nằm tại
`artifacts/modern/vibert/model/model.safetensors`; tokenizer và cấu hình đi cùng.
Đây không phải file `.py` hoặc `.joblib`. Token WordPiece có `##`/token đặc
biệt là bình thường; `[UNK]` thể hiện phần không biểu diễn được, xem
[chú giải](../../docs/guides/DOC_TOKEN_VA_KET_QUA.md).
