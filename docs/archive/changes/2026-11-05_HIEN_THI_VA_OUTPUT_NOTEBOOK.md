# Sửa hiển thị và đầu ra notebook — 05/11/2026

## Trước và sau

| Trước | Sau | Tác dụng |
|---|---|---|
| IPython Code/pandas HTML chịu màu mặc định của IDE, có chữ trắng trên nền trắng | `pipeline/presentation/notebook_display.py` dùng màu inline cho bảng/code | Đọc được trong theme sáng/tối; không cần đổi theme toàn IDE. |
| Nhiều khối source mở sẵn, che mất các bảng output | Khối “Xem code” đóng/mở, đặt sau dữ liệu | Tập trung vào comment, biến đổi và file thực tế trước. |
| Bảng/Series có nguy cơ mất tên dòng khi đổi renderer | Giữ chỉ mục trong HTML | Biết từng số liệu/môi trường/nhãn tương ứng mục nào. |
| JSON → bảng chỉ làm trên 3 bài | Làm phẳng toàn bộ 2.170 bài, bỏ rỗng, ghi và đọc lại CSV | Thực sự tạo 13.704 dòng raw CSV từ 13.833 comment/reply. |
| Clean chỉ minh họa map text | Gọi `clean_facebook_comments()` thật, có khử trùng/lọc và ghi file | 10.974 dòng sau clean; hiển thị trước/sau. |
| Chỉ đọc vài gợi ý/export nhãn để xem | Chạy CLI auto-accept offline trên cache đúng ID, xuất lại export chốt sau kiểm metadata | 6.302 nhãn áp ngưỡng, 6.303 nhãn trong export đã lưu; không giả gọi AI mới. |
| Chỉ in chung đường dẫn artifacts | Bảng file/tình trạng/số dòng/KB/nội dung sau mỗi bước và cuối notebook | Biết file nào có sẵn, file nào vừa được ghi, chứa cột gì. |
| Khó hiểu cell có chạy thật hay chỉ show source | Bảng chế độ thực thi trong notebook và hướng dẫn đầy đủ | Phân biệt đọc source, gọi hàm, CLI offline và chạy live. |

Notebook 01 hiện có 17 cell, 8 code cell. Notebook 02 giữ 36 cell, 19 code cell;
chỉ đổi cách render bảng, không đổi pipeline mô hình/dữ liệu/grid/đánh giá.
Ý 4 bổ sung vẫn để sau theo yêu cầu.

## Các file tạo bởi notebook 01

`comments_from_snapshot.csv` (13.704 dòng), `comments_cleaned_from_snapshot.csv`
(10.974), `accepted_from_saved_cache.csv` (9.398 dòng, có dòng chưa nhãn),
`labels_auto_accepted.csv` (6.302), `labels_saved_official_export.csv` (6.303),
`collection_demo_manifest.json`. Lần kiểm chứng ở
`.work/runs/readable_collection/artifacts/`; mở notebook bình thường dùng
`artifacts/collection/` hoặc `SPAMHAM_OUTPUT_DIR`.

CSV mới xử lý toàn bộ snapshot, không lọc tham chiếu như lịch sử; cache phải dùng
CSV cleaned lịch sử 9.398 dòng để giữ đúng ID. Không áp cache cũ lên CSV mới,
không nhận output notebook 01 là v7. Nguồn/nhãn sẵn có được giữ nguyên byte.

## Kiểm chứng và đánh giá

- **47/47 test đạt**. Notebook 01 chạy đủ **8/8 cell code** trong 5,78 giây;
  notebook 02 đủ **19/19** trong 1.458,22 giây CPU (khoảng 24,3 phút). Không lỗi cell.
- Cả **16 dòng thí nghiệm giữ nguyên chỉ số và tham số** so với bản trước; chỉ
  cách hiển thị và thời gian đo khác. V7 và các snapshot giữ nguyên hash.
- Kiểm tra browser lấy HTML từ output notebook thật, chạy theme sáng/tối và CSS
  mô phỏng lỗi. Màu ô bảng và code có tương phản tối thiểu khoảng 14,48:1 trong
  kiểm tra này. Có xem ảnh render local; không coi đây là thao tác trực tiếp trên IDE.
- Renderer escape nội dung comment/source; chỉ đọc AST khi show code, không chạy
  source đó. Tests kiểm tra giữ index, không đổi DataFrame và thông tin file thật.
- Notebook 01 chạy thật không gọi Facebook/LLM/Argilla. Notebook 02 được chạy lại
  với renderer mới; trạng thái và số liệu cuối ở `results/verified_run/` và
  `results/notebook_display/verification.json`.
- Model đã lưu được kiểm tra trên 7.474 test IDs: TN 6.443, FP 80, FN 354, TP 597.
  API raw comment cho cùng dự đoán trên toàn bộ các test IDs này. Bằng chứng
  `results/notebook_display/prediction_check.json`.
- Precision spam 88,18%, recall 62,78%, F1 0,7334: model còn bỏ sót spam. Tính nhất
  quán code không đồng nghĩa mọi nhãn v7 hoặc mọi dự đoán ngoài thực tế đều đúng.

Tổng quan folder/file, luồng từng bước và giải thích chạy notebook/CLI:
`docs/guides/TONG_QUAN_PROJECT_VA_CACH_THUC_THI.md`.
