# Bổ sung nguồn thu thập vào notebook

Phạm vi: mã crawler do thành viên bàn giao và 7 file trong Drive; hoàn thiện
notebook/thuyết minh kỹ thuật và kiểm chứng. Không viết báo cáo học phần, không
cào Facebook mới hoặc gọi dịch vụ gán nhãn, không thay nhãn v7.

## 1. Trước và sau

| Nội dung | Trước | Sau | Tác dụng |
|---|---|---|---|
| Nguồn thu thập | Có HF card, chưa có mã/raw đủ để xác nhận chi tiết | Kiểm kê 13 file crawler, tải/khóa hash 7 file Drive, xác định 5 group thật của Phase 2 | Phần nguồn có căn cứ, không tự điền 16 group hoặc lịch sử chưa biết. |
| Notebook giải thích crawler | Mô tả chung và mục chưa xác nhận | Trích lời gọi lấy bài, hàm ghép comments/replies từ mã bàn giao; giải thích writer/state và giới hạn | Có code để trình bày quá trình làm; không cần đăng nhập/cào live khi Run All. |
| Kiểm tra chuỗi dữ liệu | Chỉ kiểm tra CSV cuối dùng train | Tái hiện JSON → flat → clean lịch sử → merge; nối metadata/cache; so Drive với v7 | Phát hiện lỗi nguồn/phiên bản thay vì chỉ nhìn số dòng. |
| Nhãn Drive so v7 | Chưa có đối chiếu | Cùng 38.233 ID/text, khác 1.366 nhãn; ghi rõ thiếu log nguyên nhân | Tránh dùng nhầm CSV hoặc nhận là đã chứng minh tất cả nhãn được duyệt tay. |
| Cleaner và cache lịch sử | Mặc định loại thêm dòng chỉ dấu câu/emoji | Thêm `--keep-punctuation-only`; mặc định mới giữ nguyên | Giữ đúng 9.398 vị trí cũ khi tái hiện cache, tránh nối lệch 42 dòng. |
| Export/nối metadata | Text hiển thị fallback có thể bị coi là sai khi clean rỗng | Dùng cùng fallback clean rỗng → raw như uploader; vẫn kiểm tra ID/text khác nghiêm ngặt | Giải thích và hỗ trợ đúng 10 dòng emoji lịch sử; không nới lỏng nhận nhầm text tùy ý. |
| Notebook tách riêng | Có 3 module ML nhúng | Có thêm 2 module nguồn, catalog file/hash và đối soát nhúng | Không cần import thư mục crawler/pipeline ngoài; vẫn cần dữ liệu hoặc mạng. |
| Hướng dẫn nộp/demo | Hướng dẫn chạy chung | Có bảng từng mục notebook, đối chiếu đề, pipeline vs notebook và cách offline | Biết cần trình bày/nộp gì, tránh coi pipeline là lệnh train tự động. |
| Bằng chứng lần chạy | 22 code cell; 31 test của lượt trước | Runner bổ sung hash code đã chạy, test nguồn bổ sung, chạy lại notebook đầy đủ | Gắn output với đúng phiên bản code hiện tại. Kết quả thực tế ở `results/verified_run/`. |

Các cell ML cũ được giữ nguyên nội dung. Bằng chứng so sánh code:
`results/source_audit/notebook_update.json`. Notebook thêm 11 cell, thành **51 cell,
27 code cell**; trong đó thêm 5 code cell dành cho nguồn dữ liệu. Không cố thay
mô hình/tham số chỉ để tăng điểm trên test.

## 2. Những gì được xác minh và những gì chưa thể kết luận

- Raw Phase 2: 2.170 bài, 5 group, 9.438 comment và 4.395 reply.
- Tái hiện flat 9.569 dòng, clean lịch sử 9.398 dòng và bản gộp 38.233 dòng khớp
  nội dung/thứ tự. Export Phase 2 có 6.303 dòng, bỏ một trùng khi gộp còn 6.302.
- Cache 6.605 dự đoán; 6.302 đạt ngưỡng ≥0,75 đều trong export. Một ID export
  ngoài cache chưa biết lịch sử. Không suy diễn confidence thành chất lượng nhãn.
- So Drive → v7: 1.189 ham→spam, 177 spam→ham; không đổi nội dung/metadata kiểm tra.
- Chưa có raw/log Phase 1 hoặc lịch sử chỉnh nhãn v7; chưa chứng minh mã bàn giao
  chính là phiên bản lịch sử, trạng thái công khai từng group hoặc đã thu đủ replies.
- Các giới hạn crawler được ghi rõ; chưa sửa code gốc hay chạy lại crawl vì đang
  dùng tập sẵn có. Cần xử lý phân trang/checkpoint nếu sau này thu mới.

## 3. Kiểm tra và nơi xem kết quả

- **Run All thành công: 27/27 cell code, không lỗi thực thi, 1.529,5 giây CPU**
  (khoảng 25,5 phút; thời gian đo trên máy hiện tại). Notebook gồm 51 cell cả
  Markdown. Hash code cuối khớp hash code được thực thi.
- Cả 16 dòng thí nghiệm giữ nguyên chỉ số/tham số/lựa chọn so với lượt trước;
  chỉ cột thời gian chạy thay đổi. Mô hình được chọn bằng CV vẫn là TF-IDF + SVM,
  giữ stop words: test Accuracy 0,9419, Precision 0,8818, Recall 0,6278,
  F1 0,7334. Bổ sung nguồn không được ghi thành cải thiện điểm mô hình.
- **39 test đạt**, gồm 31 test cũ và 8 trường hợp mới: tái hiện legacy clean,
  fallback metadata/export, chuỗi snapshot tổng hợp, kiểm tra AST không thực thi
  crawler, module nhúng và dữ liệu Drive thật. Có một cảnh báo deprecation từ
  Argilla/Pydantic, không có test lỗi.
- Notebook chạy headless còn cảnh báo `IProgress not found` của tqdm; không ảnh
  hưởng kết quả. Không có lỗi cell hoặc cảnh báo hội tụ SVM trong output lần này.
- Kết quả thực thi notebook mới, thời lượng, số cell và hash code:
  `results/verified_run/execution_status.json`. Bảng metrics/config:
  `results/verified_run/metrics.csv`, `run_manifest.json`.
- Đối soát nguồn đầy đủ: `results/source_audit/audit.json`, manifest download và
  bằng chứng code cũ không thay đổi trong cùng thư mục.
- Đã chạy thực tế bốn CLI convert, clean legacy, auto-accept offline và merge.
  Flat/clean/merge khớp file Drive; 6.303 dòng nối metadata đúng, 6.302 dòng
  auto-accept khớp phần tương ứng trong export chính thức. Dùng export chính thức
  sẵn có cho merge, không giả lập người duyệt ID ngoài cache. Kết quả:
  `results/source_audit/cli_replay.json`.
- Đã chạy đúng cell tải nguồn với cache trống: tải đủ 7 file, đúng hash và qua
  các assertion đối soát trong 21 giây tại lần kiểm tra. Bằng chứng:
  `results/source_audit/download_verification.json`.
- Bằng chứng lượt trước được lưu dưới `results/history/62219ff/` khi công bố lần
  chạy mới, để phân biệt 22 cell cũ với 27 cell hiện tại.

Notebook được chạy trong thư mục riêng với v7 và snapshot local đúng hash. Cách
tải Drive được kiểm tra riêng bằng các URL public; lần Run All dùng bản local,
không biến kết quả đó thành khẳng định mọi mạng hoặc quyền Drive tương lai đều ổn.

## 4. File cần mở trước

1. `notebooks/spam-ham-comments.ipynb`: nội dung code/thực nghiệm chính để nộp/demo.
2. `docs/guides/HUONG_DAN_NOTEBOOK_VA_NOP_BAI.md`: trả lời thứ tự, ý nghĩa, mức đáp
   ứng các bước cốt lõi và sự khác nhau giữa notebook/pipeline.
3. `docs/references/CRAWLER_VA_DU_LIEU_DRIVE.md`: từng file crawler, số liệu, nguồn,
   giới hạn và cách chạy kiểm chứng offline.

Thao tác Git chỉ được phép ở nhánh **phi** theo phạm vi người dùng đã giao. Kết
quả kiểm tra ref ở `results/verified_run/git_scope.json`; mã commit/push thực tế
được nêu trong bàn giao sau khi hoàn tất.
