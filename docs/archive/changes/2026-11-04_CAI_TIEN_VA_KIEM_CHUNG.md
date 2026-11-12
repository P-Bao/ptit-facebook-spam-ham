# Chỉnh sửa, sắp xếp và kiểm chứng project — 04/11/2026

Đây là nhật ký lượt đầu, trước khi nhận mã crawler và dữ liệu Drive. Phần nguồn và
số cell/test hiện tại được cập nhật trong
[nhật ký bổ sung nguồn](2026-11-04_BO_SUNG_NGUON_THU_THAP.md); bằng chứng chạy lượt
đầu lưu tại `results/history/62219ff/`.

## Kết quả bàn giao

- Notebook chính: `notebooks/spam-ham-comments.ipynb`.
- Đã chạy toàn bộ **22 cell code** trong một kernel mới, trên CSV thật đủ **38,233 mẫu**, trong thư mục riêng chỉ có bản notebook lúc bắt đầu.
- Thời gian runner: **1576.3 giây**; Python **3.12.14**, CPU. Thời gian này của máy kiểm tra, không phải dự báo cho mọi máy.
- **31 kiểm tra hồi quy đạt**, kiểm tra dependencies và CLI/import đạt. Có cảnh báo deprecation từ dependency Argilla ở test; không phải ca kiểm tra thất bại.
- Mô hình hiện đại ý 4 bổ sung **chưa làm** theo yêu cầu; có heading và cell chờ. Chưa thể tuyên bố hoàn thành toàn bộ đề.
- Các thông tin nguồn chưa có được để trống; không tự thêm link group, ngày crawl hoặc người rà nhãn.

## 1. Cấu trúc cũ và mới

Trước đây script, notebook, phân công, ảnh và tài liệu nằm cùng thư mục gốc. Sau sửa:

| Trước | Sau | Lý do/tác dụng |
|---|---|---|
| `spam-ham-comments.ipynb` | `notebooks/spam-ham-comments.ipynb` | Xác định file nộp chính |
| `convert_phase2_json.py` | `pipeline/step_01_collect/` | Nhóm bước chuyển dữ liệu đã crawl |
| `clean_data_facebook_spamham.py` | `pipeline/step_02_preprocess/` | Nhóm làm sạch; thêm `text_preprocessing.py` dùng chung train/demo |
| Bốn script upload/AI/duyệt/download | `pipeline/step_03_label/` | Nhóm gán nhãn; thêm helper đối soát ID |
| `merge_labeled_data.py` | `pipeline/step_04_merge/` | Tách bước tích hợp hai phase |
| Logic CV/ngưỡng/under-sampling rải trong notebook | `pipeline/step_06_train_evaluate/training_utils.py`, được nhúng trong notebook | Dễ kiểm tra và vẫn tách notebook chạy được |
| Các Markdown kiểm tra/kế hoạch/phân công | `docs/overview`, `docs/plans`, `docs/team` | Tách tài liệu khỏi code; bản cũ được ghi là lịch sử |
| Ảnh đề bài | `docs/assignment/de_bai.png` | Tên dễ dùng, cùng tài liệu đề |
| Chưa có bộ kiểm tra/môi trường rõ ràng | `tests/`, `requirements/`, `scripts/`, `results/verified_run/` | Có bằng chứng kiểm chứng và cách chạy |

`README.md` ở gốc là điểm vào hướng dẫn; README cũ được lưu trong `docs/overview/`.
Đường dẫn mặc định của các script đã sửa về `data/` ở gốc repo. Dùng `python -m pipeline...` từ gốc repo; không tiếp tục gọi đường dẫn script cũ.

## 2. Notebook cũ → mới, vì sao sửa

| Trước | Sau | Tác dụng/mức cần thiết |
|---|---|---|
| Cài thư viện không chốt phiên bản | Kiểm tra phiên bản, `%pip` khi cần, requirements và in môi trường | Cần cho khả năng chạy/tái lập |
| Mô tả dataset gated và số liệu cũ | Khóa revision/hash, kiểm tra API công khai và đếm CSV thật | Đúng dữ liệu đang chạy |
| Lọc nhãn/thiếu text có thể không được đối soát | Kiểm tra cột, nhãn, ID, text; lỗi quan trọng dừng, không bỏ dòng âm thầm | Bảo vệ đầu vào |
| Tải nhiều từ điển tên GitHub khi chạy | Danh sách cố định từ bản cũ, ghi rõ nguồn cũ/chưa xác nhận thêm | Giảm phụ thuộc mạng; đây là cải thiện thêm |
| Cụm viết hoa như “Máy Tính” có thể bị che | Cần điều kiện họ và thành tố tên, phạm vi đầu câu được nói rõ | Sửa lỗi mất thông tin; không hứa ẩn danh hoàn toàn |
| Thêm emoji token nhưng còn emoji gốc | Loại emoji gốc và thêm đúng một token dấu hiệu | Mô tả khớp hành vi |
| Train và demo có thể khác xử lý URL/SĐT | Cùng module preprocessing, thử demo raw text | Sửa lỗi demo; demo là phần thêm ngoài đề |
| Câu chỉ dấu câu có khóa theo số dòng | Khóa ổn định theo nội dung, giữ nhóm trong cùng tập/fold | Giảm tách trùng train/validation/test |
| SVM mặc định balanced, kết luận có thể coi luôn tốt hơn | Tìm None/balanced bằng CV; so sánh riêng hai nhánh | Không ép kết quả theo giả định |
| Solver auto/dual xuất hiện cảnh báo chưa hội tụ khi kiểm tra | Chuyển primal `dual=False`, giữ cảnh báo hiển thị | Khắc phục vấn đề phát hiện khi chạy thật |
| Under-sampling trước CV, dùng `cv=5` | Lấy mẫu riêng train của từng fold nhóm, validation nguyên trạng | Sửa tính hợp lệ; kỹ thuật này không bắt buộc của đề |
| Chỉnh ngưỡng có thể sửa trực tiếp model được dùng ở bảng chính | Clone model, lấy OOF trong train; tối ưu mean F1 cùng fold | Không làm biến đổi model thí nghiệm trước, không dùng test |
| `best` chỉ nằm trong sáu mô hình chính | Chọn bằng CV trong tập ứng viên chính và bổ sung được ghi rõ; dùng đúng ngưỡng khi dự đoán/lưu | Kết luận và artifact nhất quán |
| Số liệu kết luận viết cứng | Sinh nhận xét từ kết quả lần chạy | Không dùng điểm cũ sau khi đổi code/split |
| Gọi AP là PR-AUC không giải thích | Đặt cột AP, giữ PR curve | Diễn giải đúng chỉ số; phần thêm ngoài đề |
| Output cũ chưa chứng minh code mới chạy được | Chạy riêng notebook từ kernel mới, kiểm tra load–predict và lưu manifest | Bằng chứng chạy đầy đủ |

Giữ **2 đặc trưng × 2 họ thuật toán × 2 kiểu stop words = 8 tổ hợp chính**, cộng các thí nghiệm phụ. Không thêm mô hình hiện đại hoặc số liệu giả cho phần chờ.

## 3. File ngoài đã sửa những gì?

- **Tất cả script được chuyển:** sửa root path/import để vẫn dùng `data/` ở gốc. Đây là thay đổi cần thiết do tái cấu trúc.
- **Auto-accept:** giữ ID từ CSV hoặc nhận prefix Phase 2; cache rỗng không chia 0; confidence không hữu hạn/nhãn lạ không tự duyệt; cache ID không thuộc CSV thì dừng; cập nhật server lỗi không giả vờ xuất thành công.
- **Upload/AI:** dùng ID sẵn có; `--no-upload` của AI không kết nối Argilla; cấu hình Argilla từ môi trường được dùng; score thiếu của suggestion không bị nâng lên 1.0; phản hồi AI chỉ nhận ID thuộc batch.
- **Download:** export rỗng có schema; lookup bằng ID hoặc vị trí legacy có kiểm tra; sai text/ID/mixed phase không bị âm thầm bỏ dòng hoặc ghép nhầm.
- **Merge:** giữ ID official export; chặn ID rỗng trước khi thêm prefix; kiểm tra ID/nhãn/text của Phase 1 và tập gộp. Không tự viết lại quy tắc nhãn hoặc xóa mọi text trùng.
- **Tài liệu/ignore:** phân công được chia sẻ, cập nhật cách chạy, bỏ qua data/cache/artifacts sinh ra.

**Không gọi OpenRouter hoặc cập nhật Argilla thật. Không crawl/gán nhãn/export/merge lại dataset v7.** CSV thật đã có đủ ID/nhãn Phase 2 và không có lỗi ID/text rỗng nêu trên; không có căn cứ buộc làm lại dữ liệu. Các nhánh script sửa được kiểm tra cục bộ; tính tích hợp máy chủ Argilla vẫn chưa được kiểm chứng.

## 4. Kết quả chạy thật của phiên bản mới

- Dataset SHA-256: `7305d712996da3ff0dce1ea4142844b7629e9880bcac47ee38934f0b6e6cfbee`.
- Revision: `5b904713bb2896ab8f6ff336545b9debd2489724`.
- Train **30,759**; test **7,474**, trong đó **951 spam**.
- Chọn theo CV: **TF-IDF + SVM / chính**, **giữ stop words**.
- F1-CV **0.7736**; test Accuracy **0.9419**, Precision **0.8818**, Recall **0.6278**, F1 **0.7334**, AP **0.8117**.

Toàn bộ bảng: `results/verified_run/metrics.csv`; cấu hình/dữ liệu: `run_manifest.json`; runner: `execution_status.json`; tests: `tests.xml`.
Đầu ra `.joblib` và CSV raw nằm trong thư mục local không commit; code tạo chúng có trong notebook.

**Không so trực tiếp điểm cũ 0,7943/0,8195 với điểm mới để nói mô hình đã tăng/giảm hiệu năng**: quy tắc nhóm dấu câu và tiền xử lý đã đổi, test cũ 7.480 mẫu còn test mới 7,474 mẫu. Mục tiêu lượt này là sửa tính đúng và khả năng chạy, không tối ưu điểm test.

CSV có **27 nhóm text giống nhau nhưng nhãn khác**. Giữ nguyên để người phụ trách xác nhận khi có ngữ cảnh; không tự gán nhãn lại. Điều này và lịch sử phát triển trên cùng dữ liệu là giới hạn cần nói khi trình bày.

## 5. Kaggle và nguồn dữ liệu

Đã tải và đọc notebook public Cương, phiên bản 1, 32 cell. Bản đó cũ hơn, có BoW/TF-IDF và các phần mất cân bằng. Không có kỹ thuật bắt buộc quan trọng mà repo trước sửa hoàn toàn thiếu; có một số nhận xét/số liệu không khớp nên không chép nguyên.

Chi tiết và link căn cứ: `docs/references/KAGGLE_VA_NGUON_DU_LIEU.md`. HF card cung cấp mô tả chung về nguồn/gán nhãn, nhưng chưa có thông tin group, thời gian/công cụ crawl và lịch sử v7 được nhóm xác nhận. Đã để ô trống trong notebook và tài liệu theo trả lời của người dùng; bổ sung nếu tìm được, không tự bịa.

## 6. Phạm vi kiểm chứng và phần còn thiếu

- Chạy đầy đủ notebook CPU, đủ dữ liệu thật; không dùng tập giả thay kết quả thực nghiệm.
- Kiểm tra module nhúng khớp bản chuẩn, token train/demo khớp, ID Phase 2/cache rỗng, nối metadata, nhóm fold và validation không bị under-sample.
- Kiểm tra CLI `--help` của 7 script sau chuyển thư mục, import Argilla và dependencies.
- Lượt kiểm tra đầu phát hiện cảnh báo SVM, đã dừng lượt đó và chạy lại solver đã sửa; chỉ kết quả lượt thành công cuối được đưa vào notebook chính.
- Chưa kiểm tra server Argilla/call API thật; chưa chứng minh toàn bộ lịch sử tạo nhãn. Không khẳng định toàn bộ nhãn đúng tuyệt đối.
- **Ý 4 bổ sung vẫn để sau.** Không viết báo cáo học phần ở lượt này.

## 7. Nhánh Git

Notebook còn cảnh báo giao diện tiến độ `IProgress not found` khi chạy headless, không có lỗi thực thi. Nếu cần thanh tiến độ dạng widget thì cài thêm ipywidgets; đây không phải yêu cầu của đề. Các cảnh báo quan trọng không bị tắt toàn bộ.

Nhánh làm việc được phép: **phi**. Ref push duy nhất: `HEAD:refs/heads/phi`, không force push.
SHA các nhánh khác trước sửa được ghi ở `results/verified_run/git_scope.json`; kiểm tra lại trước/sau push. Mã commit và kết quả push thực tế được nêu ở tin nhắn bàn giao.
