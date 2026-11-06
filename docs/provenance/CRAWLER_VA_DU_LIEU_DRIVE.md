> Đầu vào notebook 01 hiện dùng tên local: `posts.json` (JSON thu thập),
> `annotation_index.csv` (bảng định danh) và `annotation_predictions.json` (gợi ý),
> đặt trong `.cache/collection/`. Các tên file nguồn trong biên bản dưới đây là
> tên khi nhận dữ liệu; notebook hiện không phụ thuộc manifest/checksum.

> Biên bản nguồn lịch sử, không phải hướng dẫn chạy hiện hành. Project nay có
> ba notebook và đã hoàn tất viBERT. `.cache/collection/` giữ đầu vào tái lập,
> `data/` giữ dữ liệu làm việc. Xem [README](../../README.md) để chạy bản hiện tại;
> các số liệu đối soát bên dưới vẫn mô tả đúng thời điểm/file được kiểm tra.

# Kiểm chứng mã thu thập và dữ liệu Drive

**Cập nhật cấu trúc 05/11/2026:** đường dẫn dưới đây theo repo mới. 12 file nền
trong `external/facebook_scraper/` giữ nguyên byte; runner PTIT chuyển sang
`pipeline/collect/scrape_ptit_groups.py` và chỉnh import, đường dẫn, trợ giúp CLI.
Kiểm kê mã bàn giao nguyên bản thuộc lần kiểm tra 04/11; bản lưu trước tái cấu
trúc ở `results/history/7415098/`. Không coi runner đã chỉnh là nguyên byte mã
đã chạy lịch sử. Chi tiết: `docs/archive/changes/2026-11-05_TAI_CAU_TRUC_VA_DONG_BO.md`.

Ngày kiểm tra: 04/11/2026. Các số liệu dưới đây được tính từ file thực tế, không
lấy số trong README làm kết quả. Không đăng nhập Facebook, không cào lại, không gọi
dịch vụ gán nhãn ở lượt kiểm tra này.

Nguồn:

- Mã thành viên bàn giao: `external/facebook_scraper/` + `pipeline/collect/scrape_ptit_groups.py`.
- [Drive của nhóm](https://drive.google.com/drive/folders/1_htgS4zywI0soGbKY0gSboAtQexlm1j9).
- [CSV v7 dùng thực nghiệm](https://huggingface.co/datasets/p-storm/ptit-facebook-comments-spamham).
- Kết quả đối soát máy đọc được: `results/source_audit/audit.json`;
  URL, kích thước và SHA-256: `results/source_audit/drive_manifest.json`.

## 1. Kết luận về tính đúng của nguồn

Bộ bàn giao có đủ bằng chứng để trình bày một quy trình thu thập Phase 2 và tái hiện
các bước chuyển JSON → CSV → làm sạch → nối nhãn → gộp. Cấu trúc JSON phù hợp với
record mà runner tạo. Các bước chuyển, làm sạch theo quy tắc lịch sử và gộp tái
hiện khớp cả nội dung lẫn thứ tự file Drive.

Chưa có đủ bằng chứng để khẳng định mã hiện tại là chính xác phiên bản đã chạy,
đã lấy hết mọi bình luận hoặc toàn bộ nhãn đã được người duyệt xác nhận. Đặc biệt,
bản gộp Drive và v7 có cùng dữ liệu nhưng khác 1.366 nhãn. Giữ nguyên v7 để thực
nghiệm; không dùng bản Drive thay thế hoặc tự quyết định nhãn nào đúng.

## 2. Từng file mã thu thập làm gì?

| File | Vai trò và mức cần thiết với đề tài |
|---|---|
| `scrape_ptit_groups.py` | Runner PTIT chính: đọc cấu hình group, tham số thời gian/giới hạn, lấy batch bài, lấy comments/replies, ghi JSON và trạng thái tiếp tục. Đây là file cần giải thích đầu tiên. |
| `group_post_scraper_v2.py` | Gửi yêu cầu GraphQL lấy feed group theo cursor; tách bài, nội dung, thời gian, số bình luận và media. Runner sử dụng trực tiếp. |
| `comment_scraper.py` | Tạo payload, gửi yêu cầu và phân tích comments/replies. Comment chính có vòng cursor; replies chưa có vòng phân trang. |
| `main.py` | Các hàm URL/ID và menu CLI; `fetch_comments_for_post()` được runner gọi để ghép comments với replies. |
| `proxy_utils.py` | Tiện ích chọn/đổi proxy và nhận diện lỗi kết nối, được các module HTTP dùng. Không phải bước xử lý văn bản hoặc huấn luyện. |
| `post_scraper.py` | Nhánh lấy bài page/profile, dùng bởi menu/GUI. Không phải nhánh runner thu group PTIT đang đối soát. |
| `single_post_image.py` | Lấy ảnh của bài đơn. Không tạo đặc trưng cho mô hình văn bản hiện tại. |
| `facebook_ui.py` | Giao diện PyQt6 nhập cấu hình và điều khiển luồng scrape; có hỗ trợ trình duyệt/Selenium. Không cần mở GUI để chạy notebook. |
| `.env.example` | Mẫu tên biến `FB_COOKIES`, `FB_DTSG`; không phải phiên đăng nhập thật. Không đưa `.env` thật vào Git/notebook. |
| `requirements.txt` | Bốn dependency của bộ crawler, gồm cả GUI/browser. Notebook có bộ dependency riêng. |
| `README.md`, `README_RUN.md` | Hướng dẫn chạy bản bàn giao; hai file giống nhau. Lệnh ví dụ không phải log chứng minh lệnh nhóm đã chạy. |
| `README_ORIGINAL.md` | Hướng dẫn dự án nguồn được bộ bàn giao dẫn lại, có tham chiếu `mohdtalal3/facebook_post_comment_scraper`. Giữ để nhận diện nguồn tham khảo; chưa có commit upstream chốt cho phiên bản nhóm. |

13 file được kiểm kê hash; cả 8 file Python phân tích cú pháp thành công. Mã bàn
giao được giữ nguyên để làm căn cứ, không âm thầm sửa rồi gọi là mã đã dùng lịch sử.

## 3. Snapshot thật chứa những gì?

`groups_data_phase2.json`: **2.170 bài**, **9.438 comment chính**, **4.395 reply**.
Không có post ID trùng trong snapshot. Năm group được xác định bằng ID có trong
record; các URL dưới đây được dựng từ ID, không khẳng định hiện còn truy cập được.

| Group trong snapshot | Link từ ID | Bài | Comment | Reply |
|---|---|---:|---:|---:|
| D26 (2008) PTIT | [1130813927934700](https://www.facebook.com/groups/1130813927934700/) | 500 | 1.104 | 557 |
| PTIT Confessions | [1210825580901941](https://www.facebook.com/groups/1210825580901941/) | 500 | 1.813 | 1.262 |
| D26 Học viện… (2K8 PTIT) | [1605563914144178](https://www.facebook.com/groups/1605563914144178/) | 500 | 3.283 | 1.385 |
| Cộng đồng sinh viên PTIT | [450310163844833](https://www.facebook.com/groups/450310163844833/) | 500 | 2.927 | 918 |
| D21 PTIT Hà Nội | [778910719727961](https://www.facebook.com/groups/778910719727961/) | 170 | 311 | 273 |

Code hiện cấu hình 16 group, nhưng D21 trong raw lại không thuộc danh sách đó.
ID `450310163844833` mang tên cấu hình “2K5 PTIT”, tên raw là “Cộng đồng sinh viên
PTIT”; chưa có căn cứ kết luận sai ID hay group đã đổi tên. Không viết đã cào đủ
16 group hoặc tất cả 38.233 mẫu đều xuất phát từ 5 group này.

- `created_at` của bài: `2025-05-17 23:40:45` → `2026-09-27 12:55:52`.
- `scraped_at`: `2026-09-27 10:06:57` → `2026-09-27 14:31:22`.
- Đây là giá trị có sẵn trong JSON, không kèm múi giờ. Thời gian ở CSV là của bài
  đăng, không phải thời gian từng bình luận.
- Drive chỉ có CSV đã gán nhãn của Phase 1, không có raw/log thu thập Phase 1.
- HF card nói group công khai; README crawler đề cập phiên đăng nhập/group riêng
  tư. Chưa xác minh loại truy cập của từng group, không tự điền.

## 4. Đối soát từng chặng dữ liệu

| Chặng/file | Kết quả tính lại | Ý nghĩa |
|---|---:|---|
| JSON raw | 13.833 comment/reply | Không phải 13.833 bài hoặc mẫu đã có nhãn. |
| Bỏ text rỗng | −129 | Còn 13.704 dòng trước đối chiếu Phase 1. |
| Bỏ text làm sạch trùng Phase 1 | −4.135 | `comments_phase2.csv`: 9.569 dòng; tái hiện khớp. |
| Khử trùng `(post_id, comment_type, text)` | −171 | `comments_phase2_cleaned.csv`: 9.398 dòng; tái hiện khớp cách làm cũ. |
| Cache GLM | 6.605 ID | 6.302 đạt confidence ≥ 0,75; confidence không phải xác suất nhãn chắc chắn đúng. |
| Export Phase 2 | 6.303 dòng | 5.197 ham, 1.106 spam; một ID ngoài cache, chưa biết lịch sử duyệt. |
| Export Phase 1 | 31.931 dòng | 28.722 ham, 3.209 spam. |
| Bỏ trùng chéo khi gộp | −1 dòng Phase 2 | Giữ 6.302 Phase 2; bản gộp 38.233 dòng, tái hiện khớp. |
| Bản gộp Drive | 33.919 ham / 4.314 spam | Chưa phải phân bố nhãn v7. |
| V7 trên HF | 32.907 ham / 5.326 spam | Bản duy nhất dùng để huấn luyện/đánh giá notebook. |

Có 10 dòng export Phase 2 dùng raw text thay `cleaned_text` rỗng. Đây là fallback
có trong uploader, giải thích hết chênh lệch text; không ghi thành “đã được sửa
tay”. Mọi text/comment_type/reaction_count nối theo ID đều khớp snapshot nguồn.

Cleaner mới mặc định bỏ thêm **42 dòng** chỉ dấu câu/emoji. Áp mặc định đó rồi dùng
lại cache `cmt_<chỉ số>` sẽ nối nhầm vì vị trí dòng thay đổi. Đã thêm
`--keep-punctuation-only` để tái hiện lịch sử; không đổi mặc định của lượt làm dữ
liệu mới. Bộ đối soát và notebook dùng quy tắc lịch sử cùng hash file đã chốt.

V7 giữ nguyên tập/thứ tự 38.233 ID và toàn bộ text/cleaned_text/comment_type/
reaction_count; chỉ nhãn đổi: **1.189 ham → spam**, **177 spam → ham**. Chưa có log
giải thích 1.366 thay đổi này. Cache/CSV hiện có không chứng minh toàn bộ quá trình
người duyệt trên Argilla, tỷ lệ kiểm tay hoặc người sửa nhãn v7.

## 5. Giới hạn mã crawler và quyết định xử lý

| Phát hiện | Ảnh hưởng | Xử lý trong lượt này |
|---|---|---|
| `fetch_replies()` chỉ gửi một trang, không lặp cursor | Có thể thiếu replies | Ghi rõ giới hạn; không tuyên bố thu đủ mọi reply. |
| Khi có output nhưng thiếu state, code có thể coi group là completed | Tiếp tục phiên cũ có thể bỏ phần chưa lấy | Không coi completed là bằng chứng đủ dữ liệu. |
| Ghi state có thể trước batch cuối; kết thúc một số nhánh vẫn đánh dấu completed | Có nguy cơ checkpoint vượt dữ liệu đã lưu | Không chạy lại crawler như công cụ đã được kiểm chứng phục hồi hoàn toàn. |
| `--force-all` vẫn giữ cursor và ID đã lưu | Không đồng nghĩa bắt đầu sạch từ đầu | Không dùng tên flag để mô tả sai cách lấy dữ liệu. |
| `--year` chỉ đặt mốc dưới 01/01; bỏ video/reel | Không phải lấy tất cả loại bài trong đúng một năm đóng | Mô tả đúng bộ lọc và phạm vi snapshot. |
| Không lưu ID/timestamp gốc của comment | Khó truy vết comment về Facebook và chia theo thời gian comment | Giữ giới hạn; không biến ID dòng `cmt_*` thành ID Facebook. |
| 16 bài báo có comment nhưng mảng comments rỗng | Snapshot không chứng minh thu đủ mọi comment | Ghi nhận, không tự tạo thêm dữ liệu. |

Với mục tiêu dùng tập đã cào sẵn, **không cần cào lại để chạy notebook**. Nếu sau
này thực sự thu thêm, cần sửa phân trang replies và checkpoint/resume trước khi
tin cậy số lượng đầy đủ; đồng thời lưu ID/timestamp comment, command/config/log và
phiên bản mã. Các đề xuất đó không được trình bày như đã làm ở lần thu cũ.

## 6. Chạy lại đối soát offline

Đặt 7 file Drive vào `data/{raw,intermediate,labeled}/`, v7 vào `data/final/labeled_v7.csv`. Từ gốc repo:

```powershell
.\.venv\Scripts\python.exe -m tools.dataset_versions.audit_source --source-dir data --training-csv data/final/labeled_v7.csv --runner pipeline/collect/scrape_ptit_groups.py --crawler-dir external/facebook_scraper --output results/source_audit/audit.json
```

Lệnh chỉ đọc file và xuất JSON kiểm tra; không sửa nhãn, không gọi Facebook, LLM
hoặc Argilla. Audit được tách riêng khỏi hai notebook trình bày. Notebook 01
import hàm xử lý từ repo để minh họa snapshot; notebook 02 dùng trực tiếp v7.
