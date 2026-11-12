# Giải đáp và sơ đồ tái cấu trúc để trình bày project

**Hồ sơ kế hoạch trước triển khai.** Cấu trúc đã được thực hiện; xem
[nhật ký tái cấu trúc](../changes/2026-11-05_TAI_CAU_TRUC_VA_DONG_BO.md) và README
gốc để dùng đường dẫn/lệnh hiện tại. Bảng cũ → đề xuất dưới đây giữ nguyên để truy vết.

Ngày đối chiếu: 05/11/2026. Đây là bản đề xuất có căn cứ từ code và dữ liệu hiện
tại; chưa di chuyển module hoặc tách notebook. Phần mô hình hiện đại vẫn được
giữ chỗ, chưa triển khai trong lượt này.

## 1. Hai lớp xử lý văn bản không phải hai lần chạy lại cùng một công đoạn

Luồng chuẩn bị dữ liệu:

```text
Facebook → JSON thô → CSV → làm sạch để gán nhãn → gán/chốt nhãn → CSV có nhãn
```

Luồng thực nghiệm:

```text
CSV v7 đã có nhãn → chuẩn hóa/tách từ/đối chứng stop words
                 → chia train/test → học đặc trưng và mô hình → đánh giá → demo
```

| Module hiện tại | Đầu vào → đầu ra | Công việc thực tế |
|---|---|---|
| `pipeline/step_02_preprocess/clean_data_facebook_spamham.py` | CSV thô → thêm `cleaned_text` | NFC, HTML, URL/SĐT thành token, loại emoji, chuẩn hóa dấu câu/khoảng trắng, khử trùng/lọc dòng. Giữ cột `text` gốc. |
| `pipeline/step_02_preprocess/text_preprocessing.py` | `cleaned_text` và `text` gốc → chuỗi token | Chuẩn hóa đầu vào mô hình, che tên theo heuristic, lowercase, giảm ký tự lặp, tách từ underthesea, giữ/bỏ stop words, thêm `emoji_token` nếu văn bản gốc có emoji. |

Có phần chuẩn hóa chồng nhau, nhưng lớp sau không chỉ chạy lại cleaner: nó có tách
từ và các biến thể mô hình. Không xóa cột text gốc vì còn dùng nhận diện tín hiệu
emoji và phân tích lỗi. Không giả định gọi toàn bộ hàm tiền xử lý nhiều lần luôn
giữ nguyên kết quả; demo cần cùng một đường xử lý với train.

Đề yêu cầu thu thập, tiền xử lý, ít nhất hai đặc trưng, NB/SVM và đánh giá. Đề
không cấm làm sạch trước khi gán nhãn và không quy định gán nhãn phải đứng trước
mọi thao tác làm sạch. `step_02`, `step_03` hiện là thứ tự công cụ chuẩn bị dữ liệu,
không phải đánh số tương ứng tuyệt đối với các ý trong đề.

Để bớt nhầm, bản đề xuất dùng tên folder theo chức năng: `collect`, `prepare`,
`label`, `features`, `modeling`, `inference`; số thứ tự dành cho hai notebook.

## 2. Nguồn crawler GitHub đã được đối chiếu

Repo nguồn: https://github.com/mohdtalal3/facebook_post_comment_scraper

Nhánh main ở lần kiểm tra: `0c33360a15c1df52712b8abeaa13d38e79cc7ba5`, đã chuyển
sang cấu trúc khác. Vì vậy dùng revision cũ có các module cùng tên để đối chiếu:

https://github.com/mohdtalal3/facebook_post_comment_scraper/tree/b518ce1ab85c968754a49d627cab554100da9f81

| File bàn giao | So với revision cũ nói trên | Vai trò trong nhánh PTIT |
|---|---|---|
| `facebook_ui.py` | SHA-256 giống hệt | GUI; runner PTIT không cần mở GUI. |
| `post_scraper.py` | SHA-256 giống hệt | Được import gián tiếp qua `main.py`; nhánh lấy bài page không được runner PTIT gọi. |
| `single_post_image.py` | SHA-256 giống hệt | Được import gián tiếp qua `main.py`; nhánh bài đơn không phải luồng PTIT chính. |
| `proxy_utils.py` | SHA-256 giống hệt | Có trong bộ công cụ, không vì tồn tại file mà suy ra runner PTIT gọi các hàm đó. |
| `comment_scraper.py` | Khác byte, mức giống theo dòng khoảng 98,77% | Lấy comment/reply, có thay đổi so với bản đối chiếu. |
| `group_post_scraper_v2.py` | Khác byte, mức giống theo dòng khoảng 95,05% | Lấy bài group, có thay đổi so với bản đối chiếu. |
| `main.py` | Khác byte, mức giống theo dòng khoảng 99,21% | Runner dùng `fetch_comments_for_post()`, có thay đổi so với bản đối chiếu. |
| `scrape_ptit_groups.py` | Không có ở revision đối chiếu; API lịch sử đường dẫn này trả rỗng | Runner cấu hình/điều phối riêng cho PTIT trong bộ bàn giao. |

Mức giống theo dòng chỉ là kết quả so sánh văn bản, không phải tỷ lệ đóng góp.
Không suy ra ai viết/sửa, hoặc code đã chạy lịch sử chính xác ở commit nào. Có
căn cứ nói dùng nền một repo có sẵn, có runner PTIT riêng và một số module đã
điều chỉnh; chưa đủ căn cứ nói thành viên chỉ viết đúng một file.

Chi tiết hash/URL: `results/source_audit/upstream_comparison.json`.

## 3. Runner PTIT chạy qua đâu và sinh gì?

```text
scrape_ptit_groups.py
  ├─ đọc PTIT_GROUPS, tham số thời gian/giới hạn và cấu hình phiên
  ├─ group_post_scraper_v2.fetch_posts(): lấy các trang bài
  ├─ callback on_batch:
  │    └─ main.fetch_comments_for_post(post_id)
  │         ├─ comment_scraper.fetch_comments()
  │         └─ comment_scraper.fetch_replies()
  ├─ tạo record: group, post ID, permalink, nội dung, thời gian, comments/replies
  ├─ SingleFileWriter: ghi JSON/JSONL
  └─ StateManager: ghi trạng thái/cursor
```

Đầu ra chính:

- JSON hoặc JSONL theo `--output`; mặc định code hiện là
  `ptit_groups_data_2025.json`, dù mặc định `--year` lấy năm hiện tại.
- `state.json` hoặc đường dẫn `--state-file`.
- Nếu có `--comments-output`: thêm file chỉ chứa text, dạng TXT/JSON/JSONL.
- Module media có thể tải ảnh xuống `group_post/`; log thường ra console. File
  log riêng phụ thuộc cách gọi/redirect, không tự coi là một đầu ra luôn có.

Runner chưa chuyển thành CSV, chưa clean để gán nhãn, chưa gọi LLM, chưa train.
`--no-comments` còn có thể chỉ lấy bài. Số liệu và năm ghi trong tên file không
chứng minh số lượng hay thời gian thu thực tế.

Runner là file điều phối, không chứa toàn bộ mã HTTP. Chuyển JSON → CSV và gán
nhãn đã là các script riêng trong nhánh gốc của project; không phải dự kiến tách
tất cả từ `scrape_ptit_groups.py`.

## 4. Bảy file Drive thuộc nhiều công đoạn

| File | Xuất phát từ công đoạn |
|---|---|
| `groups_data_phase2.json` | Snapshot thu thập thô; cấu trúc phù hợp writer PTIT. |
| `comments_phase2.csv` | Converter JSON → CSV và lọc. |
| `comments_phase2_cleaned.csv` | Cleaner lịch sử. |
| `predictions_glm_cache_phase2.json` | Gợi ý nhãn AI, không phải dữ liệu crawler trực tiếp. |
| `comments_labeled_phase2.csv` | Export các bản ghi đã có nhãn được chốt. |
| `comments_labeled_export.csv` | CSV nhãn Phase 1 đã có từ trước; không có raw/log tương ứng trong Drive. |
| `comments_labeled_merged.csv` | Gộp hai CSV nhãn và loại trùng chéo. |

Notebook thực nghiệm chỉ cần v7 và môi trường phù hợp; không cần chạy merge,
Argilla hoặc cung cấp Phase 1 để train từ v7. Công cụ gộp/đối soát giữ ở nhóm
quản lý phiên bản, không cần mở trong lượt trình bày chính.

Có một phụ thuộc cần nói chính xác: converter hiện lọc trùng với CSV Phase 1.
Nếu muốn tái hiện đúng 9.569 dòng CSV lịch sử thì cần dữ liệu tham chiếu đó.
Nếu chỉ minh họa làm phẳng vài bài bằng `flatten_posts()`, không cần Phase 1;
nhưng đầu ra minh họa ấy không được nhận là toàn bộ snapshot CSV lịch sử.
Không áp cache ID dựa vào vị trí dòng lên CSV mới đã đổi thứ tự/số dòng.

## 5. “Minh họa” là dữ liệu thật đã lưu, không phải tự tạo dữ liệu giả

Notebook 1 dùng snapshot Phase 2 thành viên cung cấp, có group ID, post ID và
permalink. Các bước chuyển/clean/nối metadata đã đối soát khớp. Chưa đối chiếu
trực tiếp từng comment với Facebook hiện tại, và chưa có log phiên bản/lệnh crawl
lịch sử. Một group trong raw còn không có trong cấu hình runner hiện tại.

Nếu chạy crawler live, code thực sự gửi HTTP tới Facebook và có thể nhận dữ liệu
thật khi phiên truy cập, quyền và endpoint còn hoạt động. Lượt này chưa chạy
live; không cam kết chạy được ngay hoặc trả lại đúng snapshot cũ. Dữ liệu nguồn
có thể thay đổi, các hạn chế phân trang/resume đã ghi ở tài liệu kiểm chứng.

Không khẳng định “chạy notebook 1 nhiều lần sẽ thu được v7” hoặc “hai thành viên
cào riêng rồi gộp là v7” khi chưa có căn cứ lịch sử. Cách chuyển phần trình bày:

> Phần đầu trình bày mã và dữ liệu đã lưu của một đợt thu thập. Phần thực nghiệm
> sử dụng bộ dữ liệu bình luận PTIT phiên bản v7 do nhóm cung cấp.

Nguồn/snapshot và phiên bản nhãn được ghi ngắn trong báo cáo; chi tiết đối soát
ở phụ lục. Phần chính không cần đọc mọi con số lịch sử, nhưng không vẽ quan hệ
sinh dữ liệu mà code hiện tại không chứng minh được.

## 6. Cấu trúc đề xuất

```text
external/
  facebook_scraper/                 # Giữ 7 module nền và tài liệu bàn giao
pipeline/
  collect/
    scrape_ptit_groups.py           # Runner tách ra ngoài bộ nền
    json_to_csv.py
  prepare/
    clean_for_labeling.py
    preprocess_for_model.py
  label/
    upload_to_argilla.py
    ai_assisted_spam_ham.py
    auto_accept_labels.py
    download_labeled_data.py
    label_utils.py
  modeling/
    training_utils.py
    experiments.py
  inference/predict.py
  modern/README.md                   # Giữ chỗ ý 4 bổ sung; chưa có mô hình
notebooks/
  01_thu_thap_va_gan_nhan.ipynb
  02_thuc_nghiem_va_demo.ipynb
tools/dataset_versions/
  merge_labeled_data.py
  audit_source.py
data/
  raw/
  intermediate/
  labeled/
  final/labeled_v7.csv
docs/
  report/
  presentation/
  provenance/
results/
```

Giữ toàn bộ module nền, không xóa file chỉ vì không gọi nhánh chức năng của nó:
`main.py` hiện vẫn import `post_scraper.py` và `single_post_image.py`. Khi runner
đặt bên ngoài, phải cập nhật cách tìm thư viện, `.env`, đường dẫn output và các
tests liên quan; chỉ di chuyển file sẽ làm import lỗi. Chưa nâng cấp lên main
GitHub mới vì đó là thay đổi nền khác phạm vi tái cấu trúc.

## 7. Bảng nguồn file gốc → hiện tại → đề xuất

Đối chiếu chỉ đọc: `master` và `cuong` cùng commit
`efc4fba2a6289936fdcd43f822698f251ee9e61f`, remote được kiểm tra lại ngày
05/11/2026. Không checkout/fetch hoặc sửa hai nhánh này. Không suy tên tác giả
cá nhân từ tên nhánh.

Quy ước `G/` = `facebook_post_comment_scraper_code/facebook_post_comment_scraper/`.

| Nguồn gốc | File hiện tại trên phi | File đề xuất | Loại thay đổi |
|---|---|---|---|
| Bộ thành viên bàn giao; runner PTIT riêng trong bộ đó | `G/scrape_ptit_groups.py` | `pipeline/collect/scrape_ptit_groups.py` | Chuyển vị trí, chỉnh đường dẫn/import; chưa tách thành nhiều file. |
| Repo GitHub đối chiếu và bộ bàn giao | `G/comment_scraper.py`, `G/group_post_scraper_v2.py`, `G/main.py`, `G/post_scraper.py`, `G/single_post_image.py`, `G/proxy_utils.py`, `G/facebook_ui.py` | Cùng tên trong `external/facebook_scraper/` | Giữ nguyên snapshot bàn giao; không gọi tất cả là bản upstream nguyên xi. |
| Tài liệu/cấu hình bộ bàn giao | `G/README*.md`, `G/requirements.txt`, `G/.env.example` | Cùng nhóm `external/facebook_scraper/` | Giữ nguồn/hướng dẫn; ghi chú cách gọi runner mới. |
| `master`/`cuong`: `convert_phase2_json.py` | `pipeline/step_01_collect/convert_phase2_json.py` | `pipeline/collect/json_to_csv.py` | Chuyển/đổi tên theo chức năng; bảo toàn quy tắc/filter và dấu vết nguồn. |
| `master`/`cuong`: `clean_data_facebook_spamham.py` | `pipeline/step_02_preprocess/clean_data_facebook_spamham.py` | `pipeline/prepare/clean_for_labeling.py` | Đổi tên làm rõ clean trước label; giữ các sửa đúng đã có trên phi. |
| Notebook gốc có tiền xử lý; phi đã tách/chỉnh thành module | `pipeline/step_02_preprocess/text_preprocessing.py` | `pipeline/prepare/preprocess_for_model.py` | Chuyển/đổi tên; không coi là file nguyên bản có sẵn trên master. |
| `master`/`cuong`: `upload_to_argilla.py` | `pipeline/step_03_label/upload_to_argilla.py` | `pipeline/label/upload_to_argilla.py` | Chuyển, cập nhật import. |
| `master`/`cuong`: `ai_assisted_spam_ham.py` | `pipeline/step_03_label/ai_assisted_spam_ham.py` | `pipeline/label/ai_assisted_spam_ham.py` | Chuyển, giữ luồng gợi ý AI. |
| `master`/`cuong`: `auto_accept_labels.py` | `pipeline/step_03_label/auto_accept_labels.py` | `pipeline/label/auto_accept_labels.py` | Chuyển, giữ kiểm tra ID/cache/fallback đã sửa. |
| `master`/`cuong`: `download_labeled_data.py` | `pipeline/step_03_label/download_labeled_data.py` | `pipeline/label/download_labeled_data.py` | Chuyển, giữ luồng export nhãn đã chốt. |
| Module bổ sung trên phi trong lượt ổn định pipeline | `pipeline/step_03_label/label_utils.py` | `pipeline/label/label_utils.py` | Chuyển helper ID/metadata; không có file này ở root nhánh gốc đã kiểm tra. |
| Notebook gốc có split/huấn luyện; phi đã bổ sung/tách helper | `pipeline/step_06_train_evaluate/training_utils.py` | `pipeline/modeling/training_utils.py` | Chuyển helper CV/threshold/sampling; không phải nguyên file có sẵn trên master. |
| Notebook hiện tại | Các hàm `make_pipe`, `fit_experiment`, `evaluate`, `boot_ci` trong notebook | `pipeline/modeling/experiments.py` | Tách hàm cần dùng lại; chưa di chuyển trong lượt đề xuất. Biểu đồ/bảng chính vẫn ở notebook. |
| Notebook gốc có `predict_spam`; bản hiện tại đã sửa tính nhất quán train/demo | Hàm `predict_spam` và lưu/tải bundle trong notebook | `pipeline/inference/predict.py` | Tách phần dự đoán cần dùng lại; không tách từ runner crawler. |
| `master`/`cuong`: `merge_labeled_data.py` | `pipeline/step_04_merge/merge_labeled_data.py` | `tools/dataset_versions/merge_labeled_data.py` | Chuyển công cụ tổng hợp dữ liệu, không gọi khi train trực tiếp từ v7. |
| Bổ sung trên phi để kiểm chứng nguồn Drive | `pipeline/step_01_collect/audit_source.py` | `tools/dataset_versions/audit_source.py` | Chuyển kiểm chứng; sửa API đường dẫn vì runner và thư viện nền tách chỗ. |
| Notebook gốc + các cải tiến hiện tại | `notebooks/spam-ham-comments.ipynb` | Hai notebook mô tả ở mục 8 | Tổ chức lại nội dung; bỏ module nhúng dài ở bản trình bày, import module repo. |
| Ý 4 bổ sung chưa làm | Heading/cell chờ hiện tại | `pipeline/modern/README.md` và mục chờ notebook 2 | Giữ chỗ; không tạo kết quả hoặc coi LLM gán nhãn là mô hình bổ sung đã hoàn thành. |

Các file `__init__.py`, requirements, runner tự động, tests và tài liệu sẽ cập
nhật theo đường dẫn khi thật sự tái cấu trúc. V7, các snapshot/hash và kết quả
thực nghiệm đã kiểm chứng phải được giữ nguyên.

## 8. Hai notebook phụ trách gì và tương ứng đề nào?

### Notebook 1 — thu thập và gán nhãn

1. Bài toán PTIT spam/ham, nguồn tham khảo của công cụ.
2. Group/post thực sự có trong raw; chỉ các hàm điều phối/lấy dữ liệu.
3. Đọc snapshot JSON thật, xem schema và vài record, không cần phiên Facebook.
4. Chạy `flatten_posts()` trên một phần dữ liệu để giải thích JSON → bảng.
5. Hiển thị clean trước/sau bằng hàm chuẩn bị gán nhãn.
6. Giải thích upload → AI suggestion → chốt/duyệt → export.
7. Minh họa một bản ghi/cache nối đúng ID từ snapshot lịch sử, không renumber
   CSV minh họa rồi áp cache cũ hoặc giả là đã gọi AI mới.
8. Phân biệt CSV có nhãn của nhánh minh họa với bản v7 dùng thực nghiệm.

Phục vụ ý 1 (thu thập), một phần ý 2 (clean), và giải thích cách tạo nhãn. Mặc
định dùng dữ liệu đã lưu; chạy live là một chế độ riêng nếu sau này cần và đã
kiểm tra truy cập. Notebook này không tự tạo lại toàn bộ v7.

### Notebook 2 — thực nghiệm và demo

1. Nạp v7 từ local hoặc nguồn đã chốt, kiểm tra schema/hash/phân bố.
2. Tiền xử lý cho mô hình và đối chứng giữ/bỏ stop words — ý 2.
3. Chia train/test và folds theo nhóm — ý 4.
4. BoW/TF-IDF, fit trong train/fold — ý 3.
5. NB/SVM, tìm tham số/chọn trên CV — ý 4.
6. Mục mô hình hiện đại — giữ chỗ cho ý 4 bổ sung, triển khai sau.
7. Accuracy/Precision/Recall/F1/Confusion Matrix và so sánh — ý 5.
8. Phân tích lỗi; phần thêm như AP/CI/OOV để sau phần bắt buộc.
9. Lưu/tải mô hình, nhập comment mới, chạy cùng tiền xử lý và dự đoán — demo.
10. Nhận xét từ output thật và giới hạn nguồn/nhãn.

Không cần Phase 1, merge hoặc Argilla để notebook 2 train từ v7. LLM ở notebook 1
là công cụ tạo gợi ý nhãn, không thay cho mô hình hiện đại bổ sung ở notebook 2.

## 9. Kiểm tra cần làm khi thực hiện cấu trúc này

- Cập nhật imports, đường dẫn và CLI, không checkout/chỉnh nhánh ngoài phi.
- Đối chiếu module gốc để không mất sửa lỗi trước đây; giữ nguồn bộ crawler.
- Kiểm tra notebook 1 đọc đúng snapshot và không áp nhầm cache sau đổi dòng.
- Kiểm tra notebook 2 chạy với v7 mà không cần gọi các bước gán nhãn/gộp.
- Xác nhận train/demo cùng tiền xử lý, kết quả và dataset không bị đổi do chuyển file.
- Chạy lại các kiểm tra liên quan và notebook sau thay đổi code; giữ mục chờ ý 4.

Lượt hiện tại chỉ tạo bản giải đáp/sơ đồ và bằng chứng đối chiếu upstream; chưa
thực hiện các thay đổi code/notebook trong sơ đồ.
