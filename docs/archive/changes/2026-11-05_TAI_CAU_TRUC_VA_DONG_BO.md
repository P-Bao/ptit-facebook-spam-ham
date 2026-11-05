# Tái cấu trúc và đồng bộ chú thích — 05/11/2026

## Thay đổi chính

Từ bản `7415098`, project chuyển sang thư mục chức năng và hai notebook trình
bày. Không thay dataset v7, snapshot Drive, grid NB/SVM hoặc các phép đo để tăng
điểm. Ý 4 bổ sung vẫn có mục chờ, chưa triển khai mô hình hiện đại.

| Trước | Sau | Lý do/tác dụng |
|---|---|---|
| `facebook_post_comment_scraper_code/facebook_post_comment_scraper/` chứa cả runner và bộ nền | `external/facebook_scraper/` giữ bộ nền; runner ở `pipeline/collect/scrape_ptit_groups.py` | Phân biệt công cụ tham khảo và phần điều phối PTIT; 12 file nền/tài liệu giữ nguyên từng byte. |
| `pipeline/step_01_collect/convert_phase2_json.py` | `pipeline/collect/json_to_csv.py` | Tên theo chức năng, chưa có nhãn; giữ lựa chọn lọc trùng dữ liệu tham chiếu. |
| `step_02_preprocess/clean_data_facebook_spamham.py` | `pipeline/prepare/clean_for_labeling.py` | Làm rõ đây là clean trước gán nhãn; giữ text gốc. |
| `step_02_preprocess/text_preprocessing.py` | `pipeline/prepare/preprocess_for_model.py` | Làm rõ tiền xử lý cho train/demo, gồm tách từ và stop words. |
| `step_03_label/*.py` | `pipeline/label/` | Giữ tên năm module và logic ID/AI/chốt/export; cập nhật import/CLI. |
| `step_05_features/spam_features.py` | `pipeline/features/spam_features.py` | Module FastText được notebook/modeling import trực tiếp. |
| `step_06_train_evaluate/training_utils.py` | `pipeline/modeling/training_utils.py` | Helper CV, sampling, ngưỡng và dự đoán giữ nguyên logic. |
| Hàm tạo pipeline/fit/đánh giá/bootstrap trong notebook | `pipeline/modeling/experiments.py` | Nhận rõ X/y/folds/seed/cache/jobs thay cho biến global của notebook. |
| Hàm `predict_spam()` trong notebook | `pipeline/inference/predict.py` | Một đường raw text → preprocess → predict, dùng lại khi demo. |
| `step_04_merge/merge_labeled_data.py`, `step_01_collect/audit_source.py` | `tools/dataset_versions/` | Công cụ quản lý/đối soát phiên bản, không phải dependency train v7. |
| Một notebook 51 cell, 27 code cell có nhiều module nhúng | Notebook 01: 15 cell/7 code; notebook 02: 36 cell/19 code | Tách trình bày nguồn và thực nghiệm; không nhúng lại module bằng writefile. |
| Dataset/snapshot local ở hai folder flat | `data/raw`, `intermediate`, `labeled`, `final` | Sắp theo loại dữ liệu, giữ nguyên byte/hash và tên snapshot lịch sử. |
| `docs/references/` | `docs/provenance/` | Nguồn, giới hạn và đối chiếu phiên bản tập trung riêng. |

Bảng đường dẫn đầy đủ: `results/restructure/mapping.json`. Nguồn từ nhánh gốc và
phân biệt file có sẵn/file đã tách trên phi: bản kế hoạch
`docs/plans/2026-11-05_GIAI_DAP_VA_SO_DO_TAI_CAU_TRUC.md` (lưu nguyên như hồ sơ trước triển khai).

## Đồng bộ tên và chú thích

Các module chức năng của project có header thống nhất: đường dẫn hiện tại,
vai trò, **Đầu vào / Xử lý / Đầu ra / Liên quan**. Đã cập nhật import, tên lệnh
`python -m`, đường dẫn mặc định và hướng dẫn chuyển bước. Không thay tên snapshot
chỉ để xóa dấu vết nguồn; không nhận file CSV nhánh thu thập là v7.

Đường dẫn mặc định của luồng mới được nối khớp:

```text
runner → data/raw/posts.json
converter → data/intermediate/comments.csv
cleaner → data/intermediate/comments_cleaned.csv
công cụ nhãn → cache/intermediate và data/labeled/comments_labeled.csv
```

Output nhãn mới dùng `comments_labeled.csv`, output merge dùng `comments_merged.csv`;
các tên snapshot lịch sử được giữ riêng để tránh ghi đè mặc định.

Runner tìm bộ nền trong `external/facebook_scraper/`, `.env` ở gốc repo. Có sửa
in UTF-8 cho CLI Windows, chú thích mốc năm là giới hạn dưới, và giải thích
`--force-all` đúng với logic còn giữ cursor/ID. Không tuyên bố đã sửa phục hồi
checkpoint hoặc phân trang replies; giới hạn crawler vẫn ở tài liệu nguồn.

Các README bên trong bộ nền là snapshot tài liệu cũ, giữ nguyên để đối chiếu.
Hướng dẫn gọi hiện tại ở `external/README.md`, root README và `pipeline/collect/README.md`.

## Hai notebook và phụ thuộc

- Notebook 01 đọc 5 snapshot có hash, minh họa làm phẳng 3 bài, clean và nối nhãn
  với ID lịch sử; xuất CSV xem thử/manifest. Không gọi Facebook/LLM/Argilla.
- Notebook 02 chỉ cần v7 và module repo; không yêu cầu Phase 1, snapshot Drive
  hoặc chạy merge/label. Giữ huấn luyện và đánh giá thực, có save/load–predict.
- Do không nộp riêng notebook, các module được import từ repo; không còn mô tả
  bản mới là notebook tự chứa tách riêng. `scripts/run_notebook.py` chọn `--notebook 01/02`.
- `pipeline/modern/README.md` và mục chờ notebook 02 giữ ý 4 bổ sung. LLM nhãn
  không thay cho phần mô hình hiện đại.

## Kiểm chứng

- **44 test đạt** sau chuyển đường dẫn. Lượt đầu phát hiện fixture còn dùng vị
  trí cũ và lỗi mã hóa trợ giúp crawler; đã sửa rồi chạy lại suite thành công.
- So sánh AST với `7415098`: tiền xử lý model, lớp FastText và helper train giữ
  nguyên. Cả 6 factory BoW/TF-IDF/FastText × NB/SVM giữ estimator/tham số/grid/jobs.
  Bằng chứng `results/restructure/equivalence.json`.
- Kiểm tra cache tải sai hash không ghi đè file, payload sai không được lưu;
  train–lưu–tải–predict bằng module mới; CLI crawler `--help` không ghi dữ liệu.
- Chạy notebook 01 bằng snapshot thật thành công; lần chạy cuối và notebook 02
  có bằng chứng ở `results/verified_run/`, tổng hợp ở `results/restructure/verification.json`.
- Notebook 01: **7/7 cell code**, 2,16 giây. Notebook 02: **19/19 cell code**,
  1.469,28 giây CPU (khoảng 24,5 phút). Không có lỗi thực thi cell.
- **Cả 16 dòng thí nghiệm giữ nguyên điểm số và tham số**, chỉ thời gian đo khác.
  Mô hình chọn theo CV vẫn là TF-IDF + SVM, giữ stop words: Accuracy 0,9419,
  Precision 0,8818, Recall 0,6278, F1 0,7334. Không coi tái cấu trúc là tăng điểm.
- Đối soát JSON/CSV/clean/gộp tới v7 sau chuyển folder vẫn phải khớp số liệu cũ;
  v7 giữ SHA-256 `7305d712996da3ff0dce1ea4142844b7629e9880bcac47ee38934f0b6e6cfbee`.
- Không chạy crawl live hoặc gọi dịch vụ gán nhãn trong lượt tái cấu trúc. Không
  ghi kết quả minh họa thành lần thu/gán nhãn mới.

Các kết quả của cấu trúc cũ được lưu riêng trong `results/history/7415098/` trước
khi công bố bằng chứng mới. Cảnh báo dependency/giao diện tiến độ nếu có được
ghi theo output thực tế; không dùng cảnh báo để suy thành lỗi cell.

## Git

Chỉ cập nhật nhánh `phi`. Ref push duy nhất `HEAD:refs/heads/phi`, không force.
Kiểm tra `master`/`cuong` trước và sau push; không checkout/chỉnh hai nhánh này.
