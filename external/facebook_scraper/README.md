# Hướng Dẫn Sử Dụng Facebook Group Scraper 🕷️

Bộ công cụ cào tự động bài viết và bình luận từ các Facebook Group (hỗ trợ nhóm riêng tư/kín), tối ưu hóa để chạy nền (`nohup`), lưu toàn bộ kết quả vào một file duy nhất và tự động phục hồi phiên cào khi Cookie hết hạn.

---

## 📌 Các Tính Năng Nổi Bật

1. **Quản lý trạng thái thông minh qua `state.json`**:
   - Lưu tiến độ sau **từng trang cào** (gồm `page_num`, `cursor` phân trang của Facebook và số bài đã cào).
   - **Tự động bỏ qua nhóm đã hoàn thành** (`status: "completed"`).
   - **Tự động tiếp tục từ trang đang dở** (`status: "in_progress"`) với đúng token phân trang `cursor`, không phải quét lại từ trang 1.
2. **Tự động phát hiện Cookie hết hạn**:
   - Khi Facebook trả về lỗi hết hạn phiên (`error 1357001`), script sẽ **lập tức dừng an toàn**, lưu lại cursor hiện tại vào `state.json` và in cảnh báo rõ ràng.
3. **Lưu kết quả an toàn (Atomic Write)**:
   - Toàn bộ kết quả được ghi vào **1 file duy nhất** (mặc định: `ptit_groups_data_2025.json`).
   - Ghi thông qua file tạm `.tmp` rồi đổi tên nguyên tử (`os.replace`), đảm bảo file JSON không bao giờ bị hỏng hoặc mất dữ liệu dù bị ngắt tiến trình đột ngột.
4. **Hỗ trợ xuất riêng text của comment**:
   - Mỗi bài viết có trường `"comment_texts": [...]` chứa danh sách chuỗi text thuần túy.
   - Hỗ trợ xuất thêm file riêng biệt (ví dụ: `ptit_comments_text_only.txt` - mỗi dòng 1 bình luận) phục vụ xử lý ngôn ngữ tự nhiên (NLP).

---

## 🛠️ Cài Đặt Môi Trường

### 1. Yêu cầu
- Python 3.8 trở lên.
- Các thư viện cần thiết: `requests`, `python-dotenv`.

### 2. Cài đặt thư viện
```bash
pip install -r requirements.txt
```

### 3. Cấu hình xác thực (`.env`)
Tạo hoặc chỉnh sửa file `.env` tại thư mục dự án (tham khảo mẫu `.env.example`):

```env
# Cookie Facebook (tài khoản đã tham gia các nhóm cần cào)
FB_COOKIES="c_user=...; xs=...; datr=...; sb=...;"

# Token GraphQL fb_dtsg
FB_DTSG="NAf..."

# (Tùy chọn) Proxy nếu cần
# STATIC_PROXY=http://username:password@proxy-host:port
```

---

## 🚀 Hướng Dẫn Vận Hành

### 1. Lệnh chạy ngầm với `nohup` (Khuyên dùng)

```bash
nohup python3 -u scrape_ptit_groups.py \
  --year 2025 \
  --limit 500 \
  --output ptit_groups_data_2025.json \
  --comments-output ptit_comments_text_only.txt \
  --state-file state.json \
  > scraper.log 2>&1 & echo $! > scraper.pid
```

> **Ý nghĩa các tham số:**
> - `python3 -u`: Bắt buộc để xả log ra file tức thì (unbuffered), giúp lệnh `tail -f` hiển thị thời gian thực.
> - `--year 2025`: Chỉ cào các bài viết được đăng từ ngày `01/01/2025` đến nay.
> - `--limit 500`: Giới hạn tối đa số bài viết thu thập trên mỗi nhóm.
> - `--output ptit_groups_data_2025.json`: File kết quả bài viết duy nhất.
> - `--comments-output ptit_comments_text_only.txt`: File riêng lưu text bình luận (mỗi dòng 1 comment).
> - `--state-file state.json`: File theo dõi tiến độ và cursor phân trang.
> - `> scraper.log 2>&1 &`: Chuyển toàn bộ output vào `scraper.log` và chạy ngầm.
> - `echo $! > scraper.pid`: Lưu Process ID để tiện quản lý tiến trình.

---

### 2. Giám sát & Quản lý tiến trình

- **Xem log trực tiếp theo thời gian thực:**
  ```bash
  tail -f scraper.log
  ```
  *(Nhấn `Ctrl + C` để thoát màn hình xem log mà không làm tắt tiến trình).*

- **Xem 50 dòng log gần nhất:**
  ```bash
  tail -n 50 scraper.log
  ```

- **Kiểm tra trạng thái tiến độ các nhóm trong `state.json`:**
  ```bash
  cat state.json
  ```

- **Kiểm tra tiến trình có đang hoạt động:**
  ```bash
  kill -0 $(cat scraper.pid) 2>/dev/null && echo "✅ Đang chạy" || echo "❌ Đã dừng"
  ```

- **Dừng tiến trình khi cần:**
  ```bash
  kill $(cat scraper.pid)
  ```

---

## 🔄 Cách Xử Lý Khi Cookie Hết Hạn

Khi tài khoản Facebook hết phiên hoặc cần đăng nhập lại, log sẽ hiển thị thông báo:
```text
🚨 DỪNG TIẾN TRÌNH: PHIÊN ĐĂNG NHẬP FACEBOOK ĐÃ HẾT HẠN!
👉 Dừng tại nhóm: [Tên nhóm] (ID: ...).
👉 Điểm tiếp tục (trang & cursor) đã được lưu an toàn vào state.json.
```

**Các bước tiếp tục:**
1. Lấy chuỗi Cookie và `fb_dtsg` mới từ trình duyệt.
2. Mở file `.env` và cập nhật giá trị mới.
3. Chạy lại câu lệnh `nohup` ban đầu.
4. **Script sẽ tự động:**
   - Bỏ qua các nhóm đã hoàn thành (`status: "completed"`).
   - Tự động tiếp tục ngay từ trang đang dở của nhóm bị gián đoạn với token `cursor` đã lưu trong `state.json`.

---

## 📋 Danh Sách Tham Số Dòng Lệnh (`scrape_ptit_groups.py`)

| Tham số | Mặc định | Mô tả |
| :--- | :--- | :--- |
| `--year` | Năm hiện tại | Năm cần cào bài viết (lọc từ 01/01 của năm đến nay) |
| `--limit` | `500` | Số bài tối đa cần lấy trên mỗi nhóm |
| `--min-comments` | `0` | Lọc bài viết có số bình luận tối thiểu (0 = lấy tất cả) |
| `--output` | `ptit_groups_data_2025.json` | Đường dẫn file kết quả bài viết (`.json` hoặc `.jsonl`) |
| `--comments-output` | `None` | Đường dẫn file riêng chỉ lưu text comment (`.txt`, `.json`, `.jsonl`) |
| `--state-file` | `state.json` | Đường dẫn file theo dõi trạng thái và vị trí phân trang |
| `--force-all` | `False` | Ép cào lại tất cả các nhóm từ đầu (bỏ qua `state.json`) |
| `--no-comments` | `False` | Bỏ qua việc cào bình luận (chỉ cào bài viết để tăng tốc) |
| `--delay` | `2.0` | Thời gian nghỉ (giây) giữa các request để bảo vệ tài khoản |
