"""
Script cào bài viết và bình luận từ 16 Group PTIT trong năm chỉ định (chạy tối ưu với nohup)
- Toàn bộ kết quả được ghi vào 1 FILE DUY NHẤT (.json hoặc .jsonl)
- Hỗ trợ xuất riêng text comment ra 1 file
- Cập nhật tiến độ liên tục vào file state.json sau mỗi trang
- TỰ ĐỘNG TIẾP TỤC TỪ TRANG ĐANG DỞ khi Cookie hết hạn và được cập nhật lại
"""

import os
import sys
import time
import json
import argparse
from datetime import datetime
from dotenv import load_dotenv

# Load biến môi trường từ .env
load_dotenv()

# Import các module từ repo
import group_post_scraper_v2
import comment_scraper
from group_post_scraper_v2 import FacebookSessionExpiredException
from main import fetch_comments_for_post


def parse_cookies(cookie_string):
    """Parse cookie string in format 'key1=value1;key2=value2' into dictionary"""
    cookies = {}
    if not cookie_string:
        return cookies
    for cookie in cookie_string.split(';'):
        cookie = cookie.strip()
        if '=' in cookie:
            key, value = cookie.split('=', 1)
            cookies[key.strip()] = value.strip()
    return cookies


# Danh sách 16 Group PTIT (đã khử trùng lặp link share và resolve group ID chuẩn)
PTIT_GROUPS = [
    {
        "name": "2K5 PTIT",
        "id": "450310163844833",
        "urls": [
            "https://www.facebook.com/share/g/1aPFxrTSG4/",
            "https://www.facebook.com/groups/450310163844833/"
        ]
    },
    {
        "name": "Góc Thông Tin PTIT",
        "id": "408571091380061",
        "urls": ["https://www.facebook.com/groups/408571091380061/"]
    },
    {
        "name": "D26 (2008) PTIT",
        "id": "1130813927934700",
        "urls": ["https://www.facebook.com/groups/1130813927934700/"]
    },
    {
        "name": "D26 PTIT Hà Nội",
        "id": "873994222101062",
        "urls": ["https://www.facebook.com/groups/d26ptithanoi/"]
    },
    {
        "name": "D26 Học Viện Công Nghệ Bưu Chính Viễn Thông (2K8 PTIT)",
        "id": "1605563914144178",
        "urls": ["https://www.facebook.com/groups/1605563914144178/"]
    },
    {
        "name": "2K8 PTIT - D26",
        "id": "1379963163860326",
        "urls": ["https://www.facebook.com/groups/1379963163860326/"]
    },
    {
        "name": "PTIT Confessions",
        "id": "1210825580901941",
        "urls": ["https://www.facebook.com/groups/confessions.ptit/"]
    },
    {
        "name": "Sinh Viên PTIT giúp nhau việc làm",
        "id": "541958802253916",
        "urls": ["https://www.facebook.com/groups/541958802253916/"]
    },
    {
        "name": "Góc Thông Tin PTIT (2)",
        "id": "584397217391365",
        "urls": ["https://www.facebook.com/groups/584397217391365/"]
    },
    {
        "name": "Tuyển Dụng PTIT",
        "id": "639651753033406",
        "urls": ["https://www.facebook.com/groups/tuyendungPtit/"]
    },
    {
        "name": "2K4 PTIT",
        "id": "327017909548703",
        "urls": ["https://www.facebook.com/groups/2k4ptit/"]
    },
    {
        "name": "Sinh viên PTIT",
        "id": "3945309109133292",
        "urls": ["https://www.facebook.com/groups/sinhvienptithn/"]
    },
    {
        "name": "Hội Anti PTIT",
        "id": "1570296811505630",
        "urls": ["https://www.facebook.com/groups/groupantiptit/"]
    },
    {
        "name": "D24 Học Viện Công Nghệ Bưu Chính Viễn Thông",
        "id": "2608226055999685",
        "urls": ["https://www.facebook.com/groups/d24hocviencongnghebuuchinhvienthong/"]
    },
    {
        "name": "Cộng đồng sinh viên PTIT (Dự phòng)",
        "id": "301046308355906",
        "urls": ["https://www.facebook.com/groups/301046308355906/"]
    },
    {
        "name": "2K8 Học viện Công nghệ Bưu chính Viễn thông PTIT",
        "id": "864898861651241",
        "urls": ["https://www.facebook.com/groups/864898861651241/"]
    }
]


def log_message(msg):
    """In log kèm timestamp và flush tức thì phục vụ tail -f"""
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now_str}] {msg}", flush=True)


def extract_comment_texts(comments):
    """Trích xuất mảng text thuần túy từ danh sách comments và replies"""
    texts = []
    for c in comments:
        txt = (c.get("text") or "").strip()
        if txt:
            texts.append(txt)
        for r in c.get("replies", []):
            rtxt = (r.get("text") or "").strip()
            if rtxt:
                texts.append(rtxt)
    return texts


class StateManager:
    """Quản lý trạng thái và vị trí phân trang (cursor, page_num) của từng nhóm trong file state.json"""

    def __init__(self, state_file="state.json", output_file="ptit_groups_data_2025.json"):
        self.state_file = os.path.abspath(state_file)
        self.output_file = os.path.abspath(output_file)
        self.current_group_id = None
        self.groups = {}
        self.load_state()

    def load_state(self):
        # 1. Đọc từ file state.json nếu đã có
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.current_group_id = data.get("current_group_id")
                    self.groups = data.get("groups", {})
                log_message(f"📂 Đã nạp file trạng thái {os.path.basename(self.state_file)}: Đã có {len(self.groups)} nhóm được ghi nhận.")
            except Exception as e:
                log_message(f"⚠️ Cảnh báo đọc file {self.state_file}: {e}")

        # 2. Tự động nhận diện thêm từ file kết quả ptit_groups_data_2025.json nếu nhóm chưa có trong state.json
        # Ví dụ: Nhóm 2K5 PTIT đã có 394 bài từ lần chạy trước -> tự động đánh dấu completed để không cào trùng
        if os.path.exists(self.output_file):
            try:
                with open(self.output_file, "r", encoding="utf-8") as f:
                    records = json.load(f)
                    if isinstance(records, list):
                        group_counts = {}
                        for r in records:
                            gid = str(r.get("group_id"))
                            gname = r.get("group_name", "Unknown")
                            if gid not in group_counts:
                                group_counts[gid] = {"name": gname, "count": 0}
                            group_counts[gid]["count"] += 1

                        for gid, info in group_counts.items():
                            if gid not in self.groups and info["count"] > 0:
                                self.groups[gid] = {
                                    "name": info["name"],
                                    "status": "completed",
                                    "page_num": 135,
                                    "cursor": None,
                                    "posts_scraped": info["count"],
                                    "last_updated": "đã có sẵn trong file kết quả"
                                }
            except Exception as e:
                log_message(f"⚠️ Cảnh báo quét file kết quả để lập state: {e}")

        # In tóm tắt trạng thái các nhóm
        if self.groups:
            log_message("📋 Trạng thái tiến độ hiện tại:")
            for gid, info in self.groups.items():
                st = info.get("status", "unknown")
                p = info.get("page_num", 1)
                c = info.get("posts_scraped", 0)
                cur_text = "có cursor tiếp tục" if info.get("cursor") else "không cursor"
                log_message(f"   • {info.get('name', 'Unknown')} (ID: {gid}): [{st.upper()}] - {c} bài | Trang: {p} ({cur_text})")

    def get_group_state(self, group_id):
        return self.groups.get(str(group_id), {})

    def is_group_completed(self, group_id):
        return self.groups.get(str(group_id), {}).get("status") == "completed"

    def set_current_group(self, group_id):
        self.current_group_id = str(group_id)
        self.save_state()

    def update_group_page(self, group_id, group_name, page_num, next_cursor, posts_count, is_finished=False):
        """Cập nhật trạng thái sau mỗi trang để nếu ngắt giữa chừng có thể tiếp tục ngay từ cursor này"""
        gid = str(group_id)
        status = "completed" if is_finished else "in_progress"
        self.groups[gid] = {
            "name": group_name,
            "status": status,
            "page_num": page_num,
            "cursor": None if is_finished else next_cursor,
            "posts_scraped": posts_count,
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        self.current_group_id = gid
        self.save_state()

    def mark_group_completed(self, group_id, group_name, posts_count):
        """Đánh dấu nhóm đã hoàn thành toàn bộ"""
        gid = str(group_id)
        current = self.groups.get(gid, {})
        self.groups[gid] = {
            "name": group_name,
            "status": "completed",
            "page_num": current.get("page_num", 1),
            "cursor": None,
            "posts_scraped": posts_count,
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        self.save_state()

    def save_state(self):
        try:
            temp_path = self.state_file + ".tmp"
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump({
                    "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "current_group_id": self.current_group_id,
                    "groups": self.groups
                }, f, ensure_ascii=False, indent=2)
            os.replace(temp_path, self.state_file)
        except Exception as e:
            log_message(f"⚠️ Lỗi lưu file state: {e}")


class SingleFileWriter:
    """Quản lý lưu toàn bộ kết quả vào 1 file duy nhất (hỗ trợ cả .json và .jsonl) an toàn khi chạy background"""

    def __init__(self, output_path):
        self.output_path = os.path.abspath(output_path)
        self.is_jsonl = self.output_path.endswith(".jsonl")
        self.existing_post_ids = set()
        self.all_records = []
        
        # Tạo thư mục chứa file nếu chưa có
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        
        # Đọc dữ liệu cũ nếu file đã tồn tại để tránh cào trùng bài (hỗ trợ resume)
        if os.path.exists(self.output_path):
            try:
                if self.is_jsonl:
                    with open(self.output_path, "r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if line:
                                item = json.loads(line)
                                if "post_id" in item:
                                    self.existing_post_ids.add(str(item["post_id"]))
                else:
                    with open(self.output_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            self.all_records = data
                            for item in data:
                                if isinstance(item, dict) and "post_id" in item:
                                    self.existing_post_ids.add(str(item["post_id"]))
                log_message(f"📂 Đã đọc file kết quả cũ: tìm thấy {len(self.existing_post_ids)} bài viết đã cào trước đó.")
            except Exception as e:
                log_message(f"⚠️ Cảnh báo khi đọc file kết quả cũ: {e}")

    def is_already_saved(self, post_id):
        return str(post_id) in self.existing_post_ids

    def save_records(self, new_records):
        """Lưu danh sách bài viết mới vào file duy nhất"""
        if not new_records:
            return

        for record in new_records:
            pid = str(record.get("post_id"))
            self.existing_post_ids.add(pid)

        if self.is_jsonl:
            with open(self.output_path, "a", encoding="utf-8") as f:
                for r in new_records:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
                f.flush()
        else:
            self.all_records.extend(new_records)
            # Ghi ra file tạm rồi đổi tên (atomic write) để đảm bảo file JSON không bao giờ bị hỏng nếu bị ngắt tiến trình
            temp_path = self.output_path + ".tmp"
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(self.all_records, f, ensure_ascii=False, indent=2)
            os.replace(temp_path, self.output_path)


class CommentsOnlyWriter:
    """Tùy chọn: Ghi riêng text của comment ra 1 file chuyên biệt (.txt, .json, hoặc .jsonl)"""

    def __init__(self, output_path=None):
        self.output_path = os.path.abspath(output_path) if output_path else None
        if not self.output_path:
            return
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        self.is_txt = self.output_path.endswith(".txt")
        self.is_jsonl = self.output_path.endswith(".jsonl")
        self.all_texts = []
        if not self.is_txt and not self.is_jsonl and os.path.exists(self.output_path):
            try:
                with open(self.output_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        self.all_texts = data
            except Exception:
                self.all_texts = []

    def append_comments(self, group_name, post_id, comment_texts):
        if not self.output_path or not comment_texts:
            return

        if self.is_txt:
            # Ghi mỗi comment trên 1 dòng
            with open(self.output_path, "a", encoding="utf-8") as f:
                for t in comment_texts:
                    clean_line = " ".join(t.split())
                    f.write(clean_line + "\n")
                f.flush()
        elif self.is_jsonl:
            with open(self.output_path, "a", encoding="utf-8") as f:
                for t in comment_texts:
                    f.write(json.dumps({"group_name": group_name, "post_id": post_id, "text": t}, ensure_ascii=False) + "\n")
                f.flush()
        else:
            # File .json mảng string
            self.all_texts.extend(comment_texts)
            temp_path = self.output_path + ".tmp"
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(self.all_texts, f, ensure_ascii=False, indent=2)
            os.replace(temp_path, self.output_path)


def run_scraper():
    parser = argparse.ArgumentParser(description="Scrape Facebook Group posts for current year into a single file with state.json tracking.")
    parser.add_argument("--year", type=int, default=datetime.now().year, help="Năm cần cào (mặc định: năm hiện tại)")
    parser.add_argument("--limit", type=int, default=500, help="Số bài tối đa mỗi group (mặc định: 500)")
    parser.add_argument("--min-comments", type=int, default=0, help="Số bình luận tối thiểu (mặc định: 0)")
    parser.add_argument("--output", type=str, default="ptit_groups_data_2025.json", help="Đường dẫn file kết quả duy nhất (.json hoặc .jsonl)")
    parser.add_argument("--comments-output", type=str, default=None, help="Tùy chọn: Đường dẫn file riêng CHỈ LƯU TEXT COMMENT (.txt, .json, .jsonl)")
    parser.add_argument("--state-file", type=str, default="state.json", help="File lưu trạng thái các nhóm và vị trí trang dở (mặc định: state.json)")
    parser.add_argument("--force-all", action="store_true", default=False, help="Cào lại tất cả các nhóm từ đầu (bỏ qua state.json)")
    parser.add_argument("--cookies", type=str, default="", help="Chuỗi Cookie Facebook")
    parser.add_argument("--fb-dtsg", type=str, default="", help="Token fb_dtsg")
    parser.add_argument("--no-comments", action="store_true", help="Không cào bình luận (chỉ cào bài viết để chạy nhanh hơn)")
    parser.add_argument("--delay", type=float, default=2.0, help="Thời gian nghỉ (giây) giữa các request (mặc định: 2.0s)")
    args = parser.parse_args()

    # 1. Xác thực: Lấy Cookie & FB_DTSG
    cookie_str = args.cookies or os.getenv("FB_COOKIES", "").strip()
    fb_dtsg = args.fb_dtsg or os.getenv("FB_DTSG", "").strip()

    if not cookie_str:
        log_message("⚠️ CẢNH BÁO: Chưa có Cookie Facebook!")
        log_message("Các nhóm PTIT đều là nhóm riêng tư, bạn cần Cookie từ tài khoản đã tham gia nhóm.")
        if sys.stdin.isatty():
            choice = input("Bạn có muốn nhập Cookie trực tiếp bây giờ không? (y/n): ").strip().lower()
            if choice == 'y':
                cookie_str = input("Dán Cookie vào đây: ").strip()
                fb_dtsg = input("Dán fb_dtsg vào đây (nếu có): ").strip()
        else:
            log_message("Đang chạy background (non-interactive): Vui lòng cung cấp Cookie qua file .env hoặc tham số --cookies.")

    cookies = parse_cookies(cookie_str) if cookie_str else {}

    # Khởi tạo writer ghi vào 1 file duy nhất và StateManager
    writer = SingleFileWriter(args.output)
    comments_writer = CommentsOnlyWriter(args.comments_output)
    state_manager = StateManager(args.state_file, args.output)

    # 2. Tính timestamp bắt đầu của năm
    start_of_year = datetime(args.year, 1, 1, 0, 0, 0)
    since_timestamp = int(start_of_year.timestamp())

    log_message("=" * 65)
    log_message(f"🎯 Bắt đầu cào dữ liệu cho {len(PTIT_GROUPS)} nhóm PTIT")
    log_message(f"📅 Khoảng thời gian: Từ {start_of_year.strftime('%Y-%m-%d %H:%M:%S')} (năm {args.year}) đến nay")
    log_message(f"⏱️ Timestamp mốc: {since_timestamp}")
    log_message(f"📁 File kết quả bài viết: {writer.output_path}")
    log_message(f"⚙️ File trạng thái: {state_manager.state_file}")
    if comments_writer.output_path:
        log_message(f"💬 File riêng chỉ lưu text comment: {comments_writer.output_path}")
    log_message(f"📊 Giới hạn tối đa: {args.limit} bài/nhóm | Min comments: {args.min_comments}")
    log_message("=" * 65)

    total_scraped_all_groups = len(writer.existing_post_ids)

    # 3. Lặp qua từng nhóm
    for idx, grp in enumerate(PTIT_GROUPS, 1):
        group_id = grp["id"]
        group_name = grp["name"]

        group_state = state_manager.get_group_state(group_id)
        status = group_state.get("status")

        # Kiểm tra nếu nhóm đã hoàn thành trước đó -> Tự động bỏ qua
        if not args.force_all and status == "completed":
            post_count = group_state.get("posts_scraped", 0)
            log_message(f"[{idx}/{len(PTIT_GROUPS)}] ⏭️ [BỎ QUA NHÓM ĐÃ XONG]: {group_name} (ID: {group_id}) - Đã hoàn tất ({post_count} bài).")
            continue

        # Xác định điểm tiếp tục (Resume từ cursor & page_num)
        resume_cursor = group_state.get("cursor")
        resume_page = group_state.get("page_num", 1)
        existing_group_posts = group_state.get("posts_scraped", 0)

        if resume_cursor:
            start_cursor = resume_cursor
            start_page = resume_page + 1
            log_message(f"[{idx}/{len(PTIT_GROUPS)}] 🔄 [TIẾP TỤC TRANG DỞ]: {group_name} (ID: {group_id})")
            log_message(f"   ↳ Tiếp tục từ trang {start_page} với cursor đã lưu ({existing_group_posts} bài đã cào trước đó)")
        else:
            start_cursor = None
            start_page = 1
            log_message(f"[{idx}/{len(PTIT_GROUPS)}] 🚀 [BẮT ĐẦU NHÓM MỚI]: {group_name} (ID: {group_id})")

        state_manager.set_current_group(group_id)

        # Cập nhật context cho scraper
        group_post_scraper_v2.GROUP_ID = group_id
        group_post_scraper_v2.GROUP_NAME = group_name
        group_post_scraper_v2.HEADERS["referer"] = f"https://www.facebook.com/groups/{group_id}/"
        group_post_scraper_v2.COOKIES = cookies
        group_post_scraper_v2.FB_DTSG = fb_dtsg
        comment_scraper.FB_DTSG = fb_dtsg

        group_records_count = existing_group_posts

        # Callback xử lý lưu batch kèm bình luận vào file duy nhất
        def on_batch(batch_posts, total_so_far, total_limit):
            nonlocal group_records_count, total_scraped_all_groups
            processed_records = []

            for post in batch_posts:
                post_id = post.get("post_id")
                if not post_id:
                    continue

                if writer.is_already_saved(post_id):
                    log_message(f"  ⏭️ Bỏ qua bài đã có trong file: {post_id}")
                    continue

                comments = []
                comment_texts = []
                if not args.no_comments:
                    try:
                        comments, _ = fetch_comments_for_post(post_id, cookies=cookies)
                        comment_texts = extract_comment_texts(comments)
                        if comments_writer.output_path and comment_texts:
                            comments_writer.append_comments(group_name, post_id, comment_texts)
                        time.sleep(args.delay)
                    except FacebookSessionExpiredException:
                        raise  # Chuyển tiếp ngoại lệ session hết hạn lên cấp xử lý chính
                    except Exception as e:
                        log_message(f"  ⚠️ Lỗi lấy comment bài {post_id}: {e}")

                record = {
                    "group_id": group_id,
                    "group_name": group_name,
                    "post_id": post_id,
                    "creation_time": post.get("creation_time"),
                    "created_at": post.get("created_at"),
                    "permalink": post.get("permalink", ""),
                    "message": post.get("message", ""),
                    "comment_count": post.get("comment_count", 0),
                    "photos": post.get("photos", []),
                    "videos": post.get("videos", []),
                    "comment_texts": comment_texts,
                    "comments": comments,
                    "scraped_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                processed_records.append(record)
                group_records_count += 1
                total_scraped_all_groups += 1

            if processed_records:
                writer.save_records(processed_records)
                log_message(f"  💾 Đã lưu {len(processed_records)} bài vào file kết quả (Nhóm này: {group_records_count}, Tổng: {total_scraped_all_groups})")

        # Callback cập nhật state.json ngay sau MỖI TRANG hoàn thành
        def on_page(page_num, next_cursor, total_posts_on_run, is_finished):
            state_manager.update_group_page(
                group_id=group_id,
                group_name=group_name,
                page_num=page_num,
                next_cursor=next_cursor,
                posts_count=group_records_count,
                is_finished=is_finished
            )
            c_str = f"{next_cursor[:20]}..." if next_cursor else "None"
            log_message(f"  📄 [state.json] Cập nhật: Trang {page_num} | Cursor: {c_str} | Số bài: {group_records_count}")

        try:
            posts = group_post_scraper_v2.fetch_posts(
                limit=args.limit,
                min_comments=args.min_comments,
                batch_size=5,
                on_batch_complete=on_batch,
                since_timestamp=since_timestamp,
                initial_cursor=start_cursor,
                initial_page=start_page,
                on_page_complete=on_page
            )
            state_manager.mark_group_completed(group_id, group_name, group_records_count)
            log_message(f"✅ Hoàn tất nhóm {group_name}: Lấy được {group_records_count} bài viết trong năm {args.year}")
        except FacebookSessionExpiredException as e:
            log_message("=" * 65)
            log_message(f"🚨 DỪNG TIẾN TRÌNH: {e}")
            log_message(f"👉 Dừng tại nhóm: {group_name} (ID: {group_id}).")
            log_message("👉 Điểm tiếp tục (trang & cursor) đã được lưu an toàn vào state.json.")
            log_message("👉 HƯỚNG DẪN TIẾP TỤC:")
            log_message("   1. Mở file .env và cập nhật chuỗi FB_COOKIES và FB_DTSG mới.")
            log_message("   2. Chạy lại lệnh nohup ban đầu.")
            log_message(f"   => Script sẽ TỰ ĐỘNG TIẾP TỤC NGAY TỪ TRANG ĐANG DỞ của nhóm [{group_name}]!")
            log_message("=" * 65)
            sys.exit(1)
        except Exception as e:
            log_message(f"❌ Lỗi khi cào nhóm {group_name} ({group_id}): {e}")

        # Nghỉ giữa các nhóm để giữ an toàn cho tài khoản Facebook
        log_message("⏳ Nghỉ 5 giây trước khi chuyển sang nhóm tiếp theo...")
        time.sleep(5)

    log_message("=" * 65)
    log_message(f"🎉 TẤT CẢ ĐÃ HOÀN THÀNH!")
    log_message(f"📁 Tổng số bài viết thu thập: {total_scraped_all_groups}")
    log_message(f"📄 Kết quả toàn bộ được lưu tại: {writer.output_path}")
    log_message(f"⚙️ Trạng thái toàn bộ: {state_manager.state_file}")
    log_message("=" * 65)


if __name__ == "__main__":
    run_scraper()
