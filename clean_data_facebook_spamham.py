"""
clean_data / clean_data_facebook_spamham.py
--------------------------------------------
Tiền xử lý và làm sạch dữ liệu bình luận Facebook/PTIT (data/comments.csv,
data/comments (1).csv) cho bài toán Phát hiện Spam (Spam vs Ham) tiếng Việt.

Xử lý:
  1. Chuẩn hóa Unicode NFC  → thống nhất ký tự tiếng Việt
  2. Decode HTML entities   → & → &, < → <, ...
  3. Xóa thẻ HTML           → loại bỏ <b>, <br/>, ...
  4. Thay URL bằng token    → "urltoken"
  5. Thay SĐT VN bằng token → "phonetoken" (hỗ trợ cả SĐT có dấu cách/dấu chấm/gạch nối)
  6. Xóa emoji & symbol     → các biểu tượng cảm xúc, icon
  7. Lọc ký tự lạ           → giữ chữ cái, tiếng Việt, chữ số, dấu câu cơ bản
  8. Chuẩn hóa dấu câu      → gộp nhiều dấu "!!!" → "!", "..." giữ "..."
  9. Chuẩn hóa khoảng trắng → gộp space/tab/newline → 1 khoảng trắng

Đầu ra:
  - data/comments_cleaned.csv: 1 file duy nhất, giữ nguyên các cột gốc và toàn bộ
    dòng (sau khi gộp + khử trùng), chỉ thêm 1 cột cleaned_text ngay bên phải cột text.
"""

import argparse
import csv
import html
import re
import sys
import unicodedata
from pathlib import Path

# Fix encoding cho Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ─────────────────────────────────────────────────────────────────────────────
# CẤU HÌNH ĐƯỜNG DẪN & THAM SỐ MẶC ĐỊNH
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

DEFAULT_INPUT_FILES = [
    DATA_DIR / "comments.csv",
    DATA_DIR / "comments (1).csv",
]
DEFAULT_OUTPUT_CSV = DATA_DIR / "comments_cleaned.csv"

# ─────────────────────────────────────────────────────────────────────────────
# REGEX PATTERNS (Biên dịch 1 lần)
# ─────────────────────────────────────────────────────────────────────────────
_RE_HTML_TAG = re.compile(r"<[^>]+>")
_RE_URL = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)

# SĐT Việt Nam: hỗ trợ cả số viết liền và số có khoảng trắng, dấu chấm, dấu gạch ngang (+84, 03x, 05x, 07x, 08x, 09x)
_RE_PHONE_VN = re.compile(
    r"(?<!\d)(?:\+84|0084|84|0)(?:[\s.-]?[3-9](?:[\s.-]?\d){8})(?!\d)"
)

_RE_EMOJI = re.compile(
    "["
    "\U0001F600-\U0001F64F"   # Emoticons
    "\U0001F300-\U0001F5FF"   # Symbols & Pictographs
    "\U0001F680-\U0001F6FF"   # Transport & Map
    "\U0001F1E0-\U0001F1FF"   # Flags
    "\U00002600-\U000027BF"   # Misc symbols
    "\U0001F900-\U0001F9FF"   # Supplemental Symbols & Pictographs
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "\uFE0F"                  # Variation Selector-16
    "\u200D"                  # Zero Width Joiner
    "]+",
    flags=re.UNICODE,
)
_RE_KEEP = re.compile(
    r"[^\w\s"
    r"\u00C0-\u024F"          # Latin mở rộng
    r"\u1E00-\u1EFF"          # Latin bổ sung (tiếng Việt)
    r"\u0300-\u036F"          # Combining diacritics
    r".,!?;:()\-\"'""''…"
    r"]",
    flags=re.UNICODE,
)
_RE_PUNCT_REP = re.compile(r"([!?]){2,}")
_RE_DOTS_REP = re.compile(r"\.{4,}")
_RE_SPACE = re.compile(r"[ \t\r\n\u00A0]+")


# ─────────────────────────────────────────────────────────────────────────────
# HÀM NẠP VÀ LÀM SẠCH TEXT
# ─────────────────────────────────────────────────────────────────────────────
def load_comments_from_csv(input_file: Path) -> list:
    """
    Nạp dữ liệu bình luận từ file CSV (UTF-8 với BOM).
    Trả về list các dict (mỗi dict là 1 dòng, giữ nguyên các cột gốc).
    """
    if not input_file.exists():
        return []
    with open(input_file, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def preprocess(text: str) -> str:
    """
    Tiền xử lý văn bản tiếng Việt cho bài toán Spam/Ham.
    Trả về text_clean (URL/SĐT được thay bằng token, không xóa trắng).
    """
    if not text:
        return ""

    # 1. Unicode NFC
    text = unicodedata.normalize("NFC", text)
    # 2. Decode HTML entities
    text = html.unescape(text)
    # 3. Xóa thẻ HTML
    text = _RE_HTML_TAG.sub(" ", text)

    # 4. Thay URL bằng token (KHÔNG xóa trắng - URL là tín hiệu spam quan trọng)
    text = _RE_URL.sub(" urltoken ", text)

    # 5. Thay số điện thoại VN bằng token (tín hiệu spam quan trọng)
    text = _RE_PHONE_VN.sub(" phonetoken ", text)

    # 6. Xóa emoji & ký tự hình ảnh
    text = _RE_EMOJI.sub(" ", text)
    # 7. Xóa ký tự đặc biệt, giữ lại chữ cái, số, dấu câu (urltoken/phonetoken vẫn giữ được)
    text = _RE_KEEP.sub(" ", text)
    # 8. Chuẩn hóa dấu câu lặp
    text = _RE_PUNCT_REP.sub(r"\1", text)
    text = _RE_DOTS_REP.sub("...", text)
    # 9. Chuẩn hóa khoảng trắng
    text = _RE_SPACE.sub(" ", text)

    return text.strip()


def save_csv(records: list, fieldnames: list, output_csv: Path):
    """Lưu danh sách bản ghi ra 1 file CSV duy nhất."""
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


# ─────────────────────────────────────────────────────────────────────────────
# TIỀN XỬ LÝ DỮ LIỆU BÌNH LUẬN FACEBOOK/PTIT (Spam / Ham)
# ─────────────────────────────────────────────────────────────────────────────
def clean_facebook_comments(
    input_files: list,
    output_csv: Path = DEFAULT_OUTPUT_CSV,
):
    stats = {"per_file": [], "total": 0, "dedup": 0, "url": 0, "phone": 0}

    # --- Gộp + khử trùng: dòng trùng (post_id, comment_type, text) chỉ giữ 1 bản ---
    merged = []
    seen = set()
    for input_file in input_files:
        rows = load_comments_from_csv(input_file)
        kept = 0
        for row in rows:
            key = (row.get("post_id", ""), row.get("comment_type", ""), row.get("text", ""))
            if key in seen:
                stats["dedup"] += 1
                continue
            seen.add(key)
            kept += 1
            merged.append(row)
        stats["per_file"].append((input_file.name, len(rows), kept))
        stats["total"] += len(rows)

    if not merged:
        print("  [CẢNH BÁO] Không có dữ liệu hợp lệ từ các file nguồn")
        return

    # --- Thêm cột cleaned_text ngay bên phải cột text, giữ nguyên các cột gốc ---
    fieldnames = list(merged[0].keys())
    if "text" in fieldnames:
        fieldnames.insert(fieldnames.index("text") + 1, "cleaned_text")
    else:
        fieldnames.append("cleaned_text")

    cleaned = []
    skipped_empty = 0
    for row in merged:
        text = row.get("text", "") or ""
        # 1. Đếm thống kê URL và SĐT trên văn bản gốc TRƯỚC khi thay bằng token
        if _RE_URL.search(text):
            stats["url"] += 1
        if _RE_PHONE_VN.search(text):
            stats["phone"] += 1

        clean_text = preprocess(text)
        # 2. Loại bỏ các dòng sau khi làm sạch không chứa bất kỳ chữ cái hoặc chữ số nào (vd: ".")
        if not re.search(r"[0-9a-zA-ZÀ-ỹ]", clean_text):
            skipped_empty += 1
            continue

        new_row = dict(row)
        new_row["cleaned_text"] = clean_text
        cleaned.append(new_row)

    save_csv(cleaned, fieldnames, output_csv)

    for name, raw, kept in stats["per_file"]:
        print(f"  - File: {name:<22} | thô: {raw:>6} | giữ: {kept:>6}")
    print(f"  - Tổng dòng thô:            {stats['total']:>6}")
    print(f"  - Bỏ (dòng trùng lặp):      {stats['dedup']:>6}")
    print(f"  - Bỏ (rỗng/chỉ dấu câu):    {skipped_empty:>6}")
    print(f"  - Còn lại sau làm sạch:     {len(cleaned):>6} ({len(cleaned)/stats['total']*100:.1f}%)" if stats['total'] else "")
    print(f"  - Có chứa URL:              {stats['url']:>6}")
    print(f"  - Có chứa SĐT:              {stats['phone']:>6}")
    print(f"  -> Đã lưu CSV: {output_csv}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="Tiền xử lý và làm sạch dữ liệu bình luận Facebook/PTIT cho bài toán Spam/Ham."
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        nargs="+",
        default=[str(p) for p in DEFAULT_INPUT_FILES],
        help=f"Các file dữ liệu bình luận thô (.csv) (mặc định: {[str(p) for p in DEFAULT_INPUT_FILES]})",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=str(DEFAULT_OUTPUT_CSV),
        help=f"Đường dẫn lưu file CSV sạch (mặc định: {DEFAULT_OUTPUT_CSV})",
    )
    return parser.parse_args()


if __name__ == "__main__":
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    args = parse_args()

    print("=" * 65)
    print(" TIỀN XỬ LÝ VÀ LÀM SẠCH DỮ LIỆU FACEBOOK/PTIT CHO BÀI TOÁN SPAM/HAM")
    print("=" * 65)

    clean_facebook_comments(
        input_files=[Path(p) for p in args.input],
        output_csv=Path(args.output),
    )

    print("\n" + "=" * 65)
    print(" HOÀN TẤT: Dữ liệu Facebook/PTIT đã được làm sạch, sẵn sàng để gán nhãn spam/ham.")
    print("=" * 65)
