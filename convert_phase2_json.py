"""
convert_phase2_json.py
----------------------
Bước 0 (Phase 2): Chuyển dữ liệu crawl Phase 2 dạng JSON (data/groups_data_phase2.json)
sang CSV phẳng cùng định dạng file crawl cũ, để chạy tiếp đúng pipeline cũ:

    convert_phase2_json.py              (script này)  -> data/comments_phase2.csv
        -> clean_data_facebook_spamham.py             -> data/comments_phase2_cleaned.csv
        -> upload_to_argilla.py / ai_assisted_spam_ham.py / auto_accept_labels.py
        -> download_labeled_data.py                   -> data/comments_labeled_phase2.csv
        -> merge_labeled_data.py                      -> data/comments_labeled_merged.csv

Xử lý:
  1. Làm phẳng post -> comments -> replies thành từng dòng (comment_type = comment / reply).
  2. Bỏ dòng text rỗng.
  3. Bỏ bình luận ĐÃ CÓ trong Phase 1 (so khớp theo văn bản sau khi làm sạch bằng
     đúng hàm preprocess() của clean_data_facebook_spamham.py) -> không gán nhãn lại,
     không tốn chi phí gọi GLM cho mẫu trùng.

Đầu ra: data/comments_phase2.csv với các cột
  post_id, group_id, group_name, created_at, comment_type, text, reaction_count
"""

import argparse
import csv
import json
import sys
from pathlib import Path

# Dùng lại đúng hàm làm sạch của pipeline cũ để so khớp trùng lặp với Phase 1
from clean_data_facebook_spamham import preprocess

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

DEFAULT_INPUT_JSON = DATA_DIR / "groups_data_phase2.json"
DEFAULT_PHASE1_CSV = DATA_DIR / "comments_labeled_export.csv"
DEFAULT_OUTPUT_CSV = DATA_DIR / "comments_phase2.csv"

FIELDNAMES = ["post_id", "group_id", "group_name", "created_at", "comment_type", "text", "reaction_count"]


def to_int(value) -> int:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return 0


def flatten_posts(posts: list) -> list:
    """post -> comments -> replies  ==>  danh sách dòng phẳng."""
    rows = []
    for post in posts:
        base = {
            "post_id": post.get("post_id", ""),
            "group_id": post.get("group_id", ""),
            "group_name": post.get("group_name", ""),
            "created_at": post.get("created_at", ""),
        }
        for com in post.get("comments", []) or []:
            rows.append({
                **base,
                "comment_type": "comment",
                "text": (com.get("text") or "").strip(),
                "reaction_count": to_int(com.get("reaction_count")),
            })
            for rep in com.get("replies", []) or []:
                rows.append({
                    **base,
                    "comment_type": "reply",
                    "text": (rep.get("text") or "").strip(),
                    "reaction_count": to_int(rep.get("reaction_count")),
                })
    return rows


def load_phase1_texts(phase1_csv: Path) -> set:
    """Tập văn bản đã làm sạch của Phase 1 để loại trùng."""
    if not phase1_csv.exists():
        print(f"  [CẢNH BÁO] Không tìm thấy {phase1_csv} -> bỏ qua bước loại trùng với Phase 1")
        return set()
    texts = set()
    with open(phase1_csv, "r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            cleaned = (row.get("cleaned_text") or "").strip() or preprocess(row.get("text") or "")
            if cleaned:
                texts.add(cleaned)
    return texts


def parse_args():
    parser = argparse.ArgumentParser(description="Bước 0: chuyển JSON crawl Phase 2 sang CSV phẳng.")
    parser.add_argument("--input", "-i", type=str, default=str(DEFAULT_INPUT_JSON),
                        help=f"File JSON Phase 2 (mặc định: {DEFAULT_INPUT_JSON})")
    parser.add_argument("--phase1-csv", type=str, default=str(DEFAULT_PHASE1_CSV),
                        help=f"CSV Phase 1 đã gán nhãn để loại trùng (mặc định: {DEFAULT_PHASE1_CSV})")
    parser.add_argument("--output", "-o", type=str, default=str(DEFAULT_OUTPUT_CSV),
                        help=f"CSV đầu ra (mặc định: {DEFAULT_OUTPUT_CSV})")
    parser.add_argument("--keep-phase1-duplicates", action="store_true",
                        help="Không loại các bình luận đã có trong Phase 1")
    return parser.parse_args()


def main():
    args = parse_args()
    input_json, output_csv = Path(args.input), Path(args.output)

    print("=" * 65)
    print(" BƯỚC 0: CHUYỂN JSON PHASE 2 -> CSV")
    print("=" * 65)

    with open(input_json, "r", encoding="utf-8") as f:
        posts = json.load(f)
    rows = flatten_posts(posts)
    n_raw = len(rows)
    n_comment = sum(r["comment_type"] == "comment" for r in rows)

    rows = [r for r in rows if r["text"]]
    n_empty = n_raw - len(rows)

    n_dup_p1 = 0
    if not args.keep_phase1_duplicates:
        phase1_texts = load_phase1_texts(Path(args.phase1_csv))
        kept = []
        for r in rows:
            if preprocess(r["text"]) in phase1_texts:
                n_dup_p1 += 1
                continue
            kept.append(r)
        rows = kept

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(f"  - Số bài viết:                 {len(posts):>6}")
    print(f"  - Bình luận thô (comment+reply): {n_raw:>6}  (comment: {n_comment}, reply: {n_raw - n_comment})")
    print(f"  - Bỏ (text rỗng):              {n_empty:>6}")
    print(f"  - Bỏ (đã có trong Phase 1):    {n_dup_p1:>6}")
    print(f"  - Còn lại:                     {len(rows):>6}")
    print(f"  -> Đã lưu CSV: {output_csv}")
    print("\n Tiếp theo:")
    print(f"   python clean_data_facebook_spamham.py -i {output_csv} -o data/comments_phase2_cleaned.csv")


if __name__ == "__main__":
    main()
