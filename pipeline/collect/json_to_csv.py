"""pipeline/collect/json_to_csv.py
Chuyển JSON bình luận thành bảng CSV.

Đầu vào: JSON raw; tùy chọn CSV nhãn tham chiếu qua --phase1-csv.
Xử lý: Làm phẳng post/comment/reply, bỏ text rỗng; có thể loại trùng với dữ liệu tham chiếu.
Đầu ra: CSV gồm post_id, group_id, group_name, created_at, comment_type, text, reaction_count; chưa có label.
Liên quan: pipeline.prepare.clean_for_labeling; pipeline.label; tools.dataset_versions.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

# Dùng cùng cleaner chuẩn bị gán nhãn để so trùng với CSV tham chiếu.
from pipeline.prepare.clean_for_labeling import preprocess

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"

DEFAULT_INPUT_JSON = DATA_DIR / "raw" / "posts.json"
DEFAULT_PHASE1_CSV = DATA_DIR / "labeled" / "comments_labeled_export.csv"
DEFAULT_OUTPUT_CSV = DATA_DIR / "intermediate" / "comments.csv"

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
    parser = argparse.ArgumentParser(description="Chuyển JSON thu thập sang CSV chưa có nhãn; tùy chọn lọc trùng dữ liệu tham chiếu.")
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
    print(" CHUYỂN JSON THU THẬP -> CSV CHƯA CÓ NHÃN")
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
    print(f"   python -m pipeline.prepare.clean_for_labeling -i \"{output_csv}\" -o data/intermediate/comments_cleaned.csv")


if __name__ == "__main__":
    main()
