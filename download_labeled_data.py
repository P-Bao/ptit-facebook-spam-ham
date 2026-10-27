"""
download_labeled_data.py
------------------------
Tải các bản ghi đã gán nhãn THÀNH CÔNG (response có trạng thái 'submitted')
từ Argilla Server (container 'my-argilla') về file CSV cục bộ,
để phục vụ huấn luyện mô hình phát hiện Spam/Ham.

Pipeline trước đó:
    crawl_group_comments.py
        -> clean_data_facebook_spamham.py   (data/comments_cleaned.csv)
        -> upload_to_argilla.py             (dataset 'comments_spamham')
        -> ai_assisted_spam_ham.py          (gợi ý nhãn bằng GLM)
        -> auto_accept_labels.py            (chốt nhãn >= threshold thành 'submitted')
        -> download_labeled_data.py         (script này: tải về CSV hoàn chỉnh)
"""

import argparse
import sys
import warnings
from pathlib import Path

import pandas as pd
import argilla as rg

# Tắt cảnh báo không cần thiết từ Argilla SDK
warnings.filterwarnings("ignore", category=UserWarning, module="argilla")
warnings.filterwarnings("ignore", message=".*RequestsDependencyWarning.*")

# Fix encoding cho Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_CSV = BASE_DIR / "data" / "comments_labeled_export.csv"
DEFAULT_CSV_LOOKUP = BASE_DIR / "data" / "comments_cleaned.csv"
DEFAULT_ARGILLA_URL = "http://localhost:6900"
DEFAULT_ARGILLA_KEY = "admin.apikey"
DEFAULT_WORKSPACE = "argilla"
DEFAULT_DATASET = "comments_spamham"

VALID_LABELS = ["spam", "ham"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Tải các bản ghi đã gán nhãn thành công (submitted) từ Argilla về file CSV."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=DEFAULT_DATASET,
        help=f"Tên dataset trên Argilla (mặc định: {DEFAULT_DATASET})",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT_CSV),
        help=f"Đường dẫn file CSV xuất ra (mặc định: {DEFAULT_OUTPUT_CSV})",
    )
    parser.add_argument(
        "--csv-lookup",
        type=str,
        default=str(DEFAULT_CSV_LOOKUP),
        help="File CSV gốc đã làm sạch để nối thêm cột metadata (comment_type, reaction_count). "
             "Để trống nếu không muốn nối.",
    )
    parser.add_argument(
        "--label",
        type=str,
        default=None,
        choices=["spam", "ham"],
        help="Chỉ xuất bản ghi có nhãn này (để trống nếu xuất cả hai)",
    )
    parser.add_argument(
        "--include-suggestion",
        action="store_true",
        help="Kèm theo cột confidence từ Suggestion (nếu bản ghi có gợi ý AI)",
    )
    parser.add_argument(
        "--api-url",
        type=str,
        default=DEFAULT_ARGILLA_URL,
        help=f"URL Argilla Server (mặc định: {DEFAULT_ARGILLA_URL})",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=DEFAULT_ARGILLA_KEY,
        help=f"API Key của tài khoản quản trị (mặc định: {DEFAULT_ARGILLA_KEY})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Chỉ thống kê số lượng bản ghi đã gán nhãn, không xuất file",
    )
    return parser.parse_args()


def extract_submitted_response(rec):
    """Lấy response có trạng thái 'submitted' từ một bản ghi (ưu tiên response mới nhất)."""
    submitted = [
        r for r in getattr(rec, "responses", []) or [] if getattr(r, "status", "") == "submitted"
    ]
    if not submitted:
        return None
    return submitted[-1]


def build_records_iter(dataset):
    """
    Trả về iterator các bản ghi ĐÃ GÁN NHÃN (submitted).
    Ưu tiên lọc phía server bằng Filter, fallback về lọc thủ công nếu SDK không hỗ trợ.
    """
    try:
        from argilla import Filter, Query

        filtered = dataset.records(query=Query(filter=Filter(("response.status", "submitted"))))
        return list(filtered)
    except Exception:
        return [rec for rec in dataset.records if extract_submitted_response(rec) is not None]


def main():
    args = parse_args()

    print(f"[*] Đang kết nối tới Argilla Server tại {args.api_url}...")
    try:
        client = rg.Argilla(api_url=args.api_url, api_key=args.api_key)
        print(f"[+] Kết nối thành công! (User: '{client.me.username}', quyền: '{client.me.role}')")
    except Exception as e:
        print(f"[ERROR] Không thể kết nối tới Argilla Server: {e}")
        print("Vui lòng kiểm tra Docker container 'my-argilla' đã chạy chưa (`docker compose ps`).")
        sys.exit(1)

    dataset = client.datasets(name=args.dataset, workspace=DEFAULT_WORKSPACE)
    if dataset is None:
        print(f"[ERROR] Không tìm thấy dataset '{args.dataset}' trên workspace '{DEFAULT_WORKSPACE}'.")
        sys.exit(1)

    print(f"[*] Đang tải các bản ghi ĐÃ GÁN NHÃN (status='submitted') từ dataset '{args.dataset}'...")
    all_records = build_records_iter(dataset)
    print(f"[+] Dataset có tổng cộng {len(list(dataset.records)):,} bản ghi.")
    print(f"[+] Số bản ghi đã gán nhãn thành công: {len(all_records):,}")

    if len(all_records) == 0:
        print("[!] Không có bản ghi nào đã gán nhãn. Hãy hoàn tất gán nhãn trên giao diện Argilla.")
        sys.exit(0)

    # 1. Trích xuất dữ liệu từ các bản ghi đã gán nhãn
    rows = []
    skipped_bad_label = 0
    for rec in all_records:
        resp = extract_submitted_response(rec)
        if resp is None:
            continue

        label = str(resp.value).strip().lower()
        if label not in VALID_LABELS:
            skipped_bad_label += 1
            continue

        if args.label is not None and label != args.label:
            continue

        rid = getattr(rec, "id", None) or str(getattr(rec, "external_id", "") or "")
        fields = getattr(rec, "fields", {}) or {}
        text_display = str(fields.get("cleaned_text", "")).strip()

        row = {
            "id": rid,
            "cleaned_text": text_display,
            "label": label,
        }

        if args.include_suggestion:
            conf = None
            for s in getattr(rec, "suggestions", []) or []:
                if getattr(s, "question_name", "") == "label":
                    conf = getattr(s, "score", None)
                    break
            row["ai_confidence"] = conf

        rows.append(row)

    if skipped_bad_label:
        print(f"[!] Đã bỏ qua {skipped_bad_label} bản ghi có nhãn không hợp lệ.")

    print(f"[*] Sau khi lọc{' (label=' + args.label + ')' if args.label else ''}: {len(rows):,} bản ghi hợp lệ.")

    df = pd.DataFrame(rows)
    label_counts = df["label"].value_counts()
    print("-" * 50)
    for lbl, cnt in label_counts.items():
        print(f"    - {lbl}: {cnt:,} ({cnt / len(df) * 100:.2f}%)")
    print("-" * 50)

    if args.dry_run:
        print("\n[DRY RUN] Đã hoàn tất thống kê (không có file nào được lưu).")
        return

    # 2. Nối thêm metadata từ CSV gốc (comment_type, reaction_count, text)
    csv_lookup = Path(args.csv_lookup) if args.csv_lookup else None
    if csv_lookup and csv_lookup.exists():
        print(f"[*] Đang nối metadata từ: {csv_lookup}...")
        df_lookup = pd.read_csv(csv_lookup)

        # Ánh xạ id 'cmt_{idx}' hoặc 'p2_cmt_{idx}' -> dòng thứ idx trong CSV gốc
        df["_idx"] = df["id"].astype(str).str.replace(r"^(?:p2_)?cmt_", "", regex=True)
        df["_idx"] = pd.to_numeric(df["_idx"], errors="coerce")
        df = df.dropna(subset=["_idx"])
        df["_idx"] = df["_idx"].astype(int)

        df_lookup["_idx"] = df_lookup.index

        for col in ["text", "comment_type", "reaction_count"]:
            if col in df_lookup.columns:
                meta_map = df_lookup.set_index("_idx")[col]
                df[col] = df["_idx"].map(meta_map)

        df = df.drop(columns=["_idx"])
        print("[+] Đã nối metadata thành công.")
    elif csv_lookup:
        print(f"[!] Không tìm thấy file {csv_lookup}. Chỉ xuất dữ liệu có trên Argilla.")

    # 3. Sắp xếp thứ tự cột cho gọn
    front_cols = ["id", "label", "cleaned_text"]
    rest_cols = [c for c in df.columns if c not in front_cols]
    df = df[front_cols + rest_cols]

    # 4. Lưu ra file CSV
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 60)
    print(f"[SUCCESS] Đã tải xuống {len(df):,} bản ghi đã gán nhãn thành công!")
    print(f"👉 File lưu tại: {out_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
