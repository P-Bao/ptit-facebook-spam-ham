"""
auto_accept_labels.py
---------------------
Tự động chuyển các gợi ý (Suggestion) từ AI có độ tin cậy (Confidence) >= ngưỡng
(mặc định: 0.75) thành nhãn chính thức (Submitted Response) trên Argilla,
đồng thời có tùy chọn xuất ra file CSV hoàn chỉnh để phục vụ huấn luyện mô hình.
"""

import argparse
import json
import sys
import warnings
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import argilla as rg

# Tắt cảnh báo
warnings.filterwarnings("ignore", category=UserWarning, module="argilla")
warnings.filterwarnings("ignore", message=".*RequestsDependencyWarning.*")

# Fix UTF-8 cho Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CACHE_PATH = BASE_DIR / "data" / "predictions_glm_cache.json"
DEFAULT_CSV_PATH = BASE_DIR / "data" / "comments_cleaned.csv"
DEFAULT_OUTPUT_CSV = BASE_DIR / "data" / "comments_auto_labeled.csv"
DEFAULT_ARGILLA_URL = "http://localhost:6900"
DEFAULT_ARGILLA_KEY = "admin.apikey"
DEFAULT_DATASET = "comments_spamham"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Chuyển gợi ý AI có confidence >= ngưỡng thành nhãn chính thức."
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.75,
        help="Ngưỡng confidence tối thiểu để chấp nhận nhãn (mặc định: 0.75)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=DEFAULT_DATASET,
        help=f"Tên dataset trên Argilla (mặc định: {DEFAULT_DATASET})",
    )
    parser.add_argument(
        "--cache",
        type=str,
        default=str(DEFAULT_CACHE_PATH),
        help=f"File cache dự đoán (mặc định: {DEFAULT_CACHE_PATH})",
    )
    parser.add_argument(
        "--csv-in",
        type=str,
        default=str(DEFAULT_CSV_PATH),
        help=f"File CSV gốc đã làm sạch (mặc định: {DEFAULT_CSV_PATH})",
    )
    parser.add_argument(
        "--export-csv",
        type=str,
        default=str(DEFAULT_OUTPUT_CSV),
        help=f"Đường dẫn file CSV xuất ra (để trống nếu không muốn xuất)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Kích thước batch khi gửi lên Argilla (mặc định: 256)",
    )
    parser.add_argument(
        "--api-url",
        type=str,
        default=DEFAULT_ARGILLA_URL,
        help=f"URL server Argilla (mặc định: {DEFAULT_ARGILLA_URL})",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=DEFAULT_ARGILLA_KEY,
        help=f"Argilla API Key (mặc định: {DEFAULT_ARGILLA_KEY})",
    )
    parser.add_argument(
        "--skip-argilla",
        action="store_true",
        help="Chỉ xuất file CSV, không cập nhật lên Argilla Server",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Chỉ thống kê số lượng, không thực hiện cập nhật hoặc lưu",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cache_path = Path(args.cache)

    if not cache_path.exists():
        print(f"[ERROR] Không tìm thấy file cache: {cache_path}")
        sys.exit(1)

    with open(cache_path, "r", encoding="utf-8") as f:
        cache_data = json.load(f)

    print("=" * 65)
    print(f"[*] Tổng số bản ghi trong cache: {len(cache_data):,}")
    print(f"[*] Ngưỡng Confidence: >= {args.threshold}")
    print("=" * 65)

    auto_accept = {}
    manual_review = {}

    for rid, item in cache_data.items():
        conf = float(item.get("confidence", 0.0))
        label = item.get("label")
        if conf >= args.threshold:
            auto_accept[rid] = item
        else:
            manual_review[rid] = item

    print(f"[+] Bản ghi đủ điều kiện gán nhãn chính thức (>= {args.threshold}): {len(auto_accept):,} ({len(auto_accept)/len(cache_data)*100:.2f}%)")
    spam_accepted = sum(1 for v in auto_accept.values() if v.get("label") == "spam")
    ham_accepted = sum(1 for v in auto_accept.values() if v.get("label") == "ham")
    print(f"    - Spam: {spam_accepted:,}")
    print(f"    - Ham:  {ham_accepted:,}")
    print(f"[-] Bản ghi cần thẩm định thủ công (< {args.threshold}): {len(manual_review):,} ({len(manual_review)/len(cache_data)*100:.2f}%)")

    if args.dry_run:
        print("\n[DRY RUN] Đã hoàn tất thống kê (không có thay đổi nào được ghi nhận).")
        return

    # 1. Cập nhật lên Argilla Server
    if not args.skip_argilla:
        print(f"\n[*] Đang kết nối tới Argilla Server tại {args.api_url}...")
        try:
            client = rg.Argilla(api_url=args.api_url, api_key=args.api_key)
            dataset = client.datasets(name=args.dataset)
            if dataset is None:
                print(f"[ERROR] Không tìm thấy dataset '{args.dataset}' trên Argilla.")
                sys.exit(1)
            user_id = client.me.id
            print(f"[+] Kết nối dataset '{args.dataset}' thành công! (User ID: {user_id})")
        except Exception as e:
            print(f"[ERROR] Không thể kết nối tới Argilla Server: {e}")
            sys.exit(1)

        print(f"[*] Đang chuẩn bị {len(auto_accept):,} bản ghi để submit thành nhãn chính thức...")

        # Đọc text từ CSV để bảo toàn nội dung field
        csv_path = Path(args.csv_in)
        text_lookup = {}
        if csv_path.exists():
            df_lookup = pd.read_csv(csv_path)
            for idx, row in df_lookup.iterrows():
                rid = f"cmt_{idx}"
                text_clean = str(row["cleaned_text"]).strip() if pd.notna(row.get("cleaned_text")) else ""
                text_raw = str(row["text"]).strip() if pd.notna(row.get("text")) else ""
                text_lookup[rid] = text_clean if text_clean else text_raw

        records_to_update = []
        for rid, item in tqdm(auto_accept.items(), desc="Đóng gói records"):
            lbl = item["label"]
            conf = float(item["confidence"])
            fields = {"cleaned_text": text_lookup.get(rid, "")} if rid in text_lookup else None

            rec_kwargs = {
                "id": rid,
                "responses": [
                    rg.Response(
                        question_name="label",
                        value=lbl,
                        user_id=user_id,
                        status="submitted",
                    )
                ],
                "suggestions": [
                    rg.Suggestion(
                        question_name="label",
                        value=lbl,
                        score=conf,
                    )
                ],
                "_dataset": dataset,
            }
            if fields:
                rec_kwargs["fields"] = fields

            records_to_update.append(rg.Record(**rec_kwargs))

        print(f"[*] Đang gửi dữ liệu lên Argilla theo từng batch ({args.batch_size})...")
        try:
            dataset.records.log(records=records_to_update, batch_size=args.batch_size)
            print("\n" + "=" * 65)
            print(f"[SUCCESS] Đã chuyển thành công {len(records_to_update):,} bản ghi sang trạng thái Completed!")
            print(f"👉 Mở ngay Argilla để xem các bản ghi còn lại cần thẩm định (< {args.threshold}):")
            print(f"   {args.api_url}/dataset/{dataset.id}/annotation-mode")
            print("=" * 65)
        except Exception as e:
            print(f"[ERROR] Lỗi khi cập nhật lên Argilla: {e}")

    # 2. Xuất file CSV nếu có yêu cầu
    if args.export_csv:
        csv_in_path = Path(args.csv_in)
        if csv_in_path.exists():
            print(f"\n[*] Đang tạo file CSV tổng hợp tại {args.export_csv}...")
            df = pd.read_csv(csv_in_path)
            labels = []
            confidences = []
            is_auto = []

            for idx in range(len(df)):
                rid = f"cmt_{idx}"
                if rid in auto_accept:
                    labels.append(auto_accept[rid]["label"])
                    confidences.append(auto_accept[rid]["confidence"])
                    is_auto.append(True)
                elif rid in manual_review:
                    labels.append(None)  # Chưa duyệt chính thức
                    confidences.append(manual_review[rid]["confidence"])
                    is_auto.append(False)
                else:
                    labels.append(None)
                    confidences.append(None)
                    is_auto.append(False)

            df["official_label"] = labels
            df["ai_confidence"] = confidences
            df["is_auto_accepted"] = is_auto

            out_path = Path(args.export_csv)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(out_path, index=False, encoding="utf-8-sig")
            print(f"[+] Đã lưu file CSV: {out_path} ({len(df):,} dòng)")


if __name__ == "__main__":
    main()
