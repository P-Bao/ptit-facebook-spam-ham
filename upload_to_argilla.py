"""
upload_to_argilla.py
--------------------
Script tải tập dữ liệu bình luận Facebook/PTIT (comments_cleaned.csv) lên Argilla Server
để phục vụ gán nhãn cho bài toán Phát hiện Spam (Spam vs Ham) tiếng Việt.
"""

import argparse
import sys
import warnings
from pathlib import Path
import pandas as pd
import argilla as rg

# Tắt cảnh báo không cần thiết từ Argilla SDK và Requests
warnings.filterwarnings("ignore", category=UserWarning, module="argilla")
warnings.filterwarnings("ignore", message=".*RequestsDependencyWarning.*")

# Fix encoding cho Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ─────────────────────────────────────────────────────────────────────────────
# CẤU HÌNH MẶC ĐỊNH
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_API_URL = "http://localhost:6900"
DEFAULT_API_KEY = "admin.apikey"
DEFAULT_WORKSPACE = "argilla"
DEFAULT_DATASET_NAME = "comments_spamham"
DEFAULT_CSV_PATH = Path(__file__).resolve().parent / "data" / "comments_cleaned.csv"

DEFAULT_LABELS = ["spam", "ham"]
DEFAULT_GUIDELINES = """
## Hướng dẫn gán nhãn Spam/Ham (Phát hiện Spam):
- **spam** (Rác): Quảng cáo, rao vặt, chào hàng khóa học/sản phẩm, chứa link lạ hoặc SĐT đáng nghi,
  copy-paste hàng loạt, nội dung lừa đảo/không liên quan đến thảo luận.
- **ham** (Bình thường): Bình luận hỏi đáp, chia sẻ kinh nghiệm, thảo luận học tập/bài đăng một cách chân thực.
""".strip()


def parse_args():
    parser = argparse.ArgumentParser(description="Upload dataset lên Argilla Server để gán nhãn.")
    parser.add_argument(
        "--csv",
        type=str,
        default=str(DEFAULT_CSV_PATH),
        help=f"Đường dẫn file CSV dữ liệu (mặc định: {DEFAULT_CSV_PATH})",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=DEFAULT_DATASET_NAME,
        help=f"Tên dataset trên Argilla (mặc định: {DEFAULT_DATASET_NAME})",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Xóa dataset cũ và tạo mới lại từ đầu nếu đã tồn tại",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Số lượng bản ghi tối đa cần tải lên (để trống nếu muốn tải toàn bộ)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Kích thước batch khi gửi lên server (mặc định: 256)",
    )
    parser.add_argument(
        "--api-url",
        type=str,
        default=DEFAULT_API_URL,
        help=f"URL Argilla Server (mặc định: {DEFAULT_API_URL})",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=DEFAULT_API_KEY,
        help=f"API Key của tài khoản quản trị (mặc định: {DEFAULT_API_KEY})",
    )
    parser.add_argument(
        "--id-prefix",
        type=str,
        default="cmt_",
        help="Tiền tố ID cho bản ghi trên Argilla (mặc định: 'cmt_', dùng 'p2_cmt_' cho Phase 2)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    csv_file = Path(args.csv)

    if not csv_file.exists():
        print(f"[ERROR] Không tìm thấy file dữ liệu: {csv_file}")
        sys.exit(1)

    print(f"[*] Đang kết nối tới Argilla Server tại {args.api_url}...")
    try:
        client = rg.Argilla(api_url=args.api_url, api_key=args.api_key)
        user_info = client.me
        print(f"[+] Kết nối thành công! Đang đăng nhập với quyền: '{user_info.role}' (User: '{user_info.username}')")
    except Exception as e:
        print(f"[ERROR] Không thể kết nối tới Argilla Server: {e}")
        print("Vui lòng kiểm tra Docker container 'my-argilla' đã chạy chưa (`docker compose ps`).")
        sys.exit(1)

    # 1. Kiểm tra hoặc khởi tạo Dataset
    dataset = client.datasets(name=args.dataset, workspace=DEFAULT_WORKSPACE)

    if args.reset and dataset is not None:
        print(f"[*] Đang xóa dataset cũ '{args.dataset}' theo yêu cầu (--reset)...")
        dataset.delete()
        dataset = None

    if dataset is None:
        print(f"[*] Dataset '{args.dataset}' chưa có trên server. Đang tiến hành tạo mới...")
        settings = rg.Settings(
            guidelines=DEFAULT_GUIDELINES,
            fields=[
                rg.TextField(
                    name="cleaned_text",
                    title="Nội dung bình luận (đã làm sạch)",
                    required=True,
                    use_markdown=False,
                ),
            ],
            questions=[
                rg.LabelQuestion(
                    name="label",
                    title="Phân loại (Spam/Ham)",
                    labels=DEFAULT_LABELS,
                    required=True,
                ),
            ],
        )

        dataset = rg.Dataset(
            name=args.dataset,
            workspace=DEFAULT_WORKSPACE,
            settings=settings,
            client=client,
        )
        try:
            dataset.create()
            print(f"[+] Đã tạo mới dataset '{args.dataset}' thành công!")
        except Exception as e:
            if "already exists" in str(e).lower() or "conflict" in str(e).lower():
                print(f"[!] Dataset '{args.dataset}' đã tồn tại. Đang kết nối tới dataset hiện có...")
                dataset = client.datasets(name=args.dataset, workspace=DEFAULT_WORKSPACE)
            else:
                print(f"[ERROR] Lỗi khi tạo dataset: {e}")
                sys.exit(1)
    else:
        existing_count = len(list(dataset.records))
        print(f"[+] Tìm thấy dataset '{args.dataset}' đã tồn tại (hiện có {existing_count} bản ghi).")
        print(f"[*] Gợi ý: Dùng cờ `--reset` nếu bạn muốn xóa sạch dataset cũ và tạo lại từ đầu.")

    # 2. Đọc dữ liệu từ CSV
    print(f"[*] Đang đọc dữ liệu từ: {csv_file}...")
    df = pd.read_csv(csv_file)
    total_in_file = len(df)
    print(f"[+] Tổng số dòng trong file: {total_in_file:,}")

    if args.limit is not None and args.limit > 0:
        df = df.head(args.limit)
        print(f"[*] Giới hạn số lượng bản ghi tải lên: {len(df):,}")

    # 3. Chuẩn bị danh sách bản ghi (Records)
    records = []
    for idx, row in df.iterrows():
        # Ưu tiên cleaned_text, nếu rỗng thì lấy text gốc
        text_clean = str(row["cleaned_text"]).strip() if pd.notna(row.get("cleaned_text")) else ""
        text_raw = str(row["text"]).strip() if pd.notna(row.get("text")) else ""
        text_display = text_clean if text_clean else text_raw

        if not text_display:
            continue

        comment_type = str(row["comment_type"]).strip() if pd.notna(row.get("comment_type")) else ""
        reaction_count = str(row["reaction_count"]).strip() if pd.notna(row.get("reaction_count")) else ""

        record_data = {
            "id": f"{args.id_prefix}{idx}",
            "cleaned_text": text_display,
            "metadata": {
                "comment_type": comment_type,
                "reaction_count": reaction_count,
            },
        }

        records.append(record_data)

    print(f"[*] Bắt đầu tải {len(records):,} bản ghi lên Argilla (batch size: {args.batch_size})...")
    dataset.records.log(records=records, batch_size=args.batch_size)

    print("\n" + "=" * 60)
    print(f"[SUCCESS] Đã tải lên hoàn tất {len(records):,} bản ghi!")
    print(f"👉 Mở giao diện để gán nhãn ngay: {args.api_url}/dataset/{dataset.id}/annotation-mode")
    print("=" * 60)


if __name__ == "__main__":
    main()
