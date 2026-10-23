"""
merge_labeled_data.py
---------------------
Bước 2e: Gộp dữ liệu đã gán nhãn Phase 1 (cũ) và Phase 2 (mới) thành 1 file CSV
cho notebook spam-ham-comments-v2.ipynb.

Đầu vào:
  - data/comments_labeled_export.csv   (Phase 1, giữ NGUYÊN VẸN, đúng thứ tự dòng)
  - data/comments_labeled_phase2.csv   (Phase 2, xuất từ download_labeled_data.py)
    * Cũng chấp nhận file xuất từ auto_accept_labels.py --skip-argilla (cột official_label);
      khi đó chỉ lấy các dòng đã có nhãn chính thức.

Xử lý:
  1. Giữ đúng 6 cột cũ: id, label, cleaned_text, text, comment_type, reaction_count
     + thêm cột `source` (phase1 / phase2) để notebook so sánh trước/sau tích hợp.
  2. Id Phase 2 được thêm tiền tố "p2_" (cmt_5 -> p2_cmt_5) để không trùng id Phase 1.
  3. Chỉ khử trùng GIỮA 2 phase theo cleaned_text (giữ bản Phase 1).
     Không động vào Phase 1 -> số liệu Phase 1 giữ nguyên như báo cáo cũ (31,931 dòng).

Đầu ra: data/comments_labeled_merged.csv
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

DEFAULT_PHASE1_CSV = DATA_DIR / "comments_labeled_export.csv"
DEFAULT_PHASE2_CSV = DATA_DIR / "comments_labeled_phase2.csv"
DEFAULT_OUTPUT_CSV = DATA_DIR / "comments_labeled_merged.csv"

COLUMNS = ["id", "label", "cleaned_text", "text", "comment_type", "reaction_count"]
VALID_LABELS = {"spam", "ham"}


def load_phase2(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    # File từ auto_accept_labels.py --skip-argilla: nhãn nằm ở cột official_label, không có cột id
    if "label" not in df.columns and "official_label" in df.columns:
        df["id"] = [f"cmt_{i}" for i in range(len(df))]
        df["label"] = df["official_label"]

    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df = df[df["label"].isin(VALID_LABELS)].copy()

    for col in COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    df["id"] = "p2_" + df["id"].astype(str)
    return df[COLUMNS]


def norm(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip()


def label_stats(df: pd.DataFrame) -> str:
    n = len(df)
    spam = int((df["label"] == "spam").sum())
    ham = int((df["label"] == "ham").sum())
    if n == 0:
        return "0"
    return f"{n:,} (ham {ham:,} ≈ {ham / n * 100:.2f}% | spam {spam:,} ≈ {spam / n * 100:.2f}%)"


def parse_args():
    parser = argparse.ArgumentParser(description="Bước 2e: gộp dữ liệu gán nhãn Phase 1 + Phase 2.")
    parser.add_argument("--phase1", type=str, default=str(DEFAULT_PHASE1_CSV),
                        help=f"CSV Phase 1 (mặc định: {DEFAULT_PHASE1_CSV})")
    parser.add_argument("--phase2", type=str, default=str(DEFAULT_PHASE2_CSV),
                        help=f"CSV Phase 2 đã gán nhãn (mặc định: {DEFAULT_PHASE2_CSV})")
    parser.add_argument("--output", "-o", type=str, default=str(DEFAULT_OUTPUT_CSV),
                        help=f"CSV đầu ra (mặc định: {DEFAULT_OUTPUT_CSV})")
    return parser.parse_args()


def main():
    args = parse_args()
    p1_path, p2_path, out_path = Path(args.phase1), Path(args.phase2), Path(args.output)

    for p in (p1_path, p2_path):
        if not p.exists():
            print(f"[ERROR] Không tìm thấy file: {p}")
            sys.exit(1)

    print("=" * 65)
    print(" BƯỚC 2e: GỘP DỮ LIỆU GÁN NHÃN PHASE 1 + PHASE 2")
    print("=" * 65)

    df1 = pd.read_csv(p1_path)
    for col in COLUMNS:
        if col not in df1.columns:
            df1[col] = pd.NA
    df1 = df1[COLUMNS].copy()
    df1["source"] = "phase1"

    df2 = load_phase2(p2_path)
    n2_raw = len(df2)

    p1_texts = set(norm(df1["cleaned_text"]))
    p1_texts.discard("")
    dup_mask = norm(df2["cleaned_text"]).isin(p1_texts)
    df2 = df2[~dup_mask].copy()
    df2["source"] = "phase2"

    merged = pd.concat([df1, df2], ignore_index=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"  - Phase 1:                 {label_stats(df1)}")
    print(f"  - Phase 2 (có nhãn):       {n2_raw:,}")
    print(f"  - Phase 2 bỏ (trùng P1):   {int(dup_mask.sum()):,}")
    print(f"  - Phase 2 giữ lại:         {label_stats(df2)}")
    print(f"  - TỔNG SAU KHI GỘP:        {label_stats(merged)}")
    print(f"  -> Đã lưu CSV: {out_path}")


if __name__ == "__main__":
    main()
