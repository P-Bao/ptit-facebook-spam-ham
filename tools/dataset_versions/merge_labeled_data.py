"""tools/dataset_versions/merge_labeled_data.py
Tổng hợp hai CSV đã gán nhãn, không tạo nhãn mới.

Đầu vào: CSV có nhãn có sẵn và data/labeled/comments_labeled.csv từ notebook 01.
Xử lý: Giữ ID nguồn, thêm prefix Phase 2, loại trùng chéo theo cleaned_text, ghi source.
Đầu ra: CSV gộp; không tự tái tạo lịch sử chỉnh nhãn v7.
Liên quan: pipeline.label; audit_source; không cần chạy khi train trực tiếp data/final/labeled_v7.csv.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"

DEFAULT_PHASE1_CSV = DATA_DIR / "labeled" / "comments_labeled_phase1.csv"
DEFAULT_PHASE2_CSV = DATA_DIR / "labeled" / "comments_labeled.csv"
DEFAULT_OUTPUT_CSV = DATA_DIR / "final" / "comments_merged.csv"

COLUMNS = ["id", "label", "cleaned_text", "text", "comment_type", "reaction_count"]
VALID_LABELS = {"spam", "ham"}


def load_phase2(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    # File từ auto_accept_labels.py --skip-argilla: nhãn nằm ở cột official_label, có thể có hoặc chưa có cột id
    if "label" not in df.columns and "official_label" in df.columns:
        if "id" not in df:
            df["id"] = [f"cmt_{i}" for i in range(len(df))]
        df["label"] = df["official_label"]

    if not {"label", "cleaned_text", "id"}.issubset(df.columns):
        raise ValueError("Phase 2 thiếu label/official_label, cleaned_text hoặc id")
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df = df[df["label"].isin(VALID_LABELS)].copy()

    for col in COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    if df["id"].isna().any() or df["id"].astype(str).str.strip().eq("").any():
        raise ValueError("Phase 2 có ID rỗng; không chuyển thành p2_nan")
    # Chỉ thêm tiền tố "p2_" nếu id chưa bắt đầu bằng "p2_"
    df["id"] = df["id"].astype(str).apply(lambda x: x if x.startswith("p2_") else f"p2_{x}")
    validate_labeled(df, "Phase 2")
    return df[COLUMNS]


def validate_labeled(df, name):
    if not {"id", "label", "cleaned_text"}.issubset(df.columns):
        raise ValueError(f"{name} thiếu cột bắt buộc")
    if df["id"].isna().any() or df["id"].astype(str).str.strip().eq("").any() or df["id"].duplicated().any():
        raise ValueError(f"{name} có ID rỗng/trùng")
    if not df["label"].isin(VALID_LABELS).all():
        raise ValueError(f"{name} có nhãn không hợp lệ")
    if df["cleaned_text"].isna().any() or df["cleaned_text"].astype(str).str.strip().eq("").any():
        raise ValueError(f"{name} có cleaned_text rỗng")


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
    validate_labeled(df1, "Phase 1")
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
    validate_labeled(merged, "Tập gộp")
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
