"""pipeline/collect/snapshots.py
Đọc/tải snapshot đã lưu theo manifest, phục vụ notebook thu thập và audit.

Đầu vào: cache nguồn, tên snapshot và manifest URL/SHA-256 của nhóm.
Xử lý: tìm file flat hoặc phân tầng; chỉ tải file thiếu, kiểm tra hash trước khi lưu.
Đầu ra: đường dẫn file đúng hash; không cào Facebook hoặc tạo nhãn.
Liên quan: notebook 01; tools.dataset_versions.audit_source.
"""

import hashlib
import json
import argparse
from pathlib import Path
import shutil
import tempfile
from urllib.request import Request, urlopen

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_DIR = PROJECT_ROOT / ".cache" / "collection"
SOURCE_LOCATIONS = {
    "groups_data_phase2.json": "raw",
    "comments_phase2.csv": "intermediate",
    "comments_phase2_cleaned.csv": "intermediate",
    "predictions_glm_cache_phase2.json": "intermediate",
    "comments_labeled_phase2.csv": "labeled",
    "comments_labeled_export.csv": "labeled",
    "comments_labeled_merged.csv": "labeled",
}


def snapshot_path(data_root, name):
    """Giữ tên snapshot lịch sử; hỗ trợ cả bundle flat và data phân tầng mới."""
    folder = SOURCE_LOCATIONS[name]
    flat = Path(data_root) / name
    return flat if flat.is_file() else Path(data_root) / folder / name


def ensure_snapshots(data_root, names=None):
    """Không ghi đè file khác hash; không dùng nội dung trang HTML làm dữ liệu."""
    manifest = json.loads((PROJECT_ROOT / "results/source_audit/drive_manifest.json").read_text(encoding="utf-8"))
    entries = {item["name"]: item for item in manifest}
    result = {}
    for name in SOURCE_LOCATIONS if names is None else names:
        entry, path = entries[name], snapshot_path(data_root, name)
        if path.exists():
            content = path.read_bytes()
        else:
            request = Request(entry["download_url"], headers={"User-Agent": "Mozilla/5.0"})
            with urlopen(request, timeout=90) as response:
                content = response.read(entry["bytes"] + 1)
        if hashlib.sha256(content).hexdigest() != entry["sha256"]:
            raise ValueError("Snapshot khác hash đã chốt; kiểm tra file/quyền/mạng: " + name)
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        result[name] = path
    return result


def collection_inputs(data_root=DEFAULT_SOURCE_DIR):
    """Tên logic cho đầu vào notebook; tên/hash bản nguồn giữ trong manifest."""
    names = {"posts": "groups_data_phase2.json",
             "annotation_index": "comments_phase2_cleaned.csv",
             "predictions": "predictions_glm_cache_phase2.json"}
    paths = ensure_snapshots(data_root, names.values())
    return {key: paths[name] for key, name in names.items()}


def prepare_phase1(data_root, source_root=DEFAULT_SOURCE_DIR):
    """Lưu CSV có nhãn có sẵn, giữ nguyên byte và kiểm hash theo manifest."""
    name = "comments_labeled_export.csv"
    manifest = json.loads((PROJECT_ROOT / "results/source_audit/drive_manifest.json").read_text(encoding="utf-8"))
    entry = next(item for item in manifest if item["name"] == name)
    target = Path(data_root) / "labeled" / "comments_labeled_phase1.csv"
    if target.exists():
        if hashlib.sha256(target.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError("CSV có sẵn khác hash nguồn; không tự ghi đè: " + str(target))
        return target
    Path(source_root).mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="phase1_", dir=source_root) as temporary:
        source = ensure_snapshots(temporary, [name])[name]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    return target


def main():
    parser = argparse.ArgumentParser(description="Chuẩn bị đầu vào tái lập và kiểm SHA-256.")
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--include-phase1", action="store_true", help="Chuẩn bị thêm CSV có nhãn để gộp dữ liệu")
    args = parser.parse_args()
    for role, path in collection_inputs(args.source_dir).items():
        print(role + ":", path)
    if args.include_phase1:
        print("phase1:", prepare_phase1(args.data_dir, args.source_dir))


if __name__ == "__main__":
    main()
