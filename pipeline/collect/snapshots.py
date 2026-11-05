"""pipeline/collect/snapshots.py
Đọc/tải snapshot đã lưu theo manifest, phục vụ notebook thu thập và audit.

Đầu vào: folder data, tên snapshot và manifest URL/SHA-256 của nhóm.
Xử lý: tìm file flat hoặc phân tầng; chỉ tải file thiếu, kiểm tra hash trước khi lưu.
Đầu ra: đường dẫn file đúng hash; không cào Facebook hoặc tạo nhãn.
Liên quan: notebook 01; tools.dataset_versions.audit_source.
"""

import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen

PROJECT_ROOT = Path(__file__).resolve().parents[2]
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


def collection_inputs(data_root):
    """Tên logic cho đầu vào notebook; tên/hash bản nguồn giữ trong manifest."""
    names = {"posts": "groups_data_phase2.json",
             "annotation_index": "comments_phase2_cleaned.csv",
             "predictions": "predictions_glm_cache_phase2.json"}
    paths = ensure_snapshots(data_root, names.values())
    return {key: paths[name] for key, name in names.items()}
