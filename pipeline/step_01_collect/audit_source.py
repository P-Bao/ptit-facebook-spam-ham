"""Đối soát snapshot thu thập → CSV → nhãn → bản gộp → v7, không gọi Facebook/LLM."""

import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import re

import pandas as pd


SOURCE_FILES = (
    "groups_data_phase2.json", "comments_phase2.csv", "comments_phase2_cleaned.csv",
    "predictions_glm_cache_phase2.json", "comments_labeled_phase2.csv",
    "comments_labeled_export.csv", "comments_labeled_merged.csv",
)


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_configured_groups(crawler_dir):
    """Chỉ đọc AST literal, không import/chạy code crawler hoặc đọc phiên đăng nhập."""
    path = Path(crawler_dir) / "scrape_ptit_groups.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "PTIT_GROUPS" for t in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError("Không tìm được PTIT_GROUPS literal trong mã bàn giao")


def inspect_crawler(crawler_dir):
    inventory = []
    for path in sorted(Path(crawler_dir).iterdir()):
        if not path.is_file() or path.suffix not in {".py", ".md", ".txt", ".example"}:
            continue
        text = path.read_text(encoding="utf-8")
        item = {"name": path.name, "bytes": path.stat().st_size,
                "lines": len(text.splitlines()), "sha256": sha256_file(path)}
        if path.suffix == ".py":
            tree = ast.parse(text, filename=path.name)
            item["syntax"] = "valid"
            item["definitions"] = [{"name": n.name, "line": n.lineno, "kind": type(n).__name__}
                                   for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
        inventory.append(item)
    return inventory


def _read_csv(path):
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def _require(frame, columns, name):
    missing = set(columns) - set(frame)
    if missing:
        raise ValueError(f"{name} thiếu cột {sorted(missing)}")


def _checked_labels(frame, name):
    _require(frame, ["id", "label", "cleaned_text", "text", "comment_type", "reaction_count"], name)
    if frame.id.eq("").any() or frame.id.duplicated().any():
        raise ValueError(f"{name} có ID trùng/rỗng; không thể đối chiếu một-một")
    if not frame.label.isin(["ham", "spam"]).all():
        raise ValueError(f"{name} có nhãn ngoài ham/spam")


def audit_source_bundle(source_dir, training_csv, clean_fn, flatten_fn, configured_groups=None):
    """Tính số liệu từ file thật; không tự sửa dữ liệu hoặc suy diễn người đã rà nhãn."""
    source_dir = Path(source_dir)
    for name in SOURCE_FILES:
        if not (source_dir / name).is_file():
            raise FileNotFoundError(source_dir / name)
    posts = json.loads((source_dir / SOURCE_FILES[0]).read_text(encoding="utf-8-sig"))
    if not isinstance(posts, list) or not posts or not all(isinstance(p, dict) for p in posts):
        raise ValueError("JSON thô phải là danh sách bài đăng không rỗng")
    phase1 = _read_csv(source_dir / "comments_labeled_export.csv")
    phase2 = _read_csv(source_dir / "comments_labeled_phase2.csv")
    merged = _read_csv(source_dir / "comments_labeled_merged.csv")
    training = _read_csv(training_csv)
    for name, frame in [("phase1", phase1), ("phase2", phase2), ("merged", merged), ("training", training)]:
        _checked_labels(frame, name)
    flat_csv = _read_csv(source_dir / "comments_phase2.csv")
    cleaned_csv = _read_csv(source_dir / "comments_phase2_cleaned.csv")
    _require(flat_csv, ["post_id", "comment_type", "text"], "flat CSV")
    _require(cleaned_csv, [*flat_csv.columns, "cleaned_text"], "cleaned CSV")
    flat_rows = flatten_fn(posts)
    nonempty = [r for r in flat_rows if r["text"]]
    phase1_texts = {str(row.cleaned_text).strip() or clean_fn(row.text) for row in phase1.itertuples()}
    phase1_texts.discard("")
    kept = [r for r in nonempty if clean_fn(r["text"]) not in phase1_texts]
    expected_flat = pd.DataFrame(kept, columns=flat_csv.columns).fillna("").astype(str)
    raw_to_flat_exact = expected_flat.equals(flat_csv)

    # Snapshot cũ khử trùng nhưng giữ dòng clean rỗng/chỉ dấu câu. Không dịch chuyển ID cũ.
    dedup = flat_csv.drop_duplicates(["post_id", "comment_type", "text"]).reset_index(drop=True)
    expected_clean = dedup.copy()
    expected_clean["cleaned_text"] = expected_clean.text.map(clean_fn)
    expected_clean = expected_clean[cleaned_csv.columns]
    legacy_clean_exact = expected_clean.equals(cleaned_csv)
    punctuation_only = ~expected_clean.cleaned_text.str.contains(r"[0-9a-zA-ZÀ-ỹ]", regex=True)

    indices = pd.to_numeric(phase2.id.str.extract(r"^(?:p2_)?cmt_(\d+)$", expand=False), errors="coerce")
    if indices.isna().any() or indices.ge(len(cleaned_csv)).any():
        raise ValueError("ID Phase 2 không nối được snapshot cleaned")
    matched = cleaned_csv.iloc[indices.astype(int)].reset_index(drop=True)
    raw_columns_match = all(phase2[c].equals(matched[c]) for c in ["text", "comment_type", "reaction_count"])
    cleaned_changed = phase2.cleaned_text.ne(matched.cleaned_text)
    fallback_explains = ((matched.cleaned_text.eq("")) & phase2.cleaned_text.eq(matched.text))
    unexplained_text_edits = int((cleaned_changed & ~fallback_explains).sum())

    cache = json.loads((source_dir / "predictions_glm_cache_phase2.json").read_text(encoding="utf-8"))
    if not isinstance(cache, dict):
        raise ValueError("Cache phải là object ID → dự đoán")
    high_confidence = set()
    bad_predictions = 0
    for rid, item in cache.items():
        try:
            confidence = float(item.get("confidence", 0))
        except (TypeError, ValueError):
            confidence = float("nan")
        if item.get("label") not in {"ham", "spam"} or not math.isfinite(confidence) or not 0 <= confidence <= 1:
            bad_predictions += 1
        elif confidence >= 0.75:
            high_confidence.add(rid)
    official_ids = set(phase2.id)
    cached_official = phase2[phase2.id.isin(cache)]
    label_disagreements = sum(cache[r.id].get("label") != r.label for r in cached_official.itertuples())

    p2 = phase2.copy()
    p2["id"] = p2.id.map(lambda v: v if v.startswith("p2_") else "p2_" + v)
    duplicates = p2.cleaned_text.str.strip().isin(set(phase1.cleaned_text.str.strip()) - {""})
    expected_merge = pd.concat([phase1.assign(source="phase1"), p2.loc[~duplicates].assign(source="phase2")], ignore_index=True)
    _require(merged, ["source"], "merged")
    merge_exact = expected_merge[merged.columns].equals(merged)

    old = merged.set_index("id")
    new = training.set_index("id")
    shared = old.index.intersection(new.index)
    same_ids = set(old.index) == set(new.index)
    changes = {column: int(old.loc[shared, column].ne(new.loc[shared, column]).sum())
               for column in ["label", "cleaned_text", "text", "comment_type", "reaction_count"]}
    transitions = []
    for before, after in [("ham", "spam"), ("spam", "ham")]:
        count = int(((old.loc[shared, "label"] == before) & (new.loc[shared, "label"] == after)).sum())
        transitions.append({"from": before, "to": after, "count": count})

    configured = {str(g["id"]): g for g in configured_groups or []}
    groups = []
    for gid in sorted({str(p.get("group_id", "")) for p in posts}):
        group_posts = [p for p in posts if str(p.get("group_id", "")) == gid]
        comments = [c for p in group_posts for c in p.get("comments", []) or []]
        replies = [r for c in comments for r in c.get("replies", []) or []]
        groups.append({"group_id": gid, "names_in_snapshot": sorted({p.get("group_name", "") for p in group_posts}),
            "url_from_id": "https://www.facebook.com/groups/" + gid + "/",
            "posts": len(group_posts), "comments": len(comments), "replies": len(replies),
            "listed_in_handoff_config": gid in configured,
            "configured_name": configured.get(gid, {}).get("name")})
    def span(field):
        values = [p[field] for p in posts if p.get(field)]
        return {"min": min(values) if values else None, "max": max(values) if values else None,
                "note": "Giá trị trong snapshot; không tự gán múi giờ hoặc suy ra lệnh crawl đã dùng"}

    frames = {"phase1": phase1, "phase2_export": phase2, "merged_drive": merged, "training_v7": training}
    return {
        "files": [{"name": name, "bytes": (source_dir/name).stat().st_size, "sha256": sha256_file(source_dir/name)} for name in SOURCE_FILES],
        "training_sha256": sha256_file(training_csv),
        "raw": {"posts": len(posts), "groups": len(groups), "comments": sum(g["comments"] for g in groups),
                "replies": sum(g["replies"] for g in groups),
                "duplicate_post_ids": len(posts)-len({str(p.get("post_id")) for p in posts}),
                "posts_reporting_comments_but_empty": sum(bool(p.get("comment_count")) and not p.get("comments") for p in posts),
                "created_at": span("created_at"), "scraped_at": span("scraped_at")},
        "groups": groups, "configured_groups_count": len(configured),
        "raw_group_ids_not_in_config": [g["group_id"] for g in groups if configured and not g["listed_in_handoff_config"]],
        "conversion": {"raw_rows": len(flat_rows), "empty_removed": len(flat_rows)-len(nonempty),
            "phase1_duplicate_removed": len(nonempty)-len(kept), "flat_rows": len(flat_csv), "exact_reproduction": raw_to_flat_exact},
        "cleaning": {"duplicate_rows_removed": len(flat_csv)-len(dedup), "legacy_rows": len(cleaned_csv),
            "legacy_exact_reproduction": legacy_clean_exact, "current_filter_would_remove": int(punctuation_only.sum()),
            "current_default_rows": int((~punctuation_only).sum()), "empty_cleaned_rows_in_snapshot": int(cleaned_csv.cleaned_text.eq("").sum())},
        "annotation": {"cache_rows": len(cache), "invalid_predictions": bad_predictions, "confidence_ge_075": len(high_confidence),
            "official_rows": len(phase2), "high_confidence_without_official": len(high_confidence-official_ids),
            "official_not_in_cache": len(official_ids-set(cache)), "cached_official_label_disagreements": int(label_disagreements),
            "metadata_matches_cleaned_snapshot": raw_columns_match, "cleaned_text_fallback_to_raw": int((cleaned_changed & fallback_explains).sum()),
            "unexplained_cleaned_text_changes": unexplained_text_edits},
        "merge": {"cross_phase_duplicate_removed": int(duplicates.sum()), "phase2_retained": int((~duplicates).sum()),
                  "rows": len(merged), "exact_reproduction": merge_exact},
        "labels": {name: frame.label.value_counts().to_dict() for name, frame in frames.items()},
        "v7_comparison": {"same_id_set": same_ids, "same_id_order": old.index.tolist()==new.index.tolist(),
            "shared_ids": len(shared), "drive_only": len(old.index.difference(new.index)), "v7_only": len(new.index.difference(old.index)),
            "changed_fields": changes, "label_transitions": transitions},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--training-csv", type=Path, required=True)
    parser.add_argument("--crawler-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from .convert_phase2_json import flatten_posts
    from ..step_02_preprocess.clean_data_facebook_spamham import preprocess
    configured = read_configured_groups(args.crawler_dir) if args.crawler_dir else None
    result = audit_source_bundle(args.source_dir, args.training_csv, preprocess, flatten_posts, configured)
    if args.crawler_dir:
        result["crawler_inventory"] = inspect_crawler(args.crawler_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Đã lưu đối soát nguồn:", args.output)
    print("Raw posts:", result["raw"]["posts"], "| raw comments/replies:", result["conversion"]["raw_rows"])
    print("Nhãn thay đổi so với bản gộp Drive:", result["v7_comparison"]["changed_fields"]["label"])


if __name__ == "__main__":
    main()
