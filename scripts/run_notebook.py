"""scripts/run_notebook.py
Chạy notebook trong kernel mới với module của project và output riêng.

Đầu vào: --notebook 01/02/03/04, CSV v7 và các file thu thập/model local.
Xử lý: cấu hình PYTHONPATH/seed; ghi tiến độ và trạng thái chạy.
Đầu ra: data/ (01), artifacts/spamham/ (02), artifacts/modern/vibert/ (03),
artifacts/grid_audit/ (04), trạng thái .work/.
Liên quan: bốn notebook; 03 mặc định review checkpoint đã huấn luyện.
"""

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import time

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebook", choices=["01", "02", "03", "04"], default="02")
    parser.add_argument("--csv", type=Path, help="CSV huấn luyện local; mặc định data/final/labeled_v7.csv")
    parser.add_argument("--source-dir", type=Path, default=root / ".cache/collection")
    parser.add_argument("--work-dir", type=Path, help="Thư mục tạm cho kernel và bản sao notebook")
    parser.add_argument("--output-dir", type=Path, help="Gốc đầu ra; data/ (01), artifacts/spamham/ (02), artifacts/modern/vibert/ (03), artifacts/grid_audit/ (04)")
    parser.add_argument("--inplace", action="store_true", help="Ghi output vào notebook chính chỉ khi chạy hết thành công")
    args = parser.parse_args()
    filenames = {"01": "01_thu_thap_va_gan_nhan.ipynb", "02": "02_thuc_nghiem_va_demo.ipynb",
                 "03": "03_vibert.ipynb", "04": "04_kiem_tra_grid_search.ipynb"}
    source = root / "notebooks" / filenames[args.notebook]
    work = (args.work_dir or root / ".work/runs" / args.notebook).resolve()
    if not work.is_relative_to(root):
        raise ValueError("Thư mục kiểm tra phải nằm trong project")
    work.mkdir(parents=True, exist_ok=True)
    defaults = {"01": root / "data", "02": root / "artifacts/spamham", "03": root / "artifacts/modern/vibert",
                "04": root / "artifacts/grid_audit"}
    output = (args.output_dir or defaults[args.notebook]).resolve()
    if not output.is_relative_to(root):
        raise ValueError("Thư mục đầu ra phải nằm trong project")
    csv = args.csv or root / "data/final/labeled_v7.csv"
    os.environ.update({"SPAMHAM_CSV": str(csv.resolve()),
        "SPAMHAM_SOURCE_DIR": str(args.source_dir.resolve()),
        "PYTHONPATH": str(root) + os.pathsep + os.environ.get("PYTHONPATH", ""),
        "SPAMHAM_OUTPUT_DIR": str(output), "SPAMHAM_VIBERT_OUTPUT": str(output), "PYTHONHASHSEED": "42",
        "IPYTHONDIR": str(work / "ipython"), "JUPYTER_RUNTIME_DIR": str(work / "runtime"),
        "MPLCONFIGDIR": str(work / "matplotlib"), "MPLBACKEND": "module://matplotlib_inline.backend_inline",
        "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
    copied = work / source.name
    shutil.copy2(source, copied)
    notebook = nbformat.read(copied, as_version=4)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.execution_count = None
            cell.outputs = []
            cell.metadata.pop("execution", None)
    manager = KernelManager(kernel_name="python3", connection_file=str(work / "connection.json"))
    manager.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
    client = NotebookClient(notebook, km=manager, resources={"metadata": {"path": str(work)}},
                            timeout=3600, allow_errors=False, record_timing=True)
    start = time.monotonic()

    def progress(cell, cell_index, **kwargs):
        if cell.cell_type == "code":
            print(f"[{time.monotonic()-start:.0f}s] Cell {cell_index+1}/{len(notebook.cells)}", flush=True)

    client.on_cell_start = progress
    def save_progress(cell, cell_index, **kwargs):
        nbformat.write(notebook, copied)

    client.on_cell_executed = save_progress
    success = False
    try:
        client.execute()
        success = True
    finally:
        nbformat.write(notebook, copied)
        if manager.has_kernel:
            manager.shutdown_kernel(now=False)
        result = {"success": success, "seconds": round(time.monotonic()-start, 2),
                  "notebook": str(source.relative_to(root)), "isolated_kernel": True, "uses_project_modules": True,
                  "output_dir": str(output.relative_to(root)),
                  "python": sys.version.split()[0],
                  "executed_code_cells": sum(c.cell_type == "code" and c.execution_count is not None for c in notebook.cells)}
        (work / "execution_status.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    if success and args.inplace:
        shutil.copy2(copied, source)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
