import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import nbformat
import pytest


@pytest.mark.parametrize("explicit_csv", [False, True])
def test_runner_keeps_public_outputs_and_allows_default_dataset_download(tmp_path, monkeypatch, explicit_csv):
    source = Path(__file__).resolve().parents[1] / "scripts/run_notebook.py"
    spec = importlib.util.spec_from_file_location("runner_under_test", source)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    monkeypatch.setattr(runner, "__file__", str(tmp_path / "scripts/run_notebook.py"))
    monkeypatch.setattr(runner.os, "environ", {})
    notebook = tmp_path / "notebooks/02_thuc_nghiem_va_demo.ipynb"
    notebook.parent.mkdir(parents=True)
    nbformat.write(nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell("pass")]), notebook)
    csv_path = tmp_path / "custom.csv"
    args = ["runner", "--notebook", "02"] + (["--csv", str(csv_path)] if explicit_csv else [])
    monkeypatch.setattr(runner.sys, "argv", args)
    shutdown_modes = []
    manager = SimpleNamespace(kernel_spec=SimpleNamespace(argv=[]), has_kernel=True,
                              shutdown_kernel=lambda *, now: shutdown_modes.append(now))
    monkeypatch.setattr(runner, "KernelManager", lambda **kwargs: manager)

    class Client:
        def __init__(self, nb, **kwargs): self.notebook = nb
        def execute(self):
            assert runner.os.environ["SPAMHAM_CSV"] == (str(csv_path) if explicit_csv else "")
            assert Path(runner.os.environ["SPAMHAM_OUTPUT_DIR"]) == tmp_path / "artifacts/spamham"
            assert Path(runner.os.environ["SPAMHAM_SOURCE_DIR"]) == tmp_path / ".cache/collection"
            self.notebook.cells[0].execution_count = 1

    monkeypatch.setattr(runner, "NotebookClient", Client)
    runner.main()
    assert shutdown_modes == [False]
    status = json.loads((tmp_path / ".work/runs/02/execution_status.json").read_text(encoding="utf-8"))
    assert status["success"] and status["output_dir"] == str(Path("artifacts/spamham"))
