import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import nbformat
import pytest


@pytest.mark.parametrize("explicit_csv", [False, True])
@pytest.mark.parametrize("number", ["02", "03"])
def test_runner_passes_local_csv_and_keeps_public_outputs(tmp_path, monkeypatch, explicit_csv, number):
    source = Path(__file__).resolve().parents[1] / "scripts/run_notebook.py"
    spec = importlib.util.spec_from_file_location("runner_under_test", source)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    monkeypatch.setattr(runner, "__file__", str(tmp_path / "scripts/run_notebook.py"))
    monkeypatch.setattr(runner.os, "environ", {})
    filename = "02_thuc_nghiem_va_demo.ipynb" if number == "02" else "03_vibert.ipynb"
    expected_output = tmp_path / ("artifacts/spamham" if number == "02" else "artifacts/modern/vibert")
    notebook = tmp_path / "notebooks" / filename
    notebook.parent.mkdir(parents=True)
    nbformat.write(nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell("pass")]), notebook)
    csv_path = tmp_path / "custom.csv"
    args = ["runner", "--notebook", number] + (["--csv", str(csv_path)] if explicit_csv else [])
    monkeypatch.setattr(runner.sys, "argv", args)
    shutdown_modes = []
    manager = SimpleNamespace(kernel_spec=SimpleNamespace(argv=[]), has_kernel=True,
                              shutdown_kernel=lambda *, now: shutdown_modes.append(now))
    monkeypatch.setattr(runner, "KernelManager", lambda **kwargs: manager)

    class Client:
        def __init__(self, nb, **kwargs): self.notebook = nb
        def execute(self):
            assert runner.os.environ["SPAMHAM_CSV"] == str(csv_path if explicit_csv else tmp_path / "data/final/labeled_v7.csv")
            assert Path(runner.os.environ["SPAMHAM_OUTPUT_DIR"]) == expected_output
            assert Path(runner.os.environ["SPAMHAM_VIBERT_OUTPUT"]) == expected_output
            assert Path(runner.os.environ["SPAMHAM_SOURCE_DIR"]) == tmp_path / ".cache/collection"
            self.notebook.cells[0].execution_count = 1

    monkeypatch.setattr(runner, "NotebookClient", Client)
    runner.main()
    assert shutdown_modes == [False]
    status = json.loads((tmp_path / ".work/runs" / number / "execution_status.json").read_text(encoding="utf-8"))
    assert status["success"] and status["output_dir"] == str(expected_output.relative_to(tmp_path))
