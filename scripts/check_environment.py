"""Kiểm tra Python, môi trường .venv và phiên bản thư viện dùng cho notebook."""

from importlib import metadata
from pathlib import Path
import sys


def main():
    root = Path(__file__).resolve().parents[1]
    print("Python:", sys.version.split()[0])
    print("Interpreter:", sys.executable)
    errors = []
    if sys.version_info[:2] != (3, 12):
        errors.append("Python must be 3.12")
    if Path(sys.prefix).resolve() != (root / ".venv").resolve():
        errors.append("Use the interpreter in this project's .venv")
    for line in (root / "requirements/notebook.txt").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, expected = line.split("==", 1)
        try:
            actual = metadata.version(name)
        except metadata.PackageNotFoundError:
            actual = "missing"
        if actual != expected:
            errors.append(f"{name}: expected {expected}, found {actual}")
    if errors:
        raise SystemExit("\n".join(errors) + "\nInstall requirements/notebook.txt using .venv's Python.")
    print("Environment OK: Python 3.12, project .venv, all pinned notebook packages.")


if __name__ == "__main__":
    main()
