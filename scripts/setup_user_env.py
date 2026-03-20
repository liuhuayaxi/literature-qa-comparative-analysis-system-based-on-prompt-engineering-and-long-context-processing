"""Set up a user-scoped runtime environment without administrator privileges."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
VENV_DIR = ROOT / ".venv"
KERNEL_NAME = "literature-qa-system"
KERNEL_DISPLAY_NAME = "Literature QA System (.venv)"
JUPYTER_PACKAGES = ["ipykernel", "jupyterlab", "notebook"]


def _venv_python(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def _venv_jupyter(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "jupyter.exe"
    return venv_dir / "bin" / "jupyter"


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.check_call(cmd, cwd=ROOT)


def _setup_venv() -> None:
    if not VENV_DIR.exists():
        _run([sys.executable, "-m", "venv", str(VENV_DIR)])
    python_bin = _venv_python(VENV_DIR)
    _run([str(python_bin), "-m", "pip", "install", "--upgrade", "pip"])
    _run(
        [
            str(python_bin),
            "-m",
            "pip",
            "install",
            "-r",
            str(ROOT / "requirements.txt"),
            *JUPYTER_PACKAGES,
        ]
    )
    _run(
        [
            str(python_bin),
            "-m",
            "ipykernel",
            "install",
            "--user",
            "--name",
            KERNEL_NAME,
            "--display-name",
            KERNEL_DISPLAY_NAME,
        ]
    )
    print()
    print("Environment is ready.")
    print(f"Virtual environment: {VENV_DIR}")
    print(f"Jupyter kernel: {KERNEL_DISPLAY_NAME}")
    print(f"Launch command: {_venv_jupyter(VENV_DIR)} lab {ROOT / 'notebooks' / 'course_research_assistant.ipynb'}")


def _fallback_user_site() -> None:
    print("Virtual environment setup failed. Falling back to user-site installation.")
    _run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--user",
            "-r",
            str(ROOT / "requirements.txt"),
            *JUPYTER_PACKAGES,
        ]
    )
    _run(
        [
            sys.executable,
            "-m",
            "ipykernel",
            "install",
            "--user",
            "--name",
            f"{KERNEL_NAME}-user",
            "--display-name",
            "Literature QA System (user site)",
        ]
    )
    print()
    print("User-site environment is ready.")
    print(f"Launch command: {sys.executable} -m jupyter lab {ROOT / 'notebooks' / 'course_research_assistant.ipynb'}")


def main() -> None:
    try:
        _setup_venv()
    except Exception as exc:
        print(f"venv setup failed: {exc}")
        _fallback_user_site()


if __name__ == "__main__":
    main()
