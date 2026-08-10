from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAX_TRACKED_BYTES = 20 * 1024 * 1024
FORBIDDEN_PATH_PARTS = {
    "__pycache__",
    ".ipynb_checkpoints",
    "output",
    "outputs",
    "models",
    "models_human",
    "videos",
}
FORBIDDEN_SUFFIXES = {".safetensors", ".ckpt", ".pt", ".pth", ".mp4", ".zip", ".log"}
SECRET_PATTERNS = {
    "private key": re.compile(r"BEGIN (?:RSA|OPENSSH|EC) PRIVATE KEY"),
    "GitHub token": re.compile(r"ghp_[A-Za-z0-9]{30,}"),
    "OpenAI-style key": re.compile(r"sk-[A-Za-z0-9_-]{20,}"),
    "Google API key": re.compile(r"AIza[0-9A-Za-z_-]{30,}"),
    "private server path": re.compile(r"/root/autodl-(?:tmp|fs)/"),
}
CONTENT_PATTERNS = {
    "legacy project name": re.compile("human" + r"[ _-]?omni(?:v2)?", re.IGNORECASE),
    "Chinese text": re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]"),
}


def tracked_files() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "-co", "--exclude-standard"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return [ROOT / line for line in result.stdout.splitlines() if line]


def main() -> int:
    problems: list[str] = []
    for path in tracked_files():
        relative = path.relative_to(ROOT)
        if any(part in FORBIDDEN_PATH_PARTS for part in relative.parts):
            problems.append(f"forbidden path: {relative}")
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            problems.append(f"forbidden artifact: {relative}")
        if path.stat().st_size > MAX_TRACKED_BYTES:
            problems.append(f"large file ({path.stat().st_size} bytes): {relative}")
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for label, pattern in SECRET_PATTERNS.items():
            if pattern.search(text):
                problems.append(f"{label}: {relative}")
        if relative.as_posix() != "tools/check_release.py":
            for label, pattern in CONTENT_PATTERNS.items():
                if pattern.search(text):
                    problems.append(f"{label}: {relative}")

    if not (ROOT / "LICENSE").exists():
        problems.append("missing LICENSE: verify the upstream license before public release")

    if problems:
        print("Release check failed:")
        for item in sorted(set(problems)):
            print(f"- {item}")
        return 1
    print("Release check passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
