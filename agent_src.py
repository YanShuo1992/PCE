from __future__ import annotations

import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent

SNAPSHOT_DIR = "_agent_src"

ENV = "GEA_AGENT_SRC"

FILES = ["coding_agent.py", "coding_agent_polyglot.py", "requirements.txt", "pytest.ini",
         "llm.py", "llm_withtools.py", "harness_def.py", "model_io.py"]
DIRS = ["tools", "utils", "tests", "prompts"]

def snapshot(run_dir: str | Path, log=print) -> Path:
    dest = Path(run_dir) / SNAPSHOT_DIR
    if dest.is_dir():
        os.environ[ENV] = str(dest.resolve())
        log(f"[agent_src] reusing the existing snapshot at {dest}")
        return dest

    missing = ([n for n in FILES if not (ROOT / n).is_file()] +
               [n + "/" for n in DIRS if not (ROOT / n).is_dir()])
    if missing:
        raise SystemExit("X agent_src: declared but missing from %s: %s" % (ROOT, ", ".join(missing)))

    dest.mkdir(parents=True, exist_ok=True)
    copied = []
    for name in FILES:
        shutil.copy2(ROOT / name, dest / name)
        copied.append(name)
    for name in DIRS:
        shutil.copytree(ROOT / name, dest / name, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        copied.append(name + "/")
    os.environ[ENV] = str(dest.resolve())
    log(f"[agent_src] froze {len(copied)} entries into {dest}: {', '.join(copied)}")
    return dest

def source_root() -> Path:
    p = os.getenv(ENV, "").strip()
    return Path(p) if p and Path(p).is_dir() else ROOT
