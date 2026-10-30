#!/usr/bin/env python
"""Start gpu_keepalive.py as a detached background process and print its PID."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import subprocess
import sys
import time


DEFAULT_PROJECT_ROOT = "/mnt/bn/isp-traindata-lf-1/wq/OmniZip"


def find_existing(script: Path) -> str | None:
    pattern = re.escape(str(script))
    try:
        proc = subprocess.run(
            ["pgrep", "-f", pattern],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except FileNotFoundError:
        return None
    if proc.returncode != 0:
        return None
    for line in proc.stdout.splitlines():
        pid = line.strip()
        if pid.isdigit() and pid != str(os.getpid()):
            return pid
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=os.environ.get("PROJECT_ROOT", DEFAULT_PROJECT_ROOT))
    parser.add_argument("--target-util", default=os.environ.get("TARGET_UTIL", "0.3"))
    parser.add_argument("--mem-gb", default=os.environ.get("MEM_GB", "0.6"))
    parser.add_argument("--device", default=os.environ.get("DEVICE", "0"))
    parser.add_argument("--work-ms", default=os.environ.get("WORK_MS", None))
    parser.add_argument("--log", default=None)
    parser.add_argument("--force", action="store_true", help="Start another keepalive even if one is already running.")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    script = project_root / "scripts" / "gpu_keepalive.py"
    python = project_root / ".venv" / "bin" / "python"
    log_path = Path(args.log or os.environ.get("KEEPALIVE_LOG", project_root / "logs" / "keepalive.log"))
    log_path.parent.mkdir(parents=True, exist_ok=True)

    if not script.exists():
        print(f"ERROR: missing keepalive script: {script}", file=sys.stderr)
        return 1
    if not python.exists():
        print(f"ERROR: missing python interpreter: {python}", file=sys.stderr)
        return 1

    if not args.force:
        existing = find_existing(script)
        if existing:
            print(existing)
            return 0

    cmd = [
        str(python),
        str(script),
        "--target-util",
        str(args.target_util),
        "--mem-gb",
        str(args.mem_gb),
        "--device",
        str(args.device),
    ]
    if args.work_ms:
        cmd.extend(["--work-ms", str(args.work_ms)])

    env = os.environ.copy()
    env["PROJECT_ROOT"] = str(project_root)
    with log_path.open("ab", buffering=0) as log_fh:
        proc = subprocess.Popen(
            cmd,
            cwd=str(project_root),
            stdin=subprocess.DEVNULL,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=env,
        )

    time.sleep(3)
    if proc.poll() is not None:
        print(f"ERROR: keepalive exited early with code {proc.returncode}; log={log_path}", file=sys.stderr)
        return int(proc.returncode or 1)

    print(proc.pid)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
