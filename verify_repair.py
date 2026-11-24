"""Bounded CPU verification; no dataset download or full training campaign."""

import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time

import torch


ROOT = Path(__file__).resolve().parent


def run(args, *, timeout=90):
    result = subprocess.run([sys.executable, *args], cwd=ROOT, capture_output=True, text=True, timeout=timeout)
    print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, file=sys.stderr, end="")
    result.check_returncode()
    return result


def main():
    start = time.monotonic()
    artifacts = ROOT / "artifacts"
    artifacts.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="verify-", dir=artifacts) as directory:
        directory = Path(directory)
        result = run(["-m", "pytest", "-q", "tests", "--basetemp", str(directory / "pytest")])
        match = re.search(r"(\d+) passed", result.stdout)
        assert match and int(match.group(1)) > 0, "No actual tests passed"
        checkpoint = directory / "cli.pt"
        samples = directory / "cli.png"
        command = ["train.py", "--smoke-test", "--device", "cpu", "--max-batches", "1",
                   "--checkpoint", str(checkpoint), "--sample-output", str(samples)]
        run(command)
        first = torch.load(checkpoint, map_location="cpu", weights_only=True)
        run(command + ["--resume", str(checkpoint)])
        second = torch.load(checkpoint, map_location="cpu", weights_only=True)
        assert first["epoch"] == 1 and second["epoch"] == 2
        assert samples.stat().st_size > 0
        for key in first["optimizer_state"]["state"]:
            assert int(second["optimizer_state"]["state"][key]["step"]) == int(first["optimizer_state"]["state"][key]["step"]) + 1
        summary = {"passed_tests": int(match.group(1)), "cli_fresh_start_and_resume": True,
                   "safe_weights_only_checkpoint": True, "device_tested": "cpu", "torch": torch.__version__,
                   "elapsed_seconds": round(time.monotonic() - start, 2),
                   "data": "synthetic smoke data; no Fashion-MNIST training or quality result"}
    (artifacts / "verification.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print("FASHION_REPAIR_VERIFIED")


if __name__ == "__main__":
    main()
