"""Extract the curated ZIP into a new folder and reproduce its predictions.

Run the two commands documented in its README using an isolated interpreter.
No pre-existing experiment outputs, trained models or copied results are used.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(archive_path: Path, interpreter: Path) -> dict:
    archive_path, interpreter = archive_path.resolve(), interpreter.resolve()
    if not archive_path.is_file() or not interpreter.is_file():
        raise ValueError("Require an existing ZIP and isolated Python interpreter.")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination = (ROOT / ".tools" / ("submission-check-" + stamp)).resolve()
    if not destination.is_relative_to((ROOT / ".tools").resolve()):
        raise ValueError("Verification folder is outside the workspace tools directory.")
    destination.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(archive_path) as archive:
        for name in archive.namelist():
            if not (destination / name).resolve().is_relative_to(destination):
                raise ValueError("Archive member escapes the verification folder.")
        archive.extractall(destination)
    manifest = json.loads((destination / "submission_manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest["files_sha256"].items():
        if sha256(destination / name) != expected:
            raise ValueError(f"Extracted archive differs from manifest: {name}")
    selected = json.loads((destination / "selected_model.json").read_text(encoding="utf-8"))
    if ((destination / selected["experiment_directory"]).exists()
            or (destination / "outputs/final").exists()):
        raise ValueError("Clean reproduction must start without experiment outputs.")
    results = destination / "results.txt"
    expected = manifest["results_sha256"]
    results.unlink()  # This specific extracted copy must be recreated by inference.
    commands = [
        [str(interpreter), "-m", "pip", "install", "-r", "requirements.txt",
         "--disable-pip-version-check"],
        [str(interpreter), "classify.py"],
    ]
    logs = []
    for number, command in enumerate(commands, 1):
        completed = subprocess.run(command, cwd=destination, text=True, encoding="utf-8",
                                   errors="replace", capture_output=True)
        log_path = destination / f"command_{number}.log"
        log_path.write_text(completed.stdout + completed.stderr, encoding="utf-8")
        logs.append(str(log_path.relative_to(ROOT)))
        if completed.returncode:
            raise RuntimeError(f"Reproduction command {number} failed; see {log_path}")
    actual = sha256(results)
    labels = results.read_text(encoding="utf-8").splitlines()
    if actual != expected or len(labels) != 398:
        raise ValueError("Reproduced results are not byte-identical to the submitted 398 lines.")
    inference = json.loads((destination / "outputs/final/inference.json").read_text(encoding="utf-8"))
    if inference["results_sha256"] != actual or inference["test_metrics"] is not None:
        raise ValueError("Reproduction inference metadata is inconsistent.")
    report = {
        "verified_utc": datetime.now(timezone.utc).isoformat(),
        "archive": str(archive_path), "archive_sha256": sha256(archive_path),
        "clean_directory": str(destination.relative_to(ROOT)),
        "python": str(interpreter), "commands": commands, "logs": logs,
        "candidate_id": inference["selected_model"]["candidate_id"],
        "results_lines": len(labels), "results_sha256": actual,
        "byte_identical_results": True,
        "started_without_cv_outputs_or_results": True,
        "official_test_metrics": None,
    }
    target = ROOT / "output/submission/reproduction.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=ROOT / ".tools/submission-venv/Scripts/python.exe")
    args = parser.parse_args()
    verify(args.archive, args.python)
