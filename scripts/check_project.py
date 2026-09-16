"""Project integrity verification script for academic evaluation."""

import os
import subprocess
import sys
from pathlib import Path

# Add project root to sys.path so scripts can import src modules directly
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))


def run_check():
    print("=" * 70)
    print("  ACADEMIC EVALUATION PRE-CHECK & PROJECT AUDIT")
    print(f"  Root: {project_root}")
    print("=" * 70)

    results = []

    # 1. Required Files
    required_files = [
        "README.md",
        "statement.md",
        "LICENSE",
        "requirements.txt",
        "pyproject.toml",
        "config.yaml",
        ".gitignore",
        "report/project_report.md",
        "examples/README.md",
        "notebooks/README.md",
    ]
    for rel_file in required_files:
        fpath = project_root / rel_file
        status = "PASS" if fpath.is_file() else "FAIL"
        results.append((f"File Check: {rel_file}", status, f"Size: {fpath.stat().st_size} bytes" if fpath.is_file() else "Missing"))

    # 2. Required Directories
    required_dirs = [
        "src",
        "src/data",
        "src/models",
        "src/training",
        "src/evaluation",
        "src/inference",
        "src/utils",
        "scripts",
        "tests",
        "data",
        "models",
        "outputs",
        "report",
        "examples",
        "notebooks",
    ]
    for rel_dir in required_dirs:
        dpath = project_root / rel_dir
        status = "PASS" if dpath.is_dir() else "FAIL"
        results.append((f"Directory Check: {rel_dir}/", status, "Exists" if dpath.is_dir() else "Missing"))

    # 3. Import & Config Check
    try:
        from src.utils.config import load_config
        config = load_config(project_root / "config.yaml")
        results.append(("Configuration Load", "PASS", f"Loaded valid config ({config['project']['title']})"))
    except Exception as e:
        results.append(("Configuration Load", "FAIL", str(e)))

    # 4. CLI Import Check
    try:
        import src.main
        results.append(("CLI Module Import", "PASS", "src.main imported successfully"))
    except Exception as e:
        results.append(("CLI Module Import", "FAIL", str(e)))

    # 5. CLI Help Execution Check
    try:
        cmd = [sys.executable, "-m", "src.main", "--help"]
        res = subprocess.run(cmd, cwd=str(project_root), capture_output=True, text=True, timeout=10)
        if res.returncode == 0 and "Smart Waste" in res.stdout:
            results.append(("CLI --help Execution", "PASS", "Returned 0 with full help menu"))
        else:
            results.append(("CLI --help Execution", "FAIL", f"Exit code {res.returncode}, stderr: {res.stderr[:80]}"))
    except Exception as e:
        results.append(("CLI --help Execution", "FAIL", str(e)))

    # 6. Model Initialization Check
    try:
        from src.models.model import build_model
        model, device = build_model(config, num_classes=6)
        summary = model.get_parameter_summary()
        results.append(("Model Architecture Init", "PASS", f"MobileNetV3-Small loaded ({summary['total']:,} params) on {device.type.upper()}"))
    except Exception as e:
        results.append(("Model Architecture Init", "FAIL", str(e)))

    # Print summary
    failures = 0
    for name, status, detail in results:
        status_tag = f"[{status}]"
        print(f"  {name:<38} {status_tag:<8} {detail}")
        if status == "FAIL":
            failures += 1

    print("=" * 70)
    if failures == 0:
        print(f"  ALL {len(results)} INTEGRITY CHECKS PASSED.")
        print("=" * 70)
        sys.exit(0)
    else:
        print(f"  {failures} INTEGRITY CHECKS FAILED.")
        print("=" * 70)
        sys.exit(1)


if __name__ == "__main__":
    run_check()
