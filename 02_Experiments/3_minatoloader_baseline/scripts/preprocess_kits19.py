#!/usr/bin/env python3
"""
KiTS19 Preprocessing Script
============================
Calls MinatoLoader's original preprocess_dataset.py on our 80-case subset.
Outputs .npy files into thesis/00_datasets/kits19_preproc/

Usage:
    source ../.venv_minato/bin/activate
    python preprocess_kits19.py

Does NOT modify any file inside MinatoLoader/.
"""
import os
import sys
import subprocess
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────
SCRIPT_DIR     = Path(__file__).parent
BASE_DIR       = SCRIPT_DIR.parent
THESIS_DIR     = BASE_DIR.parent.parent   # thesis/
PREPROC_OUT    = THESIS_DIR / "00_datasets" / "kits19_preproc"
KITS19_RAW     = BASE_DIR / "MinatoLoader" / "raw-data-dir" / "kits19" / "data"
PREPROCESS_PY  = BASE_DIR / "MinatoLoader" / "preprocess_dataset.py"
EVAL_TXT       = BASE_DIR / "MinatoLoader" / "Minato" / "evaluation_cases.txt"

# ── Verify inputs ─────────────────────────────────────────────
assert PREPROCESS_PY.exists(), f"Not found: {PREPROCESS_PY}"
assert KITS19_RAW.exists(), f"Not found: {KITS19_RAW}"
PREPROC_OUT.mkdir(parents=True, exist_ok=True)

# ── Figure out which cases to preprocess ─────────────────────
with open(EVAL_TXT) as f:
    eval_cases = set(line.strip() for line in f if line.strip())

all_cases    = [f"{i:05d}" for i in range(210)]
train_cases  = [c for c in all_cases if c not in eval_cases][:39]
all_target   = sorted(list(eval_cases) + train_cases)

# Only process cases that have imaging.nii.gz
available = [c for c in all_target
             if (KITS19_RAW / f"case_{c}" / "imaging.nii.gz").exists()]

print(f"📊 Cases with imaging available: {len(available)}/80")
print(f"📂 Output dir: {PREPROC_OUT}")

# ── Check which already preprocessed ─────────────────────────
def is_preprocessed(case_id):
    x = PREPROC_OUT / f"case_{case_id}_x.npy"
    y = PREPROC_OUT / f"case_{case_id}_y.npy"
    return x.exists() and y.exists()

already_done = [c for c in available if is_preprocessed(c)]
to_process   = [c for c in available if not is_preprocessed(c)]

print(f"✅ Already preprocessed: {len(already_done)}")
print(f"⏳ To process: {len(to_process)}")

if not to_process:
    print("\n✅ All cases already preprocessed!")
    sys.exit(0)

# ── Run preprocess_dataset.py (their code, our paths) ─────────
# preprocess_dataset.py takes --data_dir (raw NIfTI) --results_dir (output npy)
# It processes ALL cases found in data_dir, so we run it with a temp symlink dir
# that only contains our 80 cases, keeping their code untouched.

import tempfile
import shutil

print(f"\n🔧 Creating temp dir with {len(to_process)} case symlinks...")
with tempfile.TemporaryDirectory(prefix="kits19_subset_") as tmpdir:
    tmp_path = Path(tmpdir)

    # Create symlinks to only our target cases
    for case_id in to_process:
        src = KITS19_RAW / f"case_{case_id}"
        dst = tmp_path / f"case_{case_id}"
        dst.symlink_to(src)

    print(f"✅ Created {len(to_process)} symlinks in {tmp_path}")
    print(f"\n🚀 Running preprocess_dataset.py...")
    print(f"   Input:  {tmp_path}")
    print(f"   Output: {PREPROC_OUT}")
    print()

    cmd = [
        sys.executable,
        str(PREPROCESS_PY),
        "--data_dir",    str(tmp_path),
        "--results_dir", str(PREPROC_OUT),
    ]

    result = subprocess.run(cmd, cwd=str(BASE_DIR / "MinatoLoader"))

if result.returncode == 0:
    # Verify outputs
    done_now = [c for c in available if is_preprocessed(c)]
    print(f"\n✅ Preprocessing complete!")
    print(f"   Preprocessed: {len(done_now)}/{len(available)} cases")

    # Check sizes
    npy_files  = list(PREPROC_OUT.glob("*.npy"))
    total_size = sum(f.stat().st_size for f in npy_files) / 1024**3
    print(f"   Total size: {total_size:.1f} GB ({len(npy_files)} files)")
else:
    print(f"\n❌ Preprocessing failed with code {result.returncode}")
    sys.exit(1)
