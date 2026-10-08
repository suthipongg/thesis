#!/usr/bin/env python3
"""
KiTS19 Subset Downloader
========================
Downloads exactly 80 cases:
  - All 41 evaluation cases (from MinatoLoader/Minato/evaluation_cases.txt)
  - First 39 non-evaluation cases as training set

Usage:
    source ../.venv_minato/bin/activate
    python download_kits19_subset.py

Downloads into: MinatoLoader/raw-data-dir/kits19/data/
"""
import os
import sys
import subprocess
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).parent
BASE_DIR     = SCRIPT_DIR.parent
KITS19_DIR   = BASE_DIR / "MinatoLoader" / "raw-data-dir" / "kits19"
DATA_DIR     = KITS19_DIR / "data"
EVAL_TXT     = BASE_DIR / "MinatoLoader" / "Minato" / "evaluation_cases.txt"
STARTER_CODE = KITS19_DIR / "starter_code"

# ── Read evaluation cases ────────────────────────────────────
with open(EVAL_TXT) as f:
    eval_cases = set(
        line.strip() for line in f if line.strip()
    )  # e.g. {'00000', '00003', ...}

TARGET_TOTAL = 80
TARGET_TRAIN = TARGET_TOTAL - len(eval_cases)

print(f"✅ Evaluation cases: {len(eval_cases)}")
print(f"✅ Training cases to download: {TARGET_TRAIN}")
print(f"✅ Total target: {TARGET_TOTAL}")

# ── Select training cases (first N non-eval cases) ───────────
all_cases = [f"{i:05d}" for i in range(210)]
train_cases = [c for c in all_cases if c not in eval_cases][:TARGET_TRAIN]
all_target  = sorted(list(eval_cases) + train_cases)

print(f"\n📋 Cases to download:")
print(f"  Eval (val): {sorted(eval_cases)[:5]}... ({len(eval_cases)} total)")
print(f"  Train:      {sorted(train_cases)[:5]}... ({len(train_cases)} total)")
print(f"  Total:      {len(all_target)} cases")

# ── Check which already have imaging.nii.gz ──────────────────
def has_imaging(case_id):
    return (DATA_DIR / f"case_{case_id}" / "imaging.nii.gz").exists()

already_done = [c for c in all_target if has_imaging(c)]
to_download  = [c for c in all_target if not has_imaging(c)]

print(f"\n📦 Already have imaging: {len(already_done)}")
print(f"📥 Need to download:     {len(to_download)}")

if not to_download:
    print("\n✅ All cases already downloaded!")
    sys.exit(0)

# ── Patch kits19 get_imaging to only download our subset ─────
# kits19 starter_code/get_imaging.py downloads all 210 cases by default.
# We monkey-patch it by overriding the case list via environment variable.
# Actually the cleanest approach: call their script case-by-case with custom range.

# Read their get_imaging.py to see how it works
get_imaging_path = STARTER_CODE / "get_imaging.py"

# Create a targeted downloader that imports their logic
wrapper_script = f"""
import sys
sys.path.insert(0, str(r'{KITS19_DIR}'))

# Import kits19 helpers
import requests
from pathlib import Path
import json

DATA_DIR = Path(r'{DATA_DIR}')
cases_to_download = {to_download}

def download_case(case_id):
    case_dir = DATA_DIR / f"case_{{case_id:05d}}"
    imaging_path = case_dir / "imaging.nii.gz"
    
    if imaging_path.exists():
        print(f"  ✅ case_{{case_id:05d}} already exists, skipping")
        return
    
    # Use kits19's official download URL pattern
    url = f"https://huggingface.co/datasets/neheller/KiTS-Challenge-Imaging/resolve/main/images/case_{{case_id:05d}}.nii.gz"
    print(f"  📥 Downloading case_{{case_id:05d}}... ", end='', flush=True)
    
    case_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
        
        total = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(imaging_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)
        
        size_mb = downloaded / 1024 / 1024
        print(f"✅ ({{size_mb:.1f}} MB)")
    except Exception as e:
        print(f"❌ Failed: {{e}}")
        if imaging_path.exists():
            imaging_path.unlink()

print(f"Starting download of {{len(cases_to_download)}} cases...")
for i, case_id in enumerate(cases_to_download):
    print(f"[{{i+1}}/{{len(cases_to_download)}}]", end=' ')
    download_case(int(case_id))

print("\\n✅ Download complete!")
"""

# Write and execute the wrapper
tmp_path = Path("/tmp/kits19_subset_download.py")
tmp_path.write_text(wrapper_script)

print(f"\n🚀 Starting download of {len(to_download)} cases...")
print("   This may take 20-40 minutes depending on internet speed.\n")

result = subprocess.run(
    [sys.executable, str(tmp_path)],
    cwd=str(KITS19_DIR)
)

if result.returncode == 0:
    print("\n✅ All downloads completed!")
    
    # Verify
    verified = [c for c in all_target if has_imaging(c)]
    missing  = [c for c in all_target if not has_imaging(c)]
    print(f"\n📊 Verification:")
    print(f"  Have imaging: {len(verified)}/{len(all_target)}")
    if missing:
        print(f"  ⚠️  Missing: {missing}")
    else:
        print(f"  ✅ All {len(all_target)} cases ready!")
else:
    print(f"\n❌ Download failed with code {result.returncode}")
    sys.exit(1)
