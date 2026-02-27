#!/usr/bin/env python3
"""
Prepare evolved/ directory for Hugging Face upload.

Copies critic (value_head, lora_adapter) and runs archive_to_strategies.
Run from project root: python scripts/prep_evolved_for_hf.py
"""
import os
import shutil
import subprocess
import sys

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SRC_CRITIC = os.path.join(PROJ_ROOT, "outputs", "ea_training", "geobench", "critic")
DST_CRITIC = os.path.join(PROJ_ROOT, "evolved", "critic")


def main():
    os.chdir(PROJ_ROOT)

    # 1. Convert archive to strategies.json
    print("[1/2] Converting archive to strategies.json...")
    ret = subprocess.run([sys.executable, "scripts/archive_to_strategies.py"], cwd=PROJ_ROOT)
    if ret.returncode != 0:
        sys.exit(ret.returncode)

    # 2. Copy critic files
    print("[2/2] Copying critic files...")
    os.makedirs(DST_CRITIC, exist_ok=True)

    value_head_src = os.path.join(SRC_CRITIC, "value_head.bin")
    if os.path.exists(value_head_src):
        shutil.copy2(value_head_src, os.path.join(DST_CRITIC, "value_head.bin"))
        print(f"  Copied value_head.bin")
    else:
        print(f"  [Warning] value_head.bin not found at {value_head_src}")

    lora_src = os.path.join(SRC_CRITIC, "lora_adapter")
    lora_dst = os.path.join(DST_CRITIC, "lora_adapter")
    if os.path.exists(lora_src):
        if os.path.exists(lora_dst):
            shutil.rmtree(lora_dst)
        shutil.copytree(lora_src, lora_dst)
        print(f"  Copied lora_adapter/")
    else:
        print(f"  [Warning] lora_adapter not found at {lora_src}")

    print(f"\n[OK] evolved/ ready for upload:")
    print(f"  evolved/critic/value_head.bin")
    print(f"  evolved/critic/lora_adapter/")
    print(f"  evolved/archive/strategies.json")


if __name__ == "__main__":
    main()
