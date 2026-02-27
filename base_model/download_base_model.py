#!/usr/bin/env python3
"""
Download Qwen2.5-1.5B-Instruct to base_model for the surrogate Critic.

Usage:
    python base_model/download_base_model.py

Requires: pip install huggingface_hub
"""

import os
import sys

# Target directory: same as this script (base_model/)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TARGET_DIR = SCRIPT_DIR
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"


def main():
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("huggingface_hub is required. Install with:")
        print("  pip install huggingface_hub")
        sys.exit(1)

    print(f"Downloading {MODEL_ID} to {TARGET_DIR}")
    print("This may take a few minutes depending on your connection...")

    try:
        path = snapshot_download(
            repo_id=MODEL_ID,
            local_dir=TARGET_DIR,
            local_dir_use_symlinks=False,
        )
        print(f"Done. Model saved to: {path}")
    except Exception as e:
        print(f"Download failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
