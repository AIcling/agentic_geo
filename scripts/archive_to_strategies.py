#!/usr/bin/env python3
"""
Convert archive_final.json to strategies.json for run_geo.

Archive format (EA output): entries with genotype dict, total_score, etc.
Strategies format (run_geo): {"strategies": [{genotype_id, strategy_type, short_prompt, full_prompt, scores}]}

Run from project root:
    python scripts/archive_to_strategies.py
"""
import json
import os
import sys

# Add project root for evolve imports
PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ_ROOT)

from evolve.genotype.schema import Genotype
from evolve.genotype.renderer import GenotypeRenderer


def main():
    archive_path = os.path.join(
        PROJ_ROOT, "outputs", "ea_training", "geobench",
        "checkpoints", "final", "archive_final.json"
    )
    out_path = os.path.join(PROJ_ROOT, "evolved", "archive", "strategies.json")

    if not os.path.exists(archive_path):
        print(f"[Error] Archive not found: {archive_path}")
        print("  Run EA training first, or set ARCHIVE_PATH env var.")
        archive_path = os.environ.get("ARCHIVE_PATH", archive_path)
        if not os.path.exists(archive_path):
            sys.exit(1)

    out_dir = os.path.dirname(out_path)
    os.makedirs(out_dir, exist_ok=True)

    with open(archive_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    entries = data.get("entries", [])
    if not entries:
        print("[Error] No entries in archive")
        sys.exit(1)

    renderer = GenotypeRenderer(include_query=False, include_content=False)
    strategies = []

    for entry in entries:
        geno_data = entry.get("genotype", {})
        if not geno_data:
            continue
        try:
            genotype = Genotype.from_dict(geno_data)
        except Exception as e:
            print(f"[Warning] Skip invalid genotype: {e}")
            continue

        short_prompt = renderer.render_strategy_only(genotype)
        full_prompt = renderer.render(genotype, query=None, content=None)

        total_score = entry.get("total_score", entry.get("fitness", 0.0))

        strategies.append({
            "genotype_id": genotype.genotype_id,
            "strategy_type": genotype.strategy_type,
            "short_prompt": short_prompt,
            "full_prompt": full_prompt,
            "scores": {"total_score": total_score},
            "fitness": getattr(genotype, "fitness", entry.get("fitness", 0.0)),
            "genotype": geno_data,
        })

    result = {"strategies": strategies}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"[OK] Exported {len(strategies)} strategies to {out_path}")


if __name__ == "__main__":
    main()
