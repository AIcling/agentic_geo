#!/usr/bin/env python3
"""
Merge split result files.

Usage:
    python src/merge_results.py

Auto-finds and merges all part*of*.json files.
"""

import json
import os
import sys
import glob
from datetime import datetime


def merge_results():
    """Merge all split result files."""

    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    pattern = os.path.join(results_dir, 'geo_results_*_part*of*.json')
    part_files = sorted(glob.glob(pattern))

    if not part_files:
        print(f"Error: No split files found in {results_dir}")
        print(f"Pattern: geo_results_*_part*of*.json")
        return False

    print("=" * 80)
    print("Merging split result files")
    print("=" * 80)
    print(f"\nFound {len(part_files)} split files:")
    for f in part_files:
        print(f"  - {os.path.basename(f)}")

    all_results = []
    metadata_list = []
    total_errors = 0

    for part_file in part_files:
        print(f"\nLoading: {os.path.basename(part_file)}")
        try:
            with open(part_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            results = data.get('results', [])
            metadata = data.get('metadata', {})

            print(f"  Records: {len(results)}")
            print(f"  Part ID: {metadata.get('part_id', 'N/A')}")

            errors = sum(1 for r in results if r.get('strategy_name') == 'error')
            if errors > 0:
                print(f"  Error records: {errors}")
                total_errors += errors

            all_results.extend(results)
            metadata_list.append(metadata)

        except json.JSONDecodeError as e:
            print(f"  Error: Failed to parse JSON - {e}")
            return False
        except Exception as e:
            print(f"  Error: {e}")
            return False

    print(f"\nMerging data...")
    print(f"  Total records: {len(all_results)}")
    print(f"  Total errors: {total_errors}")

    unique_queries = len(set(r.get('query', '') for r in all_results if r.get('strategy_name') != 'error'))
    strategy_counts = {}
    for r in all_results:
        strategy = r.get('strategy_name', 'unknown')
        strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1

    print(f"  Unique queries: {unique_queries}")
    print(f"\nStrategy distribution:")
    for strategy, count in sorted(strategy_counts.items()):
        print(f"    {strategy}: {count}")

    merged_metadata = {
        'model': metadata_list[0].get('model', 'unknown') if metadata_list else 'unknown',
        'merged_from_parts': len(part_files),
        'merge_timestamp': datetime.now().isoformat(),
        'total_records': len(all_results),
        'total_queries': unique_queries,
        'total_errors': total_errors,
        'source_files': [os.path.basename(f) for f in part_files],
        'part_metadata': metadata_list
    }

    merged_data = {
        'results': all_results,
        'metadata': merged_metadata
    }

    base_name = os.path.basename(part_files[0])
    import re
    merged_name = re.sub(r'_part\d+of\d+', '_merged', base_name)
    output_file = os.path.join(results_dir, merged_name)

    print(f"\nSaving merged file...")
    print(f"  Output: {output_file}")

    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(merged_data, f, indent=2, ensure_ascii=False)

        with open(output_file, 'r', encoding='utf-8') as f:
            json.load(f)

        file_size = os.path.getsize(output_file) / (1024 * 1024)  # MB
        print(f"  File size: {file_size:.2f} MB")
        print(f"  Merge successful!")

        print("\n" + "=" * 80)
        print("Merge Summary")
        print("=" * 80)
        print(f"Input files: {len(part_files)}")
        print(f"Total records: {len(all_results)}")
        print(f"Unique queries: {unique_queries}")
        print(f"Error records: {total_errors}")
        print(f"Output file: {output_file}")
        print("=" * 80)

        return True

    except Exception as e:
        print(f"\nError: Failed to save file - {e}")
        return False


def main():
    print()
    success = merge_results()
    print()

    if success:
        print("Merge complete!")
        return 0
    else:
        print("Merge failed")
        return 1


if __name__ == '__main__':
    sys.exit(main())
