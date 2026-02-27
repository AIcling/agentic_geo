"""Preload GEO-Bench dataset into global_cache.json.

This allows run_geo.py to use dataset sources without real-time Google crawling.
"""
import json
import os

def preload_geobench_to_cache(
    dataset_file='GEO-Bench/geo-bench-hf/test.jsonl',
    cache_file='src/global_cache.json',
    max_entries=None
):
    """Preload GEO-Bench sources into cache.

    Args:
        dataset_file: Path to GEO-Bench jsonl
        cache_file: Target cache file path
        max_entries: Max entries to load (None = all)
    """
    if os.path.exists(cache_file):
        print(f"Loading existing cache from {cache_file}")
        with open(cache_file, 'r', encoding='utf-8') as f:
            cache = json.load(f)
        print(f"Existing cache has {len(cache)} queries")
    else:
        print(f"Creating new cache file: {cache_file}")
        cache = {}

    print(f"\nLoading GEO-Bench dataset from {dataset_file}")
    entries_added = 0
    entries_skipped = 0

    with open(dataset_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if max_entries and entries_added >= max_entries:
                print(f"\nReached max_entries limit ({max_entries}), stopping.")
                break

            try:
                data = json.loads(line.strip())
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse line {line_num}: {e}")
                continue

            query = data.get('query')
            sources_list = data.get('sources', [])

            if not query or not sources_list:
                print(f"Warning: Line {line_num} missing query or sources, skipping")
                continue

            if query in cache:
                if len(cache[query][0].get('sources', [])) == 0:
                    print(f"  Overwriting empty sources for: {query[:60]}...")
                else:
                    entries_skipped += 1
                    if entries_skipped <= 5:
                        print(f"  Query already in cache, skipping: {query[:60]}...")
                    continue

            cache_sources = []
            for src in sources_list:
                cache_sources.append({
                    'url': src.get('url', ''),
                    'raw_source': src.get('raw_text', ''),
                    'source': src.get('cleaned_text', src.get('raw_text', '')),
                    'summary': src.get('cleaned_text', src.get('raw_text', '')),
                    'text': src.get('cleaned_text', src.get('raw_text', ''))
                })

            cache[query] = [{
                'sources': cache_sources,
                'responses': []
            }]

            entries_added += 1
            if entries_added % 100 == 0:
                print(f"  Processed {entries_added} queries...")

    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  Total queries added to cache: {entries_added}")
    print(f"  Queries skipped (already in cache): {entries_skipped}")
    print(f"  Final cache size: {len(cache)} queries")
    print(f"{'='*60}\n")

    print(f"Writing cache to {cache_file}...")
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    with open(cache_file, 'w', encoding='utf-8') as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)

    print(f"[OK] Cache file updated successfully!")
    print(f"  You can now run 'python src/run_geo.py' without needing to crawl Google.")

    return cache


if __name__ == '__main__':
    import sys

    max_entries = None
    if len(sys.argv) > 1:
        try:
            max_entries = int(sys.argv[1])
            print(f"Will load at most {max_entries} entries from dataset")
        except ValueError:
            print(f"Invalid max_entries argument: {sys.argv[1]}, loading all entries")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    os.chdir(project_root)

    preload_geobench_to_cache(
        dataset_file='GEO-Bench/geo-bench-hf/test.jsonl',
        cache_file='src/global_cache.json',
        max_entries=max_entries
    )

