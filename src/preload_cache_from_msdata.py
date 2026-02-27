"""Preload MSdata dataset into global_cache.json.

This allows run_geo0.py to use dataset sources without real-time Google crawling.
"""
import json
import os

def preload_msdata_to_cache(
    dataset_file='MSdata/train.json',
    cache_file='src/global_cache.json',
    max_entries=None
):
    """Preload MSdata sources into cache.

    Args:
        dataset_file: Path to MSdata JSON file (JSON array format)
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

    print(f"\nLoading MSdata dataset from {dataset_file}")
    entries_added = 0
    entries_skipped = 0

    try:
        with open(dataset_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"ERROR: Failed to parse JSON file: {e}")
        return cache
    except FileNotFoundError:
        print(f"ERROR: Dataset file not found: {dataset_file}")
        return cache

    if not isinstance(data, list):
        print(f"ERROR: Expected JSON array, got {type(data)}")
        return cache

    print(f"Loaded {len(data)} entries from dataset")

    for item_num, item in enumerate(data, 1):
        if max_entries and entries_added >= max_entries:
            print(f"\nReached max_entries limit ({max_entries}), stopping.")
            break

        try:
            query = item.get('query')
            text_list = item.get('text_list', [])

            if not query:
                print(f"Warning: Item {item_num} missing query, skipping")
                continue

            if not text_list or len(text_list) == 0:
                print(f"Warning: Item {item_num} missing or empty text_list, skipping")
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
            for idx, text in enumerate(text_list):
                text_str = str(text) if text else ''

                cache_sources.append({
                    'url': f'msdata://source_{idx}',
                    'raw_source': text_str,
                    'source': text_str,
                    'summary': text_str,
                    'text': text_str
                })

            cache[query] = [{
                'sources': cache_sources,
                'responses': []
            }]

            entries_added += 1
            if entries_added % 100 == 0:
                print(f"  Processed {entries_added} queries...")

        except Exception as e:
            print(f"Warning: Failed to process item {item_num}: {e}")
            continue

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
    print(f"  You can now run 'python src/run_geo0.py' without needing to crawl Google.")

    return cache


if __name__ == '__main__':
    import sys

    max_entries = None
    dataset_file = 'MSdata/train.json'

    if len(sys.argv) > 1:
        dataset_file = sys.argv[1]
        if not dataset_file.startswith('MSdata/'):
            dataset_file = f'MSdata/{dataset_file}'

    if len(sys.argv) > 2:
        try:
            max_entries = int(sys.argv[2])
            print(f"Will load at most {max_entries} entries from dataset")
        except ValueError:
            print(f"Invalid max_entries argument: {sys.argv[2]}, loading all entries")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    os.chdir(project_root)

    print(f"Loading from: {dataset_file}")
    preload_msdata_to_cache(
        dataset_file=dataset_file,
        cache_file='src/global_cache.json',
        max_entries=max_entries
    )
