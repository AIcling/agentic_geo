"""
GEObaseline
"""
import json
import numpy as np
from collections import defaultdict
from pathlib import Path

def load_results(file_path):
    """Load JSON result file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['results']

def merge_results(part1_path, part2_path):
    """Merge two result files."""
    print(f"Loading {part1_path}...")
    results_part1 = load_results(part1_path)
    print(f"  Found {len(results_part1)} results")

    print(f"Loading {part2_path}...")
    results_part2 = load_results(part2_path)
    print(f"  Found {len(results_part2)} results")

    merged_results = results_part1 + results_part2
    print(f"\nMerged total: {len(merged_results)} results")

    return merged_results

def calculate_method_statistics(results):
    """Calculate statistics for each method."""
    if len(results) == 0:
        print("No results to analyze!")
        return {}

    first_result = results[0]
    method_names = list(first_result['method_scores'].keys())

    print(f"\nFound {len(method_names)} methods: {method_names}")

    method_init_scores = {method: [] for method in method_names}
    method_final_scores = {method: [] for method in method_names}

    for result in results:
        suggest_idx = result['suggest_idx']
        init_scores = result['init_scores']
        method_scores = result['method_scores']

        init_score = init_scores[suggest_idx]

        for method in method_names:
            if method in method_scores:
                final_score = method_scores[method][suggest_idx]
                method_init_scores[method].append(init_score)
                method_final_scores[method].append(final_score)

    statistics = {}
    for method in method_names:
        init_scores = np.array(method_init_scores[method])
        final_scores = np.array(method_final_scores[method])

        if len(init_scores) == 0:
            continue

        improvements = final_scores - init_scores

        mean_init_score = np.mean(init_scores)
        median_init_score = np.median(init_scores)
        std_init_score = np.std(init_scores)
        q25_init = np.percentile(init_scores, 25)
        q75_init = np.percentile(init_scores, 75)
        min_init = np.min(init_scores)
        max_init = np.max(init_scores)

        mean_final_score = np.mean(final_scores)
        median_final_score = np.median(final_scores)
        std_final_score = np.std(final_scores)
        q25_final = np.percentile(final_scores, 25)
        q75_final = np.percentile(final_scores, 75)
        min_final = np.min(final_scores)
        max_final = np.max(final_scores)

        mean_improvement = np.mean(improvements)
        median_improvement = np.median(improvements)
        std_improvement = np.std(improvements)

        success_rate = np.sum(improvements > 0) / len(improvements)

        significant_improvement_rate = np.sum(improvements > 0.05) / len(improvements)

        statistics[method] = {
            'count': len(init_scores),
            'mean_init_score': float(mean_init_score),
            'median_init_score': float(median_init_score),
            'std_init_score': float(std_init_score),
            'q25_init_score': float(q25_init),
            'q75_init_score': float(q75_init),
            'min_init_score': float(min_init),
            'max_init_score': float(max_init),
            'mean_final_score': float(mean_final_score),
            'median_final_score': float(median_final_score),
            'std_final_score': float(std_final_score),
            'q25_final_score': float(q25_final),
            'q75_final_score': float(q75_final),
            'min_final_score': float(min_final),
            'max_final_score': float(max_final),
            'mean_improvement': float(mean_improvement),
            'median_improvement': float(median_improvement),
            'std_improvement': float(std_improvement),
            'success_rate': float(success_rate),
            'significant_improvement_rate': float(significant_improvement_rate),
        }

    return statistics

def print_statistics(statistics):
    """Print statistics."""
    print("\n" + "="*120)
    print("GEO Method Statistics (by init/final scores)")
    print("="*120)

    sorted_methods = sorted(statistics.items(),
                           key=lambda x: x[1]['mean_final_score'],
                           reverse=True)

    print(f"\n{'Method':<25} {'Count':<8} {'Mean Init':<14} {'Mean Final':<14} {'Mean Imp':<12} {'Success':<10} {'Sig Imp':<12}")
    print("-" * 120)

    for method, stats in sorted_methods:
        print(f"{method:<25} {stats['count']:<8} "
              f"{stats['mean_init_score']:>12.6f}  {stats['mean_final_score']:>12.6f}  "
              f"{stats['mean_improvement']:>10.6f}  {stats['success_rate']:>8.2%}  "
              f"{stats['significant_improvement_rate']:>10.2%}")

    print("\n" + "="*120)
    print("Detailed Statistics")
    print("="*120)

    for method, stats in sorted_methods:
        print(f"\n[{method}]")
        print(f"  Sample count: {stats['count']}")
        print(f"\n  Init score stats:")
        print(f"    Mean: {stats['mean_init_score']:.6f}")
        print(f"    Median: {stats['median_init_score']:.6f}")
        print(f"    Std: {stats['std_init_score']:.6f}")
        print(f"    25%: {stats['q25_init_score']:.6f}")
        print(f"    75%: {stats['q75_init_score']:.6f}")
        print(f"    Min: {stats['min_init_score']:.6f}")
        print(f"    Max: {stats['max_init_score']:.6f}")
        print(f"\n  Final score stats:")
        print(f"    Mean: {stats['mean_final_score']:.6f}")
        print(f"    Median: {stats['median_final_score']:.6f}")
        print(f"    Std: {stats['std_final_score']:.6f}")
        print(f"    25%: {stats['q25_final_score']:.6f}")
        print(f"    75%: {stats['q75_final_score']:.6f}")
        print(f"    Min: {stats['min_final_score']:.6f}")
        print(f"    Max: {stats['max_final_score']:.6f}")
        print(f"\n  Improvement stats:")
        print(f"    Mean: {stats['mean_improvement']:.6f}")
        print(f"    Median: {stats['median_improvement']:.6f}")
        print(f"    Std: {stats['std_improvement']:.6f}")
        print(f"    Success rate (final>init): {stats['success_rate']:.2%}")
        print(f"    Sig improvement (>0.05): {stats['significant_improvement_rate']:.2%}")

def save_merged_results(results, output_path):
    """Save merged results."""
    output_data = {
        'total_count': len(results),
        'results': results
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\nMerged results saved to: {output_path}")

def save_statistics(statistics, output_path):
    """Save statistics to JSON."""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(statistics, f, indent=2, ensure_ascii=False)

    print(f"Statistics saved to: {output_path}")

def main():
    script_dir = Path(__file__).parent
    results_dir = script_dir / 'results'

    part1_path = results_dir / 'geo_results_qwen-14b_part1.json'
    part2_path = results_dir / 'geo_results_qwen-14b_part2.json'

    merged_output_path = results_dir / 'geo_results_qwen-14b_merged.json'
    statistics_output_path = results_dir / 'geo_results_qwen-14b_statistics.json'

    merged_results = merge_results(part1_path, part2_path)

    save_merged_results(merged_results, merged_output_path)

    print("\nComputing statistics...")
    statistics = calculate_method_statistics(merged_results)

    print_statistics(statistics)

    save_statistics(statistics, statistics_output_path)

    print("\nDone.")

if __name__ == '__main__':
    main()

