import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def analyze_clean_images(results_path, calibration_path=None):
    """Comprehensive analysis of clean image evaluation"""
    # Load data
    data = np.load(results_path)
    distances = data['distances']
    threshold = float(data['threshold']) if 'threshold' in data.files else None
    
    # Calculate metrics
    metrics = {
        'count': len(distances),
        'mean': np.mean(distances),
        'std': np.std(distances),
        'min': np.min(distances),
        'max': np.max(distances),
        'percentiles': np.percentile(distances, [80, 90, 95, 99])
    }
    
    if threshold:
        metrics.update({
            'false_positives': np.sum(distances > threshold),
            'fpr': np.mean(distances > threshold)
        })
    
    # Calculate recommended thresholds
    p95 = metrics['percentiles'][2]
    p99 = metrics['percentiles'][3]
    metrics['recommended_thresholds'] = {
        'conservative': p95 * 1.10,
        'strict': p99 * 1.05
    }
    
    # Generate visualization
    plt.figure(figsize=(12, 6))
    n, bins, patches = plt.hist(distances, bins=50, alpha=0.7, color='blue')
    
    # Add threshold lines
    if threshold:
        plt.axvline(threshold, color='red', linestyle='--', 
                   linewidth=2, label=f'Current Threshold ({threshold:.2f})')
    
    for i, (name, val) in enumerate(metrics['recommended_thresholds'].items()):
        plt.axvline(val, color=['green','purple'][i], linestyle=':', 
                   label=f'{name.capitalize()} ({val:.2f})')
    
    plt.xlabel("Feature Distance")
    plt.ylabel("Number of Images")
    title = f"Clean Image Analysis (n={metrics['count']})"
    if threshold:
        title += f" | FPR: {metrics['fpr']:.2%}"
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save outputs
    output_dir = Path('analysis_results')
    output_dir.mkdir(exist_ok=True)
    
    plt.savefig(output_dir/'distance_distribution.png')
    np.savez(output_dir/'metrics.npz', **metrics)
    
    # Print report
    print("\n=== CLEAN IMAGE EVALUATION REPORT ===")
    print(f"Images Processed: {metrics['count']}")
    print(f"Distance Stats:")
    print(f"• Mean: {metrics['mean']:.2f} ± {metrics['std']:.2f}")
    print(f"• Range: {metrics['min']:.2f} to {metrics['max']:.2f}")
    
    if threshold:
        print(f"\nCurrent Threshold: {threshold:.2f}")
        print(f"False Positives: {metrics['false_positives']} (FPR: {metrics['fpr']:.2%})")
    
    print("\nRecommended New Thresholds:")
    for name, val in metrics['recommended_thresholds'].items():
        print(f"• {name.capitalize()}: {val:.2f}")
    
    print(f"\nVisualization saved to {output_dir/'distance_distribution.png'}")
    print(f"Metrics saved to {output_dir/'metrics.npz'}")
    
    return metrics

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('results_path', help='Path to evaluation_results.npz')
    parser.add_argument('--calibration', help='Optional calibration.npz path')
    args = parser.parse_args()
    
    analyze_clean_images(args.results_path, args.calibration)