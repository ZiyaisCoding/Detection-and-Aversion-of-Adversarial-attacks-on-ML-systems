# evaluate.py
import os
import numpy as np
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
from detector.adversarial_detector import AdversarialDetector

def evaluate(dataset_path, calibration_file='calibration.npz', output_dir='results'):
    """Evaluate detector performance on a dataset"""
    
    # 1. Setup
    os.makedirs(output_dir, exist_ok=True)
    detector = AdversarialDetector(calibration_file)
    
    # 2. Process images
    image_files = [f for f in os.listdir(dataset_path) 
                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        raise FileNotFoundError(f"No images found in {dataset_path}")
    
    print(f"Evaluating {len(image_files)} images...")
    
    results = {
        'paths': [],
        'distances': [],
        'predictions': []
    }
    
    for img_file in tqdm(image_files, desc="Processing"):
        try:
            img_path = os.path.join(dataset_path, img_file)
            is_adv, distance = detector.detect(img_path)
            
            results['paths'].append(img_path)
            results['distances'].append(distance)
            results['predictions'].append(int(is_adv))
            
        except Exception as e:
            print(f"\nSkipped {img_file}: {str(e)}")
    
    # 3. Save results
    results_file = os.path.join(output_dir, 'evaluation_results.npz')
    np.savez(
        results_file,
        paths=results['paths'],
        distances=results['distances'],
        predictions=results['predictions'],
        threshold=detector.threshold
    )
    
    # 4. Generate report
    distances = np.array(results['distances'])
    predictions = np.array(results['predictions'])
    
    print("\nEvaluation Report:")
    print(f"- Total images processed: {len(distances)}")
    print(f"- Detection threshold: {detector.threshold:.2f}")
    print(f"- Mean distance: {np.mean(distances):.2f}")
    print(f"- Max distance: {np.max(distances):.2f}")
    print(f"- Min distance: {np.min(distances):.2f}")
    
    # 5. Plot histogram
    plt.figure(figsize=(10, 6))
    plt.hist(distances, bins=50, alpha=0.7, color='blue')
    plt.axvline(x=detector.threshold, color='red', linestyle='--', 
               label=f'Threshold: {detector.threshold:.2f}')
    plt.xlabel("Distance from Mean Vector")
    plt.ylabel("Number of Images")
    plt.title("Adversarial Detection Distance Distribution")
    plt.legend()
    
    plot_file = os.path.join(output_dir, 'distance_distribution.png')
    plt.savefig(plot_file)
    print(f"\nResults saved to {results_file}")
    print(f"Plot saved to {plot_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True,
                      help='Path to evaluation dataset')
    parser.add_argument('--calibration', type=str, default='calibration.npz',
                      help='Calibration file path')
    parser.add_argument('--output', type=str, default='results',
                      help='Output directory')
    args = parser.parse_args()
    
    evaluate(args.dataset, args.calibration, args.output)