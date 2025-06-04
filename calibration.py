import os
import numpy as np
from tqdm import tqdm
import argparse
from detector.adversarial_detector import AdversarialDetector

def calibrate(dataset_path, output_file='calibration.npz'):
    print("[1/4] Initializing detector...")
    detector = AdversarialDetector()
    
    print("[2/4] Scanning dataset directory...")
    image_files = [f for f in os.listdir(dataset_path) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        raise FileNotFoundError(f"No images found in {dataset_path}")
    
    print(f"[3/4] Processing {len(image_files)} images...")
    features = []
    for img_file in tqdm(image_files, desc="Calibrating"):
        try:
            img_path = os.path.join(dataset_path, img_file)
            features.append(detector.extract_features(img_path))
        except Exception as e:
            print(f"\nSkipped {img_file}: {str(e)}")
    
    if not features:
        raise RuntimeError("No features extracted - check your images and model")
    
    print("[4/4] Calculating statistics...")
    features = np.array(features)
    mean = np.mean(features, axis=0)
    distances = np.linalg.norm(features - mean, axis=1)
    threshold = np.percentile(distances, 95)
    
    np.savez(output_file, mean=mean, threshold=threshold)
    print(f"\n✅ Calibration complete! Saved to {output_file}")
    print(f"• Mean vector shape: {mean.shape}")
    print(f"• Threshold: {threshold:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, 
                       help='Path to calibration images')
    parser.add_argument('--output', type=str, default='calibration.npz',
                       help='Output filename')
    args = parser.parse_args()
    
    calibrate(args.dataset, args.output)