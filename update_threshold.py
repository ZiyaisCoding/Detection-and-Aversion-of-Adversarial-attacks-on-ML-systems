import numpy as np
from pathlib import Path

def update_calibration(calibration_path, new_threshold, output_path=None):
    """Update calibration file with new threshold"""
    data = np.load(calibration_path)
    mean = data['mean']
    
    output_path = output_path or calibration_path
    np.savez(output_path, mean=mean, threshold=new_threshold)
    print(f"Updated threshold to {new_threshold:.2f} in {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('calibration_path', help='Path to calibration.npz')
    parser.add_argument('new_threshold', type=float, help='New threshold value')
    parser.add_argument('--output', help='Optional output path')
    args = parser.parse_args()
    
    update_calibration(args.calibration_path, args.new_threshold, args.output)