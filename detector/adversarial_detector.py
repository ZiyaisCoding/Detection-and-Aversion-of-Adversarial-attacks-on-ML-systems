import numpy as np
from io import BytesIO
from PIL import Image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input

class AdversarialDetector:
    def __init__(self, calibration_file='calibration.npz'):
        self.model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
        self._load_calibration(calibration_file)  # Initialize calibration data

    def _load_calibration(self, calibration_file):
        """Safely load calibration data"""
        try:
            data = np.load(calibration_file)
            self.mean_vector = data['mean']
            self.threshold = float(data['threshold'])
        except Exception as e:
            raise RuntimeError(f"Calibration loading failed: {str(e)}")

    @staticmethod
    def _load_image(image_input):
        """Handle different image input types"""
        if isinstance(image_input, str):
            return Image.open(image_input).convert("RGB")
        if hasattr(image_input, 'read'):
            image_input.seek(0)
            return Image.open(BytesIO(image_input.read())).convert("RGB")
        if isinstance(image_input, np.ndarray):
            return Image.fromarray(image_input.astype('uint8')).convert("RGB")
        if isinstance(image_input, Image.Image):
            return image_input.convert("RGB")
        raise ValueError("Unsupported input type")

    def extract_features(self, image_input):
        """Process image and extract features"""
        try:
            img = self._load_image(image_input)
            img = img.resize((224, 224))
            x = np.array(img)
            if x.shape != (224, 224, 3):
                raise ValueError("Invalid image dimensions")
                
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            return self.model.predict(x, verbose=0).flatten()
        except Exception as e:
            raise RuntimeError(f"Feature extraction failed: {e}")

    def detect(self, image_input):
        """Safe detection with validation"""
        if self.mean_vector is None:
            raise RuntimeError("Calibration data not loaded")
        
        features = self.extract_features(image_input)
        distance = np.linalg.norm(features - self.mean_vector)
        return (distance > self.threshold), float(distance)
