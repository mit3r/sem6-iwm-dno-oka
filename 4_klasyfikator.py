import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, balanced_accuracy_score
from skimage.measure import moments_central, moments_normalized, moments_hu

class PatchFeatureExtractor:
    """Extracts features from 5x5 patches around given coordinates in the image."""
    
    def __init__(self, patch_size: int = 5):
        if patch_size % 2 == 0:
            raise ValueError("Patch size must be an odd number.")
        self.patch_size = patch_size
        self.pad = patch_size // 2

    def extract_features(self, image: np.ndarray, coords: np.ndarray) -> np.ndarray:
        """
        For each coordinate (y, x), extracts a 5x5 patch and calculates features.
        Returns a feature matrix (N_samples, N_features).
        """
        # Padding image with "mirror reflection" to safely extract patches at edges
        padded_image = np.pad(image, self.pad, mode='reflect').astype(np.float32)
        
        features_list = []
        
        for y, x in coords:
            patch = padded_image[y : y + self.patch_size, x : x + self.patch_size]
          
            mean_val = np.mean(patch)
            var_val = np.var(patch)
       
            if np.max(patch) == 0:
                hu_moments = np.zeros(7)
            else:
                mu = moments_central(patch)
                nu = moments_normalized(mu)
                hu_moments = moments_hu(nu)
            
            # Concat features into a single vector
            feature_vector = np.hstack(([mean_val, var_val], hu_moments))
            features_list.append(feature_vector)
            
        return np.array(features_list)


class DatasetBuilder:
    """Builds a training dataset with undersampling."""
    
    @staticmethod
    def build_dataset(image: np.ndarray, expert_mask: np.ndarray, roi_mask: np.ndarray = None, patch_size: int = 5):
        """
        Returns the feature matrix X and labels Y for a single image,
        balancing the classes (N vessels = N background).
        """
        # Find vessel coordinates (positive class: 1)
        vessel_coords = np.argwhere(expert_mask > 0)
        
        # Find background coordinates (negative class: 0)
        if roi_mask is not None:
            # Sample background only from the retina area (excluding black margins)
            background_coords = np.argwhere((expert_mask == 0) & (roi_mask > 0))
        else:
            background_coords = np.argwhere(expert_mask == 0)
        
        # Sample from background as many points as we have vessel points
        n_vessels = len(vessel_coords)
        if len(background_coords) > n_vessels:
            indices = np.random.choice(len(background_coords), size=n_vessels, replace=False)
            background_coords = background_coords[indices]
            
        # Łączenie współrzędnych i etykiet
        all_coords = np.vstack((vessel_coords, background_coords))
        y = np.hstack((np.ones(n_vessels), np.zeros(len(background_coords))))
        
        # Ekstrakcja cech (X)
        extractor = PatchFeatureExtractor(patch_size)
        x = extractor.extract_features(image, all_coords)
        
        return x, y


class VesselClassifierML:
    """Random Forest classifier for retinal vessel segmentation."""
    
    def __init__(self, n_estimators: int = 50):
        self.model = RandomForestClassifier(n_estimators=n_estimators, random_state=42, n_jobs=-1)
        self.extractor = PatchFeatureExtractor(patch_size=5)
        
    def train(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)
      
        
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray):
        """Evaluate the model on a hold-out test set and print classification metrics."""
        y_pred = self.model.predict(X_test)
        
        print(classification_report(y_test, y_pred, target_names=["Tło", "Naczynie"]))
        
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        print(f"Macierz pomyłek: TP={tp}, TN={tn}, FP={fp}, FN={fn}")
        
        acc = accuracy_score(y_test, y_pred)
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        b_acc = balanced_accuracy_score(y_test, y_pred) # Średnia arytmetyczna z czułości i swoistości
        
        print(f"Accuracy (Trafność): {acc:.4f}")
        print(f"Sensitivity (Czułość): {sens:.4f}")
        print(f"Specificity (Swoistość): {spec:.4f}")
        print(f"Balanced Accuracy (Dla danych niezrównoważonych): {b_acc:.4f}\n")
        
    def predict_image(self, image: np.ndarray, roi_mask: np.ndarray = None) -> np.ndarray:
        """Predicts a binary vessel mask for the entire image using the trained model."""

        if roi_mask is not None:
            coords = np.argwhere(roi_mask > 0)
        else:
            h, w = image.shape
            y, x = np.mgrid[0:h, 0:w]
            coords = np.column_stack((y.ravel(), x.ravel()))
            
        print(f"Ekstrakcja cech dla {len(coords)} pikseli...")
        X_infer = self.extractor.extract_features(image, coords)
        y_pred = self.model.predict(X_infer)
        
        # Odbudowanie obrazu binarnego
        mask = np.zeros_like(image, dtype=np.uint8)
        for idx, (cy, cx) in enumerate(coords):
            mask[cy, cx] = int(y_pred[idx] * 255)
            
        return mask