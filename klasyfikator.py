import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, balanced_accuracy_score
from scipy.ndimage import uniform_filter
from pathlib import Path

from utils.loader import Loader
from przetwarzanie import VesselExtractor

class PatchFeatureExtractor:
    """Extracts features from 5x5 patches highly efficiently using global filters."""
    
    def __init__(self, patch_size: int = 5):
        if patch_size % 2 == 0:
            raise ValueError("Patch size must be an odd number.")
        self.patch_size = patch_size

    def extract_features(self, image: np.ndarray, coords: np.ndarray) -> np.ndarray:
        """
        Calculates features globally (vectorized) and returns values for specific coords.
        Returns a feature matrix (N_samples, N_features).
        """
     
        mean_img = uniform_filter(image, size=self.patch_size, mode='reflect')
        mean_sq_img = uniform_filter(image**2, size=self.patch_size, mode='reflect')
        var_img = np.clip(mean_sq_img - mean_img**2, 0, None) 
        
        
        y_indices = coords[:, 0]
        x_indices = coords[:, 1]
        
        sampled_raw = image[y_indices, x_indices].reshape(-1, 1)   # Sama jasność piksela
        sampled_means = mean_img[y_indices, x_indices].reshape(-1, 1) # Średnia z okienka
        sampled_vars = var_img[y_indices, x_indices].reshape(-1, 1)   # Wariancja z okienka
        
        return np.hstack((sampled_raw, sampled_means, sampled_vars))


class DatasetBuilder:
    """Builds a training dataset with undersampling and strict size limits."""
    
    @staticmethod
    def build_dataset(image: np.ndarray, expert_mask: np.ndarray, roi_mask: np.ndarray = None, patch_size: int = 5):
        """
        Returns the feature matrix X and labels Y for a single image,
        balancing the classes (N vessels = N background) and capping max samples.
        """
        vessel_coords = np.argwhere(expert_mask > 0)
        
        if roi_mask is not None:
            background_coords = np.argwhere((expert_mask == 0) & (roi_mask > 0))
        else:
            background_coords = np.argwhere(expert_mask == 0)
        
        MAX_SAMPLES = 2000 
        
        if len(vessel_coords) > MAX_SAMPLES:
            indices_v = np.random.choice(len(vessel_coords), size=MAX_SAMPLES, replace=False)
            vessel_coords = vessel_coords[indices_v]
            
        n_vessels = len(vessel_coords)
    
        if len(background_coords) > n_vessels:
            indices_b = np.random.choice(len(background_coords), size=n_vessels, replace=False)
            background_coords = background_coords[indices_b]
            
        all_coords = np.vstack((vessel_coords, background_coords))
        y = np.hstack((np.ones(n_vessels), np.zeros(len(background_coords))))
        
        extractor = PatchFeatureExtractor(patch_size)
        x = extractor.extract_features(image, all_coords)
        
        return x, y


class VesselClassifierML:
    """Random Forest classifier for retinal vessel segmentation."""
    
    def __init__(self, n_estimators: int = 50):
    
        self.model = RandomForestClassifier(n_estimators=n_estimators, min_samples_leaf=2, random_state=42, n_jobs=-1)
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
        b_acc = balanced_accuracy_score(y_test, y_pred)
        
        print(f"Accuracy (Trafność): {acc:.4f}")
        print(f"Sensitivity (Czułość): {sens:.4f}")
        print(f"Specificity (Swoistość): {spec:.4f}")
        print(f"Balanced Accuracy: {b_acc:.4f}\n")
        
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
        
        mask = np.zeros_like(image, dtype=np.uint8)
        for idx, (cy, cx) in enumerate(coords):
            mask[cy, cx] = int(y_pred[idx] * 255)
            
        return mask


    def save(self, filepath: str | Path):
        joblib.dump(self.model, filepath)
        print(f"Model zapisany pomyślnie w: {filepath}")
        
    def load(self, filepath: str | Path):
        self.model = joblib.load(filepath)
        print(f"Model wczytany pomyślnie z: {filepath}")


def main():
    TRAIN_INPUT_DIR = Path("./data/train/input/")
    TRAIN_LABEL_DIR = Path("./data/train/label/")
    TEST_INPUT_DIR = Path("./data/test/input/")
    TEST_LABEL_DIR = Path("./data/test/label/")
    SUPPORTED_EXTS = {".ppm", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    
    train_inputs = sorted([p for p in TRAIN_INPUT_DIR.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS])
    test_inputs = sorted([p for p in TEST_INPUT_DIR.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS])
    
    extractor = VesselExtractor()
    dataset_builder = DatasetBuilder()

    X_train_list = []
    y_train_list = []
    
    print(f"ZBIERANIE DANYCH ({len(train_inputs)} obrazów")
    for img_path in train_inputs:
        print(f"Przetwarzanie {img_path.name}...")
        
        label_candidates = [l for l in TRAIN_LABEL_DIR.iterdir() if l.is_file() and img_path.stem in l.name]
        if not label_candidates:
            print(f"Brak maski eksperckiej dla {img_path.name}, pomijanie...")
            continue
        label_path = label_candidates[0]
        
        rgb_image = Loader.load_rgb_image(str(img_path))
        expert_mask = Loader.load_gray_image(str(label_path))
        green_channel = rgb_image[:, :, 1]
        
        roi_mask = extractor.extract_roi(green_channel)
        _, vessel_response = extractor.enhance_vessels(green_channel)
    
        x_img, y_img = dataset_builder.build_dataset(
            image=vessel_response, 
            expert_mask=expert_mask, 
            roi_mask=roi_mask, 
            patch_size=5
        )
        
        X_train_list.append(x_img)
        y_train_list.append(y_img)

    X_train_final = np.vstack(X_train_list)
    y_train_final = np.concatenate(y_train_list)

    print(f"\nTRENOWANIE MODELU")
    classifier = VesselClassifierML(n_estimators=50)
    print(f"Trenowanie Lasu Losowego na łącznej liczbie {len(y_train_final)} wycinków...")
    classifier.train(X_train_final, y_train_final)
    
    MODEL_PATH = "model/rf_vessels.joblib"
    classifier.save(MODEL_PATH)
  
    print(f"\nTESTOWANIE")
    for img_path in test_inputs:
        print(f"Ewaluacja {img_path.name}...")
        
        label_candidates = [l for l in TEST_LABEL_DIR.iterdir() if l.is_file() and img_path.stem in l.name]
        if not label_candidates:
            print(f"Brak maski eksperckiej dla {img_path.name}, pomijanie...")
            continue
        label_path = label_candidates[0]
        
        rgb_image = Loader.load_rgb_image(str(img_path))
        expert_mask = Loader.load_gray_image(str(label_path))
        green_channel = rgb_image[:, :, 1]
        
        roi_mask = extractor.extract_roi(green_channel)
        _, vessel_response = extractor.enhance_vessels(green_channel)
        
        test_coords = np.argwhere(roi_mask > 0)
        
        X_test = classifier.extractor.extract_features(vessel_response, test_coords)
        y_test = expert_mask[test_coords[:, 0], test_coords[:, 1]] > 0
        
        classifier.evaluate(X_test=X_test, y_test=y_test)

if __name__ == "__main__":
    main()