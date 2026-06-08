from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import joblib
from PIL import Image
from torchvision import transforms

from utils.unet import UNet
from utils.filters import Filters

from skimage.filters import frangi, threshold_otsu, gaussian
from skimage.morphology import remove_small_objects, closing, disk
import warnings

from przetwarzanie import VesselExtractor
from klasyfikator import PatchFeatureExtractor 


class BasicImageProcessor:
    """Klasa implementująca algorytm detekcji naczyń oparty na klasycznym przetwarzaniu obrazów."""
    
    def process(self, rgb_image: np.ndarray, roi_mask: np.ndarray = None) -> np.ndarray:
        # a) Wstępne przetworzenie
        green_channel = rgb_image[:, :, 1]
        
        # Normalizacja histogramu wykorzystująca gotową klasę z utils
        normalized = Filters.normalize_histogram(green_channel)
        
        # Delikatne rozmycie Gaussa do usunięcia szumu
        blurred = gaussian(normalized, sigma=1.0)
        
        # b) Właściwe przetworzenie (detekcja cech)
        # Filtr Frangi'ego (wzmocnienie struktur rurkowatych)
        # Naczynia są zazwyczaj ciemniejsze, więc ustawiamy black_ridges=True
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            vessels = frangi(blurred, sigmas=range(1, 5, 1), black_ridges=True)
            
        # c) Końcowe przetworzenie
        # Automatyczne progowanie metodą Otsu na podstawie roi_mask
        if roi_mask is not None and np.any(roi_mask):
            roi_vessels = vessels[roi_mask > 0]
            thresh = threshold_otsu(roi_vessels) if len(roi_vessels) > 0 else 0
        else:
            thresh = threshold_otsu(vessels)
            
        binary_mask = vessels > thresh
        
        # Morfologia: usunięcie drobnego szumu i "połatanie" przerywanych naczyń
        binary_mask = remove_small_objects(binary_mask, min_size=50)
        binary_mask = closing(binary_mask, disk(2))
        
        if roi_mask is not None:
            binary_mask = binary_mask & (roi_mask > 0)
            
        return (binary_mask * 255).astype(np.uint8)


class VesselDetector:
    """Uniwersalny detektor naczyń obsługujący modele Deep Learning (U-Net) oraz Machine Learning (Random Forest)."""
    
    def __init__(self, model_path: str | Path, threshold: float = 0.5):
        self.model_path = Path(model_path)
        self.threshold = threshold
  
        if self.model_path.name == 'basic_ip':
            self.model_type = 'basic_ip'
            self._init_basic_ip()
        elif self.model_path.suffix == '.pth':
            self.model_type = 'unet'
            self._init_unet()
        elif self.model_path.suffix == '.joblib':
            self.model_type = 'random_forest'
            self._init_random_forest()
        else:
            raise ValueError(f"Nieobsługiwany format pliku: {self.model_path.suffix}. Użyj .pth, .joblib lub nazwy 'basic_ip'")

    def _init_unet(self):
        """Inicjalizacja środowiska dla sieci U-Net."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Inicjalizacja modelu U-Net na urządzeniu: {self.device}")
        
        self.model = UNet(n_channels=1, n_classes=1)
        state_dict = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        
        self.model.to(self.device)
        self.model.eval()

        self.preprocess = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor()
        ])

    def _init_random_forest(self):
        """Inicjalizacja środowiska dla Lasu Losowego."""
        self.model = joblib.load(self.model_path)
        

        self.feature_extractor = PatchFeatureExtractor(patch_size=5)
        self.roi_extractor = VesselExtractor()

    def _init_basic_ip(self):
        """Inicjalizacja środowiska dla klasycznego przetwarzania obrazu."""
        self.processor = BasicImageProcessor()
        self.roi_extractor = VesselExtractor()

    def detect(self, image_path: str | Path) -> np.ndarray:
        """Przetwarza obraz i zwraca binarną maskę naczyń w oryginalnej rozdzielczości."""
        if self.model_type == 'unet':
            return self._detect_unet(image_path)
        elif self.model_type == 'random_forest':
            return self._detect_random_forest(image_path)
        elif self.model_type == 'basic_ip':
            return self._detect_basic_ip(image_path)

    def _detect_unet(self, image_path: str | Path) -> np.ndarray:
        rgb_image = Image.open(image_path).convert("RGB")
        original_size = rgb_image.size 
        green_channel = rgb_image.split()[1]
        
        input_tensor = self.preprocess(green_channel).unsqueeze(0).to(self.device)
        
        with torch.no_grad(): 
            logits = self.model(input_tensor)
            probabilities = torch.sigmoid(logits)
            
        target_size = (original_size[1], original_size[0]) 
        probs_resized = F.interpolate(probabilities, size=target_size, mode='bilinear', align_corners=False)
        
        binary_mask = (probs_resized > self.threshold).squeeze().cpu().numpy()
        return (binary_mask * 255).astype(np.uint8)

    def _detect_random_forest(self, image_path: str | Path) -> np.ndarray:
 
        rgb_image = np.array(Image.open(image_path).convert("RGB"))
        green_channel = rgb_image[:, :, 1]
        
        roi_mask = self.roi_extractor.extract_roi(green_channel)
        _, vessel_response = self.roi_extractor.enhance_vessels(green_channel)
        
        coords = np.argwhere(roi_mask > 0)
  
        print(f"Ekstrakcja cech dla {len(coords)} pikseli...")
        X_infer = self.feature_extractor.extract_features(vessel_response, coords)
        
        y_pred = self.model.predict(X_infer)
       
        mask = np.zeros_like(green_channel, dtype=np.uint8)
        for idx, (cy, cx) in enumerate(coords):
            mask[cy, cx] = int(y_pred[idx] * 255)
            
        return mask

    def _detect_basic_ip(self, image_path: str | Path) -> np.ndarray:
        rgb_image = np.array(Image.open(image_path).convert("RGB"))
        green_channel = rgb_image[:, :, 1]
        
        roi_mask = self.roi_extractor.extract_roi(green_channel)
        
        return self.processor.process(rgb_image, roi_mask)


if __name__ == "__main__":
    
    test_image = "./data/input/im0001.ppm"
   
    print("\n--- Uruchamianie U-Net ---")
    detector_unet = VesselDetector(model_path="./model/unet_vessels.pth", threshold=0.5)
    mask_unet = detector_unet.detect(test_image)
    Image.fromarray(mask_unet).save("./data/output/wynik_unet.png")
    
    print("\n--- Uruchamianie Random Forest ---")
    detector_rf = VesselDetector(model_path="./model/rf_vessels.joblib")
    mask_rf = detector_rf.detect(test_image)
    Image.fromarray(mask_rf).save("./data/output/wynik_rf.png")
    
    print("\n--- Uruchamianie Basic Image Processing ---")
    detector_ip = VesselDetector(model_path="basic_ip")
    mask_ip = detector_ip.detect(test_image)
    Image.fromarray(mask_ip).save("./data/output/wynik_ip.png")
    
    print("\nGotowe! Wszystkie maski zostały zapisane w folderze wyjściowym.")