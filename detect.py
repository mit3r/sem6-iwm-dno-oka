from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import joblib
from PIL import Image
from torchvision import transforms

from utils.unet import UNet

from przetwarzanie import VesselExtractor
from klasyfikator import PatchFeatureExtractor 


class VesselDetector:
    """Uniwersalny detektor naczyń obsługujący modele Deep Learning (U-Net) oraz Machine Learning (Random Forest)."""
    
    def __init__(self, model_path: str | Path, threshold: float = 0.5):
        self.model_path = Path(model_path)
        self.threshold = threshold
  
        if self.model_path.suffix == '.pth':
            self.model_type = 'unet'
            self._init_unet()
        elif self.model_path.suffix == '.joblib':
            self.model_type = 'random_forest'
            self._init_random_forest()
        else:
            raise ValueError(f"Nieobsługiwany format pliku: {self.model_path.suffix}. Użyj .pth lub .joblib")

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

    def detect(self, image_path: str | Path) -> np.ndarray:
        """Przetwarza obraz i zwraca binarną maskę naczyń w oryginalnej rozdzielczości."""
        if self.model_type == 'unet':
            return self._detect_unet(image_path)
        elif self.model_type == 'random_forest':
            return self._detect_random_forest(image_path)

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
    
    print("\nGotowe! Obie maski zostały zapisane w folderze wyjściowym.")