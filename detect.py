from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from utils.unet import UNet

class VesselDetector:
    
    def __init__(self, model_path: str | Path, threshold: float = 0.5):
        self.threshold = threshold
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Inicjalizacja detektora na urządzeniu: {self.device}")
        
        self.model = UNet(n_channels=1, n_classes=1)
        
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        
        self.model.to(self.device)
        self.model.eval()

        self.preprocess = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor()
        ])

    def detect(self, image_path: str | Path) -> np.ndarray:
        """Przetwarza obraz i zwraca binarną maskę naczyń w oryginalnej rozdzielczości."""
        
        # --- KROK 1: Przygotowanie obrazu ---
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



if __name__ == "__main__":

    detector = VesselDetector(model_path="./model/unet_vessels.pth", threshold=0.5)
    
    test_image = "./data/input/im0001.ppm"
    
    result_mask = detector.detect(test_image)
  
    output_img = Image.fromarray(result_mask)
    output_img.save("./data/output/unet/wynik_unet.png")
    print("Gotowe! Maska zapisana jako ./data/output/unet/wynik_unet.png")