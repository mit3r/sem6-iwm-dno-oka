import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from utils.unet import UNet

class FundusDataset(Dataset):
    def __init__(self, input_paths: list[Path], label_paths: list[Path], image_size: int = 512):
        self.input_paths = input_paths
        self.label_paths = label_paths
        
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.input_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
    
        img_path = self.input_paths[idx]
        mask_path = self.label_paths[idx]

        rgb_image = Image.open(img_path).convert("RGB")
        green_channel = rgb_image.split()[1] 
       
        expert_mask = Image.open(mask_path).convert("L")

        x_tensor = self.transform(green_channel)
        y_tensor = self.transform(expert_mask)

        if not isinstance(x_tensor, torch.Tensor) or not isinstance(y_tensor, torch.Tensor):
            raise TypeError("Expected transforms to return a torch.Tensor")

        y_tensor = (y_tensor > 0.0).to(torch.float32)

        return x_tensor, y_tensor

def main():
  
    TRAIN_INPUT_DIR = Path("./data/train/input/")
    TRAIN_LABEL_DIR = Path("./data/train/label/")
    TEST_INPUT_DIR = Path("./data/test/input/")
    TEST_LABEL_DIR = Path("./data/test/label/")
    MODEL_SAVE_PATH = Path("./model/unet_vessels.pth")
    MODEL_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    BATCH_SIZE = 2      
    EPOCHS = 30        
    LEARNING_RATE = 1e-3
    IMAGE_SIZE = 512   
    
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Rozpoczynam trening na urządzeniu: {device}")

    SUPPORTED_EXTS = {".ppm", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

    train_input_paths = sorted([p for p in TRAIN_INPUT_DIR.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS])
    if not train_input_paths:
        raise RuntimeError("Brak obrazów treningowych w folderze train!")
        
    train_label_paths = []
    for p in train_input_paths:
        lbls = [l for l in TRAIN_LABEL_DIR.iterdir() if l.is_file() and p.stem in l.name]
        if not lbls:
            raise RuntimeError(f"Brak maski dla obrazu {p.name} w {TRAIN_LABEL_DIR}!")
        train_label_paths.append(lbls[0])

    test_input_paths = sorted([p for p in TEST_INPUT_DIR.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS])
    if not test_input_paths:
        raise RuntimeError("Brak obrazów testowych w folderze test!")
    test_label_paths = []
    for p in test_input_paths:
        lbls = [l for l in TEST_LABEL_DIR.iterdir() if l.is_file() and p.stem in l.name]
        if not lbls:
            raise RuntimeError(f"Brak maski dla obrazu {p.name} w {TEST_LABEL_DIR}!")
        test_label_paths.append(lbls[0])

    train_dataset = FundusDataset(train_input_paths, train_label_paths, image_size=IMAGE_SIZE)
    val_dataset = FundusDataset(test_input_paths, test_label_paths, image_size=IMAGE_SIZE)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Obrazy treningowe: {len(train_dataset)} | Obrazy walidacyjne: {len(val_dataset)}")


    model = UNet(n_channels=1, n_classes=1).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()

    best_val_loss = float('inf')

    for epoch in range(EPOCHS):
        start_time = time.time()
        
    
        model.train()
        train_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val, y_val = x_val.to(device), y_val.to(device)
                outputs = model(x_val)
                loss = criterion(outputs, y_val)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)

        elapsed_time = time.time() - start_time
        print(f"Epoka [{epoch+1:02d}/{EPOCHS}] | "
              f"Czas: {elapsed_time:.1f}s | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f}")

    
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            
            # Zapis do pliku tymczasowego i zamiana, aby uniknąć błędu 1224 na Windowsie
            temp_save_path = MODEL_SAVE_PATH.with_suffix('.tmp')
            torch.save(model.state_dict(), temp_save_path)
            try:
                temp_save_path.replace(MODEL_SAVE_PATH)
            except PermissionError:
                time.sleep(0.5)
                temp_save_path.replace(MODEL_SAVE_PATH)
                
            print(f" -> Zapisano nowy najlepszy model! (Strata z {best_val_loss:.4f})")

    print(f"\nWagi modelu zostały zapisane w {MODEL_SAVE_PATH.absolute()}")

if __name__ == "__main__":
    main()