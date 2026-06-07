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

    def __getitem__(self, idx):
    
        img_path = self.input_paths[idx]
        mask_path = self.label_paths[idx]

        rgb_image = Image.open(img_path).convert("RGB")
        green_channel = rgb_image.split()[1] 
       
        expert_mask = Image.open(mask_path).convert("L")

        x_tensor = self.transform(green_channel)
        y_tensor = self.transform(expert_mask)  

        y_tensor = (y_tensor > 0).float()

        return x_tensor, y_tensor

def main():
  
    INPUT_DIR = Path("./data/input/")
    LABEL_DIR = Path("./data/label/")
    MODEL_SAVE_PATH = Path("./model/unet_vessels.pth")
    
    BATCH_SIZE = 2      
    EPOCHS = 30        
    LEARNING_RATE = 1e-4
    IMAGE_SIZE = 512   
    
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Rozpoczynam trening na urządzeniu: {device}")

    all_input_paths = sorted(list(INPUT_DIR.glob("*.ppm")))
    if not all_input_paths:
        raise RuntimeError("Brak obrazów wejściowych!")
   
    all_label_paths = [LABEL_DIR / f"{p.stem}.vk.ppm" for p in all_input_paths]

  
    split_idx = int(len(all_input_paths) * 0.8)
    train_dataset = FundusDataset(all_input_paths[:split_idx], all_label_paths[:split_idx], image_size=IMAGE_SIZE)
    val_dataset = FundusDataset(all_input_paths[split_idx:], all_label_paths[split_idx:], image_size=IMAGE_SIZE)

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
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f" -> Zapisano nowy najlepszy model! (Strata z {best_val_loss:.4f})")

    print(f"\nWagi modelu zostały zapisane w {MODEL_SAVE_PATH.absolute()}")

if __name__ == "__main__":
    main()