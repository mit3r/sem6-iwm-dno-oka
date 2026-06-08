# Detekcja Naczyń Krwionośnych na Dnie Oka

Projekt ten służy do automatycznej detekcji (segmentacji) naczyń krwionośnych na obrazach medycznych dna oka (fundus images). System implementuje trzy różne podejścia do rozwiązania tego problemu:

1. **U-Net** (Głębokie uczenie / Deep Learning z wykorzystaniem PyTorch)
2. **Random Forest** (Uczenie maszynowe bazujące na ekstrakcji lokalnych cech)
3. **Basic Image Processing** (Klasyczne metody przetwarzania obrazów m.in. filtr Frangi'ego)

## Opis głównych plików w projekcie

- **`train_unet.py`** – Skrypt odpowiedzialny za trenowanie konwolucyjnej sieci neuronowej U-Net. Wczytuje zbiory danych treningowych i walidacyjnych (testowych), przetwarza obrazy, wykorzystując wyizolowany zielony kanał RGB (gdzie naczynia są najlepiej widoczne), a następnie rozpoczyna proces uczenia (domyślnie 30 epok). Po znalezieniu optymalnych wag z najmniejszą stratą walidacyjną (Val Loss), zapisuje model.
- **`detect.py`** – Główny skrypt inferencyjny (testujący). Wykorzystuje uniwersalną klasę `VesselDetector`, która pozwala na detekcję naczyń wybraną z trzech dostępnych metod. Skrypt pobiera pierwszy obraz ze zbioru testowego, przepuszcza go przez wszystkie 3 algorytmy i zapisuje wynikowe maski binarne.
- **`przetwarzanie.py`** oraz **`klasyfikator.py`** _(pliki pomocnicze)_ – Zawierają narzędzia niezbędne dla algorytmu Lasu Losowego (Random Forest). `VesselExtractor` zajmuje się wycinaniem odpowiedniego obszaru poszukiwań (ROI), a `PatchFeatureExtractor` generuje odpowiednie parametry numeryczne dla pojedynczych pikseli.

## Struktura katalogów

```text
DnoOka/
├── data/
│   ├── train/
│   │   ├── input/       # Obrazy treningowe (np. .jpg, .png, .tif)
│   │   └── label/       # Odpowiadające im maski eksperckie (Ground Truth)
│   ├── test/
│   │   ├── input/       # Obrazy testowe / walidacyjne
│   │   └── label/       # Maski do obrazów testowych
│   └── output/          # Tutaj skrypt detect.py zapisze przewidziane maski
├── model/               # Katalog na wagi modelu (np. unet_vessels.pth, rf_vessels.joblib)
├── utils/               # Moduły pomocnicze (filters.py, unet.py)
└── detect.py            # Główny skrypt testujący
```

## Jak uruchomić?

### 1. Trenowanie modelu (U-Net)

```bash
python train_unet.py
```

### 2. Detekcja naczyń (Testowanie modeli)

```bash
python detect.py
```

## Główne wymagania i biblioteki

- `torch`, `torchvision` (PyTorch - do sieci U-Net)
- `numpy` (do operacji macierzowych)
- `Pillow` (do wczytywania i zapisywania obrazów)
- `scikit-image` (do filtrów i morfologii w Basic Image Processing)
- `scikit-learn`, `joblib` (do wsparcia i wczytania modelu Random Forest)
