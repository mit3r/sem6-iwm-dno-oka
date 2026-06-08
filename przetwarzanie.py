from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from skimage import exposure, filters, measure, morphology
from skimage.filters import threshold_otsu

from utils import Loader
from utils.metrics import MaskEvaluator


ROOT_DIR = Path(__file__).resolve().parent
INPUT_DIR = ROOT_DIR / "data" / "input"
LABEL_DIR = ROOT_DIR / "data" / "label"
OUTPUT_DIR = ROOT_DIR / "data" / "output" / "frabgi"


@dataclass(slots=True)
class ExtractionResult:
    image_path: Path
    mask_path: Path
    overlay_path: Path
    metrics: dict[str, float] | None


class VesselExtractor:
    """Extract retinal vessels with preprocessing, Frangi response and morphological cleanup."""

    def __init__(self) -> None:
        pass

    def process(self, image_path: Path, output_mask_path: Path, save_overlay: bool = True) -> ExtractionResult:
        """Run the full vessel extraction pipeline and save the mask and optionally the overlay.
        
        Args:
            image_path: Path to the input fundus image.
            output_mask_path: Path where the binary vessel mask should be saved.
            save_overlay: If True, also save a visualization overlay. Default is True.
            
        Returns:
            ExtractionResult with paths to output files and evaluation metrics if labels exist.
        """
        rgb_image = self._load_rgb(image_path)
        green_channel = rgb_image[:, :, 1]

        denoised, vessel_response = self.enhance_vessels(green_channel)
        roi = self.extract_roi(green_channel)
        vessel_mask = self._segment_vessels(vessel_response, roi)

        output_mask_path.parent.mkdir(parents=True, exist_ok=True)
        self._save_mask(output_mask_path, vessel_mask)

        overlay_path = output_mask_path.with_name(
            f"{output_mask_path.stem.removesuffix('_mask')}_overlay.png"
        )
        if save_overlay:
            overlay = self._build_overlay(rgb_image, vessel_mask)
            self._save_overlay(overlay_path, overlay)

        metrics = MaskEvaluator.evaluate_against_label(vessel_mask, self._label_path_for(image_path))
        return ExtractionResult(
            image_path=image_path,
            mask_path=output_mask_path,
            overlay_path=overlay_path,
            metrics=metrics,
        )

    def _load_rgb(self, path: Path) -> np.ndarray:
        """Load the source image as RGB and keep it in 8-bit format."""
        return np.array(Image.open(path).convert("RGB"), dtype=np.uint8)

    def _normalize01(self, image: np.ndarray) -> np.ndarray:
        """Normalize an array to the [0, 1] range in a numerically safe way."""
        image = image.astype(np.float32)
        min_value = float(image.min())
        max_value = float(image.max())
        if max_value <= min_value:
            return np.zeros_like(image, dtype=np.float32)
        return (image - min_value) / (max_value - min_value)

    def extract_roi(self, green_channel: np.ndarray) -> np.ndarray:
        """Find the retinal field of view and suppress background outside the fundus."""
        blurred = filters.gaussian(green_channel, sigma=5.0)
        if float(blurred.max()) <= float(blurred.min()):
            return np.ones_like(green_channel, dtype=bool)

        threshold = threshold_otsu(blurred)
        roi = blurred > (threshold * 0.8)
        roi = morphology.remove_small_objects(roi, max_size=5_000)
        roi = morphology.remove_small_holes(roi, max_size=5_000)
        roi = morphology.closing(roi, morphology.disk(8))

        labeled = measure.label(roi)
        if int(np.max(labeled)) == 0:
            return roi

        largest_region = max(measure.regionprops(labeled), key=lambda region: region.area)
        return labeled == largest_region.label

    def enhance_vessels(self, green_channel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Improve contrast, suppress noise and compute the Frangi vessel response.
        
        Uses adaptive histogram equalization followed by Gaussian smoothing, then applies
        the Frangi vessel enhancement filter (ridge detector) at multiple scales.
        """
        normalized = self._normalize01(green_channel)
        equalized = exposure.equalize_adapthist(normalized, clip_limit=0.03)
        denoised = filters.gaussian(equalized, sigma=1.0)
        vessel_response = filters.frangi(
            denoised,
            sigmas=(1, 2, 3, 4, 5, 6),
            alpha=0.5,
            beta=0.5,
            black_ridges=True,
        )
        vessel_response = self._normalize01(vessel_response)
        return denoised, vessel_response

    def _segment_vessels(self, vessel_response: np.ndarray, roi: np.ndarray) -> np.ndarray:
        """Convert the vessel response into a cleaned binary mask."""
        if vessel_response.max() <= vessel_response.min():
            return np.zeros_like(vessel_response, dtype=bool)

        otsu_threshold = threshold_otsu(vessel_response)
        percentile_threshold = float(np.percentile(vessel_response[roi], 72)) if np.any(roi) else otsu_threshold
        threshold = max(otsu_threshold * 0.55, percentile_threshold)

        vessels = vessel_response > threshold
        vessels &= roi

        vessels = morphology.closing(vessels, morphology.disk(2))
        vessels = morphology.opening(vessels, morphology.disk(1))
        vessels = morphology.remove_small_objects(vessels, max_size=120)
        vessels = morphology.remove_small_holes(vessels, max_size=120)
        vessels &= roi
        return vessels

    def _build_overlay(self, rgb_image: np.ndarray, vessel_mask: np.ndarray) -> np.ndarray:
        """Paint detected vessels with a high-contrast color for visual inspection."""
        overlay = rgb_image.astype(np.float32) / 255.0
        overlay[vessel_mask] = [1.0, 0.0, 0.0]
        return (np.clip(overlay, 0.0, 1.0) * 255).astype(np.uint8)

    def _save_mask(self, path: Path, vessel_mask: np.ndarray) -> None:
        """Persist the binary vessel mask as an 8-bit grayscale image."""
        Image.fromarray(vessel_mask.astype(np.uint8) * 255, mode="L").save(path)

    def _save_overlay(self, path: Path, overlay: np.ndarray) -> None:
        """Persist the visualization overlay next to the mask."""
        Image.fromarray(overlay, mode="RGB").save(path)



    def _label_path_for(self, image_path: Path) -> Path:
        """Build the path to the expert label matching the current image."""
        candidates = [l for l in LABEL_DIR.iterdir() if l.is_file() and image_path.stem in l.name]
        return candidates[0] if candidates else LABEL_DIR / f"{image_path.stem}.vk.ppm"




def main(save_overlay: bool = True) -> None:
    """Process the whole input pool and store all masks in the output directory.
    
    Args:
        save_overlay: If True, save visualization overlays alongside masks. Default is True.
    """
    image_paths = Loader.load_images_paths("input")
    if not image_paths:
        raise RuntimeError(f"Brak obrazow w {INPUT_DIR}")

    extractor = VesselExtractor()
    
    def process_wrapper(image_path: Path, output_mask_path: Path) -> ExtractionResult:
        """Wrapper to inject save_overlay parameter into batch processing."""
        return extractor.process(image_path, output_mask_path, save_overlay=save_overlay)
    
    results = Loader.process_images("input", OUTPUT_DIR, process_wrapper)

    all_metrics: list[dict[str, float]] = []
    for result in results:
        if result.metrics is not None:
            all_metrics.append(result.metrics)
            print(
                f"{result.image_path.name}: "
                f"Dice={result.metrics['dice']:.4f}, IoU={result.metrics['iou']:.4f}, "
                f"Sensitivity={result.metrics['sensitivity']:.4f}, Specificity={result.metrics['specificity']:.4f}"
            )
        else:
            print(f"{result.image_path.name}: zapisano maske bez metryk referencyjnych")

        print(f"Maska: {result.mask_path}")
        if save_overlay:
            print(f"Overlay: {result.overlay_path}")

    if all_metrics:
        mean_dice = float(np.mean([item["dice"] for item in all_metrics]))
        mean_iou = float(np.mean([item["iou"] for item in all_metrics]))
        mean_sensitivity = float(np.mean([item["sensitivity"] for item in all_metrics]))
        mean_specificity = float(np.mean([item["specificity"] for item in all_metrics]))
        print(
            "Srednie metryki: "
            f"Dice={mean_dice:.4f}, IoU={mean_iou:.4f}, "
            f"Sensitivity={mean_sensitivity:.4f}, Specificity={mean_specificity:.4f}"
        )


if __name__ == "__main__":
    main()
