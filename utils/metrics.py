"""Mask evaluation and comparison utilities for vessel detection."""

from pathlib import Path

import numpy as np
from PIL import Image


class MaskEvaluator:
    """Evaluate and compare binary vessel masks against expert annotations."""

    @staticmethod
    def load_binary_label(path: Path) -> np.ndarray:
        """Load the expert mask and convert it to a boolean array.
        
        Args:
            path: Path to the expert label image (typically .vk.ppm file).
            
        Returns:
            Boolean array where True indicates vessel pixels.
            
        Raises:
            FileNotFoundError: If the label file does not exist.
        """
        if not path.exists():
            raise FileNotFoundError(path)
        label = np.array(Image.open(path).convert("L"), dtype=np.uint8)
        return label > 0

    @staticmethod
    def dice_score(prediction: np.ndarray, target: np.ndarray) -> float:
        """Compute the Dice coefficient for two binary masks.
        
        Dice = 2 * |intersection| / (|prediction| + |target|)
        
        Args:
            prediction: Binary mask of predicted vessels.
            target: Binary mask of expert-annotated vessels.
            
        Returns:
            Dice coefficient in range [0, 1], where 1 is perfect match.
        """
        prediction_sum = float(prediction.sum())
        target_sum = float(target.sum())
        denominator = prediction_sum + target_sum
        if denominator == 0:
            return 1.0
        intersection = float(np.logical_and(prediction, target).sum())
        return 2.0 * intersection / denominator

    @staticmethod
    def iou_score(prediction: np.ndarray, target: np.ndarray) -> float:
        """Compute intersection over union for two binary masks.
        
        IoU = |intersection| / |union|
        
        Args:
            prediction: Binary mask of predicted vessels.
            target: Binary mask of expert-annotated vessels.
            
        Returns:
            IoU coefficient in range [0, 1], where 1 is perfect match.
        """
        union = float(np.logical_or(prediction, target).sum())
        if union == 0:
            return 1.0
        intersection = float(np.logical_and(prediction, target).sum())
        return intersection / union

    @staticmethod
    def evaluate_against_label(
        prediction: np.ndarray, label_path: Path
    ) -> dict[str, float] | None:
        """Compare the mask with the expert label and compute metrics.
        
        Computes Dice, IoU, sensitivity (true positive rate), and specificity 
        (true negative rate) after aligning spatial dimensions.
        
        Args:
            prediction: Binary mask of predicted vessels.
            label_path: Path to the expert annotation file.
            
        Returns:
            Dictionary with keys 'dice', 'iou', 'sensitivity', 'specificity',
            or None if the label file does not exist.
        """
        if not label_path.exists():
            return None

        target = MaskEvaluator.load_binary_label(label_path)
        height = min(prediction.shape[0], target.shape[0])
        width = min(prediction.shape[1], target.shape[1])
        prediction = prediction[:height, :width]
        target = target[:height, :width]

        tp = float(np.logical_and(prediction, target).sum())
        tn = float(np.logical_and(~prediction, ~target).sum())
        fp = float(np.logical_and(prediction, ~target).sum())
        fn = float(np.logical_and(~prediction, target).sum())

        sensitivity = tp / (tp + fn) if (tp + fn) else 1.0
        specificity = tn / (tn + fp) if (tn + fp) else 1.0

        return {
            "dice": MaskEvaluator.dice_score(prediction, target),
            "iou": MaskEvaluator.iou_score(prediction, target),
            "sensitivity": sensitivity,
            "specificity": specificity,
        }
