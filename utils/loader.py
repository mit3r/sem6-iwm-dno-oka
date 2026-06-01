

from collections.abc import Callable
from pathlib import Path
from typing import Literal, TypeVar

import numpy as np
from PIL import Image

from .types import ImageGray, ImageRGB

TResult = TypeVar("TResult")


class Loader:
    paths: dict[Literal["train", "label", "input"], Path] = {
        "train": Path("./data/train/"),
        "label": Path("./data/label/"),
        "input": Path("./data/input/"),
    }

    @staticmethod
    def load_images_paths(src: Literal["train", "label", "input"]) -> list[Path]:
        source_dir = Loader.paths[src]
        if not source_dir.exists():
            return []
        return sorted(path for path in source_dir.iterdir() if path.is_file())

    @staticmethod
    def process_images(
        src: Literal["input"],
        output_dir: str | Path,
        processor: Callable[[Path, Path], TResult],
    ) -> list[TResult]:
        image_paths = Loader.load_images_paths(src)
        destination_dir = Path(output_dir)
        destination_dir.mkdir(parents=True, exist_ok=True)

        results: list[TResult] = []
        for image_path in image_paths:
            output_path = destination_dir / f"{image_path.stem}_mask.png"
            results.append(processor(image_path, output_path))
        return results

    @staticmethod
    def load_rgb_image(path: str) -> ImageRGB:
        return np.array(Image.open(path).convert("RGB"), dtype=np.uint8)

    @staticmethod
    def load_gray_image(path: str) -> ImageGray:
        return np.array(Image.open(path).convert("L"), dtype=np.uint8)