import numpy as np

type ImageRGB = np.ndarray[tuple[int, int, int], np.dtype[np.uint8]]
type ImageGray = np.ndarray[tuple[int, int], np.dtype[np.uint8]]