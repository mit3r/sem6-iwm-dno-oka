
from .types import ImageGray, ImageRGB
import numpy as np

from PIL import Image, ImageFilter

class Filters:
    
    @staticmethod
    def normalize_histogram(image: ImageGray) -> ImageGray:
        # Histogram, ilość pikseli o danej jasności (szarości)
        histogram = [0] * 256
        for pixel in image.flatten():
            histogram[pixel] += 1

        # CDF (Cumulative Distribution Function), skumulowana suma histogramu
        cdf = [0] * 256
        cdf[0] = histogram[0]
        for i in range(1, 256):
            cdf[i] = cdf[i - 1] + histogram[i]

        # Znormalizowanie CDF, przeskalowane do zakresu [0, 255]
        cdf_min = min(cdf)
        cdf_max = max(cdf)
        normalized_cdf = [(cdf[i] - cdf_min) / (cdf_max - cdf_min) * 255 for i in range(256)]

        # Map the pixel values to the normalized CDF
        normalized_image: ImageGray = np.zeros(image.shape, dtype=np.uint8)
        width, height = image.shape
        for i in range(width):
            for j in range(height):
                normalized_image[i, j] = int(normalized_cdf[image[i, j]])

        return normalized_image
    
    @staticmethod
    def normalize_histogram_rgb(image: ImageRGB) -> ImageRGB:
        
        # Normalizacja histogramu wspólnie dla wszystkich kanałów
        histogram = [0] * 256
        for pixel in image.reshape(-1, 3):
            for channel in pixel:
                histogram[channel] += 1
        cdf = [0] * 256
        cdf[0] = histogram[0]
        for i in range(1, 256):
            cdf[i] = cdf[i - 1] + histogram[i]
        cdf_min = min(cdf)
        cdf_max = max(cdf)
        normalized_cdf = [(cdf[i] - cdf_min) / (cdf_max - cdf_min) * 255 for i in range(256)]
        normalized_image: ImageRGB = np.zeros(image.shape, dtype=np.uint8)
        width, height, _ = image.shape
        for i in range(width):
            for j in range(height):
                for channel in range(3):
                    normalized_image[i, j, channel] = int(normalized_cdf[image[i, j, channel]])

        return normalized_image