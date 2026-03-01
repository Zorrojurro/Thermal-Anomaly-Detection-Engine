"""
Image preprocessing pipeline for infrared thermal images.

Implements: resize, bilateral denoising, CLAHE enhancement,
normalization, and ROI extraction.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional


class ThermalImageProcessor:
    """
    Preprocessing pipeline for infrared thermal images.

    Pipeline steps (in order):
        1. Load image (grayscale)
        2. Resize to target dimensions
        3. Bilateral filter denoising
        4. CLAHE contrast enhancement
        5. Min-Max normalization to [0, 1]
    """

    def __init__(
        self,
        image_size: Tuple[int, int] = (224, 224),
        bilateral_d: int = 9,
        bilateral_sigma_color: float = 75.0,
        bilateral_sigma_space: float = 75.0,
        clahe_clip_limit: float = 2.0,
        clahe_tile_grid_size: Tuple[int, int] = (8, 8),
        normalize: bool = True,
    ):
        self.image_size = image_size
        self.bilateral_d = bilateral_d
        self.bilateral_sigma_color = bilateral_sigma_color
        self.bilateral_sigma_space = bilateral_sigma_space
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_tile_grid_size = clahe_tile_grid_size
        self.normalize = normalize

    @classmethod
    def from_config(cls, config) -> "ThermalImageProcessor":
        """Create processor from a Config object."""
        img_size = tuple(config.data.image_size)
        pp = config.preprocessing
        return cls(
            image_size=img_size,
            bilateral_d=pp.bilateral_filter.d,
            bilateral_sigma_color=pp.bilateral_filter.sigma_color,
            bilateral_sigma_space=pp.bilateral_filter.sigma_space,
            clahe_clip_limit=pp.clahe.clip_limit,
            clahe_tile_grid_size=tuple(pp.clahe.tile_grid_size),
            normalize=pp.normalize,
        )

    # ------------------------------------------------------------------
    # Core pipeline
    # ------------------------------------------------------------------

    def load_image(self, image_path: str) -> np.ndarray:
        """Load an image as grayscale. Raises if file not found."""
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")

        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Failed to decode image: {path}")
        return img

    def resize(self, image: np.ndarray) -> np.ndarray:
        """Resize to the configured target size."""
        return cv2.resize(image, self.image_size, interpolation=cv2.INTER_LINEAR)

    def denoise(self, image: np.ndarray) -> np.ndarray:
        """Apply bilateral filter for edge-preserving noise removal."""
        return cv2.bilateralFilter(
            image,
            self.bilateral_d,
            self.bilateral_sigma_color,
            self.bilateral_sigma_space,
        )

    def enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """Apply CLAHE for adaptive contrast enhancement."""
        clahe = cv2.createCLAHE(
            clipLimit=self.clahe_clip_limit,
            tileGridSize=self.clahe_tile_grid_size,
        )
        return clahe.apply(image)

    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        """Min-Max normalization to [0, 1] float32."""
        img = image.astype(np.float32)
        min_val, max_val = img.min(), img.max()
        if max_val - min_val > 0:
            img = (img - min_val) / (max_val - min_val)
        else:
            img = np.zeros_like(img)
        return img

    def process(self, image_path: str) -> np.ndarray:
        """
        Run the full preprocessing pipeline on an image file.

        Args:
            image_path: Path to a thermal image.

        Returns:
            Preprocessed image as a float32 array in [0, 1], shape (H, W).
        """
        img = self.load_image(image_path)
        img = self.resize(img)
        img = self.denoise(img)
        img = self.enhance_contrast(img)
        if self.normalize:
            img = self.normalize_image(img)
        return img

    def process_array(self, image: np.ndarray) -> np.ndarray:
        """
        Run the pipeline on an already-loaded numpy array (grayscale).
        """
        img = self.resize(image)
        img = self.denoise(img)
        img = self.enhance_contrast(img)
        if self.normalize:
            img = self.normalize_image(img)
        return img

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    @staticmethod
    def extract_roi(
        image: np.ndarray,
        bbox: Optional[Tuple[int, int, int, int]] = None,
    ) -> np.ndarray:
        """
        Extract a region of interest.

        If *bbox* is None, auto-detect via thresholding + contours.

        Args:
            image: Grayscale image.
            bbox:  (x, y, w, h) or None for auto-detect.

        Returns:
            Cropped image of the ROI.
        """
        if bbox is not None:
            x, y, w, h = bbox
            return image[y : y + h, x : x + w]

        # Auto-detect: threshold → largest contour
        if image.dtype == np.float32 or image.dtype == np.float64:
            img_uint8 = (image * 255).astype(np.uint8)
        else:
            img_uint8 = image.copy()

        _, thresh = cv2.threshold(img_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return image  # fallback: return full image

        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        return image[y : y + h, x : x + w]

    @staticmethod
    def compute_thermal_stats(image: np.ndarray) -> dict:
        """Compute basic thermal statistics of an image."""
        return {
            "mean": float(np.mean(image)),
            "std": float(np.std(image)),
            "min": float(np.min(image)),
            "max": float(np.max(image)),
            "median": float(np.median(image)),
        }
