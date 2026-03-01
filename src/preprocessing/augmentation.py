"""
Data augmentation for infrared thermal images.

Uses Albumentations for efficient, GPU-friendly augmentation
with thermal-image-appropriate transforms.
"""

import numpy as np
import albumentations as A
from typing import Optional


class ThermalAugmentor:
    """
    Augmentation pipeline tailored for thermal images.

    Default transforms: rotation, horizontal flip, brightness/contrast
    shift, Gaussian noise, and random crop-resize.
    """

    def __init__(
        self,
        rotation_limit: int = 15,
        horizontal_flip: bool = True,
        vertical_flip: bool = False,
        brightness_limit: float = 0.1,
        contrast_limit: float = 0.1,
        image_size: tuple = (224, 224),
        enabled: bool = True,
    ):
        self.enabled = enabled
        self.image_size = image_size

        if not self.enabled:
            self.transform = A.Compose([A.NoOp()])
            return

        self.transform = A.Compose([
            A.Rotate(limit=rotation_limit, border_mode=0, p=0.5),
            A.HorizontalFlip(p=0.5 if horizontal_flip else 0.0),
            A.VerticalFlip(p=0.5 if vertical_flip else 0.0),
            A.RandomBrightnessContrast(
                brightness_limit=brightness_limit,
                contrast_limit=contrast_limit,
                p=0.5,
            ),
            A.GaussNoise(p=0.3),
            A.RandomResizedCrop(
                size=image_size,
                scale=(0.85, 1.0),
                ratio=(0.9, 1.1),
                p=0.3,
            ),
        ])

    @classmethod
    def from_config(cls, config) -> "ThermalAugmentor":
        """Create augmentor from a Config object."""
        aug = config.augmentation
        return cls(
            rotation_limit=aug.rotation_limit,
            horizontal_flip=aug.horizontal_flip,
            vertical_flip=aug.vertical_flip,
            brightness_limit=aug.brightness_limit,
            contrast_limit=aug.contrast_limit,
            image_size=tuple(config.data.image_size),
            enabled=aug.enabled,
        )

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        Apply augmentation pipeline to a single image.

        Args:
            image: Grayscale float32 image in [0, 1], shape (H, W).

        Returns:
            Augmented image, same shape and range.
        """
        if not self.enabled:
            return image

        # Albumentations expects uint8 or float32 with shape (H, W) or (H, W, C)
        if image.ndim == 2:
            image = image[:, :, np.newaxis]  # (H, W, 1)

        result = self.transform(image=image)
        augmented = result["image"]

        if augmented.ndim == 3 and augmented.shape[2] == 1:
            augmented = augmented[:, :, 0]

        # Clamp to [0, 1]
        augmented = np.clip(augmented, 0.0, 1.0)
        return augmented.astype(np.float32)

    def augment_sequence(self, images: list) -> list:
        """
        Apply the *same* augmentation to every image in a sequence.

        Uses a shared random seed so every frame receives identical
        spatial transforms (important for temporal consistency).

        Args:
            images: List of grayscale float32 images.

        Returns:
            List of augmented images.
        """
        if not self.enabled or len(images) == 0:
            return images

        # Pick a random seed for this sequence
        seed = np.random.randint(0, 2**31)

        augmented_images = []
        for img in images:
            # Reset random state so each frame gets identical transforms
            import random
            random.seed(seed)
            np.random.seed(seed)

            augmented_images.append(self(img))

        # Restore randomness
        np.random.seed(None)

        return augmented_images
