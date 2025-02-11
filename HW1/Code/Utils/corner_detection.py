import numpy as np
import cv2
from typing import Tuple, Optional


def find_corners(self, image: np.ndarray) -> Tuple[Optional[np.ndarray], bool]:
    """
    Enhanced corner detection with preprocessing and multiple attempts
    Args:
        image: Input image
    Returns:
        corners: Corner coordinates if found, None otherwise
        success: Whether corners were found
    """

    def try_find_corners(img):
        return cv2.findChessboardCorners(
            img,
            self.pattern_size,
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH +
                  cv2.CALIB_CB_NORMALIZE_IMAGE +
                  cv2.CALIB_CB_FILTER_QUADS +
                  cv2.CALIB_CB_FAST_CHECK
        )

    # Convert to grayscale if needed
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()

    # Try with original image
    ret, corners = try_find_corners(gray)

    if not ret:
        # Try with different preprocessing methods
        preprocessing_methods = [
            # Adaptive histogram equalization
            lambda img: cv2.equalizeHist(img),

            # Gaussian blur
            lambda img: cv2.GaussianBlur(img, (5, 5), 0),

            # Contrast enhancement
            lambda img: cv2.convertScaleAbs(img, alpha=1.2, beta=0),

            # Adaptive thresholding
            lambda img: cv2.adaptiveThreshold(
                img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            ),

            # Bilateral filtering
            lambda img: cv2.bilateralFilter(img, 9, 75, 75)
        ]

        for preprocess in preprocessing_methods:
            processed = preprocess(gray)
            ret, corners = try_find_corners(processed)
            if ret:
                break

    if ret:
        # Refine corner positions
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
        refined_corners = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1), criteria
        )

        # Verify corner positions
        if self.verify_corner_positions(refined_corners):
            return refined_corners, True

    return None, False


def verify_corner_positions(self, corners: np.ndarray) -> bool:
    """
    Verify that detected corners form a reasonable pattern
    Args:
        corners: Detected corner coordinates
    Returns:
        bool: Whether corners form a valid pattern
    """
    if corners is None:
        return False

    # Reshape corners to grid
    grid = corners.reshape(self.pattern_size[0], self.pattern_size[1], 2)

    # Check row and column spacing consistency
    row_spacing = np.diff(grid, axis=0)
    col_spacing = np.diff(grid, axis=1)

    row_spacing_std = np.std(np.linalg.norm(row_spacing, axis=2))
    col_spacing_std = np.std(np.linalg.norm(col_spacing, axis=2))

    # Define reasonable thresholds
    spacing_threshold = 5.0  # pixels

    # Check if spacing is consistent
    if row_spacing_std > spacing_threshold or col_spacing_std > spacing_threshold:
        return False

    # Check for reasonable aspect ratio
    width = np.linalg.norm(grid[0, -1] - grid[0, 0])
    height = np.linalg.norm(grid[-1, 0] - grid[0, 0])
    aspect_ratio = width / height

    # Expected aspect ratio based on pattern size
    expected_ratio = (self.pattern_size[1] - 1) / (self.pattern_size[0] - 1)
    ratio_tolerance = 0.3

    if abs(aspect_ratio - expected_ratio) > ratio_tolerance:
        return False

    return True