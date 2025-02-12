import numpy as np
import cv2
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class CalibrationResult:
    K: np.ndarray
    k: np.ndarray
    R: List[np.ndarray]
    t: List[np.ndarray]
    reprojection_error: float
    per_view_errors: List[float]


class CameraCalibration:
    def __init__(self, square_size: float = 21.5):
        self.square_size = square_size
        self.K = None
        self.k = np.zeros(2)
        self.R = []
        self.t = []

    def find_corners(self, image: np.ndarray) -> Tuple[Optional[np.ndarray], bool]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()

        # Preprocess for better detection
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # Find corners (6x7 inner corners)
        pattern_size = (6, 7)  # width x height of inner corners
        ret, corners = cv2.findChessboardCorners(
            gray, pattern_size,
            flags=(cv2.CALIB_CB_ADAPTIVE_THRESH +
                   cv2.CALIB_CB_NORMALIZE_IMAGE +
                   cv2.CALIB_CB_FILTER_QUADS)
        )

        if ret and corners is not None:
            # Refine corners
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            return corners, True
        return None, False

    def create_object_points(self) -> np.ndarray:
        objp = np.zeros((6 * 7, 3), np.float32)
        objp[:, :2] = np.mgrid[0:6, 0:7].T.reshape(-1, 2)
        objp *= self.square_size
        return objp

    def calibrate(self, images: List[np.ndarray]) -> CalibrationResult:
        object_points = []
        image_points = []
        objp = self.create_object_points()

        for img in images:
            corners, found = self.find_corners(img)
            if found:
                object_points.append(objp)
                image_points.append(corners)

        if len(object_points) < 2:
            raise ValueError("At least 2 valid calibration images required")

        # Initial calibration
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            object_points, image_points, images[0].shape[::-1], None, None,
            flags=cv2.CALIB_RATIONAL_MODEL)

        # Refine with optimization
        self.K = mtx
        self.k = dist[:2]
        self.R = [cv2.Rodrigues(r)[0] for r in rvecs]
        self.t = tvecs

        # Calculate reprojection error
        total_error = 0
        per_view_errors = []
        for i in range(len(object_points)):
            imgpoints2, _ = cv2.projectPoints(
                object_points[i], rvecs[i], tvecs[i], mtx, dist)
            error = cv2.norm(image_points[i], imgpoints2.reshape(-1, 2))
            per_view_errors.append(error)
            total_error += error

        return CalibrationResult(
            K=self.K.copy(),
            k=self.k.copy(),
            R=[R.copy() for R in self.R],
            t=[t.copy() for t in self.t],
            reprojection_error=total_error / len(object_points),
            per_view_errors=per_view_errors
        )

    def undistort_image(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
            self.K, np.hstack((self.k, [0, 0, 0])), (w, h), 1, (w, h))
        dst = cv2.undistort(image, self.K, np.hstack((self.k, [0, 0, 0])), None, newcameramtx)
        x, y, w, h = roi
        return dst[y:y + h, x:x + w]