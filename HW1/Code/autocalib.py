import numpy as np
import cv2
from typing import List, Tuple, Optional
from scipy.optimize import least_squares
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
        # self.k = np.zeros(2)
        # self.k = np.zeros((5, 1), dtype=np.float32)
        self.k = np.array([0.0, 0.0], dtype=np.float32).reshape(-1, 1)  # Explicitly initialize k = [0, 0]T
        self.R = []
        self.t = []

    def find_corners(self, image: np.ndarray, pattern_size: Tuple[int, int] = (9, 6)) -> Tuple[Optional[np.ndarray], bool]:
        """ Detect checkerboard corners dynamically with a given pattern size. """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()

        # Preprocess for better detection
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # Find corners dynamically
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
    # def find_corners(self, image: np.ndarray) -> Tuple[Optional[np.ndarray], bool]:
    #     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
    #
    #     # Preprocess for better detection
    #     gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    #     gray = cv2.GaussianBlur(gray, (5, 5), 0)
    #
    #     # Find corners (6x7 inner corners)
    #     pattern_size = (9, 6)  # width x height of inner corners
    #     ret, corners = cv2.findChessboardCorners(
    #         gray, pattern_size,
    #         flags=(cv2.CALIB_CB_ADAPTIVE_THRESH +
    #                cv2.CALIB_CB_NORMALIZE_IMAGE +
    #                cv2.CALIB_CB_FILTER_QUADS)
    #     )
    #
    #     if ret and corners is not None:
    #         # Refine corners
    #         criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    #         corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    #         return corners, True
    #     return None, False

    def create_object_points(self, pattern_size: Tuple[int, int]) -> np.ndarray:
        """ Generate 3D object points assuming Z = 0 (checkerboard pattern) dynamically. """
        objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
        objp *= self.square_size
        return objp
    # def create_object_points(self) -> np.ndarray:
    #     # objp = np.zeros((6 * 7, 3), np.float32)
    #     # objp[:, :2] = np.mgrid[0:6, 0:7].T.reshape(-1, 2)
    #     objp = np.zeros((9 * 6, 3), np.float32)  # Adjusted for correct pattern size
    #     objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)
    #     objp *= self.square_size
    #     return objp
    def calibrate(self, images: List[np.ndarray], pattern_size: Tuple[int, int] = (9, 6)):
        """ Perform initial calibration using Zhang's method. """
        object_points = []
        image_points = []
        objp = self.create_object_points(pattern_size)

        for img in images:
            corners, found = self.find_corners(img, pattern_size)
            if found:
                object_points.append(objp)
                image_points.append(corners)

        if len(object_points) < 2:
            raise ValueError("At least 2 valid calibration images required")

        ret, K, k, rvecs, tvecs = cv2.calibrateCamera(
            object_points, image_points, (images[0].shape[1], images[0].shape[0]), None, None,
            flags=cv2.CALIB_RATIONAL_MODEL)

        self.K = K
        self.k = k.flatten().reshape(-1, 1)
        self.R = [cv2.Rodrigues(r)[0] for r in rvecs]
        self.t = tvecs

        return self.refine_parameters(object_points, image_points)

    def reprojection_error(self, params, object_points, image_points, image_size):
        """ Compute the reprojection error. """

        fx, fy, cx, cy, k1, k2 = params[:6]
        rvecs = np.array(params[6:9], dtype=np.float64).reshape(3, 1)  # Ensure (3,1) shape
        tvecs = np.array(params[9:12], dtype=np.float64).reshape(3, 1)  # Ensure (3,1) shape

        # Ensure K is (3,3) and float64
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

        # Ensure k is (5,1) and float64, filling missing values with 0
        k = np.zeros((5, 1), dtype=np.float64)
        k[:2] = np.array([k1, k2]).reshape(-1, 1)  # Fill only the first two values

        # Ensure object_points is (N,1,3) and float64
        object_points = np.array(object_points, dtype=np.float64).reshape(-1, 1, 3)

        # Project 3D points to 2D
        projected_points, _ = cv2.projectPoints(object_points, rvecs, tvecs, K, k)
        projected_points = projected_points.reshape(-1, 2)

        # Compute error
        error = image_points.reshape(-1, 2) - projected_points
        return error.flatten()

    def refine_parameters(self, object_points, image_points):
        """ Refine camera parameters using scipy.optimize. """
        initial_params = np.hstack([
            self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2], self.k[0, 0], self.k[1, 0],
            cv2.Rodrigues(self.R[0])[0].flatten(), self.t[0].flatten()
        ])

        optimized = least_squares(
            self.reprojection_error, initial_params, method="lm",
            args=(object_points[0], image_points[0], (self.K.shape[1], self.K.shape[0]))
        )

        optimized_params = optimized.x
        self.K = np.array([[optimized_params[0], 0, optimized_params[2]],
                           [0, optimized_params[1], optimized_params[3]],
                           [0, 0, 1]])

        self.k = np.array([optimized_params[4], optimized_params[5]]).reshape(-1, 1)

        R, _ = cv2.Rodrigues(np.array(optimized_params[6:9]))
        self.R[0] = R
        # self.t[0] = np.array(optimized_params[9:12]).reshape(3, 1)
        self.t = (self.t[0], np.array(optimized_params[9:12]).reshape(3, 1))

    def save_rectified_image(self, image: np.ndarray, object_points, output_path="rectified_reprojected.png"):
        """ Save rectified image with reprojected corners. """
        projected_points, _ = cv2.projectPoints(object_points, cv2.Rodrigues(self.R[0])[0], self.t[0], self.K, self.k)
        projected_points = projected_points.reshape(-1, 2)

        for point in projected_points:
            cv2.circle(image, tuple(point.astype(int)), 5, (0, 0, 255), -1)

        cv2.imwrite(output_path, image)
    # def calibrate(self, images: List[np.ndarray]) -> CalibrationResult:
    #     object_points = []
    #     image_points = []
    #     objp = self.create_object_points()
    #
    #     for img in images:
    #         corners, found = self.find_corners(img)
    #         if found:
    #             object_points.append(objp)
    #             image_points.append(corners)
    #
    #     if len(object_points) < 2:
    #         raise ValueError("At least 2 valid calibration images required")
    #
    #     # Initial calibration
    #     ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
    #         object_points, image_points, (images[0].shape[1], images[0].shape[0]), None, None,  # images[0].shape[::-1]
    #         flags=cv2.CALIB_RATIONAL_MODEL)
    #
    #     # Refine with optimization
    #     self.K = mtx
    #     # self.k = dist[:2]
    #     self.k = dist.flatten()  # Ensure 1D array of all coefficients
    #     self.R = [cv2.Rodrigues(r)[0] for r in rvecs]
    #     self.t = tvecs
    #
    #     # Calculate reprojection error
    #     total_error = 0
    #     per_view_errors = []
    #     for i in range(len(object_points)):
    #         imgpoints2, _ = cv2.projectPoints(
    #             object_points[i], rvecs[i], tvecs[i], mtx, dist)
    #        # error = cv2.norm(image_points[i], imgpoints2.reshape(-1, 2))
    #         error = cv2.norm(image_points[i].reshape(-1, 2).astype(np.float32),
    #                          imgpoints2.reshape(-1, 2).astype(np.float32))
    #
    #         per_view_errors.append(error)
    #         total_error += error
    #
    #     return CalibrationResult(
    #         K=self.K.copy(),
    #         k=self.k.copy(),
    #         R=[R.copy() for R in self.R],
    #         t=[t.copy() for t in self.t],
    #         reprojection_error=total_error / len(object_points),
    #         per_view_errors=per_view_errors
    #     )

    def undistort_image(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        # Use self.k directly (already has correct coefficients)
        # newcameramtx, roi = cv2.getOptimalNewCameraMatrix(self.K, self.k, (w, h), 1, (w, h))
        # Ensure self.k is a (1, 5) numpy array
        self.k = np.array(self.k, dtype=np.float64).reshape(1, -1)

        # Now call OpenCV function
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(self.K, self.k, (w, h), 1, (w, h))

        dst = cv2.undistort(image, self.K, self.k, None, newcameramtx)
        x, y, w, h = roi
        return dst[y:y + h, x:x + w]