import numpy as np
import cv2
from typing import List, Tuple, Optional
from scipy.optimize import least_squares
from dataclasses import dataclass


@dataclass
class CalibrationResult:

    K: np.ndarray  # Camera intrinsic matrix
    k: np.ndarray  # Distortion coefficients
    R: List[np.ndarray]  # Rotation matrices for each view
    t: List[np.ndarray]  # Translation vectors for each view
    reprojection_error: float  # Overall reprojection error
    per_view_errors: List[float]  # Reprojection error for each view


class CameraCalibration:
    def __init__(self, square_size: float = 21.5):

        self.square_size = square_size
        self.K = None  # Camera intrinsic matrix
        self.k = np.zeros((5, 1), dtype=np.float32)
        self.R = []  # List of rotation matrices
        self.t = []  # List of translation vectors

    def find_corners(self, image: np.ndarray, pattern_size: Tuple[int, int] = (9, 6)) -> Tuple[
        Optional[np.ndarray], bool]:

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        ret, corners = cv2.findChessboardCorners(
            gray, pattern_size,
            flags=(cv2.CALIB_CB_ADAPTIVE_THRESH +
                   cv2.CALIB_CB_NORMALIZE_IMAGE +
                   cv2.CALIB_CB_FILTER_QUADS)
        )

        if ret and corners is not None:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            return corners, True
        return None, False

    def create_object_points(self, pattern_size: Tuple[int, int]) -> np.ndarray:

        objp = np.zeros((pattern_size[0] * pattern_size[1], 3), dtype=np.float32)
        mgrid = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T
        objp[:, :2] = mgrid.reshape(-1, 2)
        objp *= self.square_size
        return np.ascontiguousarray(objp, dtype=np.float32)

    def estimate_homography(self, obj_points: np.ndarray, img_points: np.ndarray) -> np.ndarray:

        obj_mean = np.mean(obj_points[:, :2], axis=0)
        img_mean = np.mean(img_points, axis=0)
        obj_std = np.std(obj_points[:, :2])
        img_std = np.std(img_points)

        norm_obj = (obj_points[:, :2] - obj_mean) / obj_std
        norm_img = (img_points - img_mean) / img_std

        n = len(obj_points)
        A = np.zeros((2 * n, 9))
        for i in range(n):
            X, Y = norm_obj[i]
            u, v = norm_img[i]
            A[2 * i] = [-X, -Y, -1, 0, 0, 0, X * u, Y * u, u]
            A[2 * i + 1] = [0, 0, 0, -X, -Y, -1, X * v, Y * v, v]

        _, _, Vh = np.linalg.svd(A)
        H = Vh[-1].reshape(3, 3)

        T_obj = np.array([[1 / obj_std, 0, -obj_mean[0] / obj_std],
                          [0, 1 / obj_std, -obj_mean[1] / obj_std],
                          [0, 0, 1]])
        T_img = np.array([[1 / img_std, 0, -img_mean[0] / img_std],
                          [0, 1 / img_std, -img_mean[1] / img_std],
                          [0, 0, 1]])

        H = np.linalg.inv(T_img) @ H @ T_obj
        return H / H[2, 2]

    def _zhang_closed_form(self, homographies: List[np.ndarray], image_size: Tuple[int, int]):

        V = []
        for H in homographies:
            h1, h2, h3 = H.T
            v11 = self._compute_v(h1, h1)
            v12 = self._compute_v(h1, h2)
            v22 = self._compute_v(h2, h2)
            V.append(v12)
            V.append(v11 - v22)

        V = np.array(V)
        _, _, Vh = np.linalg.svd(V)
        b = Vh[-1]

        B = np.array([
            [b[0], b[1], b[3]],
            [b[1], b[2], b[4]],
            [b[3], b[4], b[5]]
        ])

        try:
            v0 = (B[0, 1] * B[0, 2] - B[0, 0] * B[1, 2]) / (B[0, 0] * B[1, 1] - B[0, 1] ** 2)
            lam = B[2, 2] - (B[0, 2] ** 2 + v0 * (B[0, 1] * B[0, 2] - B[0, 0] * B[1, 2])) / B[0, 0]
            fx = np.sqrt(lam / B[0, 0])
            fy = np.sqrt(lam * B[0, 0] / (B[0, 0] * B[1, 1] - B[0, 1] ** 2))
            gamma = -B[0, 1] * fx ** 2 * fy / lam
            u0 = gamma * v0 / fy - B[0, 2] * fx ** 2 / lam
            v0 = v0 if not np.isnan(v0) else image_size[1] / 2
            u0 = u0 if not np.isnan(u0) else image_size[0] / 2
        except:
            fx = fy = (image_size[0] + image_size[1]) / 2
            gamma = 0
            u0, v0 = image_size[0] / 2, image_size[1] / 2

        self.K = np.array([[fx, gamma, u0],
                           [0, fy, v0],
                           [0, 0, 1]], dtype=np.float64)

    def _compute_v(self, h1, h2):
        return np.array([
            h1[0] * h2[0],
            h1[0] * h2[1] + h1[1] * h2[0],
            h1[1] * h2[1],
            h1[2] * h2[0] + h1[0] * h2[2],
            h1[2] * h2[1] + h1[1] * h2[2],
            h1[2] * h2[2]
        ])

    def calibrate(self, images: List[np.ndarray], pattern_size: Tuple[int, int] = (9, 6)) -> CalibrationResult:

        objp = self.create_object_points(pattern_size)
        image_points = []
        homographies = []

        for img in images:
            corners, found = self.find_corners(img, pattern_size)
            if found:
                H = self.estimate_homography(objp, corners.squeeze())
                homographies.append(H)
                image_points.append(corners.astype(np.float32))

        if len(homographies) < 2:
            raise ValueError("At least 2 valid images required")

        h, w = images[0].shape[:2]
        self._zhang_closed_form(homographies, (w, h))

        # Initialize distortion coefficients to zero
        self.k = np.zeros((5, 1), dtype=np.float64)

        # Convert OpenCV format to our parameter structure
        self.R, self.t = [], []
        for H in homographies:
            h1, h2, h3 = H.T
            lam = 1 / np.linalg.norm(np.linalg.inv(self.K) @ h1)
            r1 = lam * np.linalg.inv(self.K) @ h1
            r2 = lam * np.linalg.inv(self.K) @ h2
            r3 = np.cross(r1, r2)
            R = np.vstack([r1, r2, r3]).T
            U, _, Vt = np.linalg.svd(R)
            R = U @ Vt
            t = lam * np.linalg.inv(self.K) @ h3
            self.R.append(R)
            self.t.append(t)

        return self.refine_parameters([objp] * len(image_points), image_points, (w, h))

    def refine_parameters(self, object_points: List[np.ndarray], image_points: List[np.ndarray],
                          image_size: Tuple[int, int]) -> CalibrationResult:

        w, h = image_size

        def compute_residuals(params):
            K = np.array([
                [params[0], params[1], params[3]],  # fx, gamma, cx
                [0, params[2], params[4]],  # fy, cy
                [0, 0, 1]
            ], dtype=np.float64)

            k = np.array([params[5], params[6], 0, 0, 0], dtype=np.float64)
            residuals = []
            idx = 7  # Start of extrinsic parameters

            for i in range(len(object_points)):
                rvec = params[idx:idx + 3]
                tvec = params[idx + 3:idx + 6]
                idx += 6

                proj, _ = cv2.projectPoints(object_points[i], rvec, tvec, K, k)
                error = (image_points[i].reshape(-1, 2) - proj.reshape(-1, 2)).ravel()
                residuals.extend(error.tolist())

            return np.array(residuals, dtype=np.float64)

        # Initial parameters [fx, gamma, fy, cx, cy, k1, k2]
        initial_params = [
            self.K[0, 0], self.K[0, 1], self.K[1, 1],  # fx, gamma, fy
            self.K[0, 2], self.K[1, 2],  # cx, cy
            0.0, 0.0  # k1, k2
        ]

        # Add extrinsics
        for R, t in zip(self.R, self.t):
            rvec, _ = cv2.Rodrigues(R)
            initial_params.extend(rvec.ravel())
            initial_params.extend(t.ravel())

        # Set bounds
        n_params = 7 + 6 * len(object_points)
        lb = np.full(n_params, -np.inf)
        ub = np.full(n_params, np.inf)

        # Intrinsic bounds
        lb[0:3] = [0.5 * w, -100, 0.5 * h]  # fx, gamma, fy
        ub[0:3] = [2.0 * w, 100, 2.0 * h]
        lb[3:5] = [0.4 * w, 0.4 * h]  # cx, cy
        ub[3:5] = [0.6 * w, 0.6 * h]
        lb[5:7] = [-2.0, -2.0]  # k1, k2
        ub[5:7] = [2.0, 2.0]

        # Optimize
        result = least_squares(
            compute_residuals,
            np.array(initial_params),
            bounds=(lb, ub),
            method='trf',
            loss='soft_l1',
            ftol=1e-4,
            xtol=1e-4,
            max_nfev=600,
            verbose=1
        )

        # Extract results
        self.K = np.array([
            [result.x[0], result.x[1], result.x[3]],
            [0, result.x[2], result.x[4]],
            [0, 0, 1]
        ])
        self.k = np.array([result.x[5], result.x[6], 0, 0, 0])

        # Calculate errors
        residuals = compute_residuals(result.x)
        total_error = np.sqrt(np.mean(residuals ** 2))
        per_view = [np.sqrt(np.mean(chunk ** 2))
                    for chunk in np.array_split(residuals, len(object_points))]

        return CalibrationResult(
            K=self.K.copy(),
            k=self.k.copy(),
            R=[R.copy() for R in self.R],
            t=[t.copy() for t in self.t],
            reprojection_error=total_error,
            per_view_errors=per_view
        )

    def undistort_image(self, image: np.ndarray) -> np.ndarray:

        h, w = image.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(self.K, self.k, (w, h), 1, (w, h))
        dst = cv2.undistort(image, self.K, self.k, None, newcameramtx)
        x, y, w, h = roi
        return dst[y:y + h, x:x + w]

    def save_rectified_image(self, image: np.ndarray, object_points: np.ndarray, corners: np.ndarray, index:int,
                             output_path: str = "rectified_reprojected.png") -> None:

        # First undistort the image
        h, w = image.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(self.K, self.k, (w, h), 1, (w, h))
        undist = cv2.undistort(image, self.K, self.k, None, newcameramtx)

        # Make visualization image (copy of undistorted image)
        vis_img = undist.copy()

        R = self.R[index]
        t = self.t[index]

        # Existing projection code
        projected_points, _ = cv2.projectPoints(object_points,
                                                cv2.Rodrigues(R)[0],
                                                t,
                                                self.K,
                                                self.k)

        # Draw detected corners in red
        for corner in corners:
            point = tuple(corner.ravel().astype(int))
            cv2.circle(vis_img, point, 10, (0, 0, 255), -1)  # Red circles

        # Draw reprojected points in green
        for point in projected_points:
            point = tuple(point.ravel().astype(int))
            cv2.circle(vis_img, point, 5, (0, 255, 0), -1)  # Green circles

        # Add legend
        legend_height = 60
        legend = np.full((legend_height, w, 3), 255, dtype=np.uint8)

        # Add legend text and samples
        cv2.circle(legend, (30, 20), 5, (0, 0, 255), -1)
        cv2.putText(legend, "Detected Corners", (50, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        cv2.circle(legend, (30, 45), 5, (0, 255, 0), -1)
        cv2.putText(legend, "Reprojected Points", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        # Combine visualization and legend
        final_img = np.vstack((vis_img, legend))

        # Add error information
        error = np.mean(np.linalg.norm(corners - projected_points, axis=2))
        cv2.putText(final_img, f"Reprojection Error: {error:.3f} pixels",
                    (w - 300, h + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        # Save the final visualization
        cv2.imwrite(output_path, final_img)

        # Optional: Also save undistorted image without visualization
        undist_path = output_path.replace('.png', '_undistorted.png')
        cv2.imwrite(undist_path, undist)