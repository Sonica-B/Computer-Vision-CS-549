import numpy as np
import cv2


class ZhangCalibration:
    def __init__(self, square_size: float = 21.5):
        self.square_size = square_size
        self.K = None

    def find_corners(self, image):
        """Find checkerboard corners using OpenCV"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        pattern_size = (9, 6)

        # Detect corners
        ret, corners = cv2.findChessboardCorners(gray, pattern_size,
                                                 flags=cv2.CALIB_CB_ADAPTIVE_THRESH +
                                                       cv2.CALIB_CB_NORMALIZE_IMAGE +
                                                       cv2.CALIB_CB_FILTER_QUADS)

        if ret:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            return corners
        return None

    def estimate_homography(self, object_points: np.ndarray, image_points: np.ndarray) -> np.ndarray:
        """
        Estimate homography matrix H using DLT (Section 3.1 in Zhang's paper)
        Args:
            object_points: 3D points in model plane (Z=0)
            image_points: 2D image points
        Returns:
            3x3 homography matrix
        """

        # Normalize points to improve numerical stability
        def normalize_points(points):
            centroid = np.mean(points, axis=0)
            scale = np.sqrt(2) / np.mean(np.linalg.norm(points - centroid, axis=1))
            T = np.array([[scale, 0, -scale * centroid[0]],
                          [0, scale, -scale * centroid[1]],
                          [0, 0, 1]])
            return T, cv2.transform(points.reshape(-1, 1, 2), T).reshape(-1, 2)

        # Normalize image and model points
        T_img, img_normalized = normalize_points(image_points)
        T_obj, obj_normalized = normalize_points(object_points[:, :2])  # Use only X,Y coordinates

        # Build homogeneous system of equations
        n = len(img_normalized)
        A = np.zeros((2 * n, 9))

        for i in range(n):
            x, y = obj_normalized[i]
            u, v = img_normalized[i]
            A[2 * i] = [-x, -y, -1, 0, 0, 0, x * u, y * u, u]
            A[2 * i + 1] = [0, 0, 0, -x, -y, -1, x * v, y * v, v]

        # Solve using SVD
        _, _, Vt = np.linalg.svd(A)
        H = Vt[-1].reshape(3, 3)

        # Denormalize
        H = np.linalg.inv(T_img) @ H @ T_obj
        return H / H[2, 2]

    def extract_intrinsic_parameters(self, homographies: list) -> np.ndarray:
        """
        Extract camera intrinsic matrix K from homographies (Section 3.1 in Zhang's paper)
        Args:
            homographies: List of homography matrices
        Returns:
            Camera intrinsic matrix K
        """
        # Build matrix V for equation Vb = 0
        V = []
        for H in homographies:
            h1, h2 = H[:, 0], H[:, 1]
            V.append(self._create_v_row(h1, h2))
            V.append(self._create_v_row(h1, h1) - self._create_v_row(h2, h2))
        V = np.array(V)

        # Solve using SVD
        _, _, Vt = np.linalg.svd(V)
        b = Vt[-1]
        B11, B12, B22, B13, B23, B33 = b

        # Extract intrinsic parameters
        v0 = (B12 * B13 - B11 * B23) / (B11 * B22 - B12 ** 2)
        lambda_ = B33 - (B13 ** 2 + v0 * (B12 * B13 - B11 * B23)) / B11
        alpha = np.sqrt(lambda_ / B11)
        beta = np.sqrt(lambda_ * B11 / (B11 * B22 - B12 ** 2))
        gamma = -B12 * alpha ** 2 * beta / lambda_
        u0 = gamma * v0 / beta - B13 * alpha ** 2 / lambda_

        return np.array([[alpha, gamma, u0],
                         [0, beta, v0],
                         [0, 0, 1]])

    def _create_v_row(self, h1: np.ndarray, h2: np.ndarray) -> np.ndarray:
        """Create a row of V matrix for intrinsic parameter estimation"""
        return np.array([
            h1[0] * h2[0],
            h1[0] * h2[1] + h1[1] * h2[0],
            h1[1] * h2[1],
            h1[2] * h2[0] + h1[0] * h2[2],
            h1[2] * h2[1] + h1[1] * h2[2],
            h1[2] * h2[2]
        ])

    def estimate_extrinsics(self, H: np.ndarray) -> tuple:
        """
        Estimate extrinsic parameters R,t from homography (Section 3.1 in Zhang's paper)
        Args:
            H: 3x3 homography matrix
        Returns:
            R: 3x3 rotation matrix
            t: 3x1 translation vector
        """
        # Extract rotation columns with proper scale
        h1, h2, h3 = H[:, 0], H[:, 1], H[:, 2]
        K_inv = np.linalg.inv(self.K)

        lambda_ = 1 / np.linalg.norm(K_inv @ h1)
        r1 = lambda_ * K_inv @ h1
        r2 = lambda_ * K_inv @ h2
        r3 = np.cross(r1, r2)
        t = lambda_ * K_inv @ h3

        R = np.column_stack([r1, r2, r3])

        # Ensure proper rotation matrix
        U, _, Vt = np.linalg.svd(R)
        R = U @ Vt

        return R, t

    def calibrate_initial(self, images: list) -> tuple:
        """
        Perform initial calibration (Sections 4.1 and 4.2)
        Args:
            images: List of calibration images
        Returns:
            K: Camera intrinsic matrix
            Rs: List of rotation matrices
            ts: List of translation vectors
        """
        # Create object points
        object_points = np.zeros((6 * 7, 3), np.float32)
        object_points[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2) * self.square_size

        # Get corner points and compute homographies
        homographies = []
        for img in images:
            corners = self.find_corners(img)
            if corners is not None:
                H = self.estimate_homography(object_points, corners.reshape(-1, 2))
                homographies.append(H)

        # Extract camera intrinsic matrix
        self.K = self.extract_intrinsic_parameters(homographies)

        # Estimate extrinsics for each view
        Rs, ts = [], []
        for H in homographies:
            R, t = self.estimate_extrinsics(H)
            Rs.append(R)
            ts.append(t)

        return self.K, Rs, ts


def load_images(directory: str) -> list:
    """Load calibration images from directory"""
    import os
    from pathlib import Path
    import cv2

    image_paths = sorted(Path(directory).glob('*.jpg'))
    images = []

    for path in image_paths:
        img = cv2.imread(str(path))
        if img is not None:
            images.append(img)

    if not images:
        raise ValueError(f"No valid images found in {directory}")

    return images


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, default='D:\\WPI Assignments\\Computer Vision CS549\\HW1\\Data\\Calibration_Imgs',
                        help='Directory containing calibration images')
    parser.add_argument('--square_size', type=float, default=21.5,
                        help='Size of checkerboard squares in mm')
    args = parser.parse_args()

    # Load images
    images = load_images(args.image_dir)

    # Run calibration
    calibrator = ZhangCalibration(square_size=args.square_size)
    K, Rs, ts = calibrator.calibrate_initial(images)

    # Print results
    print("\nCamera Matrix K:")
    print(K)