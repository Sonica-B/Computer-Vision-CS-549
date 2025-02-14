import argparse
import logging
from pathlib import Path
#from autocalib import *
from Utils.corner_detection import *

# class CalibrationWrapper:
#     def __init__(self, image_dir: str, output_dir: str, square_size: float = 21.5):
#         self.image_dir = Path(image_dir)
#         self.output_dir = Path(output_dir)
#         self.output_dir.mkdir(parents=True, exist_ok=True)
#         self.calibrator = CameraCalibration(square_size=square_size)
#         self._setup_logger()
#
#     def _setup_logger(self):
#         log_file = self.output_dir / 'calibration.log'
#         format_str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
#         logging.basicConfig(
#             level=logging.INFO,
#             format=format_str,
#             handlers=[
#                 logging.FileHandler(log_file),
#                 logging.StreamHandler()
#             ]
#         )
#         self.logger = logging.getLogger("CalibrationWrapper")
#
#     def run_calibration(self):
#         try:
#             # Load images
#             images = []
#             image_paths = sorted(self.image_dir.glob('*.jpg'))
#             for path in image_paths:
#                 img = cv2.imread(str(path))
#                 if img is not None:
#                     images.append(img)
#
#             self.logger.info(f"Loaded {len(images)} images")
#
#             # Debug directory for visualizations
#             debug_dir = self.output_dir / 'debug'
#             debug_dir.mkdir(exist_ok=True)
#
#             # Test corner detection
#             for i, img in enumerate(images):
#                 corners, ret = self.calibrator.find_corners(img)
#                 if ret:
#                     vis = img.copy()
#                     cv2.drawChessboardCorners(vis, (6, 7), corners, True)
#                     cv2.imwrite(str(debug_dir / f'corners_{i:02d}.jpg'), vis)
#                     self.logger.info(f"Found corners in image {i}")
#                 else:
#                     self.logger.warning(f"Failed to find corners in image {i}")
#
#             # Perform calibration
#             result = self.calibrator.calibrate(images)
#
#             # Save undistorted images
#             for i, img in enumerate(images):
#                 undistorted = self.calibrator.undistort_image(img)
#                 cv2.imwrite(str(debug_dir / f'undistorted_{i:02d}.jpg'), undistorted)
#
#             # Save calibration parameters
#             np.savez(self.output_dir / 'calibration.npz',
#                      K=result.K,
#                      k=result.k,
#                      R=result.R,
#                      t=result.t,
#                      error=result.reprojection_error)
#
#
#             self.logger.info(f"""
#             Calibration completed successfully:
#             - Processed {len(images)} images
#             - Final reprojection error: {result.reprojection_error:.4f} pixels
#             - Results saved to {self.output_dir}
#             """)
#
#             return result
#
#         except Exception as e:
#             self.logger.error(f"Calibration failed: {str(e)}")
#             raise


class CalibrationWrapper:
    """Wrapper class for camera calibration process."""

    def __init__(self, image_dir: str, output_dir: str, square_size: float = 21.5):
        """
        Initialize the calibration wrapper.

        Args:
            image_dir: Directory containing calibration images
            output_dir: Directory to save results
            square_size: Size of checkerboard squares in mm
        """
        self.image_dir = Path(image_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.calibrator = CameraCalibration(square_size=square_size)
        self._setup_logger()

    def _setup_logger(self):
        """Setup logging configuration."""
        log_file = self.output_dir / 'calibration.log'
        format_str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        logging.basicConfig(
            level=logging.INFO,
            format=format_str,
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger("CalibrationWrapper")

    def run_calibration(self) -> CalibrationResult:
        """
        Run the complete calibration pipeline.

        Returns:
            CalibrationResult object containing calibration parameters
        """
        try:
            # Create debug directory for visualizations
            debug_dir = self.output_dir / 'debug'
            debug_dir.mkdir(exist_ok=True)

            # Load and process images
            images = []
            image_paths = sorted(self.image_dir.glob('*.jpg'))
            if not image_paths:
                self.logger.warning("No .jpg images found, trying .png")
                image_paths = sorted(self.image_dir.glob('*.png'))

            self.logger.info(f"Found {len(image_paths)} images")

            # First pass: detect corners and save visualizations
            valid_images = []
            corners_list = []
            pattern_size = (9, 6)  # Inner corners of the checkerboard

            for i, path in enumerate(image_paths):
                img = cv2.imread(str(path))
                if img is None:
                    self.logger.error(f"Failed to load image: {path}")
                    continue

                corners, ret = self.calibrator.find_corners(img, pattern_size)

                if ret:
                    # Save corner detection visualization
                    vis = img.copy()
                    cv2.drawChessboardCorners(vis, pattern_size, corners, ret)
                    cv2.imwrite(str(debug_dir / f'corners_{i:02d}.jpg'), vis)

                    valid_images.append(img)
                    corners_list.append(corners)
                    self.logger.info(f"Successfully found corners in image {i}")
                else:
                    self.logger.warning(f"Failed to find corners in image {i}")

            if len(valid_images) < 2:
                raise ValueError("At least 2 valid calibration images required")

            # Perform calibration
            self.logger.info("Starting calibration...")
            result = self.calibrator.calibrate(valid_images, pattern_size)

            # Save undistorted images and corner reprojections
            objp = self.calibrator.create_object_points(pattern_size)
            for i, (img, corners) in enumerate(zip(valid_images, corners_list)):
                # Save undistorted image with reprojected corners
                self.calibrator.save_rectified_image(
                    img,
                    objp,
                    corners,
                    index=i,
                    output_path=str(debug_dir / f'calibration_result_{i:02d}.png')
                )

            # Save calibration parameters
            np.savez(self.output_dir / 'calibration.npz',
                     K=result.K,
                     k=result.k,
                     R=result.R,
                     t=result.t,
                     error=result.reprojection_error,
                     per_view_errors=result.per_view_errors)

            # Log results
            self.logger.info(f"""
            Calibration completed successfully:
            - Processed {len(valid_images)} images
            - Final reprojection error: {result.reprojection_error:.4f} pixels
            - Camera matrix K:
            {result.K}
            - Distortion coefficients k:
            {result.k.ravel()}
            - Per-view reprojection errors:
            {', '.join(f'{err:.4f}' for err in result.per_view_errors)}
            - Results saved to {self.output_dir}
            """)

            return result

        except Exception as e:
            self.logger.error(f"Calibration failed: {str(e)}")
            raise


def main():
    parser = argparse.ArgumentParser(description='Camera Calibration using Zhang\'s method')
    parser.add_argument('--image_dir', type=str, default='D:\\WPI Assignments\\Computer Vision CS549\\HW1\\Data\\Calibration_Imgs',
                        help='Directory containing calibration images')
    parser.add_argument('--output_dir', type=str, default='Outputs',
                        help='Directory to save results')
    parser.add_argument('--square_size', type=float, default=21.5,
                        help='Size of checkerboard squares in mm')


    args = parser.parse_args()

    wrapper = CalibrationWrapper(
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        square_size=args.square_size
    )

    try:
        wrapper.run_calibration()
    except Exception as e:
        logging.error(f"Calibration failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()