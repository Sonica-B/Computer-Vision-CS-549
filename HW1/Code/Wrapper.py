import argparse
from pathlib import Path
import logging
from autocalib import *
import matplotlib.pyplot as plt
from datetime import datetime
import json
import os


class CalibrationWrapper:
    def __init__(self, image_dir: str, output_dir: str, square_size: float = 21.5):
        """
        Initialize calibration wrapper
        Args:
            image_dir: Directory containing calibration images
            output_dir: Directory to save results
            square_size: Size of checkerboard squares in mm
        """
        self.image_dir = Path(image_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.square_size = square_size

        # Setup logging
        self.setup_logger()

        # Initialize calibrator
        self.calibrator = CameraCalibration(
            square_size=square_size,
            pattern_size=(3, 6)  # Inner corners as per assignment
        )

    def setup_logger(self):
        """Setup logging configuration"""
        log_file = self.output_dir / 'calibration.log'

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger("CalibrationWrapper")

    def load_images(self) -> list:
        """Load calibration images"""
        image_paths = sorted(self.image_dir.glob("*.jpg"))
        if not image_paths:
            raise ValueError(f"No images found in {self.image_dir}")

        images = []
        for path in image_paths:
            img = cv2.imread(str(path))
            if img is None:
                self.logger.warning(f"Could not read image: {path}")
                continue
            images.append(img)

        self.logger.info(f"Loaded {len(images)} images")
        return images

    def process_image(self, img: np.ndarray, idx: int) -> Tuple[Optional[np.ndarray], bool]:
        """
        Process single calibration image with visualization
        Args:
            img: Input image
            idx: Image index
        Returns:
            corners: Detected corners if found
            success: Whether corners were found
        """
        # Create debug directory
        debug_dir = self.output_dir / 'debug'
        debug_dir.mkdir(exist_ok=True)

        # Save original image
        cv2.imwrite(str(debug_dir / f'original_{idx}.jpg'), img)

        # Find corners
        corners, success = self.calibrator.find_corners(img)

        if success:
            # Draw and save corner visualization
            vis = self.calibrator.draw_corners(img, corners)
            cv2.imwrite(str(debug_dir / f'corners_{idx}.jpg'), vis)
            self.logger.info(f"Successfully found corners in image {idx}")
        else:
            self.logger.warning(f"No corners found in image {idx}")

        return corners, success



    def save_corner_detection_results(self, image: np.ndarray, corners: np.ndarray,
                                      idx: int) -> None:
        """Save corner detection visualization"""
        vis_img = image.copy()
        cv2.drawChessboardCorners(vis_img, (3, 6), corners, True)
        cv2.imwrite(str(self.output_dir / f'corners_{idx}.jpg'), vis_img)

    def plot_reprojection_errors(self, errors: list) -> None:
        """Plot reprojection errors"""
        plt.figure(figsize=(10, 5))
        plt.plot(errors, 'bo-')
        plt.title('Reprojection Error per Image')
        plt.xlabel('Image Index')
        plt.ylabel('RMS Error (pixels)')
        plt.grid(True)
        plt.savefig(self.output_dir / 'reprojection_errors.png')
        plt.close()

    def generate_report(self, results: dict) -> None:
        """Generate calibration report"""
        report_path = self.output_dir / 'calibration_report.txt'

        with open(report_path, 'w') as f:
            f.write("Camera Calibration Report\n")
            f.write("=======================\n\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("Camera Matrix (K):\n")
            f.write(f"{results['K']}\n\n")

            f.write("Distortion Coefficients [k1, k2]:\n")
            f.write(f"{results['k']}\n\n")

            f.write("Reprojection Error Statistics:\n")
            f.write(f"Mean error: {results['mean_error']:.3f} pixels\n")
            f.write(f"Max error: {results['max_error']:.3f} pixels\n")
            f.write(f"Standard deviation: {results['std_error']:.3f} pixels\n\n")

            f.write("Per-image Reprojection Errors:\n")
            for i, error in enumerate(results['per_view_errors']):
                f.write(f"Image {i + 1}: {error:.3f} pixels\n")

    def run_calibration(self):
        """
        Run complete camera calibration pipeline following Zhang's method
        - Handles 13 images from Google Pixel XL
        - Uses 3x6 inner corners (5x8 pattern)
        - Implements zhang's constraints
        - Performs full optimization
        - Generates all required outputs
        """
        try:
            # Create debug directory
            debug_dir = self.output_dir / 'debug'
            debug_dir.mkdir(exist_ok=True)

            # 1. Load and validate images
            self.logger.info("Loading calibration images...")
            images = self.load_images()

            if len(images) != 13:
                self.logger.warning(f"Expected 13 images, found {len(images)}")

            # 2. Process images and detect corners
            object_points = []
            image_points = []
            successful_images = []

            obj_pattern = self.calibrator.create_object_points()

            for idx, image in enumerate(images, 1):
                self.logger.info(f"Processing image {idx}/{len(images)}")

                # Save original image
                cv2.imwrite(str(debug_dir / f'original_{idx}.jpg'), image)

                # Find checkerboard corners
                corners, success = self.calibrator.find_corners(image)

                if success:
                    # Verify corners form valid pattern
                    if self.calibrator.verify_corner_positions(corners):
                        # Draw and save corner visualization
                        vis = self.calibrator.draw_corners(image, corners)
                        cv2.imwrite(str(debug_dir / f'corners_{idx}.jpg'), vis)

                        object_points.append(obj_pattern)
                        image_points.append(corners)
                        successful_images.append(image)

                        self.logger.info(f"Successfully processed image {idx}")
                    else:
                        self.logger.warning(f"Invalid corner pattern in image {idx}")
                else:
                    self.logger.warning(f"No corners found in image {idx}")

            if len(successful_images) < 2:
                raise ValueError("Need at least 2 valid calibration images!")

            # 3. Initial parameter estimation
            self.logger.info("Performing initial calibration...")

            # 3.1 Estimate homographies
            homographies = []
            valid_indices = []

            for idx, (obj_pts, img_pts) in enumerate(zip(object_points, image_points)):
                H = self.calibrator.estimate_homography(obj_pts[:, :2], img_pts.reshape(-1, 2))

                # Validate homography
                if self.calibrator.validate_homography_constraints(H):
                    homographies.append(H)
                    valid_indices.append(idx)
                    self.logger.info(f"Valid homography for image {idx + 1}")
                else:
                    self.logger.warning(f"Invalid homography for image {idx + 1}")

            # Filter points based on valid homographies
            object_points = [object_points[i] for i in valid_indices]
            image_points = [image_points[i] for i in valid_indices]
            successful_images = [successful_images[i] for i in valid_indices]

            # 3.2 Check for degenerate configurations
            valid_configs = self.calibrator.check_critical_configurations(successful_images)

            # Filter out degenerate configurations
            object_points = [obj for obj, valid in zip(object_points, valid_configs) if valid]
            image_points = [img for img, valid in zip(image_points, valid_configs) if valid]

            self.logger.info(f"Using {len(object_points)} valid images for calibration")

            # 4. Perform calibration
            self.logger.info("Starting Zhang's calibration method...")
            result = self.calibrator.calibrate(successful_images)

            # 5. Generate and save outputs
            self.logger.info("Generating calibration outputs...")

            # Save calibration parameters
            np.savez(self.output_dir / 'calibration_params.npz',
                     K=result.K,
                     k=result.k,
                     R=result.R,
                     t=result.t,
                     error=result.reprojection_error)

            # Save undistorted images
            for idx, image in enumerate(successful_images):
                undistorted = self.calibrator.undistort_image(image)
                cv2.imwrite(str(self.output_dir / f'undistorted_{idx + 1}.jpg'), undistorted)

            # Generate reprojection error visualization
            self.plot_reprojection_errors(result.per_view_errors)

            # Generate comprehensive report
            self.generate_calibration_report(result)

            self.logger.info(f"""
            Calibration completed successfully:
            - Processed {len(images)} images
            - Used {len(object_points)} valid configurations
            - Final reprojection error: {result.reprojection_error:.4f} pixels
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

    wrapper.run_calibration()


if __name__ == "__main__":
    main()