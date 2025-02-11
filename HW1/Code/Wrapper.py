import argparse
from pathlib import Path
from calibration import CameraCalibration
import numpy as np
import cv2
import matplotlib.pyplot as plt


def main(args):
    # Initialize calibrator with checkerboard parameters
    calibrator = CameraCalibration(
        square_size=21.5,  # mm
        board_pattern=(3, 6)  # inner corners as specified
    )

    # Get all calibration images
    image_paths = list(Path(args.image_dir).glob('*.jpg'))
    if not image_paths:
        raise ValueError(f"No images found in {args.image_dir}")

    # Step 1: Initial parameter estimation
    print("Estimating initial parameters...")
    calibrator.estimate_initial_parameters(image_paths)

    # Print initial parameters
    print("\nInitial Parameters:")
    print(f"Camera Matrix K:\n{calibrator.K}")
    print(f"Distortion Coefficients k: {calibrator.k}")

    # Step 2: Non-linear optimization
    print("\nOptimizing parameters...")
    obj_points = calibrator.create_object_points()
    img_points = []
    for image_path in image_paths:
        corners, _ = calibrator.find_checkerboard_corners(str(image_path))
        if corners is not None:
            img_points.append(corners)

    final_error = calibrator.optimize_parameters(obj_points, img_points)

    # Print final results
    print("\nFinal Parameters:")
    print(f"Camera Matrix K:\n{calibrator.K}")
    print(f"Distortion Coefficients k: {calibrator.k}")
    print(f"Re-projection Error: {final_error}")

    # Save results if output directory is specified
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True)

        # Save parameters
        np.savez(output_dir / 'calibration_results.npz',
                 K=calibrator.K,
                 k=calibrator.k,
                 R=calibrator.R,
                 t=calibrator.t)

        # Generate and save rectified images
        for i, image_path in enumerate(image_paths):
            img = cv2.imread(str(image_path))
            h, w = img.shape[:2]
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
                calibrator.K,
                np.hstack((calibrator.k, [0, 0, 0])),  # OpenCV expects 5 distortion coefficients
                (w, h),
                1,
                (w, h)
            )

            # Rectify image
            dst = cv2.undistort(img, calibrator.K,
                                np.hstack((calibrator.k, [0, 0, 0])),
                                None,
                                newcameramtx)

            # Save rectified image
            cv2.imwrite(str(output_dir / f'rectified_{i}.jpg'), dst)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Camera Calibration')
    parser.add_argument('--image_dir', default='../Data/Calibration_Imgs', type=str, required=True,
                        help='Directory containing calibration images')
    parser.add_argument('--output_dir', type=str, default='Output',
                        help='Directory to save results')
    args = parser.parse_args()

    main(args)