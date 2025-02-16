# Camera Calibration

This project implements Zhang's method for camera calibration using a checkerboard pattern.

## Requirements

- Python 3.6+
- OpenCV (cv2)
- NumPy
- SciPy
- Matplotlib

Install requirements using:
```bash
pip install opencv-python numpy scipy matplotlib
```

## Project Structure

```
.
├── Wrapper.py            # Main script to run calibration
├── calibration.py        # Core calibration implementation
└── README.md            # This file
```

## Usage

1. Prepare your checkerboard images in a directory

2. Run the calibration:
```bash
python Wrapper.py --image_dir path/to/images --output_dir path/to/output
```

Arguments:
- `--image_dir`: Directory containing checkerboard calibration images (required)
- `--output_dir`: Directory to save results (default: 'output')

## Output

The script will generate:
1. Calibration parameters saved in NPZ format
2. Rectified images
3. Console output with camera matrix K and distortion coefficients k

## Notes

- The checkerboard pattern is assumed to be 5x8 with a square size of 21.5mm
- The implementation uses the inner 3x6 grid for computation as recommended
- Initial distortion is assumed to be minimal (k = [0, 0])