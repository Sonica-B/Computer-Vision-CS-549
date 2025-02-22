import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
# def plot_correspondences(image1_path, image2_path, points1, points2, key):
#     output = "D:\\WPI Assignments\\Computer Vision CS549\\YourDirectoryID_p2\\Group5_p2\\Phase1\\Outputs\\"
#     img1 = cv2.imread(image1_path)
#     img2 = cv2.imread(image2_path)
#
#     h1, w1, _ = img1.shape
#     h2, w2, _ = img2.shape
#
#     canvas = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
#     canvas[:h1, :w1] = img1
#     canvas[:h2, w1:w1 + w2] = img2
#
#     points1 = np.array(points1, dtype=int)
#     points2 = np.array(points2, dtype=int) + np.array([w1, 0])
#
#     for (x1, y1), (x2, y2) in zip(points1, points2):
#         cv2.circle(canvas, (x1, y1), 3, (0, 0, 255), -1)
#         cv2.circle(canvas, (x2, y2), 3, (255, 0, 0), -1)
#         cv2.line(canvas, (x1, y1), (x2, y2), (0, 255, 0), 1)
#
#     cv2.imwrite(f"{output}matching_{key}.jpg", canvas)
#     cv2.imshow("Feature Correspondences", canvas)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

def plot_correspondences(image1_path, image2_path, points1, points2, key):
    """
    Plot and save feature correspondences between two images.
    Saves output without displaying.
    """
    # Define output path - using relative path
    output = "D:\\WPI Assignments\\Computer Vision CS549\\YourDirectoryID_p2\\Group5_p2\\Phase1\\Outputs\\"

    # Create output directory if it doesn't exist
    os.makedirs(output, exist_ok=True)

    # Read images
    img1 = cv2.imread(image1_path)
    img2 = cv2.imread(image2_path)

    if img1 is None or img2 is None:
        print(f"Error: Could not read images: {image1_path} or {image2_path}")
        return

    # Get dimensions
    h1, w1, _ = img1.shape
    h2, w2, _ = img2.shape

    # Create canvas for side-by-side visualization
    canvas = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    canvas[:h1, :w1] = img1
    canvas[:h2, w1:w1 + w2] = img2

    # Convert points to integer type and adjust second image points
    points1 = np.array(points1, dtype=int)
    points2 = np.array(points2, dtype=int) + np.array([w1, 0])

    # Draw correspondences
    for (x1, y1), (x2, y2) in zip(points1, points2):
        cv2.circle(canvas, (x1, y1), 3, (0, 0, 255), -1)  # Red circles for first image
        cv2.circle(canvas, (x2, y2), 3, (255, 0, 0), -1)  # Blue circles for second image
        cv2.line(canvas, (x1, y1), (x2, y2), (0, 255, 0), 1)  # Green lines for matches

    # Save the visualization
    output_path = f"{output}matching_{key}.jpg"
    cv2.imwrite(output_path, canvas)
    print(f"Saved correspondence visualization to: {output_path}")

def plot_inliers(img1, img2, pts1, pts2, inliers, key):
    output = "D:\\WPI Assignments\\Computer Vision CS549\\YourDirectoryID_p2\\Group5_p2\\Phase1\\Outputs\\"
    img1_color = cv2.imread(img1)
    img2_color = cv2.imread(img2)

    inlier_pts1 = np.array([pts1[i] for i in inliers])
    inlier_pts2 = np.array([pts2[i] for i in inliers])

    combined_img = np.hstack((img1_color, img2_color))
    offset = img1_color.shape[1]

    for i in range(len(inlier_pts1)):
        pt1 = tuple(map(int, inlier_pts1[i]))
        pt2 = tuple(map(int, inlier_pts2[i] + np.array([offset, 0])))
        cv2.line(combined_img, pt1, pt2, (0, 255, 0), 1)
        cv2.circle(combined_img, pt1, 4, (0, 0, 255), -1)
        cv2.circle(combined_img, pt2, 4, (255, 0, 0), -1)

    cv2.imwrite(f"{output}ransac_{key}.jpg", combined_img)
    # cv2.imshow("Matched Inliers", combined_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

## PUT BELOW IN WRAPPER.PY FOR RANSAC VISUALISATION
# img_folder = "/Users/tvidk/Library/CloudStorage/OneDrive-WorcesterPolytechnicInstitute(wpi.edu)/WPI/Education/RBE 549/Project 2/Divam/YourDirectoryID_p2/Phase1/P2Data/"
# output_path = "/Users/tvidk/Library/CloudStorage/OneDrive-WorcesterPolytechnicInstitute(wpi.edu)/WPI/Education/RBE 549/Project 2/Divam/YourDirectoryID_p2/Phase1/Outputs/"
# for i, key in enumerate(matches):
#     pts1 = np.array(matches[key][0])
#     pts2 = np.array(matches[key][1])
#     best_inliers = inliers[i]
#     img1 = f"{img_folder}{key.split('_')[0]}.png"
#     img2 = f"{img_folder}{key.split('_')[1]}.png"
#     plot_inliers(img1, img2, pts1, pts2, best_inliers, key)

def plot_3d_points_and_cameras(points_3d, camera_poses, save_path):
    """
    Plot 3D points and camera positions with proper scaling

    Args:
        points_3d: Dictionary with image pair keys and corresponding 3D points
        camera_poses: Dictionary with camera indices and their poses (C, R)
        save_path: Optional path to save the plot
    """
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Collect all points and camera centers for normalization
    all_points = []
    all_centers = []

    for points in points_3d.values():
        if points is not None and len(points) > 0:
            all_points.extend(points)

    for C, _ in camera_poses.values():
        all_centers.append(C.ravel())

    all_points = np.array(all_points)
    all_centers = np.array(all_centers)

    if len(all_points) > 0:
        # Compute center and scale
        center = np.mean(all_points, axis=0)
        scale = np.max(np.abs(all_points - center)) + 1e-10  # Add small epsilon to avoid division by zero

        # Plot 3D points with different colors for each image pair
        colors = plt.cm.rainbow(np.linspace(0, 1, len(points_3d)))
        for (key, points), color in zip(points_3d.items(), colors):
            if points is not None and len(points) > 0:
                # Normalize points
                normalized_points = (points - center) / scale
                ax.scatter(normalized_points[:, 0],
                           normalized_points[:, 1],
                           normalized_points[:, 2],
                           c=[color], s=1, label=f'Points {key}', alpha=0.6)

        # Plot cameras
        for cam_idx, (C, R) in camera_poses.items():
            # Normalize camera center
            C_normalized = (C.ravel() - center) / scale

            # Plot camera center
            ax.scatter(C_normalized[0], C_normalized[1], C_normalized[2],
                       c='red', s=100, marker='o', label=f'Camera {cam_idx}')

            # Plot camera axes
            axis_length = 0.1  # Adjust this value to change the length of camera axes
            colors = ['r', 'g', 'b']
            for i, c in enumerate(colors):
                direction = R[:, i] * axis_length
                ax.quiver(C_normalized[0], C_normalized[1], C_normalized[2],
                          direction[0], direction[1], direction[2],
                          color=c, alpha=0.6)

        # Set equal aspect ratio
        ax.set_box_aspect([1, 1, 1])

        # Set reasonable axis limits
        limit = 1.5  # Adjust this value to change the plot boundaries
        ax.set_xlim([-limit, limit])
        ax.set_ylim([-limit, limit])
        ax.set_zlim([-limit, limit])

        # Add labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('3D Reconstruction with Camera Poses')

        # Adjust legend
        handles, labels = ax.get_legend_handles_labels()
        # Remove duplicate camera labels
        unique_labels = []
        unique_handles = []
        for handle, label in zip(handles, labels):
            if label not in unique_labels:
                unique_labels.append(label)
                unique_handles.append(handle)
        ax.legend(unique_handles, unique_labels, bbox_to_anchor=(1.05, 1),
                  loc='upper left', borderaxespad=0.)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)

        plt.show()
    else:
        print("No points to plot!")


def plot_initial_triangulation_with_poses(points_3d_list, centers_list, rotations_list):
    """Plot initial triangulation showing all four possible camera poses."""
    plt.figure(figsize=(10, 10))
    ax = plt.axes(projection='3d')

    colors = ['r', 'g', 'b', 'y']
    for i in range(4):
        # Plot 3D points
        points = points_3d_list[i]
        ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                   c=colors[i], s=1, alpha=0.6, label=f'Config {i + 1}')

        # Plot camera center
        C = centers_list[i]
        ax.scatter(C[0], C[1], C[2], c=colors[i], s=100, marker='o')

        # Plot camera orientation
        R = rotations_list[i]
        scale = 1.0
        ax.quiver(C[0], C[1], C[2],
                  R[0, 0] * scale, R[1, 0] * scale, R[2, 0] * scale,
                  color=colors[i])
        ax.quiver(C[0], C[1], C[2],
                  R[0, 1] * scale, R[1, 1] * scale, R[2, 1] * scale,
                  color=colors[i])
        ax.quiver(C[0], C[1], C[2],
                  R[0, 2] * scale, R[1, 2] * scale, R[2, 2] * scale,
                  color=colors[i])

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Initial Triangulation with Four Possible Camera Poses')
    plt.legend()
    plt.savefig('Outputs/initial_triangulation.png')
    plt.close()


def compare_linear_nonlinear_triangulation(points_3d_linear, points_3d_nonlinear, index):
    """Compare 3D points from linear and non-linear triangulation."""
    plt.figure(figsize=(15, 7))

    # Linear triangulation plot
    ax1 = plt.subplot(121, projection='3d')
    ax1.scatter(points_3d_linear[:, 0], points_3d_linear[:, 1], points_3d_linear[:, 2],
                c='b', s=1)
    ax1.set_title('Linear Triangulation')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')

    # Non-linear triangulation plot
    ax2 = plt.subplot(122, projection='3d')
    ax2.scatter(points_3d_nonlinear[:, 0], points_3d_nonlinear[:, 1], points_3d_nonlinear[:, 2],
                c='r', s=1)
    ax2.set_title('Non-linear Triangulation')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')

    plt.savefig(f'Outputs/triangulation_comparison_{index}.png')
    plt.close()


def plot_reprojection_comparison(img1_path, img2_path, pts1, pts2,
                                 proj_pts1_linear, proj_pts2_linear,
                                 proj_pts1_nonlinear, proj_pts2_nonlinear, index):
    """Compare reprojections between linear and non-linear triangulation."""
    # Read images
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Linear triangulation results - First row
    axes[0, 0].imshow(img1)
    axes[0, 0].scatter(pts1[:, 0], pts1[:, 1], c='g', s=20, label='Features')
    axes[0, 0].scatter(proj_pts1_linear[:, 0], proj_pts1_linear[:, 1],
                       c='r', s=20, label='Reprojections')
    axes[0, 0].set_title('Image 1 - Linear Triangulation')
    axes[0, 0].legend()

    axes[0, 1].imshow(img2)
    axes[0, 1].scatter(pts2[:, 0], pts2[:, 1], c='g', s=20, label='Features')
    axes[0, 1].scatter(proj_pts2_linear[:, 0], proj_pts2_linear[:, 1],
                       c='r', s=20, label='Reprojections')
    axes[0, 1].set_title('Image 2 - Linear Triangulation')
    axes[0, 1].legend()

    # Non-linear triangulation results - Second row
    axes[1, 0].imshow(img1)
    axes[1, 0].scatter(pts1[:, 0], pts1[:, 1], c='g', s=20, label='Features')
    axes[1, 0].scatter(proj_pts1_nonlinear[:, 0], proj_pts1_nonlinear[:, 1],
                       c='r', s=20, label='Reprojections')
    axes[1, 0].set_title('Image 1 - Non-linear Triangulation')
    axes[1, 0].legend()

    axes[1, 1].imshow(img2)
    axes[1, 1].scatter(pts2[:, 0], pts2[:, 1], c='g', s=20, label='Features')
    axes[1, 1].scatter(proj_pts2_nonlinear[:, 0], proj_pts2_nonlinear[:, 1],
                       c='r', s=20, label='Reprojections')
    axes[1, 1].set_title('Image 2 - Non-linear Triangulation')
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(f'Outputs/reprojection_comparison_{index}.png')
    plt.close()