import cv2
import numpy as np
import matplotlib.pyplot as plt
def plot_correspondences(image1_path, image2_path, points1, points2, key):
    output = "D:\\WPI Assignments\\Computer Vision CS549\\YourDirectoryID_p2\\Group5_p2\\Phase1\\Outputs\\"
    img1 = cv2.imread(image1_path)
    img2 = cv2.imread(image2_path)

    h1, w1, _ = img1.shape
    h2, w2, _ = img2.shape

    canvas = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    canvas[:h1, :w1] = img1
    canvas[:h2, w1:w1 + w2] = img2

    points1 = np.array(points1, dtype=int)
    points2 = np.array(points2, dtype=int) + np.array([w1, 0])

    for (x1, y1), (x2, y2) in zip(points1, points2):
        cv2.circle(canvas, (x1, y1), 3, (0, 0, 255), -1)
        cv2.circle(canvas, (x2, y2), 3, (255, 0, 0), -1)
        cv2.line(canvas, (x1, y1), (x2, y2), (0, 255, 0), 1)
    
    cv2.imwrite(f"{output}matching_{key}.jpg", canvas)
    cv2.imshow("Feature Correspondences", canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


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