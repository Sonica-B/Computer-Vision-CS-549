import cv2
import numpy as np

def plot_correspondences(image1_path, image2_path, points1, points2, key):
    output = "/Users/tvidk/Library/CloudStorage/OneDrive-WorcesterPolytechnicInstitute(wpi.edu)/WPI/Education/RBE 549/Project 2/Divam/YourDirectoryID_p2/Phase1/Outputs/"
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
    output = "/Users/tvidk/Library/CloudStorage/OneDrive-WorcesterPolytechnicInstitute(wpi.edu)/WPI/Education/RBE 549/Project 2/Divam/YourDirectoryID_p2/Phase1/Outputs/"
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