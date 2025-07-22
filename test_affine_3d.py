
import cv2
import cv2.aruco as aruco
import numpy as np
import pdb
from example_plot import plot_pts_correspondence


from fundamental_matrix import detect_aruco


file1 = "./phone_stream/01830.jpg"
file2 = "./ipad_stream/00744.jpg"

image = cv2.imread(file1)
print(image.shape)

corners1 = detect_aruco(file1)
corners2 = detect_aruco(file2)

print(corners1)
print(corners2)


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
from matplotlib.image import imread

def plot_correspondences_on_images(pts1: np.ndarray, pts2: np.ndarray,
                                   img_path1: str, img_path2: str,
                                   title1="Image 1", title2="Image 2"):
    """
    Overlay corresponding points on two images and draw lines between them.

    Parameters
    ----------
    pts1, pts2 : (N,2) float arrays
        Row i in pts1 corresponds to row i in pts2. Coordinates are (x, y)
        in pixel units (x = column index, y = row index).
    img_path1, img_path2 : str
        File paths to the two images.
    """
    pts1 = np.asarray(pts1, dtype=float)
    pts2 = np.asarray(pts2, dtype=float)
    if pts1.shape != pts2.shape or pts1.ndim != 2 or pts1.shape[1] != 2:
        raise ValueError("pts1 and pts2 must both be of shape (N, 2).")

    img1 = imread(img_path1)
    img2 = imread(img_path2)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    # Show images
    ax1.imshow(img1, origin='upper')
    ax2.imshow(img2, origin='upper')
    ax1.set_title(title1)
    ax2.set_title(title2)

    # Plot points
    ax1.scatter(pts1[:, 0], pts1[:, 1], s=30, edgecolors='white', facecolors='none')
    ax2.scatter(pts2[:, 0], pts2[:, 1], s=30, edgecolors='white', facecolors='none')

    # Make axes tidy
    for ax in (ax1, ax2):
        ax.set_axis_off()
        ax.set_xlim([0, img1.shape[1] if ax is ax1 else img2.shape[1]])
        ax.set_ylim([img1.shape[0] if ax is ax1 else img2.shape[0], 0])  # invert y for image coords

    # Draw correspondence lines
    for (x1, y1), (x2, y2) in zip(pts1, pts2):
        con = ConnectionPatch(xyA=(x2, y2), xyB=(x1, y1),
                              coordsA="data", coordsB="data",
                              axesA=ax2, axesB=ax1,
                              color="yellow", linewidth=1, alpha=0.6)
        fig.add_artist(con)

    plt.tight_layout()
    plt.show()

# Example:
# pts1 = np.array([[50, 60], [120, 90], [200, 150]])
# pts2 = pts1 + np.array([10, -5])
corner_indices = [0,1,3,8]
print()
print(corners1[corner_indices, :], corners2[corner_indices, :])
# quit()
plot_correspondences_on_images(corners1[corner_indices, :], corners2[corner_indices, :], file1, file2)

# # Example usage:
# pts1 = np.random.rand(10,2)
# pts2 = pts1 + np.array([0.5, 0.2])  # translated
# plot_correspondences(corners1, corners2)


# mark the tetrahedron points you want
# 0,1,2,3
# 4,5,6,7
# 8,9,10,11

# 0
# 5
# 8
# 9