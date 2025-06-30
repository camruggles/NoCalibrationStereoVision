import numpy as np

# A is your 4x12 array
# Example dummy data (replace with your own)
A = np.random.rand(4, 12)

# --- core operation -------------------------------------------------
# Broadcast the division: first 3 rows ÷ 4th row (column-wise)
B = A[:3] / A[3]          # shape: (3, 12)
# --------------------------------------------------------------------

print("Original shape:", A.shape)   # (4, 12)
print("Converted shape:", B.shape)  # (3, 12)
print(B)


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D   # registers 3-D projection

# pts: shape (3, 12)  →  rows = x, y, z
# example dummy data (replace with your own)
# pts = np.random.rand(3, 12)

def make_3d_plot(pts):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # scatter-plot the 12 points
    ax.scatter(pts[0], pts[1], pts[2])

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3-D point cloud (12 points)')

    plt.tight_layout()
    plt.show()

print(np.eye(3))
make_3d_plot(np.eye(3))