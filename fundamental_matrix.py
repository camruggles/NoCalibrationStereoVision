
import cv2
import cv2.aruco as aruco
import numpy as np
import pdb
from example_plot import plot_pts_correspondence

def detect_aruco(image_path):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image not found!")
        return
    
    # Convert image to grayscale as required for detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Define the ArUco dictionary we are using (e.g., 6x6 with 250 possible ids)
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_7X7_100)
    parameters =  cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

    # Initialize detector parameters using default values
    
    # Detect the markers in the image
    # import pdb
    # pdb.set_trace()
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict)
    # print(ids,corners)
    
    sorted_indices = np.argsort(ids.flatten())  # Get sorted indices based on ids
    sorted_corners = [corners[i] for i in sorted_indices]
    sorted_ids = ids[sorted_indices]
    print(sorted_corners, sorted_ids)
    # quit()
    # If markers are detected, draw them and overlay the ids
    if ids is not None:
        # Draw the detected markers on the image
        aruco.drawDetectedMarkers(image, corners, ids)
        
        # Loop through each detected marker and overlay the id at its center
        for i, corner in enumerate(sorted_corners):
            # corner is an array with shape (1, 4, 2): 4 corner points (x, y)
            corners_arr = corner[0]
            center = np.mean(corners_arr, axis=0).astype(int)
            marker_id = str(ids[i][0])
            cv2.putText(image, marker_id, tuple(center), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    else:
        print("No ArUco markers detected.")

    # Display the result
    print(image.shape)
    image[750:762, 1880:1896 :] = [0,0,0]
    print(image.shape)
    window_name = "Detected Aruco Corners"
    # cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    # cv2.resizeWindow(window_name, 1920,1080)
    # cv2.imshow(window_name, image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # pdb.set_trace()
    return np.concat([corner.squeeze() for corner in sorted_corners],axis=0)
    



file1 = "./phone_stream/01830.jpg"
file2 = "./ipad_stream/00744.jpg"

image = cv2.imread(file1)
print(image.shape)

corners1 = detect_aruco(file1)
corners2 = detect_aruco(file2)

print(corners1)
print(corners2)

# plot_pts_correspondence(corners1, corners2)

# print(dir(cv2))
F,mask = cv2.findFundamentalMat(corners1, corners2, cv2.FM_8POINT)
print(F)
print(mask)


from scipy.linalg import null_space

left_epipole = null_space(F.T)

# output = cv2.stereoRectifyUncalibrated(corners1, corners2, F, image.shape[:2])
# print(output)
# retval, H1, H2 = output
# print(H1)
# print(H2)

print(left_epipole)
print(np.dot(left_epipole.T, F))

ex = left_epipole[0,0]
ey = left_epipole[1,0]
ez = left_epipole[2,0]

e_cross = np.array([[0,-ez,ey],[ez,0,-ex],[-ey, ex, 0]])

print('e_cross')
print(e_cross)

factor_one = np.matmul(e_cross, F)
print('factor one', factor_one)

# pdb.set_trace()
camera_matrix2 = np.hstack((factor_one, left_epipole))
print('camera 2', camera_matrix2)
camera_matrix1 = np.hstack((np.eye(3), np.zeros((3,1))))
print('camera 1', camera_matrix1)

# print(corners)
print(corners1.T.shape, corners2.T.shape)
points_4d = cv2.triangulatePoints(camera_matrix1, camera_matrix2, corners1.T, corners2.T)
print(points_4d.T)
print(points_4d.T.shape)

points_4d_flipped = cv2.triangulatePoints(camera_matrix1, camera_matrix2, corners2.T, corners1.T)
print(points_4d_flipped)
print(points_4d_flipped.shape)

points_3d = points_4d[:3] / points_4d[3]

points_3d_flipped = points_4d_flipped[:3] / points_4d_flipped[3]

print(points_3d.T, points_3d.T.shape)

# points4d = cv2.triangulatePoints(H1, H2, corners1.transpose(), corners2.transpose())
'''
open both files
get 12 aruco tag corners

calculate the fundamental matrix

call the 3 functions from chat gpt
'''
# https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html

# https://cmsc426.github.io/sfm/

# multiple view geometry textbook


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

make_3d_plot(points_3d)