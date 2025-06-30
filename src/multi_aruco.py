import cv2
import numpy as np
import pdb

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def calibrate_camera_from_images(image1, image2, aruco_dict, aruco_params):
    # Load the ArUco dictionary
    # aruco_dict = cv2.aruco.Dictionary_get(aruco_dict)
    # aruco_params = cv2.aruco.DetectorParameters_create()

    # Prepare 3D points in the real world coordinates (for each ArUco marker)
    # Here we define two markers
    real_world_points = np.array([
        [0, 0, 0],        # Marker 1 (position of first marker in world coordinates)
        [0.1, 1.0, 0]       # Marker 2 (position of second marker in world coordinates)
    ], dtype=np.float32)

    # Prepare lists to store detected points and their corresponding 3D world points
    obj_points = []  # 3D world coordinates
    img_points = []  # 2D image coordinates

    # Detect ArUco markers in both images
    for img in [image1, image2]:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detect ArUco markers and their ids in the image
        corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)

        print(corners, ids)


        if ids is not None and len(ids) >= 2:  # Expecting exactly two markers
            
            # pdb.set_trace()
            # Sort the detected markers by their ids in ascending order
            sorted_indices = np.argsort(ids.flatten())  # Get sorted indices based on ids
            sorted_corners = [corners[i] for i in sorted_indices]
            sorted_ids = ids[sorted_indices]
            print(sorted_corners, sorted_ids)
            # quit()
            # Add the sorted markers to the object and image points lists
            for i in range(2): #len(sorted_ids)):
                print(sorted_corners[i].shape)
                top_left_corner = sorted_corners[i][0, 0].reshape(-1, 2)
                print('corner', top_left_corner)
                img_points.append(list(top_left_corner))
                obj_points.append(list(real_world_points[i]))
        else:
            print("Error: Exactly two markers must be detected in each image.")
            return None, None, None, None

    # Calibrate the cameras based on the detected points (obj_points, img_points)
    # quit()
    pdb.set_trace()
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, gray.shape[::-1], None, None
    )
    
    pdb.set_trace()
    return camera_matrix, dist_coeffs


def plot_3points(p1,p2):
    # Stack the points into a single numpy array for easy manipulation
    p3 = np.array([0,0,0])
    points = np.array([p1, p2, p3])

    # Unzip the points into separate lists for X, Y, and Z coordinates
    x, y, z = points[:, 0], points[:, 1], points[:, 2]

    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot the points
    ax.scatter(x, y, z, color='r', s=100)

    # Label the axes
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Set the title
    ax.set_title('3D Plot of Three Points')

    # Enable interactive rotation
    plt.show()

filename1 = './phone/frames/00000.jpg'

filename2 = './ipad/frames/00000.jpg'
# Load images from both cameras
image1 = cv2.imread(filename1)  # First image from camera 1
image2 = cv2.imread(filename2)  # Second image from camera 2

# Load the pre-built ArUco dictionary
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_7X7_100)
parameters =  cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

calibrate_camera_from_images(image1, image2, aruco_dict, parameters)

# Assume we have the camera calibration matrices for both cameras
camera_matrix1 = np.array([[1000, 0, image1.shape[1] // 2],
                           [0, 1000, image1.shape[0] // 2],
                           [0, 0, 1]], dtype=np.float32)  # Example matrix for camera 1

camera_matrix2 = np.array([[1000, 0, image2.shape[1] // 2],
                           [0, 1000, image2.shape[0] // 2],
                           [0, 0, 1]], dtype=np.float32)  # Example matrix for camera 2

dist_coeffs1 = np.zeros((4, 1), dtype=np.float32)  # Assuming no distortion for camera 1
dist_coeffs2 = np.zeros((4, 1), dtype=np.float32)  # Assuming no distortion for camera 2

# Detect markers in both images
corners1, ids1, _ = cv2.aruco.detectMarkers(image1, aruco_dict, parameters=parameters)
corners2, ids2, _ = cv2.aruco.detectMarkers(image2, aruco_dict, parameters=parameters)

# Marker size in meters (known size)
marker_size = 0.1  # 10 cm (for example)

# If markers are detected in both images
if len(corners1) > 0 and len(corners2) > 0:
    # Get the first ArUco marker position in both images (Assuming the first marker is the reference)
    print('ids', ids1, ids2)
    # quit()
    marker_corners1 = corners1[2]  # Corners of the first detected marker in image1
    marker_corners2 = corners2[1]  # Corners of the first detected marker in image2
    print(marker_corners1.shape)

    # Estimate the pose of the first marker in both images
    retval1, rvec1, tvec1 = cv2.aruco.estimatePoseSingleMarkers(marker_corners1, marker_size, camera_matrix1, dist_coeffs1)
    retval2, rvec2, tvec2 = cv2.aruco.estimatePoseSingleMarkers(marker_corners2, marker_size, camera_matrix2, dist_coeffs2)

    # The translation vectors tvec1 and tvec2 give us the position of the first marker in 3D space relative to each camera
    print("Camera 1 position (tvec1):", tvec1)
    print("Camera 2 position (tvec2):", tvec2)

    # The rotation vectors (rvec1, rvec2) describe the orientation of the markers in the camera frame
    print("Camera 1 rotation (rvec1):", rvec1)
    print("Camera 2 rotation (rvec2):", rvec2)

    # To get the origin of each camera relative to the first ArUco marker (in the world frame),
    # we transform the positions using the inverse of the marker's rotation and translation.
    # The position of the camera in the world frame (relative to the first ArUco tag) is the negative of the translation vector
    # after applying the marker's rotation.

    # Inverse transformation of the ArUco marker's position (camera position in world frame)
    R1, _ = cv2.Rodrigues(rvec1[0])  # Convert rotation vector to rotation matrix for camera 1
    R2, _ = cv2.Rodrigues(rvec2[0])  # Convert rotation vector to rotation matrix for camera 2

    print(R1.shape)
    print(tvec1.shape)

    # Compute the camera positions relative to the first marker
    tvec1 = np.moveaxis(tvec1, [1,2], [2,1])
    print(tvec1.shape)
    tvec2 = np.moveaxis(tvec2, [1,2], [2,1])
    camera1_world_pos = -np.dot(R1.T, tvec1[0])  # Inverse of the rotation matrix times the negative translation
    camera2_world_pos = -np.dot(R2.T, tvec2[0])  # Similarly for camera 2

    print("Camera 1 world position relative to ArUco tag:", camera1_world_pos)
    print("Camera 2 world position relative to ArUco tag:", camera2_world_pos)

    
    pdb.set_trace()
    plot_3points(camera1_world_pos.squeeze(), camera2_world_pos.squeeze())

    # Optionally, visualize the pose of the markers in both images by drawing the axis
    # cv2.aruco.drawAxis(image1, camera_matrix1, dist_coeffs1, rvec1, tvec1, marker_size)
    # cv2.aruco.drawAxis(image2, camera_matrix2, dist_coeffs2, rvec2, tvec2, marker_size)

    # Show the images with the drawn axis
    cv2.imshow('Image 1 with Axis', image1)
    cv2.imshow('Image 2 with Axis', image2)
    # cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("ArUco markers not detected in one or both images.")
