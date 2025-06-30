import cv2
import numpy as np

filename1 = './phone/frames/00000.jpg'
filename2 = 'phone_aruco.jpg'

filename1 = './ipad/frames/00000.jpg'
filename2 = 'ipad_aruco.jpg'

# Load the image
image = cv2.imread(filename1)  # Provide the path to your image

# Load the pre-built ArUco dictionary and the corresponding parameters for detection
# aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)  # You can change the dictionary type
# parameters = cv2.aruco.DetectorParameters_create()

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_7X7_100)
parameters =  cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

# Detect the markers in the image
corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(image, aruco_dict, parameters=parameters)


camera_matrix = np.array([[1000, 0, image.shape[1] // 2],
                          [0, 1000, image.shape[0] // 2],
                          [0, 0, 1]], dtype=np.float32)
dist_coeffs = np.zeros((4, 1), dtype=np.float32)  # Assuming no distortion
# If markers are detected, overlay their positions and IDs
if len(corners) > 0:
    # Draw detected markers
    image = cv2.aruco.drawDetectedMarkers(image, corners, ids)
    
    # Iterate through each marker and add overlays (like the marker ID)
    for i in range(len(ids)):
        # Extract the corners of the marker
        # import pdb
        # pdb.set_trace()
        marker_corners = corners[i]
        print(marker_corners.shape)

        ####### orientation and position #######
        # Estimate the pose of the marker
        retval, rvec, tvec = cv2.aruco.estimatePoseSingleMarkers(marker_corners, 0.1, camera_matrix, dist_coeffs)

        # rvec (rotation vector) gives the orientation of the marker
        # tvec (translation vector) gives the position of the marker
        print(f"Marker ID: {ids[i][0]}")
        print(f"Rotation Vector (rvec): {rvec[0]}")
        print(f"Translation Vector (tvec): {tvec[0]}")

        # Convert rotation vector to rotation matrix
        rotation_matrix, _ = cv2.Rodrigues(rvec[0])

        print(f"Rotation Matrix:\n{rotation_matrix}")
        
        # Optionally, convert the rotation matrix to Euler angles
        # Euler angles (roll, pitch, yaw) from rotation matrix
        sy = np.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2)
        singular = sy < 1e-6

        if not singular:
            x = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])  # roll
            y = np.arctan2(-rotation_matrix[2, 0], sy)                     # pitch
            z = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])   # yaw
        else:
            x = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])  # roll
            y = np.arctan2(-rotation_matrix[2, 0], sy)                     # pitch
            z = 0

        # Print Euler angles (in radians)
        print(f"Euler Angles (roll, pitch, yaw) in radians: ({x}, {y}, {z})")

        # Optionally, you can convert the angles from radians to degrees:
        print(f"Euler Angles (roll, pitch, yaw) in degrees: ({np.degrees(x)}, {np.degrees(y)}, {np.degrees(z)})")

        # Visualize the marker pose by drawing the axis on the image (optional)
        # cv2.aruco.drawAxis(image, camera_matrix, dist_coeffs, rvec, tvec, 0.1)  # Draw a 3D axis on the marker


        ##### normal vector ######
        marker_corners = corners[i][0]

        # Get two vectors that lie on the plane of the ArUco marker
        vector1 = marker_corners[1] - marker_corners[0]  # Vector from corner 0 to corner 1
        vector2 = marker_corners[2] - marker_corners[0]  # Vector from corner 0 to corner 2

        # 3d needed for 3d normal vector
        vector1 = np.append(vector1,0.0)
        vector2 = np.append(vector2,0.0)

        # Calculate the normal vector using the cross product of these two vectors
        normal_vector = np.cross(vector1, vector2)

        # Normalize the normal vector (optional but typically useful for visualization)
        normal_vector = normal_vector / np.linalg.norm(normal_vector)

        # Print the normal vector
        print(f"Marker ID: {ids[i][0]} - Normal Vector: {normal_vector}")


        ###### CENTER of the marker ##########
        # Get the center of the marker
        center = np.mean(corners[i][0], axis=0).astype(int)
        
        # Draw the marker ID near the center
        cv2.putText(image, str(ids[i][0]), (center[0] - 10, center[1] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
        
        # You can add other information (like the corner positions, etc.)
        for corner in corners[i][0]:
            corner = tuple(corner.astype(int))
            cv2.circle(image, corner, 5, (0, 0, 255), -1)  # Draw circles on the corners
        print()
# Display the resulting image
# cv2.imshow('Image with ArUco Markers', image)

# Wait for a key press and close the display window
# cv2.waitKey(0)
cv2.destroyAllWindows()

# Optionally save the image with overlays
cv2.imwrite(filename2, image)
