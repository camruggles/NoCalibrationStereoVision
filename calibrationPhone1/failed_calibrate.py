import cv2
import cv2.aruco as aruco
import numpy as np

# Define the ArUco dictionary and create the ChArUco board.
# Adjust squaresX, squaresY, squareLength, and markerLength as needed.


aruco_dict = cv2.aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
# parameters =  cv2.aruco.DetectorParameters()
# detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
print(dir(cv2.aruco.CharucoBoard))
# import pdb; pdb.set_trace()
# aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)
board = aruco.CharucoBoard((11,8),squareLength=0.015,  markerLength=0.011, dictionary=aruco_dict)
charucodetector = cv2.aruco.CharucoDetector(board)

# List the image file paths (adjust these paths to your images)
# image_files = ['image1.jpg', 'image2.jpg', 'image3.jpg', 'image4.jpg']
image_files = ['20250320_113104.jpg',  '20250320_113106.jpg',  '20250320_113108.jpg',  '20250320_113110.jpg']

# Containers to hold all detected ChArUco corners and their corresponding IDs
all_corners = []
all_ids = []
image_size = None

# Process each image
for img_file in image_files:
    img = cv2.imread(img_file)
    if img is None:
        print(f"Could not load image {img_file}")
        continue

    # Convert to grayscale
    
    charuco_corners, charuco_ids, marker_corners, marker_ids = charucodetector.detectBoard(img)
    print(charuco_corners, charuco_ids, marker_corners, marker_ids)
    # quit()
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    image_size = img.shape[::-1]  # (width, height)
    print(image_size)

    # # Detect ArUco markers in the image
    # marker_corners, marker_ids, _ = aruco.detectMarkers(gray, aruco_dict)
    # if marker_ids is not None and len(marker_ids) > 0:
    #     # Interpolate ChArUco corners using the detected markers
    #     print(marker_corners, marker_ids)
    #     retval, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(marker_corners,
    #                                                                           marker_ids,
    #                                                                           img, board)
    #     # Only use images with a sufficient number of detected corners
    #     print(retval, charuco_corners, charuco_ids)
    if charuco_ids is not None and charuco_corners is not None and len(charuco_ids) > 3:
        all_corners.append(charuco_corners)
        all_ids.append(charuco_ids)
    else:
        print(f"Not enough ChArUco corners detected in {img_file}")

# Make sure we have valid detections from enough images before calibration
if len(all_corners) > 0:
    # Calibrate the camera using the ChArUco detections.
    # This function returns the reprojection error, camera matrix, distortion coefficients,
    # rotation vectors, and translation vectors.
    ret, cameraMatrix, distCoeffs, rvecs, tvecs = aruco.calibrateCameraCharuco(
        charucoCorners=all_corners,
        charucoIds=all_ids,
        board=board,
        imageSize=image_size,
        cameraMatrix=None,
        distCoeffs=None)
    
    print("Reprojection error:", ret)
    print("Camera matrix:\n", cameraMatrix)
    print("Distortion coefficients:\n", distCoeffs)
else:
    print("Not enough detections for calibration.")
