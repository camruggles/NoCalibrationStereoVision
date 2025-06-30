import cv2
import numpy as np
import glob
import random
from tqdm import tqdm

def get_world_corners():
    a = (0,0)
    b = (0.015, 0.0)
    c = (0.015, 0.015)
    d = (0.0, 0.015)
    arr = [a,b,c,d]
    id_dict =  {}
    id_dict[0] = arr
    count = 0
    for y in range(8):
        for x in range(11):
            if count % 2 == 0:
                id = count //2
                [a,b,c,d] = id_dict[0]
                a1,a2 = a
                b1,b2 = b 
                c1, c2 = c
                d1, d2 = d
                id_dict[id] = [[a1 + x * 0.02, a2 + y * 0.02], [b1 + x * 0.02, b2 + y * 0.02], [c1 + x * 0.02, c2 + y * 0.02], [d1 + x * 0.02, d2 + y * 0.02]]
                # print(id, id_dict[id])
            count += 1
    return id_dict

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
parameters =  cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
# Prepare object points for the Charuco board
obj_points = []
img_points = []

# List of image paths (adjust this with the paths to your images)
image_paths = [f"../calibration_and_squat/{i:05d}.jpg" for i in range(700,1501)]
# print(image_paths)
# quit()
corner_id_dict = get_world_corners()
for image_path in tqdm(image_paths):
    # Read the image
    # print(image_path)
    if random.random() < 0.4:
        continue
    img = cv2.imread(image_path)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect ArUco markers in the image
    corners, ids, rejected_img_points = cv2.aruco.detectMarkers(gray, aruco_dict)
    if not isinstance(ids, type(None)):
        # print(len(corners), len(ids))
        sorted_indices = np.argsort(ids.flatten())  # Get sorted indices based on ids
        sorted_corners = [corners[i] for i in sorted_indices]
        sorted_ids = ids[sorted_indices]
        # print(sorted_corners, sorted_ids)
        # import pdb; pdb.set_trace()
        for i,tag in enumerate(sorted_corners):
            arr_corners = tag[0].tolist()
            for corner in arr_corners:
                img_points.append( np.array(corner) )
            world_corners = corner_id_dict[i]
            for corner in world_corners:
                x,y = corner
                obj_points.append(  np.array([x,y,0])   )

    # quit()
    # Refine the marker detection
    # if len(sorted_corners) > 0:
    #     # Interpolate the Charuco corners based on ArUco markers
    #     # ret, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, charuco_board)

    #     if ret >= 4:  # Need at least 4 corners for calibration
    #         img_points.append(charuco_corners)
    #         obj_points.append(np.zeros_like(charuco_corners, dtype=np.float32))  # Placeholder object points (we'll use a dummy)

# Perform camera calibration
if len(obj_points) > 0:
    # Camera calibration using the detected Charuco corners
    # import pdb; pdb.set_trace()
    np_obj_points = np.array(obj_points, dtype=np.float32)
    np_img_points = np.array(img_points, dtype=np.float32)
    # print(np_obj_points.shape)
    # print(np_img_points.shape)
    object_points = [np_obj_points]
    image_points = [np_img_points]
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, gray.shape[::-1], None, None)

    print("Camera matrix:", mtx)
    print("Distortion coefficients:", dist)
    print("Rotation vectors:", rvecs)
    print("Translation vectors:", tvecs)
else:
    print("No valid corners detected for calibration.")

# Optionally, visualize the detected markers (for debugging)
# for image_path in image_paths:
#     img = cv2.imread(image_path)
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     corners, ids, _ = cv2.aruco.detectMarkers(gray, cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250))
    
#     if len(corners) > 0:
#         img = cv2.aruco.drawDetectedMarkers(img, corners, ids)
#     cv2.imshow("Detected Markers", img)
#     cv2.waitKey(0)

# cv2.destroyAllWindows()

