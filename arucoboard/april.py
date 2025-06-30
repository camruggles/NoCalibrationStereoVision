import cv2
import numpy as np
import apriltag

import pdb
# -------------------------------
# Board configuration (in meters)
# -------------------------------
tag_size = 0.068       # Tag side length in meters.
tag_spacing = 0.02    # Gap between adjacent tags (in meters).
board_rows = 2        # Number of tag rows on the board.
board_cols = 3        # Number of tag columns on the board.

# Create a dictionary mapping tag IDs to their 3D corner coordinates.
# We assume tags are arranged in a grid, numbered sequentially row-wise.
# Each tag has four corners (top-left, top-right, bottom-right, bottom-left) in world coordinates (z=0).
board_dict = {}
for i in range(board_rows):
    for j in range(board_cols):
        tag_id = i * board_cols + j
        x0 = j * (tag_size + tag_spacing)
        y0 = i * (tag_size + tag_spacing)
        board_dict[tag_id] = np.array([
            [x0,           y0,            0],
            [x0 + tag_size, y0,            0],
            [x0 + tag_size, y0 + tag_size, 0],
            [x0,           y0 + tag_size, 0]
        ], dtype=np.float32)

# -------------------------------
# Initialize AprilTag detector
# -------------------------------
# Use the apriltag library. Adjust the 'families' parameter if needed.
options = apriltag.DetectorOptions(families="tag36h11")
detector = apriltag.Detector(options)

pdb.set_trace()
# -------------------------------
# Process images and collect correspondences
# -------------------------------
# List your image filenames (adjust paths as needed)
image_files = ["20250321_134141.jpg"]

all_obj_points = []  # List of 3D object points per image.
all_img_points = []  # List of corresponding 2D image points per image.
image_size = None

for fname in image_files:
    img = cv2.imread(fname)
    if img is None:
        print(f"Could not load image: {fname}")
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if image_size is None:
        image_size = gray.shape[::-1]  # (width, height)

    # Detect AprilTags in the image
    detections = detector.detect(gray)
    
    obj_points = []  # 3D points for this image.
    img_points = []  # 2D points for this image.
    
    for detection in detections:
        tag_id = detection.tag_id
        if tag_id in board_dict:
            # Get the board's 3D corner positions for this tag.
            board_corners = board_dict[tag_id]
            # The detected corners (order: typically top-left, top-right, bottom-right, bottom-left)
            detected_corners = detection.corners.astype(np.float32)
            # Append these four correspondences.
            obj_points.extend(board_corners)
            img_points.extend(detected_corners)
        else:
            print(f"Detected tag id {tag_id} is not in board configuration.")

    if len(obj_points) > 0 and len(img_points) > 0:
        all_obj_points.append(np.array(obj_points, dtype=np.float32))
        all_img_points.append(np.array(img_points, dtype=np.float32))
    else:
        print(f"No valid tag detections in image {fname}")

if len(all_obj_points) == 0 or len(all_img_points) == 0:
    raise RuntimeError("No valid tag detections were found in any image. Cannot calibrate.")

# -------------------------------
# Calibrate the camera
# -------------------------------
# The calibrateCamera function computes the camera matrix, distortion coefficients, 
# rotation vectors, and translation vectors.
ret, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(
    objectPoints=all_obj_points,
    imagePoints=all_img_points,
    imageSize=image_size,
    cameraMatrix=None,
    distCoeffs=None)

print("Reprojection error:", ret)
print("Camera Matrix:\n", cameraMatrix)
print("Distortion Coefficients:\n", distCoeffs)
