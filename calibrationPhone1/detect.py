
import cv2
import cv2.aruco as aruco
import numpy as np

def detect_aruco(image_path):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image not found!")
        return
    
    # Convert image to grayscale as required for detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Define the ArUco dictionary we are using (e.g., 6x6 with 250 possible ids)
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters =  cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
        
    # Initialize detector parameters using default values
    
    # Detect the markers in the image
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
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
    cv2.imshow('Detected ArUco Markers', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # Replace 'image.jpg' with your input image path containing ArUco markers
    detect_aruco('20250320_113104.jpg')
