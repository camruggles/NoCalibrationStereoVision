import cv2
import cv2.aruco as aruco
import numpy as np

def create_and_draw_charuco_board():
    # Define board parameters
    squares_x = 8  # Number of squares in X-direction
    squares_y = 11  # Number of squares in Y-direction
    square_length = 0.015  # Length of a chessboard square in pixels (world unit)
    marker_length = 0.011   # Length of the ArUco marker side in pixels (world unit)

    # Create the dictionary and the Charuco board object
    aruco_dict = cv2.aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    board = aruco.CharucoBoard((11,8),squareLength=0.015,  markerLength=0.011, dictionary=aruco_dict)

    # Draw the board image.
    # We choose an image size that fits the board.
    board_img = board.draw((squares_x * square_length, squares_y * square_length))
    board_img = cv2.aruco.drawPlanarBoard(board, imageSize, margins, borderBits)
    return board, board_img, squares_x, squares_y, square_length

def compute_charuco_world_coordinates(squares_x, squares_y, square_length):
    """
    Compute the object/world coordinates (in the board frame) of the internal chessboard corners.
    For a board with squares_x and squares_y squares, there are (squares_x-1) x (squares_y-1) internal corners.
    We assume the board lies on the Z=0 plane.
    """
    obj_points = []
    for i in range(squares_y - 1):
        for j in range(squares_x - 1):
            # Each corner in world coordinates (x, y, 0)
            pt = (j * square_length, i * square_length, 0)
            obj_points.append(pt)
    return obj_points

def overlay_world_coordinates(image, obj_points):
    """
    Overlay the computed world coordinates as text on the image.
    We'll mark each computed corner with a circle and its coordinate.
    """
    annotated = image.copy()
    for pt in obj_points:
        # We use the (x,y) values for drawing on the image.
        x, y, _ = pt
        # Draw a small circle at the corner location
        cv2.circle(annotated, (int(x), int(y)), 5, (0, 0, 255), -1)
        # Overlay text with the coordinate values
        text = f"({int(x)},{int(y)})"
        cv2.putText(annotated, text, (int(x) + 5, int(y) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    return annotated

def main():
    board, board_img, squares_x, squares_y, square_length = create_and_draw_charuco_board()

    # Compute the world coordinates of the internal chessboard corners.
    obj_points = compute_charuco_world_coordinates(squares_x, squares_y, square_length)

    # Overlay the computed coordinates on the drawn board image.
    annotated_img = overlay_world_coordinates(board_img, obj_points)

    # Display the result
    cv2.imshow('Charuco Board with World Coordinates', annotated_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
