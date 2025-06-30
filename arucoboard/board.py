import cv2
import cv2.aruco as aruco
import numpy as np
from PIL import Image

# -------------------------------
# Create an ArUco Board
# -------------------------------
# Define the ArUco dictionary to use
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)

# Create a GridBoard: adjust the number of markers, marker size, and spacing as needed.
# For example, here we create a board with 5 markers in width and 7 markers in height.
markersX = 5
markersY = 7
markerLength = 60    # marker side length in pixels (for drawing)
markerSeparation = 15  # separation between markers in pixels
margins = markerSeparation
borderBits = 1

board = aruco.GridBoard((markersX, markersY), markerLength, markerSeparation, aruco_dict)

imageSize = (800,1100)
# -------------------------------
# Draw the Board to an Image
# -------------------------------
# Determine the size of the output image. Adjust as necessary.
img_width = 600
img_height = 800
print(dir(board))
board_image = cv2.aruco.drawPlanarBoard(board, imageSize, margins, borderBits)
# board_image = board.draw((img_width, img_height))

# Save the board as an image (PNG format)
cv2.imwrite("aruco_board.png", board_image)
print("Aruco board image saved as 'aruco_board.png'.")

# -------------------------------
# Save the Board as a PDF
# -------------------------------
# Use Pillow to convert the image to PDF.
# Note: The image is initially in a numpy array (BGR), but Pillow expects RGB.
board_image_rgb = cv2.cvtColor(board_image, cv2.COLOR_BGR2RGB)
image_pil = Image.fromarray(board_image_rgb)
image_pil.save("aruco_board.pdf", "PDF", resolution=100.0)
print("Aruco board PDF saved as 'aruco_board.pdf'.")
