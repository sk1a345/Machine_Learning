import cv2
import numpy as np

# Print OpenCV version
print("OpenCV version:", cv2.__version__)

# Create a simple black image
image = cv2.imread("test.jpeg")  # You can use any image path

if image is None:
    print("Image not found. Showing a black image instead.")
    image = 255 * np.ones((300, 300, 3), dtype=np.uint8)  # White image

cv2.imshow("OpenCV Test Window", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
