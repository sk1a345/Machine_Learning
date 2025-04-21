import cv2
import numpy as np
img = cv2.imread(r"C:\Users\HP\OneDrive\python_Pandas\OpenCV\nature.png")

re_img = cv2.resize(img,(500,700)) #resizing the image:
# cv2.imshow("Nature",re_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Multiple images:
v = np.array([[1,2,1,2],[1,2,1,2]])

img = cv2.imread(r"C:\Users\HP\OneDrive\python_Pandas\OpenCV\nature.png")
# print(img) #RGB 


# horizontally joining two images
img1 = cv2.imread(r"C:\Users\HP\OneDrive\python_Pandas\OpenCV\nature.png")
re_img1 = cv2.resize(img1,(300,300))
h = np.hstack((re_img1,re_img1))
cv2.imshow("Horizontallyjoint",h)
cv2.waitKey(0)
cv2.destroyAllWindows()


# Vertically joining two images:
img2 = cv2.imread(r"C:\Users\HP\OneDrive\python_Pandas\OpenCV\nature.png")
re_img2 = cv2.resize(img2,(300,300))
v = np.vstack((re_img2,re_img2))
cv2.imshow("VerticallyJoint",v)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Joining vertically and horizontally:
img3 = cv2.imread(r"C:\Users\HP\OneDrive\python_Pandas\OpenCV\nature.png")
re_img3 = cv2.resize(img3,(300,300))
h = np.hstack((re_img3,re_img3,re_img3))
v = np.vstack((h,h))
cv2.imshow("Hozi-Verti-Display",v)
cv2.waitKey(0)
cv2.destroyAllWindows()