import cv2

img = cv2.imread('OpenCV/test.jpeg')  #read image:
cv2.imshow("Kanha",img) #image show:
# cv2.imshow("Kanha1",img)
cv2.waitKey(10000) #image frame wait time in millisecond and if 0 is passed=> keyboard any key press and window will be closed and if 10000 if passed =>exactly after 10 senonds the window will be closed automatically
cv2.destroyAllWindows() #all windows close:
# cv2.destroyWindow("Kanha1") #only one window close:


