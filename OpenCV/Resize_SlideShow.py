import cv2 
import numpy as np
import os

# Slide show

list_name = os.listdir(r"C:\Users\HP\OneDrive\Pictures\New folder")
print(list_name)

for name in list_name:
    path = "C:\\Users\\HP\\OneDrive\\Pictures\\New folder"
    img_name = path+"\\"+name
    img =cv2.imread(img_name)
    img_resize = cv2.resize(img,(300,300))
    cv2.imshow("images_slide_show",img_resize)
    cv2.waitKey(4000)
cv2.destroyAllWindows()
