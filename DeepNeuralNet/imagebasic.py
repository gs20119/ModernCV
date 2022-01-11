
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('C:/Users/Epsilon/Desktop/images/sample.jpeg')
img = img[50:250,40:240,:] # crop image
print(img.shape) # pixel x pixel x BGR


img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_gray_s = cv2.resize(img_gray,(25,25))


plt.imshow(img_gray, cmap='gray')
plt.show()

plt.imshow(img_gray_s, cmap='gray')
plt.show()

plt.imshow(img_rgb)
plt.show()





