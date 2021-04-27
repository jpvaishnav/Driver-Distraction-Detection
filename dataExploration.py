import cv2 
import numpy as np 
import glob

row=64
col=64

Y=['c0']
a = row*col
DDEPTH = cv2.CV_16S
vector_newX = np.zeros((a, 1))
vector_newY = []

img_path="phone.jpg"
oriimg = cv2.imread(img_path,0)
img0 = cv2.resize(oriimg,(row,col))
cv2.imwrite('resized.jpg',img0)
cv2.imshow('Resized',img0)
cv2.waitKey(0)
img = cv2.GaussianBlur(img0, (3, 3), 2)
cv2.imwrite('blur.jpg',img)
cv2.imshow("Gaussian Smoothing",np.hstack((img0, img)))
cv2.waitKey(0) # waits until a key is pressed
cv2.imshow('GaussianBlur',img)
cv2.waitKey(0)


gradx = cv2.Sobel(img, DDEPTH , 1, 0, ksize=3, scale=1, delta=0)
gradx = cv2.convertScaleAbs(gradx)
cv2.imshow('Gradx',gradx)
cv2.waitKey(0)


grady = cv2.Sobel(img, DDEPTH , 0, 1, ksize=3, scale=1, delta=0)
grady = cv2.convertScaleAbs(grady)
cv2.imshow('GradY',grady)
cv2.waitKey(0)

        
grad = cv2.addWeighted(gradx, 0.5, grady, 0.5, 0)
cv2.imshow('Blending',grad)
cv2.imwrite('Grad.jpg',grad)
cv2.waitKey(0)

print("Shape of blended grad is " + str(np.shape(grad)))
flat = grad.reshape(a,1)
print("size of reshaped 1d vector is: "+str(np.shape(flat)))
print(flat)
#for row in flat:
#	print(row)

vector_newX = np.c_[vector_newX,flat]
print("vector_newX is: ")
print(vector_newX)
vector_newY = np.append(vector_newY,Y)
print("vector_newY is: ")
print(vector_newY)
print(img_path)

cv2.destroyAllWindows()