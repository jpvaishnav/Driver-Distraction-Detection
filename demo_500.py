from __future__ import division
import argparse
import glob
#import tensorflow as tf
import cv2
import numpy as np
#import imutils
import time
import cv2
import os
#import dlib
import time
#import threading
import math
#from util import *
import os.path as osp
import pandas as pd
import random 
import pickle as pkl
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

import cnn1

#import forest
def label(x):
    if x==0:
        return "Normal Driving"
    if x==1:
        return "texting - right"
    if x==2:
        return "talking on phone - right"
    if x==3:
        return "texting - left"
    if x==4:
        return "talking on phone - left"
    if x==5:
        return "operating the radio"
    if x==6:
        return "drinking"
    if x==7:
        return "reaching behind"
    if x==8:
        return "hair and makeup"
    if x==9:
        return "talking to passenger"

f=open('lb.txt','r')
Lines = f.readlines()
Y_data=[]
for l in Lines:
	x=l.split(",")
	Y_data.append(str(x[1])[1])
Y_data=Y_data[0:530]
print("Y_data is : ")
print(Y_data)
print("Size of Y_data: **********************")
print(len(Y_data))


imgs = glob.glob(r"F:\Trimester2\BTP\BTP\test 500\*.jpg")
# pathIn='F:/Trimester2/BTP/BTP/test 500/'
# files = [f for f in os.listdir(pathIn)]
temp=cv2.imread(imgs[0])
frame_size = temp.shape[:2][::-1]
# files = [f for f in os.listdir(pathIn)]
#files.sort(key= lambda x: int(x.split('.')[0].split('_')[1]))
X_data=[]
for ele in imgs:
	img0=cv2.imread(ele)
	frame = cv2.imread(ele)
	fr = cv2.resize(frame, (224,224))
	X_data.append(fr)
X_data=np.array(X_data)
print("X_data is : ")
print(X_data)
#fr=np.array(fr)
#frame_vector=fr.reshape(1,64*64)
#print(frame_vector)
#set the frame_size equal to the required image size in cnn file
y=cnn1.Predict(X_data)
print("Prediction of 500 images is: ")
print(y)
print("Size of Y: **********************")
print(len(y))

output_dir='./data/output/'
print(confusion_matrix(Y_data, y))
target_names = ['c0', 'c1','c2','c3','c4','c5','c6','c7','c8','c9']
print(classification_report(Y_data, y, target_names=target_names))
i=1
for ele in imgs:
	if i==530:
		break
	print(str(y[i]))
	lb=int(str(y[i])[0])
	print(lb)
	msg=label(lb)
	img0=cv2.imread(ele)
	frame = cv2.imread(ele)
	window_width = int(frame.shape[1])
	window_height = int(frame.shape[0])

	#cv2.WINDOW_NORMAL makes the output window resizealbe
	cv2.namedWindow('Distraction', cv2.WINDOW_NORMAL)

	#resize the window according to the screen resolution
	cv2.resizeWindow('Distraction', window_width, window_height)

	#cv2.imshow('Distraction', img0)
	cv2.putText(img0, msg, (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.15, (0, 255, 0), 2)
	cv2.imshow('Distraction',img0)
	cv2.imwrite(os.path.join(output_dir , 'out_{}.png'.format(i)), img0)
	i+=1
cv2.destroyAllWindows()
