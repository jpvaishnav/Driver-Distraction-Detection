import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split
import cv2
import glob
from sklearn import tree
import time
import matplotlib
import pydot
from io import StringIO
##############GENERALIZE CHECKING DATA GENERATE#################################

row = 64
col = 64
a = row*col

oimg1 = cv2.imread(r"F:\Trimester2\BTP\BTP\phone.jpg",0)
img1 = cv2.resize(oimg1,(row,col))

vector_newX = np.reshape(img1,(row*col,1))

imgs = glob.glob(r"F:\Trimester2\BTP\BTP\test 500\*.jpg")

for img in imgs:
    oriimg = cv2.imread(img,0)
    img0 = cv2.resize(oriimg,(row,col))
    flat = img0.reshape(a,1)
    vector_newX = np.c_[vector_newX,flat]
    print(img)

vector_newX = vector_newX.T

finalX_gen = vector_newX[1:,:]

###################IMPORT TRAINING DATA##########################################
data = pd.read_csv(r'F:\Trimester2\BTP\BTP\Sobel_train.csv'  , header = None)
#print(data)

Xo = data.drop(data.columns[-1], axis=1)
print("Xo is:")
print(Xo)

Yo = data[data.columns[-1]]
print("Yo is: ")
print(Yo)

X_train,X_test,y_train,y_test=train_test_split(Xo , Yo , test_size = 0.15, random_state = 100)
#print(X_train)
#print(y_train)
classifier = DecisionTreeClassifier()
classifier=classifier.fit(X_train,y_train)
print("Drcesion Tree:")

#dot_data = StringIO() 
#tree.export_graphviz(classifier, out_file=dot_data) 
#graph = pydot.graph_from_dot_data(dot_data.getvalue())
#graph[0].write_pdf("DecisionTree.pdf") 

r=tree.export_text(classifier)
print(r)

print("Decision Tree ended: ")
start_time = time.time()
y_pred = classifier.predict(X_test)
print("--- %s seconds ---" % (time.time() - start_time))
print('Toal Images predicted: ')
print(len(y_pred))
print(y_pred)
#y_generalize = classifier.predict(finalX_gen)

#print(len(y_generalize))
#print(y_generalize)

print("Accuracy:",metrics.accuracy_score(y_test,y_pred))

################################################################################



'''oimg1 = cv2.imread(r"D:\Kaggle_Distracted_Driver\imgs\test\img_7.jpg")
img1 = cv2.resize(oimg1,(64,64))

vector_newX1 = np.reshape(img1,(64*64*3,1))
vector_newX1 = vector_newX1.T

y_new1 = classifier.predict(vector_newX1)
print("The actual class is c0, predicted class for first real world image is: ",y_new1)

oimg2 = cv2.imread(r"D:\Kaggle_Distracted_Driver\imgs\test\img_8.jpg")
img2 = cv2.resize(oimg2,(64,64))

vector_newX2 = np.reshape(img2,(64*64*3,1))
vector_newX2 = vector_newX2.T

y_new2 = classifier.predict(vector_newX2)
print("The actual class is c3, predicted class for second world image is: ",y_new2)

oimg3 = cv2.imread(r"D:\Kaggle_Distracted_Driver\imgs\test\img_9.jpg")
img3 = cv2.resize(oimg3,(64,64))

vector_newX3 = np.reshape(img3,(64*64*3,1))
vector_newX3 = vector_newX3.T

y_new3 = classifier.predict(vector_newX3)
print("The actual class is c5, predicted class for third world image is: ",y_new3)

oimg4 = cv2.imread(r"D:\Kaggle_Distracted_Driver\imgs\test\img_11.jpg")
img4 = cv2.resize(oimg4,(64,64))

vector_newX4 = np.reshape(img4,(64*64*3,1))
vector_newX4 = vector_newX4.T

y_new4 = classifier.predict(vector_newX4)
print("The actual class is c1, predicted class for fourth real world image is: ",y_new4)'''

