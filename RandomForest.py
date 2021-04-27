import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics 
import time
import cv2
import glob
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import matplotlib.pyplot as plt
import pydot
from io import StringIO
from sklearn.tree import export_graphviz
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
def matrix(row,col,imgs):
    a = row*col
    vector_newX = np.zeros((a, 1))
    imgSeq = []
    for img in imgs :
        o=cv2.imread(img,0)
        #print(o)
        oriimg = cv2.imread(img, cv2.IMREAD_GRAYSCALE)/255.
        #pixel values divide by 255 so that they can be be between 0 and 1
        #print(oriimg)
        imgSeq.append(img.split('img_')[1].split('.jpg')[0])
        img0 = cv2.resize(oriimg,(row,col))
        flat = img0.reshape(a,1)
        vector_newX = np.c_[vector_newX,flat]
        #print(img)
    print(imgSeq[0],imgSeq[30],imgSeq[263])
    vector_newX = vector_newX.T
    finalX_train = vector_newX[1:,:]
    print('size of feature martix is:',np.shape(finalX_train))
    return  finalX_train,imgSeq

def tree_function():
    
    mydata = pd.read_csv(r"F:\Trimester2\BTP\BTP\Jan16\Sobel_train.csv",header = None)
    
    X = mydata.iloc[0:,:-1].values  #iloc is a --> Purely integer-location based indexing for selection by position from data.
    print(X)
    Y = mydata.iloc[0:,-1].values
    print(Y)
    X=X/255.
    print(X)
    print(np.shape(X))
    print(np.shape(Y))
        
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.2,random_state = 50)

    #Create Random Forest classifer object
    clf = RandomForestClassifier(n_estimators=50,criterion="entropy")
    # Train Random Forest Classifer
    clf = clf.fit(X_train,Y_train)
    print('Random forest classifer: ')
    cnt=0
    for t in clf.estimators_:
        cnt+=1
        print("Decision Tree: ",cnt)
        r=tree.export_text(t)
        #print(r)
    print('total decision tree drawn: ',cnt)
    '''
    fn=data.feature_names
    cn=data.target_names
    fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=800)
    tree.plot_tree(clf.estimators_[0],feature_names = fn,class_names=cn,filled = True);
    fig.savefig('clf_individualtree.png')
    '''
    #Predict the response for test dataset
    Y_pred = clf.predict(X_test)
    
    Accuracy = metrics.accuracy_score(Y_test, Y_pred)
    print(Y_pred)
    print(" Accuracy:",(Accuracy*100))
    return clf,Accuracy

def predict(X,imgSeq):
    Y_pred_case = clf.predict(X)
    print('Predicted Result:',Y_pred_case)
    imgLabel = pd.read_csv("F:\Trimester2\BTP\BTP\images\ImageLabels.csv")
    imgLabel["RandomForest_Gray"]=''
    print(imgLabel.head(10))
    correct=0
    for i,val in enumerate(imgSeq):
        imgLabel.loc[imgLabel["image"]==int(val),"RandomForest_Gray"]=Y_pred_case[i]
        if (Y_pred_case[i] in str(imgLabel.loc[imgLabel["image"]==int(val)]["label"])):
            correct = correct+1
    print(imgLabel.head(10))
    total = len(imgSeq)
    genAcc = (correct/total) *100
    print(correct, "Images classified correctly out of ",total, "images")
    print("General Accuracy: ",genAcc)
    imgLabel.to_csv(r"F:\Trimester2\BTP\BTP\images\updated_ImageLabels.csv")


row = 64        #height of the image 
col = 64        #width of the image    

imgs = glob.glob(r"F:\Trimester2\BTP\BTP\test 500\*.jpg")
combined_train,imgSeq = matrix(row,col,imgs)
clf,acc = tree_function()
start_time = time.time()
predict(combined_train,imgSeq)
#print("--- %s seconds ---" % (time.time() - start_time))

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

imgs = glob.glob(r"F:\Trimester2\BTP\BTP\Jan11\Test\*.jpg")
DDEPTH = cv2.CV_16S
for img in imgs:
    oriimg = cv2.imread(img,0)
    img0 = cv2.resize(oriimg,(64,64))

    img1 = cv2.GaussianBlur(img0, (3, 3), 2)

    gradx = cv2.Sobel(img1, DDEPTH , 1, 0, ksize=3, scale=1, delta=0)
    gradx = cv2.convertScaleAbs(gradx)
    
    grady = cv2.Sobel(img1, DDEPTH , 0, 1, ksize=3, scale=1, delta=0)
    grady = cv2.convertScaleAbs(grady)
    
    grad = cv2.addWeighted(gradx, 0.5, grady, 0.5, 0)
    #print(np.shape(grad))
    flat = grad.reshape(1,64*64)


    #flat = img0.reshape(1,64*64)
    state=clf.predict(flat)
    print(str(state[0]))
    lb=int(str(state[0])[1])
    print(lb)
    msg=label(lb)
    cv2.rectangle(oriimg, (40, 52), (420, 8), (255,255,255), cv2.FILLED)
    cv2.rectangle(oriimg, (40, 52), (420, 8), (0,0,0), 3)
    cv2.putText(oriimg, msg, (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.15, (0, 255, 0), 2)
    cv2.imshow('Distraction',oriimg)
    cv2.waitKey(0)