import cv2 
import numpy as np 
import glob

def matrix(row,col,Y,imgs):
    a = row*col
    DDEPTH = cv2.CV_16S
    vector_newX = np.zeros((a, 1))
    vector_newY = []
    
    for img_path in imgs:
        oriimg = cv2.imread(img_path,0)
        img0 = cv2.resize(oriimg,(row,col))
        img = cv2.GaussianBlur(img0, (3, 3), 2)

        gradx = cv2.Sobel(img, DDEPTH , 1, 0, ksize=3, scale=1, delta=0)
        gradx = cv2.convertScaleAbs(gradx)
        
        grady = cv2.Sobel(img, DDEPTH , 0, 1, ksize=3, scale=1, delta=0)
        grady = cv2.convertScaleAbs(grady)
        
        grad = cv2.addWeighted(gradx, 0.5, grady, 0.5, 0)
        #print(np.shape(grad))
        flat = grad.reshape(a,1)
        print('size of flattened vector : '+str(np.shape(flat)))
        vector_newX = np.c_[vector_newX,flat]
        print('size of newX vector : '+str(np.shape(vector_newX)))
        vector_newY = np.append(vector_newY,Y)
        print('size of newY vector : '+str(np.shape(vector_newY)))
        #print(img_path)
    
    vector_newX = vector_newX.T
    print('size of flattened vector : '+str(np.shape(vector_newX)))
    #1st-last column items along the second axis(column axis)=all the values from 1st-last columns
    finalX_train = vector_newX[1:,:]
    print('size of flattened vector : '+str(np.shape(finalX_train)))
    combined_train = np.c_[finalX_train,vector_newY]
    print('size of feature martix is:',np.shape(combined_train))
    return  combined_train

row = 100        #height of the image 
col = 100        #width of the image    

Y = ['c0']
imgs = glob.glob(r"F:\Trimester2\BTP\BTP\images\c0\*.jpg")
#print("h",imgs[0])
combined_train = matrix(row,col,Y,imgs)


Y = ['c1']
imgs1 = glob.glob(r"F:\Trimester2\BTP\BTP\images\c1\*.jpg")
combined_train1 = matrix(row,col,Y,imgs1)
X1 = np.concatenate((combined_train,combined_train1))

np.savetxt('Sobel_train.csv',X1, delimiter=',',fmt='%s')

print('size of feature martix is:',np.shape(X1))


#print(combined_train[:,13440])

##################  Checking each features with images: #######################
#check1 = finalX_train[0]
#check = check1.reshape(col,row,3)

#oimg = cv2.imread(r"C:\Users\vrush\Jupyter Noteboks\ML\img_104.jpg")
#img1 = cv2.resize(oimg,(row,col))
#print(img1)
#print(np.array_equal(check, img1))
###############################################################################