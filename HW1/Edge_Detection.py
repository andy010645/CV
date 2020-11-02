import cv2
import numpy as np
from matplotlib import pyplot as plt
import math
import time

def padding_conv(img,kernel):
    h,w=img.shape
    #padding
    pad=np.ones((h+2,w+2))
    for i in range(1,w+1):
        for j in range(1,h+1):
            pad[j][i]=img[j-1][i-1]
    res=np.zeros((h,w))
    #conv
    for x in range(w):
        for y in range(h):
            filter=0
            xx=0
            for filter_x in range(x,x+3):
                yy=0
                for filter_y in range(y,y+3):
                    filter+=pad[filter_y][filter_x]*kernel[yy][xx]
                    yy+=1
                xx+=1
            if filter<0:
                filter=0
            res[y][x]=filter
    res=res.astype(np.uint8)
    return res

def gaussian_filter():
    file_name = ".\\Dataset_opencvdl\\Q3_Image\\Chihiro.jpg"
    img = cv2.imread(file_name)
    img_gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow("GRAY",img_gray)   
    # Gaussian filter
    x,y=np.mgrid[-1:2,-1:2]
    kernel=np.exp(-x**2-y**2)
    kernel=kernel/kernel.sum()

    res=padding_conv(img_gray,kernel)
    cv2.imwrite("Chihiro_gaussian.jpg",res)
    cv2_judge=cv2.filter2D(img_gray,-1,kernel)
    cv2.imshow("Gaussian blur",res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def sobel_x():
    file_name = "Chihiro_gaussian.jpg"
    img = cv2.imread(file_name)
    img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Sobel X
    sobel_x=np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    # conv
    res=padding_conv(img,sobel_x)
    cv2.imshow("Sobel X",res)
    cv2.imwrite("Chihiro_Sobel_X.jpg",res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def sobel_y():
    file_name = "Chihiro_gaussian.jpg"
    img = cv2.imread(file_name)
    img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Sobel Y
    sobel_y=np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
    # conv
    res=padding_conv(img,sobel_y)
    cv2.imshow("Sobel Y",res)
    cv2.imwrite("Chihiro_Sobel_Y.jpg",res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
def magnitude():
    file_name = "Chihiro_Sobel_X.jpg"
    img = cv2.imread(file_name)
    file_name = "Chihiro_Sobel_Y.jpg"
    img2 = cv2.imread(file_name)
    img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img2=cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    # Gaussian filter
    res=np.hypot(img,img2)
    res=res.astype(np.uint8)
    cv2.imshow("Magnitude",res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
