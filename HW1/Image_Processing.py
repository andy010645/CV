import cv2
import numpy as np

def load_image():
    file_name = ".\\Dataset_opencvdl\\Q1_Image\\Uncle_Roger.jpg"
    img = cv2.imread(file_name)
    cv2.imshow("image", img)
    h,w,c=img.shape
    print("Height = ",h,"\nWidth = ",w)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def color_separation():
    file_name = ".\\Dataset_opencvdl\\Q1_Image\\Flower.jpg"
    img = cv2.imread(file_name)
    
    b,g,r=cv2.split(img)
    zeros = np.zeros(img.shape[:2], dtype = "uint8")
    cv2.imshow("Blue", cv2.merge([b, zeros, zeros]))
    cv2.imshow("Green", cv2.merge([zeros, g, zeros]))
    cv2.imshow("Red", cv2.merge([zeros, zeros, r]))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def image_flipping():
    file_name = ".\\Dataset_opencvdl\\Q1_Image\\Uncle_Roger.jpg"
    img = cv2.imread(file_name)
    flip=cv2.flip(img,1)
    cv2.imshow("image",flip)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



def blending():
    
    def Change(x):
        min1 = x /255
        max1 = 1 - min1
        dst = cv2.addWeighted(img, min1, flip, max1, 0)
        cv2.imshow('BLENDING', dst)

    file_name = ".\\Dataset_opencvdl\\Q1_Image\\Uncle_Roger.jpg"
    img = cv2.imread(file_name)
    flip=cv2.flip(img,1)
    cv2.namedWindow('BLENDING')
    cv2.createTrackbar("BLEND","BLENDING",0,255,Change)
    #show picture in 0
    Change(0)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



