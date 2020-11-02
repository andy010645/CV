import cv2
import numpy as np
from matplotlib import pyplot as plt

def median_filter():
    file_name = ".\\Dataset_opencvdl\\Q2_Image\\Cat.png"
    img = cv2.imread(file_name)
    cv2.imshow("Original",img)
    blur=cv2.medianBlur(img,7)
    cv2.imshow("Filter",blur)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def gaussian_blur():
    file_name = ".\\Dataset_opencvdl\\Q2_Image\\Cat.png"
    img = cv2.imread(file_name)
    cv2.imshow("Original",img)
    blur=cv2.GaussianBlur(img,(3,3),0)
    cv2.imshow("Blur",blur)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def bilateral_filter():
    file_name = ".\\Dataset_opencvdl\\Q2_Image\\Cat.png"
    img = cv2.imread(file_name)
    cv2.imshow("Original",img)
    blur=cv2.bilateralFilter(img,9,90,90)
    cv2.imshow("Filter",blur)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
