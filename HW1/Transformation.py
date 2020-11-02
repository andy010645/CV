import cv2
import numpy as np
from matplotlib import pyplot as plt


def transformation_(r,s,tx,ty):
    file_name = ".\\Dataset_opencvdl\\Q4_Image\\Parrot.png"
    img = cv2.imread(file_name)
    cv2.imshow("Original",img)
    rows,cols=img.shape[:2]
    newx,newy=160+float(tx),84+float(ty)
    # tx,ty
    H = np.float32([[1,0,tx],[0,1,ty]])
    res=cv2.warpAffine(img,H,(cols,rows))
    # rotate scale
    M = cv2.getRotationMatrix2D((newx,newy),float(r),float(s))
    res2=cv2.warpAffine(res,M,(cols,rows))
    cv2.imshow("Image RST",res2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    


