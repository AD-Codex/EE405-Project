import numpy as np
import cv2
import matplotlib.pyplot as mp

img= cv2.imread("2.png")
img2=img.reshape((-1,3))

img2= np.float32(img2)

criteria= (cv2.TERM_CRITERIA_EPS+cv2.TermCriteria_MAX_ITER,10,1.0)

k=3

attempts=10

ret,label,center=cv2.kmeans(img2,k,None,criteria,attempts,cv2.KMEANS_PP_CENTERS)

center=np.uint8(center)

res=center[label.flatten()]
res2=res.reshape((img.shape))
mp.imshow(res2)
mp.show()
# cv2.imshow('ss',res2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# print(res2)