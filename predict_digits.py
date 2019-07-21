__author__ = "Chukwumeme Tadinma Johnpaul"

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from keras.models import load_model
import sys


# Load digits
image = cv2.imread(sys.argv[1])
if image.all() == None:
	print("No image available.")
	sys.exit(1)

print("Image preprocessing----------------------------")
if sys.argv[1] == "digits2.jpeg":
    image = image[0:175,0:-1]


gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

ret, threshed = cv2.threshold(gray,160,255,cv2.THRESH_BINARY_INV)

kernel = np.ones((3,3),np.uint8)
threshed = cv2.dilate(threshed,kernel,iterations=1)
dilated = cv2.morphologyEx(threshed,cv2.MORPH_CLOSE,kernel)

cnts, heir = cv2.findContours(dilated.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
print("Image preprocessing done_---------------------------------")

print("Loading model---------------------------------")
model = load_model("mydigitrecognizer.h5")

print("Scanning image for digits ------------------------------")
for c in cnts:
    if cv2.contourArea(c) < 1:
        continue
    (x, y, w, h) = cv2.boundingRect(c)
    #print("x = ",x,"y = " ,y,"w = ",w,"h = ",h)
    width = x + w
    height= y + h
    #if width < 0 or height < 0:
        #width = height = 0
    cv2.rectangle(image, (x-4,y-4), (width+4, height+4),(0,255,255),2)
    #image = cv2.drawContours(image, c, -1,(0,255,0),3)
    roi = dilated[y-4:height+4,x-4:width+4]
    #mask = np.zeros((28,28))
    font = cv2.FONT_HERSHEY_SIMPLEX
    roi = cv2.resize(roi,(28,28))
    roi = roi[np.newaxis,:,:]
    roi = np.array([roi])
    #roi = cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
    prediction = model.predict(roi)
    prediction = np.argmax(prediction)
    print(prediction)
    cv2.putText(image,str(prediction),(x,y-5),font, 1,(255,0,0),2,cv2.LINE_AA)
print("Done reading image-----------------------------------")

print("saving recognised digits--------------")
cv2.imwrite("recognized_digits1.jpg",image)
print("Done")