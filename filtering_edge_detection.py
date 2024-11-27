import cv2 as cv
import numpy as np
import matplotlib as plt

# read the Mandrill image
mandrill_img = cv.imread('mandrill.jpg')

# check and raise error if needed
if mandrill_img is not None:
    cv.imshow('Loaded Image', mandrill_img)
    cv.waitKey(0)
    cv.destroyAllWindows()
else:
    print("Error loading the image")