## Assignment 2: Filtering and Edge Detection
## Francesca Salute --> bhn327
## Martin
## Nicole


# import libraries
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

# Gaussian filtering. Show the result using σ = 1, 2, 4, 8 and explain in detail what can be seen.



























# Gradient magnitude computation using Gaussian derivatives. Use σ = 1, 2, 4, 8 pixels, and 
# explain in detail what can be seen and how the results differ.
























# Laplacian-Gaussian filtering. You may implement this as a difference og Gaussians. Again, use σ = 1, 2, 4, 8 pixels, and 
# explain in detail what can be seen and how the results differ.





























# Canny (or similar) edge detection. Describe the parameter values and their impact on the result. Select what you think 
# is a set of good parameter values, apply, show and decribe the result.
























# Experiment and testing