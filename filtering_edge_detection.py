## Assignment 2: Filtering and Edge Detection
## Francesca Salute --> bhn327
## Martin
## Nicole


# import libraries
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage


# read the Mandrill image
mandrill_img = cv.imread('mandrill.jpg')

# check and raise error if needed
if mandrill_img is not None:
    cv.imshow('Loaded Image', mandrill_img)
    cv.waitKey(0) # press whatever key to stop it
    cv.destroyAllWindows()
else:
    print("Error loading the image")

# convert the image to grayscale and show it
gray_mandrill = cv.cvtColor(mandrill_img, cv.COLOR_BGR2GRAY)
cv.imshow('Grayscale Image', gray_mandrill)
cv.waitKey(0) 
cv.destroyAllWindows()

# Gaussian filtering. Show the result using σ = 1, 2, 4, 8 and explain in detail what can be seen.


























# Gradient magnitude computation using Gaussian derivatives. Use σ = 1, 2, 4, 8 pixels, and 
# explain in detail what can be seen and how the results differ.

# using the scipy library 

sigmas = [1,2,4,8]

for sigma in sigmas:
    result = ndimage.gaussian_gradient_magnitude(gray_mandrill, sigma=sigma)
    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(121)  # left side with original image
    ax2 = fig.add_subplot(122)  # right side with gradient computation

    ax1.imshow(gray_mandrill, cmap='gray')
    ax1.set_title("Grayscale Image")
    ax1.axis('off')

    ax2.imshow(result, cmap='gray')
    ax2.set_title(f"Gradient Magnitude (σ={sigma})")
    ax2.axis('off')
    plt.show()
    plt.close(fig) # to avoid overlapping each time a plot get closed the new one with the other sigma pops up


# manually done with opencv


















# Laplacian-Gaussian filtering. You may implement this as a difference og Gaussians. Again, use σ = 1, 2, 4, 8 pixels, and 
# explain in detail what can be seen and how the results differ.

def apply_gaussian_filter(image, sigma):
    """
    Apply Gaussian filter to an image.
    :param image: Grayscale input image
    :param sigma: Standard deviation of the Gaussian
    :return: Smoothed image
    """
    if sigma < 0.5:
        raise ValueError("Sigma values below 0.5 are not valid for discrete pixel grids.")
  
    # Calculate kernel size based on ±3σ truncation rule
    ksize = max(3, int(6 * sigma + 1))  # Ensure minimum kernel size of 3
    if ksize % 2 == 0:
        ksize += 1  # Make kernel size odd if it's even

    # Apply Gaussian blur using OpenCV's built-in function
    return cv.GaussianBlur(image, (ksize, ksize), sigma)

def apply_laplacian(image):
    """
    Apply Laplacian operator to an image.
    :param image: Grayscale input image
    :return: Image after Laplacian
    """
    # Compute the Laplacian of the image - second derivative
    laplacian = cv.Laplacian(image, cv.CV_64F) # Use a 64-bit float to preserve precision
    return laplacian


def laplacian_of_gaussian(image, sigma):
    """
    Apply Laplacian of Gaussian to an image.
    :param image: Grayscale input image
    :param sigma: Standard deviation of the Gaussian
    :return: LoG result
    """
    # Smooth the image with a the Gaussian filter function
    smoothed = apply_gaussian_filter(image, sigma)

    # Apply the Laplacian function to the smoothed image
    laplacian = apply_laplacian(smoothed)

    return laplacian
"""
# Apply Laplacian of Gaussian with different sigmas
sigmas = [1, 2, 4, 8] #These values control the extent of smoothing (larger sigma = more smoothing)

results = [laplacian_of_gaussian(gray_image, sigma) for sigma in sigmas]

# Plot the results for each sigma value
plt.figure(figsize=(12, 8))
for i, sigma in enumerate(sigmas):
    plt.subplot(2, 2, i+1)
    plt.imshow(results[i], cmap='gray')
    plt.title(f"LoG (sigma={sigma})")
    plt.axis('off')
plt.tight_layout()
plt.show()
"""

























# Canny (or similar) edge detection. Describe the parameter values and their impact on the result. Select what you think 
# is a set of good parameter values, apply, show and decribe the result.
























# Experiment and testing
"""
# Part 3: Experiments to understand how the different sigmas influence the edges detection


# Sigma set - small values
sigmas_small = [0.5, 1, 2, 3]
results_small = [laplacian_of_gaussian(gray_image, sigma) for sigma in sigmas_small]

# Sigma set - medium values
sigmas_medium = [4, 5, 6, 7]
results_medium = [laplacian_of_gaussian(gray_image, sigma) for sigma in sigmas_medium]

# Sigma set - large values 
sigmas_large = [8, 9, 10, 11]
results_large = [laplacian_of_gaussian(gray_image, sigma) for sigma in sigmas_large]

# Plot the results for the first set of sigmas (small values)
plt.figure(figsize=(12, 8))
for i, sigma in enumerate(sigmas_small):
    plt.subplot(2, 2, i + 1)
    plt.imshow(results_small[i], cmap='gray')
    plt.title(f"LoG (sigma={sigma})")
    plt.axis('off')
plt.tight_layout()
plt.show()

# Plot the results for the second set of sigmas (medium values)
plt.figure(figsize=(12, 8))
for i, sigma in enumerate(sigmas_medium):
    plt.subplot(2, 2, i + 1)
    plt.imshow(results_medium[i], cmap='gray')
    plt.title(f"LoG (sigma={sigma})")
    plt.axis('off')
plt.tight_layout()
plt.show()

# Plot the results for the third set of sigmas (large values)
plt.figure(figsize=(12, 8))
for i, sigma in enumerate(sigmas_large):
    plt.subplot(2, 2, i + 1)
    plt.imshow(results_large[i], cmap='gray')
    plt.title(f"LoG (sigma={sigma})")
    plt.axis('off')
plt.tight_layout()
plt.show()
"""