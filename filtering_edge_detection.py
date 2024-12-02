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

sigma_values = [1, 2, 4, 8]

def gaussian_filter(image, sigma):

    # Calculate kernel size based on ±3σ truncation rule
    ksize = max(3, int(6 * sigma + 1))  # Ensure minimum kernel size of 3
    if ksize % 2 == 0:
        ksize += 1  # Make kernel size odd if it's even

    image_filtered = cv.GaussianBlur(image, (ksize,ksize), sigma)

    return image_filtered

def gaussian_filter_diff(original_image, filtered_image):
    image_diff = original_image - filtered_image

    return image_diff


for i, sigma in enumerate(sigma_values):
    # Apply Gaussian filter
    filtered_image = gaussian_filter(gray_mandrill, sigma)
    # Compute difference
    diff_image = gaussian_filter_diff(gray_mandrill, filtered_image)

    # Plot the filtered image
    plt.subplot(len(sigma_values), 2, i * 2 + 1)
    plt.imshow(filtered_image, cmap='gray')
    plt.title(f'Filtered Image (σ={sigma})')
    plt.axis('off')

    # Plot the difference image
    plt.subplot(len(sigma_values), 2, i * 2 + 2)
    plt.imshow(diff_image, cmap='gray')
    plt.title(f'Difference (σ={sigma})')
    plt.axis('off')
plt.tight_layout()
plt.show()
plt.close()

# Gradient magnitude computation using Gaussian derivatives. Use σ = 1, 2, 4, 8 pixels, and 
# explain in detail what can be seen and how the results differ. 

# manually done with opencv

def gaussian_kernels(sigma):
    """
    Generate 2D Gaussian derivative kernels for x and y directions.
    
    :param sigma: Standard deviation of the Gaussian
    :return: The derivative kernels for x and y directions
    """
    # Calculate kernel size based on ±3σ rule
    ksize = max(3, int(6 * sigma + 1))  # Ensure minimum kernel size is 3
    if ksize % 2 == 0:
        ksize += 1  # Make sure the kernel size is odd

    # 1D Gaussian kernel
    gaussian = cv.getGaussianKernel(ksize, sigma)  
    gaussian_first_deriv = -np.diff(gaussian[:, 0])  

    # 2D derivative kernels
    Gx = np.outer(gaussian_first_deriv, gaussian[:, 0])  # x direction
    Gy = Gx.T  # y direction 

    return Gx, Gy

def gradient_magnitude_gaussian(image, sigma):
    """
    Gradient magnitude using Gaussian derivatives.

    :param image: Original grayscale image
    :param sigma: Standard deviation of the Gaussian
    :return: Gradient magnitude image
    """

    Gx_kernel, Gy_kernel = gaussian_kernels(sigma)

    # Compute gradients in x and y directions and magnitude
    gradient_x = cv.filter2D(image, cv.CV_64F, Gx_kernel)  
    gradient_y = cv.filter2D(image, cv.CV_64F, Gy_kernel)  
    grad_mag = np.sqrt(gradient_x**2 + gradient_y**2)

    return grad_mag

sigmas = [1,2,4,8]
gradient_magnitudes = []

for sigma in sigmas:
    grad_mag = gradient_magnitude_gaussian(gray_mandrill, sigma)
    gradient_magnitudes.append((sigma, grad_mag))

plt.figure(figsize=(12, 8))
for i, (sigma, grad_mag) in enumerate(gradient_magnitudes):
    plt.subplot(2, 2, i + 1)
    plt.imshow(grad_mag, cmap='gray')
    plt.title(f"Gradient Magnitude (σ={sigma})")
    plt.axis('off')
plt.tight_layout()
plt.show()
plt.close()

# using the scipy library out of curiosity

sigmas = [1,2,4,8]
plt.figure(figsize=(12, 8))

for i,sigma in enumerate(sigmas):
    result = ndimage.gaussian_gradient_magnitude(gray_mandrill, sigma=sigma)
    plt.subplot(2, 2, i + 1)
    plt.imshow(result, cmap='gray')
    plt.title(f"Gradient Magnitude (σ={sigma})")
    plt.axis('off')
plt.tight_layout()
plt.show()
plt.close()

# Laplacian-Gaussian filtering. You may implement this as a difference og Gaussians. Again, use σ = 1, 2, 4, 8 pixels, and 
# explain in detail what can be seen and how the results differ.

def apply_gaussian_filter(image, sigma):
    """
    Apply Gaussian filter to an image.
    :param image: Grayscale input image
    :param sigma: Standard deviation of the Gaussian
    :return: Smoothed image
    """
    
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

# Apply Laplacian of Gaussian with different sigmas
sigmas = [1, 2, 4, 8] #These values control the extent of smoothing (larger sigma = more smoothing)

results = [laplacian_of_gaussian(gray_mandrill, sigma) for sigma in sigmas]

# Plot the results for each sigma value
plt.figure(figsize=(12, 8))
for i, sigma in enumerate(sigmas):
    plt.subplot(2, 2, i+1)
    plt.imshow(results[i], cmap='gray')
    plt.title(f"LoG (sigma={sigma})")
    plt.axis('off')
plt.tight_layout()
plt.show()
plt.close()

# Canny (or similar) edge detection. Describe the parameter values and their impact on the result. Select what you think 
# is a set of good parameter values, apply, show and decribe the result.

# apply gaussian smoothing before applying Canny
filtered_image = gaussian_filter(gray_mandrill, sigma=1) 
# set different ranges for lower bound x and upper bound y for finding the right values
thresholds = [(50, 100),(100, 150), (100, 200),( 150, 300)]
# looping through the thresholds of the canny algorhytm and plotting them for inpsection
for i, (x, y) in enumerate(thresholds):
    edges = cv.Canny(filtered_image, x, y)
    plt.subplot(1, len(thresholds), i+1)
    plt.imshow(edges, cmap= 'gray')
    plt.title(f'x={x} y={y}')
    plt.axis('off')
plt.tight_layout()
plt.show()
plt.close()






















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
