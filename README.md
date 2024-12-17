# Filtering Edges

This code is for the assignment 2 in the Vision and Image Processing - frasalute is one of the main contributors. 
All of the assignment questions are answered within. The code is meant to run in one go. 

## Overview
The code applies image processing techniques to analyze and visualize the impact of smoothing and edge detection with varying σ (sigma) values. The following methods are implemented:

1. **Gaussian Filtering**
2. **Gradient Magnitude Computation** using Gaussian Derivatives
3. **Laplacian of Gaussian (LoG)**
4. **Canny Edge Detection**

## Libraries Used
- OpenCV
- NumPy
- Matplotlib
- SciPy

## How to Run
1. Ensure the required libraries are installed.
   ```bash
   pip install opencv-python numpy matplotlib scipy
   ```
2. Place the *mandrill.jpg* image in the same directory as the script.
3. Run the script:
   ```bash
   python assignment2.py
   ```

## Instructions
- The script runs all sections sequentially.
- Press any key to close the original and grayscale images.
- Close each displayed plot to proceed to the next one.
- The impact of different σ values will be shown for each method.

## Key Features
- **Gaussian Filtering**: Smooths the image with σ = 1, 2, 4, and 8.
- **Gradient Magnitude**: Computes image gradients to highlight edges.
- **Laplacian of Gaussian (LoG)**: Combines Gaussian smoothing with Laplacian for edge detection.
- **Canny Edge Detection**: Tests multiple threshold parameters for edge refinement.

## Results
The code generates plots for each method to compare how σ and parameters affect the output, showcasing the smoothing, edge detection, and gradient effects on the *Mandrill* image.

## Notes
- All results are displayed one after another; you can inspect each plot before closing it.
- The differences between methods and parameter choices are clearly visible in the generated visualizations.
