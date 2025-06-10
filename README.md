# CUDA-Based Canny Edge Detection for Image Batches

Course: Parallel Programming and Algorithms
This project implements a parallelized version of the Canny Edge Detection algorithm using CUDA. It processes batches of 200×200 BMP images by performing grayscale conversion, Gaussian blur, Sobel edge detection, non-maximum suppression, double thresholding, and hysteresis — all on the GPU.

## 🔧 Features

- Processes **batches of 32 images** in parallel
- Fully implemented using **CUDA kernels**
- Records and prints **execution time** for each stage using CUDA Events
- Reads and writes **BMP images** using helper functions from `bmp.c`

## 🛠️ Tech Stack

- C, CUDA
- NVCC (CUDA compiler)

## 🧪 Image Processing Stages

1. **Convert RGB to Grayscale**
2. **Apply Gaussian Blur** (3×3 kernel, σ ≈ 1)
3. **Sobel Edge Detection** (X and Y gradients)
4. **Gradient Magnitude and Direction Calculation**
5. **Non-Maximum Suppression**
6. **Double Thresholding** (Strong/Weak/Noise separation)
7. **Edge Tracking by Hysteresis**
