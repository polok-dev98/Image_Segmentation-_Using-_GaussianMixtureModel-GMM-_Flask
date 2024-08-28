# Image Segmentation Using Gaussian Mixture Model (GMM)

## Overview

This project demonstrates how to use a Gaussian Mixture Model (GMM) for image segmentation. GMM is an unsupervised learning algorithm that is particularly useful for identifying and separating distinct regions within an image based on pixel intensities.

## Gaussian Mixture Model (GMM) for Image Segmentation

### How GMM Works

Gaussian Mixture Models assume that the data points (in this case, pixel values) are generated from a mixture of several Gaussian distributions. Each distribution corresponds to a cluster within the image. The algorithm estimates the parameters of these distributions using the Expectation-Maximization (EM) algorithm and assigns each pixel to a cluster, resulting in a segmented image.

### Steps to Build a GMM for Image Segmentation

1. **Data Preparation**: Convert the image into a format suitable for GMM, usually a flattened 2D array of pixel values.
2. **Model Initialization**: Initialize a GMM with a specified number of components (clusters).
3. **Training the Model**: Fit the GMM to the pixel data using the EM algorithm.
4. **Segmenting the Image**: Assign each pixel to a cluster and reshape the result to form a segmented image.
5. **Post-processing**: Optionally, generate binary maps or other visual representations of the segmentation.

   
## Features

- Upload images for segmentation.
- Choose the number of clusters for segmentation using Gaussian Mixture Models (GMM).
- View the original, segmented, and binary images in the browser.

## Technologies Used

- **Python**: The core programming language.
- **Flask**: Web framework used for building the web application.
- **Pillow**: Python Imaging Library (PIL) used for image processing.
- **NumPy**: Library for handling numerical data and image arrays.
- **Scikit-learn**: Machine learning library used for Gaussian Mixture Models (GMM).
- **HTML/CSS**: For frontend development.

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/polok-dev98/Image_Segmentation_Using_GaussianMixtureModel_Flask.git
   cd image-segmentation-app

2. **Install the required dependencies:**

    ```bash
    pip install -r requirements.txt

3.  **Start the Flask app:**
    ```bash
    python app.py

## Output Sample

![image](https://github.com/user-attachments/assets/287d68ec-ab37-4429-90e1-6df38c52d0bf)

![image](https://github.com/user-attachments/assets/d7841900-381b-4481-8a7a-3b950b86285b)

![image](https://github.com/user-attachments/assets/89c7172c-0bc3-4fd2-bf70-064299f1687d)



   
