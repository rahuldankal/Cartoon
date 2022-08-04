#############**TURN PHOTOS INTO CARTOON USING PYTHON**#############

#At first we are importing all the necessary libraries that we need for this project.  

import cv2  
#Also called OpenCV, it provides a common infrastructure for computer vision applications and has optimized-machine-learning algorithms. 
#It can be used to recognize objects, detect, and produce high-resolution images.

import numpy as np
#NumPy is a Python library used for working with arrays. 
#It also has functions for working in domain of linear algebra, fourier transform, and matrices.

import skimage.io
#Skimage reads the image, converts it from JPEG into a NumPy array, and returns the array.
#We save the array in a variable named img.

import matplotlib.pyplot as plt
#Matplotlib is a cross-platform, data visualization and graphical plotting library for Python and its numerical extension NumPy.

import skimage.filters
#This library is used image processing, and using natively NumPy arrays as image objects.
from google.colab.patches import cv2_imshow
from google.colab import files
#The first main step is loading the image. 
#Define the read_file function, which includes the cv2_imshow to load our selected image
def read_file(filename):
  img = cv2.imread(filename)
  cv2_imshow(img)
  return img
  #Call the created function to load the image.
#This will allow us to choose any image from the system.
uploaded = files.upload()

filename = next(iter(uploaded))
img = read_file(filename)
#PERFORMING EDGE MASK#
#In this function, we transform the image into grayscale. 
#Then, we reduce the noise of the blurred grayscale image by using cv2.medianBlur. 
#The larger blur value means fewer black noises appear in the image. 
#And then, apply adaptiveThreshold function, and define the line size of the edge. 
#A larger line size means the thicker edges that will be emphasized in the image.
def edge_mask(img, line_size, blur_value):
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  gray_blur = cv2.medianBlur(gray, blur_value)
  edges = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, line_size, blur_value)
  return edges
  
#After defining the function, call it and see the result.
line_size = 7
blur_value = 7
edges = edge_mask(img, line_size, blur_value)
cv2_imshow(edges)
#PERFORMING COLOR QUANTIZATION USING K-MEANS ALGORITHM#
#To do color quantization, we apply the K-Means clustering algorithm which is provided by the OpenCV library.
#To make it easier in the next steps, we can define the COLOR QUANTIZATION function as below.
#We can adjust the k value to determine the number of colors that we want to apply to the image.
#Defining K-Means Algorithm
def quantimage(img,k):
    i = np.float32(img).reshape(-30,30)
    condition = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,30,2.0)
    ret,label,center = cv2.kmeans(i, k , None, condition,6,cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    final_img = center[label.flatten()]
    final_img = final_img.reshape(img.shape)
    return final_img
#Using this function, we are showing 4 different output of 1 image.
#Using different values of k as (2,5,12,30)
#It shows that the higher the value of k is, the more colors will be used in output.
#Thus, more colors give clearer picture.
plt.imshow(quantimage(img,2))
plt.show()
plt.imshow(quantimage(img,5))
plt.show()
plt.imshow(quantimage(img,12))
plt.show()
plt.imshow(quantimage(img,30))
plt.show()
#PERFORMING BLUR EFFECT ON IMAGE
#Gaussian blur is the result of blurring an image by a Gaussian function. 
#It is a widely used effect in graphics software, typically to reduce image noise and reduce detail. 
#It is also used as a preprocessing stage before applying our machine learning or deep learning models.
image = skimage.io.imread(fname="xx.jpg")

#Sigma function plays a very vital role here, the value of sigma defines how many pixels are we combining.
#The more the value of sigma, the more blurred image will be. 
sigma = 9

#Apply Gaussian Blur, creating a new image
blurred = skimage.filters.gaussian(
    image, sigma=(sigma, sigma), truncate=9.0, multichannel=True)
#Display Blurred Image
fig, ax = plt.subplots()
plt.imshow(blurred)
plt.show()
#COMBINING EDGE MASK IMAGE, BILATERAL IMAGE AND ORIGINAL IMAGE 
#USING OPEN CV AND BITWISE FUNCTION
color = cv2.bilateralFilter(img, 1, 1, 1)
color2 = cv2.bilateralFilter(img, 151, 151, 151)
cartoon = cv2.bitwise_and(color, color2,mask=edges)

#OUTPUT SHOWS THE ORIGINAL IMAGE, EDGE MASKED IMAGE AND THEN THEIR COMBINED VERSION
#cv2_imshow(img)
#cv2_imshow(edges)
cv2_imshow(cartoon)
