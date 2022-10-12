import pytesseract
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

#############################################
# Functions

# Get Grayscale Image
def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Noise Removal
def remove_noise(image):
    return cv2.medianBlur(image, 5)

# Remove GridLines
def remove_gridLines(image):
    image = 255 - image

    image = cv2.medianBlur(image, 3)

    ret,image = cv2.threshold(image, 100, 255, cv2.THRESH_BINARY)

    image = cv2.medianBlur(image, 3)

    '''
    kernal = np.ones((1,3), np.uint8)
    image = cv2.erode(image, kernal, iterations=1)

    kernal = np.ones((1,3), np.uint8)
    image = cv2.dilate(image, kernal, iterations=1)

    kernal = np.ones((3,1), np.uint8)
    image = cv2.erode(image, kernal, iterations=1)

    kernal = np.ones((3,1), np.uint8)
    image = cv2.dilate(image, kernal, iterations=1)
    '''
    return image

# Side-by-Side Plot
def sideBySidePlot(imageOne, imageTwo):
    plt.subplot(121),plt.imshow(imageOne),plt.title('Original')
    plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(imageTwo),plt.title('Updated')
    plt.xticks([]), plt.yticks([])
    plt.show()

#############################################

# Start of Main

##############################
# Reading the original Image
#img = Image.open('Input Images/Test.PNG')
img = cv2.imread('Input Images/Test.PNG')

##############################
# Pre-Processing

# Remove GridLines
imgPrime = remove_gridLines(img)

# Show the result of Pre-Processing
sideBySidePlot(img, imgPrime)

# Saving Changes
#cv2.imwrite("Output.jpg", img)


##############################
# OCR Section
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\jharol.perez\AppData\Local\Tesseract-OCR\tesseract.exe'

# Adding custom options
custom_config = r'--oem 3 --psm 6'
#text = pytesseract.image_to_string(img)

#print(text)


