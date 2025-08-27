import cv2
import numpy as py
import face_recognition

# Load the reference image of Gustavo Cerati and convert image from BGR to RGB
imgCerati = face_recognition.load_image_file('imagesBasic/cerati1.png')
imgCerati = cv2.cvtColor(imgCerati, cv2.COLOR_BGR2RGB)

# Load the test image to compare and convert image from BGR to RGB
imgTest = face_recognition.load_image_file('imagesBasic/testCerati.jpg')
imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)

# Resize the reference image for display (50% of original size)
scale_percent = 50
width = int(imgCerati.shape[1] * scale_percent / 100)
height = int(imgCerati.shape[0] * scale_percent / 100)
dim = (width, height)
imgCeratiSmall = cv2.resize(imgCerati, dim, interpolation=cv2.INTER_AREA)

# Display the images in separate windows
cv2.imshow('Gustavo cerati', imgCeratiSmall)
cv2.imshow('Test Cerati', imgTest)
cv2.waitKey(0)

