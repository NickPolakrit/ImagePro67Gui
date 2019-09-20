import cv2

# Taking input from a directory path
img = cv2.imread('card.png')
# Applying Erosion
img_erosion = cv2.erode(img, kernel, iterations=1)
# Applying Dilation
img_dilation = cv2.dilate(img, kernel, iterations=1)
# Display Images
cv2.imshow('Input', img)
cv2.imshow('Erosion', img_erosion)
cv2.imshow('Dilation', img_dilation)
cv2.waitKey(0)
