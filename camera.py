import cv2
import numpy as np


#capture from camera at location 0
# cap = cv2.VideoCapture(0)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1000)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 760)
#set the width and height, and UNSUCCESSFULLY set the exposure time
# cap.set(3, 1280)
# cap.set(4, 1024)
# cap.set(15, 0.1)
# cap = cv2.VideoCapture(0)
# cap.set(3, 640)
# cap.set(4, 480)


# while True:
#     ret, img = cap.read()
#     cv2.imshow("input", img)
#     #cv2.imshow("thresholded", imgray*thresh2)

#     key = cv2.waitKey(10)
#     if key == 27:
#         break


# cv2.destroyAllWindows()
# cv2.VideoCapture(0).release()



while True:

    # ret, img = cap.read()
    img = cv2.imread("corpTest.png")
    # cv2.imshow("input", img)
    # cv2.imshow("thresholded", imgray*thresh2)
    cv2.imshow("Original image", img)

    # # CLAHE (Contrast Limited Adaptive Histogram Equalization)
    # clahe = cv2.createCLAHE(clipLimit=3., tileGridSize=(8, 8))

    # # convert from BGR to LAB color space
    # lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    # l, a, b = cv2.split(lab)  # split on 3 different channels

    # l2 = clahe.apply(l)  # apply CLAHE to the L-channel

    # lab = cv2.merge((l2, a, b))  # merge channels
    # img2 = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)  # convert from LAB to BGR
    kernel = np.ones((5, 5), np.uint8)

    Mblurred = cv2.medianBlur(img, 5)
    color_lower = np.array(
        [0, 7, 0], np.uint8)
    color_upper = np.array(
        [179, 255, 255], np.uint8)

    hsv = cv2.cvtColor(Mblurred, cv2.COLOR_BGR2HSV)

    color_mask = cv2.inRange(hsv, color_lower, color_upper)
    card = cv2.bitwise_and(
        img, img, mask=color_mask)

    bFilterC2 = cv2.bilateralFilter(color_mask, 9, 75, 75)
    openingC2 = cv2.morphologyEx(bFilterC2, cv2.MORPH_OPEN, kernel)
    closingC2 = cv2.morphologyEx(openingC2, cv2.MORPH_CLOSE, kernel)
    cv2.imshow('Increased contrast', color_mask)
    #cv2.imwrite('sunset_modified.jpg', img2)

    key = cv2.waitKey(10)
    if key == 27:
        break

cv2.destroyAllWindows()
cv2.VideoCapture(0).release()
