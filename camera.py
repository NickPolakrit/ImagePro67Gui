import cv2


#capture from camera at location 0
cap = cv2.VideoCapture(0)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1000)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 760)
#set the width and height, and UNSUCCESSFULLY set the exposure time
# cap.set(3, 1280)
# cap.set(4, 1024)
# cap.set(15, 0.1)
# cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)


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

    ret, img = cap.read()
    cv2.imshow("input", img)
    # cv2.imshow("thresholded", imgray*thresh2)
    cv2.imshow("Original image", img)

    # CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=3., tileGridSize=(8, 8))

    # convert from BGR to LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)  # split on 3 different channels

    l2 = clahe.apply(l)  # apply CLAHE to the L-channel

    lab = cv2.merge((l2, a, b))  # merge channels
    img2 = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)  # convert from LAB to BGR
    cv2.imshow('Increased contrast', img2)
    #cv2.imwrite('sunset_modified.jpg', img2)

    key = cv2.waitKey(10)
    if key == 27:
        break

cv2.destroyAllWindows()
cv2.VideoCapture(0).release()
