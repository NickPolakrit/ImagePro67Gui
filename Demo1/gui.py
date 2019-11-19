# import cv2
# import numpy as np

# cap = cv2.VideoCapture(0)
# cap1 = cv2.VideoCapture(0)

# # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1000)
# # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 760)

# cap.set(3, 640)
# cap.set(4, 480)

# while(cap.isOpened()):

#     ret, frame = cap.read()
#     # ret1, frame1 = cap1.read()
#     if ret == True:

#         both = np.concatenate((frame, frame), axis=1)
#         # both = np.concatenate((frame, frame))
#         # both = np.column_stack((frame, frame))

#         cv2.imshow('Frame', both)
#         # out.write(frame)

#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     else:
#         break

# cap.release()
# out.release()


# cv2.waitKey(0)
# cv2.destroyAllWindows()


import cv2
import numpy as np

img = cv2.imread('opencv_frame_0.png')
# img = cv2.imread('demo1.png')
# ret, thresh = cv2.threshold(img, 127, 255, 0)
# contours, hierarchy = cv2.findContours(thresh, 1, 2)

# Card
hsv_frame2 = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
cardCount = 0
low = np.array([0, 10, 96])
high = np.array([25, 74, 255])
mask = cv2.inRange(hsv_frame2, low, high)
result = cv2.bitwise_and(img, img, mask=mask)
cv2.imshow("card", result)
contoursCard, _ = cv2.findContours(
    mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
ret, thresh = cv2.threshold(result, 127, 255, 0)
contours, hierarchy = cv2.findContours(thresh, 1, 2)

cnt = contours[0]
area = cv2.contourArea(cnt)
print(str(area))


cv2.waitKey(0)
# cnt = contours[0]
# M = cv2.moments(cnt)
# print M

# area = cv2.contourArea(cnt)
# print(str(area))
