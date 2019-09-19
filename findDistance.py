import cv2
import pandas as pd
import numpy as np
import imutils
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours

# using cam built-in to computer
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1000)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 760)


def safe_div(x, y):  # so we don't crash so often
    if y == 0:
        return 0
    return x/y


def nothing(x):  # for trackbar
    pass


def rescale_result(result, percent=80):  # make the video windows a bit smaller
    width = int(result.shape[1] * percent / 100)
    height = int(result.shape[0] * percent / 100)
    dim = (width, height)
    return cv2.resize(result, dim, interpolation=cv2.INTER_AREA)


if not cap.isOpened():
    print("can't open camera")
    exit()

windowName = "Result"

cv2.namedWindow(windowName)

# Sliders to adjust image
# https://medium.com/@manivannan_data/set-trackbar-on-image-using-opencv-python-58c57fbee1ee
cv2.createTrackbar("threshold", windowName, 75, 255, nothing)
cv2.createTrackbar("kernel", windowName, 5, 30, nothing)
cv2.createTrackbar("iterations", windowName, 1, 10, nothing)

showLive = True
while(showLive):

    ret, frame = cap.read()

    cv2.circle(frame, (247, 92), 5, (0, 0, 255), -1)
    cv2.circle(frame, (734, 93), 5, (0, 0, 255), -1)
    cv2.circle(frame, (23, 533), 5, (0, 0, 255), -1)
    cv2.circle(frame, (994, 533), 5, (0, 0, 255), -1)
    pts1 = np.float32([[247, 92],
                       [734, 93],
                       [23, 533],
                       [994, 533]])
    pts2 = np.float32([[0, 0], [500, 0], [0, 500], [500, 500]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)

    result = cv2.warpPerspective(frame, matrix, (500, 500))

    cv2.imshow("Frame", frame)
    cv2.imshow("Output", result)

    key = cv2.waitKey(1)
    if key == 27:
        break

    result_resize = rescale_result(result)
    if not ret:
        print("cannot capture the frame")
        exit()

    thresh = cv2.getTrackbarPos("threshold", windowName)
    ret, thresh1 = cv2.threshold(result_resize, thresh, 255, cv2.THRESH_BINARY)

    kern = cv2.getTrackbarPos("kernel", windowName)
    # square image kernel used for erosion
    kernel = np.ones((kern, kern), np.uint8)

    itera = cv2.getTrackbarPos("iterations", windowName)
    dilation = cv2.dilate(thresh1, kernel, iterations=itera)
    # refines all edges in the binary image
    erosion = cv2.erode(dilation, kernel, iterations=itera)

    opening = cv2.morphologyEx(erosion, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    closing = cv2.cvtColor(closing, cv2.COLOR_BGR2GRAY)

    # find contours with simple approximation cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE
    contours, hierarchy = cv2.findContours(
        closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    closing = cv2.cvtColor(closing, cv2.COLOR_GRAY2RGB)
    cv2.drawContours(closing, contours, -1, (128, 255, 0), 1)

    # focus on only the largest outline by area
    areas = []  # list to hold all areas

    for contour in contours:
        ar = cv2.contourArea(contour)
        areas.append(ar)

    max_area = max(areas)
    # index of the list element with largest area
    max_area_index = areas.index(max_area)

    # largest area contour is usually the viewing window itself, why?
    cnt = contours[max_area_index - 1]

    cv2.drawContours(closing, [cnt], 0, (0, 0, 255), 1)

    def midpoint(ptA, ptB):
        return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

    # compute the rotated bounding box of the contour
    orig = result_resize.copy()
    box = cv2.minAreaRect(cnt)
    box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
    box = np.array(box, dtype="int")

    # order the points in the contour such that they appear
    # in top-left, top-right, bottom-right, and bottom-left
    # order, then draw the outline of the rotated bounding
    # box
    box = perspective.order_points(box)
    cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 1)

    # loop over the original points and draw them
    for (x, y) in box:
        cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)

    # unpack the ordered bounding box, then compute the midpoint
    # between the top-left and top-right coordinates, followed by
    # the midpoint between bottom-left and bottom-right coordinates
    (tl, tr, br, bl) = box
    (tltrX, tltrY) = midpoint(tl, tr)
    (blbrX, blbrY) = midpoint(bl, br)

    # compute the midpoint between the top-left and top-right points,
    # followed by the midpoint between the top-righ and bottom-right
    (tlblX, tlblY) = midpoint(tl, bl)
    (trbrX, trbrY) = midpoint(tr, br)

    # draw the midpoints on the image
    cv2.circle(orig, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
    cv2.circle(orig, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
    cv2.circle(orig, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
    cv2.circle(orig, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)

    # draw lines between the midpoints
    cv2.line(orig, (int(tltrX), int(tltrY)),
             (int(blbrX), int(blbrY)), (255, 0, 255), 1)
    cv2.line(orig, (int(tlblX), int(tlblY)),
             (int(trbrX), int(trbrY)), (255, 0, 255), 1)
    cv2.drawContours(orig, [cnt], 0, (0, 0, 255), 1)

    # compute the Euclidean distance between the midpoints
    dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
    dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

    # compute the size of the object
    # more to do here to get actual measurements that have meaning in the real world
    pixelsPerMetric = 1
    dimA = dA / pixelsPerMetric
    dimB = dB / pixelsPerMetric

    # draw the object sizes on the image
    cv2.putText(orig, "{:.1f}mm".format(dimA), (int(
        tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    cv2.putText(orig, "{:.1f}mm".format(dimB), (int(
        trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    # compute the center of the contour
    M = cv2.moments(cnt)
    cX = int(safe_div(M["m10"], M["m00"]))
    cY = int(safe_div(M["m01"], M["m00"]))

    # draw the contour and center of the shape on the image
    cv2.circle(orig, (cX, cY), 5, (255, 255, 255), -1)
    cv2.putText(orig, "center", (cX - 20, cY - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    cv2.imshow(windowName, orig)
    cv2.imshow('', closing)
    if cv2.waitKey(30) >= 0:
        showLive = False


cap.release()
cv2.destroyAllWindows()
