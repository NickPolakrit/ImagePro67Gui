
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
# cap.set(3, 640)
# cap.set(4, 480)


def nothing(x):  # for trackbar
    pass


cv2.namedWindow('xyPosition')
cv2.createTrackbar('X1', 'xyPosition', 0, 1022, nothing)
cv2.createTrackbar('Y1', 'xyPosition', 0, 575, nothing)
cv2.createTrackbar('X2', 'xyPosition', 0, 1022, nothing)
cv2.createTrackbar('Y2', 'xyPosition', 0, 575, nothing)
cv2.createTrackbar('X3', 'xyPosition', 0, 1022, nothing)
cv2.createTrackbar('Y3', 'xyPosition', 0, 575, nothing)
cv2.createTrackbar('X4', 'xyPosition', 0, 1022, nothing)
cv2.createTrackbar('Y4', 'xyPosition', 0, 575, nothing)


def safe_div(x, y):  # so we don't crash so often
    if y == 0:
        return 0
    return x/y


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
cv2.createTrackbar("threshold", windowName, 75, 255, nothing)
cv2.createTrackbar("kernel", windowName, 5, 30, nothing)
cv2.createTrackbar("iterations", windowName, 1, 10, nothing)

cv2.setTrackbarPos("threshold", windowName, 210)
cv2.setTrackbarPos("kernel", windowName, 13)
cv2.setTrackbarPos("iterations", windowName, 1)
# Sliders to adjust image
# https://medium.com/@manivannan_data/set-trackbar-on-image-using-opencv-python-58c57fbee1ee


showLive = True
while(showLive):

    ret, frame = cap.read()

    x1 = cv2.getTrackbarPos('X1', 'xyPosition')
    y1 = cv2.getTrackbarPos('Y1', 'xyPosition')
    x2 = cv2.getTrackbarPos('X2', 'xyPosition')
    y2 = cv2.getTrackbarPos('Y2', 'xyPosition')
    x3 = cv2.getTrackbarPos('X3', 'xyPosition')
    y3 = cv2.getTrackbarPos('Y3', 'xyPosition')
    x4 = cv2.getTrackbarPos('X4', 'xyPosition')
    y4 = cv2.getTrackbarPos('Y4', 'xyPosition')

    cv2.setTrackbarPos("X1", "xyPosition", 219)
    cv2.setTrackbarPos("Y1", "xyPosition", 0)
    cv2.setTrackbarPos("X2", "xyPosition", 803)
    cv2.setTrackbarPos("Y2", "xyPosition", 0)
    cv2.setTrackbarPos("X3", "xyPosition", 26)
    cv2.setTrackbarPos("Y3", "xyPosition", 575)
    cv2.setTrackbarPos("X4", "xyPosition", 970)
    cv2.setTrackbarPos("Y4", "xyPosition", 575)

    cv2.circle(frame, (x1, y1), 5, (0, 0, 255), -1)
    cv2.circle(frame, (x2, y2), 5, (0, 0, 255), -1)
    cv2.circle(frame, (x3, y3), 5, (0, 0, 255), -1)
    cv2.circle(frame, (x4, y4), 5, (0, 0, 255), -1)
    pts1 = np.float32([[x1, y1],
                       [x2, y2],
                       [x3, y3],
                       [x4, y4]])
    pts2 = np.float32([[0, 0], [500, 0], [0, 500], [500, 500]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)

    resultWarp = cv2.warpPerspective(frame, matrix, (500, 500))

    Gblurred = cv2.GaussianBlur(resultWarp, (5, 5), 0)
    # Bblurred = cv2.bilateralFilter(Gblurred, 9, 75, 75)
    Mblurred = cv2.medianBlur(Gblurred, 5)
    hsv_frame = cv2.cvtColor(Mblurred, cv2.COLOR_BGR2HSV)

    # gray = cv2.cvtColor(Gblurred, cv2.COLOR_BGR2GRAY)
    # edged = cv2.Canny(gray, 10, 250)
    # cv2.imshow("Edged", edged)

    # both = np.concatenate((frame, result), axis=1)
    # cv2.imshow('Frame', both)

    result_resize = rescale_result(resultWarp)
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

    try:
        max_area = max(areas)
    except:
        pass
    # index of the list element with largest area

    try:
        max_area_index = areas.index(max_area)
    except:
        pass

    # largest area contour is usually the viewing window itself, why?
    try:
        cnt = contours[max_area_index - 1]
    except:
        pass

    # cv2.drawContours(closing, [cnt], 0, (0, 0, 255), 4)

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
    cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 4)

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
    # cv2.circle(orig, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
    # cv2.circle(orig, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
    # cv2.circle(orig, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
    # cv2.circle(orig, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)

    # draw lines between the midpoints
    # cv2.line(orig, (int(tltrX), int(tltrY)),
    #          (int(blbrX), int(blbrY)), (255, 0, 255), 1)
    # cv2.line(orig, (int(tlblX), int(tlblY)),
    #          (int(trbrX), int(trbrY)), (255, 0, 255), 1)
    # cv2.drawContours(orig, [cnt], 0, (0, 0, 255), 1)

    # compute the Euclidean distance between the midpoints
    dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
    dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

    # compute the size of the object
    # more to do here to get actual measurements that have meaning in the real world
    pixelsPerMetric = 1
    dimA = dA / pixelsPerMetric
    dimB = dB / pixelsPerMetric

    # draw the object sizes on the image , text size card
    cv2.putText(orig, "{:.1f}mm".format(dimA), (int(
        tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    cv2.putText(orig, "{:.1f}mm".format(dimB), (int(
        trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    # compute the center of the contour
    M = cv2.moments(cnt)
    cX = int(safe_div(M["m10"], M["m00"]))
    cY = int(safe_div(M["m01"], M["m00"]))
    # draw the contour and center of the shape on the image
    # cv2.circle(orig, (cX, cY), 5, (255, 255, 255), -1)
    # cv2.putText(orig, "center", (cX - 20, cY - 20),
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # ------ COLOR ------------
    # Red color
    low_red = np.array([106, 136, 0])
    high_red = np.array([180, 255, 255])
    red_mask = cv2.inRange(hsv_frame, low_red, high_red)
    red = cv2.bitwise_and(resultWarp, resultWarp, mask=red_mask)
    # Blue color
    low_blue = np.array([94, 189, 2])
    high_blue = np.array([126, 255, 255])
    blue_mask = cv2.inRange(hsv_frame, low_blue, high_blue)
    blue = cv2.bitwise_and(resultWarp, resultWarp, mask=blue_mask)

    # Green color
    low_green = np.array([69, 32, 0])
    high_green = np.array([94, 255, 234])
    green_mask = cv2.inRange(hsv_frame, low_green, high_green)
    green = cv2.bitwise_and(resultWarp, resultWarp, mask=green_mask)

    # Yellow color
    low_yellow = np.array([15, 0, 180])
    high_yellow = np.array([80, 255, 255])
    yellow_mask = cv2.inRange(hsv_frame, low_yellow, high_yellow)
    yellow = cv2.bitwise_and(resultWarp, resultWarp, mask=yellow_mask)

    # Black color
    low_black = np.array([79, 0, 0])
    high_black = np.array([130, 197, 197])
    black_mask = cv2.inRange(hsv_frame, low_black, high_black)
    black = cv2.bitwise_and(resultWarp, resultWarp, mask=black_mask)

    # Every color except white
    low = np.array([0, 42, 0])
    high = np.array([179, 255, 255])
    mask = cv2.inRange(hsv_frame, low, high)
    result = cv2.bitwise_and(resultWarp, resultWarp, mask=mask)

    # # Contour Red
    contoursRed, _ = cv2.findContours(
        red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    try:
        biggest_contoursRed = max(contoursRed, key=cv2.contourArea)
        (x, y, w, h) = cv2.boundingRect(biggest_contoursRed)
        cv2.rectangle(resultWarp, (x, y), (x+w, y+h), (0, 0, 255), 2)
    except:
        pass

    contoursBlue, _ = cv2.findContours(
        blue_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    try:
        biggest_contoursBlue = max(contoursBlue, key=cv2.contourArea)
        (x, y, w, h) = cv2.boundingRect(biggest_contoursBlue)
        cv2.rectangle(resultWarp, (x, y), (x+w, y+h), (255, 0, 0), 2)
    except:
        pass

    contoursGreen, _ = cv2.findContours(
        green_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    try:
        biggest_contoursGreen = max(contoursGreen, key=cv2.contourArea)
        (x, y, w, h) = cv2.boundingRect(biggest_contoursGreen)
        cv2.rectangle(resultWarp, (x, y), (x+w, y+h), (0, 255, 0), 2)
    except:
        pass

    contoursYellow, _ = cv2.findContours(
        yellow_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    try:
        biggest_contoursYellow = max(contoursYellow, key=cv2.contourArea)
        (x, y, w, h) = cv2.boundingRect(biggest_contoursYellow)
        cv2.rectangle(resultWarp, (x, y), (x+w, y+h), (0, 255, 255), 2)
    except:
        pass

    contoursBlack, _ = cv2.findContours(
        black_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    try:
        biggest_contoursBlack = max(contoursBlack, key=cv2.contourArea)
        (x, y, w, h) = cv2.boundingRect(biggest_contoursBlack)
        cv2.rectangle(resultWarp, (x, y), (x+w, y+h), (0, 0, 0), 2)
    except:
        pass

    # ------ COLOR ------------
    # countRed = str(len(contoursRed))
    # count = 0
    # for c in contoursRed:
    #     rect = cv2.boundingRect(c)
    #     x, y, w, h = rect
    #     area = w * h
    #     if area > 1000:
    #         count = count + 1  # นับ object ที่มีพื้นที่มากกว่า 1000 pixel
    #         cv2.rectangle(resultWarp, (x, y), (x+w, y+h),  5)
    # cv2.rectangle(resultWarp, (0, 0), (200, 120), (0, 0, 0), -1)
    # cv2.putText(resultWarp, 'Red Count : ' + str(len(contoursRed)),
    #             (10, 50),                  # bottomLeftCornerOfText
    #             cv2.FONT_HERSHEY_SIMPLEX,  # font
    #             0.5,                      # fontScale
    #             (0, 0, 255),            # fontColor
    #             1)                        # lineType

    # cv2.putText(resultWarp, "Red : " + countRed, (int(
    #     tltrX - 0), int(tltrY - 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    # countBlue = str(len(biggest_contoursBlue))
    # cv2.putText(resultWarp, "Blue : " + countBlue, (int(
    #     tltrX - 20), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    # countGreen = str(len(biggest_contoursGreen))
    # cv2.putText(resultWarp, "Red : " + countGreen, (int(
    #     tltrX - 30), int(tltrY - 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    # countYellow = str(len(biggest_contoursYellow))
    # cv2.putText(resultWarp, "Yellow : " + countYellow, (int(
    #     tltrX - 40), int(tltrY - 30)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    # countBlack = str(len(biggest_contoursBlack))
    # cv2.putText(resultWarp, "Black : " + countBlack, (int(
    #     tltrX - 50), int(tltrY - 40)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    # cv2.imshow(windowName, orig)
    # cv2.imshow('', closing)
    cv2.imshow('Color', closing)
    frame2 = np.concatenate((orig, closing), axis=1)
    # cv2.imshow('window', frame2)
    cv2.imshow('result', orig)
    cv2.imshow('Red', red)
    frame3 = np.concatenate((black, yellow), axis=1)
    # frame2 = np.concatenate((frame, red), axis=1)
    frame1 = np.concatenate((blue, green), axis=1)

    # cv2.imshow('Frame, Red', frame2)
    cv2.imshow('Blue, Green', frame1)
    cv2.imshow('Black, Yellow', frame3)
    cv2.imshow("Frame", frame)
    cv2.imshow("Output", resultWarp)
    cv2.imshow("Output Blurre", Mblurred)

    # if cv2.waitKey(30) >= 0:
    #     showLive = False
    key = cv2.waitKey(1)
    if key == 27:
        break


cap.release()
cv2.destroyAllWindows()
