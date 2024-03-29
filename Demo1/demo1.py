
import cv2
import pandas as pd
import numpy as np
import imutils
import serial
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours

# using cam built-in to computer
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1000)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 760)
# cap.set(3, 640)
# cap.set(4, 480)

# serialDevice = serial.Serial(
#     "/dev/cu.usbserial-AC00YIZF", 115200, 8, 'N', 1, 0, 0, 0, 0, 0)

# serialDevice.setRTS(0)
# serialDevice.setDTR(0)


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

cv2.setTrackbarPos("X1", "xyPosition", 277)
cv2.setTrackbarPos("Y1", "xyPosition", 123)
cv2.setTrackbarPos("X2", "xyPosition", 620)
cv2.setTrackbarPos("Y2", "xyPosition", 121)
cv2.setTrackbarPos("X3", "xyPosition", 193)
cv2.setTrackbarPos("Y3", "xyPosition", 441)
cv2.setTrackbarPos("X4", "xyPosition", 667)
cv2.setTrackbarPos("Y4", "xyPosition", 441)


def safe_div(x, y):  # so we don't crash so often
    if y == 0:
        return 0
    return x/y


def rescale_result(result, percent=100):  # make the video windows a bit smaller
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
cv2.setTrackbarPos("kernel", windowName, 19)
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

    positionDraw = cv2.warpPerspective(frame, matrix, (500, 500))

    Gblurred = cv2.GaussianBlur(resultWarp, (5, 5), 0)
    Mblurred = cv2.medianBlur(Gblurred, 5)
    # Denoise = cv2.fastNlMeansDenoisingMulti(resultWarp, 2, 5, None, 4, 7, 35)
    # unsharpV = cv2.addWeighted(
    #     resultWarp, 1.5, Mblurred, -0.5, 0, resultWarp)
    # Bblurred = cv2.bilateralFilter(unsharpV, 9, 75, 75)

    hsv_frame = cv2.cvtColor(Mblurred, cv2.COLOR_BGR2HSV)

    # gray = cv2.cvtColor(Gblurred, cv2.COLOR_BGR2GRAY)
    # edged = cv2.Canny(gray, 10, 250)
    # cv2.imshow("Edged", edged)
    # cv2.imshow("Denoise", Denoise)

    # both = np.concatenate((frame, result), axis=1)
    # cv2.imshow('Frame', both)

    

    # ------ COLOR ------------
    # Red color
    low_red = np.array([0, 23, 32])
    high_red = np.array([11, 255, 255])
    red_mask = cv2.inRange(hsv_frame, low_red, high_red)
    red = cv2.bitwise_and(resultWarp, resultWarp, mask=red_mask)
    # Blue color
    low_blue = np.array([108, 0, 70])
    high_blue = np.array([163, 255, 237])
    blue_mask = cv2.inRange(hsv_frame, low_blue, high_blue)
    blue = cv2.bitwise_and(resultWarp, resultWarp, mask=blue_mask)

    # Green color
    low_green = np.array([26, 24, 42])
    high_green = np.array([97, 102, 166])
    green_mask = cv2.inRange(hsv_frame, low_green, high_green)
    green = cv2.bitwise_and(resultWarp, resultWarp, mask=green_mask)

    # Yellow color
    low_yellow = np.array([21, 39, 121])
    high_yellow = np.array([94, 178, 254])
    yellow_mask = cv2.inRange(hsv_frame, low_yellow, high_yellow)
    yellow = cv2.bitwise_and(resultWarp, resultWarp, mask=yellow_mask)

    # Black color
    low_black = np.array([88, 13, 106])
    high_black = np.array([128, 255, 209])
    black_mask = cv2.inRange(hsv_frame, low_black, high_black)
    black = cv2.bitwise_and(resultWarp, resultWarp, mask=black_mask)

    # Card
    low = np.array([0, 10, 96])
    high = np.array([25, 74, 255])
    mask = cv2.inRange(hsv_frame, low, high)
    result = cv2.bitwise_and(resultWarp, resultWarp, mask=mask)

    countRed = 0
    countBlue = 0
    countYellow = 0
    countGreen = 0
    countBlack = 0

    # ----------- Serial -------------

    # # Contour Red
    contoursRed, _ = cv2.findContours(
        red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    try:
        biggest_contoursRed = max(contoursRed, key=cv2.contourArea)
        (x, y, w, h) = cv2.boundingRect(biggest_contoursRed)
        cv2.rectangle(resultWarp, (x, y), (x+w, y+h), (0, 0, 255), 2)
        countRed = 1

        Mred = cv2.moments(biggest_contoursRed)
        rX = int(Mred["m10"] / Mred["m00"])
        rY = int(Mred["m01"] / Mred["m00"])
        cv2.circle(resultWarp, (rX, rY), 5, (0, 0, 255), -1)
        cv2.putText(resultWarp, 'X :' + str(rX) + " Y :" + str(rY),
                    # bottomLeftCornerOfText
                    (rX + 30, rY),
                    cv2.FONT_HERSHEY_SIMPLEX,  # font
                    0.55,                      # fontScale
                    (0, 0, 255),            # fontColor
                    1)

    # cv2.putText(resultWarp, ":" + rX, (rX - 25, rY - 25),
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    except:
        pass
    contoursBlue, _ = cv2.findContours(
        blue_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    try:
        biggest_contoursBlue = max(contoursBlue, key=cv2.contourArea)
        (x, y, w, h) = cv2.boundingRect(biggest_contoursBlue)
        cv2.rectangle(resultWarp, (x, y), (x+w, y+h), (255, 0, 0), 2)
        countBlue = 1

        Mblue = cv2.moments(biggest_contoursBlue)
        bX = int(Mblue["m10"] / Mblue["m00"])
        bY = int(Mblue["m01"] / Mblue["m00"])
        cv2.circle(resultWarp, (bX, bY), 5, (255, 0, 0), -1)
        cv2.putText(resultWarp, 'X :' + str(bX) + " Y :" + str(bY),
                    # bottomLeftCornerOfText
                    (bX + 30, bY),
                    cv2.FONT_HERSHEY_SIMPLEX,  # font
                    0.55,                      # fontScale
                    (255, 0, 0),            # fontColor
                    1)

    except:
        pass

    contoursGreen, _ = cv2.findContours(
        green_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    try:
        biggest_contoursGreen = max(contoursGreen, key=cv2.contourArea)
        (x, y, w, h) = cv2.boundingRect(biggest_contoursGreen)
        cv2.rectangle(resultWarp, (x, y), (x+w, y+h), (0, 255, 0), 2)
        countGreen = 1

        Mgreen = cv2.moments(biggest_contoursGreen)
        gX = int(Mgreen["m10"] / Mgreen["m00"])
        gY = int(Mgreen["m01"] / Mgreen["m00"])
        cv2.circle(resultWarp, (gX, gY), 5, (0, 255, 0), -1)
        cv2.putText(resultWarp, 'X :' + str(gX) + " Y :" + str(gY),
                    # bottomLeftCornerOfText
                    (gX + 30, gY),
                    cv2.FONT_HERSHEY_SIMPLEX,  # font
                    0.55,                      # fontScale
                    (0, 255, 0),            # fontColor
                    1)

        pGx = gX*0.8
        pGy = gY*0.8
        cv2.circle(positionDraw, (pGx, pGy), 5, (0, 255, 0), -1)

    except:
        pass

    contoursYellow, _ = cv2.findContours(
        yellow_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    try:
        biggest_contoursYellow = max(contoursYellow, key=cv2.contourArea)
        (x, y, w, h) = cv2.boundingRect(biggest_contoursYellow)
        cv2.rectangle(resultWarp, (x, y), (x+w, y+h), (0, 255, 255), 2)
        countYellow = 1

        Myellow = cv2.moments(biggest_contoursYellow)
        yX = int(Myellow["m10"] / Myellow["m00"])
        yY = int(Myellow["m01"] / Myellow["m00"])
        cv2.circle(resultWarp, (yX, yY), 5, (0, 255, 255), -1)
        cv2.putText(resultWarp, 'X :' + str(yX) + " Y :" + str(yY),
                    # bottomLeftCornerOfText
                    (yX + 30, yY),
                    cv2.FONT_HERSHEY_SIMPLEX,  # font
                    0.55,                      # fontScale
                    (0, 255, 255),            # fontColor
                    1)

    except:
        pass

    contoursBlack, _ = cv2.findContours(
        black_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    try:
        biggest_contoursBlack = max(contoursBlack, key=cv2.contourArea)
        (x, y, w, h) = cv2.boundingRect(biggest_contoursBlack)
        cv2.rectangle(resultWarp, (x, y), (x+w, y+h), (0, 0, 0), 2)
        countBlack = 1

        Mblack = cv2.moments(biggest_contoursBlack)
        blX = int(Mblack["m10"] / Mblack["m00"])
        blY = int(Mblack["m01"] / Mblack["m00"])
        cv2.circle(resultWarp, (blX, blY), 5, (0, 0, 0), -1)
        cv2.putText(resultWarp, 'X :' + str(blX) + " Y :" + str(blY),
                    # bottomLeftCornerOfText
                    (blX + 30, blY),
                    cv2.FONT_HERSHEY_SIMPLEX,  # font
                    0.55,                      # fontScale
                    (0, 0, 0),            # fontColor
                    1)

    except:
        pass

    contoursCard, _ = cv2.findContours(
        mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    try:
        biggest_contoursCard = max(contoursCard, key=cv2.contourArea)
        (x, y, w, h) = cv2.boundingRect(biggest_contoursCard)
        cv2.rectangle(resultWarp, (x, y), (x+w, y+h), (0, 0, 0), 2)

        Mcard = cv2.moments(biggest_contoursCard)

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

    # Draw Text
    cv2.rectangle(resultWarp, (0, 0), (100, 110), (0, 0, 0), -1)
    cv2.putText(resultWarp, 'Red    :' + str(countRed),
                (10, 20),                  # bottomLeftCornerOfText
                cv2.FONT_HERSHEY_SIMPLEX,  # font
                0.5,                      # fontScale
                (0, 0, 255),            # fontColor
                1)                        # lineType
    cv2.putText(resultWarp, 'Yellow :' + str(countYellow),
                (10, 40),                  # bottomLeftCornerOfText
                cv2.FONT_HERSHEY_SIMPLEX,  # font
                0.5,                      # fontScale
                (0, 255, 255),            # fontColor
                1)
    cv2.putText(resultWarp, 'Blue   :' + str(countBlue),
                (10, 60),                  # bottomLeftCornerOfText
                cv2.FONT_HERSHEY_SIMPLEX,  # font
                0.5,                      # fontScale
                (255, 238, 0),            # fontColor
                1)
    cv2.putText(resultWarp, 'Green  :' + str(countGreen),
                (10, 80),                  # bottomLeftCornerOfText
                cv2.FONT_HERSHEY_SIMPLEX,  # font
                0.5,                      # fontScale
                (0, 255, 0),            # fontColor
                1)
    cv2.putText(resultWarp, 'Black  :' + str(countBlack),
                (10, 100),                  # bottomLeftCornerOfText
                cv2.FONT_HERSHEY_SIMPLEX,  # font
                0.5,                      # fontScale
                (255, 255, 255),            # fontColor
                1)

    # ------------
    k = cv2.waitKey(1)
    if k % 256 == 32:
        # print('Red X' + str(rX) + ' Y' + str(rY))
        # print('Blue X' + str(bX) + ' Y' + str(bY))
        # print('Green X' + str(gX) + ' Y' + str(gY))
        # print('Yellow X' + str(yX) + ' Y' + str(yY))
        # print('Black X' + str(blX) + ' Y' + str(blY))
        # cv2.putText(resultWarp, 'Sending...',
        #             (100, 200),                  # bottomLeftCornerOfText
        #             cv2.FONT_HERSHEY_SIMPLEX,  # font
        #             1,                      # fontScale
        #             (0, 255, 0),            # fontColor
        #             2)
        # break

        
    else:
        pass
        
    # ------------

    # cv2.imshow(windowName, orig)
    # cv2.imshow('', closing)
    cv2.imshow('Mask All', result)
    # cv2.imshow('Color', closing)
    # frame2 = np.concatenate((orig, closing), axis=1)
    # cv2.imshow('window', frame2)
    # cv2.imshow('result', orig)
    cv2.imshow('Red', red)
    frame3 = np.concatenate((black, yellow), axis=1)
    # frame2 = np.concatenate((frame, red), axis=1)
    frame1 = np.concatenate((blue, green), axis=1)

    # cv2.imshow('Frame, Red', frame2)
    cv2.imshow('Blue, Green', frame1)
    cv2.imshow('Black, Yellow', frame3)
    cv2.imshow("Frame", frame)
    cv2.imshow("Output", resultWarp)

    # cv2.imshow("Output Blurre", Mblurred)

    # if cv2.waitKey(30) >= 0:
    #     showLive = False
    key = cv2.waitKey(1)
    if key == 27:
        break


cap.release()
cv2.destroyAllWindows()
