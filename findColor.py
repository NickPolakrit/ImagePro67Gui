import cv2
import numpy as np

cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1000)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 760)

while True:
    _, frame = cap.read()
    Gblurred = cv2.GaussianBlur(frame, (5, 5), 0)
    Bblurred = cv2.bilateralFilter(Gblurred, 9, 75, 75)
    Mblurred = cv2.medianBlur(Bblurred, 5)
    hsv_frame = cv2.cvtColor(Mblurred, cv2.COLOR_BGR2HSV)

    # Red color
    low_red = np.array([105, 0, 0])
    high_red = np.array([180, 255, 255])
    red_mask = cv2.inRange(hsv_frame, low_red, high_red)
    red = cv2.bitwise_and(frame, frame, mask=red_mask)
    # Blue color
    low_blue = np.array([94, 80, 2])
    high_blue = np.array([126, 255, 255])
    blue_mask = cv2.inRange(hsv_frame, low_blue, high_blue)
    blue = cv2.bitwise_and(frame, frame, mask=blue_mask)

    # Green color
    low_green = np.array([23, 38, 57])
    high_green = np.array([90, 255, 195])
    green_mask = cv2.inRange(hsv_frame, low_green, high_green)
    green = cv2.bitwise_and(frame, frame, mask=green_mask)

    # Yellow color
    low_yellow = np.array([0, 0, 191])
    high_yellow = np.array([64, 255, 255])
    yellow_mask = cv2.inRange(hsv_frame, low_yellow, high_yellow)
    yellow = cv2.bitwise_and(frame, frame, mask=yellow_mask)

    # Black color
    low_black = np.array([0, 0, 0])
    high_black = np.array([180, 90, 160])
    black_mask = cv2.inRange(hsv_frame, low_black, high_black)
    black = cv2.bitwise_and(frame, frame, mask=black_mask)

    # Every color except white
    low = np.array([0, 42, 0])
    high = np.array([179, 255, 255])
    mask = cv2.inRange(hsv_frame, low, high)
    result = cv2.bitwise_and(frame, frame, mask=mask)
    # cv2.imshow("Frame", frame)
    # cv2.imshow("Red", red)
    # cv2.imshow("Blue", blue)
    # cv2.imshow("Green", green)
    contours, _ = cv2.findContours(
        red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    for contour in contours:
        cv2.drawContours(frame, contour, -1, (0, 0, 255), 2)

    # GBR
    # cv2.imshow("Output Blurre", Mblurred)
    cv2.imshow("Result", result)
    frame3 = np.concatenate((black, yellow), axis=1)
    frame2 = np.concatenate((frame, red), axis=1)
    frame1 = np.concatenate((blue, green), axis=1)

    cv2.imshow('Frame, Red', frame2)
    cv2.imshow('Blue, Green', frame1)
    cv2.imshow('Black, Yellow', frame3)

    key = cv2.waitKey(1)
    if key == 27:
        break
