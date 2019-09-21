import cv2
import numpy as np
import imutils

cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1000)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 760)

while True:
    _, frame = cap.read()
    Gblurred = cv2.GaussianBlur(frame, (5, 5), 0)
    # Bblurred = cv2.bilateralFilter(Gblurred, 9, 75, 75)
    Mblurred = cv2.medianBlur(Gblurred, 5)
    hsv_frame = cv2.cvtColor(Mblurred, cv2.COLOR_BGR2HSV)

    # Red color
    low_red = np.array([106, 136, 0])
    high_red = np.array([180, 255, 255])
    red_mask = cv2.inRange(hsv_frame, low_red, high_red)
    red = cv2.bitwise_and(frame, frame, mask=red_mask)
    # Blue color
    low_blue = np.array([94, 189, 2])
    high_blue = np.array([126, 255, 255])
    blue_mask = cv2.inRange(hsv_frame, low_blue, high_blue)
    blue = cv2.bitwise_and(frame, frame, mask=blue_mask)

    # Green color
    low_green = np.array([69, 32, 0])
    high_green = np.array([94, 255, 234])
    green_mask = cv2.inRange(hsv_frame, low_green, high_green)
    green = cv2.bitwise_and(frame, frame, mask=green_mask)

    # Yellow color
    low_yellow = np.array([15, 0, 180])
    high_yellow = np.array([80, 255, 255])
    yellow_mask = cv2.inRange(hsv_frame, low_yellow, high_yellow)
    yellow = cv2.bitwise_and(frame, frame, mask=yellow_mask)

    # Black color
    low_black = np.array([80, 173, 91])
    high_black = np.array([115, 255, 165])
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

    # Contour Red
    contoursRed, _ = cv2.findContours(
        red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    biggest_contourRed = max(contoursRed, key=cv2.contourArea)
    (x, y, w, h) = cv2.boundingRect(biggest_contourRed)
    cv2.rectangle(result, (x, y), (x+w, y+h), (0, 0, 255), 2)

    contoursBlue, _ = cv2.findContours(
        blue_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    biggest_contourBlue = max(contoursBlue, key=cv2.contourArea)
    (x, y, w, h) = cv2.boundingRect(biggest_contourBlue)
    cv2.rectangle(result, (x, y), (x+w, y+h), (255, 0, 0), 2)
    # contour_sizes = [(cv2.contourArea(contour), contour)
    #                  for contour in contoursRed]
    # biggest_contour = max(contour_sizes, key=lambda x: x[0])[1]
    # for contour in contoursRed:
    #     cv2.drawContours(result, [biggest_contour], -1, (0, 0, 255), 2)
    # contoursRed = imutils.grab_contours(contoursRed)
    # for c in contoursRed:
    #     # compute the center of the contour
    #     M = cv2.moments(c)
    #     cX = int(M["m10"] / M["m00"])
    #     cY = int(M["m01"] / M["m00"])

    #     # draw the contour and center of the shape on the image
    #     cv2.drawContours(result, [c], -1, (0, 255, 0), 2)
    #     cv2.circle(result, (cX, cY), 7, (255, 255, 255), -1)
    #     cv2.putText(result, "center", (cX - 20, cY - 20),
    #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # # Contour Blue
    # contoursBlue, _=cv2.findContours(
    #     blue_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # contour_sizes=[(cv2.contourArea(contour), contour)
    #                  for contour in contoursBlue]
    # biggest_contour=max(contour_sizes, key=lambda x: x[0])[1]
    # for contour in contoursBlue:
    #     cv2.drawContours(result, [biggest_contour], -1, (255, 0, 0), 2)
    # # Contour Green
    # contoursGreen, _=cv2.findContours(
    #     green_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # contour_sizes=[(cv2.contourArea(contour), contour)
    #                  for contour in contoursGreen]
    # biggest_contour=max(contour_sizes, key=lambda x: x[0])[1]
    # for contour in contoursGreen:
    #     cv2.drawContours(result, [biggest_contour], -1, (0, 255, 0), 2)
    # # Contour Yellow
    # contoursYellow, _=cv2.findContours(
    #     yellow_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # contour_sizes=[(cv2.contourArea(contour), contour)
    #                  for contour in contoursYellow]
    # biggest_contour=max(contour_sizes, key=lambda x: x[0])[1]
    # for contour in contoursYellow:
    #     cv2.drawContours(result, [biggest_contour], -1, (0, 255, 255), 2)
    # # Contour Black
    # contoursBlack, _=cv2.findContours(
    #     black_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # contour_sizes=[(cv2.contourArea(contour), contour)
    #                  for contour in contoursBlack]
    # biggest_contour=max(contour_sizes, key=lambda x: x[0])[1]
    # for contour in contoursBlack:
    #     cv2.drawContours(result, [biggest_contour], -1, (0, 0, 0), 2)

    # for contour in contours:
    #     cv2.drawContours(frame, contour, -1, (0, 0, 255), 2)

    # for c in contours:
    #     peri = cv2.arcLength(c, True)
    #     approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    #     x, y, w, h = cv2.boundingRect(approx)
    #     cv2.rectangle(result, (x, y), (x+w, y+h), (0, 0, 255), 1)

    # x, y, w, h = cv2.boundingRect(contours)
    # cv2.rectangle(result, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # x, y, w, h = cv2.boundingRect(contours)
    # cv2.rectangle(result, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # GBR
    cv2.imshow("Output Blurre", Mblurred)
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
