import cv2
import numpy as np

cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1000)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 760)

while True:
    _, frame = cap.read()

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

    result = cv2.warpPerspective(frame, matrix, (500, 505))

    #######
    hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)

    # red color

    red_lower = np.array([136, 87, 111], np.uint8)
    red_upper = np.array([180, 255, 255], np.uint8)

    # blue color

    blue_lower = np.array([99, 115, 150], np.uint8)
    blue_upper = np.array([110, 255, 255], np.uint8)

    # yellow color

    yellow_lower = np.array([22, 60, 200], np.uint8)
    yellow_upper = np.array([60, 255, 255], np.uint8)

    # white color

    white_lower = np.array([0, 0, 200], np.uint8)
    white_upper = np.array([180, 20, 255], np.uint8)

    # green color

    green_lower = np.array([25, 52, 72], np.uint8)
    green_upper = np.array([102, 255, 255], np.uint8)

    # black color

    black_lower = np.array([0, 0, 0], np.uint8)
    black_upper = np.array([180, 255, 30], np.uint8)

    # all color together

    red = cv2.inRange(hsv, red_lower, red_upper)
    blue = cv2.inRange(hsv, blue_lower, blue_upper)
    yellow = cv2.inRange(hsv, yellow_lower, yellow_upper)
    white = cv2.inRange(hsv, white_lower, white_upper)
    green = cv2.inRange(hsv, green_lower, green_upper)
    black = cv2.inRange(hsv, black_lower, black_upper)

    # Morphological Transform, Dilation

    kernal = np.ones((5, 5), "uint8")

    red = cv2.dilate(red, kernal)
    res_red = cv2.bitwise_and(result, result, mask=red)

    blue = cv2.dilate(blue, kernal)
    res_blue = cv2.bitwise_and(result, result, mask=blue)

    yellow = cv2.dilate(yellow, kernal)
    res_yellow = cv2.bitwise_and(result, result, mask=yellow)

    # white = cv2.dilate(white, kernal)
    # res_white = cv2.bitwise_and(result, result, mask=white)

    green = cv2.dilate(green, kernal)
    res_green = cv2.bitwise_and(result, result, mask=green)

    black = cv2.dilate(black, kernal)
    res_black = cv2.bitwise_and(result, result, mask=black)

    # Tracking red
    (contours, hierarchy) = cv2.findContours(
        red, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if(area > 700):
            x, y, w, h = cv2.boundingRect(contour)
            result = cv2.rectangle(
                result, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(result, "Red Colour", (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255))  # BGR

    # Tracking blue
    (contours, hierarchy) = cv2.findContours(
        blue, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if(area > 700):
            x, y, w, h = cv2.boundingRect(contour)
            result = cv2.rectangle(
                result, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(result, "Blue Colour", (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0))

    # Tracking yellow
    (contours, hierarchy) = cv2.findContours(
        yellow, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if(area > 700):
            x, y, w, h = cv2.boundingRect(contour)
            result = cv2.rectangle(
                result, (x, y), (x + w, y + h), (255, 242, 0), 2)
            cv2.putText(result, "Yellow Colour", (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0))

    # Tracking white
    # (contours, hierarchy) = cv2.findContours(
    #     white, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # for pic, contour in enumerate(contours):
    #     area = cv2.contourArea(contour)
    #     if(area > 700):
    #         x, y, w, h = cv2.boundingRect(contour)
    #         result = cv2.rectangle(
    #             result, (x, y), (x + w, y + h), (255, 255, 255), 2)
    #         cv2.putText(result, "White Colour", (x, y),
    #                     cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255))

    # Tracking green
    (contours, hierarchy) = cv2.findContours(
        green, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if(area > 700):
            x, y, w, h = cv2.boundingRect(contour)
            result = cv2.rectangle(
                result, (x, y), (x + w, y + h), (25, 255, 0), 2)
            cv2.putText(result, "Green Colour", (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255))

    # Tracking black
    (contours, hierarchy) = cv2.findContours(
        black, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if(area > 700):
            x, y, w, h = cv2.boundingRect(contour)
            result = cv2.rectangle(
                result, (x, y), (x + w, y + h), (0, 0, 0), 2)
            cv2.putText(result, "Black Colour", (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0))

    cv2.imshow("Color Tracking", result)
    ##

    cv2.imshow("Frame", frame)
    # cv2.imshow("Perspective transformation", result)

    key = cv2.waitKey(1)
    if key == 27:
        break


cap.release()
cv2.destroyAllWindows()
