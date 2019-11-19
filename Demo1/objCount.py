import cv2
import numpy as np

# Capture the input frame from webcam


cap = cv2.VideoCapture(0)
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

    cv2.imshow("Frame", frame)
    # cv2.imshow("Perspective transformation", result)

    key = cv2.waitKey(1)
    if key == 27:
        break

    # Convert the HSV colorspace
    hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)

    # Define 'blue' range in HSV colorspace
    lower = np.array([60, 100, 100])
    upper = np.array([180, 255, 255])

    # # Define 'blue' range in HSV colorspace
    # lower = np.array([60, 100, 100])
    # upper = np.array([180, 255, 255])

    # Threshold the HSV image to get only blue color
    mask = cv2.inRange(hsv, lower, upper)

    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(result, result, mask=mask)
    res = cv2.medianBlur(res, 5)

    cv2.imshow('Original image', result)
    cv2.imshow('Color Detector', res)

    # Check if the user pressed ESC key
    c = cv2.waitKey(5)
    if c == 27:
        break

cap.release()
cv2.destroyAllWindows()
