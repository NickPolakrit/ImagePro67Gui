import numpy as np
import cv2


cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1000)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 760)

cv2.namedWindow('xyPosition')
cv2.createTrackbar('X1', 'xyPosition', 0, 1022, nothing)
cv2.createTrackbar('Y1', 'xyPosition', 0, 575, nothing)
cv2.createTrackbar('X2', 'xyPosition', 0, 1022, nothing)
cv2.createTrackbar('Y2', 'xyPosition', 0, 575, nothing)
cv2.createTrackbar('X3', 'xyPosition', 0, 1022, nothing)
cv2.createTrackbar('Y3', 'xyPosition', 0, 575, nothing)
cv2.createTrackbar('X4', 'xyPosition', 0, 1022, nothing)
cv2.createTrackbar('Y4', 'xyPosition', 0, 575, nothing)


while True:
    _, frame = cap.read()

    x1 = cv2.getTrackbarPos('X1', 'xyPosition')
    y1 = cv2.getTrackbarPos('Y2', 'xyPosition')
    x2 = cv2.getTrackbarPos('X2', 'xyPosition')
    y2 = cv2.getTrackbarPos('Y2', 'xyPosition')
    x3 = cv2.getTrackbarPos('X3', 'xyPosition')
    y3 = cv2.getTrackbarPos('Y3', 'xyPosition')
    x4 = cv2.getTrackbarPos('X4', 'xyPosition')
    y4 = cv2.getTrackbarPos('X4', 'xyPosition')

    cv2.circle(frame, (x1, y1), 5, (0, 0, 255), -1)
    cv2.circle(frame, (x2, y2), 5, (0, 0, 255), -1)
    cv2.circle(frame, (x3, y3), 5, (0, 0, 255), -1)
    cv2.circle(frame, (994, 533), 5, (0, 0, 255), -1)
    pts1 = np.float32([[x1, y1],
                       [x2, y2],
                       [x3, y3],
                       [x4, y4]])
    pts2 = np.float32([[0, 0], [500, 0], [0, 500], [500, 500]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)

    result = cv2.warpPerspective(frame, matrix, (500, 500))

    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    Bblurred = cv2.bilateralFilter(gray, 9, 75, 75)
    Mblurred = cv2.medianBlur(Bblurred, 5)
    cv2.imshow("Gray", gray)

    # detect edges in the image
    edged = cv2.Canny(Mblurred, 10, 250)
    cv2.imshow("Edged", edged)

    # construct and apply a closing kernel to 'close' gaps between 'white'
    # pixels
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
    cv2.imshow("Closed", closed)

    # find contours (i.e. the 'outlines') in the image and initialize the
    # total number of books found
    (cnts, _) = cv2.findContours(closed.copy(),
                                 cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    total = 0

    # loop over the contours
    for c in cnts:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

    if len(approx) == 4:
        cv2.drawContours(result, [approx], -1, (0, 255, 0), 4)
        total += 1

    cv2.imshow("Frame", frame)
    cv2.imshow("Perspective transformation", result)

    key = cv2.waitKey(1)
    if key == 27:
        break


cap.release()
cv2.destroyAllWindows()
