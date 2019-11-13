

from __future__ import print_function
import imutils
from imutils import perspective
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2
import time


cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1000)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 760)

cv2.namedWindow("test")


def nothing(x):  # for trackbar
    pass

img_counter = 0

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



while True:
    ret, frame = cam.read()

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

    cv2.imshow("Warp", resultWarp)
    cv2.imshow("test", frame)

    if not ret:
        break
    k = cv2.waitKey(1)

    if k % 256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k % 256 == 32:
        # SPACE pressed
        img_name = "opencv_frame_{}.png".format(img_counter)
        cv2.imwrite(img_name, resultWarp)
        print("{} written!".format(img_name))
        print(img_name)
        img_counter += 1

        time.sleep(2)

        img1 = cv2.imread('opencv_frame_0.png')
        cv2.imshow("shotimg", img1)

        def order_points_old(pts):
            rect = np.zeros((4, 2), dtype="float32")


            s = pts.sum(axis=1)
            rect[0] = pts[np.argmin(s)]
            rect[2] = pts[np.argmax(s)]

            diff = np.diff(pts, axis=1)
            rect[1] = pts[np.argmin(diff)]
            rect[3] = pts[np.argmax(diff)]

            return rect

        ap = argparse.ArgumentParser()
        ap.add_argument("-n", "--new", type=int, default=-1,
                        help="whether or not the new order points should should be used")
        args = vars(ap.parse_args())

        # load our input image, convert it to grayscale, and blur it slightly
        image = cv2.imread("opencv_frame_0.png")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        # perform edge detection, then perform a dilation + erosion to
        # close gaps in between object edges
        edged = cv2.Canny(gray, 10, 40)
        edged = cv2.dilate(edged, None, iterations=1)
        edged = cv2.erode(edged, None, iterations=1)
        cv2.imshow("edge", edged)


        # find contours in the edge map
        cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        # sort the contours from left-to-right and initialize the bounding box
        # point colors
        (cnts, _) = contours.sort_contours(cnts)
        colors = ((0, 0, 255), (240, 0, 159), (255, 0, 0), (255, 255, 0))
        for (i, c) in enumerate(cnts):
        	# if the contour is not sufficiently large, ignore it
            if cv2.contourArea(c) < 100:
                continue

            # compute the rotated bounding box of the contour, then
            # draw the contours
            box = cv2.minAreaRect(c)
            box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
            box = np.array(box, dtype="int")
            cv2.drawContours(image, [box], -1, (0, 255, 0), 2)

            # show the original coordinates
            print("Object #{}:".format(i + 1))
            print(box)

            # order the points in the contour such that they appear
            # in top-left, top-right, bottom-right, and bottom-left
            # order, then draw the outline of the rotated bounding
            # box
            rect = order_points_old(box)

            # check to see if the new method should be used for
            # ordering the coordinates
            if args["new"] > 0:
                rect = perspective.order_points(box)

            # show the re-ordered coordinates
            print(rect.astype("int"))
            print("")

            # loop over the original points and draw them
            for ((x, y), color) in zip(rect, colors):
                cv2.circle(image, (int(x), int(y)), 5, color, -1)

            # draw the object num at the top-left corner
            cv2.putText(image, "Object #{}".format(i + 1),
                    (int(rect[0][0] - 15), int(rect[0][1] - 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

            # show the image
            cv2.imshow("Image", image)
            cv2.waitKey(0)
                
        


        
        


cam.release()

cv2.destroyAllWindows()


