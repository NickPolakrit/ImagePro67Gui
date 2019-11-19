
from transform import four_point_transform
# from __future__ import print_function
import imutils
from imutils import perspective
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2
import time


cam = cv2.VideoCapture(1)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1000)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 760)

# cv2.namedWindow("test")


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

cv2.setTrackbarPos("X1", "xyPosition", 249)
cv2.setTrackbarPos("Y1", "xyPosition", 14)
cv2.setTrackbarPos("X2", "xyPosition", 753)
cv2.setTrackbarPos("Y2", "xyPosition", 24)
cv2.setTrackbarPos("X3", "xyPosition", 176)
cv2.setTrackbarPos("Y3", "xyPosition", 546)
cv2.setTrackbarPos("X4", "xyPosition", 798)
cv2.setTrackbarPos("Y4", "xyPosition", 549)

stateWork = 1

while True:
    ret, frame = cam.read()
    ret, frame1 = cam.read()


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

    resultWarp = cv2.warpPerspective(frame1, matrix, (500, 500))

    
    cv2.imshow("original", frame1)
    cv2.imshow("circle", frame)
    # cv2.imshow("Warp", resultWarp)

    Gblurred = cv2.GaussianBlur(resultWarp, (5, 5), 0)
    Mblurred = cv2.medianBlur(Gblurred, 5)
    hsv_frame2 = cv2.cvtColor(Mblurred, cv2.COLOR_BGR2HSV)

    # time.sleep(10)
    # Card
    cardCount = 0
    low = np.array([0, 20, 101])
    high = np.array([180, 255, 255])
    mask = cv2.inRange(hsv_frame2, low, high)
    result = cv2.bitwise_and(resultWarp, resultWarp, mask=mask)
    
    contoursCard, _ = cv2.findContours(
        mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # try:
    #     biggest_contoursCard = max(contoursCard, key=cv2.contourArea)
    #     (x, y, w, h) = cv2.boundingRect(biggest_contoursCard)
    #     # cv2.rectangle(resultWarp, (x, y), (x+w, y+h), (0, 0, 0), 2)

    #     cv2.rectangle(result, (x, y), (x+w, y+h), (0, 0, 255), 2)
    #     Mcard = cv2.moments(biggest_contoursCard)
    #     cX = int(Mcard["m10"] / Mcard["m00"])
    #     cY = int(Mcard["m01"] / Mcard["m00"])
    #     cv2.circle(result, (cX, cY), 5, (0, 0, 255), -1)
    # except:
    #     pass

    
    cv2.imshow("card", result)

    for cnt1 in contoursCard:
        area12 = cv2.contourArea(cnt1)
        approx = cv2.approxPolyDP(cnt1, 0.02*cv2.arcLength(cnt1, True), True)
        # x = approx.ravel()[0]
        # y = approx.ravel()[1]
        # biggest_contoursCard = max(contoursCard, key=cv2.contourArea)
        # (x, y, w, h) = cv2.boundingRect(biggest_contoursCard)
        # # cv2.rectangle(resultWarp, (x, y), (x+w, y+h), (0, 0, 0), 2)

        # cv2.rectangle(result, (x, y), (x+w, y+h), (0, 0, 255), 2)
        # Mcard = cv2.moments(biggest_contoursCard)
        # cX = int(Mcard["m10"] / Mcard["m00"])
        # cY = int(Mcard["m01"] / Mcard["m00"])
        # cv2.circle(result, (cX, cY), 5, (0, 0, 255), -1)

        # Mcard = cv2.moments(biggest_contoursCard)
    
        # print("area " + str(approx[0][0][0]))

        if approx[0][0][0] > 200:
            time.sleep(1)

            try:
                biggest_contoursCard = max(contoursCard, key=cv2.contourArea)
                (x, y, w, h) = cv2.boundingRect(biggest_contoursCard)
                # cv2.rectangle(resultWarp, (x, y), (x+w, y+h), (0, 0, 0), 2)

                # cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)
                # Mcard = cv2.moments(biggest_contoursCard)
                # cX = int(Mcard["m10"] / Mcard["m00"])
                # cY = int(Mcard["m01"] / Mcard["m00"])
                # cv2.circle(image, (cX, cY), 5, (0, 0, 255), -1)

                # Mcard = cv2.moments(biggest_contoursCard)
                time.sleep(1)
                cardCount = 1

            except:
                cardCount = 0
                pass

            # cv2.imshow("card", result)
            print("staetWork : "+str(stateWork) + " Card : " + str(cardCount))

            if cardCount == 1 and stateWork == 1:
                time.sleep(1)
                img_name = "opencv_frame_0.png"
                cv2.imwrite(img_name, resultWarp)
                # print("{} written!".format(img_name))
                print(img_name)
                # img_counter += 1

                cv2.imshow("Warp", resultWarp)

                img1 = cv2.imread('opencv_frame_0.png')

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
                gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (7, 7), 0)
                cv2.imshow("gray", gray)

                # perform edge detection, then perform a dilation + erosion to
                # close gaps in between object edges
                edged = cv2.Canny(gray, 30, 40)
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
                    # if cv2.contourArea(c) < 100:
                    #     continue

                    # compute the rotated bounding box of the contour, then
                    # draw the contours
                    box = cv2.minAreaRect(c)
                    box = cv2.cv.BoxPoints(
                        box) if imutils.is_cv2() else cv2.boxPoints(box)
                    box = np.array(box, dtype="int")
                    cv2.drawContours(image, [box], -1, (0, 255, 0), 2)

                    # show the original coordinates
                    print("Object #{}:".format(i + 1))
                    print(box[0][0])
                    print(box[0][1])
                    print(box[1][0])
                    print(box[1][1])
                    print(box[2][0])
                    print(box[2][1])
                    print(box[3][0])
                    print(box[3][1])

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
                    # print(rect.astype("int"))

                    print("")

                    # loop over the original points and draw them
                    for ((x, y), color) in zip(rect, colors):
                        cv2.circle(image, (int(x), int(y)), 5, color, -1)

                    # draw the object num at the top-left corner
                    # cv2.putText(image, "Object #{}".format(i + 1),
                    #         (int(rect[0][0] - 15), int(rect[0][1] - 15)),
                    #         cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

                    # show the image

                cv2.circle(image, (box[2][0], box[2][1]), 2, (0, 0, 255), -1)
                cv2.circle(image, (box[3][0], box[3][1]), 2, (0, 0, 255), -1)
                cv2.circle(image, (box[1][0], box[1][1]), 2, (0, 0, 255), -1)
                cv2.circle(image, (box[0][0], box[0][1]), 2, (0, 0, 255), -1)
                pts1 = np.float32([[box[2][0], box[2][1]],
                                [box[3][0], box[3][1]],
                                [box[1][0], box[1][1]],
                                [box[0][0], box[0][1]]])
                pts2 = np.float32([[0, 0], [500, 0], [0, 500], [500, 500]])
                matrix = cv2.getPerspectiveTransform(pts1, pts2)

                resultCrop = cv2.warpPerspective(img1, matrix, (500, 500))
                hsv_frame = cv2.cvtColor(resultCrop, cv2.COLOR_BGR2HSV)
                # ------ COLOR ------------
                # Red color
                low_red = np.array([0, 34, 0])
                high_red = np.array([12, 255, 255])
                red_mask = cv2.inRange(hsv_frame, low_red, high_red)
                red = cv2.bitwise_and(resultCrop, resultCrop, mask=red_mask)
                # Blue color
                low_blue = np.array([62, 11, 0])
                high_blue = np.array([180, 255, 255])
                blue_mask = cv2.inRange(hsv_frame, low_blue, high_blue)
                blue = cv2.bitwise_and(resultCrop, resultCrop, mask=blue_mask)

                # Green color
                low_green = np.array([26, 0, 0])
                high_green = np.array([105, 111, 213])
                green_mask = cv2.inRange(hsv_frame, low_green, high_green)
                green = cv2.bitwise_and(resultCrop, resultCrop, mask=green_mask)

                # Yellow color
                low_yellow = np.array([19, 49, 135])
                high_yellow = np.array([33, 255, 255])
                yellow_mask = cv2.inRange(hsv_frame, low_yellow, high_yellow)
                yellow = cv2.bitwise_and(resultCrop, resultCrop, mask=yellow_mask)

                # Black color
                low_black = np.array([133, 0, 0])
                high_black = np.array([180, 43, 147])
                black_mask = cv2.inRange(hsv_frame, low_black, high_black)
                black = cv2.bitwise_and(resultCrop, resultCrop, mask=black_mask)
                cv2.imshow("black", black)

                countRed = 0
                countBlue = 0
                countYellow = 0
                countGreen = 0
                countBlack = 0
                # # Contour Red
                contoursRed, _ = cv2.findContours(
                    red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                try:
                    biggest_contoursRed = max(contoursRed, key=cv2.contourArea)
                    (x, y, w, h) = cv2.boundingRect(biggest_contoursRed)
                    cv2.rectangle(resultCrop, (x, y), (x+w, y+h), (0, 0, 255), 2)
                    countRed = 1

                    Mred = cv2.moments(biggest_contoursRed)
                    rX = int(Mred["m10"] / Mred["m00"])
                    rY = int(Mred["m01"] / Mred["m00"])
                    cv2.circle(resultCrop, (rX, rY), 5, (0, 0, 255), -1)
                    cv2.putText(resultCrop, 'X :' + str(rX) + " Y :" + str(rY),
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
                    cv2.rectangle(resultCrop, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    countBlue = 1

                    Mblue = cv2.moments(biggest_contoursBlue)
                    bX = int(Mblue["m10"] / Mblue["m00"])
                    bY = int(Mblue["m01"] / Mblue["m00"])
                    cv2.circle(resultCrop, (bX, bY), 5, (255, 0, 0), -1)
                    cv2.putText(resultCrop, 'X :' + str(bX) + " Y :" + str(bY),
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
                    cv2.rectangle(resultCrop, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    countGreen = 1

                    Mgreen = cv2.moments(biggest_contoursGreen)
                    gX = int(Mgreen["m10"] / Mgreen["m00"])
                    gY = int(Mgreen["m01"] / Mgreen["m00"])
                    cv2.circle(resultCrop, (gX, gY), 5, (0, 255, 0), -1)
                    cv2.putText(resultCrop, 'X :' + str(gX) + " Y :" + str(gY),
                                # bottomLeftCornerOfText
                                (gX + 30, gY),
                                cv2.FONT_HERSHEY_SIMPLEX,  # font
                                0.55,                      # fontScale
                                (0, 255, 0),            # fontColor
                                1)

                    pGx = gX*0.8
                    pGy = gY*0.8
                    cv2.circle(resultCrop, (pGx, pGy), 5, (0, 255, 0), -1)

                except:
                    pass

                contoursYellow, _ = cv2.findContours(
                    yellow_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

                try:
                    biggest_contoursYellow = max(
                        contoursYellow, key=cv2.contourArea)
                    (x, y, w, h) = cv2.boundingRect(biggest_contoursYellow)
                    cv2.rectangle(resultCrop, (x, y), (x+w, y+h), (0, 255, 255), 2)
                    countYellow = 1

                    Myellow = cv2.moments(biggest_contoursYellow)
                    yX = int(Myellow["m10"] / Myellow["m00"])
                    yY = int(Myellow["m01"] / Myellow["m00"])
                    cv2.circle(resultCrop, (yX, yY), 5, (0, 255, 255), -1)
                    cv2.putText(resultCrop, 'X :' + str(yX) + " Y :" + str(yY),
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
                    cv2.rectangle(resultCrop, (x, y), (x+w, y+h), (0, 0, 0), 2)
                    countBlack = 1

                    Mblack = cv2.moments(biggest_contoursBlack)
                    blX = int(Mblack["m10"] / Mblack["m00"])
                    blY = int(Mblack["m01"] / Mblack["m00"])
                    cv2.circle(resultCrop, (blX, blY), 5, (0, 0, 0), -1)
                    cv2.putText(resultCrop, 'X :' + str(blX) + " Y :" + str(blY),
                                # bottomLeftCornerOfText
                                (blX + 30, blY),
                                cv2.FONT_HERSHEY_SIMPLEX,  # font
                                0.55,                      # fontScale
                                (0, 0, 0),            # fontColor
                                1)

                except:
                    pass

                # Draw Text
                cv2.rectangle(resultCrop, (0, 0), (100, 110), (0, 0, 0), -1)
                cv2.putText(resultCrop, 'Red    :' + str(countRed),
                            (10, 20),                  # bottomLeftCornerOfText
                            cv2.FONT_HERSHEY_SIMPLEX,  # font
                            0.5,                      # fontScale
                            (0, 0, 255),            # fontColor
                            1)                        # lineType
                cv2.putText(resultCrop, 'Yellow :' + str(countYellow),
                            (10, 40),                  # bottomLeftCornerOfText
                            cv2.FONT_HERSHEY_SIMPLEX,  # font
                            0.5,                      # fontScale
                            (0, 255, 255),            # fontColor
                            1)
                cv2.putText(resultCrop, 'Blue   :' + str(countBlue),
                            (10, 60),                  # bottomLeftCornerOfText
                            cv2.FONT_HERSHEY_SIMPLEX,  # font
                            0.5,                      # fontScale
                            (255, 238, 0),            # fontColor
                            1)
                cv2.putText(resultCrop, 'Green  :' + str(countGreen),
                            (10, 80),                  # bottomLeftCornerOfText
                            cv2.FONT_HERSHEY_SIMPLEX,  # font
                            0.5,                      # fontScale
                            (0, 255, 0),            # fontColor
                            1)
                cv2.putText(resultCrop, 'Black  :' + str(countBlack),
                            (10, 100),                  # bottomLeftCornerOfText
                            cv2.FONT_HERSHEY_SIMPLEX,  # font
                            0.5,                      # fontScale
                            (255, 255, 255),            # fontColor
                            1)

                # if not ret:
                # break
                # k = cv2.waitKey(1)
                # if k % 256 == 27:
                #     # ESC pressed
                #     print("Escape hit, closing...")
                #     break
                # elif k % 256 == 32:
                #     print("spacebar...")

                # time.sleep(10)
                cardCount = 0
                stateWork = 0

                cv2.imshow("Image", image)
                cv2.imshow("Crop", resultCrop)
                cv2.imshow("img1", img1)
            

            else:
                pass

        k = cv2.waitKey(1)

        if k % 256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break
        elif k % 256 == 32:
            print("spacebar Reset...")
            cv2.putText(frame, 'Reset...',
                                    # bottomLeftCornerOfText
                                    (100, 200),
                                    cv2.FONT_HERSHEY_SIMPLEX,  # font
                                    2,                      # fontScale
                                    (0, 255, 0),            # fontColor
                                    2)
            stateWork = 1



    
        
        

        # SPACE pressed
        
        # continue
    
    else:
        pass
        # continue

    # if not ret:
    #     break
    

    #     # SPACE pressed
        
    #     break
    
    
    
    
    # cv2.waitKey(0)
    



# print("finish") was 

cam.release()
cv2.destroyAllWindows()
       
                
        


        
        





