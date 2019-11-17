from transform import four_point_transform
import cv2
import numpy as np
import imutils
import time
import serial
import struct 


def nothing(x):
    # any operation
    pass


cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1000)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 760)

l_h = 6
l_s = 20
l_v = 0
u_h = 180
u_s = 255
u_v = 255

x1 = 249
y1 = 14
x2 = 753
y2 = 24
x3 = 176
y3 = 546
x4 = 798
y4 = 549

r_lh = 0
r_ls = 43
r_lv = 40
r_uh = 12
r_us = 255
r_uv = 255

# serialPort = serial.Serial(
#     "/dev/cu.usbserial-AC00YIZF", 115200, 8, 'N', 1, 0, 0, 0, 0, 0)

# serialPort.setRTS(0)
# serialPort.setDTR(0)


FOUND_CARD = False


stateWork = 1
font = cv2.FONT_HERSHEY_COMPLEX

while True:
    _, frame = cap.read()
    _, original = cap.read()

    

    

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

    resultWarp = cv2.warpPerspective(original, matrix, (500, 500))
    resultOut = cv2.warpPerspective(original, matrix, (500, 500))

    cv2.imshow("Frame", frame)

    
    cardCount = 0

    Mblurred = cv2.medianBlur(resultWarp, 5)
    hsv = cv2.cvtColor(Mblurred, cv2.COLOR_BGR2HSV)

    lower_crop = np.array([l_h, l_s, l_v])
    upper_crop = np.array([u_h, u_s, u_v])

    mask = cv2.inRange(hsv, lower_crop, upper_crop)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel)

    edged = cv2.Canny(mask, 50, 220)

    contours, _ = cv2.findContours(
        mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        approx = cv2.approxPolyDP(cnt, 0.02*cv2.arcLength(cnt, True), True)
        x = approx.ravel()[0]
        y = approx.ravel()[1]

        if 15000 > area > 5000:
            time.sleep(1)
            # cv2.drawContours(resultWarp, [approx], 0, (0, 0, 0), 5)

            if len(approx) == 4 and stateWork == 1:
                time.sleep(1)
                # cv2.putText(mask, "CARD", (x, y-20), font, 1, (0, 0, 0))
                FOUND_CARD = True
                cardCount = 1

                
                cnts = cv2.findContours(
                    edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                cnts = imutils.grab_contours(cnts)
                cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
                # for c in cnts:
                #     peri = cv2.arcLength(c, True)
                #     approx = cv2.approxPolyDP(c, 0.02 * peri, True)
                #     if len(approx) == 4:
                #         Approx = approx
                #         break

                Approx = approx
                Outline = cv2.drawContours(
                    resultOut.copy(), [Approx], -5, (0, 0, 255), 1)
                ratio = Outline.shape[0] / 500
                Crop_card = four_point_transform(
                    resultWarp, Approx.reshape(4, 2) * ratio)
                Crop_card = cv2.resize(Crop_card, (int(500), int(500)))
                cv2.imshow("Outline", Outline)
                cv2.imshow("warp crop", resultWarp)
                cv2.imshow("Your_CARD", Crop_card)
                img_name = "crop_card.png"
                cv2.imwrite(img_name, Crop_card)

                imgCrop = cv2.imread("crop_card.png")

                # cardCount = 0
                stateWork = 0
                # ------ COLOR ------------
                

                cropBlur = cv2.medianBlur(imgCrop, 5)
                hsv_frame = cv2.cvtColor(cropBlur, cv2.COLOR_BGR2HSV)

                # Blackground color
                low_bg = np.array([0, 0, 150])
                high_bg = np.array([34, 74, 255])
                bg_mask = cv2.inRange(hsv_frame, low_bg, high_bg)
                bg = cv2.bitwise_and(
                    imgCrop, imgCrop, mask=bg_mask)
                cv2.imshow("bg", bg)

                # Red color
                low_red = np.array([0, 43, 40])
                high_red = np.array([12, 255, 255])
                red_mask = cv2.inRange(hsv_frame, low_red, high_red)
                red = cv2.bitwise_and(imgCrop, imgCrop, mask=red_mask)
                # Blue color
                low_blue = np.array([62, 11, 0])
                high_blue = np.array([180, 255, 255])
                blue_mask = cv2.inRange(hsv_frame, low_blue, high_blue)
                blue = cv2.bitwise_and(imgCrop, imgCrop, mask=blue_mask)

                # Green color
                low_green = np.array([26, 0, 0])
                high_green = np.array([105, 111, 213])
                green_mask = cv2.inRange(hsv_frame, low_green, high_green)
                green = cv2.bitwise_and(
                    imgCrop, imgCrop, mask=green_mask)

                # Yellow color
                low_yellow = np.array([24, 72, 0])
                high_yellow = np.array([30, 255, 255])
                yellow_mask = cv2.inRange(hsv_frame, low_yellow, high_yellow)
                yellow = cv2.bitwise_and(
                    imgCrop, imgCrop, mask=yellow_mask)

                # Black color
                low_black = np.array([133, 0, 0])
                high_black = np.array([180, 43, 147])
                black_mask = cv2.inRange(hsv_frame, low_black, high_black)
                black = cv2.bitwise_and(
                    imgCrop, imgCrop, mask=black_mask)
                cv2.imshow("red", red)

                

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
                    cv2.rectangle(imgCrop, (x, y),
                                  (x+w, y+h), (0, 0, 255), 2)
                    countRed = 1

                    Mred = cv2.moments(biggest_contoursRed)
                    rX = int(Mred["m10"] / Mred["m00"])
                    rY = int(Mred["m01"] / Mred["m00"])
                    cv2.circle(imgCrop, (rX, rY), 5, (0, 0, 255), -1)
                    cv2.putText(imgCrop, 'X :' + str(rX) + " Y :" + str(rY),
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
                    biggest_contoursBlue = max(
                        contoursBlue, key=cv2.contourArea)
                    (x, y, w, h) = cv2.boundingRect(biggest_contoursBlue)
                    cv2.rectangle(imgCrop, (x, y),
                                  (x+w, y+h), (255, 0, 0), 2)
                    countBlue = 1

                    Mblue = cv2.moments(biggest_contoursBlue)
                    bX = int(Mblue["m10"] / Mblue["m00"])
                    bY = int(Mblue["m01"] / Mblue["m00"])
                    cv2.circle(imgCrop, (bX, bY), 5, (255, 0, 0), -1)
                    cv2.putText(imgCrop, 'X :' + str(bX) + " Y :" + str(bY),
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
                    biggest_contoursGreen = max(
                        contoursGreen, key=cv2.contourArea)
                    (x, y, w, h) = cv2.boundingRect(biggest_contoursGreen)
                    cv2.rectangle(imgCrop, (x, y),
                                  (x+w, y+h), (0, 255, 0), 2)
                    countGreen = 1

                    Mgreen = cv2.moments(biggest_contoursGreen)
                    gX = int(Mgreen["m10"] / Mgreen["m00"])
                    gY = int(Mgreen["m01"] / Mgreen["m00"])
                    cv2.circle(imgCrop, (gX, gY), 5, (0, 255, 0), -1)
                    cv2.putText(imgCrop, 'X :' + str(gX) + " Y :" + str(gY),
                                # bottomLeftCornerOfText
                                (gX + 30, gY),
                                cv2.FONT_HERSHEY_SIMPLEX,  # font
                                0.55,                      # fontScale
                                (0, 255, 0),            # fontColor
                                1)

                    pGx = gX*0.8
                    pGy = gY*0.8
                    cv2.circle(imgCrop, (pGx, pGy), 5, (0, 255, 0), -1)

                except:
                    pass

                contoursYellow, _ = cv2.findContours(
                    yellow_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

                try:
                    biggest_contoursYellow = max(
                        contoursYellow, key=cv2.contourArea)
                    (x, y, w, h) = cv2.boundingRect(biggest_contoursYellow)
                    cv2.rectangle(imgCrop, (x, y),
                                  (x+w, y+h), (0, 255, 255), 2)
                    countYellow = 1

                    Myellow = cv2.moments(biggest_contoursYellow)
                    yX = int(Myellow["m10"] / Myellow["m00"])
                    yY = int(Myellow["m01"] / Myellow["m00"])
                    cv2.circle(imgCrop, (yX, yY), 5, (0, 255, 255), -1)
                    cv2.putText(imgCrop, 'X :' + str(yX) + " Y :" + str(yY),
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
                    biggest_contoursBlack = max(
                        contoursBlack, key=cv2.contourArea)
                    (x, y, w, h) = cv2.boundingRect(biggest_contoursBlack)
                    cv2.rectangle(imgCrop, (x, y), (x+w, y+h), (0, 0, 0), 2)
                    countBlack = 1

                    Mblack = cv2.moments(biggest_contoursBlack)
                    blX = int(Mblack["m10"] / Mblack["m00"])
                    blY = int(Mblack["m01"] / Mblack["m00"])
                    cv2.circle(imgCrop, (blX, blY), 5, (0, 0, 0), -1)
                    cv2.putText(imgCrop, 'X :' + str(blX) + " Y :" + str(blY),
                                # bottomLeftCornerOfText
                                (blX + 30, blY),
                                cv2.FONT_HERSHEY_SIMPLEX,  # font
                                0.55,                      # fontScale
                                (0, 0, 0),            # fontColor
                                1)


                

                except:
                    pass

                cv2.imshow("IMAGE", imgCrop)
                
                Send = [int(rX/2),int(rY/2),int(gX/2),int(gY/2),int(bX/2),int(bY/2),int(yX/2),int(yY/2),int(blX/2),int(blY/2)]
                print(Send)
                # serialPort.write(Send)
                # print(serialPort.readline())
                # totalall = [1]

                # for i in Send:
                #     time.sleep(0.1)
                #     c = struct.pack('B', i)
                #     serialPort.write(c)
                #     print(i)

                # r g b y black

                


            else:
                pass

            # print(" Loop staetWork : "+str(stateWork) + " Card : " + str(cardCount))

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
        
        else:
            pass


    cv2.imshow("edged", edged)

    cv2.imshow("Mask", mask)

    # print("staetWork : "+str(stateWork) + " Card : " + str(cardCount))
    

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
