from transform import four_point_transform
import cv2
import numpy as np
import imutils


def nothing(x):
    # any operation
    pass


cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)

l_h = 6
l_s = 20
l_v = 0
u_h = 180
u_s = 255
u_v = 255

FOUND_CARD = False


stateWork = 1
font = cv2.FONT_HERSHEY_COMPLEX

while True:
    _, frame = cap.read()
    # _, original = cap.read()

    

    x1 = 249
    y1 = 14
    x2 = 753
    y2 = 24
    x3 = 176
    y3 = 546
    x4 = 798
    y4 = 549

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
    resultOut = cv2.warpPerspective(frame, matrix, (500, 500))

    cv2.imshow("Original", frame)

    
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
            # cv2.drawContours(resultWarp, [approx], 0, (0, 0, 0), 5)

            if len(approx) == 4 and stateWork == 1:
                cv2.putText(mask, "CARD", (x, y-20), font, 1, (0, 0, 0))
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
                cv2.imshow("Your_CARD", Crop_card)
                img_name = "crop_card.png"
                cv2.imwrite(img_name, Crop_card)

                # cardCount = 0
                stateWork = 0
                


            else:
                pass

            print(" Loop staetWork : "+str(stateWork) + " Card : " + str(cardCount))

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

    print("staetWork : "+str(stateWork) + " Card : " + str(cardCount))

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
