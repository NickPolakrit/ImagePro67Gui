from pyimagesearch.transform import four_point_transform
import cv2
import numpy as np
import imutils

def nothing(x):
    # any operation
    pass


cap = cv2.VideoCapture("VDO/X.mp4")
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)

l_h = 66
l_s = 0
l_v = 53
u_h = 168
u_s = 49
u_v = 255
FOUND_CARD = False

font = cv2.FONT_HERSHEY_COMPLEX

while True:
    _, frame = cap.read()
    frame =  cv2.resize(frame,(int(800),int(600)))
    original = frame.copy()
    Mblurred = cv2.medianBlur(frame, 7)
    hsv = cv2.cvtColor(Mblurred, cv2.COLOR_BGR2HSV)


    lower_red = np.array([l_h, l_s, l_v])
    upper_red = np.array([u_h, u_s, u_v])

    mask = cv2.inRange(hsv, lower_red, upper_red)
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
            cv2.drawContours(frame, [approx], 0, (0, 0, 0), 5)
            
            if len(approx) == 4:
                cv2.putText(mask, "CARD", (x, y-20), font, 1, (0, 0, 0))
                FOUND_CARD = True

                if FOUND_CARD == True:
                    cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                    cnts = imutils.grab_contours(cnts)
                    cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]
                    for c in cnts:		
                        peri = cv2.arcLength(c, True)
                        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
                        
                        if len(approx) == 4:
                            Approx = approx
                            break
                Outline = cv2.drawContours(original.copy(), [Approx], -5, (0, 0, 255), 2)
                ratio = Outline.shape[0] / 600.0 
                Your_CARD = four_point_transform(original, Approx.reshape(4, 2) * ratio)
                Your_CARD =  cv2.resize(Your_CARD,(int(600),int(600)))
                cv2.imshow("Outline", Outline)
                cv2.imshow("Your_CARD", Your_CARD)
                img_name = "img_CARD.png"
                cv2.imwrite(img_name, Your_CARD)
                cap.release()
                cv2.destroyAllWindows()
                
                            

            
            


    cv2.imshow("edged", edged)
    
    cv2.imshow("Mask", mask)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()