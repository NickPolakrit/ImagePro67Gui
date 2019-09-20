import cv2
import numpy as np

cap = cv2.VideoCapture('Video3.avi', 0)
cap1 = cv2.VideoCapture('Video4.avi', 0)

while(cap.isOpened()):

    ret, frame = cap.read()
    ret1, frame1 = cap1.read()
    if ret == True and ret1 == True:  # you *have* to check *both* captures !
        h, w, c = frame.shape
        h1, w1, c1 = frame1.shape

        if h != h1 or w != w1:  # resize right img to left size
            frame1 = cv2.resize(frame1, (w, h))

        both = np.concatenate((frame, frame1), axis=1)
        # or like this:
        # both = cv2.hconcat([frame, frame1])

    else:
        break

cap.release()
out.release()


cv2.waitKey(0)
cv2.destroyAllWindows()
