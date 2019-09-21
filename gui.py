import cv2
import numpy as np

cap = cv2.VideoCapture(0)
cap1 = cv2.VideoCapture(0)

# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1000)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 760)

cap.set(3, 640)
cap.set(4, 480)

while(cap.isOpened()):

    ret, frame = cap.read()
    # ret1, frame1 = cap1.read()
    if ret == True:

        both = np.concatenate((frame, frame), axis=1)
        # both = np.concatenate((frame, frame))
        # both = np.column_stack((frame, frame))

        cv2.imshow('Frame', both)
        # out.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    else:
        break

cap.release()
out.release()


cv2.waitKey(0)
cv2.destroyAllWindows()
