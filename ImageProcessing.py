import cv2
import numpy as np

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

    result = cv2.warpPerspective(frame, matrix, (500, 500))

    cv2.imshow("Frame", frame)
    cv2.imshow("Perspective transformation", result)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
