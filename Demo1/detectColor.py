# import the necessary packages
import numpy as np
import argparse
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", help="path to the image")
args = vars(ap.parse_args())

# load the image
image = cv2.imread('./test.jpg')

# define the list of boundaries
boundaries = {
    "red": ([17, 15, 100], [50, 56, 200]),  # เข้ม อ่อน
    "blue":  ([105, 45, 0], [255, 165, 90]),
    "yellow":  ([34, 121, 173], [109, 156, 180]),
    "green": ([49, 104, 20], [127, 163, 115])
}

# loop over the boundaries
for color_name, (lower, upper) in boundaries.items():
    # create NumPy arrays from the boundaries
    lower = np.array(lower, dtype="uint8")
    upper = np.array(upper, dtype="uint8")

    # find the colors within the specified boundaries and apply
    # the mask
    mask = cv2.inRange(image, lower, upper)
    output = cv2.bitwise_and(image, image, mask=mask)
    if mask.any():
        print(f"{color_name}: {mask.sum()}")

    # show the images
    cv2.imshow("images", np.hstack([image, output]))
    cv2.waitKey(0)
