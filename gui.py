import sys
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QApplication, QDialog
from PyQt5.uic import loadUi
import cv2
import numpy as np
import imutils
from transform import four_point_transform
import time
import serial
import struct
import subprocess


serialPIC = serial.Serial(
    "/dev/cu.usbserial-AC00YIZF", 115200, 8, 'N', 1, 0, 0, 0, 0, 0)

serialPIC.setRTS(0)
serialPIC.setDTR(0)

serialAD = serial.Serial(
    "/dev/cu.usbserial-14520", 115200, 8, 'N', 1, 0, 0, 0, 0, 0)

serialAD.setRTS(0)
serialAD.setDTR(0)


rX = 100
rY = 100
gX = 100
gY = 100
bX = 100
bY = 100
yX = 100
yY = 100
blX = 100
blY = 100



class OpencvImg(QDialog):
    def __init__(self):
        super(OpencvImg, self).__init__()
        loadUi('demo2gui.ui', self)
        self.image = None
        self.start_button.clicked.connect(self.start_webcam)
        self.stop_button.clicked.connect(self.stop_webcam)

        self.commandLinkButton.clicked.connect(self.start_card)
        
        
    def start_card(self):
        self.timer.timeout.connect(self.find_card)
        self.timer.start(5)


    def start_webcam(self):
        
        self.capture = cv2.VideoCapture(1)
        # self.capture = cap
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 300)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 400)
        
        self.x1_slider.setValue(549)
        self.y1_slider.setValue(41)
        self.x2_slider.setValue(153)
        self.y2_slider.setValue(36)
        self.x3_slider.setValue(587)
        self.y3_slider.setValue(427)
        self.x4_slider.setValue(123)
        self.y4_slider.setValue(418)

        self.card_lh.setValue(0)
        self.card_ls.setValue(12)
        self.card_lv.setValue(0)
        self.card_uh.setValue(179)
        self.card_us.setValue(255)
        self.card_uv.setValue(255)


        self.red_lh.setValue(0)
        self.red_ls.setValue(30)
        self.red_lv.setValue(6)
        self.red_uh.setValue(6)
        self.red_us.setValue(255)
        self.red_uv.setValue(255)


        self.green_lh.setValue(41)
        self.green_ls.setValue(21)
        self.green_lv.setValue(58)
        self.green_uh.setValue(93)
        self.green_us.setValue(255)
        self.green_uv.setValue(246)


        self.blue_lh.setValue(94)
        self.blue_ls.setValue(50)
        self.blue_lv.setValue(131)
        self.blue_uh.setValue(139)
        self.blue_us.setValue(255)
        self.blue_uv.setValue(255)


        self.yellow_lh.setValue(16)
        self.yellow_ls.setValue(46)
        self.yellow_lv.setValue(0)
        self.yellow_uh.setValue(30)
        self.yellow_us.setValue(255)
        self.yellow_uv.setValue(206)


        self.black_lh.setValue(90)
        self.black_ls.setValue(0)
        self.black_lv.setValue(0)
        self.black_uh.setValue(104)
        self.black_us.setValue(152)
        self.black_uv.setValue(255)
        # self.black_lh.setValue(0)
        # self.black_ls.setValue(0)
        # self.black_lv.setValue(0)
        # self.black_uh.setValue(179)
        # self.black_us.setValue(30)
        # self.black_uv.setValue(120)



        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(5)


    def update_frame(self):


        ret, self.image = self.capture.read()

        self.notCircle = self.image.copy()

        cv2.circle(self.image, (self.x1_slider.value(),
                                self.y1_slider.value()), 5, (0, 0, 255), -1)
        self.current_value.setText(
            'X1 Y1 -> :'+str(self.x1_slider.value()) + ' '+str(self.y1_slider.value()))
        cv2.circle(self.image, (self.x2_slider.value(),
                                self.y2_slider.value()), 5, (0, 0, 255), -1)
        self.current_value2.setText(
            'X2 Y2 -> :'+str(self.x2_slider.value()) + ' '+str(self.y2_slider.value()))
        cv2.circle(self.image, (self.x3_slider.value(),
                                self.y3_slider.value()), 5, (0, 0, 255), -1)
        self.current_value3.setText(
            'X3 Y3 -> :'+str(self.x3_slider.value()) + ' '+str(self.y3_slider.value()))
        cv2.circle(self.image, (self.x4_slider.value(),
                                self.y4_slider.value()), 5, (0, 0, 255), -1)
        self.current_value4.setText(
            'X4 Y4 -> :'+str(self.x4_slider.value()) + ' '+str(self.y4_slider.value()))

        pts1 = np.float32([[self.x1_slider.value(), self.y1_slider.value()],
                           [self.x2_slider.value(), self.y2_slider.value()],
                           [self.x3_slider.value(), self.y3_slider.value()],
                           [self.x4_slider.value(), self.y4_slider.value()]])
        pts2 = np.float32([[0, 0], [500, 0], [0, 500], [500, 500]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        resultWarp = cv2.warpPerspective(self.notCircle, matrix, (500, 500))
        self.imageWarp = resultWarp

        # self.image = cv2.flip(self.image, 1)
        self.imageWarp = cv2.flip(self.imageWarp, 1)
        # self.displayImage(self.image, self.imageWarp, 1)
        
        # Mblurred = cv2.medianBlur(self.imageWarp, 5)
        hsv = cv2.cvtColor(self.imageWarp, cv2.COLOR_BGR2HSV)

        color_lower = np.array([self.card_lh.value(), self.card_ls.value(), self.card_lv.value()], np.uint8)
        color_upper = np.array([self.card_uh.value(), self.card_us.value(), self.card_uv.value()], np.uint8)

        mask = cv2.inRange(hsv, color_lower, color_upper)
        kernel = np.ones((5, 5), np.uint8)
        

        red_lower = np.array(
            [self.red_lh.value(), self.red_ls.value(), self.red_lv.value()], np.uint8)
        red_upper = np.array(
            [self.red_uh.value(), self.red_us.value(), self.red_uv.value()], np.uint8)
        green_lower = np.array(
            [self.green_lh.value(), self.green_ls.value(), self.green_lv.value()], np.uint8)
        green_upper = np.array(
            [self.green_uh.value(), self.green_us.value(), self.green_uv.value()], np.uint8)
        blue_lower = np.array(
            [self.blue_lh.value(), self.blue_ls.value(), self.blue_lv.value()], np.uint8)
        blue_upper = np.array(
            [self.blue_uh.value(), self.blue_us.value(), self.blue_uv.value()], np.uint8)
        yellow_lower = np.array(
            [self.yellow_lh.value(), self.yellow_ls.value(), self.yellow_lv.value()], np.uint8)
        yellow_upper = np.array(
            [self.yellow_uh.value(), self.yellow_us.value(), self.yellow_uv.value()], np.uint8)
        black_lower = np.array(
            [self.black_lh.value(), self.black_ls.value(), self.black_lv.value()], np.uint8)
        black_upper = np.array(
            [self.black_uh.value(), self.black_us.value(), self.black_uv.value()], np.uint8)

        self.card_value.setText(
            'Current Value -> Min :'+str(color_lower)+' Max: '+str(color_upper))
        self.red_value.setText('Current Value -> Min :'+str(red_lower)+' Max: '+str(red_upper))
        self.green_value.setText('Current Value -> Min :'+str(green_lower)+' Max: '+str(green_upper))
        self.blue_value.setText('Current Value -> Min :'+str(blue_lower)+' Max: '+str(blue_upper))
        self.yellow_value.setText('Current Value -> Min :'+str(yellow_lower)+' Max: '+str(yellow_upper))
        self.black_value.setText('Current Value -> Min :'+str(black_lower)+' Max: '+str(black_upper))


        color_mask = cv2.inRange(hsv, color_lower, color_upper)
        card = cv2.bitwise_and(
            self.imageWarp, self.imageWarp, mask=color_mask)
        

        bFilterC2 = cv2.bilateralFilter(color_mask, 9, 75, 75)
        openingC2 = cv2.morphologyEx(bFilterC2, cv2.MORPH_OPEN, kernel)
        closingC2 = cv2.morphologyEx(openingC2, cv2.MORPH_CLOSE, kernel)

        bFilterC = cv2.bilateralFilter(card, 9, 75, 75)
        openingC = cv2.morphologyEx(bFilterC, cv2.MORPH_OPEN, kernel)
        closingC = cv2.morphologyEx(openingC, cv2.MORPH_CLOSE, kernel)
        mask = cv2.erode(closingC2, kernel)
        edged = cv2.Canny(mask, 50, 220)
        contours, _ = cv2.findContours(
            mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)



        red_mask = cv2.inRange(hsv, red_lower, red_upper)
        red = cv2.bitwise_and(
            self.imageWarp, self.imageWarp, mask=red_mask)
        bFilterR2 = cv2.bilateralFilter(red_mask, 9, 75, 75)
        dilationR2 = cv2.dilate(bFilterR2, kernel, iterations=1)
        openingR2 = cv2.morphologyEx(dilationR2, cv2.MORPH_OPEN, kernel)
        closingR2 = cv2.morphologyEx(openingR2, cv2.MORPH_CLOSE, kernel)
        
        bFilterR2 = cv2.bilateralFilter(red, 9, 75, 75)
        dilationR = cv2.dilate(bFilterR2, kernel, iterations=1)
        openingR = cv2.morphologyEx(dilationR, cv2.MORPH_OPEN, kernel)
        closingR = cv2.morphologyEx(openingR, cv2.MORPH_CLOSE, kernel)

        


        green_mask = cv2.inRange(hsv, green_lower, green_upper)
        green = cv2.bitwise_and(
            self.imageWarp, self.imageWarp, mask=green_mask)
        dilationG2 = cv2.dilate(green_mask, kernel, iterations=1)
        openingG2 = cv2.morphologyEx(dilationG2, cv2.MORPH_OPEN, kernel)
        closingG2 = cv2.morphologyEx(openingG2, cv2.MORPH_CLOSE, kernel)

        dilationG = cv2.dilate(green, kernel, iterations=1)
        openingG = cv2.morphologyEx(dilationG, cv2.MORPH_OPEN, kernel)
        closingG = cv2.morphologyEx(openingG, cv2.MORPH_CLOSE, kernel)

        blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)
        blue = cv2.bitwise_and(
            self.imageWarp, self.imageWarp, mask=blue_mask)
        dilationB2 = cv2.dilate(blue_mask, kernel, iterations=1)
        openingB2 = cv2.morphologyEx(dilationB2, cv2.MORPH_OPEN, kernel)
        closingB2 = cv2.morphologyEx(openingB2, cv2.MORPH_CLOSE, kernel)

        dilationB = cv2.dilate(blue, kernel, iterations=1)
        openingB = cv2.morphologyEx(dilationB, cv2.MORPH_OPEN, kernel)
        closingB = cv2.morphologyEx(openingB, cv2.MORPH_CLOSE, kernel)

        yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
        yellow = cv2.bitwise_and(
            self.imageWarp, self.imageWarp, mask=yellow_mask)
        dilationY2 = cv2.dilate(yellow_mask, kernel, iterations=1)
        openingY2 = cv2.morphologyEx(dilationY2, cv2.MORPH_OPEN, kernel)
        closingY2 = cv2.morphologyEx(openingY2, cv2.MORPH_CLOSE, kernel)
        
        dilationY = cv2.dilate(yellow, kernel, iterations=1)
        openingY = cv2.morphologyEx(dilationY, cv2.MORPH_OPEN, kernel)
        closingY = cv2.morphologyEx(openingY, cv2.MORPH_CLOSE, kernel)
        

        black_mask = cv2.inRange(hsv, black_lower, black_upper)
        black = cv2.bitwise_and(
            self.imageWarp, self.imageWarp, mask=black_mask)
        dilationbl2 = cv2.dilate(black_mask, kernel, iterations=1)
        openingBl2 = cv2.morphologyEx(dilationbl2, cv2.MORPH_OPEN, kernel)
        closingBl2 = cv2.morphologyEx(openingBl2, cv2.MORPH_CLOSE, kernel)
        # openingBl2 = cv2.morphologyEx(black, cv2.MORPH_OPEN, kernel)
        # closingBl2 = cv2.morphologyEx(openingBl2, cv2.MORPH_CLOSE, kernel)
        # dilationBl2 = cv2.dilate(closingBl2, kernel, iterations=1)

        dilationBl = cv2.dilate(black, kernel, iterations=1)
        openingBl = cv2.morphologyEx(dilationBl, cv2.MORPH_OPEN, kernel)
        closingBl = cv2.morphologyEx(openingBl, cv2.MORPH_CLOSE, kernel)
        
        self.displayImage(self.image, closingC, 1)
        self.displayImage(self.image, mask, 2)
        self.displayImage(self.image, edged, 13)
        # self.displayImage(self.image, edge, 14)

        self.displayImage(self.image, closingR2, 3)
        self.displayImage(self.image, closingR, 4)
        self.displayImage(self.image, self.imageWarp, 15)

        self.displayImage(self.image, closingG2, 5)
        self.displayImage(self.image, closingG, 6)

        self.displayImage(self.image, closingB2, 7)
        self.displayImage(self.image, closingB, 8)

        self.displayImage(self.image, closingY2, 9)
        self.displayImage(self.image, closingY, 10)

        self.displayImage(self.image, closingBl2, 11)
        self.displayImage(self.image, closingBl, 12)
        
        
    def find_card(self):
        # print("HOME ... !!!")
        rX = 100
        rY = 100
        gX = 100
        gY = 100
        bX = 100
        bY = 100
        yX = 100
        yY = 100
        blX = 100
        blY = 100
        self.debugTextBrowser.append("Home ...")
        ret, self.image = self.capture.read()

        self.notCircle = self.image.copy()

        cv2.circle(self.image, (self.x1_slider.value(),
                                self.y1_slider.value()), 5, (0, 0, 255), -1)
        self.current_value.setText(
            'X1 Y1 -> :'+str(self.x1_slider.value()) + ' '+str(self.y1_slider.value()))
        cv2.circle(self.image, (self.x2_slider.value(),
                                self.y2_slider.value()), 5, (0, 0, 255), -1)
        self.current_value2.setText(
            'X2 Y2 -> :'+str(self.x2_slider.value()) + ' '+str(self.y2_slider.value()))
        cv2.circle(self.image, (self.x3_slider.value(),
                                self.y3_slider.value()), 5, (0, 0, 255), -1)
        self.current_value3.setText(
            'X3 Y3 -> :'+str(self.x3_slider.value()) + ' '+str(self.y3_slider.value()))
        cv2.circle(self.image, (self.x4_slider.value(),
                                self.y4_slider.value()), 5, (0, 0, 255), -1)
        self.current_value4.setText(
            'X4 Y4 -> :'+str(self.x4_slider.value()) + ' '+str(self.y4_slider.value()))

        pts1 = np.float32([[self.x1_slider.value(), self.y1_slider.value()],
                           [self.x2_slider.value(), self.y2_slider.value()],
                           [self.x3_slider.value(), self.y3_slider.value()],
                           [self.x4_slider.value(), self.y4_slider.value()]])
        pts2 = np.float32([[0, 0], [500, 0], [0, 500], [500, 500]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        resultWarp = cv2.warpPerspective(self.notCircle, matrix, (500, 500))
        self.imageWarp = resultWarp

        self.imageWarp = cv2.flip(self.imageWarp, 1)
        # self.displayImage(self.image, self.imageWarp, 1)

        Mblurred = cv2.medianBlur(self.imageWarp, 5)
        hsv = cv2.cvtColor(Mblurred, cv2.COLOR_BGR2HSV)

        color_lower = np.array(
            [self.card_lh.value(), self.card_ls.value(), self.card_lv.value()], np.uint8)
        color_upper = np.array(
            [self.card_uh.value(), self.card_us.value(), self.card_uv.value()], np.uint8)

        mask = cv2.inRange(hsv, color_lower, color_upper)
        kernel = np.ones((5, 5), np.uint8)

        red_lower = np.array(
            [self.red_lh.value(), self.red_ls.value(), self.red_lv.value()], np.uint8)
        red_upper = np.array(
            [self.red_uh.value(), self.red_us.value(), self.red_uv.value()], np.uint8)
        green_lower = np.array(
            [self.green_lh.value(), self.green_ls.value(), self.green_lv.value()], np.uint8)
        green_upper = np.array(
            [self.green_uh.value(), self.green_us.value(), self.green_uv.value()], np.uint8)
        blue_lower = np.array(
            [self.blue_lh.value(), self.blue_ls.value(), self.blue_lv.value()], np.uint8)
        blue_upper = np.array(
            [self.blue_uh.value(), self.blue_us.value(), self.blue_uv.value()], np.uint8)
        yellow_lower = np.array(
            [self.yellow_lh.value(), self.yellow_ls.value(), self.yellow_lv.value()], np.uint8)
        yellow_upper = np.array(
            [self.yellow_uh.value(), self.yellow_us.value(), self.yellow_uv.value()], np.uint8)
        black_lower = np.array(
            [self.black_lh.value(), self.black_ls.value(), self.black_lv.value()], np.uint8)
        black_upper = np.array(
            [self.black_uh.value(), self.black_us.value(), self.black_uv.value()], np.uint8)

        self.card_value.setText(
            'Current Value -> Min :'+str(color_lower)+' Max: '+str(color_upper))
        self.red_value.setText('Current Value -> Min :' +
                               str(red_lower)+' Max: '+str(red_upper))
        self.green_value.setText(
            'Current Value -> Min :'+str(green_lower)+' Max: '+str(green_upper))
        self.blue_value.setText(
            'Current Value -> Min :'+str(blue_lower)+' Max: '+str(blue_upper))
        self.yellow_value.setText(
            'Current Value -> Min :'+str(yellow_lower)+' Max: '+str(yellow_upper))
        self.black_value.setText(
            'Current Value -> Min :'+str(black_lower)+' Max: '+str(black_upper))

        color_mask = cv2.inRange(hsv, color_lower, color_upper)
        card = cv2.bitwise_and(
            self.imageWarp, self.imageWarp, mask=color_mask)
        openingC2 = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel)
        closingC2 = cv2.morphologyEx(openingC2, cv2.MORPH_CLOSE, kernel)
        openingC = cv2.morphologyEx(card, cv2.MORPH_OPEN, kernel)
        closingC = cv2.morphologyEx(openingC, cv2.MORPH_CLOSE, kernel)
        mask = cv2.erode(closingC2, kernel)
        edged = cv2.Canny(mask, 50, 220)
        contours, _ = cv2.findContours(
            mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


        for cnt in contours:
            # time.sleep(0.1)
            area = cv2.contourArea(cnt)
            approx = cv2.approxPolyDP(cnt, 0.02*cv2.arcLength(cnt, True), True)
            x = approx.ravel()[0]
            y = approx.ravel()[1]
            self.debugTextBrowser.append(str(area))

            if 10000 > area > 5000:
                stateWork = 1
                if len(approx) == 4 :
                    time.sleep(1)
                    cnts = cv2.findContours(
                        edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                    cnts = imutils.grab_contours(cnts)
                    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
                    Approx = approx
                    Outline = cv2.drawContours(
                        self.imageWarp.copy(), [Approx], -5, (0, 0, 255), 1)
                    ratio = Outline.shape[0] / 500
                    Crop_card = four_point_transform(
                        self.imageWarp, Approx.reshape(4, 2) * ratio)
                    Crop_card = cv2.resize(Crop_card, (int(500), int(500)))
                    img_name = "crop_card.png"

                    
                    # time.sleep(0.1)
                    cv2.imwrite(img_name, Crop_card)
                    imgCrop = cv2.imread("crop_card.png")

                    # -----------------
                    clahe = cv2.createCLAHE(clipLimit=3., tileGridSize=(8, 8))
                    lab = cv2.cvtColor(imgCrop, cv2.COLOR_BGR2LAB)
                    l, a, b = cv2.split(lab)  # split on 3 different channels
                    l2 = clahe.apply(l)  # apply CLAHE to the L-channel
                    lab = cv2.merge((l2, a, b))  # merge channels
                    imgCrop = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
                    # -----------------
                    # cardCount = 0
                    stateWork = 0
                    countRed = 0
                    countBlue = 0
                    countYellow = 0
                    countGreen = 0
                    countBlack = 0


                    cropBlur = cv2.medianBlur(imgCrop, 5)
                    hsv = cv2.cvtColor(cropBlur, cv2.COLOR_BGR2HSV)

                    red_mask = cv2.inRange(hsv, red_lower, red_upper)
                    red = cv2.bitwise_and(
                        self.imageWarp, self.imageWarp, mask=red_mask)

                    bFilterR2 = cv2.bilateralFilter(red_mask, 9, 75, 75)
                    dilationR2 = cv2.dilate(bFilterR2, kernel, iterations=1)
                    openingR2 = cv2.morphologyEx(dilationR2, cv2.MORPH_OPEN, kernel)
                    closingR2 = cv2.morphologyEx(openingR2, cv2.MORPH_CLOSE, kernel)

                    # bFilterR2 = cv2.bilateralFilter(red_mask, 9, 75, 75)
                    dilationR = cv2.dilate(red, kernel, iterations=1)
                    openingR = cv2.morphologyEx(dilationR, cv2.MORPH_OPEN, kernel)
                    closingR = cv2.morphologyEx(openingR, cv2.MORPH_CLOSE, kernel)

                    green_mask = cv2.inRange(hsv, green_lower, green_upper)
                    green = cv2.bitwise_and(
                        self.imageWarp, self.imageWarp, mask=green_mask)
                    dilationG2 = cv2.dilate(green_mask, kernel, iterations=1)
                    openingG2 = cv2.morphologyEx(dilationG2, cv2.MORPH_OPEN, kernel)
                    closingG2 = cv2.morphologyEx(openingG2, cv2.MORPH_CLOSE, kernel)

                    dilationG = cv2.dilate(green, kernel, iterations=1)
                    openingG = cv2.morphologyEx(dilationG, cv2.MORPH_OPEN, kernel)
                    closingG = cv2.morphologyEx(openingG, cv2.MORPH_CLOSE, kernel)

                    blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)
                    blue = cv2.bitwise_and(
                        self.imageWarp, self.imageWarp, mask=blue_mask)
                    dilationB2 = cv2.dilate(blue_mask, kernel, iterations=1)
                    openingB2 = cv2.morphologyEx(dilationB2, cv2.MORPH_OPEN, kernel)
                    closingB2 = cv2.morphologyEx(openingB2, cv2.MORPH_CLOSE, kernel)

                    dilationB = cv2.dilate(blue, kernel, iterations=1)
                    openingB = cv2.morphologyEx(dilationB, cv2.MORPH_OPEN, kernel)
                    closingB = cv2.morphologyEx(openingB, cv2.MORPH_CLOSE, kernel)

                    yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
                    yellow = cv2.bitwise_and(
                        self.imageWarp, self.imageWarp, mask=yellow_mask)
                    dilationY2 = cv2.dilate(yellow_mask, kernel, iterations=1)
                    openingY2 = cv2.morphologyEx(dilationY2, cv2.MORPH_OPEN, kernel)
                    closingY2 = cv2.morphologyEx(openingY2, cv2.MORPH_CLOSE, kernel)

                    dilationY = cv2.dilate(yellow, kernel, iterations=1)
                    openingY = cv2.morphologyEx(dilationY, cv2.MORPH_OPEN, kernel)
                    closingY = cv2.morphologyEx(openingY, cv2.MORPH_CLOSE, kernel)

                    black_mask = cv2.inRange(hsv, black_lower, black_upper)
                    black = cv2.bitwise_and(
                        self.imageWarp, self.imageWarp, mask=black_mask)
                    bFilterBl2 = cv2.bilateralFilter(black_mask, 9, 75, 75)
                    dilationbl2 = cv2.dilate(bFilterBl2, kernel, iterations=1)
                    openingBl2 = cv2.morphologyEx(dilationbl2, cv2.MORPH_OPEN, kernel)
                    closingBl2 = cv2.morphologyEx(openingBl2, cv2.MORPH_CLOSE, kernel)


                    dilationBl = cv2.dilate(black, kernel, iterations=1)
                    openingBl = cv2.morphologyEx(dilationBl, cv2.MORPH_OPEN, kernel)
                    closingBl = cv2.morphologyEx(openingBl, cv2.MORPH_CLOSE, kernel)

                    contoursRed, _ = cv2.findContours(
                        closingR2, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                    try:
                        biggest_contoursRed = max(contoursRed, key=cv2.contourArea)
                        (x, y, w, h) = cv2.boundingRect(biggest_contoursRed)
                        # cv2.rectangle(imgCrop, (x, y),
                        #             (x+w, y+h), (0, 0, 255), 2)
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
                        closingB2, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                    try:
                        biggest_contoursBlue = max(
                            contoursBlue, key=cv2.contourArea)
                        (x, y, w, h) = cv2.boundingRect(biggest_contoursBlue)
                        # cv2.rectangle(imgCrop, (x, y),
                        #             (x+w, y+h), (255, 0, 0), 2)
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
                        closingG2, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                    try:
                        biggest_contoursGreen = max(
                            contoursGreen, key=cv2.contourArea)
                        (x, y, w, h) = cv2.boundingRect(biggest_contoursGreen)
                        # cv2.rectangle(imgCrop, (x, y),
                        #             (x+w, y+h), (0, 255, 0), 2)
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
                        closingY2, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

                    try:
                        biggest_contoursYellow = max(
                            contoursYellow, key=cv2.contourArea)
                        (x, y, w, h) = cv2.boundingRect(biggest_contoursYellow)
                        # cv2.rectangle(imgCrop, (x, y),
                        #             (x+w, y+h), (0, 255, 255), 2)
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
                        closingBl2, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

                    try:
                        biggest_contoursBlack = max(
                            contoursBlack, key=cv2.contourArea)
                        (x, y, w, h) = cv2.boundingRect(biggest_contoursBlack)
                        # cv2.rectangle(imgCrop, (x, y), (x+w, y+h), (0, 0, 0), 2)
                        countBlack = 1

                        Mblack = cv2.moments(biggest_contoursBlack)
                        blX = int(Mblack["m10"] / Mblack["m00"])
                        blY = int(Mblack["m01"] / Mblack["m00"])
                        cv2.circle(imgCrop, (blX, blY), 5, (191, 191, 191), -1)
                        cv2.putText(imgCrop, 'X :' + str(blX) + " Y :" + str(blY),
                                    # bottomLeftCornerOfText
                                    (blX + 30, blY),
                                    cv2.FONT_HERSHEY_SIMPLEX,  # font
                                    0.55,                      # fontScale
                                    (0, 0, 0),            # fontColor
                                    1)

                    except:
                        pass

                    rX = int(250-(rX/2))
                    rY = int(250-(rY/2))
                    gX = int(250-(gX/2))
                    gY = int(250-(gY/2))
                    bX = int(250-(bX/2))
                    bY = int(250-(bY/2))
                    yX = int(250-(yX/2))
                    yY = int(250-(yY/2))
                    blX = int(250-(blX/2))
                    blY = int(250-(blY/2))

                    Send = [rY,rX,gY,gX,bY,bX,yY,yX,blY,blX]
                    # Send = [250,250,0,250,250,0,0,0,0,0]
                    self.debugTextBrowser.append(str(Send))

                    self.displayImage(Outline, imgCrop, 14)

                    for i in Send:
                        time.sleep(0.2)
                        c = struct.pack('B', i)
                        serialPIC.write(c)
                        # self.debugTextBrowser.append(str(i))



                    # # r g b y black

                    sCount = 0
                    while sCount < 10:
                        # print("start "+ str(sCount))
                        Rpic = serialPIC.read()
                        Radr = serialAD.read()
                        if Rpic == b'1':
                            keep = struct.pack('B', 49)
                            serialAD.write(keep)
                            print("keep")
                        elif Rpic == b'0':
                            paste = struct.pack('B', 48)
                            serialAD.write(paste)
                            print("past")
                        else:
                            continue
                        # print(sCount)
                        sCount += 1

                            
                    # ------------------
                    print("------FINISH-------")
                    subprocess.call(["afplay", "beep-06.wav"])
                    self.timer.stop()
                    
        
        
        


        
        


    def stop_webcam(self):
        self.capture.release()
        self.timer.stop()

    # def stop_card(self):
    #     # self.capture.release()
    #     self.timer.stop()

    def displayImage(self, img, imageWarp, window=1):
        qformat = QImage.Format_Indexed8
        if len(imageWarp.shape) == 3:  # [0]=rows, [1]=cols, [2]=channels
            if imageWarp.shape[2] == 4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888

        outImage = QImage(img, img.shape[1],
                          img.shape[0], img.strides[0], qformat)
        outImage = outImage.rgbSwapped()

        outImageWarp = QImage(imageWarp, imageWarp.shape[1],
                              imageWarp.shape[0], imageWarp.strides[0], qformat)
        outImageWarp = outImageWarp.rgbSwapped()

        if window == 1:
            
            self.frame_card1.setPixmap(QPixmap.fromImage(outImageWarp))
            self.frame_card1.setScaledContents(True)
            self.camera_set.setPixmap(QPixmap.fromImage(outImage))
            self.camera_set.setScaledContents(True)

        if window == 2:
            self.frame_card2.setPixmap(QPixmap.fromImage(outImageWarp))
            self.frame_card2.setScaledContents(True)
        elif window == 3:
            self.frame_red.setPixmap(QPixmap.fromImage(outImageWarp))
            self.frame_red.setScaledContents(True)
        elif window == 4:
            self.frame_red2.setPixmap(QPixmap.fromImage(outImageWarp))
            self.frame_red2.setScaledContents(True)
        
        elif window == 5:
            self.frame_green.setPixmap(QPixmap.fromImage(outImageWarp))
            self.frame_green.setScaledContents(True)
        elif window == 6:
            self.frame_green2.setPixmap(QPixmap.fromImage(outImageWarp))
            self.frame_green2.setScaledContents(True)
        elif window == 7:
            self.frame_blue.setPixmap(QPixmap.fromImage(outImageWarp))
            self.frame_blue.setScaledContents(True)
        elif window == 8:
            self.frame_blue2.setPixmap(QPixmap.fromImage(outImageWarp))
            self.frame_blue2.setScaledContents(True)
        elif window == 9:
            self.frame_yellow.setPixmap(QPixmap.fromImage(outImageWarp))
            self.frame_yellow.setScaledContents(True)
        elif window == 10:
            self.frame_yellow2.setPixmap(QPixmap.fromImage(outImageWarp))
            self.frame_yellow2.setScaledContents(True)
        elif window == 11:
            self.frame_black.setPixmap(QPixmap.fromImage(outImageWarp))
            self.frame_black.setScaledContents(True)
        elif window == 12:
            self.frame_black2.setPixmap(QPixmap.fromImage(outImageWarp))
            self.frame_black2.setScaledContents(True)
        elif window == 13:
            self.frame_card3.setPixmap(QPixmap.fromImage(outImageWarp))
            self.frame_card3.setScaledContents(True)

        elif window == 14:
            self.card_output.setPixmap(QPixmap.fromImage(outImageWarp))
            self.card_output.setScaledContents(True)
            self.card_output2.setPixmap(QPixmap.fromImage(outImage))
            self.card_output2.setScaledContents(True)
            self.card_output3.setPixmap(QPixmap.fromImage(outImageWarp))
            self.card_output3.setScaledContents(True)

        elif window == 15:
            self.frame_red3.setPixmap(QPixmap.fromImage(outImageWarp))
            self.frame_red3.setScaledContents(True)
            self.frame_green3.setPixmap(QPixmap.fromImage(outImageWarp))
            self.frame_green3.setScaledContents(True)
            self.frame_blue3.setPixmap(QPixmap.fromImage(outImageWarp))
            self.frame_blue3.setScaledContents(True)
            self.frame_yellow3.setPixmap(QPixmap.fromImage(outImageWarp))
            self.frame_yellow3.setScaledContents(True)
            self.frame_black3.setPixmap(QPixmap.fromImage(outImageWarp))
            self.frame_black3.setScaledContents(True)
    

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = OpencvImg()
    window.setWindowTitle('NICKY')
    window.show()
    sys.exit(app.exec_())
