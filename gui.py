import sys
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QApplication, QDialog
from PyQt5.uic import loadUi
import cv2
import numpy as np
import imutils
from transform import four_point_transform


class ColorDetector(QDialog):
    def __init__(self):
        super(ColorDetector, self).__init__()
        loadUi('demo2gui.ui', self)
        self.image = None
        self.start_button.clicked.connect(self.start_webcam)
        self.stop_button.clicked.connect(self.stop_webcam)


    # def start_card(self, imgWarp):

        
    


    def start_webcam(self):
        self.capture = cv2.VideoCapture(1)
        # self.capture = cap

        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 300)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 400)
        
        self.x1_slider.setValue(545)
        self.y1_slider.setValue(55)
        self.x2_slider.setValue(86)
        self.y2_slider.setValue(55)
        self.x3_slider.setValue(545)
        self.y3_slider.setValue(400)
        self.x4_slider.setValue(86)
        self.y4_slider.setValue(410)

        self.card_lh.setValue(0)
        self.card_ls.setValue(19)
        self.card_lv.setValue(0)
        self.card_uh.setValue(63)
        self.card_us.setValue(255)
        self.card_uv.setValue(255)

        self.red_lh.setValue(166)
        self.red_ls.setValue(84)
        self.red_lv.setValue(141)
        self.red_uh.setValue(186)
        self.red_us.setValue(255)
        self.red_uv.setValue(255)

        self.green_lh.setValue(66)
        self.green_ls.setValue(122)
        self.green_lv.setValue(129)
        self.green_uh.setValue(86)
        self.green_us.setValue(255)
        self.green_uv.setValue(255)

        self.blue_lh.setValue(97)
        self.blue_ls.setValue(100)
        self.blue_lv.setValue(117)
        self.blue_uh.setValue(117)
        self.blue_us.setValue(255)
        self.blue_uv.setValue(255)

        self.yellow_lh.setValue(23)
        self.yellow_ls.setValue(59)
        self.yellow_lv.setValue(19)
        self.yellow_uh.setValue(54)
        self.yellow_us.setValue(255)
        self.yellow_uv.setValue(255)

        self.black_lh.setValue(23)
        self.black_ls.setValue(59)
        self.black_lv.setValue(19)
        self.black_uh.setValue(54)
        self.black_us.setValue(255)
        self.black_uv.setValue(255)


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

        self.image = cv2.flip(self.image, 1)
        # self.imageWarp = cv2.flip(self.imageWarp, 1)
        # self.displayImage(self.image, self.imageWarp, 1)
        
        Mblurred = cv2.medianBlur(self.imageWarp, 5)
        hsv = cv2.cvtColor(Mblurred, cv2.COLOR_BGR2HSV)

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
        openingC2 = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel)
        closingC2 = cv2.morphologyEx(openingC2, cv2.MORPH_CLOSE, kernel)
        openingC = cv2.morphologyEx(card, cv2.MORPH_OPEN, kernel)
        closingC = cv2.morphologyEx(openingC, cv2.MORPH_CLOSE, kernel)
        mask = cv2.erode(closingC2, kernel)
        edged = cv2.Canny(mask, 50, 220)
        contours, _ = cv2.findContours(
            mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)



        red_mask = cv2.inRange(hsv, red_lower, red_upper)
        red = cv2.bitwise_and(
            self.imageWarp, self.imageWarp, mask=red_mask)
        openingR2 = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
        closingR2 = cv2.morphologyEx(openingR2, cv2.MORPH_CLOSE, kernel)
        openingR = cv2.morphologyEx(red, cv2.MORPH_OPEN, kernel)
        closingR = cv2.morphologyEx(openingR, cv2.MORPH_CLOSE, kernel)


        green_mask = cv2.inRange(hsv, green_lower, green_upper)
        green = cv2.bitwise_and(
            self.imageWarp, self.imageWarp, mask=green_mask)
        openingG2 = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)
        closingG2 = cv2.morphologyEx(openingG2, cv2.MORPH_CLOSE, kernel)
        openingG = cv2.morphologyEx(green, cv2.MORPH_OPEN, kernel)
        closingG = cv2.morphologyEx(openingG, cv2.MORPH_CLOSE, kernel)

        blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)
        blue = cv2.bitwise_and(
            self.imageWarp, self.imageWarp, mask=blue_mask)
        openingB2 = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel)
        closingB2 = cv2.morphologyEx(openingB2, cv2.MORPH_CLOSE, kernel)
        openingB = cv2.morphologyEx(blue, cv2.MORPH_OPEN, kernel)
        closingB = cv2.morphologyEx(openingB, cv2.MORPH_CLOSE, kernel)

        yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
        yellow = cv2.bitwise_and(
            self.imageWarp, self.imageWarp, mask=yellow_mask)
        openingY2 = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, kernel)
        closingY2 = cv2.morphologyEx(openingY2, cv2.MORPH_CLOSE, kernel)
        openingY = cv2.morphologyEx(yellow, cv2.MORPH_OPEN, kernel)
        closingY = cv2.morphologyEx(openingY, cv2.MORPH_CLOSE, kernel)

        black_mask = cv2.inRange(hsv, black_lower, black_upper)
        black = cv2.bitwise_and(
            self.imageWarp, self.imageWarp, mask=black_mask)
        openingBl2 = cv2.morphologyEx(black_mask, cv2.MORPH_OPEN, kernel)
        closingBl2 = cv2.morphologyEx(openingBl2, cv2.MORPH_CLOSE, kernel)
        openingBl = cv2.morphologyEx(black, cv2.MORPH_OPEN, kernel)
        closingBl = cv2.morphologyEx(openingBl, cv2.MORPH_CLOSE, kernel)
        
        self.displayImage(self.image, closingC, 1)
        self.displayImage(self.image, color_mask, 2)
        self.displayImage(self.image, edged, 13)
        # self.displayImage(self.image, edge, 14)

        self.displayImage(self.image, closingR2, 3)
        self.displayImage(self.image, closingR, 4)
        self.displayImage(self.image, closingG2, 5)
        self.displayImage(self.image, closingG, 6)
        self.displayImage(self.image, closingB2, 7)
        self.displayImage(self.image, closingB, 8)
        self.displayImage(self.image, closingY2, 9)
        self.displayImage(self.image, closingY, 10)
        self.displayImage(self.image, closingBl2, 11)
        self.displayImage(self.image, closingBl, 12)
        
        # self.start_card.clicked.connect(self.start_card, self.imageWarp)
        # self.stop_card.clicked.connect(self.stop_card)
        




    def stop_webcam(self):
        self.capture.release()
        self.timer.stop()

    def stop_card(self):
        # self.capture.release()
        self.timer.stop()

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
    

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ColorDetector()
    window.setWindowTitle('OpenCV Color Detector')
    window.show()
    sys.exit(app.exec_())
