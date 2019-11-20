import sys
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QApplication, QDialog
from PyQt5.uic import loadUi
import cv2
import numpy as np
import imutils


class ColorDetector(QDialog):
    def __init__(self):
        super(ColorDetector, self).__init__()
        loadUi('demo2gui.ui', self)
        self.image = None
        self.start_button.clicked.connect(self.start_webcam)
        self.stop_button.clicked.connect(self.stop_webcam)

    def start_webcam(self):
        self.capture = cv2.VideoCapture(0)
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
        self.displayImage(self.image, self.imageWarp, 1)


    def stop_webcam(self):
        self.capture.release()
        self.timer.stop()

    def displayImage(self, img, imageWarp, window=1):
        qformat = QImage.Format_Indexed8
        if len(img.shape) == 3:  # [0]=rows, [1]=cols, [2]=channels
            if img.shape[2] == 4:
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
            self.frame_card2.setPixmap(QPixmap.fromImage(outImage))
            self.frame_card2.setScaledContents(True)

            
            self.camera_set.setPixmap(QPixmap.fromImage(outImage))
            self.camera_set.setScaledContents(True)

        # if window == 2:
        #     self.frame_card2.setPixmap(QPixmap.fromImage(outImage))
        #     self.frame_card2.setScaledContents(True)
    

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ColorDetector()
    window.setWindowTitle('OpenCV Color Detector')
    window.show()
    sys.exit(app.exec_())
