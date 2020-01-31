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


class OpencvImg(QDialog):
    def __init__(self):
        super(OpencvImg, self).__init__()
        loadUi('module.ui', self)
        self.image = None
        self.runbt.clicked.connect(self.start_webcam)
        self.stopbt.clicked.connect(self.stop_webcam)

    def start_webcam(self):
        self.debugTextBrowser.append("RUN")

    def stop_webcam(self):
        self.debugTextBrowser.append("STOP")

    # def stop_card(self):
    #     # self.capture.release()
    #     self.timer.stop()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = OpencvImg()
    window.setWindowTitle('NICKY')
    window.show()
    sys.exit(app.exec_())
