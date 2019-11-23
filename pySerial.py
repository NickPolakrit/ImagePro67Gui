# import serial.tools.list_ports
# comlist = serial.tools.list_ports.comports()
# connected = []
# for element in comlist:
#     connected.append(element.device)
# print("Connected COM ports: " + str(connected))

import time
import serial 
import struct

serialAD = serial.Serial(
    "/dev/cu.usbserial-14320", 115200, 8, 'N', 1, 0, 0, 0, 0, 0)

serialAD.setRTS(0)
serialAD.setDTR(0)

while True:


    Radr = serialAD.read()

    print("start")
    keep = struct.pack('B', 49)
    serialAD.write(keep)
    

    if Radr == b'1':
        print("01")
    elif Radr == b'0':
        print("02")
    

    time.sleep(2)
