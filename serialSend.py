# import serial
# ser = serial.Serial('/dev/cu.usbserial-AC00YIZ6')  # open serial port
# print(ser.name)         # check which port was really used
# ser.write(b's')


import serial
serialPort = serial.Serial(
    "/dev/cu.usbserial-AC00YIZF", 115200, 8, 'N', 1, 0, 0, 0, 0, 0)

serialDevice.setRTS(0)
serialDevice.setDTR(0)

while (1):
    g = str(input("Commands : "))
    print(g)
    serialPort.write(g.encode())
