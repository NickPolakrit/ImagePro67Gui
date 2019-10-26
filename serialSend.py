import serial
ser = serial.Serial('/dev/cu.usbserial-AC00YIZF')  # open serial port
print(ser.name)         # check which port was really used
ser.write(b'hello')
