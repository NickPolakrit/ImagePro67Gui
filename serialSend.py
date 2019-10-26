import serial
ser = serial.Serial('/dev/cu.SOC')  # open serial port
print(ser.name)         # check which port was really used
ser.write(b'hello')
