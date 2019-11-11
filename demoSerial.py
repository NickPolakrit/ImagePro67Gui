import serial
serialPort = serial.Serial(
    "/dev/cu.VirtualSerialPort", 115200, 8, 'N', 1, 0, 0, 0, 0, 0)

serialPort.setRTS(0)
serialPort.setDTR(0)

while (1):
    cInput = str(input("Commands : "))
    print(cInput)
    serialPort.write(cInput.encode())

    print(serialPort.readline())
    # print(serialPort.readline().decode('ascii').strip().split(','))
