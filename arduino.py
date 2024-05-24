import serial
import time


def read_arduino(arduino):
    data = arduino.readline().decode("utf-8")
    if len(data) > 0:
        return data
    return ""


def send_arduino(arduino, value_to_send):
    arduino.write(str(value_to_send).encode())


class Arduino:
    def __init__(self, serport="/dev/ttyACM0", baud=9600, timeout=0.08):
        self.port = serport
        self.baud = baud
        self.timeout = timeout
        self.arduino = None

    def init_arduino(self, time_limit=5):
        self.arduino = serial.Serial(port=self.port, baudrate=self.baud, timeout=self.timeout)
        print("Wait arduino ...")
        exit_count = 0
        sleeptime = 0.25
        while True:
            exit_count += 1
            time.sleep(sleeptime)
            send_arduino(self.arduino, "start_e")
            if "OK" in read_arduino(self.arduino):
                print("Connected!")
                return True
            elif sleeptime * exit_count >= time_limit:
                print("Failed, timeout!")
                return False

    def wait_for_btn(self, time_limit=60):
        exit_count = 0
        sleeptime = 0.25
        print("Wait for btn ...")
        while True:
            exit_count += 1
            if "btn" in read_arduino(self.arduino):
                print("button click!")
                return True
            elif sleeptime * exit_count >= time_limit:
                return False

    def set_xy(self, setx, sety):
        tosend = "x{x}y{y}e".format(x=str(round(setx)), y=str(round(sety)))
        send_arduino(self.arduino, tosend)

# arduinoinit
