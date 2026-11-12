"""Fire the laser trigger by hand - no Time Tagger involved. Windows.

Press FIRE_KEY (set below) to send a trigger to the Arduino running
laser_trigger.ino.
The pulse itself (1 ms, 5 V on D8 by default) is set in laser_trigger.ino;
upload.py only flashes it. The sketch answers "fired" once the pulse is
out, so a missing reply means the Arduino never got the command.

Use it to check the Arduino -> laser link on its own: watch D8 on a scope,
or look for the laser's sync pulse, without starting a measurement.
Close measure_delay.py and the Serial Monitor first - only one program
can hold the COM port.

    python fire_trigger.py
"""

import msvcrt
import time

import serial

# -------------------- settings --------------------
FIRE_KEY = "s"           # keyboard key that fires the trigger (q quits)
# Amplitude cannot be set here: the Arduino pin always outputs 5 V.

ARDUINO_PORT = "COM3"    # same port as in measure_delay.py
ARDUINO_BAUD = 115200
REPLY_TIMEOUT = 2.0      # s to wait for "fired"; raise it if N_PULSES in the
                         # sketch makes a burst last longer than this
# --------------------------------------------------

arduino = serial.Serial(ARDUINO_PORT, ARDUINO_BAUD, timeout=REPLY_TIMEOUT)
time.sleep(2.0)                   # the board reboots when the port is opened

shots = 0
print(f"\n  {FIRE_KEY}  fire the trigger      q  quit\n")

while True:
    key = msvcrt.getch().decode(errors="ignore").lower()
    if key == "q":
        break
    if key != FIRE_KEY.lower():
        continue

    arduino.reset_input_buffer()  # drop any late reply from a previous shot
    arduino.write(FIRE_KEY.encode())
    reply = arduino.readline().decode(errors="ignore").strip()
    shots += 1

    if reply == "fired":
        print(f"shot {shots:4d}   fired")
    else:
        print(f"shot {shots:4d}   no reply - is laser_trigger.ino uploaded, and is {ARDUINO_PORT} right?")

arduino.close()
