"""Arduino pulse -> Time Tagger test. Windows.

Wire Arduino D8 into TT20 input CH through a 1 kohm series resistor, with
Arduino GND on the SMA shield. Press FIRE_KEY to fire: every pulse the
tagger sees is printed with its width and the time since the previous one.

    python tag_pulses.py
"""

import msvcrt
import time

import serial
import TimeTagger

# -------------------- settings --------------------
CH = 1
TRIGGER_LEVEL = 0.12     # V

ARDUINO_PORT = "COM3"
ARDUINO_BAUD = 115200
FIRE_KEY = "s"
# --------------------------------------------------

tagger = TimeTagger.createTimeTagger()
tagger.setTriggerLevel(CH, TRIGGER_LEVEL)

arduino = serial.Serial(ARDUINO_PORT, ARDUINO_BAUD, timeout=2)
time.sleep(2.0)                  

stream = TimeTagger.TimeTagStream(tagger, 1000, [CH, -CH])   # rising and falling edges
tagger.sync()
stream.getData()                 # throw away anything from before the first shot

previous = None
print(f"\n  {FIRE_KEY}  fire      q  quit\n")

while True:
    key = msvcrt.getch().decode(errors="ignore").lower()
    if key == "q":
        break
    if key != FIRE_KEY.lower():
        continue

    arduino.reset_input_buffer()
    arduino.write(FIRE_KEY.encode())
    if arduino.readline().decode(errors="ignore").strip() != "fired":
        print("no reply from the Arduino - check ARDUINO_PORT and that the sketch is uploaded")
        continue

    tagger.sync()                # wait until this shot's tags are processed
    buf = stream.getData()
    ts, chs = buf.getTimestamps(), buf.getChannels()
    rising, falling = ts[chs == CH], ts[chs == -CH]

    if len(rising) == 0:
        print(f"Arduino fired, but nothing on input {CH} - check wiring, ground and trigger level")
        continue

    for t in rising:
        after = falling[falling > t]
        width = f"{(after[0] - t) / 1e9:.3f} ms" if len(after) else "?"
        since = "" if previous is None else f"   {(t - previous) / 1e12:.6f} s since previous"
        print(f"pulse at {t / 1e12:12.6f} s   width {width}{since}")
        previous = t

arduino.close()
TimeTagger.freeTimeTagger(tagger)
