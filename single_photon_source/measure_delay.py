
import msvcrt
import sys
import time

import numpy as np
import serial
import TimeTagger

# -------------------- settings --------------------
SYNC_CH = 1              
DETECTOR_CH = 2          

SYNC_TRIGGER_LEVEL = 0.5  
DET_TRIGGER_LEVEL = 0.5    

BINWIDTH = 100        

ARDUINO_PORT = "COM3"
ARDUINO_BAUD = 115200
FIRE_KEY = "s"

SETTLE = 0.2     
OUTFILE = "delays_ps.txt"
# --------------------------------------------------


def read_key():
    return msvcrt.getch().decode(errors="ignore")


tagger = TimeTagger.createTimeTagger()
tagger.setTriggerLevel(SYNC_CH, SYNC_TRIGGER_LEVEL)
tagger.setTriggerLevel(DETECTOR_CH, DET_TRIGGER_LEVEL)


rate = TimeTagger.Countrate(tagger, [SYNC_CH, DETECTOR_CH])
rate.startFor(int(1e12))
rate.waitUntilFinished()
r_sync, r_det = rate.getData()
print(f"idle rates:  sync {r_sync:.0f} /s   detector {r_det:.0f} /s")

arduino = serial.Serial(ARDUINO_PORT, ARDUINO_BAUD, timeout=1)
time.sleep(2.0)
arduino.reset_input_buffer()

meas = TimeTagger.StartStop(tagger, DETECTOR_CH, SYNC_CH, BINWIDTH)
meas.start()

shots = 0
print(f"\n  {FIRE_KEY}  fire the laser      q  stop and save\n")

while True:
    key = read_key().lower()
    if key == "q":
        break
    if key != FIRE_KEY.lower():
        continue

    arduino.write(FIRE_KEY.encode())
    shots += 1
    time.sleep(SETTLE)

    data = meas.getData()
    total = int(data[:, 1].sum()) if len(data) else 0
    if total:
        peak = data[np.argmax(data[:, 1]), 0]
        print(f"shot {shots:4d}   pairs {total:6d}   peak delay {peak / 1000:.3f} ns")
    else:
        print(f"shot {shots:4d}   no start-stop pairs yet")

meas.stop()
data = meas.getData()
arduino.close()
TimeTagger.freeTimeTagger(tagger)

if len(data) == 0:
    print("\nNo start-stop pairs recorded - check the wiring and trigger levels.")
    sys.exit()

times, counts = data[:, 0], data[:, 1]
peak = times[np.argmax(counts)]

# The peak is the delay. The weighted mean is only meaningful once the peak
# sits well above the dark-count background, since every dark count that
# lands before the real photon is recorded as a pair too.
print(f"\nshots fired : {shots}")
print(f"pairs       : {int(counts.sum())}")
print(f"peak delay  : {peak / 1000:.3f} ns")
print(f"mean delay  : {np.average(times, weights=counts) / 1000:.3f} ns")

np.savetxt(OUTFILE, data, fmt="%d", header="delay_ps  counts")
print(f"saved       : {OUTFILE}")
