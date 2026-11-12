"""Record all channels while firing the laser by hand. Windows.

Type start + ENTER to begin recording. While it records, press FIRE_KEY to
fire the laser through the Arduino (laser_trigger.ino). Press q to stop:
the tags are saved to the Desktop as a CSV with one column per channel,
with the raw .ttbin kept next to it.

    python single_photons_v1.py
"""

import csv
import msvcrt
import time
from datetime import datetime
from itertools import zip_longest
from pathlib import Path

import numpy as np
import serial
import TimeTagger

# -------------------- settings --------------------
CHANNELS = [1, 2, 3, 4, 5, 6]    # rising edges only; -1..-6 would be the falling edges
TRIGGER_LEVEL = 2      # V, applied to every channel

ARDUINO_PORT = "COM4"
ARDUINO_BAUD = 115200
FIRE_KEY = "s"

OUTPUT_DIR = Path.home() / "Desktop"
# --------------------------------------------------


def export_by_channel(ttbin_path, csv_path):
    """Write one column of timestamps (ps) per channel. Returns tags per channel."""
    reader = TimeTagger.FileReader(str(ttbin_path))
    chunks = {ch: [] for ch in CHANNELS}
    overflow = False

    while reader.hasData():
        buf = reader.getData(1_000_000)
        ts, chs, types = buf.getTimestamps(), buf.getChannels(), buf.getEventTypes()
        overflow |= bool((types != 0).any())     # 0 = normal time tag
        for ch in CHANNELS:
            chunks[ch].append(ts[(chs == ch) & (types == 0)])

    columns = [np.concatenate(chunks[ch]).tolist() if chunks[ch] else [] for ch in CHANNELS]

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([f"ch{ch}_ps" for ch in CHANNELS])
        writer.writerows(zip_longest(*columns, fillvalue=""))

    return [len(col) for col in columns], overflow


stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
ttbin_path = OUTPUT_DIR / f"single_photons_v1_{stamp}.ttbin"
csv_path = OUTPUT_DIR / f"single_photons_v1_{stamp}.csv"

tagger = TimeTagger.createTimeTagger()
for ch in CHANNELS:
    tagger.setTriggerLevel(ch, TRIGGER_LEVEL)

arduino = serial.Serial(ARDUINO_PORT, ARDUINO_BAUD, timeout=2)
time.sleep(2.0)                   # the board reboots when the port is opened

while input("\ntype start + ENTER to begin recording: ").strip().lower() != "start":
    pass

file_writer = TimeTagger.FileWriter(tagger, str(ttbin_path), CHANNELS)
tagger.sync()                     # recording is live before the first shot
t0 = time.monotonic()
shots = 0
print(f"\nrecording channels {CHANNELS}      {FIRE_KEY}  fire      q  stop and save\n")

try:
    while True:
        key = msvcrt.getch().decode(errors="ignore").lower()
        if key == "q":
            break
        if key != FIRE_KEY.lower():
            continue

        arduino.reset_input_buffer()
        arduino.write(FIRE_KEY.encode())
        reply = arduino.readline().decode(errors="ignore").strip()
        shots += 1
        status = "fired" if reply == "fired" else f"no reply - check {ARDUINO_PORT}"
        print(f"shot {shots:4d}   {status}   t = {time.monotonic() - t0:8.1f} s")
except KeyboardInterrupt:
    pass                          # Ctrl+C stops and saves, same as q

duration = time.monotonic() - t0
file_writer.stop()
TimeTagger.freeTimeTagger(tagger)
arduino.close()

print(f"\nrecorded {duration:.1f} s, {shots} shots - exporting...")
counts, overflow = export_by_channel(ttbin_path, csv_path)
for ch, n in zip(CHANNELS, counts):
    print(f"  ch{ch}: {n} tags")
if overflow:
    print("  WARNING: the tagger overflowed during the run - some tags are missing")
print(f"saved {csv_path}")
print(f"raw   {ttbin_path}")
