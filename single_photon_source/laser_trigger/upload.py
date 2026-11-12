"""Compile and upload laser_trigger.ino to the Arduino. Windows.

Python cannot compile AVR code on its own - it needs a compiler and an
uploader. arduino-cli carries both, so install it once:

    winget install ArduinoSA.CLI

(if winget cannot find it, `winget search arduino-cli` shows the current
package id, or grab the .zip from arduino.github.io/arduino-cli and put
arduino-cli.exe somewhere on your PATH)

After that just run:

    python upload.py

The board and its COM port are detected automatically. If detection fails
(unrecognised clone, or the board sitting in bootloader mode), set FQBN
and PORT below by hand.

Close the Arduino Serial Monitor and measure_delay.py first - only one
program can hold the COM port at a time.
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

# -------------------- settings --------------------
FQBN = ""    # leave empty to auto-detect. Uno WiFi Rev2 is "arduino:megaavr:uno2018",
             # the original Uno WiFi is "arduino:avr:unowifi"
PORT = ""    # leave empty to auto-detect, e.g. "COM3"
# --------------------------------------------------

SKETCH_DIR = Path(__file__).resolve().parent


def run(*args):
    """Run arduino-cli, and stop the script if the step fails."""
    print("$ arduino-cli", " ".join(args))
    if subprocess.run(["arduino-cli", *args]).returncode != 0:
        sys.exit(1)


if shutil.which("arduino-cli") is None:
    sys.exit(
        "arduino-cli not found. Install it with:\n"
        "    winget install ArduinoSA.CLI\n"
        "then open a new terminal so the PATH change takes effect."
    )

fqbn, port = FQBN, PORT

if not fqbn or not port:
    listing = subprocess.run(
        ["arduino-cli", "board", "list", "--format", "json"],
        capture_output=True, text=True, check=True,
    ).stdout
    detected = json.loads(listing)
    if isinstance(detected, dict):          # arduino-cli 1.x wraps the list
        detected = detected.get("detected_ports", [])

    for entry in detected:
        boards = entry.get("matching_boards") or []
        if not boards:
            continue                        # a COM port with no board behind it
        fqbn = fqbn or boards[0]["fqbn"]
        port = port or entry.get("port", entry).get("address", "")
        print(f"detected {boards[0]['name']} on {port}")
        break

if not fqbn or not port:
    sys.exit(
        "No board detected. Check the USB cable and that Windows has a driver\n"
        "for the board (Device Manager -> Ports), then rerun. Or run\n"
        "    arduino-cli board list\n"
        "to see what is there and set FQBN and PORT at the top of this script."
    )

core = ":".join(fqbn.split(":")[:2])        # e.g. arduino:megaavr
run("core", "update-index")
run("core", "install", core)
run("compile", "--fqbn", fqbn, str(SKETCH_DIR))
run("upload", "--fqbn", fqbn, "--port", port, str(SKETCH_DIR))

print(f"\nuploaded to {port}")
print(f"Test it:  arduino-cli monitor -p {port} -c baudrate=115200   then type any character")
print(f"Set ARDUINO_PORT = \"{port}\" in measure_delay.py")
