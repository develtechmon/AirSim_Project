r"""
get_logs_esp32.py - PC: copy the throw results from the ESP32-S3 (USB) into runs\throw_esp32\
=====================================================================================================
Why: on the Pi you used  scp -r ...runs .  The ESP32-S3 keeps its files on its own flash; this script asks it
for every file over the USB cable (command getall) and saves them, so the thesis data is never only on the board.

1. HOW TO RUN  (Windows PowerShell, folder Q250_Hardware_Throw_Fly; Arduino Serial Monitor CLOSED)
------------------------------------------------------------------------------------------------------
    py -m pip install pyserial                       (once)

    py .\get_logs_esp32.py --port COM5               (your COM port: Arduino IDE -> Tools -> Port)

    py .\get_logs_esp32.py --port COM5 --delete      (same, then empties the board - only after a good copy)
    Works with the battery out (USB power only): the board answers while it waits for the FC / GPS.

2. EXPECTED RESULT
------------------------------------------------------------------------------------------------------
    | Line printed                                         | Means                                      |
    |------------------------------------------------------|--------------------------------------------|
    | saved runs\throw_esp32\run0003.csv (812 B) OK        | file copied, size checked                  |
    | 6 files saved, 0 bad -> runs\throw_esp32\            | all done                                   |
    | board emptied (rmall)                                | only with --delete and 0 bad               |
    "no answer": wrong COM port, Serial Monitor still open, or the board is in the middle of a throw.
    "SIZE WRONG": copy again (USB cable / port); --delete is skipped automatically.

3. OUTPUTS  (same columns as the Pi's files - see throw_esp32s3.ino section 4)
------------------------------------------------------------------------------------------------------
    runs\throw_esp32\run<boot>.csv / run<boot>_steps.csv / run<boot>.json / console.log
    A file that already exists on the PC is kept and the new one is saved as name_1.csv (nothing overwritten).
"""

import argparse
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "runs" / "throw_esp32"


def read_all(ser, quiet_s=3.0, max_s=300.0):
    """Read until the board has been quiet for quiet_s (a full getall ends with 'N files, free flash')."""
    data, t_last, t0 = bytearray(), time.time(), time.time()
    while time.time() - t_last < quiet_s and time.time() - t0 < max_s:
        chunk = ser.read(4096)
        if chunk:
            data += chunk
            t_last = time.time()
            if b" files, free flash " in data[-200:]:
                break
    return bytes(data)


def parse(data):
    """Files between '=== BEGIN name size' and '=== END name'."""
    files, i = [], 0
    while True:
        b = data.find(b"=== BEGIN ", i)
        if b < 0:
            return files
        nl = data.find(b"\n", b)
        name, size = data[b + 10:nl].decode(errors="replace").strip().rsplit(" ", 1)
        size = int(size)
        body = data[nl + 1:nl + 1 + size]
        end = data.find(b"=== END " + name.encode(), nl + 1 + size)
        files.append((name, size, body, end >= 0))
        i = nl + 1 + size


def main(args):
    import serial
    OUT.mkdir(parents=True, exist_ok=True)
    with serial.Serial(args.port, 115200, timeout=0.2) as ser:
        time.sleep(1.0)
        ser.reset_input_buffer()
        ser.write(b"getall\n")
        data = read_all(ser)
        files = parse(data)
        if not files:
            raise SystemExit("no answer: wrong COM port, Serial Monitor still open, or the board is busy (throwing)")
        bad = 0
        for name, size, body, ended in files:
            ok = len(body) == size and ended
            bad += not ok
            dst = OUT / name
            k = 1
            while dst.exists():
                dst = OUT / f"{Path(name).stem}_{k}{Path(name).suffix}"
                k += 1
            dst.write_bytes(body)
            print(f"saved {dst.relative_to(HERE)} ({len(body)} B) {'OK' if ok else 'SIZE WRONG'}")
        print(f"{len(files)} files saved, {bad} bad -> {OUT.relative_to(HERE)}\\")
        if args.delete:
            if bad:
                print("--delete skipped: some files were not copied completely")
            else:
                ser.write(b"rmall\n")
                time.sleep(1.0)
                print("board emptied (rmall)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", required=True, help="e.g. COM5 (Arduino IDE -> Tools -> Port)")
    ap.add_argument("--delete", action="store_true", help="empty the board after a complete copy")
    main(ap.parse_args())
