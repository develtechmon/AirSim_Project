==========================================================================================
STAGE 2 - HARDWARE FOLDER  (what to do on the PC, and what to do on the Pi)
==========================================================================================

1. WHAT THIS FOLDER IS
----------------------
This folder is the "lunchbox" for the Raspberry Pi Zero 2 W:
exactly the files the Pi needs to fly the real drone. Nothing else.

    PC  = the kitchen  -> turns the trained brain into an .onnx file (export_onnx.py)
    Pi  = eats lunch   -> flies the drone with that .onnx file (real_fan_test.py)

    | Script            | Runs on | Lives in            | Job                                  |
    |-------------------|---------|---------------------|--------------------------------------|
    | export_onnx.py    | PC      | Stage_2\  (NOT here)| trained model (.zip) -> .onnx file   |
    | real_fan_test.py  | Pi      | Stage_2\Hardware\   | flies the drone in fan wind          |
    | ap_link.py        | Pi      | Stage_2\Hardware\   | MAVLink link to the flight controller|
    | pid_expert.py     | Pi      | Stage_2\Hardware\   | only needed for --model pid          |

    NOTE: ap_link.py and pid_expert.py are COPIES of the files in Stage_2\.
          If the Stage_2\ version is ever updated, copy it here again.


2. FOLDER LAYOUT (target)
-------------------------
    Stage_2\
       export_onnx.py                <- PC script
       ap_link.py, pid_expert.py, drone_hover_disturbance_env.py, ...
       runs\run1\onnx\               <- export_onnx.py writes warm.onnx + stage1.onnx here

       Hardware\                     <- copy THIS WHOLE FOLDER to the Pi
          README.txt
          ap_link.py
          real_fan_test.py
          pid_expert.py
          runs\run1\onnx\warm.onnx   <- copied from Stage_2\runs\run1\onnx\
          runs\run1\onnx\stage1.onnx
          results\                   <- real-drone results copied BACK from the Pi (step 6)


3. STEP 1 - ON THE PC: EXPORT THE BRAINS TO ONNX
------------------------------------------------
In the Stage_2\ folder (not Hardware\):

    py .\export_onnx.py

    py .\export_onnx.py --model stage1

    | Command                  | Exports                     | File made                    |
    |--------------------------|-----------------------------|------------------------------|
    | (no flag)                | Stage 2 PPO (warm) - main   | runs\run1\onnx\warm.onnx     |
    | --model stage1           | Stage 1 network (baseline)  | runs\run1\onnx\stage1.onnx   |

    EXPECTED RESULT: "PASS" - ONNX output matches the PPO model (max difference < 1e-4 m/s).
    If FAIL: do not fly. Send the printout.


4. STEP 2 - ON THE PC: FILL THE LUNCHBOX
----------------------------------------
    Copy   Stage_2\runs\run1\onnx\    ->   Stage_2\Hardware\runs\run1\onnx\
    (keep the same folder path: runs\run1\onnx\)

    Then copy the whole Hardware\ folder to the Pi as  ~/Stage_2/
    (USB stick, WinSCP, or scp). On the Pi it must look like:

    ~/Stage_2/
       ap_link.py
       real_fan_test.py
       pid_expert.py
       runs/run1/onnx/warm.onnx
       runs/run1/onnx/stage1.onnx


5. STEP 3 - ON THE PI: ONE-TIME SETUP
-------------------------------------
    pip3 install numpy pymavlink onnxruntime

    (No PyTorch / Stable-Baselines3 needed on the Pi - the .onnx file has everything.)

    Connection defaults in real_fan_test.py:  /dev/serial0  @ 921600 baud
    -> the flight controller's telemetry port must use the same baud rate.


6. STEP 4 - ON THE PI: LINK CHECK (PROPS OFF)
---------------------------------------------
    cd ~/Stage_2

    python3 real_fan_test.py --check

    EXPECTED RESULT: "Position and attitude received" + heading printout. Nothing is sent
    to the motors. If it hangs at "Connecting": check wiring, /dev/serial0 and baud rate.

    IMPORTANT: start the script BEFORE takeoff (indoor, no GPS: it sends the EKF origin).


7. STEP 5 - ON THE PI: FLY THE TESTS
------------------------------------
Who flies:
    1. YOU take off in AltHold, switch to PosHold, hover at about 1.5-2 m
    2. flip the mode switch to GUIDED   -> the Pi flies one episode (25 s)
    3. the Pi switches back to PosHold by itself and prints the result
    4. next episode: flip away from GUIDED and back
    ANY TIME: flip away from GUIDED -> you have control instantly

Run in this order (same fan position for every model):

    python3 real_fan_test.py --model warm --fan off --episodes 3

    python3 real_fan_test.py --model warm --fan high --fan-dist 1.5 --episodes 3

    python3 real_fan_test.py --model stage1 --fan high --fan-dist 1.5 --episodes 3

    python3 real_fan_test.py --model pid --fan high --fan-dist 1.5 --episodes 3

    | Test                       | Purpose                                   |
    |----------------------------|-------------------------------------------|
    | warm, fan off              | calm-air hover check (does it fly at all) |
    | warm, fan high             | Stage 2 PPO in real wind  (MAIN RESULT)   |
    | stage1, fan high           | "before Stage 2" baseline                 |
    | pid, fan high              | Stage 1 teacher reference (optional)      |

    The Pi stops and switches to PosHold by itself if:
    > 1.5 m from target, > 0.75 m height error, tilt > 35 deg, height < 0.4 m,
    no position for 0.5 s, or heading > 20 deg from the training heading.

    EXPECTED RESULT: warm stays closer to the target than stage1 with the fan on
    (same pattern as SITL: PPO about 94% within 0.5 m vs Stage 1 about 84%).


8. STEP 6 - BACK TO THE PC: SAVE THE RESULTS
--------------------------------------------
    On the Pi, results are in:   ~/Stage_2/runs/run1/results/
        real_<model>_fan-<fan>_<date>.json / .csv / _steps.csv

    Copy them back to the PC into:   Stage_2\Hardware\results\
    (keeps real-drone results separate from the SITL results in Stage_2\runs\run1\results\)


9. QUICK TROUBLESHOOTING
------------------------
    | Message                                   | Fix                                         |
    |-------------------------------------------|---------------------------------------------|
    | ModuleNotFoundError: ap_link              | ap_link.py missing next to real_fan_test.py |
    | ModuleNotFoundError: onnxruntime          | pip3 install onnxruntime  (on the Pi)       |
    | ...onnx not found - run export_onnx.py    | step 1 + step 2 not done / wrong folder     |
    | No runs/run<N> folder                     | runs/run1/onnx/ not copied to the Pi        |
    | refuses to fly: heading                   | yaw the drone to face the training heading  |
    | export_onnx.py: ModuleNotFoundError       | run it in Stage_2\, not in Hardware\        |