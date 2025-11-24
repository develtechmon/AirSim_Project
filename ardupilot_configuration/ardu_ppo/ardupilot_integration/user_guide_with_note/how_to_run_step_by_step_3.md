✅ STEP 3: TEST CONNECTION
Terminal 2 (new terminal):

cd ~/phd_ardupilot_deploy/ardupilot_integration
python 1_connection_test.py
```

**Expected output:**
```
======================================================================
🔌 TESTING ARDUPILOT CONNECTION
======================================================================
Connection string: 127.0.0.1:14550
Baud rate: 57600
======================================================================

Connecting to vehicle...
✅ Connection successful!

======================================================================
📊 VEHICLE INFORMATION
======================================================================
Autopilot: APM:Copter V4.3.0
Mode: STABILIZE
Armed: False
System status: STANDBY
GPS: GPSInfo:fix=3,num_sat=10
Battery: Battery:voltage=12.587,current=0.0,level=100

======================================================================
🎯 CURRENT STATE
======================================================================
Position: Lat=-35.363262, Lon=149.165237, Alt=0.00m
Attitude: Roll=0.001rad, Pitch=-0.002rad, Yaw=0.000rad
Velocity: North=0.00m/s, East=0.00m/s, Down=0.00m/s
Heading: 0°

======================================================================
📡 TESTING DATA STREAM (5 seconds)
======================================================================
[1/5] Alt=0.00m, Roll=0.001rad, Pitch=-0.002rad
[2/5] Alt=0.00m, Roll=0.001rad, Pitch=-0.002rad
[3/5] Alt=0.00m, Roll=0.001rad, Pitch=-0.002rad
[4/5] Alt=0.00m, Roll=0.001rad, Pitch=-0.002rad
[5/5] Alt=0.00m, Roll=0.001rad, Pitch=-0.002rad

✅ Data streaming works!

======================================================================
Closing connection...
✅ Connection test complete!
======================================================================