@echo off
title FlightGear quad view (ArduPilot SITL)

rem ---- 1. Paths (change FG_EXE if your FlightGear is somewhere else) ----
set "FG_EXE=C:\Program Files\FlightGear 2024.1\bin\fgfs.exe"
set "FG_AIRCRAFT=%USERPROFILE%\fg_aircraft"

rem ---- 2. Check FlightGear is there ----
if not exist "%FG_EXE%" (
    echo [ERROR] fgfs.exe not found here:
    echo         %FG_EXE%
    echo FlightGear folders in Program Files:
    dir /b "C:\Program Files\FlightGear*"
    echo Fix the FG_EXE line at the top of this file.
    pause
    exit /b 1
)

rem ---- 3. Check the ArduPilot quad model is there ----
if not exist "%FG_AIRCRAFT%\arducopter\arducopter-set.xml" (
    echo [ERROR] ArduPilot quad model not found here:
    echo         %FG_AIRCRAFT%\arducopter
    echo Run this ONCE in WSL, then try again:
    echo   cp -r ~/ardupilot/Tools/autotest/aircraft /mnt/c/Users/lukas/fg_aircraft
    pause
    exit /b 1
)

rem ---- 4. Start FlightGear: listens on UDP 5503 for ArduPilot SITL ----
echo Starting FlightGear (KSFO, noon, quad) ...
"%FG_EXE%" --native-fdm=socket,in,10,,5503,udp --fdm=external --aircraft=arducopter "--fg-aircraft=%FG_AIRCRAFT%" --airport=KSFO --timeofday=noon --geometry=1400x900 --disable-sound --disable-ai-models --disable-random-objects --wind=0@0 --log-level=info

echo.
echo FlightGear closed (exit code %ERRORLEVEL%). Read any error above.
pause