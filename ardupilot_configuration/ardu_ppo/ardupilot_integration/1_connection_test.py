"""
ARDUPILOT CONNECTION TEST
=========================
Tests connection to ArduPilot SITL or real drone via MAVLink.

Usage:
    # For SITL (simulation)
    python 1_connection_test.py --connect 127.0.0.1:14550
    
    # For real drone via USB
    python 1_connection_test.py --connect /dev/ttyUSB0
    
    # For real drone via telemetry
    python 1_connection_test.py --connect /dev/ttyAMA0 --baud 57600
"""

from dronekit import connect, VehicleMode, LocationGlobalRelative
import time
import argparse


def test_connection(connection_string, baud_rate=57600):
    """Test connection to ArduPilot"""
    
    print("\n" + "="*70)
    print("🔌 TESTING ARDUPILOT CONNECTION")
    print("="*70)
    print(f"Connection string: {connection_string}")
    print(f"Baud rate: {baud_rate}")
    print("="*70 + "\n")
    
    try:
        # Connect to vehicle
        print("Connecting to vehicle...")
        if connection_string.startswith('/dev/'):
            vehicle = connect(connection_string, wait_ready=True, baud=baud_rate)
        else:
            vehicle = connect(connection_string, wait_ready=True)
        
        print("✅ Connection successful!\n")
        
        # Get vehicle info
        print("="*70)
        print("📊 VEHICLE INFORMATION")
        print("="*70)
        print(f"Autopilot: {vehicle.version}")
        print(f"Mode: {vehicle.mode.name}")
        print(f"Armed: {vehicle.armed}")
        print(f"System status: {vehicle.system_status.state}")
        print(f"GPS: {vehicle.gps_0}")
        print(f"Battery: {vehicle.battery}")
        
        # Get attitude
        print("\n" + "="*70)
        print("🎯 CURRENT STATE")
        print("="*70)
        print(f"Position: Lat={vehicle.location.global_relative_frame.lat:.6f}, "
              f"Lon={vehicle.location.global_relative_frame.lon:.6f}, "
              f"Alt={vehicle.location.global_relative_frame.alt:.2f}m")
        print(f"Attitude: Roll={vehicle.attitude.roll:.3f}rad, "
              f"Pitch={vehicle.attitude.pitch:.3f}rad, "
              f"Yaw={vehicle.attitude.yaw:.3f}rad")
        print(f"Velocity: North={vehicle.velocity[0]:.2f}m/s, "
              f"East={vehicle.velocity[1]:.2f}m/s, "
              f"Down={vehicle.velocity[2]:.2f}m/s")
        print(f"Heading: {vehicle.heading}°")
        
        # Test data streaming
        print("\n" + "="*70)
        print("📡 TESTING DATA STREAM (5 seconds)")
        print("="*70)
        
        for i in range(5):
            print(f"[{i+1}/5] Alt={vehicle.location.global_relative_frame.alt:.2f}m, "
                  f"Roll={vehicle.attitude.roll:.3f}rad, "
                  f"Pitch={vehicle.attitude.pitch:.3f}rad")
            time.sleep(1)
        
        print("\n✅ Data streaming works!")
        
        # Close connection
        print("\n" + "="*70)
        print("Closing connection...")
        vehicle.close()
        print("✅ Connection test complete!")
        print("="*70 + "\n")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Connection failed: {e}\n")
        print("Troubleshooting:")
        print("  1. Check if ArduPilot SITL is running:")
        print("     sim_vehicle.py -v ArduCopter --console --map")
        print("  2. Check connection string (127.0.0.1:14550 for SITL)")
        print("  3. Check if device exists: ls /dev/tty*")
        print("  4. Check permissions: sudo chmod 666 /dev/ttyUSB0")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test connection to ArduPilot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # SITL (simulation)
  python 1_connection_test.py --connect 127.0.0.1:14550
  
  # Real drone via USB
  python 1_connection_test.py --connect /dev/ttyUSB0
  
  # Real drone via telemetry
  python 1_connection_test.py --connect /dev/ttyAMA0 --baud 57600
        """
    )
    
    parser.add_argument('--connect', type=str, default='127.0.0.1:14550',
                        help='Connection string (default: 127.0.0.1:14550 for SITL)')
    parser.add_argument('--baud', type=int, default=57600,
                        help='Baud rate for serial connection (default: 57600)')
    
    args = parser.parse_args()
    
    test_connection(args.connect, args.baud)