import airsim
import time
import random
import numpy as np

class ExtremeWeatherTester:
    def __init__(self):
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        self.client.simEnableWeather(True)
        
    def test_wind_gust(self, duration=30):
        """Simulate sudden wind gusts"""
        print("Testing wind gust response...")
        self.client.takeoffAsync().join()
        
        for i in range(duration):
            if random.random() < 0.2:  # 20% chance of gust each second
                gust = airsim.Vector3r(
                    random.uniform(20, 40),
                    random.uniform(-15, 15),
                    random.uniform(-5, 5)
                )
                self.client.simSetWind(gust)
                print(f"GUST: {gust.x_val:.1f}, {gust.y_val:.1f}, {gust.z_val:.1f}")
                time.sleep(0.5)
                # Return to moderate wind
                self.client.simSetWind(airsim.Vector3r(10, 5, 0))
            time.sleep(1)
    
    def test_progressive_degradation(self):
        """Test how drone handles gradually worsening conditions"""
        print("Testing progressive weather degradation...")
        self.client.takeoffAsync().join()
        
        for wind_speed in range(0, 35, 5):
            print(f"Wind speed: {wind_speed} m/s")
            self.client.simSetWind(airsim.Vector3r(wind_speed, wind_speed*0.5, -wind_speed*0.2))
            self.client.simSetWeatherParameter(airsim.WeatherParameter.Rain, min(wind_speed/30.0, 1.0))
            time.sleep(10)
            
            # Check if drone is still controllable
            state = self.client.getMultirotorState()
            if abs(state.kinematics_estimated.position.z_val) > 50:
                print("FAIL: Drone lost altitude control!")
                break
    
    def test_all_extreme_conditions(self):
        """Run the full extreme weather test suite"""
        self.test_wind_gust()
        time.sleep(5)
        self.test_progressive_degradation()
        
        # Land
        self.client.landAsync().join()
        self.client.armDisarm(False)

# Run tests
tester = ExtremeWeatherTester()
tester.test_all_extreme_conditions()