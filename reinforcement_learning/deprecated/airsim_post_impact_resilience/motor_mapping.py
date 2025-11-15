import airsim, time

client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

hover_pwm = 0.63
bump = 0.75

labels = ["M1", "M2", "M3", "M4"]
for i in range(4):
    print(f"\nTesting {labels[i]} ...")
    pwm = [hover_pwm] * 4
    pwm[i] = bump
    client.moveByMotorPWMsAsync(*pwm, duration=1.0).join()
    time.sleep(2)

client.armDisarm(False)
client.enableApiControl(False)
