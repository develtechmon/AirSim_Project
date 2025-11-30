how to run gazebo and not airsim version

roscore
rosrun gazebo_ros gazebo ~/ardupilot_gazebo/worlds/iris_arducopter_runway.world

sim_vehicle.py -v ArduCopter -f gazebo-iris --console --out=127.0.0.1:14550

python3 stage_3_dk/ppo_controller_gazebo.py --altitude 30
python3 stage_3_dk/disturbance_gazebo.py 
