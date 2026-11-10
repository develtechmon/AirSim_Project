#!/usr/bin/env bash
# =====================================================================================
# SETUP F450 (AirSim-matched) FOR GAZEBO CLASSIC 11 + ardupilot_gazebo   - run in WSL
# =====================================================================================
#
# VERSION 2 (23 Oct 2026) - final values verified in SITL (Stage 3 easy flips: 3/3 recovered)
#
# 1. WHAT IT DOES
#    | Creates                                              | What                                  |
#    |------------------------------------------------------|---------------------------------------|
#    | ~/ardupilot_gazebo/models/f450_airsim_base/           | F450 body: 1.0 kg, 455 mm X           |
#    | ~/ardupilot_gazebo/models/f450_airsim/                | + ArduPilot plugin, rotors            |
#    | ~/ardupilot_gazebo/worlds/f450_airsim_runway.world    | copy of the Iris runway world         |
#    | ~/sitl_f450/f450_airsim_match.parm                    | AirSim-matched ArduPilot settings     |
#    Your Iris files are NOT changed.
#
# 2. HOW TO RUN (WSL, file in your home folder)
#      bash ~/setup_f450_airsim.sh
#
# 3. EXPECTED RESULT
#      "OK: world uses model://f450_airsim"  and  "Setup complete"
#      If it says "WARNING: world does not include model://iris_with_ardupilot" -> send the
#      output of:  grep -n "iris" ~/ardupilot_gazebo/worlds/iris_arducopter_runway.world
#
# 4. FINAL MODEL VALUES (vs Iris)
#    | Item                        | Iris            | f450_airsim (this script)          |
#    |-----------------------------|-----------------|------------------------------------|
#    | Total mass                  | ~2.4 kg + gimbal| 1.0 kg  (gz model check: 1.0)      |
#    | Rotor positions             | x 0.13 y 0.22   | +/-0.1609 m (F450 X)               |
#    | Blade area (8 blades)       | 0.002           | 0.00061 -> T/W 1.63 (AirSim 1.62)  |
#    | Blade drag cda              | 0.10            | 0.67 -> yaw torque = AirSim        |
#    | Motor torque cap cmd_max    | 2.5 (ceiling!)  | 5.0                                |
#    | Prop inertia izz            | 1.676e-4        | 5.7e-5 (real 9-inch prop)          |
#    | Gimbal                      | yes             | none                               |
#
# 5. AFTER INSTALL (once): load the ArduPilot settings in MAVProxy, then reboot
#      param load /home/<you>/sitl_f450/f450_airsim_match.parm
#      reboot
# =====================================================================================
set -e
G=~/ardupilot_gazebo
[ -d "$G/models" ] || { echo "ERROR: $G/models not found"; exit 1; }
mkdir -p "$G/models/f450_airsim_base" "$G/models/f450_airsim"
cat > "$G/models/f450_airsim_base/model.sdf" <<'F450EOF'
<?xml version='1.0'?>
<sdf version='1.5'>
  <!-- F450 matched to AirSim's default quad (AirLib MultiRotorParams.hpp setupFrameGenericQuad + RotorParams.hpp)
       total mass 1.0 kg | arm 0.2275 m (X) | inertia total ~0.0067 / 0.0080 / 0.0143 kg m2
       Nested model keeps the name 'iris' so joint/link names used by the ArduPilot plugin stay valid.
       Rotor (prop + motor bell) inertia: Iris 1.676e-4 -> 5.7e-5 kg m2 (real 9-inch prop), ixx = iyy = 2.85e-5
       (heavy Iris props acted like gyroscopes and stalled the flip near upside-down). -->
  <model name='iris'>
    <pose>0 0 0.194923 0 0 0</pose>
    <link name='base_link'>
      <enable_wind>true</enable_wind>
      <wind>10 10 0</wind>
      <velocity_decay>
        <linear>0.0</linear>
        <angular>0.0</angular>
      </velocity_decay>
      <inertial>
        <pose>0 0 0 0 0 0</pose>
        <mass>0.898</mass>  <!-- 1.0 kg total - 4 rotors (0.1) - sensor links (0.002) -->
        <inertia>             <!-- AirSim total minus what the 4 rotor links add -->
          <ixx>0.0037</ixx>
          <ixy>0</ixy>
          <ixz>0</ixz>
          <iyy>0.0050</iyy>
          <iyz>0</iyz>
          <izz>0.0084</izz>
        </inertia>
      </inertial>
      <collision name='base_link_collision'>
        <pose>0 0 -0.08 0 0 0</pose>
        <geometry>
          <box>
            <size>0.34 0.34 0.20</size>
          </box>
        </geometry>
        <surface>
          <contact>
            <ode>
              <max_vel>100.0</max_vel>
              <min_depth>0.001</min_depth>
            </ode>
          </contact>
          <friction>
            <ode>
              <mu>100000.0</mu>
              <mu2>100000.0</mu2>
            </ode>
          </friction>
        </surface>
      </collision>
      <visual name='base_link_visual'>
        <geometry>
          <mesh>
            <uri>model://iris_with_standoffs/meshes/iris.dae</uri>
          </mesh>
        </geometry>
        <material>
          <script>
            <name>Gazebo/DarkGrey</name>
          </script>
        </material>
      </visual>

      <visual name='front_left_leg_visual'>
        <pose>0.10 0.10 -0.11 0 0 0</pose>
        <geometry>
          <cylinder>
            <radius>0.005</radius>
            <length>0.17</length>
          </cylinder>
        </geometry>
        <material>
          <script>
            <name>Gazebo/FlatBlack</name>
            <uri>file://media/materials/scripts/gazebo.material</uri>
          </script>
        </material>
      </visual>
      <visual name='front_right_leg_visual'>
        <pose>0.10 -0.10 -0.11 0 0 0</pose>
        <geometry>
          <cylinder>
            <radius>0.005</radius>
            <length>0.17</length>
          </cylinder>
        </geometry>
        <material>
          <script>
            <name>Gazebo/FlatBlack</name>
            <uri>file://media/materials/scripts/gazebo.material</uri>
          </script>
        </material>
      </visual>
      <visual name='rear_left_leg_visual'>
        <pose>-0.10 0.10 -0.11 0 0 0</pose>
        <geometry>
          <cylinder>
            <radius>0.005</radius>
            <length>0.17</length>
          </cylinder>
        </geometry>
        <material>
          <script>
            <name>Gazebo/FlatBlack</name>
            <uri>file://media/materials/scripts/gazebo.material</uri>
          </script>
        </material>
      </visual>
      <visual name='rear_right_leg_visual'>
        <pose>-0.10 -0.10 -0.11 0 0 0</pose>
        <geometry>
          <cylinder>
            <radius>0.005</radius>
            <length>0.17</length>
          </cylinder>
        </geometry>
        <material>
          <script>
            <name>Gazebo/FlatBlack</name>
            <uri>file://media/materials/scripts/gazebo.material</uri>
          </script>
        </material>
      </visual>
    </link>
    <link name='iris/ground_truth/odometry_sensorgt_link'>
      <pose>0 0 0 0 -0 0</pose>
      <inertial>
        <pose>0 0 0 0 -0 0</pose>
        <mass>0.001</mass>
        <inertia>
          <ixx>1e-07</ixx>
          <ixy>0</ixy>
          <ixz>0</ixz>
          <iyy>1e-07</iyy>
          <iyz>0</iyz>
          <izz>1e-07</izz>
        </inertia>
      </inertial>
    </link>
    <joint name='iris/ground_truth/odometry_sensorgt_joint' type='revolute'>
      <child>iris/ground_truth/odometry_sensorgt_link</child>
      <parent>base_link</parent>
      <axis>
        <xyz>0 0 1</xyz>
        <limit>
          <lower>0</lower>
          <upper>0</upper>
          <effort>0</effort>
          <velocity>0</velocity>
        </limit>
        <dynamics>
          <damping>1.0</damping>
        </dynamics>
        <use_parent_model_frame>1</use_parent_model_frame>
      </axis>
      <physics>
        <ode>
          <implicit_spring_damper>1</implicit_spring_damper>
        </ode>
      </physics>
    </joint>
    <link name='iris/imu_link'>
      <inertial>
        <pose>0 0 0 0 0 0</pose>
        <mass>0.001</mass>
        <inertia>
          <ixx>1e-07</ixx>
          <ixy>0</ixy>
          <ixz>0</ixz>
          <iyy>1e-07</iyy>
          <iyz>0</iyz>
          <izz>1e-07</izz>
        </inertia>
      </inertial>
      <sensor name="imu_sensor" type="imu">
        <pose>0 0 0 3.141593 0 0</pose>
        <always_on>1</always_on>
        <update_rate>1000.0</update_rate>
      </sensor>
    </link>
    <joint name='iris/imu_joint' type='revolute'>
      <child>iris/imu_link</child>
      <parent>base_link</parent>
      <axis>
        <xyz>0 0 1</xyz>
        <limit>
          <lower>0</lower>
          <upper>0</upper>
          <effort>0</effort>
          <velocity>0</velocity>
        </limit>
        <dynamics>
          <damping>1.0</damping>
        </dynamics>
        <use_parent_model_frame>1</use_parent_model_frame>
      </axis>
      <physics>
        <ode>
          <implicit_spring_damper>1</implicit_spring_damper>
        </ode>
      </physics>
    </joint>
<!--
    <link name='iris/imugt_link'>
      <pose>0 0 0 0 -0 0</pose>
      <inertial>
        <pose>0 0 0 0 -0 0</pose>
        <mass>1e-02</mass>
        <inertia>
          <ixx>0.0002</ixx>
          <ixy>0</ixy>
          <ixz>0</ixz>
          <iyy>0.0002</iyy>
          <iyz>0</iyz>
          <izz>0.0002</izz>
        </inertia>
      </inertial>
    </link>
    <joint name='iris/imugt_joint' type='revolute'>
      <child>iris/imugt_link</child>
      <parent>base_link</parent>
      <axis>
        <xyz>0 0 1</xyz>
        <limit>
          <lower>0</lower>
          <upper>0</upper>
          <effort>0</effort>
          <velocity>0</velocity>
        </limit>
        <dynamics>
          <spring_reference>0</spring_reference>
          <spring_stiffness>0</spring_stiffness>
        </dynamics>
        <use_parent_model_frame>1</use_parent_model_frame>
      </axis>
    </joint>
-->
    <link name='rotor_0'>
      <pose>0.1609 -0.1609 0.023 0 -0 0</pose>
      <inertial>
        <pose>0 0 0 0 -0 0</pose>
        <mass>0.025</mass>
        <inertia>
          <ixx>2.85e-05</ixx>
          <ixy>0</ixy>
          <ixz>0</ixz>
          <iyy>2.85e-05</iyy>
          <iyz>0</iyz>
          <izz>5.7e-05</izz>
        </inertia>
      </inertial>
      <collision name='rotor_0_collision'>
        <pose>0 0 0 0 -0 0</pose>
        <geometry>
          <cylinder>
            <length>0.005</length>
            <radius>0.1</radius>
          </cylinder>
        </geometry>
        <surface>
          <contact>
            <ode/>
          </contact>
          <friction>
            <ode/>
          </friction>
        </surface>
      </collision>
      <visual name='rotor_0_visual'>
        <pose>0 0 0 0 -0 0</pose>
        <geometry>
          <mesh>
            <scale>1 1 1</scale>
            <uri>model://iris_with_standoffs/meshes/iris_prop_ccw.dae</uri>
          </mesh>
        </geometry>
        <material>
          <script>
            <name>Gazebo/Blue</name>
            <uri>__default__</uri>
          </script>
        </material>
      </visual>
      <gravity>1</gravity>
      <velocity_decay/>
      <self_collide>0</self_collide>
    </link>
    <joint name='rotor_0_joint' type='revolute'>
      <child>rotor_0</child>
      <parent>base_link</parent>
      <axis>
        <xyz>0 0 1</xyz>
        <limit>
          <lower>-1e+16</lower>
          <upper>1e+16</upper>
        </limit>
        <dynamics>
          <damping>0.004</damping>
        </dynamics>
        <use_parent_model_frame>1</use_parent_model_frame>
      </axis>
      <physics>
        <ode>
          <implicit_spring_damper>1</implicit_spring_damper>
        </ode>
      </physics>
    </joint>
    <link name='rotor_1'>
      <pose>-0.1609 0.1609 0.023 0 -0 0</pose>
      <inertial>
        <pose>0 0 0 0 -0 0</pose>
        <mass>0.025</mass>
        <inertia>
          <ixx>2.85e-05</ixx>
          <ixy>0</ixy>
          <ixz>0</ixz>
          <iyy>2.85e-05</iyy>
          <iyz>0</iyz>
          <izz>5.7e-05</izz>
        </inertia>
      </inertial>
      <collision name='rotor_1_collision'>
        <pose>0 0 0 0 -0 0</pose>
        <geometry>
          <cylinder>
            <length>0.005</length>
            <radius>0.1</radius>
          </cylinder>
        </geometry>
        <surface>
          <contact>
            <ode/>
          </contact>
          <friction>
            <ode/>
          </friction>
        </surface>
      </collision>
      <visual name='rotor_1_visual'>
        <pose>0 0 0 0 -0 0</pose>
        <geometry>
          <mesh>
            <scale>1 1 1</scale>
            <uri>model://iris_with_standoffs/meshes/iris_prop_ccw.dae</uri>
          </mesh>
        </geometry>
        <material>
          <script>
            <name>Gazebo/DarkGrey</name>
            <uri>__default__</uri>
          </script>
        </material>
      </visual>
      <gravity>1</gravity>
      <velocity_decay/>
      <self_collide>0</self_collide>
    </link>
    <joint name='rotor_1_joint' type='revolute'>
      <child>rotor_1</child>
      <parent>base_link</parent>
      <axis>
        <xyz>0 0 1</xyz>
        <limit>
          <lower>-1e+16</lower>
          <upper>1e+16</upper>
        </limit>
        <dynamics>
          <damping>0.004</damping>
        </dynamics>
        <use_parent_model_frame>1</use_parent_model_frame>
      </axis>
      <physics>
        <ode>
          <implicit_spring_damper>1</implicit_spring_damper>
        </ode>
      </physics>
    </joint>
    <link name='rotor_2'>
      <pose>0.1609 0.1609 0.023 0 -0 0</pose>
      <inertial>
        <pose>0 0 0 0 -0 0</pose>
        <mass>0.025</mass>
        <inertia>
          <ixx>2.85e-05</ixx>
          <ixy>0</ixy>
          <ixz>0</ixz>
          <iyy>2.85e-05</iyy>
          <iyz>0</iyz>
          <izz>5.7e-05</izz>
        </inertia>
      </inertial>
      <collision name='rotor_2_collision'>
        <pose>0 0 0 0 -0 0</pose>
        <geometry>
          <cylinder>
            <length>0.005</length>
            <radius>0.1</radius>
          </cylinder>
        </geometry>
        <surface>
          <contact>
            <ode/>
          </contact>
          <friction>
            <ode/>
          </friction>
        </surface>
      </collision>
      <visual name='rotor_2_visual'>
        <pose>0 0 0 0 -0 0</pose>
        <geometry>
          <mesh>
            <scale>1 1 1</scale>
            <uri>model://iris_with_standoffs/meshes/iris_prop_cw.dae</uri>
          </mesh>
        </geometry>
        <material>
          <script>
            <name>Gazebo/Blue</name>
            <uri>__default__</uri>
          </script>
        </material>
      </visual>
      <gravity>1</gravity>
      <velocity_decay/>
      <self_collide>0</self_collide>
    </link>
    <joint name='rotor_2_joint' type='revolute'>
      <child>rotor_2</child>
      <parent>base_link</parent>
      <axis>
        <xyz>0 0 1</xyz>
        <limit>
          <lower>-1e+16</lower>
          <upper>1e+16</upper>
        </limit>
        <dynamics>
          <damping>0.004</damping>
        </dynamics>
        <use_parent_model_frame>1</use_parent_model_frame>
      </axis>
      <physics>
        <ode>
          <implicit_spring_damper>1</implicit_spring_damper>
        </ode>
      </physics>
    </joint>
    <link name='rotor_3'>
      <pose>-0.1609 -0.1609 0.023 0 -0 0</pose>
      <inertial>
        <pose>0 0 0 0 -0 0</pose>
        <mass>0.025</mass>
        <inertia>
          <ixx>2.85e-05</ixx>
          <ixy>0</ixy>
          <ixz>0</ixz>
          <iyy>2.85e-05</iyy>
          <iyz>0</iyz>
          <izz>5.7e-05</izz>
        </inertia>
      </inertial>
      <collision name='rotor_3_collision'>
        <pose>0 0 0 0 -0 0</pose>
        <geometry>
          <cylinder>
            <length>0.005</length>
            <radius>0.1</radius>
          </cylinder>
        </geometry>
        <surface>
          <contact>
            <ode/>
          </contact>
          <friction>
            <ode/>
          </friction>
        </surface>
      </collision>
      <visual name='rotor_3_visual'>
        <pose>0 0 0 0 -0 0</pose>
        <geometry>
          <mesh>
            <scale>1 1 1</scale>
            <uri>model://iris_with_standoffs/meshes/iris_prop_cw.dae</uri>
          </mesh>
        </geometry>
        <material>
          <script>
            <name>Gazebo/DarkGrey</name>
            <uri>__default__</uri>
          </script>
        </material>
      </visual>
      <gravity>1</gravity>
      <velocity_decay/>
      <self_collide>0</self_collide>
    </link>
    <joint name='rotor_3_joint' type='revolute'>
      <child>rotor_3</child>
      <parent>base_link</parent>
      <axis>
        <xyz>0 0 1</xyz>
        <limit>
          <lower>-1e+16</lower>
          <upper>1e+16</upper>
        </limit>
        <dynamics>
          <damping>0.004</damping>
        </dynamics>
        <use_parent_model_frame>1</use_parent_model_frame>
      </axis>
      <physics>
        <ode>
          <implicit_spring_damper>1</implicit_spring_damper>
        </ode>
      </physics>
    </joint>
    <static>0</static>
  </model>
</sdf>
F450EOF
cat > "$G/models/f450_airsim_base/model.config" <<'F450EOF'
<?xml version="1.0"?>
<model>
  <name>f450_airsim_base</name>
  <version>1.0</version>
  <sdf version="1.6">model.sdf</sdf>
  <author><name>Lukas Johnny</name></author>
  <description>F450 body matched to AirSim default quad (1.0 kg, 450 mm)</description>
</model>
F450EOF
cat > "$G/models/f450_airsim/model.sdf" <<'F450EOF'
<?xml version='1.0'?>
<sdf version="1.6" xmlns:xacro='http://ros.org/wiki/xacro'>
  <model name="iris_demo_test">
    <!-- F450 matched to AirSim (see f450_airsim_base). No gimbal (AirSim's quad has none).
         Model name kept as iris_demo_test so imuName below and your world files stay valid.
         Rotor thrust: blade area 0.002 -> 0.00061  => thrust-to-weight 1.63 at full throttle (AirSim 1.62)
                                                       verified in SITL: hover PWM 1741-1742 (predicted 1745)
         Rotor yaw torque: cda 0.10 -> 0.67          => ~0.056 N m per rotor (AirSim max_torque)
         Motor speed-up: cmd_max 2.5 -> 5.0          => faster rotor response (AirSim: 5 ms filter) -->
    <include>
      <uri>model://f450_airsim_base</uri>
    </include>
    

    <!-- visual markers for debugging
    <link name="rotor_0_blade_1_cp">
      <gravity>0</gravity>
      <pose>0.13 -0.22 0.216 0 -0 0</pose>
      <visual name='rotor_0_visual_root'>
        <pose>0 0 0 0 -0 0</pose>
        <geometry>
          <sphere>
            <radius>0.01</radius>
          </sphere>
        </geometry>
      </visual>
      <visual name='rotor_0_visual_tip'>
        <pose>0.12 0 0 0 -0 0</pose>
        <geometry>
          <sphere>
            <radius>0.01</radius>
          </sphere>
        </geometry>
      </visual>
      <visual name='rotor_0_visual_cp'>
        <pose>0.084 0 0 0 -0 0</pose>
        <geometry>
          <sphere>
            <radius>0.003</radius>
          </sphere>
        </geometry>
      </visual>
      <visual name='rotor_0_visual_cp_forward'>
        <pose>0.084 0.02 0 0 -0 0</pose>
        <geometry>
          <sphere>
            <radius>0.003</radius>
          </sphere>
        </geometry>
      </visual>
      <visual name='rotor_0_visual_cp_upward'>
        <pose>0.084 0 0.02 0 -0 0</pose>
        <geometry>
          <sphere>
            <radius>0.003</radius>
          </sphere>
        </geometry>
      </visual>
    </link>
    <link name="rotor_0_blade_2_cp">
      <gravity>0</gravity>
      <pose>0.13 -0.22 0.216 0 -0 0</pose>
      <visual name='rotor_0_visual_root'>
        <pose>0 0 0 0 -0 0</pose>
        <geometry>
          <sphere>
            <radius>0.01</radius>
          </sphere>
        </geometry>
      </visual>
      <visual name='rotor_0_visual_tip'>
        <pose>-0.12 0 0 0 -0 0</pose>
        <geometry>
          <sphere>
            <radius>0.01</radius>
          </sphere>
        </geometry>
      </visual>
      <visual name='rotor_0_visual_cp'>
        <pose>-0.084 0 0 0 -0 0</pose>
        <geometry>
          <sphere>
            <radius>0.003</radius>
          </sphere>
        </geometry>
      </visual>
      <visual name='rotor_0_visual_cp_forward'>
        <pose>-0.084 -0.02 0 0 -0 0</pose>
        <geometry>
          <sphere>
            <radius>0.003</radius>
          </sphere>
        </geometry>
      </visual>
      <visual name='rotor_0_visual_cp_upward'>
        <pose>-0.084 0 0.02 0 -0 0</pose>
        <geometry>
          <sphere>
            <radius>0.003</radius>
          </sphere>
        </geometry>
      </visual>
    </link>

    <link name="rotor_1_blade_1_cp">
      <gravity>0</gravity>
      <pose>-0.13 0.2 0.216 0 -0 0</pose>
      <visual name='rotor_1_visual_root'>
        <pose>0 0 0 0 -0 0</pose>
        <geometry>
          <sphere>
            <radius>0.01</radius>
          </sphere>
        </geometry>
      </visual>
      <visual name='rotor_1_visual_tip'>
        <pose>0.12 0 0 0 -0 0</pose>
        <geometry>
          <sphere>
            <radius>0.01</radius>
          </sphere>
        </geometry>
      </visual>
      <visual name='rotor_1_visual_cp'>
        <pose>0.084 0 0 0 -0 0</pose>
        <geometry>
          <sphere>
            <radius>0.003</radius>
          </sphere>
        </geometry>
      </visual>
      <visual name='rotor_1_visual_cp_forward'>
        <pose>0.084 0.02 0 0 -0 0</pose>
        <geometry>
          <sphere>
            <radius>0.003</radius>
          </sphere>
        </geometry>
      </visual>
      <visual name='rotor_1_visual_cp_upward'>
        <pose>0.084 0 0.02 0 -0 0</pose>
        <geometry>
          <sphere>
            <radius>0.003</radius>
          </sphere>
        </geometry>
      </visual>
    </link>
    <link name="rotor_1_blade_2_cp">
      <gravity>0</gravity>
      <pose>-0.13 0.2 0.216 0 -0 0</pose>
      <visual name='rotor_1_visual_root'>
        <pose>0 0 0 0 -0 0</pose>
        <geometry>
          <sphere>
            <radius>0.01</radius>
          </sphere>
        </geometry>
      </visual>
      <visual name='rotor_1_visual_tip'>
        <pose>-0.12 0 0 0 -0 0</pose>
        <geometry>
          <sphere>
            <radius>0.01</radius>
          </sphere>
        </geometry>
      </visual>
      <visual name='rotor_1_visual_cp'>
        <pose>-0.084 0 0 0 -0 0</pose>
        <geometry>
          <sphere>
            <radius>0.003</radius>
          </sphere>
        </geometry>
      </visual>
      <visual name='rotor_1_visual_cp_forward'>
        <pose>-0.084 -0.02 0 0 -0 0</pose>
        <geometry>
          <sphere>
            <radius>0.003</radius>
          </sphere>
        </geometry>
      </visual>
      <visual name='rotor_1_visual_cp_upward'>
        <pose>-0.084 0 0.02 0 -0 0</pose>
        <geometry>
          <sphere>
            <radius>0.003</radius>
          </sphere>
        </geometry>
      </visual>
    </link>

    <link name="rotor_2_blade_1_cp">
      <gravity>0</gravity>
      <pose>0.13 0.22 0.216 0 -0 0</pose>
      <visual name='rotor_2_visual_root'>
        <pose>0 0 0 0 -0 0</pose>
        <geometry>
          <sphere>
            <radius>0.01</radius>
          </sphere>
        </geometry>
      </visual>
      <visual name='rotor_2_visual_tip'>
        <pose>0.12 0 0 0 -0 0</pose>
        <geometry>
          <sphere>
            <radius>0.01</radius>
          </sphere>
        </geometry>
      </visual>
      <visual name='rotor_2_visual_cp'>
        <pose>0.084 0 0 0 -0 0</pose>
        <geometry>
          <sphere>
            <radius>0.003</radius>
          </sphere>
        </geometry>
      </visual>
      <visual name='rotor_2_visual_cp_forward'>
        <pose>0.084 -0.02 0 0 -0 0</pose>
        <geometry>
          <sphere>
            <radius>0.003</radius>
          </sphere>
        </geometry>
      </visual>
      <visual name='rotor_2_visual_cp_upward'>
        <pose>0.084 0 0.02 0 -0 0</pose>
        <geometry>
          <sphere>
            <radius>0.003</radius>
          </sphere>
        </geometry>
      </visual>
    </link>
    <link name="rotor_2_blade_2_cp">
      <gravity>0</gravity>
      <pose>0.13 0.22 0.216 0 -0 0</pose>
      <visual name='rotor_2_visual_root'>
        <pose>0 0 0 0 -0 0</pose>
        <geometry>
          <sphere>
            <radius>0.01</radius>
          </sphere>
        </geometry>
      </visual>
      <visual name='rotor_2_visual_tip'>
        <pose>-0.12 0 0 0 -0 0</pose>
        <geometry>
          <sphere>
            <radius>0.01</radius>
          </sphere>
        </geometry>
      </visual>
      <visual name='rotor_2_visual_cp'>
        <pose>-0.084 0 0 0 -0 0</pose>
        <geometry>
          <sphere>
            <radius>0.003</radius>
          </sphere>
        </geometry>
      </visual>
      <visual name='rotor_2_visual_cp_forward'>
        <pose>-0.084 0.02 0 0 -0 0</pose>
        <geometry>
          <sphere>
            <radius>0.003</radius>
          </sphere>
        </geometry>
      </visual>
      <visual name='rotor_2_visual_cp_upward'>
        <pose>-0.084 0 0.02 0 -0 0</pose>
        <geometry>
          <sphere>
            <radius>0.003</radius>
          </sphere>
        </geometry>
      </visual>
    </link>

    <link name="rotor_3_blade_1_cp">
      <gravity>0</gravity>
      <pose>-0.13 -0.2 0.216 0 -0 0</pose>
      <visual name='rotor_3_visual_root'>
        <pose>0 0 0 0 -0 0</pose>
        <geometry>
          <sphere>
            <radius>0.01</radius>
          </sphere>
        </geometry>
      </visual>
      <visual name='rotor_3_visual_tip'>
        <pose>0.12 0 0 0 -0 0</pose>
        <geometry>
          <sphere>
            <radius>0.01</radius>
          </sphere>
        </geometry>
      </visual>
      <visual name='rotor_3_visual_cp'>
        <pose>0.084 0 0 0 -0 0</pose>
        <geometry>
          <sphere>
            <radius>0.003</radius>
          </sphere>
        </geometry>
      </visual>
      <visual name='rotor_3_visual_cp_forward'>
        <pose>0.084 -0.02 0 0 -0 0</pose>
        <geometry>
          <sphere>
            <radius>0.003</radius>
          </sphere>
        </geometry>
      </visual>
      <visual name='rotor_3_visual_cp_upward'>
        <pose>0.084 0 0.02 0 -0 0</pose>
        <geometry>
          <sphere>
            <radius>0.003</radius>
          </sphere>
        </geometry>
      </visual>
    </link>
    <link name="rotor_3_blade_2_cp">
      <gravity>0</gravity>
      <pose>-0.13 -0.2 0.216 0 -0 0</pose>
      <visual name='rotor_3_visual_root'>
        <pose>0 0 0 0 -0 0</pose>
        <geometry>
          <sphere>
            <radius>0.01</radius>
          </sphere>
        </geometry>
      </visual>
      <visual name='rotor_3_visual_tip'>
        <pose>-0.12 0 0 0 -0 0</pose>
        <geometry>
          <sphere>
            <radius>0.01</radius>
          </sphere>
        </geometry>
      </visual>
      <visual name='rotor_3_visual_cp'>
        <pose>-0.084 0 0 0 -0 0</pose>
        <geometry>
          <sphere>
            <radius>0.003</radius>
          </sphere>
        </geometry>
      </visual>
      <visual name='rotor_3_visual_cp_forward'>
        <pose>-0.084 0.02 0 0 -0 0</pose>
        <geometry>
          <sphere>
            <radius>0.003</radius>
          </sphere>
        </geometry>
      </visual>
      <visual name='rotor_3_visual_cp_upward'>
        <pose>-0.084 0 0.02 0 -0 0</pose>
        <geometry>
          <sphere>
            <radius>0.003</radius>
          </sphere>
        </geometry>
      </visual>
    </link>
    -->

    <!-- plugins -->
    <plugin name="rotor_0_blade_1" filename="libLiftDragPlugin.so">
      <a0>0.3</a0>
      <alpha_stall>1.4</alpha_stall>
      <cla>4.2500</cla>
      <cda>0.67</cda>
      <cma>0.00</cma>
      <cla_stall>-0.025</cla_stall>
      <cda_stall>0.0</cda_stall>
      <cma_stall>0.0</cma_stall>
      <area>0.00061</area>
      <air_density>1.2041</air_density>
      <cp>0.084 0 0</cp>
      <forward>0 1 0</forward>
      <upward>0 0 1</upward>
      <link_name>iris::rotor_0</link_name>
    </plugin>
    <plugin name="rotor_0_blade_2" filename="libLiftDragPlugin.so">
      <a0>0.3</a0>
      <alpha_stall>1.4</alpha_stall>
      <cla>4.2500</cla>
      <cda>0.67</cda>
      <cma>0.00</cma>
      <cla_stall>-0.025</cla_stall>
      <cda_stall>0.0</cda_stall>
      <cma_stall>0.0</cma_stall>
      <area>0.00061</area>
      <air_density>1.2041</air_density>
      <cp>-0.084 0 0</cp>
      <forward>0 -1 0</forward>
      <upward>0 0 1</upward>
      <link_name>iris::rotor_0</link_name>
    </plugin>

    <plugin name="rotor_1_blade_1" filename="libLiftDragPlugin.so">
      <a0>0.3</a0>
      <alpha_stall>1.4</alpha_stall>
      <cla>4.2500</cla>
      <cda>0.67</cda>
      <cma>0.00</cma>
      <cla_stall>-0.025</cla_stall>
      <cda_stall>0.0</cda_stall>
      <cma_stall>0.0</cma_stall>
      <area>0.00061</area>
      <air_density>1.2041</air_density>
      <cp>0.084 0 0</cp>
      <forward>0 1 0</forward>
      <upward>0 0 1</upward>
      <link_name>iris::rotor_1</link_name>
    </plugin>
    <plugin name="rotor_1_blade_2" filename="libLiftDragPlugin.so">
      <a0>0.3</a0>
      <alpha_stall>1.4</alpha_stall>
      <cla>4.2500</cla>
      <cda>0.67</cda>
      <cma>0.00</cma>
      <cla_stall>-0.025</cla_stall>
      <cda_stall>0.0</cda_stall>
      <cma_stall>0.0</cma_stall>
      <area>0.00061</area>
      <air_density>1.2041</air_density>
      <cp>-0.084 0 0</cp>
      <forward>0 -1 0</forward>
      <upward>0 0 1</upward>
      <link_name>iris::rotor_1</link_name>
    </plugin>

    <plugin name="rotor_2_blade_1" filename="libLiftDragPlugin.so">
      <a0>0.3</a0>
      <alpha_stall>1.4</alpha_stall>
      <cla>4.2500</cla>
      <cda>0.67</cda>
      <cma>0.00</cma>
      <cla_stall>-0.025</cla_stall>
      <cda_stall>0.0</cda_stall>
      <cma_stall>0.0</cma_stall>
      <area>0.00061</area>
      <air_density>1.2041</air_density>
      <cp>0.084 0 0</cp>
      <forward>0 -1 0</forward>
      <upward>0 0 1</upward>
      <link_name>iris::rotor_2</link_name>
    </plugin>
    <plugin name="rotor_2_blade_2" filename="libLiftDragPlugin.so">
      <a0>0.3</a0>
      <alpha_stall>1.4</alpha_stall>
      <cla>4.2500</cla>
      <cda>0.67</cda>
      <cma>0.00</cma>
      <cla_stall>-0.025</cla_stall>
      <cda_stall>0.0</cda_stall>
      <cma_stall>0.0</cma_stall>
      <area>0.00061</area>
      <air_density>1.2041</air_density>
      <cp>-0.084 0 0</cp>
      <forward>0 1 0</forward>
      <upward>0 0 1</upward>
      <link_name>iris::rotor_2</link_name>
    </plugin>

    <plugin name="rotor_3_blade_1" filename="libLiftDragPlugin.so">
      <a0>0.3</a0>
      <alpha_stall>1.4</alpha_stall>
      <cla>4.2500</cla>
      <cda>0.67</cda>
      <cma>0.00</cma>
      <cla_stall>-0.025</cla_stall>
      <cda_stall>0.0</cda_stall>
      <cma_stall>0.0</cma_stall>
      <area>0.00061</area>
      <air_density>1.2041</air_density>
      <cp>0.084 0 0</cp>
      <forward>0 -1 0</forward>
      <upward>0 0 1</upward>
      <link_name>iris::rotor_3</link_name>
    </plugin>
    <plugin name="rotor_3_blade_2" filename="libLiftDragPlugin.so">
      <a0>0.3</a0>
      <alpha_stall>1.4</alpha_stall>
      <cla>4.2500</cla>
      <cda>0.67</cda>
      <cma>0.00</cma>
      <cla_stall>-0.025</cla_stall>
      <cda_stall>0.0</cda_stall>
      <cma_stall>0.0</cma_stall>
      <area>0.00061</area>
      <air_density>1.2041</air_density>
      <cp>-0.084 0 0</cp>
      <forward>0 1 0</forward>
      <upward>0 0 1</upward>
      <link_name>iris::rotor_3</link_name>
    </plugin>
    <plugin name="arducopter_plugin" filename="libArduPilotPlugin.so">
      <fdm_addr>127.0.0.1</fdm_addr>
      <fdm_port_in>9002</fdm_port_in>
      <fdm_port_out>9003</fdm_port_out>
      <!--
          Require by APM :
          Only change model and gazebo from XYZ to XY-Z coordinates
      -->
      <modelXYZToAirplaneXForwardZDown>0 0 0 3.141593 0 0</modelXYZToAirplaneXForwardZDown>
      <gazeboXYZToNED>0 0 0 3.141593 0 0</gazeboXYZToNED>
      <imuName>iris_demo_test::iris::iris/imu_link::imu_sensor</imuName>
      <connectionTimeoutMaxCount>5</connectionTimeoutMaxCount>
      <control channel="0">
      <!--
          incoming control command [0, 1]
          so offset it by 0 to get [0, 1]
          and divide max target by 1.
          offset = 0
          multiplier = 838 max rpm / 1 = 838
        -->
        <type>VELOCITY</type>
        <offset>0</offset>
        <p_gain>0.20</p_gain>
        <i_gain>0</i_gain>
        <d_gain>0</d_gain>
        <i_max>0</i_max>
        <i_min>0</i_min>
        <cmd_max>5.0</cmd_max>
        <cmd_min>-5.0</cmd_min>
        <jointName>iris::rotor_0_joint</jointName>
        <multiplier>838</multiplier>
        <controlVelocitySlowdownSim>1</controlVelocitySlowdownSim>
      </control>
      <control channel="1">
        <type>VELOCITY</type>
        <offset>0</offset>
        <p_gain>0.20</p_gain>
        <i_gain>0</i_gain>
        <d_gain>0</d_gain>
        <i_max>0</i_max>
        <i_min>0</i_min>
        <cmd_max>5.0</cmd_max>
        <cmd_min>-5.0</cmd_min>
        <jointName>iris::rotor_1_joint</jointName>
        <multiplier>838</multiplier>
        <controlVelocitySlowdownSim>1</controlVelocitySlowdownSim>
      </control>
      <control channel="2">
        <type>VELOCITY</type>
        <offset>0</offset>
        <p_gain>0.20</p_gain>
        <i_gain>0</i_gain>
        <d_gain>0</d_gain>
        <i_max>0</i_max>
        <i_min>0</i_min>
        <cmd_max>5.0</cmd_max>
        <cmd_min>-5.0</cmd_min>
        <jointName>iris::rotor_2_joint</jointName>
        <multiplier>-838</multiplier>
        <controlVelocitySlowdownSim>1</controlVelocitySlowdownSim>
      </control>
      <control channel="3">
        <type>VELOCITY</type>
        <offset>0</offset>
        <p_gain>0.20</p_gain>
        <i_gain>0</i_gain>
        <d_gain>0</d_gain>
        <i_max>0</i_max>
        <i_min>0</i_min>
        <cmd_max>5.0</cmd_max>
        <cmd_min>-5.0</cmd_min>
        <jointName>iris::rotor_3_joint</jointName>
        <multiplier>-838</multiplier>
        <controlVelocitySlowdownSim>1</controlVelocitySlowdownSim>
      </control>
    </plugin>

  </model>
</sdf>
F450EOF
cat > "$G/models/f450_airsim/model.config" <<'F450EOF'
<?xml version="1.0"?>
<model>
  <name>f450_airsim</name>
  <version>1.0</version>
  <sdf version="1.6">model.sdf</sdf>
  <author><name>Lukas Johnny</name></author>
  <description>F450 (AirSim-matched) with ArduPilot plugin</description>
</model>
F450EOF
echo "Models written: $G/models/f450_airsim_base, $G/models/f450_airsim"
SRC="$G/worlds/iris_arducopter_runway.world"
DST="$G/worlds/f450_airsim_runway.world"
if grep -q "model://iris_with_ardupilot" "$SRC"; then
  sed 's#model://iris_with_ardupilot#model://f450_airsim#g' "$SRC" > "$DST"
  grep -q "model://f450_airsim" "$DST" && echo "OK: world uses model://f450_airsim -> $DST"
else
  echo "WARNING: world does not include model://iris_with_ardupilot - world NOT created"
  echo "Send me: grep -n \"iris\" $SRC"
  exit 1
fi
mkdir -p ~/sitl_f450
cat > ~/sitl_f450/f450_airsim_match.parm <<'F450EOF'
# ArduPilot SITL settings matched to ArduPilot's AirSim defaults (Tools/autotest/default_params/airsim-quadX.parm)
# MOT_THST_EXPO is NOT copied: Gazebo motors (thrust ~ speed^2) need 0.65; AirSim motors are linear (0.0)
# SIMULATION ONLY - never load on the real drone (use AutoTune there)
ATC_RAT_PIT_P    0.25
ATC_RAT_PIT_I    0.25
ATC_RAT_PIT_D    0.003
ATC_RAT_RLL_P    0.25
ATC_RAT_RLL_I    0.25
ATC_RAT_RLL_D    0.003
ATC_RAT_PIT_FLTD 50
ATC_RAT_PIT_FLTT 50
ATC_RAT_RLL_FLTD 50
ATC_RAT_RLL_FLTT 50
INS_GYRO_FILTER  50
INS_ACCEL_FILTER 50
SCHED_LOOP_RATE  300
F450EOF
echo "ArduPilot settings file: ~/sitl_f450/f450_airsim_match.parm"
echo "Setup complete. Start Gazebo with:"
echo "  gazebo --verbose $DST"
