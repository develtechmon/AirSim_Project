"""
MANUAL DISTURBANCE DEMONSTRATION WITH REAL-TIME DASHBOARD
==========================================================
Interactive demo with comprehensive real-time visualization!

Controls:
    W/A/S/D - Directional bird attacks
    F/G/T - Flip/Spin/Drop
    
    UP/DOWN ARROW - Altitude control
    +/- - Intensity control
    
    SPACE - Apply disturbance
    R - RESET
    Q - Quit

Features:
    - Real-time sensor data display
    - Recovery percentage tracking
    - Impact intensity visualization
    - PhD research metrics
"""

import airsim
import numpy as np
import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from drone_flip_recovery_env_injector_gated import DroneFlipRecoveryEnv
import time
import sys
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch, Rectangle

try:
    from pynput import keyboard
    KEYBOARD_AVAILABLE = True
except ImportError:
    print("Install: pip install pynput matplotlib")
    sys.exit(1)

from disturbance_injector import DisturbanceInjector, DisturbanceType


class ProductionDashboard:
    """Production-ready dashboard with clean layout"""
    
    def __init__(self):
        self.max_points = 100
        
        # Data buffers
        self.time_data = deque(maxlen=self.max_points)
        self.roll_data = deque(maxlen=self.max_points)
        self.pitch_data = deque(maxlen=self.max_points)
        self.yaw_data = deque(maxlen=self.max_points)
        self.ang_vel_data = deque(maxlen=self.max_points)
        self.altitude_data = deque(maxlen=self.max_points)
        
        # Current state
        self.current_roll = 0
        self.current_pitch = 0
        self.current_yaw = 0
        self.current_ang_vel = 0
        self.current_altitude = 0
        self.current_intensity = 1.0
        self.current_pos_x = 0
        self.current_pos_y = 0
        self.target_altitude = 30.0
        
        # Status
        self.is_recovering = False
        self.disturbance_active = False
        self.selected_disturbance = "bird_attack"
        
        # Statistics
        self.total_disturbances = 0
        self.successful_recoveries = 0
        self.recovery_percentage = 0
        self.last_recovery_time = 0
        self.avg_recovery_time = 0
        self.recovery_times = []
        
        self.start_time = time.time()
        
        # Create figure with CLEAN layout
        plt.style.use('dark_background')
        self.fig = plt.figure(figsize=(18, 10))
        plt.ion()
        
        self.fig.canvas.manager.set_window_title('PhD Research Dashboard - Impact-Resilient UAV')
        
        # Professional title
        self.fig.text(0.5, 0.975, 
                     'IMPACT-RESILIENT UAV EMBEDDED ELECTRONICS SYSTEM',
                     ha='center', va='top', fontsize=13, fontweight='bold', 
                     color='#00ffff', family='sans-serif')
        self.fig.text(0.5, 0.955, 
                     'Autonomous Mid-Air Recovery Through PPO and Adaptive PID Control',
                     ha='center', va='top', fontsize=11, fontweight='bold', 
                     color='#00ffff', family='sans-serif')
        
        # Grid: 3 rows x 4 columns
        gs = GridSpec(3, 4, figure=self.fig, 
                     height_ratios=[1, 1, 0.8],
                     hspace=0.40, wspace=0.30,
                     top=0.92, bottom=0.08, left=0.06, right=0.96)
        
        # Row 1
        self.ax_roll = self.fig.add_subplot(gs[0, 0])
        self.ax_pitch = self.fig.add_subplot(gs[0, 1])
        self.ax_yaw = self.fig.add_subplot(gs[0, 2])
        self.ax_recovery = self.fig.add_subplot(gs[0, 3])
        
        # Row 2
        self.ax_ang_vel = self.fig.add_subplot(gs[1, 0])
        self.ax_altitude = self.fig.add_subplot(gs[1, 1])
        self.ax_intensity = self.fig.add_subplot(gs[1, 2])
        self.ax_status = self.fig.add_subplot(gs[1, 3])
        
        # Row 3
        self.ax_info = self.fig.add_subplot(gs[2, :])
        
        self._setup_plots()
        
        plt.show(block=False)
        plt.pause(0.001)
    
    def _setup_plots(self):
        """Setup all plots with professional styling"""
        
        # Roll Angle
        self.ax_roll.set_title('Roll Angle (Lateral Tilt)', fontsize=11, fontweight='bold', pad=10)
        self.ax_roll.set_ylim(-180, 180)
        self.ax_roll.axhline(0, color='white', linestyle='--', linewidth=0.8, alpha=0.3)
        self.ax_roll.grid(True, alpha=0.2, color='gray')
        self.line_roll, = self.ax_roll.plot([], [], color='#00d4ff', linewidth=2.5)
        self.ax_roll.set_ylabel('Degrees (°)', fontsize=9)
        self.ax_roll.set_xlabel('Time (s)', fontsize=9)
        self.ax_roll.tick_params(labelsize=8)
        
        # Pitch Angle
        self.ax_pitch.set_title('Pitch Angle (Forward/Backward Tilt)', fontsize=11, fontweight='bold', pad=10)
        self.ax_pitch.set_ylim(-180, 180)
        self.ax_pitch.axhline(0, color='white', linestyle='--', linewidth=0.8, alpha=0.3)
        self.ax_pitch.grid(True, alpha=0.2, color='gray')
        self.line_pitch, = self.ax_pitch.plot([], [], color='#00ff00', linewidth=2.5)
        self.ax_pitch.set_ylabel('Degrees (°)', fontsize=9)
        self.ax_pitch.set_xlabel('Time (s)', fontsize=9)
        self.ax_pitch.tick_params(labelsize=8)
        
        # Yaw Angle
        self.ax_yaw.set_title('Yaw Angle (Rotation)', fontsize=11, fontweight='bold', pad=10)
        self.ax_yaw.set_ylim(-180, 180)
        self.ax_yaw.axhline(0, color='white', linestyle='--', linewidth=0.8, alpha=0.3)
        self.ax_yaw.grid(True, alpha=0.2, color='gray')
        self.line_yaw, = self.ax_yaw.plot([], [], color='#ffff00', linewidth=2.5)
        self.ax_yaw.set_ylabel('Degrees (°)', fontsize=9)
        self.ax_yaw.set_xlabel('Time (s)', fontsize=9)
        self.ax_yaw.tick_params(labelsize=8)
        
        # Recovery Success Rate
        self.ax_recovery.axis('off')
        
        # Angular Velocity
        self.ax_ang_vel.set_title('Angular Velocity (Rotation Speed)', fontsize=11, fontweight='bold', pad=10)
        self.ax_ang_vel.set_ylim(0, 8)
        self.ax_ang_vel.axhline(1.2, color='#ff3333', linestyle='--', linewidth=1.5, 
                                label='Recovery Threshold', alpha=0.8)
        self.ax_ang_vel.grid(True, alpha=0.2, color='gray')
        self.line_ang_vel, = self.ax_ang_vel.plot([], [], color='#ff9900', linewidth=2.5)
        self.ax_ang_vel.legend(loc='upper right', fontsize=8, framealpha=0.7)
        self.ax_ang_vel.set_ylabel('rad/s', fontsize=9)
        self.ax_ang_vel.set_xlabel('Time (s)', fontsize=9)
        self.ax_ang_vel.tick_params(labelsize=8)
        
        # Altitude
        self.ax_altitude.set_title('Altitude Above Ground', fontsize=11, fontweight='bold', pad=10)
        self.ax_altitude.set_ylim(0, 45)
        self.ax_altitude.axhline(30, color='#00ff00', linestyle='--', linewidth=1.5, 
                                label='Target (30m)', alpha=0.8)
        self.ax_altitude.axhline(5, color='#ff3333', linestyle='--', linewidth=1.5, 
                                label='Danger (5m)', alpha=0.8)
        self.ax_altitude.grid(True, alpha=0.2, color='gray')
        self.line_altitude, = self.ax_altitude.plot([], [], color='#ff00ff', linewidth=2.5)
        self.ax_altitude.legend(loc='upper right', fontsize=8, framealpha=0.7)
        self.ax_altitude.set_ylabel('Meters (m)', fontsize=9)
        self.ax_altitude.set_xlabel('Time (s)', fontsize=9)
        self.ax_altitude.tick_params(labelsize=8)
        
        # Impact Intensity
        self.ax_intensity.axis('off')
        
        # System Status
        self.ax_status.axis('off')
        
        # Information Panel
        self.ax_info.axis('off')
    
    def update(self):
        """Update dashboard in real-time"""
        
        if len(self.time_data) < 2:
            plt.pause(0.001)
            return
        
        times = list(self.time_data)
        
        if times[-1] > times[0]:
            # Update time series
            self.line_roll.set_data(times, list(self.roll_data))
            self.ax_roll.set_xlim(times[0], times[-1])
            
            self.line_pitch.set_data(times, list(self.pitch_data))
            self.ax_pitch.set_xlim(times[0], times[-1])
            
            self.line_yaw.set_data(times, list(self.yaw_data))
            self.ax_yaw.set_xlim(times[0], times[-1])
            
            self.line_ang_vel.set_data(times, list(self.ang_vel_data))
            self.ax_ang_vel.set_xlim(times[0], times[-1])
            
            self.line_altitude.set_data(times, list(self.altitude_data))
            self.ax_altitude.set_xlim(times[0], times[-1])
        
        # Update Recovery Success Rate
        self.ax_recovery.clear()
        self.ax_recovery.axis('off')
        
        if self.recovery_percentage >= 75:
            color = '#00ff00'
            grade = 'EXCELLENT'
        elif self.recovery_percentage >= 60:
            color = '#ffff00'
            grade = 'GOOD'
        elif self.recovery_percentage >= 40:
            color = '#ff9900'
            grade = 'FAIR'
        else:
            color = '#ff3333'
            grade = 'NEEDS WORK'
        
        self.ax_recovery.text(0.5, 0.95, 'RECOVERY SUCCESS RATE', 
                             fontsize=11, fontweight='bold', color='white',
                             ha='center', va='top', transform=self.ax_recovery.transAxes)
        
        self.ax_recovery.text(0.5, 0.55, f'{self.recovery_percentage:.0f}%', 
                             fontsize=70, fontweight='bold', color=color,
                             ha='center', va='center', transform=self.ax_recovery.transAxes)
        
        self.ax_recovery.text(0.5, 0.25, grade, 
                             fontsize=14, fontweight='bold', color=color,
                             ha='center', va='center', transform=self.ax_recovery.transAxes)
        
        self.ax_recovery.text(0.5, 0.08, f'{self.successful_recoveries}/{self.total_disturbances} Recovered', 
                             fontsize=10, color='white',
                             ha='center', va='center', transform=self.ax_recovery.transAxes)
        
        # Update Impact Intensity
        self.ax_intensity.clear()
        self.ax_intensity.set_xlim(0, 1)
        self.ax_intensity.set_ylim(0, 1)
        self.ax_intensity.axis('off')
        
        self.ax_intensity.text(0.5, 0.95, 'IMPACT INTENSITY', 
                              fontsize=11, fontweight='bold', color='#ffff00',
                              ha='center', va='top', transform=self.ax_intensity.transAxes)
        
        if self.current_intensity < 0.8:
            bar_color = '#00ff00'
            intensity_label = 'LOW'
        elif self.current_intensity < 1.2:
            bar_color = '#ffff00'
            intensity_label = 'MEDIUM'
        elif self.current_intensity < 1.6:
            bar_color = '#ff9900'
            intensity_label = 'HIGH'
        else:
            bar_color = '#ff3333'
            intensity_label = 'EXTREME'
        
        # Draw intensity bar
        bar_height = self.current_intensity / 2.0
        bar = FancyBboxPatch((0.25, 0.15), 0.5, bar_height * 0.65, 
                            boxstyle="round,pad=0.02", 
                            facecolor=bar_color, edgecolor='white', 
                            linewidth=3, alpha=0.9,
                            transform=self.ax_intensity.transAxes)
        self.ax_intensity.add_patch(bar)
        
        # Scale markers - FIXED: Use plot() instead of axhline()
        for val in [0.5, 1.0, 1.5, 2.0]:
            y_pos = 0.15 + (val / 2.0) * 0.65
            # Draw horizontal line using plot
            self.ax_intensity.plot([0.2, 0.8], [y_pos, y_pos], 
                                  color='white', linestyle='--', linewidth=0.8, alpha=0.3,
                                  transform=self.ax_intensity.transAxes)
            self.ax_intensity.text(0.82, y_pos, f'{val:.1f}x', 
                                  fontsize=9, color='white',
                                  va='center', transform=self.ax_intensity.transAxes)
        
        self.ax_intensity.text(0.5, 0.15 + bar_height * 0.65 + 0.08, 
                              f'{self.current_intensity:.2f}x\n{intensity_label}',
                              fontsize=13, fontweight='bold', color=bar_color,
                              ha='center', va='bottom', transform=self.ax_intensity.transAxes)
        
        # Update System Status
        self.ax_status.clear()
        self.ax_status.axis('off')
        
        self.ax_status.text(0.5, 0.95, 'SYSTEM STATUS', 
                           fontsize=11, fontweight='bold', color='white',
                           ha='center', va='top', transform=self.ax_status.transAxes)
        
        if self.disturbance_active:
            status_color = '#ff3333'
            bg_color = '#330000'
            status_lines = [
                'DISTURBANCE ACTIVE',
                '',
                self.selected_disturbance.upper(),
                '',
                f'{self.current_intensity:.2f}x Intensity',
                '',
                'PPO Policy Active'
            ]
        elif self.is_recovering:
            status_color = '#ff9900'
            bg_color = '#332200'
            status_lines = [
                'RECOVERING',
                '',
                'Stabilizing Control',
                '',
                f'{self.current_ang_vel:.2f} rad/s',
                '',
                'Adaptive PID Active'
            ]
        else:
            status_color = '#00ff00'
            bg_color = '#003300'
            status_lines = [
                'STABLE',
                '',
                'Hover Mode',
                '',
                'Ready for Impact',
                '',
                'System Armed'
            ]
        
        self.ax_status.set_facecolor(bg_color)
        
        status_text = '\n'.join(status_lines)
        self.ax_status.text(0.5, 0.5, status_text,
                           fontsize=11, fontweight='bold', color=status_color,
                           ha='center', va='center', transform=self.ax_status.transAxes)
        
        # Update Information Panel
        self.ax_info.clear()
        self.ax_info.axis('off')
        
        elapsed_time = time.time() - self.start_time
        
        info_text = f"""╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
║  PhD RESEARCH: IMPACT-RESILIENT UAV EMBEDDED ELECTRONICS SYSTEM                                                                      ║
║  Novel Contribution: Variable-Intensity Impact Recovery Using Proximal Policy Optimization and Adaptive PID Control                 ║
╠══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣
║  CURRENT SENSOR READINGS              │  RECOVERY STATISTICS                │  SYSTEM METRICS                                       ║
║  ────────────────────────────────────  │  ──────────────────────────────────  │  ─────────────────────────────────────────────────  ║
║  Position:                             │  Total Disturbances:  {self.total_disturbances:3d}            │  Session Time:        {elapsed_time/60:6.1f} min             ║
║    X: {self.current_pos_x:7.2f} m                      │  Successful:          {self.successful_recoveries:3d}            │  Avg Recovery:        {self.avg_recovery_time:5.2f} s              ║
║    Y: {self.current_pos_y:7.2f} m                      │  Success Rate:     {self.recovery_percentage:5.1f}%           │  Last Recovery:       {self.last_recovery_time:5.2f} s              ║
║    Z: {self.current_altitude:7.2f} m (Target: {self.target_altitude:.1f}m)   │                                     │  Impact Range:      0.5-2.0x              ║
║                                        │  KEY CAPABILITIES:                  │  Control Method:    PPO + PID             ║
║  Orientation:                          │    • Real-time policy adaptation    │  Operating Mode:    Manual Demo           ║
║    Roll:  {self.current_roll:7.1f}°                    │    • Adaptive PID stabilization     │                                       ║
║    Pitch: {self.current_pitch:7.1f}°                    │    • Variable intensity recovery    │  CONTROLS:                            ║
║    Yaw:   {self.current_yaw:7.1f}°                    │    • Autonomous system              │    W/A/S/D: Attack direction          ║
║                                        │                                     │    F/G/T: Flip/Spin/Drop              ║
║  Dynamics:                             │                                     │    +/-: Adjust intensity              ║
║    Angular Velocity: {self.current_ang_vel:5.2f} rad/s      │                                     │    SPACE: Apply | R: Reset            ║
╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝"""
        
        self.ax_info.text(0.01, 0.98, info_text,
                         fontsize=9, family='monospace', color='white',
                         va='top', ha='left', transform=self.ax_info.transAxes)
        
        # Force redraw
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        plt.pause(0.001)
    
    def update_data(self, obs, disturbance_info, target_alt):
        """Update with new data"""
        
        pos = obs[0:3]
        ori = obs[6:10]
        ang_vel = obs[10:13]
        
        qw, qx, qy, qz = ori
        
        sinr_cosp = 2 * (qw * qx + qy * qz)
        cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
        roll = np.degrees(np.arctan2(sinr_cosp, cosr_cosp))
        
        sinp = 2 * (qw * qy - qz * qx)
        pitch = np.degrees(np.arcsin(np.clip(sinp, -1, 1)))
        
        siny_cosp = 2 * (qw * qz + qx * qy)
        cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
        yaw = np.degrees(np.arctan2(siny_cosp, cosy_cosp))
        
        ang_vel_mag = np.linalg.norm(ang_vel)
        altitude = -pos[2]
        
        current_time = time.time() - self.start_time
        
        self.time_data.append(current_time)
        self.roll_data.append(roll)
        self.pitch_data.append(pitch)
        self.yaw_data.append(yaw)
        self.ang_vel_data.append(ang_vel_mag)
        self.altitude_data.append(altitude)
        
        self.current_roll = roll
        self.current_pitch = pitch
        self.current_yaw = yaw
        self.current_ang_vel = ang_vel_mag
        self.current_altitude = altitude
        self.current_pos_x = pos[0]
        self.current_pos_y = pos[1]
        self.target_altitude = target_alt
        
        if disturbance_info:
            self.current_intensity = disturbance_info.get('intensity', 1.0)
            self.is_recovering = disturbance_info.get('is_recovering', False)
            self.disturbance_active = disturbance_info.get('disturbance_active', False)
            self.selected_disturbance = disturbance_info.get('selected_disturbance', 'unknown')
    
    def record_disturbance(self):
        self.total_disturbances += 1
        self._update_recovery_percentage()
    
    def record_recovery(self, recovery_time_steps):
        self.successful_recoveries += 1
        recovery_time_sec = recovery_time_steps * 0.05
        self.last_recovery_time = recovery_time_sec
        self.recovery_times.append(recovery_time_sec)
        self.avg_recovery_time = np.mean(self.recovery_times)
        self._update_recovery_percentage()
    
    def _update_recovery_percentage(self):
        self.recovery_percentage = (self.successful_recoveries / self.total_disturbances * 100) if self.total_disturbances > 0 else 0


class ManualControlProduction:
    """Manual control with production dashboard"""
    
    def __init__(self, model_path, vecnorm_path):
        print("\n" + "="*70)
        print("PRODUCTION DASHBOARD - PhD Research Demo")
        print("="*70)
        
        def make_env():
            return DroneFlipRecoveryEnv(
                target_altitude=30.0, max_steps=1000,
                wind_strength=3.0, flip_prob=0.0, debug=False
            )
        
        self.env = DummyVecEnv([make_env])
        self.env = VecNormalize.load(vecnorm_path, self.env)
        self.env.training = False
        self.env.norm_reward = False
        
        self.model = PPO.load(model_path, device='cpu')
        
        base_env = self.env.venv.envs[0]
        self.actual_env = base_env.env if hasattr(base_env, 'env') else base_env
        
        self.client = self.actual_env.client
        self.injector = self.actual_env.disturbance_injector
        
        self.intensity = 1.0
        self.selected_disturbance = DisturbanceType.BIRD_ATTACK
        self.selected_direction = 'front'
        self.running = True
        self.obs = None
        self.episode_step = 0
        self.currently_recovering = False
        
        self.target_altitude = 30.0
        self.altitude_step = 2.0
        self.min_altitude = 5.0
        self.max_altitude = 50.0
        
        self.pending_disturbance = False
        self.pending_reset = False
        
        self.dashboard = ProductionDashboard()
        
        print("System ready!")
        print("="*70 + "\n")
        
        self.listener = keyboard.Listener(on_press=self._on_key_press)
        self.listener.start()
    
    def _on_key_press(self, key):
        try:
            if hasattr(key, 'char') and key.char:
                char = key.char.lower()
                
                if char == 'w':
                    self.selected_disturbance = DisturbanceType.BIRD_ATTACK
                    self.selected_direction = 'front'
                elif char == 'a':
                    self.selected_disturbance = DisturbanceType.BIRD_ATTACK
                    self.selected_direction = 'left'
                elif char == 's':
                    self.selected_disturbance = DisturbanceType.BIRD_ATTACK
                    self.selected_direction = 'back'
                elif char == 'd':
                    self.selected_disturbance = DisturbanceType.BIRD_ATTACK
                    self.selected_direction = 'right'
                elif char == 'f':
                    self.selected_disturbance = DisturbanceType.FLIP
                elif char == 'g':
                    self.selected_disturbance = DisturbanceType.SPIN
                elif char == 't':
                    self.selected_disturbance = DisturbanceType.DROP
                elif char == 'r':
                    self.pending_reset = True
                elif char == 'q':
                    self.running = False
                elif char in ['+', '=']:
                    self.intensity = min(2.0, self.intensity + 0.1)
                elif char == '-':
                    self.intensity = max(0.5, self.intensity - 0.1)
            
            if key == keyboard.Key.space:
                self.pending_disturbance = True
            elif key == keyboard.Key.up:
                self.target_altitude = np.clip(self.target_altitude + self.altitude_step, 
                                               self.min_altitude, self.max_altitude)
            elif key == keyboard.Key.down:
                self.target_altitude = np.clip(self.target_altitude - self.altitude_step, 
                                               self.min_altitude, self.max_altitude)
        except AttributeError:
            pass
    
    def _reset_drone(self):
        self.client.reset()
        time.sleep(0.5)
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        time.sleep(0.5)
        self.client.takeoffAsync().join()
        time.sleep(1.0)
        self.client.moveToPositionAsync(0, 0, -self.target_altitude, 5).join()
        time.sleep(1.0)
        self.obs = self.env.reset()
        self.episode_step = 0
        self.currently_recovering = False
    
    def _apply_disturbance(self):
        if self.selected_disturbance == DisturbanceType.BIRD_ATTACK:
            force = np.random.uniform(40, 80) * self.intensity
            directions = {'front': (force, 0, 0), 'back': (-force, 0, 0),
                         'left': (0, -force, 0), 'right': (0, force, 0)}
            fx, fy, fz = directions[self.selected_direction]
            
            self.client.moveByVelocityAsync(
                fx * 0.1, fy * 0.1, fz * 0.1, duration=0.15,
                drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                yaw_mode=airsim.YawMode(False, 0)
            ).join()
            
            angular_vel = np.random.uniform(360, 540) * self.intensity
            axis = np.random.choice(['roll', 'pitch'])
            
            self.client.moveByAngleRatesThrottleAsync(
                roll_rate=np.radians(angular_vel) if axis == 'roll' else 0,
                pitch_rate=np.radians(angular_vel) if axis == 'pitch' else 0,
                yaw_rate=0, throttle=0.59375, duration=0.8
            ).join()
        else:
            self.injector.inject_disturbance(self.selected_disturbance, intensity=self.intensity)
        
        self.dashboard.record_disturbance()
        self.currently_recovering = True
        self.actual_env.disturbance_initiated = True
        self.actual_env.disturbance_recovered = False
        self.actual_env.disturbance_start_step = self.episode_step
    
    def run(self):
        self._reset_drone()
        
        while self.running:
            try:
                if self.pending_reset:
                    self._reset_drone()
                    self.pending_reset = False
                
                if self.pending_disturbance:
                    self._apply_disturbance()
                    self.pending_disturbance = False
                
                action, _ = self.model.predict(self.obs, deterministic=True)
                self.obs, reward, done, info = self.env.step(action)
                self.episode_step += 1
                
                if self.currently_recovering and self.actual_env.disturbance_recovered:
                    self.dashboard.record_recovery(self.actual_env.recovery_steps)
                    self.currently_recovering = False
                
                disturbance_info = {
                    'intensity': self.intensity,
                    'is_recovering': self.currently_recovering,
                    'disturbance_active': self.pending_disturbance or self.currently_recovering,
                    'selected_disturbance': self.selected_disturbance.value
                }
                
                self.dashboard.update_data(self.obs[0], disturbance_info, self.target_altitude)
                self.dashboard.update()
                
                if done:
                    reason = info[0].get('reason', 'unknown') if isinstance(info, list) else info.get('reason', 'unknown')
                    if reason != 'timeout':
                        time.sleep(2)
                        self._reset_drone()
                
                time.sleep(0.05)
                
            except KeyboardInterrupt:
                self.running = False
        
        self.listener.stop()
        self.env.close()
        input("Press Enter to close...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str,
                        default='./models/stage3_checkpoints/gated_curriculum_policy.zip')
    parser.add_argument('--vecnorm', type=str,
                        default='./models/stage3_checkpoints/gated_curriculum_vecnormalize.pkl')
    args = parser.parse_args()
    
    if not KEYBOARD_AVAILABLE:
        print("pip install pynput matplotlib")
        sys.exit(1)
    
    demo = ManualControlProduction(args.model, args.vecnorm)
    demo.run()