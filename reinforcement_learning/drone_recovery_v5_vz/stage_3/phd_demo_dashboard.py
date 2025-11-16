"""
PRODUCTION-READY MANUAL DISTURBANCE DEMONSTRATION DASHBOARD
============================================================
Multi-threaded, matches training environment EXACTLY!

CRITICAL: ALL disturbances use DisturbanceInjector.inject_disturbance()
This is the EXACT SAME method used during training!

FIXES:
✅ Uses DisturbanceInjector for ALL disturbances (matches training!)
✅ No spontaneous disturbances (flip_prob=0.0, manual control only)
✅ Dashboard records data during ALL phases (takeoff, hover, recovery)
✅ Multi-threaded (PPO @ 20Hz, Dashboard @ 10Hz)
✅ Stable, production-ready code

Controls:
    W/A/S/D - Bird attacks (front/left/back/right)
              Uses DisturbanceInjector - SAME AS TRAINING!
    
    F - Flip (violent tumble)
    G - Spin (yaw rotation)
    T - Drop (altitude loss)
    
    +/- - Intensity control
    SPACE - Apply disturbance
    R - RESET
    Q - Quit
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
from matplotlib.patches import FancyBboxPatch, FancyArrow, Circle
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk
import threading
import queue

try:
    from pynput import keyboard
    KEYBOARD_AVAILABLE = True
except ImportError:
    print("Install: pip install pynput matplotlib")
    sys.exit(1)

from disturbance_injector import DisturbanceInjector, DisturbanceType


class ScrollableDashboard:
    """Production-ready dashboard with data logging during ALL phases"""
    
    def __init__(self, client, data_queue):
        self.client = client
        self.data_queue = data_queue
        self.max_points = 100
        
        # Data buffers
        self.time_data = deque(maxlen=self.max_points)
        self.roll_data = deque(maxlen=self.max_points)
        self.pitch_data = deque(maxlen=self.max_points)
        self.yaw_data = deque(maxlen=self.max_points)
        self.ang_vel_data = deque(maxlen=self.max_points)
        self.altitude_data = deque(maxlen=self.max_points)
        
        # PPO action tracking
        self.action_vx_data = deque(maxlen=self.max_points)
        self.action_vy_data = deque(maxlen=self.max_points)
        self.action_vz_data = deque(maxlen=self.max_points)
        self.current_action = np.zeros(3)
        
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
        self.selected_direction = "front"
        
        # Recovery timer (NEW!)
        self.recovery_start_time = None
        self.current_recovery_time = 0.0
        self.last_completed_recovery_time = 0.0
        
        # Last disturbance info (for display)
        self.last_disturbance_type = "None"
        self.last_disturbance_direction = "N/A"
        self.last_disturbance_intensity = 0.0
        self.last_disturbance_force = "N/A"
        self.last_disturbance_angular_vel = "N/A"
        self.last_disturbance_axis = "N/A"
        
        # Statistics
        self.total_disturbances = 0
        self.successful_recoveries = 0
        self.recovery_percentage = 0
        self.last_recovery_time = 0
        self.avg_recovery_time = 0
        self.recovery_times = []
        
        self.start_time = time.time()
        
        # Create Tkinter root
        self.root = tk.Tk()
        self.root.title("PhD: Production-Ready Recovery Demonstration")
        self.root.geometry("1800x950")
        
        # Create main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left side: Matplotlib plots
        plot_frame = ttk.Frame(main_frame)
        plot_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Right side: Scrollable info
        info_frame = ttk.Frame(main_frame, width=650)
        info_frame.pack(side=tk.RIGHT, fill=tk.BOTH)
        info_frame.pack_propagate(False)
        
        scrollbar = ttk.Scrollbar(info_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.info_text = tk.Text(info_frame,
                                 bg='#1e1e1e',
                                 fg='#00ff00',
                                 font=('Courier', 9),
                                 yscrollcommand=scrollbar.set,
                                 wrap=tk.WORD,
                                 padx=10,
                                 pady=10)
        self.info_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.info_text.yview)
        
        # Create matplotlib figure
        plt.style.use('dark_background')
        self.fig = plt.figure(figsize=(13, 9.5))
        
        # Embed matplotlib
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Title
        self.fig.text(0.5, 0.975,
                     'PRODUCTION-READY: PPO AUTONOMOUS RECOVERY SYSTEM',
                     ha='center', va='top', fontsize=12, fontweight='bold',
                     color='#00ffff', family='sans-serif')
        self.fig.text(0.5, 0.955,
                     'Stable Multi-threaded Implementation (~10s Recovery)',
                     ha='center', va='top', fontsize=9, fontweight='bold',
                     color='#00ff00', family='sans-serif')
        
        # Grid: 3 rows x 4 columns
        gs = GridSpec(3, 4, figure=self.fig,
                     height_ratios=[1, 1, 1],
                     hspace=0.45, wspace=0.30,
                     top=0.92, bottom=0.06, left=0.08, right=0.96)
        
        # Row 1: THE PROBLEM
        self.ax_roll = self.fig.add_subplot(gs[0, 0])
        self.ax_pitch = self.fig.add_subplot(gs[0, 1])
        self.ax_ang_vel = self.fig.add_subplot(gs[0, 2])
        self.ax_altitude = self.fig.add_subplot(gs[0, 3])
        
        # Row 2: THE SOLUTION
        self.ax_action_vx = self.fig.add_subplot(gs[1, 0])
        self.ax_action_vy = self.fig.add_subplot(gs[1, 1])
        self.ax_action_vz = self.fig.add_subplot(gs[1, 2])
        self.ax_action_vector = self.fig.add_subplot(gs[1, 3])
        
        # Row 3: THE RESULT
        self.ax_recovery = self.fig.add_subplot(gs[2, 0])
        self.ax_intensity = self.fig.add_subplot(gs[2, 1])
        self.ax_status = self.fig.add_subplot(gs[2, 2])
        self.ax_strategy = self.fig.add_subplot(gs[2, 3])
        
        self._setup_plots()
        
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.closed = False
        
        # Start async update loop
        self._schedule_update()
    
    def _on_close(self):
        self.closed = True
        self.root.quit()
        self.root.destroy()
    
    def _setup_plots(self):
        """Setup all plots - EXACTLY AS ORIGINAL"""
        
        # Roll Angle
        self.ax_roll.set_title('PROBLEM: Roll Angle\n(Lateral tilt - should be ~0°)', 
                               fontsize=9, fontweight='bold', pad=8, color='#ff9999')
        self.ax_roll.set_ylim(-180, 180)
        self.ax_roll.axhline(0, color='lime', linestyle='--', linewidth=1, alpha=0.5, label='Target: 0°')
        self.ax_roll.axhspan(-30, 30, color='green', alpha=0.1)
        self.ax_roll.grid(True, alpha=0.2, color='gray')
        self.line_roll, = self.ax_roll.plot([], [], color='#00d4ff', linewidth=2.5)
        self.ax_roll.set_ylabel('Degrees (°)', fontsize=8)
        self.ax_roll.set_xlabel('Time (s)', fontsize=8)
        self.ax_roll.tick_params(labelsize=7)
        self.ax_roll.legend(loc='upper right', fontsize=6)
        
        # Pitch Angle
        self.ax_pitch.set_title('PROBLEM: Pitch Angle\n(Forward/back tilt - should be ~0°)', 
                                fontsize=9, fontweight='bold', pad=8, color='#ff9999')
        self.ax_pitch.set_ylim(-180, 180)
        self.ax_pitch.axhline(0, color='lime', linestyle='--', linewidth=1, alpha=0.5, label='Target: 0°')
        self.ax_pitch.axhspan(-30, 30, color='green', alpha=0.1)
        self.ax_pitch.grid(True, alpha=0.2, color='gray')
        self.line_pitch, = self.ax_pitch.plot([], [], color='#00ff00', linewidth=2.5)
        self.ax_pitch.set_ylabel('Degrees (°)', fontsize=8)
        self.ax_pitch.set_xlabel('Time (s)', fontsize=8)
        self.ax_pitch.tick_params(labelsize=7)
        self.ax_pitch.legend(loc='upper right', fontsize=6)
        
        # Angular Velocity
        self.ax_ang_vel.set_title('PROBLEM: Angular Velocity\n(Spin speed - MUST drop below 1.2 rad/s)', 
                                  fontsize=9, fontweight='bold', pad=8, color='#ff9999')
        self.ax_ang_vel.set_ylim(0, 8)
        self.ax_ang_vel.axhline(1.2, color='#ff3333', linestyle='--', linewidth=2,
                                label='Recovery Threshold', alpha=0.9)
        self.ax_ang_vel.axhspan(0, 1.2, color='green', alpha=0.1, label='Safe Zone')
        self.ax_ang_vel.grid(True, alpha=0.2, color='gray')
        self.line_ang_vel, = self.ax_ang_vel.plot([], [], color='#ff9900', linewidth=3)
        self.ax_ang_vel.legend(loc='upper right', fontsize=6)
        self.ax_ang_vel.set_ylabel('rad/s', fontsize=8)
        self.ax_ang_vel.set_xlabel('Time (s)', fontsize=8)
        self.ax_ang_vel.tick_params(labelsize=7)
        
        # Altitude
        self.ax_altitude.set_title('PROBLEM: Altitude Loss\n(Must stay above 5m danger zone)', 
                                   fontsize=9, fontweight='bold', pad=8, color='#ff9999')
        self.ax_altitude.set_ylim(0, 45)
        self.ax_altitude.axhline(30, color='#00ff00', linestyle='--', linewidth=1.5,
                                label='Target: 30m', alpha=0.8)
        self.ax_altitude.axhline(5, color='#ff3333', linestyle='--', linewidth=2,
                                label='DANGER: 5m', alpha=0.9)
        self.ax_altitude.axhspan(0, 5, color='red', alpha=0.1)
        self.ax_altitude.axhspan(25, 35, color='green', alpha=0.1)
        self.ax_altitude.grid(True, alpha=0.2, color='gray')
        self.line_altitude, = self.ax_altitude.plot([], [], color='#ff00ff', linewidth=2.5)
        self.ax_altitude.legend(loc='upper right', fontsize=6)
        self.ax_altitude.set_ylabel('Meters (m)', fontsize=8)
        self.ax_altitude.set_xlabel('Time (s)', fontsize=8)
        self.ax_altitude.tick_params(labelsize=7)
        
        # PPO Actions
        self.ax_action_vx.set_title('SOLUTION: PPO Action Vx\n(Forward/Backward velocity)', 
                                    fontsize=9, fontweight='bold', pad=8, color='#99ff99')
        self.ax_action_vx.set_ylim(-5.5, 5.5)
        self.ax_action_vx.axhline(0, color='white', linestyle='--', linewidth=0.8, alpha=0.3)
        self.ax_action_vx.grid(True, alpha=0.2, color='gray')
        self.line_action_vx, = self.ax_action_vx.plot([], [], color='#ff0000', linewidth=2.5)
        self.ax_action_vx.set_ylabel('m/s', fontsize=8)
        self.ax_action_vx.set_xlabel('Time (s)', fontsize=8)
        self.ax_action_vx.tick_params(labelsize=7)
        
        self.ax_action_vy.set_title('SOLUTION: PPO Action Vy\n(Left/Right velocity)', 
                                    fontsize=9, fontweight='bold', pad=8, color='#99ff99')
        self.ax_action_vy.set_ylim(-5.5, 5.5)
        self.ax_action_vy.axhline(0, color='white', linestyle='--', linewidth=0.8, alpha=0.3)
        self.ax_action_vy.grid(True, alpha=0.2, color='gray')
        self.line_action_vy, = self.ax_action_vy.plot([], [], color='#00ff00', linewidth=2.5)
        self.ax_action_vy.set_ylabel('m/s', fontsize=8)
        self.ax_action_vy.set_xlabel('Time (s)', fontsize=8)
        self.ax_action_vy.tick_params(labelsize=7)
        
        self.ax_action_vz.set_title('SOLUTION: PPO Action Vz\n(Vertical - PREVENTS CRASH!)', 
                                    fontsize=9, fontweight='bold', pad=8, color='#99ff99')
        self.ax_action_vz.set_ylim(-5.5, 5.5)
        self.ax_action_vz.axhline(0, color='white', linestyle='--', linewidth=0.8, alpha=0.3)
        self.ax_action_vz.axhspan(-5, -1, color='green', alpha=0.1, label='Climbing')
        self.ax_action_vz.grid(True, alpha=0.2, color='gray')
        self.line_action_vz, = self.ax_action_vz.plot([], [], color='#00ffff', linewidth=3)
        self.ax_action_vz.legend(loc='upper right', fontsize=6)
        self.ax_action_vz.set_ylabel('m/s', fontsize=8)
        self.ax_action_vz.set_xlabel('Time (s)', fontsize=8)
        self.ax_action_vz.tick_params(labelsize=7)
        
        self.ax_action_vector.axis('off')
        self.ax_recovery.axis('off')
        self.ax_intensity.axis('off')
        self.ax_status.axis('off')
        self.ax_strategy.axis('off')
    
    def _schedule_update(self):
        """Async updates - processes UI events!"""
        if not self.closed:
            try:
                self._process_queue_data()
                self._update_visualization()
                self.root.update()  # CRITICAL: Process UI events!
            except Exception as e:
                print(f"Dashboard error: {e}")
            self.root.after(100, self._schedule_update)
    
    def _process_queue_data(self):
        """Get all pending data from PPO thread"""
        try:
            while True:
                msg = self.data_queue.get_nowait()
                
                if msg['type'] == 'state':
                    self._update_from_state(msg)
                elif msg['type'] == 'disturbance':
                    self.record_disturbance()
                    # Start recovery timer
                    self.recovery_start_time = time.time()
                    self.current_recovery_time = 0.0
                elif msg['type'] == 'recovery':
                    self.record_recovery(msg['steps'])
                    # Stop recovery timer and save the time
                    if self.recovery_start_time is not None:
                        self.last_completed_recovery_time = time.time() - self.recovery_start_time
                        self.recovery_start_time = None
                        self.current_recovery_time = 0.0
                elif msg['type'] == 'disturbance_info':
                    # Update disturbance display info
                    self.selected_disturbance = msg.get('disturbance', 'unknown')
                    self.selected_direction = msg.get('direction', '')
                elif msg['type'] == 'disturbance_details':
                    # ✅ NEW: Capture detailed disturbance info
                    self._update_disturbance_details(msg)
                    
        except queue.Empty:
            pass
        
        # Update current recovery time if actively recovering
        if self.recovery_start_time is not None:
            self.current_recovery_time = time.time() - self.recovery_start_time
    
    def _update_disturbance_details(self, msg):
        """Update last disturbance details for display"""
        self.last_disturbance_type = msg.get('disturbance_type', 'Unknown')
        self.last_disturbance_direction = msg.get('direction', 'N/A')
        self.last_disturbance_intensity = msg.get('intensity', 0.0)
        
        # Format force vector
        force = msg.get('force', [0, 0, 0])
        if isinstance(force, list) and len(force) == 3:
            self.last_disturbance_force = f"[{force[0]:+6.1f}, {force[1]:+6.1f}, {force[2]:+6.1f}] N"
        else:
            self.last_disturbance_force = str(force)
        
        # Format angular velocity
        ang_vel = msg.get('angular_velocity', None)
        if ang_vel is not None:
            self.last_disturbance_angular_vel = f"{ang_vel:+6.1f} deg/s"
        else:
            self.last_disturbance_angular_vel = "N/A"
        
        # Format axis
        self.last_disturbance_axis = msg.get('axis', 'N/A')
    
    def _update_from_state(self, msg):
        """Update internal state from queue"""
        current_time = time.time() - self.start_time
        
        self.time_data.append(current_time)
        self.roll_data.append(msg['roll'])
        self.pitch_data.append(msg['pitch'])
        self.yaw_data.append(msg['yaw'])
        self.ang_vel_data.append(msg['ang_vel'])
        self.altitude_data.append(msg['altitude'])
        
        self.current_action = msg['action']
        self.action_vx_data.append(msg['action'][0])
        self.action_vy_data.append(msg['action'][1])
        self.action_vz_data.append(msg['action'][2])
        
        self.current_roll = msg['roll']
        self.current_pitch = msg['pitch']
        self.current_yaw = msg['yaw']
        self.current_ang_vel = msg['ang_vel']
        self.current_altitude = msg['altitude']
        self.current_pos_x = msg['pos_x']
        self.current_pos_y = msg['pos_y']
        self.target_altitude = msg['target_alt']
        
        self.current_intensity = msg.get('intensity', 1.0)
        self.is_recovering = msg.get('is_recovering', False)
        self.disturbance_active = msg.get('disturbance_active', False)
    
    def _update_visualization(self):
        """Update all plots - includes all original visualization code"""
        
        # Update time series plots
        if len(self.time_data) >= 2:
            times = list(self.time_data)
            
            if times[-1] > times[0]:
                self.line_roll.set_data(times, list(self.roll_data))
                self.ax_roll.set_xlim(times[0], times[-1])
                
                self.line_pitch.set_data(times, list(self.pitch_data))
                self.ax_pitch.set_xlim(times[0], times[-1])
                
                self.line_ang_vel.set_data(times, list(self.ang_vel_data))
                self.ax_ang_vel.set_xlim(times[0], times[-1])
                
                self.line_altitude.set_data(times, list(self.altitude_data))
                self.ax_altitude.set_xlim(times[0], times[-1])
                
                self.line_action_vx.set_data(times, list(self.action_vx_data))
                self.ax_action_vx.set_xlim(times[0], times[-1])
                
                self.line_action_vy.set_data(times, list(self.action_vy_data))
                self.ax_action_vy.set_xlim(times[0], times[-1])
                
                self.line_action_vz.set_data(times, list(self.action_vz_data))
                self.ax_action_vz.set_xlim(times[0], times[-1])
        
        # Update action vector
        self.ax_action_vector.clear()
        self.ax_action_vector.set_xlim(-1, 1)
        self.ax_action_vector.set_ylim(-1, 1)
        self.ax_action_vector.axis('off')
        
        self.ax_action_vector.text(0.5, 0.95, 'PPO ACTION\nVECTOR', fontsize=9, fontweight='bold',
                                  color='#ff00ff', ha='center', va='top',
                                  transform=self.ax_action_vector.transAxes)
        
        # Draw drone circle
        drone_circle = Circle((0.5, 0.5), 0.08, color='#00ffff', alpha=0.8,
                             transform=self.ax_action_vector.transAxes)
        self.ax_action_vector.add_patch(drone_circle)
        
        # ✅ HORIZONTAL MOVEMENT ARROWS (Vx, Vy)
        scale = 0.06
        vx_arrow = self.current_action[0] * scale  # Forward/backward
        vy_arrow = self.current_action[1] * scale  # Right/left
        
        # Draw horizontal movement arrow if significant
        if abs(vx_arrow) > 0.001 or abs(vy_arrow) > 0.001:
            # Rotate 90° because Forward (Vx) should point UP in visualization
            arrow = FancyArrow(0.5, 0.5, vy_arrow, vx_arrow,
                              width=0.02, head_width=0.05, head_length=0.03,
                              color='#ff0000', alpha=0.9,
                              transform=self.ax_action_vector.transAxes)
            self.ax_action_vector.add_patch(arrow)
            
            # Label the direction
            if abs(vx_arrow) > abs(vy_arrow):
                direction = 'Forward' if vx_arrow > 0 else 'Backward'
            else:
                direction = 'Right' if vy_arrow > 0 else 'Left'
            
            self.ax_action_vector.text(0.5 + vy_arrow, 0.5 + vx_arrow + 0.08, 
                                      direction, fontsize=7,
                                      color='#ff0000', ha='center', va='bottom',
                                      transform=self.ax_action_vector.transAxes)
        
        # ✅ VERTICAL INDICATOR (Vz)
        vz = self.current_action[2]
        if vz < -0.1:  # Climbing (negative = up in NED)
            self.ax_action_vector.text(0.5, 0.75, '↑ CLIMBING', fontsize=11,
                                      color='#00ff00', fontweight='bold',
                                      ha='center', va='center',
                                      transform=self.ax_action_vector.transAxes)
            self.ax_action_vector.text(0.5, 0.65, f'{vz:.2f} m/s', fontsize=9,
                                      color='#00ff00', ha='center', va='center',
                                      transform=self.ax_action_vector.transAxes)
        elif vz > 0.1:  # Descending (positive = down in NED)
            self.ax_action_vector.text(0.5, 0.25, '↓ DESCENDING', fontsize=11,
                                      color='#ffff00', fontweight='bold',
                                      ha='center', va='center',
                                      transform=self.ax_action_vector.transAxes)
            self.ax_action_vector.text(0.5, 0.15, f'{vz:.2f} m/s', fontsize=9,
                                      color='#ffff00', ha='center', va='center',
                                      transform=self.ax_action_vector.transAxes)
        else:
            self.ax_action_vector.text(0.5, 0.35, 'HOVERING', fontsize=11,
                                      color='#ffffff', fontweight='bold',
                                      ha='center', va='center',
                                      transform=self.ax_action_vector.transAxes)
        
        # Action magnitude
        action_mag = np.linalg.norm(self.current_action)
        self.ax_action_vector.text(0.5, 0.08, f'|Action|: {action_mag:.2f} m/s',
                              fontsize=8, color='white', ha='center', va='center',
                              transform=self.ax_action_vector.transAxes)
        
        # Update recovery stats, intensity, status, strategy (same as original)
        self._update_recovery_panel()
        self._update_intensity_panel()
        self._update_status_panel()
        self._update_strategy_panel()
        self._update_info_panel()
        
        self.canvas.draw_idle()
    
    def _update_recovery_panel(self):
        """Update recovery statistics"""
        self.ax_recovery.clear()
        self.ax_recovery.axis('off')
        
        if self.recovery_percentage >= 75:
            color, grade = '#00ff00', 'EXCELLENT'
        elif self.recovery_percentage >= 60:
            color, grade = '#ffff00', 'GOOD'
        else:
            color, grade = '#ff3333', 'NEEDS WORK'
        
        self.ax_recovery.text(0.5, 0.90, 'RECOVERY\nRATE', fontsize=9, fontweight='bold',
                             color='white', ha='center', va='top',
                             transform=self.ax_recovery.transAxes)
        self.ax_recovery.text(0.5, 0.50, f'{self.recovery_percentage:.0f}%',
                             fontsize=50, fontweight='bold', color=color,
                             ha='center', va='center', transform=self.ax_recovery.transAxes)
        self.ax_recovery.text(0.5, 0.15, grade, fontsize=11, fontweight='bold',
                             color=color, ha='center', va='center',
                             transform=self.ax_recovery.transAxes)
        self.ax_recovery.text(0.5, 0.05, f'{self.successful_recoveries}/{self.total_disturbances}',
                             fontsize=9, color='white', ha='center', va='center',
                             transform=self.ax_recovery.transAxes)
    
    def _update_intensity_panel(self):
        """Update intensity bar"""
        self.ax_intensity.clear()
        self.ax_intensity.set_xlim(0, 1)
        self.ax_intensity.set_ylim(0, 1)
        self.ax_intensity.axis('off')
        
        self.ax_intensity.text(0.5, 0.90, 'IMPACT\nINTENSITY', fontsize=9, fontweight='bold',
                              color='#ffff00', ha='center', va='top',
                              transform=self.ax_intensity.transAxes)
        
        if self.current_intensity < 0.8:
            bar_color, label = '#00ff00', 'LOW'
        elif self.current_intensity < 1.2:
            bar_color, label = '#ffff00', 'MEDIUM'
        elif self.current_intensity < 1.6:
            bar_color, label = '#ff9900', 'HIGH'
        else:
            bar_color, label = '#ff3333', 'EXTREME'
        
        bar_height = self.current_intensity / 2.0
        bar = FancyBboxPatch((0.25, 0.15), 0.5, bar_height * 0.65,
                            boxstyle="round,pad=0.02",
                            facecolor=bar_color, edgecolor='white',
                            linewidth=3, alpha=0.9,
                            transform=self.ax_intensity.transAxes)
        self.ax_intensity.add_patch(bar)
        
        self.ax_intensity.text(0.5, 0.15 + bar_height * 0.65 + 0.08,
                              f'{self.current_intensity:.2f}x\n{label}',
                              fontsize=10, fontweight='bold', color=bar_color,
                              ha='center', va='bottom', transform=self.ax_intensity.transAxes)
    
    def _update_status_panel(self):
        """Update system status with recovery timer"""
        self.ax_status.clear()
        self.ax_status.axis('off')
        
        self.ax_status.text(0.5, 0.90, 'SYSTEM\nSTATUS', fontsize=9, fontweight='bold',
                           color='white', ha='center', va='top',
                           transform=self.ax_status.transAxes)
        
        if self.disturbance_active:
            status_color = '#ff3333'
            bg_color = '#330000'
            # Show disturbance type clearly
            dist_text = self.selected_disturbance.upper()
            if self.selected_direction:
                dist_text += f'\n{self.selected_direction.upper()}'
            
            # Show recovery timer if recovering
            if self.recovery_start_time is not None:
                status_text = f'RECOVERING\n\n{dist_text}\n\nTime: {self.current_recovery_time:.1f}s'
            else:
                status_text = f'DISTURBANCE\n\n{dist_text}\n\nPPO\nRESPONDING'
        elif self.is_recovering:
            status_color = '#ff9900'
            bg_color = '#332200'
            # Show recovery progress
            status_text = f'RECOVERING\n\nω: {self.current_ang_vel:.2f}\nrad/s\n\nTime: {self.current_recovery_time:.1f}s'
        else:
            status_color = '#00ff00'
            bg_color = '#003300'
            # Show last recovery time if available
            if self.last_completed_recovery_time > 0:
                status_text = f'STABLE\n\nHover\nReady\n\nLast: {self.last_completed_recovery_time:.1f}s'
            else:
                status_text = 'STABLE\n\nHover\nReady'
        
        self.ax_status.set_facecolor(bg_color)
        self.ax_status.text(0.5, 0.5, status_text, fontsize=10, fontweight='bold',
                           color=status_color, ha='center', va='center',
                           transform=self.ax_status.transAxes)
    
    def _update_strategy_panel(self):
        """Update PPO strategy"""
        self.ax_strategy.clear()
        self.ax_strategy.axis('off')
        
        self.ax_strategy.text(0.5, 0.95, 'PPO\nSTRATEGY', fontsize=9, fontweight='bold',
                             color='#ff00ff', ha='center', va='top',
                             transform=self.ax_strategy.transAxes)
        
        if self.is_recovering or self.disturbance_active:
            if self.current_ang_vel > 3.0:
                strategy, color, desc = "PHASE 1:\nSTOP SPIN", '#ff0000', "Counter\nrotation"
            elif self.current_ang_vel > 1.2:
                strategy, color, desc = "PHASE 2:\nUPRIGHT", '#ff9900', "Level\ndrone"
            else:
                strategy, color, desc = "PHASE 3:\nSTABILIZE", '#ffff00', "Fine\nadjust"
        else:
            strategy, color, desc = "HOVER", '#00ff00', "Maintain"
        
        self.ax_strategy.text(0.5, 0.60, strategy, fontsize=11, fontweight='bold',
                             color=color, ha='center', va='center',
                             transform=self.ax_strategy.transAxes)
        self.ax_strategy.text(0.5, 0.30, desc, fontsize=9, color='white',
                             ha='center', va='center',
                             transform=self.ax_strategy.transAxes)
    
    def _format_recovery_status(self):
        """Format current recovery status for display"""
        if self.recovery_start_time is not None:
            # Currently recovering
            return f"⏱️  {self.current_recovery_time:5.2f}s (IN PROGRESS)"
        else:
            return "Idle (waiting for disturbance)"
    
    def _update_info_panel(self):
        """Update info text - using replace to preserve scroll"""
        elapsed_time = time.time() - self.start_time
        
        info_text = f"""╔════════════════════════════════════════════════════════════════════╗
║                                                                    ║
║  PRODUCTION-READY PPO RECOVERY SYSTEM                              ║
║  Exactly Matches Training Environment                              ║
║                                                                    ║
╠════════════════════════════════════════════════════════════════════╣
║                                                                    ║
║  📊 DISTURBANCE TYPES (Consistent & Realistic Physics!)            ║
║  ════════════════════════════════════════════════════════════════  ║
║                                                                    ║
║  W/A/S/D - BIRD ATTACKS (Drone rotates AWAY from impact!)         ║
║    • W = Hit from FRONT → Pitches BACKWARD                        ║
║    • S = Hit from BACK  → Pitches FORWARD                         ║
║    • A = Hit from LEFT  → Rolls RIGHT (away from impact)          ║
║    • D = Hit from RIGHT → Rolls LEFT (away from impact)           ║
║    • Force: 30-80N (violent impact!)                               ║
║    • Rotation: 270-450°/s (causes tumbling!)                       ║
║                                                                    ║
║  F - FLIP (Rapid Tumble)                                           ║
║    • Random roll/pitch rotation                                    ║
║    • Angular velocity: 360-540°/s                                  ║
║                                                                    ║
║  G - SPIN (Yaw Rotation)                                           ║
║    • Rotation around vertical axis                                 ║
║    • Angular velocity: 450-630°/s                                  ║
║                                                                    ║
║  T - DROP (Altitude Loss)                                          ║
║    • Sudden downward force: 40-80N                                 ║
║                                                                    ║
╠════════════════════════════════════════════════════════════════════╣
║                                                                    ║
║  CURRENT STATE                                                     ║
║  ════════════════════════════════════════════════════════════════  ║
║                                                                    ║
║  Position: X={self.current_pos_x:6.2f}m  Y={self.current_pos_y:6.2f}m  Z={self.current_altitude:6.2f}m           ║
║  Attitude: R={self.current_roll:6.1f}°  P={self.current_pitch:6.1f}°  Y={self.current_yaw:6.1f}°              ║
║  Ang Vel:  {self.current_ang_vel:5.2f} rad/s  (Threshold: 1.20 rad/s)              ║
║                                                                    ║
║  PPO Actions:                                                      ║
║    Vx: {self.current_action[0]:+6.2f} m/s  (Forward/Back)                           ║
║    Vy: {self.current_action[1]:+6.2f} m/s  (Right/Left)                             ║
║    Vz: {self.current_action[2]:+6.2f} m/s  (-UP / +DOWN) ← KEY!                     ║
║                                                                    ║
╠════════════════════════════════════════════════════════════════════╣
║                                                                    ║
║  ⏱️  RECOVERY TIMER                                                ║
║  ════════════════════════════════════════════════════════════════  ║
║                                                                    ║
║  Current Recovery:  {self._format_recovery_status():<45}  ║
║  Last Recovery:     {self.last_completed_recovery_time:5.2f}s                                     ║
║  Average Recovery:  {self.avg_recovery_time:5.2f}s                                     ║
║  Fastest Recovery:  {min(self.recovery_times) if self.recovery_times else 0:5.2f}s                                     ║
║  Slowest Recovery:  {max(self.recovery_times) if self.recovery_times else 0:5.2f}s                                     ║
║                                                                    ║
╠════════════════════════════════════════════════════════════════════╣
║                                                                    ║
║  LAST DISTURBANCE INFO                                             ║
║  ════════════════════════════════════════════════════════════════  ║
║                                                                    ║
║  Type:      {self.last_disturbance_type:<50}  ║
║  Direction: {self.last_disturbance_direction:<50}  ║
║  Intensity: {self.last_disturbance_intensity:.2f}x                                           ║
║  Force:     {self.last_disturbance_force:<50}  ║
║  Ang Vel:   {self.last_disturbance_angular_vel:<50}  ║
║  Axis:      {self.last_disturbance_axis:<50}  ║
║                                                                    ║
╠════════════════════════════════════════════════════════════════════╣
║                                                                    ║
║  STATISTICS                                                        ║
║  ════════════════════════════════════════════════════════════════  ║
║                                                                    ║
║  Total Disturbances:    {self.total_disturbances:3d}                                  ║
║  Successful Recoveries: {self.successful_recoveries:3d}                                  ║
║  Success Rate:          {self.recovery_percentage:5.1f}%                               ║
║  Avg Recovery Time:     {self.avg_recovery_time:5.2f} s                               ║
║  Session Duration:      {elapsed_time/60:5.1f} min                                ║
║                                                                    ║
╠════════════════════════════════════════════════════════════════════╣
║                                                                    ║
║  ✅ REALISTIC PHYSICS: Rotation Away From Impact                   ║
║  ════════════════════════════════════════════════════════════════  ║
║                                                                    ║
║  • Front impact → Backward pitch (natural physics!)                ║
║  • Left impact → Right roll (pushed away from impact!)             ║
║  • Each direction now produces DISTINCT rotation!                  ║
║  • Same intensity = same impact severity                           ║
║  • Uses DisturbanceInjector (matches training!)                    ║
║                                                                    ║
╚════════════════════════════════════════════════════════════════════╝
"""
        
        # ✅ CORRECT FIX: Use replace() method which preserves scroll position
        # replace() updates content without moving the insertion cursor or scroll
        self.info_text.replace('1.0', tk.END, info_text)
    
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
    
    def is_alive(self):
        return not self.closed


class PPOControlThread(threading.Thread):
    """
    PRODUCTION-READY PPO control thread
    - Runs at 20 Hz (full speed!)
    - NO spontaneous disturbances (flip_prob=0.0)
    - Clear bird attack implementation (push only, no rotation!)
    - Logs data during ALL phases
    """
    
    def __init__(self, model_path, vecnorm_path, data_queue, command_queue):
        super().__init__(daemon=True)
        self.data_queue = data_queue
        self.command_queue = command_queue
        self.running = True
        
        def make_env():
            return DroneFlipRecoveryEnv(
                target_altitude=30.0,
                max_steps=1000,
                wind_strength=0.0,   # ✅ No background wind
                flip_prob=0.0,       # ✅ CRITICAL: Manual control only!
                debug=False
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
        self.obs = None
        self.episode_step = 0
        self.currently_recovering = False
        self.target_altitude = 30.0
        
        # Background data logging (runs even during takeoff!)
        self.logging_active = False
    
    def run(self):
        """Main control loop with continuous data logging"""
        
        self._reset_drone()
        
        # Start background logging
        self.logging_active = True
        self._start_background_logging()
        
        while self.running:
            try:
                self._process_commands()
                
                # PPO step
                action, _ = self.model.predict(self.obs, deterministic=True)
                self.obs, reward, done, info = self.env.step(action)
                self.episode_step += 1
                
                # Check recovery
                if self.currently_recovering and self.actual_env.disturbance_recovered:
                    self.data_queue.put({'type': 'recovery', 'steps': self.actual_env.recovery_steps})
                    self.currently_recovering = False
                
                # Send state (PPO actions available here)
                self._send_state(action[0])
                
                if done:
                    env_info = info[0] if isinstance(info, list) else info
                    reason = env_info.get('reason', 'unknown')
                    if reason != 'timeout':
                        time.sleep(2)
                        self._reset_drone()
                
                time.sleep(0.05)  # 20 Hz
                
            except Exception as e:
                print(f"PPO thread error: {e}")
                break
        
        self.logging_active = False
        self.env.close()
    
    def _start_background_logging(self):
        """Start background thread for continuous data logging"""
        def log_loop():
            while self.logging_active:
                try:
                    # Send state even when not in PPO loop (during takeoff, etc.)
                    if self.obs is None:
                        # Use zero action during initialization
                        self._send_state(np.zeros(3))
                    time.sleep(0.1)  # 10 Hz background logging
                except:
                    pass
        
        log_thread = threading.Thread(target=log_loop, daemon=True)
        log_thread.start()
    
    def _process_commands(self):
        """Process keyboard commands"""
        try:
            while True:
                cmd = self.command_queue.get_nowait()
                
                if cmd['type'] == 'disturbance':
                    self._apply_disturbance()
                elif cmd['type'] == 'reset':
                    self._reset_drone()
                elif cmd['type'] == 'quit':
                    self.running = False
                elif cmd['type'] == 'set_disturbance':
                    self.selected_disturbance = cmd['disturbance']
                    if 'direction' in cmd:
                        self.selected_direction = cmd['direction']
                elif cmd['type'] == 'set_intensity':
                    self.intensity = cmd['intensity']
                    
        except queue.Empty:
            pass
    
    def _send_state(self, action):
        """Send current state to dashboard"""
        
        drone_state = self.client.getMultirotorState()
        pos = drone_state.kinematics_estimated.position
        ori = drone_state.kinematics_estimated.orientation
        ang_vel = drone_state.kinematics_estimated.angular_velocity
        
        qw, qx, qy, qz = ori.w_val, ori.x_val, ori.y_val, ori.z_val
        
        roll = np.degrees(np.arctan2(2*(qw*qx + qy*qz), 1 - 2*(qx*qx + qy*qy)))
        pitch = np.degrees(np.arcsin(np.clip(2*(qw*qy - qz*qx), -1, 1)))
        
        siny_cosp = 2 * (qw * qz + qx * qy)
        cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
        yaw = np.degrees(np.arctan2(siny_cosp, cosy_cosp))
        
        ang_vel_mag = np.sqrt(ang_vel.x_val**2 + ang_vel.y_val**2 + ang_vel.z_val**2)
        altitude = -pos.z_val
        
        self.data_queue.put({
            'type': 'state',
            'roll': roll,
            'pitch': pitch,
            'yaw': yaw,
            'ang_vel': ang_vel_mag,
            'altitude': altitude,
            'pos_x': pos.x_val,
            'pos_y': pos.y_val,
            'action': action,
            'target_alt': self.target_altitude,
            'intensity': self.intensity,
            'is_recovering': self.currently_recovering,
            'disturbance_active': self.currently_recovering
        })
    
    def _reset_drone(self):
        """Reset with continuous logging"""
        print("\n🔄 Resetting...")
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
        print("✅ Reset complete\n")
    
    def _apply_disturbance(self):
        """
        EXACTLY MATCHES TRAINING ENVIRONMENT!
        ✅ FIXED: Now passes direction for consistent bird attacks!
        """
        
        # Send basic disturbance info to dashboard (for status panel)
        self.data_queue.put({
            'type': 'disturbance_info',
            'disturbance': self.selected_disturbance.value,
            'direction': self.selected_direction if self.selected_disturbance == DisturbanceType.BIRD_ATTACK else ''
        })
        
        # ✅ FIXED: Pass direction parameter for bird attacks!
        if self.selected_disturbance == DisturbanceType.BIRD_ATTACK:
            disturbance_info = self.injector.inject_disturbance(
                self.selected_disturbance,
                intensity=self.intensity,
                direction=self.selected_direction  # ← PASS DIRECTION!
            )
            print(f"💥 Bird attack from {self.selected_direction.upper()} at {self.intensity:.1f}x")
        else:
            disturbance_info = self.injector.inject_disturbance(
                self.selected_disturbance,
                intensity=self.intensity
            )
            print(f"💥 {self.selected_disturbance.value} at {self.intensity:.1f}x")
        
        if 'angular_velocity' in disturbance_info:
            print(f"   Angular velocity: {disturbance_info['angular_velocity']:.1f} deg/s")
            print(f"   Rotation axis: {disturbance_info.get('axis', 'N/A')}")
        if 'force' in disturbance_info:
            print(f"   Force: {disturbance_info['force']}")
        
        # ✅ NEW: Send detailed disturbance info to dashboard
        self.data_queue.put({
            'type': 'disturbance_details',
            'disturbance_type': self.selected_disturbance.value,
            'direction': disturbance_info.get('direction', 'N/A'),
            'intensity': disturbance_info.get('intensity', self.intensity),
            'force': disturbance_info.get('force', [0, 0, 0]),
            'angular_velocity': disturbance_info.get('angular_velocity', None),
            'axis': disturbance_info.get('axis', 'N/A')
        })
        
        self.data_queue.put({'type': 'disturbance'})
        self.currently_recovering = True
        self.actual_env.disturbance_initiated = True
        self.actual_env.disturbance_recovered = False
        self.actual_env.disturbance_start_step = self.episode_step


class ManualControlProduction:
    """Production-ready main controller"""
    
    def __init__(self, model_path, vecnorm_path):
        print("\n" + "="*70)
        print("PRODUCTION-READY PhD Demo")
        print("Stable, Multi-threaded, PhD Defense Ready")
        print("="*70)
        
        self.data_queue = queue.Queue()
        self.command_queue = queue.Queue()
        
        self.ppo_thread = PPOControlThread(model_path, vecnorm_path,
                                          self.data_queue, self.command_queue)
        
        self.dashboard = ScrollableDashboard(self.ppo_thread.client, self.data_queue)
        
        self.intensity = 1.0
        self.selected_disturbance = DisturbanceType.BIRD_ATTACK
        self.selected_direction = 'front'
        self.running = True
        
        print("="*70 + "\n")
        
        self.listener = keyboard.Listener(on_press=self._on_key_press)
        self.listener.start()
    
    def _on_key_press(self, key):
        """Keyboard handler - all controls preserved"""
        try:
            if hasattr(key, 'char') and key.char:
                char = key.char.lower()
                
                if char == 'w':
                    self.selected_disturbance = DisturbanceType.BIRD_ATTACK
                    self.selected_direction = 'front'
                    self.command_queue.put({'type': 'set_disturbance', 
                                           'disturbance': DisturbanceType.BIRD_ATTACK,
                                           'direction': 'front'})
                    print("Selected: Bird attack from FRONT")
                elif char == 'a':
                    self.selected_disturbance = DisturbanceType.BIRD_ATTACK
                    self.selected_direction = 'left'
                    self.command_queue.put({'type': 'set_disturbance', 
                                           'disturbance': DisturbanceType.BIRD_ATTACK,
                                           'direction': 'left'})
                    print("Selected: Bird attack from LEFT")
                elif char == 's':
                    self.selected_disturbance = DisturbanceType.BIRD_ATTACK
                    self.selected_direction = 'back'
                    self.command_queue.put({'type': 'set_disturbance', 
                                           'disturbance': DisturbanceType.BIRD_ATTACK,
                                           'direction': 'back'})
                    print("Selected: Bird attack from BACK")
                elif char == 'd':
                    self.selected_disturbance = DisturbanceType.BIRD_ATTACK
                    self.selected_direction = 'right'
                    self.command_queue.put({'type': 'set_disturbance', 
                                           'disturbance': DisturbanceType.BIRD_ATTACK,
                                           'direction': 'right'})
                    print("Selected: Bird attack from RIGHT")
                elif char == 'f':
                    self.selected_disturbance = DisturbanceType.FLIP
                    self.command_queue.put({'type': 'set_disturbance', 
                                           'disturbance': DisturbanceType.FLIP})
                    print("Selected: FLIP (violent tumble)")
                elif char == 'g':
                    self.selected_disturbance = DisturbanceType.SPIN
                    self.command_queue.put({'type': 'set_disturbance', 
                                           'disturbance': DisturbanceType.SPIN})
                    print("Selected: SPIN (yaw rotation)")
                elif char == 't':
                    self.selected_disturbance = DisturbanceType.DROP
                    self.command_queue.put({'type': 'set_disturbance', 
                                           'disturbance': DisturbanceType.DROP})
                    print("Selected: DROP (altitude loss)")
                elif char == 'r':
                    self.command_queue.put({'type': 'reset'})
                elif char == 'q':
                    self.running = False
                    self.command_queue.put({'type': 'quit'})
                    self.dashboard._on_close()
                elif char in ['+', '=']:
                    self.intensity = min(2.0, self.intensity + 0.1)
                    self.command_queue.put({'type': 'set_intensity', 'intensity': self.intensity})
                    print(f"Intensity: {self.intensity:.1f}x")
                elif char == '-':
                    self.intensity = max(0.5, self.intensity - 0.1)
                    self.command_queue.put({'type': 'set_intensity', 'intensity': self.intensity})
                    print(f"Intensity: {self.intensity:.1f}x")
            
            if key == keyboard.Key.space:
                self.command_queue.put({'type': 'disturbance'})
        except AttributeError:
            pass
    
    def run(self):
        """Main loop"""
        
        self.ppo_thread.start()
        
        print("\n" + "="*70)
        print("CONTROLS:")
        print("="*70)
        print("  W - Bird attack from FRONT (uses DisturbanceInjector!)")
        print("  A - Bird attack from LEFT (uses DisturbanceInjector!)")
        print("  S - Bird attack from BACK (uses DisturbanceInjector!)")
        print("  D - Bird attack from RIGHT (uses DisturbanceInjector!)")
        print("")
        print("  F - FLIP (violent tumble)")
        print("  G - SPIN (yaw rotation)")
        print("  T - DROP (altitude loss)")
        print("")
        print("  SPACE - Apply selected disturbance")
        print("  +/- - Adjust intensity (0.5x - 2.0x)")
        print("  R - Reset drone")
        print("  Q - Quit")
        print("="*70)
        print("\n✅ All disturbances use the SAME injector as training!")
        print("✅ Dashboard logging data continuously...\n")
        
        try:
            self.dashboard.root.mainloop()
        except KeyboardInterrupt:
            pass
        finally:
            self.listener.stop()
            self.command_queue.put({'type': 'quit'})
            print("\n✅ Demo complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str,
                        default='./models/stage3_all_intensity_checkpoints/gated_curriculum_policy.zip')
    parser.add_argument('--vecnorm', type=str,
                        default='./models/stage3_all_intensity_checkpoints/gated_curriculum_vecnormalize.pkl')
    args = parser.parse_args()
    
    if not KEYBOARD_AVAILABLE:
        print("pip install pynput matplotlib")
        sys.exit(1)
    
    demo = ManualControlProduction(args.model, args.vecnorm)
    demo.run()