"""
IMU Feature Extraction for Impact Classification
=================================================

Extracts features from IMU data to classify impact types.
Based on peer-reviewed literature:

- Paper #3 (Sensors 2021): FFT + Wavelet Packet Decomposition
- Paper #6 (Applied Sciences 2023): Statistical features, rolling windows
- Paper #7 (Sensors 2018): Spectrogram-based features

Feature Categories:
1. Jerk features (rate of acceleration change)
2. Frequency domain features (FFT)
3. Wavelet features (WPD energy)
4. Statistical features (mean, std, variance)
5. Temporal features (duration, sustained count)

These features enable classification into:
- Sharp Collision (high jerk, brief duration)
- Sustained Force (low jerk, long duration)
- Rotational (high angular acceleration)
- Free-fall (vertical acceleration drop)
"""

import numpy as np
from scipy import signal
from scipy.fft import fft, fftfreq
import pywt  # PyWavelets for wavelet decomposition
from collections import deque
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class IMUBuffer:
    """
    Rolling buffer for IMU data.
    
    Think of this as a "memory" - we need to remember recent sensor
    readings to detect patterns (like how you'd notice if someone
    keeps pushing you vs a single shove).
    """
    accel: deque  # Linear acceleration (m/s²)
    gyro: deque   # Angular velocity (rad/s)
    timestamp: deque
    maxlen: int = 50  # ~5 seconds at 10Hz, ~2.5s at 20Hz
    
    def __init__(self, maxlen: int = 50):
        self.maxlen = maxlen
        self.accel = deque(maxlen=maxlen)
        self.gyro = deque(maxlen=maxlen)
        self.timestamp = deque(maxlen=maxlen)
    
    def add_sample(self, accel: np.ndarray, gyro: np.ndarray, timestamp: float):
        """Add new IMU sample to buffer"""
        self.accel.append(accel.copy())
        self.gyro.append(gyro.copy())
        self.timestamp.append(timestamp)
    
    def get_arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get IMU data as numpy arrays"""
        return np.array(self.accel), np.array(self.gyro)
    
    def is_full(self) -> bool:
        """Check if buffer has enough samples"""
        return len(self.accel) >= self.maxlen
    
    def clear(self):
        """Clear buffer"""
        self.accel.clear()
        self.gyro.clear()
        self.timestamp.clear()


class IMUFeatureExtractor:
    """
    Extracts features from IMU data for impact classification.
    
    Uses methods from literature:
    1. Time-domain: jerk, statistical features
    2. Frequency-domain: FFT, dominant frequencies
    3. Time-frequency: Wavelet Packet Decomposition
    
    Analogy: Like a music analyzer that extracts tempo (frequency),
    loudness (amplitude), and rhythm patterns (wavelets) from audio.
    """
    
    def __init__(self, sampling_rate: float = 10.0):
        """
        Args:
            sampling_rate: IMU sampling rate in Hz (default 10Hz)
        """
        self.sampling_rate = sampling_rate
        self.gravity = 9.81  # m/s²
    
    def extract_features(self, imu_buffer: IMUBuffer) -> Dict[str, float]:
        """
        Extract comprehensive feature set from IMU buffer.
        
        Returns dictionary with ~25 features for classification.
        
        Args:
            imu_buffer: Buffer containing recent IMU samples
            
        Returns:
            Dictionary of features with keys:
            - jerk_*: Jerk-based features
            - fft_*: Frequency domain features
            - wpd_*: Wavelet features
            - stat_*: Statistical features
            - temporal_*: Temporal pattern features
        """
        if not imu_buffer.is_full():
            # Not enough data yet
            return self._empty_features()
        
        accel_array, gyro_array = imu_buffer.get_arrays()
        
        features = {}
        
        # 1. JERK FEATURES (Paper #3 - critical for collision detection)
        features.update(self._extract_jerk_features(accel_array))
        
        # 2. FREQUENCY FEATURES (Paper #3 - FFT best performance)
        features.update(self._extract_frequency_features(accel_array, gyro_array))
        
        # 3. WAVELET FEATURES (Paper #3 - WPD energy)
        features.update(self._extract_wavelet_features(accel_array))
        
        # 4. STATISTICAL FEATURES (Paper #6 - rolling windows)
        features.update(self._extract_statistical_features(accel_array, gyro_array))
        
        # 5. TEMPORAL FEATURES (impact duration, sustained count)
        features.update(self._extract_temporal_features(accel_array, gyro_array))
        
        return features
    
    def _extract_jerk_features(self, accel_array: np.ndarray) -> Dict[str, float]:
        """
        Extract jerk-based features (rate of change of acceleration).
        
        Jerk is THE key feature for distinguishing sharp vs sustained impacts.
        
        Analogy: Jerk is like the "sharpness" of a push. A slap (sharp collision)
        has high jerk, while leaning on something (sustained force) has low jerk.
        """
        # Compute jerk (derivative of acceleration)
        jerk = np.diff(accel_array, axis=0)
        jerk_magnitude = np.linalg.norm(jerk, axis=1)
        
        # Key features
        max_jerk = np.max(jerk_magnitude)
        mean_jerk = np.mean(jerk_magnitude)
        std_jerk = np.std(jerk_magnitude)
        
        # Jerk in each axis (for direction detection)
        jerk_x = np.max(np.abs(jerk[:, 0]))
        jerk_y = np.max(np.abs(jerk[:, 1]))
        jerk_z = np.max(np.abs(jerk[:, 2]))
        
        # Dominant jerk axis (which direction is impact from?)
        dominant_axis = np.argmax([jerk_x, jerk_y, jerk_z])
        
        return {
            'jerk_max': max_jerk,
            'jerk_mean': mean_jerk,
            'jerk_std': std_jerk,
            'jerk_x': jerk_x,
            'jerk_y': jerk_y,
            'jerk_z': jerk_z,
            'jerk_dominant_axis': float(dominant_axis),
        }
    
    def _extract_frequency_features(self, 
                                   accel_array: np.ndarray, 
                                   gyro_array: np.ndarray) -> Dict[str, float]:
        """
        Extract frequency-domain features using FFT.
        
        Different impacts have different frequency signatures:
        - Sharp collision: High frequency content (sudden change)
        - Sustained force: Low frequency content (slow variation)
        
        Analogy: Like how a drum hit (sharp) has high-pitch sound vs
        a tuba note (sustained) has low-pitch sound.
        """
        # FFT on acceleration magnitude
        accel_magnitude = np.linalg.norm(accel_array, axis=1)
        fft_accel = fft(accel_magnitude)
        fft_freq = fftfreq(len(accel_magnitude), 1.0 / self.sampling_rate)
        
        # Only positive frequencies
        positive_freq_idx = fft_freq > 0
        fft_magnitude = np.abs(fft_accel[positive_freq_idx])
        fft_freq_positive = fft_freq[positive_freq_idx]
        
        # Dominant frequency (frequency with max power)
        dominant_freq_idx = np.argmax(fft_magnitude)
        dominant_frequency = fft_freq_positive[dominant_freq_idx]
        
        # Power spectral density features
        total_power = np.sum(fft_magnitude ** 2)
        low_freq_power = np.sum(fft_magnitude[fft_freq_positive < 2.0] ** 2)  # <2Hz
        high_freq_power = np.sum(fft_magnitude[fft_freq_positive > 5.0] ** 2)  # >5Hz
        
        # Spectral centroid (weighted average frequency)
        if total_power > 1e-6:
            spectral_centroid = np.sum(fft_freq_positive * fft_magnitude) / np.sum(fft_magnitude)
        else:
            spectral_centroid = 0.0
        
        # FFT on gyro (angular velocity)
        gyro_magnitude = np.linalg.norm(gyro_array, axis=1)
        fft_gyro = fft(gyro_magnitude)
        gyro_dominant_freq = fftfreq(len(gyro_magnitude), 1.0 / self.sampling_rate)[np.argmax(np.abs(fft_gyro[1:len(fft_gyro)//2]))+1]
        
        return {
            'fft_dominant_freq': dominant_frequency,
            'fft_total_power': total_power,
            'fft_low_freq_ratio': low_freq_power / (total_power + 1e-6),
            'fft_high_freq_ratio': high_freq_power / (total_power + 1e-6),
            'fft_spectral_centroid': spectral_centroid,
            'fft_gyro_dominant_freq': gyro_dominant_freq,
        }
    
    def _extract_wavelet_features(self, accel_array: np.ndarray) -> Dict[str, float]:
        """
        Extract wavelet features using Wavelet Packet Decomposition (WPD).
        
        Paper #3 found WPD + FFT gave best classification performance.
        
        Wavelets capture transient features better than FFT - good for
        detecting brief impacts within longer time series.
        
        Analogy: Like a microscope that can zoom in on different time scales.
        FFT sees the whole picture, wavelets see details at different scales.
        """
        # Use acceleration magnitude
        accel_magnitude = np.linalg.norm(accel_array, axis=1)
        
        # Wavelet Packet Decomposition (Daubechies wavelet)
        wavelet = 'db4'
        level = 3
        
        try:
            wp = pywt.WaveletPacket(data=accel_magnitude, wavelet=wavelet, maxlevel=level)
            
            # Extract energy from each decomposition level
            wpd_energies = []
            for node in wp.get_level(level, order='freq'):
                wpd_energies.append(np.sum(node.data ** 2))
            
            wpd_energies = np.array(wpd_energies)
            total_energy = np.sum(wpd_energies)
            
            # Energy distribution features
            wpd_energy_max = np.max(wpd_energies)
            wpd_energy_mean = np.mean(wpd_energies)
            wpd_energy_std = np.std(wpd_energies)
            
            # Entropy (measure of energy spread)
            wpd_energies_norm = wpd_energies / (total_energy + 1e-6)
            wpd_entropy = -np.sum(wpd_energies_norm * np.log(wpd_energies_norm + 1e-6))
            
        except Exception as e:
            # If WPD fails, return zeros
            wpd_energy_max = 0.0
            wpd_energy_mean = 0.0
            wpd_energy_std = 0.0
            wpd_entropy = 0.0
        
        return {
            'wpd_energy_max': wpd_energy_max,
            'wpd_energy_mean': wpd_energy_mean,
            'wpd_energy_std': wpd_energy_std,
            'wpd_entropy': wpd_entropy,
        }
    
    def _extract_statistical_features(self, 
                                     accel_array: np.ndarray,
                                     gyro_array: np.ndarray) -> Dict[str, float]:
        """
        Extract statistical features from IMU data.
        
        Paper #6 recommends rolling window statistics for time-series classification.
        """
        # Current acceleration anomaly (deviation from gravity)
        current_accel = accel_array[-1]
        expected_accel = np.array([0, 0, self.gravity])  # Hovering expectation
        accel_anomaly = np.linalg.norm(current_accel - expected_accel)
        
        # Rolling window statistics (last 10 samples = ~1 second)
        window_size = min(10, len(accel_array))
        recent_accel = accel_array[-window_size:]
        recent_gyro = gyro_array[-window_size:]
        
        # Acceleration statistics
        accel_mean = np.mean(np.linalg.norm(recent_accel, axis=1))
        accel_std = np.std(np.linalg.norm(recent_accel, axis=1))
        accel_max = np.max(np.linalg.norm(recent_accel, axis=1))
        accel_min = np.min(np.linalg.norm(recent_accel, axis=1))
        
        # Gyro statistics
        gyro_mean = np.mean(np.linalg.norm(recent_gyro, axis=1))
        gyro_std = np.std(np.linalg.norm(recent_gyro, axis=1))
        gyro_max = np.max(np.linalg.norm(recent_gyro, axis=1))
        
        return {
            'stat_accel_anomaly': accel_anomaly,
            'stat_accel_mean': accel_mean,
            'stat_accel_std': accel_std,
            'stat_accel_range': accel_max - accel_min,
            'stat_gyro_mean': gyro_mean,
            'stat_gyro_std': gyro_std,
            'stat_gyro_max': gyro_max,
        }
    
    def _extract_temporal_features(self,
                                  accel_array: np.ndarray,
                                  gyro_array: np.ndarray) -> Dict[str, float]:
        """
        Extract temporal pattern features.
        
        Key for distinguishing sharp (brief) vs sustained (long) impacts.
        """
        # Threshold for "anomalous" acceleration (>2G)
        anomaly_threshold = 2.0 * self.gravity
        
        # Count consecutive samples above threshold (sustained duration)
        accel_magnitudes = np.linalg.norm(accel_array, axis=1)
        expected_mag = self.gravity
        
        sustained_count = 0
        for mag in reversed(accel_magnitudes):
            if abs(mag - expected_mag) > anomaly_threshold:
                sustained_count += 1
            else:
                break
        
        # Angular acceleration (gyro derivative)
        angular_accel = np.diff(gyro_array, axis=0)
        angular_accel_magnitude = np.linalg.norm(angular_accel, axis=1)
        max_angular_accel = np.max(angular_accel_magnitude)
        
        # Vertical component (detect free-fall)
        vertical_accel = accel_array[:, 2]  # Z-axis
        vertical_accel_mean = np.mean(vertical_accel[-10:])  # Recent average
        
        return {
            'temporal_sustained_count': float(sustained_count),
            'temporal_max_angular_accel': max_angular_accel,
            'temporal_vertical_accel': vertical_accel_mean,
        }
    
    def _empty_features(self) -> Dict[str, float]:
        """Return empty feature dict (when buffer not full)"""
        return {
            'jerk_max': 0.0, 'jerk_mean': 0.0, 'jerk_std': 0.0,
            'jerk_x': 0.0, 'jerk_y': 0.0, 'jerk_z': 0.0,
            'jerk_dominant_axis': 0.0,
            'fft_dominant_freq': 0.0, 'fft_total_power': 0.0,
            'fft_low_freq_ratio': 0.0, 'fft_high_freq_ratio': 0.0,
            'fft_spectral_centroid': 0.0, 'fft_gyro_dominant_freq': 0.0,
            'wpd_energy_max': 0.0, 'wpd_energy_mean': 0.0,
            'wpd_energy_std': 0.0, 'wpd_entropy': 0.0,
            'stat_accel_anomaly': 0.0, 'stat_accel_mean': 0.0,
            'stat_accel_std': 0.0, 'stat_accel_range': 0.0,
            'stat_gyro_mean': 0.0, 'stat_gyro_std': 0.0,
            'stat_gyro_max': 0.0,
            'temporal_sustained_count': 0.0,
            'temporal_max_angular_accel': 0.0,
            'temporal_vertical_accel': 0.0,
        }


# Test the feature extractor
if __name__ == "__main__":
    print("="*70)
    print("IMU FEATURE EXTRACTION TEST")
    print("="*70)
    
    # Create synthetic IMU data
    np.random.seed(42)
    
    # Test case 1: Sharp collision
    print("\n" + "-"*70)
    print("Test 1: Sharp Collision Signature")
    print("-"*70)
    
    buffer = IMUBuffer(maxlen=50)
    extractor = IMUFeatureExtractor(sampling_rate=10.0)
    
    # Normal flight data
    for i in range(40):
        accel = np.array([0, 0, 9.81]) + np.random.randn(3) * 0.1
        gyro = np.random.randn(3) * 0.05
        buffer.add_sample(accel, gyro, i * 0.1)
    
    # Sharp collision (high jerk, brief)
    for i in range(10):
        if i < 3:  # Brief spike
            accel = np.array([15, 0, 9.81]) + np.random.randn(3) * 0.5
            gyro = np.random.randn(3) * 2.0
        else:
            accel = np.array([0, 0, 9.81]) + np.random.randn(3) * 0.1
            gyro = np.random.randn(3) * 0.05
        buffer.add_sample(accel, gyro, (40 + i) * 0.1)
    
    features = extractor.extract_features(buffer)
    
    print(f"Jerk Max: {features['jerk_max']:.2f} m/s³  (HIGH = sharp)")
    print(f"Sustained Count: {features['temporal_sustained_count']:.0f} samples  (LOW = brief)")
    print(f"FFT High Freq Ratio: {features['fft_high_freq_ratio']:.3f}  (HIGH = sharp)")
    print(f"Spectral Centroid: {features['fft_spectral_centroid']:.2f} Hz  (HIGH = sharp)")
    
    # Test case 2: Sustained force
    print("\n" + "-"*70)
    print("Test 2: Sustained Force Signature (Wind)")
    print("-"*70)
    
    buffer.clear()
    
    # Normal flight
    for i in range(20):
        accel = np.array([0, 0, 9.81]) + np.random.randn(3) * 0.1
        gyro = np.random.randn(3) * 0.05
        buffer.add_sample(accel, gyro, i * 0.1)
    
    # Sustained push (low jerk, long duration)
    for i in range(30):
        accel = np.array([3, 0, 9.81]) + np.random.randn(3) * 0.2  # Constant push
        gyro = np.random.randn(3) * 0.1
        buffer.add_sample(accel, gyro, (20 + i) * 0.1)
    
    features = extractor.extract_features(buffer)
    
    print(f"Jerk Max: {features['jerk_max']:.2f} m/s³  (LOW = sustained)")
    print(f"Sustained Count: {features['temporal_sustained_count']:.0f} samples  (HIGH = sustained)")
    print(f"FFT Low Freq Ratio: {features['fft_low_freq_ratio']:.3f}  (HIGH = sustained)")
    print(f"Spectral Centroid: {features['fft_spectral_centroid']:.2f} Hz  (LOW = sustained)")
    
    print("\n" + "="*70)
    print("✅ Feature extraction test complete!")
    print("="*70)
    print("\nKey Observations:")
    print("• Sharp collision: High jerk, brief duration, high frequency")
    print("• Sustained force: Low jerk, long duration, low frequency")
    print("• These features enable ML classification!")