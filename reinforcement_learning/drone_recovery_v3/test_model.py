"""
Model Testing Script for Drone Impact Recovery
===============================================
Test trained PPO models with various impact scenarios and visualize recovery performance.
"""

import argparse
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import json

from stable_baselines3 import PPO
from airsim_recovery_env import AirSimDroneRecoveryEnv
import airsim


class RecoveryTester:
    """Test drone recovery models with various impact scenarios."""
    
    def __init__(self, model_path: str, stage: int = 3, verbose: bool = True):
        """
        Initialize tester.
        
        Args:
            model_path: Path to trained PPO model
            stage: Stage to test (1, 2, or 3)
            verbose: Print detailed output
        """
        self.model_path = model_path
        self.stage = stage
        self.verbose = verbose
        
        # Load model
        print(f"\n📂 Loading model: {model_path}")
        self.model = PPO.load(model_path)
        print("✅ Model loaded successfully")
        
        # Create environment
        print(f"\n🔧 Creating test environment (Stage {stage})...")
        self.env = AirSimDroneRecoveryEnv(stage=stage, debug=verbose)
        print("✅ Environment created")
        
        # AirSim client for manual disturbances
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
    
    def test_hover_stability(self, duration: float = 30.0) -> Dict:
        """
        Test basic hover stability without disturbances.
        
        Args:
            duration: Test duration in seconds
        
        Returns:
            Test results dictionary
        """
        print("\n" + "="*80)
        print("TEST 1: HOVER STABILITY (No Disturbances)")
        print("="*80)
        print(f"Duration: {duration}s")
        print("Objective: Maintain stable hover at target position\n")
        
        obs, _ = self.env.reset()
        start_time = time.time()
        
        positions = []
        orientations = []
        rewards = []
        step_count = 0
        
        while time.time() - start_time < duration:
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = self.env.step(action)
            
            positions.append(obs[0:3].copy())
            orientations.append(obs[3])  # qw
            rewards.append(reward)
            step_count += 1
            
            if terminated or truncated:
                print(f"⚠️  Episode terminated early: {info.get('reason', 'unknown')}")
                break
            
            if step_count % 100 == 0 and self.verbose:
                pos_error = np.linalg.norm(obs[0:3] - np.array([0, 0, -10]))
                print(f"  Step {step_count}: Pos Error={pos_error:.3f}m, qw={obs[3]:.3f}, Reward={reward:.2f}")
        
        # Compute statistics
        positions = np.array(positions)
        target = np.array([0, 0, -10])
        pos_errors = np.linalg.norm(positions - target, axis=1)
        
        results = {
            "test": "hover_stability",
            "duration": time.time() - start_time,
            "steps": step_count,
            "mean_position_error": float(np.mean(pos_errors)),
            "max_position_error": float(np.max(pos_errors)),
            "mean_orientation": float(np.mean(orientations)),
            "min_orientation": float(np.min(orientations)),
            "mean_reward": float(np.mean(rewards)),
            "total_reward": float(np.sum(rewards)),
            "success": np.mean(pos_errors) < 0.5 and np.mean(orientations) > 0.95
        }
        
        self._print_results(results)
        return results
    
    def test_impact_scenario(self, impact_type: str, n_trials: int = 5) -> Dict:
        """
        Test recovery from specific impact type.
        
        Args:
            impact_type: Type of impact ('flip', 'spin', 'tumble', 'collision')
            n_trials: Number of test trials
        
        Returns:
            Test results dictionary
        """
        print("\n" + "="*80)
        print(f"TEST: {impact_type.upper()} RECOVERY")
        print("="*80)
        print(f"Trials: {n_trials}")
        print(f"Objective: Recover from {impact_type} and resume hover\n")
        
        trial_results = []
        
        for trial in range(n_trials):
            print(f"\n--- Trial {trial + 1}/{n_trials} ---")
            
            # Reset and hover
            obs, _ = self.env.reset()
            time.sleep(2.0)  # Let it stabilize
            
            # Apply impact
            print(f"💥 Applying {impact_type} impact...")
            self._apply_impact(impact_type)
            
            # Record recovery attempt
            start_time = time.time()
            step_count = 0
            recovered = False
            min_altitude = 10.0
            max_angular_vel = 0.0
            positions = []
            
            for _ in range(500):  # Max 500 steps (~25 seconds)
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.env.step(action)
                
                altitude = -obs[2]  # Convert NED to positive-up
                min_altitude = min(min_altitude, altitude)
                ang_vel_mag = np.linalg.norm(obs[10:13])
                max_angular_vel = max(max_angular_vel, ang_vel_mag)
                positions.append(obs[0:3].copy())
                
                step_count += 1
                
                # Check if recovered
                pos_error = np.linalg.norm(obs[0:3] - np.array([0, 0, -10]))
                if pos_error < 1.0 and abs(obs[3]) > 0.95 and ang_vel_mag < 0.5:
                    recovered = True
                    recovery_time = time.time() - start_time
                    print(f"✅ RECOVERED in {recovery_time:.2f}s at step {step_count}")
                    print(f"   Min altitude: {min_altitude:.2f}m")
                    print(f"   Max angular vel: {max_angular_vel:.2f} rad/s")
                    break
                
                if terminated:
                    reason = info.get('reason', 'unknown')
                    print(f"❌ FAILED: {reason}")
                    print(f"   Min altitude: {min_altitude:.2f}m")
                    print(f"   Steps survived: {step_count}")
                    break
            
            if not recovered and not terminated:
                print(f"⚠️  TIMEOUT: Did not recover in 500 steps")
                print(f"   Min altitude: {min_altitude:.2f}m")
            
            trial_results.append({
                "trial": trial + 1,
                "recovered": recovered,
                "recovery_time": recovery_time if recovered else None,
                "min_altitude": float(min_altitude),
                "max_angular_velocity": float(max_angular_vel),
                "steps": step_count,
                "terminated": terminated
            })
        
        # Aggregate results
        success_count = sum(1 for r in trial_results if r["recovered"])
        recovery_times = [r["recovery_time"] for r in trial_results if r["recovery_time"] is not None]
        min_altitudes = [r["min_altitude"] for r in trial_results]
        
        results = {
            "test": f"{impact_type}_recovery",
            "trials": n_trials,
            "successes": success_count,
            "success_rate": success_count / n_trials,
            "mean_recovery_time": float(np.mean(recovery_times)) if recovery_times else None,
            "mean_min_altitude": float(np.mean(min_altitudes)),
            "lowest_altitude_survived": float(np.min(min_altitudes)),
            "trial_details": trial_results
        }
        
        self._print_results(results)
        return results
    
    def test_all_impacts(self, n_trials: int = 5) -> Dict:
        """
        Test all impact types.
        
        Args:
            n_trials: Number of trials per impact type
        
        Returns:
            Combined test results
        """
        impact_types = ['flip', 'spin', 'tumble', 'collision']
        all_results = {}
        
        for impact_type in impact_types:
            all_results[impact_type] = self.test_impact_scenario(impact_type, n_trials)
        
        # Summary
        print("\n" + "="*80)
        print("📊 OVERALL IMPACT RECOVERY SUMMARY")
        print("="*80)
        for impact_type, results in all_results.items():
            success_rate = results["success_rate"]
            status = "✅" if success_rate >= 0.7 else "⚠️" if success_rate >= 0.5 else "❌"
            print(f"{status} {impact_type.upper():<12} Success Rate: {success_rate:>5.1%} ({results['successes']}/{results['trials']})")
        
        overall_success_rate = np.mean([r["success_rate"] for r in all_results.values()])
        print(f"\n{'='*80}")
        print(f"Overall Success Rate: {overall_success_rate:.1%}")
        print(f"{'='*80}\n")
        
        return all_results
    
    def test_continuous_disturbances(self, duration: float = 60.0, disturbance_freq: float = 0.1) -> Dict:
        """
        Test with continuous random disturbances.
        
        Args:
            duration: Test duration in seconds
            disturbance_freq: Probability of disturbance per step
        
        Returns:
            Test results
        """
        print("\n" + "="*80)
        print("TEST: CONTINUOUS DISTURBANCES")
        print("="*80)
        print(f"Duration: {duration}s")
        print(f"Disturbance Frequency: {disturbance_freq:.1%} per step")
        print("Objective: Maintain stability under continuous random impacts\n")
        
        obs, _ = self.env.reset()
        start_time = time.time()
        
        step_count = 0
        crash_count = 0
        disturbance_count = 0
        positions = []
        min_altitude = 10.0
        
        while time.time() - start_time < duration:
            action, _ = self.model.predict(obs, deterministic=True)
            
            # Random disturbance
            if np.random.rand() < disturbance_freq:
                impact_type = np.random.choice(['flip', 'spin', 'tumble'])
                self._apply_impact(impact_type)
                disturbance_count += 1
                if self.verbose and disturbance_count <= 5:
                    print(f"  💥 Disturbance {disturbance_count}: {impact_type}")
            
            obs, reward, terminated, truncated, info = self.env.step(action)
            
            altitude = -obs[2]
            min_altitude = min(min_altitude, altitude)
            positions.append(obs[0:3].copy())
            step_count += 1
            
            if terminated:
                crash_count += 1
                print(f"  ❌ Crash {crash_count} at step {step_count}: {info.get('reason', 'unknown')}")
                obs, _ = self.env.reset()
                time.sleep(1.0)
        
        # Statistics
        positions = np.array(positions)
        target = np.array([0, 0, -10])
        pos_errors = np.linalg.norm(positions - target, axis=1)
        
        results = {
            "test": "continuous_disturbances",
            "duration": time.time() - start_time,
            "steps": step_count,
            "disturbances_applied": disturbance_count,
            "crashes": crash_count,
            "crash_rate": crash_count / disturbance_count if disturbance_count > 0 else 0,
            "mean_position_error": float(np.mean(pos_errors)),
            "min_altitude": float(min_altitude),
            "success": crash_count == 0
        }
        
        self._print_results(results)
        return results
    
    def _apply_impact(self, impact_type: str) -> None:
        """Apply specific impact to drone."""
        state = self.client.simGetGroundTruthKinematics()
        
        if impact_type == 'flip':
            # Rapid pitch rotation (bird strike from front/back)
            state.angular_velocity.y_val = float(np.random.uniform(-8, 8))
        
        elif impact_type == 'spin':
            # Rapid yaw rotation (propeller strike)
            state.angular_velocity.z_val = float(np.random.uniform(-6, 6))
        
        elif impact_type == 'tumble':
            # Multi-axis rotation (collision)
            state.angular_velocity.x_val = float(np.random.uniform(-5, 5))
            state.angular_velocity.y_val = float(np.random.uniform(-5, 5))
            state.angular_velocity.z_val = float(np.random.uniform(-3, 3))
        
        elif impact_type == 'collision':
            # Combined rotation + translation
            state.angular_velocity.x_val = float(np.random.uniform(-4, 4))
            state.angular_velocity.y_val = float(np.random.uniform(-4, 4))
            state.linear_velocity.x_val += float(np.random.uniform(-3, 3))
            state.linear_velocity.y_val += float(np.random.uniform(-3, 3))
        
        self.client.simSetKinematics(state, ignore_collision=True)
        time.sleep(0.1)  # Let physics settle
    
    def _print_results(self, results: Dict) -> None:
        """Print formatted test results."""
        print("\n" + "-"*80)
        print("📊 RESULTS")
        print("-"*80)
        
        for key, value in results.items():
            if key not in ["trial_details"]:
                if isinstance(value, float):
                    print(f"   {key}: {value:.4f}")
                elif isinstance(value, bool):
                    status = "✅ PASS" if value else "❌ FAIL"
                    print(f"   {key}: {status}")
                else:
                    print(f"   {key}: {value}")
        
        print("-"*80 + "\n")
    
    def run_full_test_suite(self, save_results: bool = True) -> Dict:
        """
        Run complete test suite.
        
        Args:
            save_results: Save results to JSON file
        
        Returns:
            All test results
        """
        print("\n" + "="*80)
        print("🧪 FULL TEST SUITE")
        print("="*80)
        print(f"Model: {self.model_path}")
        print(f"Stage: {self.stage}")
        print("="*80)
        
        all_results = {}
        
        # Test 1: Hover stability
        all_results["hover_stability"] = self.test_hover_stability(duration=30.0)
        
        # Test 2: Individual impact recovery
        all_results["impact_recovery"] = self.test_all_impacts(n_trials=5)
        
        # Test 3: Continuous disturbances
        all_results["continuous_disturbances"] = self.test_continuous_disturbances(
            duration=60.0, 
            disturbance_freq=0.1
        )
        
        # Overall assessment
        hover_pass = all_results["hover_stability"]["success"]
        impact_pass = np.mean([
            r["success_rate"] for r in all_results["impact_recovery"].values()
        ]) >= 0.7
        continuous_pass = all_results["continuous_disturbances"]["success"]
        
        overall_pass = hover_pass and impact_pass and continuous_pass
        
        all_results["overall"] = {
            "hover_stability_pass": hover_pass,
            "impact_recovery_pass": impact_pass,
            "continuous_disturbances_pass": continuous_pass,
            "overall_pass": overall_pass
        }
        
        # Print final summary
        print("\n" + "="*80)
        print("🏆 FINAL ASSESSMENT")
        print("="*80)
        print(f"✅ Hover Stability: {'PASS' if hover_pass else 'FAIL'}")
        print(f"✅ Impact Recovery: {'PASS' if impact_pass else 'FAIL'}")
        print(f"✅ Continuous Disturbances: {'PASS' if continuous_pass else 'FAIL'}")
        print(f"\n{'='*80}")
        print(f"Overall: {'✅ PASS' if overall_pass else '❌ FAIL'}")
        print(f"{'='*80}\n")
        
        # Save results
        if save_results:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"test_results_stage{self.stage}_{timestamp}.json"
            with open(filename, 'w') as f:
                json.dump(all_results, f, indent=2)
            print(f"💾 Results saved to: {filename}\n")
        
        return all_results
    
    def close(self) -> None:
        """Cleanup resources."""
        self.env.close()


def main():
    parser = argparse.ArgumentParser(description="Test trained drone recovery model")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model (.zip)")
    parser.add_argument("--stage", type=int, default=3, choices=[1, 2, 3], help="Stage to test")
    parser.add_argument("--test", type=str, default="full", 
                       choices=["full", "hover", "flip", "spin", "tumble", "collision", "continuous"],
                       help="Test type to run")
    parser.add_argument("--trials", type=int, default=5, help="Number of trials for impact tests")
    parser.add_argument("--duration", type=float, default=30.0, help="Duration for timed tests (seconds)")
    parser.add_argument("--save", action="store_true", help="Save results to JSON")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Check model exists
    if not Path(args.model).exists():
        print(f"❌ Error: Model not found: {args.model}")
        return 1
    
    # Create tester
    tester = RecoveryTester(args.model, stage=args.stage, verbose=args.verbose)
    
    try:
        # Run requested test
        if args.test == "full":
            results = tester.run_full_test_suite(save_results=args.save)
        elif args.test == "hover":
            results = tester.test_hover_stability(duration=args.duration)
        elif args.test in ["flip", "spin", "tumble", "collision"]:
            results = tester.test_impact_scenario(args.test, n_trials=args.trials)
        elif args.test == "continuous":
            results = tester.test_continuous_disturbances(duration=args.duration)
        
        return 0
    
    except KeyboardInterrupt:
        print("\n⚠️  Test interrupted by user")
        return 1
    
    finally:
        tester.close()


if __name__ == "__main__":
    import sys
    sys.exit(main())