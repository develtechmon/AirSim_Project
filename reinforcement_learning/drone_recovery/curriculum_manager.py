import numpy as np

class CurriculumManager:
    """Manages automatic stage progression"""
    
    def __init__(self):
        self.current_stage = 1
        self.stage_performance = {1: [], 2: [], 3: []}
        
        self.thresholds = {
            1: 50.0,
            2: 40.0,
            3: 30.0
        }
        
        self.window_size = 50
        
    def record_episode(self, stage, reward):
        """Record episode performance"""
        if stage in self.stage_performance:
            self.stage_performance[stage].append(reward)
    
    def should_advance(self):
        """Check if we should move to next stage"""
        if self.current_stage >= 3:
            return False
        
        rewards = self.stage_performance[self.current_stage]
        
        if len(rewards) < self.window_size:
            return False
        
        recent_rewards = rewards[-self.window_size:]
        avg_reward = np.mean(recent_rewards)
        threshold = self.thresholds[self.current_stage]
        
        print(f"📊 Stage {self.current_stage} - Avg Reward: {avg_reward:.2f} (Threshold: {threshold})")
        
        if avg_reward >= threshold:
            print(f"✅ Stage {self.current_stage} PASSED! Advancing to Stage {self.current_stage + 1}")
            return True
        
        return False
    
    def advance_stage(self):
        """Move to next stage"""
        if self.current_stage < 3:
            self.current_stage += 1
            return self.current_stage
        return self.current_stage
    
    def get_current_stage(self):
        return self.current_stage