================================================================================
📊 TRAINING STATISTICS SUMMARY - Table 5.1
================================================================================

================================================================================
OVERALL PERFORMANCE
================================================================================
  Total Episodes:           1024
  Total Training Time:      14.7 hours
  Final Recovery Rate:      98.0% (last 50)
  Final Avg Reward:         12684.9 (last 10)

================================================================================
CURRICULUM PROGRESSION - Table 5.2
================================================================================

Level Transition          Episode         Time (hours)    Episodes Trained    
--------------------------------------------------------------------------------
Level 0 → 1               51              0.8             50                  
Level 1 → 2               101             1.5             50                  
Level 2 (Final)           101             14.7            924                 

================================================================================
BY CURRICULUM LEVEL - Table 5.4
================================================================================

Level                Episodes        Recovery Rate        Avg Intensity  
--------------------------------------------------------------------------------
EASY (0.7-0.9×)      50              100.0                0.79           
MEDIUM (0.9-1.1×)    50              98.0                 1.01           
HARD (1.1-1.5×)      924             97.7                 1.30           
================================================================================

📄 LaTeX tables saved:
   - analysis_results/table_5_1_training_statistics.tex
   - analysis_results/table_5_2_curriculum_advancement.tex

✅ Statistics saved to: thesis_figures/training_statistics.json

🎉 Analysis complete! Plots saved to: ./thesis_figures/
