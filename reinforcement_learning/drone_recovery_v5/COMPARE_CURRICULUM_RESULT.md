  Loading model: ./models/stage3_checkpoints/gated_curriculum_policy.zip
Connected!
Client Ver:1 (Min Req: 1), Server Ver:1 (Min Req: 1)

   Loading VecNormalize: ./models/stage3_checkpoints/gated_curriculum_vecnormalize.pkl
   Model loaded successfully!
   Testing at EASY (0.7-0.9×)...
      Episode 5/20: 5/5 recovered (100%)
      Episode 10/20: 10/10 recovered (100%)
      Episode 15/20: 15/15 recovered (100%)
      Episode 20/20: 20/20 recovered (100%)
   ✅ Complete: 100.0% recovery (20/20)

   Loading model: ./models/stage3_checkpoints/gated_curriculum_policy.zip
Connected!
Client Ver:1 (Min Req: 1), Server Ver:1 (Min Req: 1)

   Loading VecNormalize: ./models/stage3_checkpoints/gated_curriculum_vecnormalize.pkl
   Model loaded successfully!
   Testing at MEDIUM (0.9-1.1×)...
      Episode 5/20: 5/5 recovered (100%)
      Episode 10/20: 10/10 recovered (100%)
      Episode 15/20: 15/15 recovered (100%)
      Episode 20/20: 20/20 recovered (100%)
   ✅ Complete: 100.0% recovery (20/20)

   Loading model: ./models/stage3_checkpoints/gated_curriculum_policy.zip
Connected!
Client Ver:1 (Min Req: 1), Server Ver:1 (Min Req: 1)

   Loading VecNormalize: ./models/stage3_checkpoints/gated_curriculum_vecnormalize.pkl
   Model loaded successfully!
   Testing at HARD (1.1-1.5×)...
      Episode 5/20: 5/5 recovered (100%)
      Episode 10/20: 10/10 recovered (100%)
      Episode 15/20: 15/15 recovered (100%)



      Episode 20/20: 20/20 recovered (100%)
   ✅ Complete: 100.0% recovery (20/20)


================================================================================
📊 COMPREHENSIVE RESULTS - PhD Table 5.3
================================================================================
Cross-Level Performance Evaluation: No Catastrophic Forgetting
--------------------------------------------------------------------------------

Model                     Trained On           Test Intensity  Recovery Rate        Avg Time (steps)
--------------------------------------------------------------------------------
Level 0 (EASY)            EASY (0.7-0.9×)      EASY            100.0% (20/20)       12.1           
                                               MEDIUM          100.0% (20/20)       11.3           
                                               HARD            100.0% (20/20)       15.0           
--------------------------------------------------------------------------------
Level 1 (MEDIUM)          MEDIUM (0.9-1.1×)    EASY            100.0% (20/20)       15.4           
                                               MEDIUM          100.0% (20/20)       15.3           
                                               HARD            100.0% (20/20)       16.8           
--------------------------------------------------------------------------------
Final Model               HARD (1.1-1.5×)      EASY            100.0% (20/20)       9.8            
                                               MEDIUM          100.0% (20/20)       14.8           
                                               HARD            100.0% (20/20)       17.9           
================================================================================

💾 Results saved to: analysis_results/curriculum_comparison_results.json

✅ Analysis Complete!
