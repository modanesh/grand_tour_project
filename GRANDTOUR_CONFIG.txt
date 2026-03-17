
======================================================================
Grand Tour Dataset – Observation & Action Space
======================================================================
  Observation dims : 48 (with prev_actions)
  Action dims      : 12
  Joint order      : ['LF_HAA', 'LF_HFE', 'LF_KFE', 'RF_HAA', 'RF_HFE', 'RF_KFE', 'LH_HAA', 'LH_HFE', 'LH_KFE', 'RH_HAA', 'RH_HFE', 'RH_KFE']

======================================================================
Grand Tour default joint angles (centering reference for joint_pos / actions):
======================================================================
  Joint        Default (rad)   Default (deg)
  ----------  --------------  --------------
  LF_HAA             -0.3818          -21.88
  LF_HFE              0.8445           48.39
  LF_KFE             -1.3450          -77.06
  RF_HAA              0.4191           24.01
  RF_HFE              0.7747           44.39
  RF_KFE             -1.3316          -76.30
  LH_HAA             -0.3890          -22.29
  LH_HFE             -0.5935          -34.01
  LH_KFE              1.2910           73.97
  RH_HAA              0.4031           23.10
  RH_HFE             -0.7371          -42.23
  RH_KFE              1.3671           78.33

======================================================================

  Observation space:
       Dim  Term                       Sub-label     Scale / note
  --------  -------------------------  ------------  ------------------------------
         0  base_lin_vel               x             × 2.0 (body frame)
         1  base_lin_vel               y             
         2  base_lin_vel               z             
         3  base_ang_vel               p             × 0.25 (body frame)
         4  base_ang_vel               q             
         5  base_ang_vel               r             
         6  projected_gravity          x             no scaling
         7  projected_gravity          y             
         8  projected_gravity          z             
         9  velocity_commands          vx            × [2.0, 2.0, 0.25]
        10  velocity_commands          vy            
        11  velocity_commands          yaw_rate      
        12  joint_pos                  LF_HAA        (joint_pos − GT_default) × 1.0
        13  joint_pos                  LF_HFE        
        14  joint_pos                  LF_KFE        
        15  joint_pos                  RF_HAA        
        16  joint_pos                  RF_HFE        
        17  joint_pos                  RF_KFE        
        18  joint_pos                  LH_HAA        
        19  joint_pos                  LH_HFE        
        20  joint_pos                  LH_KFE        
        21  joint_pos                  RH_HAA        
        22  joint_pos                  RH_HFE        
        23  joint_pos                  RH_KFE        
        24  joint_vel                  LF_HAA        × 0.05
        25  joint_vel                  LF_HFE        
        26  joint_vel                  LF_KFE        
        27  joint_vel                  RF_HAA        
        28  joint_vel                  RF_HFE        
        29  joint_vel                  RF_KFE        
        30  joint_vel                  LH_HAA        
        31  joint_vel                  LH_HFE        
        32  joint_vel                  LH_KFE        
        33  joint_vel                  RH_HAA        
        34  joint_vel                  RH_HFE        
        35  joint_vel                  RH_KFE        
        36  prev_actions               LF_HAA        (target − GT_default) / 0.5
        37  prev_actions               LF_HFE        
        38  prev_actions               LF_KFE        
        39  prev_actions               RF_HAA        
        40  prev_actions               RF_HFE        
        41  prev_actions               RF_KFE        
        42  prev_actions               LH_HAA        
        43  prev_actions               LH_HFE        
        44  prev_actions               LH_KFE        
        45  prev_actions               RH_HAA        
        46  prev_actions               RH_HFE        
        47  prev_actions               RH_KFE        

======================================================================

  Action space:
       Dim  Term                       Sub-label     Scale / note
  --------  -------------------------  ------------  ------------------------------
         0  joint_position_target      LF_HAA        (target − GT_default) / 0.5
         1  joint_position_target      LF_HFE        
         2  joint_position_target      LF_KFE        
         3  joint_position_target      RF_HAA        
         4  joint_position_target      RF_HFE        
         5  joint_position_target      RF_KFE        
         6  joint_position_target      LH_HAA        
         7  joint_position_target      LH_HFE        
         8  joint_position_target      LH_KFE        
         9  joint_position_target      RH_HAA        
        10  joint_position_target      RH_HFE        
        11  joint_position_target      RH_KFE        

======================================================================
