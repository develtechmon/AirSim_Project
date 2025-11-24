# 1. Create project structure
cd ~
mkdir -p phd_ardupilot_deploy
cd phd_ardupilot_deploy

# 2. Create folders
mkdir -p ardupilot_integration/utils
mkdir -p models

# 3. Install Python packages
pip install dronekit pymavlink torch stable-baselines3 numpy

# 4. Copy your trained models
cp /path/to/your/models/hover_policy_best.pth models/
cp /path/to/your/models/hover_disturbance_policy.zip models/
cp /path/to/your/models/hover_disturbance_vecnormalize.pkl models/