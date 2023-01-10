# rl_morphing_airfoil
Jupyter notebook code and data for running my morphing airfoil DRL project

These folders are included:

data: flex sensor signal, MFC voltage, and true deflection measurements used to train the state inference and dynamics models

env_models: LSTM dynamics model used to simulate the MFC morphing airfoil environment. 

flex_and_laz_models: the two LSTM models ensembled for state inference

tmp: temporary folder used to hold trained actor and critic models

The jupyter files include:

Time_series_MFC_data_flex_to_laz_github.ipynb: 
This jupyter notebook file trains state inference models using CNN, MSD, and LSTM network structures

Time_series_MFC_dynamics_training_github.ipynb: 
This jupyter notebook file trains dynamics models of the MFC morphing system.

Run_ppo_torch_with_last_10obs_github.ipynb:
This file uses proximal policy optimization to train a policy to control the morphing airfoil

Run_ppo_torch_with_last_10obs_no_overshoot_reward_github.ipynb:
This file uses proximal policy optimization to train a policy to control the morphing airfoil with the added reward penalty to mitigate overshoot.

Additional jupyter notebook files include the label "CONNECTED", and require a serial connection to the hardware exoeriment in order to run. I have included these files to show the structure used for testing on the hardware. These files include testing when using the PID controller as well as the two PPO controllers. Within each CONNECTED run file, one must import the appropriate mfc_env file for the desired state inference (laz, simple, LSTM). Within the CONNECTED ppo run file, one must select whether to use the "RL" or "MO" controller by commenting out the appropriate line of inputs when defining the agent. 

All necessary environment python files are included.

Additionally the "ppo_torch_conv1d.py" file is included to train a PPO controller. This file was based from an open source PPO example by Phil Tabor, found here: https://github.com/philtabor/Youtube-Code-Repository/tree/master/ReinforcementLearning/PolicyGradient/PPO/torch 
