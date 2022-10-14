# VAE-Model-based-RL-for-navigation
Model-based Reinforcement Learning algorithm for navigation using VAE and DDPG.
This is a model-based RL algorithm using a VAE with an angular latent representation of the environment to perform a navigation task for autonomous vehicles.

The algorithm is inspired by model-based RL neuroscience experiments, where mice can pre-acquire a latent representation of a the map of a maze, that will accelerate the learning process when placed again in the maze. In general, for a naigation task an angular latent representation (compass) can be pre-aquired by having a look to the map. In our environment the vehcile should arrive at a target by reducing Euclidean distance and alignment with respect to the target.

<img src="https://github.com/AngelCanelo/VAE-Model-based-RL-for-navigation/blob/main/images/Fig_1_Navigation_Environments.png" width=50% height=50%>

In our algorithm the VAE with angular latent representation is combined with a policy network updated using Deep Deterministic Policy Gradient (DDPG).

<img src="https://github.com/AngelCanelo/VAE-Model-based-RL-for-navigation/blob/main/images/Fig_2_workflow_diagram.png" width=50% height=50%>

Our algorithm outperformed state-of-the-art algorithms such as Twin Delayed Deep Deterministic policy gradient (TD3), and DDPG in performing this navigation task.

<img src="https://github.com/AngelCanelo/VAE-Model-based-RL-for-navigation/blob/main/images/Fig_3_results.png" width=50% height=50%>

- **images** folder contains the enviroments, workflow diagram, and the results.
- **models** folder contains the different files for training the models, and the file to test them.
- **weights** folder contains the weight files for each network used by the test file.
