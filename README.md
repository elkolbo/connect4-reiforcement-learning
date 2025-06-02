# Connect 4 Reinforcement Learning Agent

This project allows you to play the classic game "Connect 4" against a trained Reinforcement Learning (RL) agent. The RL agent is developed through training in the RL-agent.py file and can then compete against a human player in the game.

Developer: Lorenz Kolb, Ilaria Di Sabatino and Moritz Huhle

## Files and Directories

- connect4_game.py: This script contains the main logic of the "Connect 4" game. It is used to start the game and play against the RL agent.

- game_functions.py: This file implements the necessary functions for the "Connect 4" game. These functions support the gameplay and are called by connect4_game.py.

- RL-agent.py: This script is responsible for training the RL agent. By running this script, the agent is further trained to play the game effectively. The trained weights of the agent are saved in the "checkpoints"-folder and can be loaded later.

- agent_helper_functions.py: This file contains supporting functions for the RL agent. It provides functions used in training to improve the agent.

- config.py: This file contains configuration variables that can be adjusted for training and game design. Parameters such as learning rate, number of episodes, and other training settings can be set here.

## Installing

With the following configurations the repo will be in a working state:

1. Create a enviroment with python 3.10.13
2. Install the required packages from requirements.txt
3. Install tensorflow from another source e.g. conda


## Instructions

1. Open the config.py file to adjust the settings for training or game design.

2. Run the script RL-agent.py to train the RL agent. The trained weights will be saved.

3. Start the game with connect4_game.py. Here, you can play against the trained RL agent.

4. Enjoy the game!
