# fighting-game-two-tier-agent-system

## Project Structure
The project is organized as follows:

- `agent_models/` 
  - `policy_pool`: The directly of the policy pool that store the historical version of the agent model.

- `data/`
  - `Gym Retro Integration UI/`The interactive emulator UI provided by Gym Retro. 
  - `environment.yml`: Anaconda environment configuration file, used for setting up project dependencies.
  - `copy_all_of_these`: As its name. Usage will be described in "Environment setup".

- `src/`
  - `environments/`: Custom environments for training the agent.
  - `models/`: Neural networks for the agent.
  - `self_play`: Self-play training features, including SelfPlay class, Environment and Agent Manager, and policy pool management methods.
  - `play_game/`: Scripts related to playing the game.
  - `utils/`: Utility functions and helper scripts.
  - Additional Python scripts include:
    - `play.py`: Script to play the game.
    - `train.py`: Main script for training the agent.
    
- `configs/`: The configuration files for experiments. If you want to customize a new experiment, add a new configuration file here.

- `LICENSE`
  - License information for the project.

- `README.md`
  - This file, providing an overview of the project, setup instructions, and usage.

## Environment Setup

### Hardware Requirements

The project can be run on either Windows or Linux. While a high-end machine is not strictly necessary, since reinforcement learning environment simulations are CPU-dependent, high performance CPU is more perferable than other deep learning tasks. Addtionally, the more VRAM, the more sub-process environments you can run. 

### Environment Installation
#### Recommended
We use Anaconda to manage the project's environment. To set up the environment, please use the provided environment.yml file along with the following command to install all necessary dependencies:

0. Install anaconda (for local machine) or miniconda (for HPC). If you are installing on your local machine and haven't install Microsoft Visual C++ Build Tools yet, please use the following link to download and install Microsoft Visual C++ Build Tools: `https://visualstudio.microsoft.com/zh-hant/visual-cpp-build-tools/`

1. go to the directory `<your_github_folder_path>/SFAI/data`, where you can find `environment.yml`. (If you want to customize the conda environment name, rename the "name" parameter (the first line) in `environment.yml` to your `<env_name>`)

2. Run the command: `conda env create -f environment.yml`

3. go to the directory `<your_github_folder_path>/SFAI/src/utils`

4. Run the command: `python print_game_lib_folder.py`

5. Go to `<your_github_folder_path>/SFAI/data`. Copy all the files in  `<your_github_folder_path>/SFAI/data/copy_all_of_these` to the directory outputed by step 4.

After the installation, to activate the conda environment, run the command `conda activate sfai` (or `conda activate <env_name>` if you changed the name in step 1).


#### Singularity & Docker Container Deployment (Future)

In the future, we will provide a Singularity or Docker container to simplify deployment. For now, please proceed with the manual installation process described above.

## Start up
### Play The Game Against The Agent Model
1. Go to the root directory of this project.
2. Run this command for playing agianst an agent model
`python src/play.py`

### Train The Agent Via Self-Play
1. Go to the root directory of this project.
2. Run this command for training an agent via self-play:
`python src/train.py --config <path_to_your_configuration_file>.py`
