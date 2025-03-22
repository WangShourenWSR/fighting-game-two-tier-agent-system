# fighting-game-two-tier-agent-system

## Project Structure
The project is organized as follows:

- `agent_models/`
  - `policy_pool`: The directly of the policy pool that store the historical version of the agent model.

- `data/`
  - Stores various data files required for the project, such as game states and configuration files.
  - Subfolders include:
    - `Gym Retro Integration UI/`The interactive emulator UI provided by Gym Retro. 
    - `environment.yml`: Anaconda environment configuration file, used for setting up project dependencies.
    - `copy_all_of_these`: As its name. Usage will be described in "Environment setup".

- `src/`
  - Contains source code for the project.
  - Subfolders include:
    - `environments/`: Custom environments for training the agent.
    - `models/`: Neural network models used for the agent.
    - `self_play`: Self-play training features, including SelfPlay class, Environment and Agent Manager, and policy pool management methods.
    - `play_game/`: Scripts related to playing the game.
    - `utils/`: Utility functions and helper scripts.
    - `old_codes/`: Old version codes, won't be used and will be removed in the future.
  - Additional Python scripts include:
    - `play.py`: Script to play the game.
    - `train.py`: Main script for training the agent.
    - `unit_test.py`: Script for unit-testing the code implementation.

- `configs/`: The configuration files for experiments. If you want to customize a new experiment, add a new configuration file here.

- `LICENSE`
  - License information for the project.

- `README.md`
  - This file, providing an overview of the project, setup instructions, and usage.

## Environment Setup

### Hardware Requirements

The project can be run on either Windows or Linux. While a high-end machine is not strictly necessary, having an NVIDIA GPU is highly recommended, as optimizing neural networks without one can be very slow and make debugging more difficult. Additionally, a high-performance CPU is preferable, since reinforcement learning environment simulations are CPU-dependent.

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

#### Alternative
If the previous installation doesn't work, please go the teh directory `<your_github_folder_path>/SFAI/old_codes/dependency_old/`. Try the installation method in `dependencies_installation_old_version.txt`.


### Singularity & Docker Container Deployment (Future)

In the future, we will provide a Singularity or Docker container to simplify deployment. For now, please proceed with the manual installation process described above.

## Start up
### Play The Game Against The Agent Model
1. Go to the root directory of this project.
2. Run this command for playing agianst an agent model
`python src/play.py`

### Train The Agent Via Self-Play
1. Go to the root directory of this project.
2. Run this command for training an agent via self-play:
`python src/train_self_play.py --task_name sp_default_projectile_reward --with_aux_obj True --features_extractor_class CustomResNet18 --multi_input_policy False --sticky_action_mode False --stickiness 0.0 --default_a2c False --log2file False --network_graph_tensorboard False`
