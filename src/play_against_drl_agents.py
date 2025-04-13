import subprocess
import os
import sys
import argparse
# Import all the supported play mode here.
# Add new play mode to src.play package and import here.
import play_game.play_game_pvai  # Play against the trained agent. Now only support fight against Ryu.
import play_game.play_game_pvp  # Play against another player(Need 2 people to play). Now only support fight against Ryu.
import play_game.play_game_ai_vs_ai
import play_game.play_game_pve

from stable_baselines3 import PPO
from models.aux_obj_ppo import AuxObjPPO

# Select the play mode here. Must be one kof ["PvAI", "PvP", "AIvAI", "PvE"]
PLAY_MODE = "AIvAI"
# Set if the model is multi-input model.
MULTI_INPUT = True
# Modify the game state(.state file name) here.
# GAME_STATE = "PvP.RyuVsRyu"
# GAME_STATE = "PvP.KenVsKen"
# GAME_STATE = "PvP.ChunliVsChunli"
# GAME_STATE = "PvP.ZangiefVsZangief"
# GAME_STATE = "PvP.DhalsimVsDhalsim"
# GAME_STATE = "PvP.GuileVsGuile"
# GAME_STATE = "PvP.BlankaVsBlanka"
# GAME_STATE = "PvP.EHondaVsEHonda"
# GAME_STATE = "PvP.BalrogVsBalrog"
# GAME_STATE = "PvP.VegaVsVega"
# GAME_STATE = "PvP.SagatVsSagat"
GAME_STATE = "PvP.BisonVsBison"
# GAME_STATE = "Champion.Level11.RyuVsSagat"
# Modify model's path and name here.
# ENEMY_MODEL_DIR = r'agent_models/best_eval_models'
# ENEMY_MODEL_DIR = r'agent_models/agents_archive'
ENEMY_MODEL_NAME = r'rushdown_type/1'
# ENEMY_MODEL_NAME = r'newbie/1'
# ENEMY_MODEL_NAME = r'best_model'
# This is for 'ai_vs_ai' mode. Modify the models' path and name for both agent and enemy here.
ENEMY_MODEL_DIR = r'agent_models/AIvAI'
ENEMY_MODEL_NAME_PLAYER1 = '1'
ENEMY_MODEL_NAME_PLAYER2 = "2"
# ENEMY_MODEL_NAME = ENEMY_MODEL_NAME_PLAYER1

# Set reward_kwargs for evaluation.
REWARD_KWARGS = {
    'reward_scale': 0.001,
    'raw_reward_coef': 1.0, # How much HP change is rewarded, only happens during the fighting (not round over)
    'special_move_reward': 0.0, # Reward for using special moves
    'special_move_bonus': 1.0, # Bonus for dealing damage with special moves. The raw reward will be multiplied by this value, so set it to 1.0 if you don't want to use this feature.
    'projectile_reward': 1.0, # Reward for using projectiles
    'projectile_bonus': 3.0, # Bonus for dealing damage with projectiles. The raw reward will be multiplied by this value, so set it to 1.0 if you don't want to use this feature.
    'distance_reward': 0.00, # Set it to positive value to encourage the agent to stay far from the enemy, negative value to encourage the agent to stay close to the enemy
    'distance_bonus': 2.0, # Bonus for dealing damage with projectiles. 0.0 means no bonus, positive/negative value means reward being far/close to the enemy
    'cost_coef': 1.0, # Ratio between the cost and the reward
    'special_move_cost': 2.0,
    'regular_attack_cost': 1.0,
    'jump_cost': 3.0,
    'vulnerable_frame_cost':0.05,
}


if __name__ == "__main__":
    # Extract model's class name from model's name and set the model class.
    if "%" in ENEMY_MODEL_NAME:
        model_class_name = ENEMY_MODEL_NAME.split("%")[1]
        if model_class_name == "AuxObjPPO":
            model_class = AuxObjPPO
        elif model_class_name == "PPO":
            model_class = PPO
        else:
            raise ValueError(f"Invalid model class name: {model_class_name}. Must be one of ['AuxObjPPO', 'PPO']")
    else:
        model_class = PPO
    
    if PLAY_MODE == "PvAI":
        play_game.play_game_pvai.main(
            model_class = model_class,
            multi_input= MULTI_INPUT,
            game_state = GAME_STATE,
            model_dir = ENEMY_MODEL_DIR,
            model_name = ENEMY_MODEL_NAME,
            character_flip_rate = 0.0,
            reward_kwargs = REWARD_KWARGS,
        )
    elif PLAY_MODE == "PvP":
        play_game.play_game_pvp.main(
            game_state = GAME_STATE,
            rendering_interval = 0.015,
            reward_kwargs = REWARD_KWARGS,
        )
    elif PLAY_MODE == "AIvAI":
        play_game.play_game_ai_vs_ai.main(
            model_class = model_class,
            multi_input= MULTI_INPUT,
            game_state = GAME_STATE,
            model_dir_player1 = ENEMY_MODEL_DIR,
            model_name_player1 = ENEMY_MODEL_NAME_PLAYER1,
            model_dir_player2 = ENEMY_MODEL_DIR,
            model_name_player2 = ENEMY_MODEL_NAME_PLAYER2,
            reward_kwargs = REWARD_KWARGS,
        )
    elif PLAY_MODE == "PvE":
        play_game.play_game_pve.main(
            game_state = GAME_STATE,
            rendering_interval = 0.015,
            reward_kwargs = REWARD_KWARGS,
        )
    else:
        raise ValueError(f"Invalid play mode: {PLAY_MODE}. Must be one of ['PvAI', 'PvP', 'AIvAI', 'PvE']")
    

