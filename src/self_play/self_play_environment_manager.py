# Environment Controller class for managing multiple environments for self-play
# Author: Shouren Wang
# TODO: np.random.choice(Replace = True) 这个需要改。需要优先选择不同的，当没有不同的时，再选择相同的。
# ==============================================================================================================
import os
import sys
import argparse
from typing import Union
from types import ModuleType
import warnings

import numpy as np
import torch as th
import retro

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.torch_layers import NatureCNN
from stable_baselines3.common.base_class import BaseAlgorithm

from environments.sfai_wrapper import SFAIWrapper
from environments.self_play_wrapper import SelfPlayWrapper
from utils.extended_checkpoint_callback import GraphCheckpointCallback
from models.custom_models import CustomCNN, CustomResNet18, CustomResNet50

from self_play.policy_selection import policy_selection_vanilla, policy_selection_random, policy_selection_all

class SelfPlayEnvironmentManager:
    '''
    Environment Manager class for managing multiple environments for self-play
    
    After initialization, the manager will construct a list of SubprocVecEnv environments.
    iterate through the list to access the environments.

    :param game: The name of the game
    :param make_env: The function for creating the environment
    :param num_envs: The number of environments maintained by the manager, setting this value is to control the VRAM usage.
    :param rendering: The flag for rendering the environment
    :param sticky_action_mode: The flag for sticky action mode
    :param stickiness: The stickiness of the sticky action mode
    :param pvp_ratio: The ratio of PVP environments to create
    :param enemy_policy_selection: The method for selecting enemy policies
    :param model_classes: The dictionary of enemy model classes                                              
    :param policy_pool_dir: The directory containing the policy pool
    :param game_states_pve: The list of PVE game states
    :param game_states_pvp: The list of PVP game states
    '''   
    def __init__(
            self, 
            game: str = None,
            make_env: callable = None,
            rendering: bool = None,
            num_envs: int = None,
            sticky_action_mode: bool = None,
            stickiness: float = None,
            pvp_ratio: float = None,
            enemy_policy_selection: Union[int, str] = None,
            model_classes: dict = None,
            policy_pool_dir: str = None,
            game_states_pve: list = None,
            game_states_pvp: list = None,
        ):
        # Initialize the environment controller with the given settings
        if game == None:
            raise ValueError("The game should be provided.")
        self.game = game

        if num_envs == None:
            raise ValueError("The number of environments should be provided.")
        elif num_envs <= 0:
            raise ValueError("The number of environments should be positive.")
        self.num_envs = num_envs

        if make_env == None:
            raise ValueError("The make_env function should be provided.")
        self.make_env = make_env

        if rendering == None:
            raise ValueError("The rendering flag should be provided.")
        self.rendering = rendering

        if sticky_action_mode == None:
            raise ValueError("The sticky action mode should be provided.")
        self.sticky_action_mode = sticky_action_mode

        if stickiness == None:
            raise ValueError("The stickiness should be provided.")
        elif stickiness < 0 or stickiness > 1:
            raise ValueError("The stickiness should be between 0 and 1.")
        self.stickiness = stickiness

        if pvp_ratio == None:
            raise ValueError("The PVP ratio should be provided.")
        elif pvp_ratio < 0 or pvp_ratio > 1:
            raise ValueError("The PVP ratio should be between 0 and 1.")
        self.pvp_ratio = pvp_ratio

        if enemy_policy_selection == None:
            raise ValueError("The enemy policy selection method should be provided.")
        if isinstance(enemy_policy_selection, int):
            if enemy_policy_selection < 1:
                raise ValueError("The number of top enemies to select should be larger than 0.")
        elif isinstance(enemy_policy_selection, str):
            if enemy_policy_selection not in ["All", "Random"]:
                raise ValueError("The enemy policy selection method should be 'All' or 'Random'.")
        else:
            raise ValueError("The enemy policy selection should be an integer or a string.")
        self.enemy_policy_selection = enemy_policy_selection

        if model_classes == None:
            raise ValueError("The model classes should be provided.")
        self.model_classes = model_classes

        if policy_pool_dir == None:
            raise ValueError("The policy pool directory should be provided.")
        self.policy_pool_dir = policy_pool_dir

        if game_states_pve == None:
            raise ValueError("The PVE game states should be provided.")
        self.game_states_pve = game_states_pve

        if game_states_pvp == None:
            raise ValueError("The PVP game states should be provided.")
        self.game_states_pvp = game_states_pvp


        # Initialize the environment
        self.num_envs_pve = 0
        self.num_envs_pvp = 0
        self.raw_envs_list = None
        self.num_subproc_vecenvs = 0
        self.envs = None
        self.construct_environments()

        # Print initialization information
        print("----------------- Environment Controller -----------------")
        print("Successfully initialized the environment controller.")
        print("Number of PVE environments: {}".format(self.num_envs_pve))
        print("Number of PVP environments: {}".format(self.num_envs_pvp))
        print("----------------------------------------------------------")

    def construct_environments(
            self,
            env_idx: int = 0,    
        ):
        '''
        Construct the environments based on the settings.
        Due to the VRAM limitation, the number of environments might be less than the number of the tasks.
        To solve this, this function returns a list of SubproVecEnv environments.
        For each SubprocVecEnv environment, it contains num_envs environments.
        Training should be conducted on each SubprocVecEnv environment.

        :param env_idx: The index of the SubprocVecEnv in self.raw_envs_list that used to construct the SubprocVecEnv environments. Default value is 0.


        :return: A list of SubprocVecEnv environments
        '''
        # Allocate the environments
        # Do not use int() here because it will round down. 
        # For example, when num_envs=4, pvp_ratio=0.7, int(4*0.7)=2, round(4*0.7)=3. 
        # 3 is definitely better than 2 in this case.
        # Check if the enemy policy selection method is an integer
        if isinstance(self.enemy_policy_selection, int):
            # NOTE: stable-baselines3's model.set_env() method DOES NOT support changing the number of environments, to guarantee that, the total number of environments must be multiples of num_envs
            # The following calculation method is to ensure that the total number of environments is multiples of num_envs
            # The number of top enemies to select
            num_pvp_enemy = self.enemy_policy_selection
            # Calculate the number or total environments based on num_pvp_enemy and pvp_ratio
            if self.pvp_ratio != 0:
                num_total_envs = round(num_pvp_enemy / self.pvp_ratio)
            else:
                # This is only for pure PVE training
                num_total_envs = self.num_envs
            # If num_total_envs is not multiples of num_envs, round up it to the nearest multiple of num_envs
            if num_total_envs % self.num_envs != 0:
                num_total_envs = (num_total_envs // self.num_envs + 1) * self.num_envs
            # Assign the number of PvP environments and PVE environments by num_total_envs and pvp_ratio
            self.num_envs_pvp = round(num_total_envs * self.pvp_ratio)
            self.num_envs_pve = num_total_envs - self.num_envs_pvp
        elif isinstance(self.enemy_policy_selection, str) and self.enemy_policy_selection == "Random":
            self.num_envs_pvp = round(self.num_envs * self.pvp_ratio) 
            self.num_envs_pve = self.num_envs - self.num_envs_pvp
        elif isinstance(self.enemy_policy_selection, str) and self.enemy_policy_selection == "All":
            # NOTE: stable-baselines3's model.set_env() method DOES NOT support changing the number of environments, to guarantee that, the total number of environments must be multiples of num_envs
            # The following calculation method is to ensure that the total number of environments is multiples of num_envs
            # The number of top enemies to select
            # Read the number of models in the policy pool
            num_envs_pvp = len(os.listdir(self.policy_pool_dir))
            # Calculate the number or total environments based on num_pvp_enemy and pvp_ratio
            if self.pvp_ratio != 0:
                num_total_envs = round(num_envs_pvp / self.pvp_ratio)
            else:
                # This is only for pure PVE training
                num_total_envs = self.num_envs
            # If num_total_envs is not multiples of num_envs, round up it to the nearest multiple of num_envs
            if num_total_envs % self.num_envs != 0:
                num_total_envs = (num_total_envs // self.num_envs + 1) * self.num_envs
            # Assign the number of PvP environments and PVE environments by num_total_envs and pvp_ratio
            self.num_envs_pvp = round(num_total_envs * self.pvp_ratio)
            self.num_envs_pve = num_total_envs - self.num_envs_pvp
            
        else:
            raise ValueError("The enemy policy selection method should be an integer or a string(Currently only support Random and All).")
        # Construct the environments
        # PvP environments:
        # Randomly select num_envs_pvp game states from the PvP game states
        # Be aware of that, the number of PvP games states may not equal to the number of pvp environments
        # If less, repeat the game states; if more, randomly select the game states
        pvp_game_states = np.random.choice(self.game_states_pvp, self.num_envs_pvp, replace=True)
        # Select the policy for PvP environments
        selected_enemy_models, selected_enemy_model_classes = self.select_enemy_policy()
        # Check if the enemy model classes are the supported model classes
        for model_class in selected_enemy_model_classes:
            if model_class not in self.model_classes:
                raise ValueError("The model class {} is not supported.".format(model_class))
        # Construct the enemy models paths
        enemy_models_paths = [os.path.join(self.policy_pool_dir, model_name) for model_name in selected_enemy_models]
        # Extract the enemy model classes
        enemy_model_classes = [self.model_classes[model_class] for model_class in selected_enemy_model_classes]
        # Construct the PvP environments
        # Be aware of that, the number of enemy models may not equal to the number of pvp environments
        # If less, repeat the enemy models; if more, select the top a few enemy models because policy selection is based on the win rate
        pvp_envs =[]
        for i in range(self.num_envs_pvp):
            pvp_env = self.make_env(
                game = self.game,
                state = pvp_game_states[i % len(pvp_game_states)],
                rendering = self.rendering,
                enemy_model_path = enemy_models_paths[i % len(enemy_models_paths)],
                enemy_model_class = enemy_model_classes[i % len(enemy_model_classes)],
                seed = i,
                sticky_action_mode = self.sticky_action_mode,
                stickiness = self.stickiness,
                players = 2
            )
            pvp_envs.append(pvp_env)

        # PVE environments:
        pve_envs = []
        if self.num_envs_pve > 0:
            # Randomly select num_envs_pve game states from the PVE game states
            pve_game_states = np.random.choice(self.game_states_pve, self.num_envs_pve, replace=True)
            for i in range(self.num_envs_pve):
                pve_env = self.make_env(
                    game = self.game,
                    state = pve_game_states[i % len(pve_game_states)],
                    rendering = self.rendering,
                    seed = i,
                    sticky_action_mode = self.sticky_action_mode,
                    stickiness = self.stickiness,
                    players = 1
                )
                pve_envs.append(pve_env)

        # Combine the PvP and PVE environments    
        raw_envs = pvp_envs + pve_envs
        # Shuffle the environments
        np.random.shuffle(raw_envs)
        # Split the raw_envs in to several parts, each part contains num_envs environments
        self.raw_envs_list = [raw_envs[i:i+self.num_envs] for i in range(0, len(raw_envs), self.num_envs)]
        self.num_subproc_vecenvs = len(self.raw_envs_list)

        # Construct the SubprocVecEnv environments
        self.update_environments(env_idx=env_idx)
    
    def update_environments(
            self,
            env_idx: int = None,
        ):
        '''
        Update the environments in each iteration of the self-play training process.

        :param env_idx: The index of the SubprocVecEnv in self.raw_envs_list. Start from 0.

        The update includes:
        1. Update the enemy policy models
        2. Reselect the game states

        The update is conducted in the following steps:
        1. Close the existing environments
        2. Reconstruct the environments using self.construct_environments()
        '''
        # Check if the env_idx is valid
        # Constrcut the SubprocVecEnv environments according to 'env_idx'
        if env_idx == None:
            raise ValueError("The 'env_idx' should be provided.")
        if env_idx > len(self.raw_envs_list):
            # If env_idx is larger than the number of SubprocVecEnv environments, set it to the length of the list, raise a warning as well
            env_idx = len(self.raw_envs_list)
            warnings.warn("The 'env_idx' is larger than the number of SubprocVecEnv environments, set it to the length of the list.", UserWarning)
        elif env_idx < 0:
            raise ValueError("The 'env_idx' should be a non-negative integer.")
        
        # Check if self.raw_envs_list is None or empty
        if self.raw_envs_list == None:
            raise ValueError("The raw_envs_list is None.")
        elif len(self.raw_envs_list) == 0:
            raise ValueError("The raw_envs_list is empty.")

        # Close the existing environments
        self.close_environments()

        # Construct the new environments
        self.envs = SubprocVecEnv(self.raw_envs_list[env_idx])
        
        

    def update_settings(
            self,
            make_env: callable = None,
            rendering: bool = None,
            num_envs: int = None,
            sticky_action_mode: bool = None,
            stickiness: float = None,
            pvp_ratio: float = None,
            enemy_policy_selection: Union[int, str] = None,
            model_classes: dict = None,
            policy_pool_dir: str = None,
            game_states_pve: list = None,
            game_states_pvp: list = None,
        ):
        # Update the settings
        # Just update the parameters without changing the environments
        # For the arugments above, if no value passed, keep the original value
        if make_env != None:
            self.make_env = make_env
        if rendering != None:
            self.rendering = rendering
        if num_envs != None:
            self.num_envs = num_envs
        if sticky_action_mode != None:
            self.sticky_action_mode = sticky_action_mode
        if stickiness != None:
            self.stickiness = stickiness
        if pvp_ratio != None:
            self.pvp_ratio = pvp_ratio
        if enemy_policy_selection != None:
            self.enemy_policy_selection = enemy_policy_selection
        if model_classes != None:
            self.model_classes = model_classes
        if policy_pool_dir != None:
            self.policy_pool_dir = policy_pool_dir
        if game_states_pve != None:
            self.game_states_pve = game_states_pve
        if game_states_pvp != None:
            self.game_states_pvp = game_states_pvp

    # TODO: 代码运行无误后，删除下方的代码               
    # def reconstruct_environments(
    #         self,
    #         make_env: callable = None,
    #         rendering: bool = None,
    #         num_envs: int = None,
    #         sticky_action_mode: bool = None,
    #         stickiness: float = None,
    #         pvp_ratio: float = None,
    #         enemy_policy_selection: Union[int, str] = None,
    #         model_classes: dict = None,
    #         policy_pool_dir: str = None,
    #         game_states_pve: list = None,
    #         game_states_pvp: list = None,
    #     ):
    #     '''
    #     Completely reconstruct the environments based on the new settings.
    #     '''
    #     # Update the environments
    #     # Close the existing environments
    #     self.close_environments()

    #     # Update the settings
    #     self.update_settings(
    #         make_env = make_env,
    #         rendering = rendering,
    #         num_envs = num_envs,
    #         sticky_action_mode = sticky_action_mode,
    #         stickiness = stickiness,
    #         pvp_ratio = pvp_ratio,
    #         enemy_policy_selection = enemy_policy_selection,
    #         model_classes = model_classes,
    #         policy_pool_dir = policy_pool_dir,
    #         game_states_pve = game_states_pve,
    #         game_states_pvp = game_states_pvp,
    #     )
    #     # Reconstruct the environments
    #     self.envs = self.construct_environments()
        

    def select_enemy_policy(self):
        # Select the enemy policy models
        if isinstance(self.enemy_policy_selection, int):
            # The largest number for the top N enemies to select is the number of models in the policy pool
            # Check if the number of top enemies to select is larger than the number of models in the policy pool
            # if so, select all the models in the pool instead
            if self.enemy_policy_selection > len(self.policy_pool_dir):
                top_n_enemy = len(self.policy_pool_dir)
            else:
                top_n_enemy = self.enemy_policy_selection
            enemy_models, _, enemy_model_classes = policy_selection_vanilla(policy_pool_dir=self.policy_pool_dir, top_n=top_n_enemy)
            
        elif isinstance(self.enemy_policy_selection, str):
            if self.enemy_policy_selection == "All":
                # Select all the models in the policy pool
                enemy_models, _, enemy_model_classes = policy_selection_all(policy_pool_dir=self.policy_pool_dir)
            elif self.enemy_policy_selection == "Random":
                # Randomly select num_envs_pvp enemy models from the policy pool
                enemy_models, _, enemy_model_classes = policy_selection_random(policy_pool_dir=self.policy_pool_dir, top_n=self.num_envs_pvp)
            else:
                raise ValueError("The enemy policy selection method should be 'All' or 'Random'.")
        else:
            raise ValueError("The enemy policy selection should be an integer or a string.")
        
        return enemy_models, enemy_model_classes
        
    def close_environments(self):
        if self.envs != None:
            if self.rendering:
                self.envs.env_method('render', close=True)
            self.envs.close()
        else:
            print("No environments running. Skip closing.")
        # for env in self.envs:
        #     env.close()