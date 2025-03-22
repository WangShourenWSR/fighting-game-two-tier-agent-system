import math
import time
import collections
import random 

import gym
import numpy as np
from stable_baselines3.common.base_class import BaseAlgorithm



class SelfPlayWrapper(gym.Wrapper):
    def __init__(
            self, 
            env,
            enemy_model: BaseAlgorithm = None,
            enemy_model_path: str = None,
            enemy_model_class: BaseAlgorithm = None,
            ):
        """
        Custom environment wrapper for self-play.
        :param env: (gym.Env) The environment
        :param enemy_model: (BaseAlgorithm) The enemy model. Do not specify this if enemy_model_path is provided.
        :param enemy_model_path: (str) The enemy model path. Do not specify this if enemy_model is provided.
        :param enemy_model_class: (BaseAlgorithm) The enemy model class. Do not specify this if enemy_model is provided.
        """
        super(SelfPlayWrapper, self).__init__(env)
        self.env = env
        # Check if the action space is MultiBinary with 24 dimensions.
        if not isinstance(env.action_space, gym.spaces.MultiBinary) or env.action_space.n != 24:
            raise ValueError("For PvP mode, the action space must be MultiBinary with 24 dimensions. Please double check your input env.")

        # Set the action space to MultiBinary with 12 dimensions, because the agent being trained only controls the first 12 dimensions.
        self.action_space = gym.spaces.MultiBinary(12)
        if enemy_model is not None:
            # Model Loading Method1: Use the provided enemy model
            self.enemy_model = enemy_model
        elif enemy_model_path is not None:
            # Model Loading Method2: Load the enemy model from the provided path
            self.enemy_model = enemy_model_class.load(enemy_model_path)
            # NOTE: Uncomment the following line if you want to print the path of the loaded enemy model
            # print("Enemy model loaded from path: ", enemy_model_path)
        else:
            raise ValueError("Either enemy_model or model_path must be provided.")
        
        # Define the last observation for the enemy model's predict() method
        self.last_observation = None
    

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self.last_observation = obs
        return obs
    
        
    def step(self, action):
        """
        Extend the step method to include the enemy model's action to support self-play.

        """
        # Get the action of player1, which is the agent being trained
        player1_action = action
        # Get the action of player2, which is the enemy model
        if self.last_observation is not None:
            player2_action, _ = self.enemy_model.predict(self.last_observation)
        else:
            raise RuntimeError("The last observation is None. Please call the reset() method to get an observation for enemy model's predict() method first.")

        # Combine the actions of player1 and player2
        action_both = [0]*24
        action_both[:12] = player1_action
        action_both[12:24] = player2_action

        # Perform the combined actions
        observation, reward, done, info = self.env.step(action_both)

        # Update the last observation
        self.last_observation = observation

        return observation, reward, done, info