import math
import time
import collections
import random 

import gym
import numpy as np
from stable_baselines3.common.base_class import BaseAlgorithm


class EvaluationWrapper(gym.Wrapper):
    '''
    Wrapper for evaluating a model. Wrap environment with thsi wrapper after warpping with SFAIWrapper.  
    TODO: FINISH ENJOYABILITY SCORE  
    '''
    def __init__(
            self,
            env,
            score_enjoyability: callable = None,
            score_enjoyability_kwargs: dict = None,
    ):
        '''
        Args:
            env (gym.Env): The environment to wrap.
            score_enjoyability (callable): A function that takes in a reward and returns a score for the reward. If None, the default function will be used.
        '''
        super().__init__(env)
        self.env = env

        # initialize the statistics
        self.steps = 0
        self.total_reward = 0
        self.total_regular_attacks = 0
        self.total_special_moves = 0
        self.total_distance = 0
        self.total_air_time = 0
        self.total_projectiles = 0
        self.total_jumps = 0
    
    def reset(self):
        '''
        Reset the environment.
        '''
        self.steps = 0
        self.total_reward = 0
        self.total_regular_attacks = 0
        self.total_special_moves = 0
        self.total_distance = 0
        self.total_air_time = 0
        self.total_projectiles = 0
        self.total_jumps = 0
        return self.env.reset()
    

    def step(self, action):
        '''
        Take a step in the environment.
        '''
        obs, reward, done, info = self.env.step(action)
        
        # Get the behavior of the agent
        behavior = self._check_behavior(info)
        # Update the statistics
        self.steps += 1
        self.total_reward += reward
        self.total_regular_attacks += behavior.get('Regular Attack', 0)
        self.total_special_moves += behavior.get('Special Move', 0)
        self.total_distance += behavior.get('distance', 0)
        self.total_air_time += behavior.get('In Air', 0)
        self.total_projectiles += behavior.get('Projectile', 0)
        self.total_jumps += behavior.get('Jump', 0)
        
        # Calculate the enjoyability and update the behavior data when the episode is done
        if done:
            average_distance = self.total_distance / self.steps
            average_air_time = self.total_air_time / self.steps
            # Calculate the enjoyability score
            enjoyability_score = self.default_score_enjoyability()
            
            # Check if agent wins or loses
            win = 0
            if info['agent_hp'] > info['enemy_hp']:
                win = 1
            elif info['agent_hp'] == info['enemy_hp']:
                win = 0.5
            else:
                win = 0


            # Update info
            behavior_metrics = {}
            behavior_metrics['enjoyability_score'] = enjoyability_score
            behavior_metrics['episode_reward'] = self.total_reward
            behavior_metrics['average_distance'] = average_distance
            behavior_metrics['average_air_time'] = average_air_time
            behavior_metrics['regular_attacks'] = self.total_regular_attacks
            behavior_metrics['jumps'] = self.total_jumps
            behavior_metrics['special_moves'] = self.total_special_moves
            behavior_metrics['projectiles'] = self.total_projectiles
            behavior_metrics['win'] = win

            info['behavior_metrics'] = behavior_metrics

        # Return the observation, reward, done, and info        
        return obs, reward, done, info


    def _check_behavior(self, info):
        '''
        This function is used to detect the behavior of the agent based on the change of information from the environment.
        The behavior includes the following:
        - Special moves: The agent performs special moves.
        - Projectiles: The agent performs projectiles.
        - Jumps: The agent performs jumps.
        - Regular attacks: The agent performs regular attacks.
        - In air: The agent is in the air.

        Parameters:
        - info: The information from the environment. Note the it must contain the 'info_sequence_buffer' key (for the 'info_sequence_buffer' please see sfai_wrapper.py for more details).
        '''
        # Check if the 'info_sequence_buffer' key is in the info dictionary, if not, raise an error
        if 'info_sequence_buffer' not in info:
            raise ValueError("The 'info' provided does not contain the 'info_sequence_buffer' dictionary. Please check the 'info' returned in  your environment wrapper's step() function.")
        

        # Retrieve the informations
        agent_status = info['agent_status']
        prev_info = info['info_sequence_buffer'][-2]
        prev_agent_status = prev_info.get('agent_status', 0)

        # Status Check
        # Special Move triggerring status
        status_special_move = False
        if agent_status == 524 and prev_agent_status != 524:
            status_special_move = True

        # Projectile triggerring status
        status_projectile = False
        if info['agent_projectile_status'] == 34048 and prev_info['agent_projectile_status'] != 34048:
            status_projectile = True
            
        # Regular attack status(regular attacks are light~heavy punches, light~heavy kicks)
        status_regular_attack = False
        if agent_status == 522 and prev_agent_status != 522:
            status_regular_attack = True

        # Jumping status
        status_jump = False
        if agent_status == 516 and prev_agent_status != 516:
            status_jump = True

        # In air status
        status_in_air = False
        if agent_status == 516:
            status_in_air = True

        # Distance
        distance = abs(info.get('agent_x', 0) - info.get('enemy_x', 0))


        behavior = {
            "Special Move": status_special_move,
            "Projectile": status_projectile,
            "Regular Attack": status_regular_attack,
            "Jump": status_jump,
            "In Air": status_in_air,
            "distance": distance
        }

        return behavior

    def default_score_enjoyability(
            self, 
            score_enjoyability_kwargs: dict = {}
        ):
        '''
        TODO: FINISH THIS PART
        Default function for scoring enjoyability.
        In this function, we calculate the enjoyability based on:
        1. Average reward——it should not be too high or too low, too high means too strong and too low means too weak to players.
        2. Average regular attacks——it should not be too high or too low, too high means key spamming and too low means no attacks.
        3. Average special moves——usually, it's the higher the better
        4. Average distance——the more extreme the better, very low means aggressive and very high means defensive.
        5. Average air time——Not yet sure how it affects the enjoyability, just pass it in for now.
        6. Average projectiles——Not yet sure how it affects the enjoyability, usually, the higher the better.

        Args:
            score_enjoyability_kwargs (dict): A dictionary of keyword arguments for the function.

        Returns:
            float: The sigmoid score for the reward normalized to [0, 1].
        '''
        # Get the kwargs
        difficulty_coef = score_enjoyability_kwargs.get('difficulty_coef', 1.0)
        regular_attack_coef = score_enjoyability_kwargs.get('regular_attack_coef', 0.0)
        special_move_coef = score_enjoyability_kwargs.get('special_move_coef', 0.0)
        distance_coef = score_enjoyability_kwargs.get('distance_coef', 0.0)
        air_time_coef = score_enjoyability_kwargs.get('air_time_coef', 0.0)
        projectile_coef = score_enjoyability_kwargs.get('projectile_coef', 0.0)

        enjoyability_score = 0.0

        difficulty_score = self.total_reward * difficulty_coef
        special_move_score = self.total_special_moves * special_move_coef

        # TODO: FInish this part 
        
        

        return enjoyability_score
