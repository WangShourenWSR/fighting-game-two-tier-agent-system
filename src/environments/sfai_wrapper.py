# Copyright 2023 LIN Yi. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import math
import time
import collections
import random 
import copy

import gym
import numpy as np
import warnings

from environments.reward_functions import sfai_reward, naruto_reward


# Custom environment wrapper
class SFAIWrapper(gym.Wrapper):
    def __init__(
            self, 
            env, 
            character_flip_rate = 0,
            reset_round=True, 
            rendering=False, 
            stack_observation=True, # TODO: Make it false if the result is better. Setting it to True is just for compatibility with the original configuration and implementations.
            sticky_action_mode=False, 
            stickiness=0.0, 
            rendering_interval=0.01, 
            num_step_frames=1, 
            reward_function_idx = 0,
            reward_kwargs: dict = {}
        ):
        '''
        Custom environment wrapper for SFAI. This wrapper provides features for training deep reinforcement learning agents in street fighter II.

        :param env: (gym.Env) The environment
        :param character_flip_rate: (float) The probability of flipping the player 1 and player 2's character.
        :param reset_round: (bool) Whether to reset the round after each episode.
        :param rendering: (bool) Whether to render the game.
        :param sticky_action_mode: (bool) Whether to use sticky action mode.
        :param stickiness: (float) The probability of using the previous action in sticky action mode.
        :param rendering_interval: (float) The interval of rendering.
        :param num_step_frames: (int) The number of frames to keep the button pressed.
        :param reward_function_idx: (int) The index of the reward function to use.
        :param reward_kwargs: (dict) The keyword arguments for the reward function.
        '''
        super(SFAIWrapper, self).__init__(env)
        self.env = env

        self.character_flip_rate = character_flip_rate
        # Set character flip
        if random.random() < self.character_flip_rate:
            if self.env.players == 1:
                # Raise a warning if there is "PvP" in the state name.
                if 'PvP' in self.env.statename:
                    warnings.warn("The players parameter of the environment is 1, make sure it is a PvE environment but not PvP. the state file is: " + self.env.statename)
                self.character_flip = False
            else:
                # Flip the characters
                self.character_flip = True
        else:
            self.character_flip = False

        # Stack the observation
        self.stack_observation = stack_observation
        if self.stack_observation:
            # Use a deque to store the last 9 frames
            self.num_frames = 10
            self.frame_stack = collections.deque(maxlen=self.num_frames)

        # Number of frames to keep the button pressed, default is 1.
        self.num_step_frames = num_step_frames

        self.reward_coeff = 1.0

        self.total_timesteps = 0

        self.full_hp = 176
        self.prev_player_health = self.full_hp 
        self.prev_oppont_health = self.full_hp 

        if self.stack_observation:
            self.observation_space = gym.spaces.Box(low=0, high=255, shape=(100, 128, 12), dtype=np.uint8)
        else:
            self.observation_space = gym.spaces.Box(low=0, high=255, shape=(200, 256, 3), dtype=np.uint8)

        self.reset_round = reset_round
        self.rendering = rendering

        # Step Counter:
        self.step_counter = 0

        # Rendering interval
        self.rendering_interval = rendering_interval

        # Set the reward function parameters
        self.reward_function_list = [sfai_reward,  naruto_reward]
        self.reward_function_idx = reward_function_idx
        self.reward_function = self.reward_function_list[self.reward_function_idx]
        self.reward_kwargs = reward_kwargs

        # sticky action feature
        self.sticky_action_mode = sticky_action_mode
        self.stickiness = stickiness
        self.previous_action = np.zeros(12)

        # Use a deque to store the last 20 infos, each info is a dictionary.
        # return the stored info sequence as an additional information to help the agent to learn the special moves.
        # Initialize the info sequence buffer to all empty dictionary at the beginning.
        self.info_sequence_buffer = collections.deque(maxlen=100)
        for _ in range(100):
            self.info_sequence_buffer.append({})

        # Added Modification 1: Round Over Reward
        # Usually the reward should only be given once when the round is over, because the agent's action is frozen when the round is over.
        # But in original implementation, the reward is given in each step even during the animation of round over.
        # To fix this, we need to add a flag to check if the round is over, and only give the reward once when the round is over.
        self.round_over_reward_given = False
        
    
    def _stack_observation(self):
        # return np.stack([self.frame_stack[i * 3 + 2][:, :, i] for i in range(3)], axis=-1)
        return np.stack([self.frame_stack[i][:, :, j] for i in [0,3,6,9] for j in range(3)], axis=-1)


    def reset(self):
        observation = self.env.reset()
        
        self.prev_player_health = self.full_hp 
        self.prev_oppont_health = self.full_hp 

        self.total_timesteps = 0

        # Set character flip
        if random.random() < self.character_flip_rate:
            if self.env.players == 1:
                # Raise a warning if there is "PvP" in the state name.
                if 'PvP' in self.env.statename:
                    warnings.warn("The players parameter of the environment is 1, make sure it is a PvE environment but not PvP. the state file is: " + self.env.statename)
                self.character_flip = False
            else:
                # Flip the characters
                self.character_flip = True
        else:
            self.character_flip = False

        # Step Counter:
        self.step_counter = 0

        # Stack the observation
        if self.stack_observation:
            # Clear the frame stack and add the first observation [num_frames] times
            self.frame_stack.clear()
            for _ in range(self.num_frames):
                self.frame_stack.append(observation[::2, ::2, :])

        # Reset previous_action
        self.previous_action = np.zeros(12)

        # Clear the info sequence buffer, and add all empty dictionary to the info sequence.'
        self.info_sequence_buffer.clear()
        for _ in range(100):
            self.info_sequence_buffer.append({})


        # Reset the round over reward given flag.
        self.round_over_reward_given = False

        # return np.stack([self.frame_stack[i * 3 + 2][:, :, i] for i in range(3)], axis=-1)
        if self.stack_observation:
            return np.stack([self.frame_stack[i][:, :, j] for i in [0,3,6,9] for j in range(3)], axis=-1)
        else:
            return observation


    def step(self, action):
        
        # Sticky Action: 
        # If sticky action mode is enabled, the action in this step is the same as the previous step with a certain probability(stickiness).
        if self.sticky_action_mode and random.random() < self.stickiness:
            action = self.previous_action
            print("Sticky action triggered")

        # Character Flip
        # If character flip is enabled in this environment, flip the actions of player 1 and player 2.
        if self.character_flip:
            # Flip the player 1's action (action[0:12]) and player 2's action (action[12:24])
            action = np.concatenate([action[12:24], action[0:12]])

        custom_done = False

        obs, _reward, _done, info = self.env.step(action)

        # Character Flip
        # Flip the returned info if character flip is enabled in this environment.
        if self.character_flip:
            # Flip the info
            info = self._flip_info(info)
        
        # Previous Infos:
        # Add the previous infos to the info returned by the environment.
        custom_info = copy.deepcopy(info)
        custom_info['info_sequence_buffer'] = self.info_sequence_buffer
        
        if self.stack_observation:
            self.frame_stack.append(obs[::2, ::2, :])

        # Render the game if rendering flag is set to True.
        if self.rendering:
            self.env.render()
            if self.rendering_interval > 0:
                time.sleep(self.rendering_interval)

        curr_player_health = info['agent_hp']
        curr_oppont_health = info['enemy_hp']
        
        self.total_timesteps += self.num_step_frames
        custom_info['step_number'] = self.total_timesteps
        info['step_number'] = self.total_timesteps
        # Reward function

        # First, check if the first element is empty. If so, pass and set the reward to 0.
        # The reward_functions relies on at least one previous info to compute the reward.
        if self.info_sequence_buffer[0] == {}: 
            custom_reward = 0
        else:
            if self.reward_function.__name__ == 'sfai_reward':
                custom_reward, self.round_over_reward_given = self.reward_function(
                    info = custom_info, 
                    round_over_reward_given = self.round_over_reward_given,
                    reward_kwargs = self.reward_kwargs
                )
            else:
                custom_reward, self.round_over_reward_given = self.reward_function(custom_info, self.round_over_reward_given)

        # Check if round is over and set the custom reward and done flag.
        custom_done = (curr_player_health < 0 or 
               curr_oppont_health < 0 or 
               info['round_countdown'] == 0)
        # When reset_round flag is set to False (never reset), the session should always keep going.
        if not self.reset_round:
            custom_done = False
        
        # Sticky Action: 
        # Update the previous action
        self.previous_action = action

        # Previous Infos:
        # Update the info sequence buffer with the current info, then proceed to next step.
        
        # Before appending info to the sequence buffer, first append action to info 
        info['action'] = action
        self.info_sequence_buffer.append(info)  

        # This original code is from LinYi's implementation, but it's better not scale the reward here, but change vf_coef in PPO algorithm instead.
        # If still have to scale reward, do that by passing the reward_kwargs to the reward function.
        # return self._stack_observation(), 0.001 * custom_reward, custom_done, custom_info # reward normalization
        if self.stack_observation:
            observation = self._stack_observation()
        else:
            observation = obs
        return observation, custom_reward, custom_done, custom_info 

        # Max reward is 6 * full_hp = 1054 (damage * 3 + winning_reward * 3) norm_coefficient = 0.001
    
    
    def _flip_info(
            self, 
            info
        ):
        '''
        Flip the informations in the info dictionary.
        The naming rule of the info dictionary is:
        agent_<info_name> for player 1 and enemy_<info_name> for player 2. For example, info['agent_hp'] and info['enemy_hp'].
        According to this rule, the infos of player 1 and player 2 will be flipped using this function.

        :param info: (dict) The info dictionary

        :return: (dict) The flipped info dictionary
        '''
        flipped_info = {}
        # Iterate through the info dictionary and flip the values of the infos of player 1 (agent_<info_name>) and player 2 (enemy_<info_name>).
        for key, value in info.items():
            if key.startswith('agent_'):
                flipped_info[key.replace('agent_', 'enemy_')] = value
            elif key.startswith('enemy_'):
                flipped_info[key.replace('enemy_', 'agent_')] = value
            else:
                flipped_info[key] = value

        # Projectile status extra processing:
        if flipped_info['agent_projectile_status'] == 34176:
            flipped_info['agent_projectile_status'] = 34048
        if flipped_info['enemy_projectile_status'] == 34048:
            flipped_info['enemy_projectile_status'] = 34176
            

        return flipped_info