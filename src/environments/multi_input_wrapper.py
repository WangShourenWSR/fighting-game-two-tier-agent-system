import gym
import gym.spaces
import gym.spaces.multi_binary
import numpy as np
from stable_baselines3.common.base_class import BaseAlgorithm

class MultiInputWrapper(gym.Wrapper):
    def __init__(
            self, 
            env,
        ):
        super(MultiInputWrapper, self).__init__(env)
        self.env = env
        cnn_space_shape = env.observation_space.shape

        # Redefine the observation space for MultiInputWrapper, which is a gym.space.Dict
        self.observation_space = gym.spaces.Dict(
            {
                'game_pixel': gym.spaces.Box(low=0, high=255, shape=cnn_space_shape, dtype=np.uint8),
                'action_sequence': gym.spaces.MultiBinary((100,12)),
                'player_character': gym.spaces.Discrete(12),
                'player_status': gym.spaces.Discrete(600),
                'enemy_character': gym.spaces.Discrete(12),
                'enemy_status': gym.spaces.Discrete(600)
            }
        )

    def reset(self, **kwargs):
        image_obs = self.env.reset(**kwargs)
        # initialize a dict to store the observation with all zero values accoriding to self.observation_space
        obs = {
            'game_pixel': image_obs,
            'action_sequence': np.zeros((100,12)),
            'player_character': 0,
            'player_status': 0,
            'enemy_character': 0,
            'enemy_status': 0
        }
        self.last_observation = obs
        return obs
    
    def step(self, action):
        dict_obs = {}

        obs, _reward, _done, info = self.env.step(action)
        dict_obs['game_pixel'] = obs
        
        # Extended the information for observation dict
        if info['info_sequence_buffer'] is not None:
            info_sequence_buffer = info['info_sequence_buffer']
        else:
            raise ValueError("info_sequence_buffer is None. Please wrap SFAIWrapper first, then wrap this wrapper.")
        action_sequence = np.zeros((100,12))

        # Iterate over info_sequence_buffer, it is a deque, elements are added from tail to head. So we need to reverse the order of the elements.
        # Reverse iteration over the deque:
        for i in range(len(info_sequence_buffer)):
            # Check if there is a key's name is 'info_sequence_buffer', becasue at the very beginning, info sequence buffer is empty. if so, set this action all 0.
            if 'action' in info_sequence_buffer[-(i+1)].keys():
                action_sequence[i] = info_sequence_buffer[-(i+1)]['action'][0:12] # Only the first 12 elements are needed (player's action)
            else:
                continue

        dict_obs['action_sequence'] = action_sequence

        # Extract the agent's character and status
        player_character = info['agent_character']
        player_status = info['agent_status']
        dict_obs['player_character'] = player_character
        dict_obs['player_status'] = player_status

        # Extract the enemy's character and status
        enemy_character = info['enemy_character']
        enemy_status = info['enemy_status']
        dict_obs['enemy_character'] = enemy_character
        dict_obs['enemy_status'] = enemy_status

        return dict_obs, _reward, _done, info




