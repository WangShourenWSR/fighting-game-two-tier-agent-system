import time
import gym 
import retro
import pygame
from pygame.locals import *
import keyboard
from environments.sfai_wrapper import SFAIWrapper
pygame.init()
GAME_STATE = "PvP.RyuVsRyu"
DEFAULT_REWARD_KWARGS = {
    'reward_scale': 0.001,
    'raw_reward_coef': 1.0, # How much HP change is rewarded, only happens during the fighting (not round over)
    'special_move_reward': 0.0, # Reward for using special moves
    'special_move_bonus': 1.0, # Bonus for dealing damage with special moves. The raw reward will be multiplied by this value, so set it to 1.0 if you don't want to use this feature.
    'projectile_reward': 0.0, # Reward for using projectiles
    'projectile_bonus': 3.0, # Bonus for dealing damage with projectiles. The raw reward will be multiplied by this value, so set it to 1.0 if you don't want to use this feature.
    'distance_reward': 0.00, # Set it to positive value to encourage the agent to stay far from the enemy, negative value to encourage the agent to stay close to the enemy
    'distance_bonus': 0.0, # Bonus for dealing damage with projectiles. 0.0 means no bonus, positive/negative value means reward being far/close to the enemy
    'cost_coef': 0.0, # Ratio between the cost and the reward
    'special_move_cost': 2.0,
    'regular_attack_cost': 1.0,
    'jump_cost': 3.0,
    'vulnerable_frame_cost':0.1,
}

def action_translate(a):
    action = ''
    if a[4]==1:
        action = 'jump'
    elif a[5]==1:
        action = 'crouch'
    elif a[6]==1:
        action = 'left'
    elif a[7]==1:
        action = 'right'
    elif a[10]==1:
        action = 'light_punch'
    elif a[1]==1:
        action = 'light_kick'
    elif a[9]==1:
        action = 'medium_punch'
    elif a[0]==1:
        action = 'medium_kick'
    elif a[11]==1:
        action = 'hard_punch'
    elif a[8]==1:
        action = 'hard_kick'
    else:
        action = 'nothing'

    return action

def make_env(game, state, players = 1, rendering_interval = 0.002, reward_kwargs = DEFAULT_REWARD_KWARGS):
    def _init():
        env = retro.make(
            game=game, 
            state=state, 
            use_restricted_actions=retro.Actions.FILTERED,
            obs_type=retro.Observations.IMAGE,
            players=players
        )
        # env = StreetFighterCustomWrapper(env, reset_round=True, rendering=True, reverse_env=False, render_interval= 0.1)
        env = SFAIWrapper(env, reset_round= True, rendering = True, rendering_interval = rendering_interval, reward_function_idx = 5, reward_kwargs=reward_kwargs)
        return env
    return _init

def main(
        game_state: str = GAME_STATE,
        rendering_interval: float = 0.002,
        reward_kwargs = DEFAULT_REWARD_KWARGS
):
    game = "StreetFighterIISpecialChampionEdition-Genesis"
    env = make_env(
        game, 
        state=game_state, 
        players = 1,
        rendering_interval=rendering_interval,
        reward_kwargs=reward_kwargs
        )()

    # init the pygame screen
    screen = pygame.display.set_mode((env.observation_space.shape[1], env.observation_space.shape[0]))
    pygame.display.set_caption('Street Fighter with Pygame Controls')
    _obs = env.reset()

    # init the accumulated reward
    accumulated_reward = 0

    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # 按键检测
        keys = pygame.key.get_pressed() 

        # 检查pygame是否捕获到了按键事件，如果有，就返回一个包含所有按键状态的列表。
        # key_values = [keys[K_w], keys[K_s], keys[K_a], keys[K_d], keys[K_u], keys[K_i], keys[K_o], keys[K_j], keys[K_k], keys[K_l]]

        # Key input detection
        action = [0] * 12  # 初始化一个全0的动作列表

        # The keys and its corresponding actions
        # ['medium_kick', 'light_kick', ?, ?, 'jump', 'crouch', 'left', 'right', 'hard_kick', 'medium_punch', 'light_punch', 'hard_punch']q
        if keys[K_w]:
            action[4]=1
        if keys[K_s]:
            action[5]=1
        if keys[K_a]:
            action[6]=1
        if keys[K_d]:
            action[7]=1
        if keys[K_u]:
            action[10]=1
        if keys[K_j]:
            action[1]=1
        if keys[K_i]:
            action[9]=1
        if keys[K_k]:
            action[0]=1
        if keys[K_o]:
            action[11]=1
        if keys[K_l]:
            action[8]=1

        observation, reward, done, info = env.step(action)
        # if reward != 0:
        #     print("reward of current step: ", reward)
        accumulated_reward += reward
        # print(accumulated_reward)

        if done:
            observation = env.reset()
            # print the total reward
            print("Total reward: {}".format(accumulated_reward))
        
        # 刷新屏幕
        pygame.display.flip()

    env.close()



