import time
import gym 
import retro
import pygame
from pygame.locals import *
import keyboard
from environments.sfai_wrapper import SFAIWrapper
from environments.evaluation_wrapper import EvaluationWrapper
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

def make_env(game, state, players = 1, rendering_interval = 0.002, reward_kwargs = DEFAULT_REWARD_KWARGS, character_flip_rate = 0.0):
    def _init():
        env = retro.make(
            game=game, 
            state=state, 
            use_restricted_actions=retro.Actions.FILTERED,
            obs_type=retro.Observations.IMAGE,
            players=players
        )
        # env = StreetFighterCustomWrapper(env, reset_round=True, rendering=True, reverse_env=False, render_interval= 0.1)
        env = SFAIWrapper(env, reset_round= True, rendering = True, rendering_interval = rendering_interval, reward_function_idx = 0, reward_kwargs=reward_kwargs, character_flip_rate = character_flip_rate)
        env = EvaluationWrapper(env)
        return env
    return _init

def main(
        game_state: str = GAME_STATE,
        rendering_interval: float = 0.002,
        reward_kwargs = DEFAULT_REWARD_KWARGS,
        character_flip_rate = 0.0
):
    game = "StreetFighterIISpecialChampionEdition-Genesis"
    env = make_env(
        game, 
        state=game_state, 
        players = 2,
        rendering_interval=rendering_interval,
        reward_kwargs=reward_kwargs,
        character_flip_rate = character_flip_rate
        )()


    # 每次重新开始游戏时，删除agent_action_status.txt文件中的内容
    with open('agent_action_status.txt', 'w') as f:
        f.write('')

    # init the pygame screen
    screen = pygame.display.set_mode((env.observation_space.shape[1], env.observation_space.shape[0]))
    pygame.display.set_caption('Street Fighter with Pygame Controls')
    observation = env.reset()

    # init the accumulated reward
    accumulated_reward = 0

    # Define the key names for player 1 and player 2
    key_names_player1 = ['w', 's', 'a', 'd', 'u', 'i', 'o', 'j', 'k', 'l']
    key_names_player2 = ['up', 'down', 'left', 'right', 'numpad1', 'numpad2', 'numpad3', 'numpad4', 'numpad5', 'numpad6']
    # key_names_player2 = ['m', 'f', 'g', 'r', 'z', 'x', 'c', 'v', 'b', 'n']

    save_action = False
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # 按键检测
        keys = pygame.key.get_pressed() 

        # 检查pygame是否捕获到了按键事件，如果有，就返回一个包含所有按键状态的列表。
        key_values = [keys[K_w], keys[K_s], keys[K_a], keys[K_d], keys[K_u], keys[K_i], keys[K_o], keys[K_j], keys[K_k], keys[K_l],
                            # keys[K_z], keys[K_x], keys[K_c], keys[K_v], keys[K_b], keys[K_n], keys[K_m], keys[K_f], keys[K_g], keys[K_r]]
                    keys[K_b], keys[K_DOWN], keys[K_LEFT], keys[K_RIGHT], keys[K_1], keys[K_2], keys[K_3], keys[K_4], keys[K_5], keys[K_6]]
        pressed_keys_player1 = [name for name, value in zip(key_names_player1, key_values) if value]
        pressed_keys_player2 = [name for name, value in zip(key_names_player2, key_values) if value]
        # if pressed_keys_player1:
        #     print(f"Player 1 Pressed keys: {', '.join(pressed_keys_player1)}")
        # if pressed_keys_player2:
        #     print(f"Player 2 Pressed keys: {', '.join(pressed_keys_player2)}")

        
        # 按键检测
        action = [0] * 24  # 初始化一个全0的动作列表
        # 按键对应的动作
        # ['中腿', '轻腿', '？', '？', '跳', '蹲', '左', '右', '重腿', '中拳', '轻拳', '重拳']
        if keys[K_w]:
            action[4]=1
        # action = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
        if keys[K_s]:
            action[5]=1
            # action = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
        if keys[K_a]:
            action[6]=1
            # action = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
        if keys[K_d]:
            action[7]=1
            # action = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
        if keys[K_u]:
            action[10]=1
            # action[0]=1
            # action = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        if keys[K_j]:
            action[1]=1
            # action[1]=1
            # action = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        if keys[K_i]:
            action[9]=1
            # action = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
        if keys[K_k]:
            action[0]=1
            # action[10]=1
            # action = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
        if keys[K_o]:
            action[11]=1
            # action[8]=1
            # action = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
        if keys[K_l]:
            action[8]=1
            # action[11]=1
            # action = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
        if keys[K_UP]:
            action[16]=1
        if keys[K_DOWN]:
            action[17]=1
        if keys[K_LEFT]:
            action[18]=1
        if keys[K_RIGHT]:
            action[19]=1
        if keys[K_1]:
            action[22]=1
        if keys[K_2]:
            action[21]=1
        if keys[K_3]:
            action[23]=1
        if keys[K_4]:
            action[13]=1
        if keys[K_5]:
            action[12]=1
        if keys[K_6]:
            action[20]=1
        # print(action)

        observation, reward, done, info = env.step(action)

        if done:
            print('behavior_metrics: ', info['behavior_metrics']) 

        if done:
            for key, value in info['behavior_metrics'].items():
                print("{}: ".format(key), value)
        # if reward != 0:
        #     print("reward of current step: ", reward)
        accumulated_reward += reward
        if reward != 0:
            print("reward of current step: ", reward)
            print("accumulated_reward: ", accumulated_reward)

        # if reward > 0:
        #     print("reward of current step: ", reward)
        #     print("accumulated_reward: ", accumulated_reward)

        # print(accumulated_reward)
        # Save the action and status observation results
        # Save the result to a new line in the file
        if save_action and action_translate(action) != 'nothing':
            agent_status = info['agent_status']
            agent_action = action_translate(action)
            # Save the result to a new line
            with open('agent_action_status.txt', 'a') as f:
                f.write(str(agent_status) + ' ' + agent_action + '\n')




        if done:
            observation = env.reset()
            # print the total reward
            print("Total reward: {}".format(accumulated_reward))
            # reset accumulated reward
            accumulated_reward = 0
        
        # 刷新屏幕
        pygame.display.flip()



    env.close()



