import time
import gym 
import retro
import pygame
from pygame.locals import *
import keyboard
from src.environments.sfai_wrapper import SFAIWrapper
# 初始化pygame
pygame.init()

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
# 创建gymretro游戏环境
# env = retro.make(game='StreetFighterIISpecialChampionEdition-Genesis', state='Champion.Level1.RyuVsGuile')
def make_env(game, state):
    def _init():
        env = retro.make(
            game=game, 
            state=state, 
            use_restricted_actions=retro.Actions.FILTERED,
            obs_type=retro.Observations.IMAGE
        )
        # env = StreetFighterCustomWrapper(env, reset_round=True, rendering=True, reverse_env=False, render_interval= 0.1)
        env = SFAIWrapper(env, reset_round= True, rendering = True, rendering_interval= 0.02)
        return env
    return _init

game = "StreetFighterIISpecialChampionEdition-Genesis"
#env = make_env(game, state="Champion.Level1.RyuVsGuile")()
env = make_env(game, state="PvP.RyuvsRyu")()
# env = make_env(game, state="Champion.Level12.RyuVsBison")()

# 每次重新开始游戏时，删除agent_action_status.txt文件中的内容
with open('agent_action_status.txt', 'w') as f:
    f.write('')

# 初始化环境和pygame窗口
screen = pygame.display.set_mode((env.observation_space.shape[1], env.observation_space.shape[0]))
pygame.display.set_caption('Street Fighter with Pygame Controls')
observation = env.reset()

# 初始化累计奖励
total_reward = 0

# 游戏主循环
save_action = False
running = True
key_names = ['w', 's', 'a', 'd', 'u', 'i', 'o', 'j', 'k', 'l']

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # 按键检测
    keys = pygame.key.get_pressed() 
    # 
    # 检查pygame是否捕获到了按键事件，如果有，就返回一个包含所有按键状态的列表。
    key_values = [keys[K_w], keys[K_s], keys[K_a], keys[K_d], keys[K_u], keys[K_i], keys[K_o], keys[K_j], keys[K_k], keys[K_l]]
    pressed_keys = [name for name, value in zip(key_names, key_values) if value]
    # if pressed_keys:
    #     print(f"Pressed keys: {', '.join(pressed_keys)}")
    
    # 按键检测
    action = [0] * 12  # 初始化一个全0的动作列表
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
    if keys[K_1]:
        action[2]=1
        # action = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    if keys[K_2]:
        # action = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
        action[3]=1


    observation, reward, done, info = env.step(action)
    total_reward += reward
    # print(info['combo_num'])
    # print(info['jump_num'])
    # print(info['attack_num'])
    # print("The position of agent: {}".format(info['agent_x']))
    # print("The position of enemy: {}".format(info['enemy_x']))
    # print("The distance between two players: {}".format(info['distance_average']))
    # print("The round countdown is: {}".format(info['round_countdown']))
    # env.render()
    
    # print("The status of the agent: {}".format(info['agent_status']))
    # print("The status of the enemy: {}".format(info['enemy_status']))


    # time.sleep(0.01)

    # 保存动作和状态的观察结果
    # 将动作和状态的观察结果保存到文件中，每一个观察结果占一行。
    if save_action and action_translate(action) != 'nothing':
        agent_status = info['agent_status']
        agent_action = action_translate(action)
        # 保存结果到文件的新一行
        with open('agent_action_status.txt', 'a') as f:
            f.write(str(agent_status) + ' ' + agent_action + '\n')




    if done:
        observation = env.reset()
        # 输出累计奖励
        print("Total reward: {}".format(total_reward))
    
    # 刷新屏幕
    pygame.display.flip()





env.close()



