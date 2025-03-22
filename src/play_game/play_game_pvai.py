import time
import os
import retro
import pygame
from pygame.locals import *
from environments.sfai_wrapper import SFAIWrapper
from environments.multi_input_wrapper import MultiInputWrapper
from models.aux_obj_ppo import AuxObjPPO
from stable_baselines3 import PPO
from stable_baselines3.common.base_class import BaseAlgorithm

NUM_EPISODES = 3 # The number of episodes to play
MODEL_DIR = r"agent_models/large_files" # Specify the model directory.
MODEL_NAME ="28800000_steps_%PPO%self_play_ppo$w_0.4_r_0.2853500634809375$"
GAME_STATE = "PvP.RyuvsRyu"
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

# Create the gymretro game environment
def make_env(game, state, players = 1, reward_kwargs = None, character_flip_rate = 0.0):
    def _init():
        env = retro.make(
            game=game, 
            state=state, 
            use_restricted_actions=retro.Actions.FILTERED,
            obs_type=retro.Observations.IMAGE,
            players=players
        )
        # env = StreetFighterCustomWrapper(env, reset_round=True, rendering=True, reverse_env=False, render_interval= 0.1)
        env = SFAIWrapper(env, reset_round= True, rendering = True, rendering_interval= 0.005, reward_kwargs=reward_kwargs, character_flip_rate=character_flip_rate)
        return env
    return _init

def main(
        model_class: BaseAlgorithm = PPO,
        multi_input: bool = False,
        game_state: str = GAME_STATE,
        model_dir: str = MODEL_DIR,
        model_name: str = MODEL_NAME,
        reward_kwargs = DEFAULT_REWARD_KWARGS,
        character_flip_rate = 0.0
):
    game = "StreetFighterIISpecialChampionEdition-Genesis"
    env = make_env(
        game, state = game_state, 
        players = 2, 
        reward_kwargs = reward_kwargs,
        character_flip_rate = character_flip_rate
        )()
    if multi_input:
        env = MultiInputWrapper(env)

    print("loading model")
    # Load the agent's model
    model = model_class.load(os.path.join(model_dir, model_name))
    print("model loaded")

    # Initialize pygame
    pygame.init()
    screen = pygame.display.set_mode((320, 224))
    pygame.display.set_caption("Street Fighter II PvP")

    # Main loop for the game
    num_episodes = NUM_EPISODES
    print_info = False

    for _ in range(num_episodes):
        accumulated_reward = 0
        done = False
        running = True
        obs = env.reset()
        action24 = [0]*24 # The actions for both player(0~11) and the agent(12~23).
        # Loop for each episode
        num_steps = 0
        start_time = time.time()
        while running:
            time.sleep(0.003)
            num_steps += 1
            if num_steps % 100 == 0:
                print('FPS: ', num_steps/(time.time()-start_time))
                print('Total time: ', time.time()-start_time)
                start_time = time.time()
                num_steps = 0

            # Get the actions from the agent model
            action_agent, _states = model.predict(obs)
            # action24[12:24] = action_agent
            action24[0:12] = action_agent

            # Get the actions from the player
            keys = pygame.key.get_pressed()
            key_values = [keys[K_w], keys[K_s], keys[K_a], keys[K_d], keys[K_u], keys[K_i], keys[K_o], keys[K_j], keys[K_k], keys[K_l]]
            action_player = [0]*12
            if keys[K_w]:
                action_player[4]=1
            if keys[K_s]:
                action_player[5]=1
            if keys[K_a]:
                action_player[6]=1
            if keys[K_d]:
                action_player[7]=1
            if keys[K_u]:
                action_player[10]=1
            if keys[K_j]:
                action_player[1]=1
            if keys[K_i]:
                action_player[9]=1
            if keys[K_k]:
                action_player[0]=1
            if keys[K_o]:
                action_player[11]=1
            if keys[K_l]:
                action_player[8]=1
            if keys[K_1]:
                action_player[2]=1
            if keys[K_2]:
                action_player[3]=1
            
            # action24[0:12] = action_player
            action24[12:24] = action_player
            
            # Perform the actions
            obs, reward, done, info = env.step(action24) 

            accumulated_reward += reward
            # if reward != 0:
            #     print("reward of current step: ", reward)
            #     print("accumulated_reward: ", accumulated_reward)
            # print("The accumulated reward is : ", accumulated_reward)
            # print('Eneny HP: ', info['enemy_hp'])
            # print("Enemy HP previous: ",  info['info_sequence_buffer'][-1]['enemy_hp'])
            # print("Enemy HP changed: ", info['info_sequence_buffer'][-1]['enemy_hp'] != info['enemy_hp'])

            if info['round_countdown'] >= 30000 and info['round_countdown']<= 36000 and not print_info:
                # print('The info sequence buffer is : ', info['info_sequence_buffer'])
                # Save info to a file
                # Write the element of info in one line, and write the elements of info_sequence_buffer line by line.
                with open('info_sequence_buffer.txt', 'w') as f:
                    # First, write the elements of info except info_sequence_buffer.
                    for key in info:
                        if key != 'info_sequence_buffer':
                            f.write(key+':'+str(info[key])+'\n')
                    for i in info['info_sequence_buffer']:
                        f.write(str(i)+'\n')
                print_info = True


if __name__ == "__main__":
    main()
    


            


        
        


