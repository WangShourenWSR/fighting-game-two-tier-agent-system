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

NUM_EPISODES = 1 # The number of episodes to play
MODEL_DIR = r"agent_models/large_files" # Specify the model directory.
MODEL_NAME ="28800000_steps_%PPO%self_play_ppo$w_0.4_r_0.2853500634809375$"
GAME_STATE = "PvP.RyuvsRyu"

REWARD_KWARGS = {
    'raw_reward_coef': 0.0, # reward only based on projectile
    'special_move_reward': 0.0, # Reward for using special moves
    'special_move_bonus': 0.0, 
    'projectile_reward': 10.0, # Reward for using projectiles
    'projectile_bonus': 10.0, # Bonus for dealing damage with projectiles. The raw reward will be multiplied by this value.
    'cost_coef': 0.05, # Ratio between the cost and the reward
    'special_move_cost': 2.0,
    'regular_attack_cost': 1.0,
    'jump_cost': 3.0,
}

# Create the gymretro game environment
def make_env(game, state, players = 1,reward_kwargs = {}):
    def _init():
        env = retro.make(
            game=game, 
            state=state, 
            use_restricted_actions=retro.Actions.FILTERED,
            obs_type=retro.Observations.IMAGE,
            players=players
        )
        # env = StreetFighterCustomWrapper(env, reset_round=True, rendering=True, reverse_env=False, render_interval= 0.1)
        env = SFAIWrapper(env, reset_round= True, rendering = True, rendering_interval= 0.005, reward_function_idx= 0,reward_kwargs=reward_kwargs, character_flip_rate=0.0)
        env = MultiInputWrapper(env)
        return env
    return _init

def main(
        model_class: BaseAlgorithm = PPO,
        multi_input: bool = False,
        game_state: str = GAME_STATE,
        model_dir_player1: str = MODEL_DIR,
        model_name_player1: str = MODEL_NAME,
        model_dir_player2: str = MODEL_DIR,
        model_name_player2: str = MODEL_NAME,
        reward_kwargs = REWARD_KWARGS
):
    game = "StreetFighterIISpecialChampionEdition-Genesis"
    env = make_env(game, state = game_state, players = 2, reward_kwargs=reward_kwargs)()
    # if multi_input:
    #     env = MultiInputWrapper(env)

    print("loading model")
    # Load the agent's model
    model_player1 = model_class.load(os.path.join(model_dir_player1, model_name_player1))
    model_player2 = model_class.load(os.path.join(model_dir_player2, model_name_player2))
    print("model loaded")

    # Main loop for the game
    num_episodes = NUM_EPISODES
    print_info = False

    player1_wins = 0
    for num_rounds in range(num_episodes):
        accumulated_reward = 0
        done = False
        running = True
        obs = env.reset()
        action24 = [0]*24 # The actions for both player(0~11) and the agent(12~23).
        # Loop for each episode
        while running:
            # Get the actions from the agent model
            action_player1, _states = model_player1.predict(obs, deterministic=True)
            action_player2, _states = model_player2.predict(obs['game_pixel'],deterministic=True)
            # action24[12:24] = action_agent
            action24[0:12] = action_player1
            action24[12:24] = action_player2
            
            # Perform the actions
            obs, reward, done, info = env.step(action24) 
            running = not done
            if done:
                print('player1 hp : ', info['agent_hp'])
                print('enemy hp : ', info['enemy_hp'])
                if info['agent_hp'] > info['enemy_hp']:
                    player1_wins += 1
                else:
                    player1_wins += 0

            accumulated_reward += reward
            # print("The accumulated reward is : ", accumulated_reward)
            # print(f"The projectile index: {info['agent_projectile_status']}")
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
    env.render(close = True)
    env.close()
    print("\033[1;32m" + f"Player 1 wins: {player1_wins} out of {num_episodes}" + "\033[0m")

if __name__ == "__main__":
    main()
    


            


        
        


