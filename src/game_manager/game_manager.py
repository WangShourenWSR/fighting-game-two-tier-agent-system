import os
import sys
import json
import re
import random
import time

import pygame
from pygame.locals import *
import retro
from models.aux_obj_ppo import AuxObjPPO
from stable_baselines3 import PPO
from stable_baselines3.common.base_class import BaseAlgorithm

from environments.sfai_wrapper import SFAIWrapper
from environments.multi_input_wrapper import MultiInputWrapper
from environments.evaluation_wrapper import EvaluationWrapper

from hyper_agent.hyper_agent import HyperAgent


class GameManager:
    """
    This class is responsible for:
      1) Managing gameplay workflow: selecting characters, opponents, starting matches.
      2) Collecting data during battles (via wrappers, callbacks, or internal logging).
      3) Summarizing relevant player data that can be passed to the HyperAgent.
    """
    def __init__(
            self,
            hyper_agent_model_path = r'C:/Users/SR.W/LLMs/DeepSeek-R1-Distill-Llama-8B/snapshots/6a6f4aa4197940add57724a7707d069478df56b1'
        ):
        self._game = "StreetFighterIISpecialChampionEdition-Genesis"
        # load game states from game_states.json
        with open("src/game_manager/game_states.json") as f:
            self._game_states = json.load(f)   
        # load agents_archive from agents_archive.json
        with open("src/game_manager/agents_archive.json") as f:
            self._agents_archive = json.load(f)
        
        self.playing_data = {
            "current_character": 'Ryu',
            "total_matches": 0,
            "win_rate": 0,
            "total_wins": 0,
            "total_losses": 0,
            "current_win_streak": 0,
            "current_loss_streak": 0,
            "average_score_per_match": "0/100",
            # "average_reward_per_match": 0,
            "average_special_moves_per_match": 0,
            "faced_agents_times": {
                "projectile_type": 0,
                "special_move_type": 0,
                "defensive_type": 0,
                "aggressive_type": 0,
                "air_type": 0,
                "coward_type": 0,
                "newbie_type": 0,
                "key_spamming_type": 0
            },
            "faced_characters_times": {
                "Ryu": 0,
                "Ken": 0,
                "Chunli": 0,
                "Guile": 0,
                "Blanka": 0,
                "Zangief": 0,
                "Dhalsim": 0,
                "Balrog": 0,
                "Vega": 0,
                "Sagat": 0,
                "Bison": 0,
                "EHonda": 0,    
            },
            "the_last_opponents": {
                "type": "This is the first match, no last opponent.",
                "character": "This is the first match, no last opponent.",
                "model_path": "This is the first match, no last opponent.",
                "difficulty": "This is the first match, no last opponent."
            },
            "player's_feedback": {
                "First match, no feedback yet."
            }
        }
        
        self._default_agent_model_path = r"agent_models/agents_archive/air_type/1_0.08"
        self._default_agent_type = "air_type"

        self.hyper_agent = HyperAgent(
            model_path = hyper_agent_model_path,
            agents_archive=self._agents_archive,
            state_files=self._game_states,
        )

    def start_game(
            self,
            blind_test: bool = False
        ):
        """
        Starts the game.
        """
        # 1) d basic introduction and available characters
        print("\033[1;3;38;5;208mWelcome to Enjoyability Focused Street Fighter II Game-Playing Agent Project!\033[0m")

        # print("Welcome to Enjoyability Focused Street Fighter II Game-Playing Agent Project!")
        print("\033[1;32mHere are the available characters you can choose to play as:\033[0m")
        # Iterate over self._game_states to print available characters
        for key, _value in self._game_states.items():
            print("\033[32m{}\033[0m".format(key))

        # 2) Ask the player to select a character, if not in the list, ask player to re-enter
        while True:
            selected_character = input("\033[1;33mNow,Please select a character from the list above: \033[0m").strip()
            if selected_character in self._game_states:
                self.player_character = selected_character
                self.playing_data["current_character"] = selected_character
                break
            else:
                print("\033[33mInvalid character selected. Please try again.\033[0m")

        # 4) For blind tests (randomly select an opponent for the player).
        if blind_test:
            # num_matches_random = 0
            # while True:
            #     num_matches_random += 1
            #     if num_matches_random > 1: 
            #         # Randomly wait for 1 to 2 minutes to simuate the time for hyper agent to select an opponent.
            #         print("Please wait for 1 to 2 minutes for selecting next opponent...")
            #         time.sleep(random.randint(60,62))
            #     # Randomly select an opponent(model and state files) for the player.
            #     random_type = random.choice(list(self._agents_archive.keys()))
            #     randon_chosen_opponent = random.choice(list(self._agents_archive[random_type]["suggested_characters_for_this_type"]))
            #     random_model_path = random.choice(self._agents_archive[random_type]["agent_models"])["model_path"]
            #     # Construct the state file name
            #     random_state_file = f"PvP.{self.player_character}Vs{randon_chosen_opponent}"

            #     # Update the playing data
            #     self.playing_data["faced_agents_times"][random_type] += 1
            #     self.playing_data["faced_characters_times"][randon_chosen_opponent] += 1
                
            #     print(f"\033[33mMatch {self.playing_data['total_matches']} starting...")
            #     print(f"Your character: {self.playing_data['current_character']}")
            #     print(f"Your opponent: {randon_chosen_opponent}\033[0m")
            #     # print(f"Opponent: [Model: {chosen_opponent['model_path']}, Character: {chosen_opponent['character']}]")

            #     chosen_opponent = random.choice(['Ryu', 'Ken', 'Chunli', 'Guile', 'Blanka', 'Zangief', 'Dhalsim', 'Balrog', 'Vega', 'Sagat', 'Bison', 'EHonda'])
            #     chosen_model_path = "agent_models/baseline_blind_test/CNN_self_play"
            #     chosen_state_file = f"PvP.{self.player_character}Vs{randon_chosen_opponent}"
            #     # Run the match and get the behavior metrics
            #     match_behavior_metrics = self._run_match(
            #         game_state=chosen_state_file,
            #         agent_model_path=chosen_model_path,
            #     )

            #     # Update the playing data
            #     self._update_playing_behavior_data(match_behavior_metrics)

            #     # Ask the player for feedback
            #     provide_feed_back = input("\033[1;33mDo you want to provide some feedback for this match? (yes/no) \033[0m").strip()
            #     while provide_feed_back not in ["yes", "no"]:
            #         print("Invalid input. Please enter 'yes' or 'no'.")
            #         provide_feed_back = input("Do you want to provide some feedback for this match? (yes/no): ").strip().lower()
            #     if provide_feed_back == "yes":
            #         feedback = input("\033[1;33mPlease provide your feedback for this match: \033[0m").strip()
            #         self.playing_data["player's_feedback"] = feedback
            #     else:
            #         self.playing_data["player's_feedback"] = "The player did not provide feedback for the last match."       

            #     # Print the match results
            #     print(f"\033[1;33mMatch {self.playing_data['total_matches']} ended.")
            #     print(self.playing_data)
            #     print("\033[0m")

            #     if blind_test and num_matches_random >= 5:
            #         print("\033[1;31mTest 1 complete, Thank you!\033[0m")
            #         break

            #     # 询问玩家是否进入下一局
            #     next_round = input("Do you want to play another round? (yes/no): ").strip().lower()
            #     # If the player's input is not 'yes' or 'no', ask the player to re-enter
            #     while next_round not in ["yes", "no"]:
            #         print("Invalid input. Please enter 'yes' or 'no'.")
            #         next_round = input("Do you want to play another round? (yes/no): ").strip().lower()
            #     if next_round == "no":
            #         break


            num_matches_random = 0
            while True:
                num_matches_random += 1
                if num_matches_random > 1: 
                    # Randomly wait for 1 to 2 minutes to simuate the time for hyper agent to select an opponent.
                    print("Please wait for 1 to 2 minutes for selecting next opponent...")
                    time.sleep(random.randint(60, 120))
                # Randomly select an opponent(model and state files) for the player.
                random_type = random.choice(list(self._agents_archive.keys()))
                randon_chosen_opponent = random.choice(list(self._agents_archive[random_type]["suggested_characters_for_this_type"]))
                random_model_path = random.choice(self._agents_archive[random_type]["agent_models"])["model_path"]
                # Construct the state file name
                random_state_file = f"PvP.{self.player_character}Vs{randon_chosen_opponent}"

                # Update the playing data
                self.playing_data["faced_agents_times"][random_type] += 1
                self.playing_data["faced_characters_times"][randon_chosen_opponent] += 1
                
                print(f"\033[33mMatch {self.playing_data['total_matches']} starting...")
                print(f"Your character: {self.playing_data['current_character']}")
                print(f"Your opponent: {randon_chosen_opponent}\033[0m")
                # print(f"Opponent: [Model: {chosen_opponent['model_path']}, Character: {chosen_opponent['character']}]")

                # Run the match and get the behavior metrics
                match_behavior_metrics = self._run_match(
                    game_state=random_state_file,
                    agent_model_path=random_model_path,
                )

                # Update the playing data
                self._update_playing_behavior_data(match_behavior_metrics)

                # Ask the player for feedback
                provide_feed_back = input("\033[1;33mDo you want to provide some feedback for this match? (yes/no) \033[0m").strip()
                while provide_feed_back not in ["yes", "no"]:
                    print("Invalid input. Please enter 'yes' or 'no'.")
                    provide_feed_back = input("Do you want to provide some feedback for this match? (yes/no): ").strip().lower()
                if provide_feed_back == "yes":
                    feedback = input("\033[1;33mPlease provide your feedback for this match: \033[0m").strip()
                    self.playing_data["player's_feedback"] = feedback
                else:
                    self.playing_data["player's_feedback"] = "The player did not provide feedback for the last match."       

                # Print the match results
                print(f"\033[1;33mMatch {self.playing_data['total_matches']} ended.")
                print(self.playing_data)
                print("\033[0m")

                if blind_test and num_matches_random >= 5:
                    print("\033[1;31mTest 1 complete, Thank you!\033[0m")
                    break

                # 询问玩家是否进入下一局
                next_round = input("Do you want to play another round? (yes/no): ").strip().lower()
                # If the player's input is not 'yes' or 'no', ask the player to re-enter
                while next_round not in ["yes", "no"]:
                    print("Invalid input. Please enter 'yes' or 'no'.")
                    next_round = input("Do you want to play another round? (yes/no): ").strip().lower()
                if next_round == "no":
                    break
        
        if blind_test:
            ready_for_next_test = False
            while ready_for_next_test != "yes":
                ready_for_next_test = input("\033[1;31mNow we are entering test 2, please enter yes when you're ready\033[0m").strip().lower()
                if ready_for_next_test != "yes":
                    print("\033[1;31mInvalid input. Please enter 'yes' when you're ready.\033[0m")


        # 3) Enter the main loop for repeated matches
        if blind_test:
            self._reset_playing_data()
        while True:
            # For the first match, randomly select an opponent for the player.
            if self.playing_data["total_matches"] == 0:
                possible_opponents = self._game_states.get(self.player_character, {})
                chosen_opponent,chosen_state = random.choice(list(possible_opponents.items()))
                chosen_model_path = self._default_agent_model_path
                chosen_agent_type = self._default_agent_type
                self.playing_data["faced_agents_times"]['air_type'] += 1
                self.playing_data["faced_characters_times"][chosen_opponent] += 1
            else:
                if blind_test:
                    sys.stdout = open(os.devnull, 'w', encoding="utf-8")  
                chosen_agent_type, chosen_model_path, chosen_opponent, chosen_state  = self.hyper_agent.select_agent(playing_data=self.playing_data)
                if blind_test:
                    sys.stdout = sys.__stdout__
                self.playing_data["faced_agents_times"][chosen_agent_type] += 1
                self.playing_data["faced_characters_times"][chosen_opponent] += 1

            print(f"\033[33mMatch {self.playing_data['total_matches']} starting...")
            print(f"Your character: {self.playing_data['current_character']}")
            print(f"Your opponent: {chosen_opponent}\033[0m")
            # print(f"Opponent: [Model: {chosen_opponent['model_path']}, Character: {chosen_opponent['character']}]")

            # Run the match and get the behavior metrics
            match_behavior_metrics = self._run_match(
                game_state=chosen_state,
                agent_model_path=chosen_model_path,
            )

            # Update info of the last opponent
            self.playing_data["the_last_opponents"] = {
                "type": chosen_agent_type,
                "character": chosen_opponent,
                "model_path": chosen_model_path,
                "difficulty": self._agents_archive[chosen_agent_type]["agent_models"][0]["model_difficulty_score"]
            }
            
            # Ask the player for feedback
            provide_feed_back = input("\033[1;33mDo you want to provide some feedback for this match? (yes/no) \033[0m").strip()
            while provide_feed_back not in ["yes", "no"]:
                print("Invalid input. Please enter 'yes' or 'no'.")
                provide_feed_back = input("Do you want to provide some feedback for this match? (yes/no): ").strip().lower()
            if provide_feed_back == "yes":
                feedback = input("\033[1;33mPlease provide your feedback for this match: \033[0m").strip()
                self.playing_data["player's_feedback"] = feedback
            else:
                self.playing_data["player's_feedback"] = "The player did not provide feedback for the last match."            

            # Update the playing data
            self._update_playing_behavior_data(match_behavior_metrics)

            # Print the match results
            print(f"\033[1;33mMatch {self.playing_data['total_matches']} ended.")
            print(self.playing_data)
            print("\033[0m")

            if blind_test and self.playing_data["total_matches"] >= 5:
                print("\033[1;31mBlind test complete, Thank you!\033[0m")
                break

            # 询问玩家是否进入下一局
            next_round = input("Do you want to play another round? (yes/no): ").strip().lower()
            # If the player's input is not 'yes' or 'no', ask the player to re-enter
            while next_round not in ["yes", "no"]:
                print("Invalid input. Please enter 'yes' or 'no'.")
                next_round = input("Do you want to play another round? (yes/no): ").strip().lower()
            
            if next_round == "no":
                break


        # End the game
        print("\033[1;3;38;5;208mThank you for playing!\033[0m")
        

    def _run_match(
            self,
            game_state: str,
            agent_model_path: str,
    ):
        env = self.make_env(
            state=game_state,
            players=2,
        )()

        # Load the agent's model
        model = PPO.load(agent_model_path)

        # # Initialize pygame
        pygame.init()
        screen = pygame.display.set_mode((320, 224))
        pygame.display.set_caption("Street Fighter II PvP")

        # Initialize the variables for the match
        accumulated_reward = 0
        obs = env.reset()
        action24 = [0]*24 # The actions for both player(0~11) and the agent(12~23).
        num_steps = 0
        match_start_time = time.time()
        done = False 

        # Loop for the match 
        while not done:
            num_steps += 1
            if num_steps == 2:
                # Wait for 10 seconds for the players get ready
                print("Match starting in 10 seconds...")
                time.sleep(10)
            # Get the actions from the agent model
            action_agent, _states = model.predict(obs)
            action24[12:24] = action_agent
            # action24[0:12] = action_agent

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
            
            action24[0:12] = action_player
            # action24[12:24] = action_player
            
            # Perform the actions
            obs, reward, done, info = env.step(action24) 
            accumulated_reward += reward
        
        match_time = time.time() - match_start_time # in seconds
        
        print(f"Match ended")
        env.render(close = True)
        env.close()

        return info['behavior_metrics']

    def _update_playing_behavior_data(self, behavior_metrics):
        """
        Update the playing data based on the behavior metrics of the match.
        """
        self.playing_data["total_matches"] += 1

        num_matches = self.playing_data["total_matches"]

        # Update the average special moves per match
        average_special_moves_per_match = self.playing_data["average_special_moves_per_match"]
        total_special_moves = average_special_moves_per_match * (num_matches - 1) + behavior_metrics['special_moves']
        average_special_moves_per_match = total_special_moves / num_matches
        self.playing_data["average_special_moves_per_match"] = average_special_moves_per_match

        # Update the average reward per match
        # Calculate the score for this match according to the reward. Reward is from -0.35 to 0.35, map it to 0 to 100 as the score.
        match_score = (behavior_metrics['episode_reward'] + 0.35) * 100 / 0.7
        # make sure the score is between 0 and 100
        if match_score < 0:
            match_score = 0
        if match_score > 100:
            match_score = 100
        # Extract the average score per match from the playing data. It is a string like "50/100", extract the first part.
        average_score_per_match = int(re.search(r'\d+', self.playing_data["average_score_per_match"]).group())
        # Calculate the new average score per match
        total_score = average_score_per_match * (num_matches - 1) + match_score
        average_score_per_match = total_score / num_matches
        self.playing_data["average_score_per_match"] = f"{int(average_score_per_match)}/100"

        # Update the win streak and loss streak
        if behavior_metrics['win'] == 1:
            self.playing_data["total_wins"] += 1
            self.playing_data["current_win_streak"] += 1
            self.playing_data["current_loss_streak"] = 0
        else:
            self.playing_data["total_losses"] += 1
            self.playing_data["current_loss_streak"] += 1
            self.playing_data["current_win_streak"] = 0

        # Calculate win rate
        self.playing_data["win_rate"] = self.playing_data["total_wins"] / num_matches

    def _reset_playing_data(self):
        """
        Reset the playing data for a new round of matches.
        """
        self.playing_data = {
            "current_character": self.playing_data["current_character"],
            "total_matches": 0,
            "win_rate": 0,
            "total_wins": 0,
            "total_losses": 0,
            "current_win_streak": 0,
            "current_loss_streak": 0,
            "average_score_per_match": "0/100",
            "average_special_moves_per_match": 0,
            "faced_agents_times": {
                "projectile_type": 0,
                "special_move_type": 0,
                "defensive_type": 0,
                "aggressive_type": 0,
                "air_type": 0,
                "coward_type": 0,
                "newbie_type": 0,
                "key_spamming_type": 0
            },
            "faced_characters_times": {
                "Ryu": 0,
                "Ken": 0,
                "Chunli": 0,
                "Guile": 0,
                "Blanka": 0,
                "Zangief": 0,
                "Dhalsim": 0,
                "Balrog": 0,
                "Vega": 0,
                "Sagat": 0,
                "Bison": 0,
                "EHonda": 0,    
            },
            "the_last_opponents": {
                "type": "This is the first match, no last opponent.",
                "character": "This is the first match, no last opponent.",
                "model_path": "This is the first match, no last opponent.",
                "difficulty": "This is the first match, no last opponent."
            },
            "player's_feedback": {
                "First match, no feedback yet."
            }
        }
    
    REWARD_KWARGS = {
        'reward_scale': 0.001,
        'raw_reward_coef': 1.0, # How much HP change is rewarded, only happens during the fighting (not round over)
        'special_move_reward': 0.0, # Reward for using special moves
        'special_move_bonus': 1.0, # Bonus for dealing damage with special moves. The raw reward will be multiplied by this value, so set it to 1.0 if you don't want to use this feature.
        'projectile_reward': 0.0, # Reward for using projectiles
        'projectile_bonus': 1.0, # Bonus for dealing damage with projectiles. The raw reward will be multiplied by this value, so set it to 1.0 if you don't want to use this feature.
        'distance_reward': 0.0, # Set it to positive value to encourage the agent to stay far from the enemy, negative value to encourage the agent to stay close to the enemy
        'distance_bonus': 0.0, # Bonus for dealing damage with projectiles. 0.0 means no bonus, positive/negative value means reward being far/close to the enemy
        'in_air_reward': 0.00, # Reward for being in the air
        'time_reward_bonus': 0.0, # Bonus for the duration of the round. Set it to positive value to encourage the agent to stay longer in the round, negative value to encourage the agent to finish the round quickly
        'cost_coef': 0.0, # Ratio between the cost and the reward
        'special_move_cost': 2.0,
        'regular_attack_cost': 0.5,
        'jump_cost': 0.0,
        'vulnerable_frame_cost':0.00,
    }

    def make_env(self, state, players = 2 , reward_kwargs = REWARD_KWARGS, character_flip_rate = 0.0):
        def _init():
            env = retro.make(
                game=self._game, 
                state=state, 
                use_restricted_actions=retro.Actions.FILTERED,
                obs_type=retro.Observations.IMAGE,
                players=players
            )
            env = SFAIWrapper(
                env, 
                reset_round= True, 
                rendering = True, 
                rendering_interval= 0.02, 
                reward_kwargs=reward_kwargs, 
                character_flip_rate=character_flip_rate
            )
            env = MultiInputWrapper(env)
            env = EvaluationWrapper(env)
            return env
        return _init

       
        