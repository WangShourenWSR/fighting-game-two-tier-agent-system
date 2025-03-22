# In this file, we define the policy assessment function, which is used to assess the policy of the agent.
# The policy assessment function is used to assess the trained agent's policy by serveral metrics, such as the win rate, the average reward.
# The pipeline of the policy assessment is as follows:
# 1. Load the model to be assessed.
# 2. Initialize the following test tasks:
#    - Model vs Model, initialize the PvP environment using PvP .state file, and load the enemy model.
#    - Model vs Built-in AI, initialize the PvE environment using PvE .state file.
# 3. Run the 2 types of test tasks for a certain number of episodes. 
#    - In each episode of PvP task, gather the win rate, average reward for both the agent and the enemy.
#    - In each episode of PvE task, gather the average reward for the agent.
# 4. Return the gathered data. Update the model name according to the gathered data if needed.
# 
# CAN BE IMPROVED:
# Currently the assessment tasks are not parallelized. It can be parallelized to speed up the assessment process.
# But should use SubprocVecEnv to create the environments for parallelization.
# Still not clear how to implement. Can be implemented in the future.
# ==============================================================================================================
import os
import time
import numpy as np
import math

import retro
from environments.sfai_wrapper import SFAIWrapper

from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3 import PPO

from self_play.policy_selection import policy_selection_vanilla

# Define some constants as default values
TEST_EPISODES = 3 # The number of episodes to play
ENEMIES_MODEL_PATH = r"agent_models/policy_pool/assessment_enemies" # Specify the model directory
RENDERING = True
RENDERING_INTERVAL = -0.001
GAME = "StreetFighterIISpecialChampionEdition-Genesis"

DEFAULT_BUILT_IN_AI_STATES = [
    # "Champion.Level1.RyuVsGuile",
    # "Champion.Level2.RyuVsKen",
    "Champion.Level3.RyuVsChunli",
    # "Champion.Level4.RyuVsZangief",
    "Champion.Level5.RyuVsDhalsim",
    "Champion.Level6.RyuVsRyu",
    # "Champion.Level7.RyuVsHonda",
    "Champion.Level8.RyuVsBlanka",
    "Champion.Level9.RyuVsBalrog",
    # "Champion.Level10.RyuVsVega",
    "Champion.Level11.RyuVsSagat",
    "Champion.Level12.RyuVsBison"
]

DEFAULT_PVP_STATES = [
    "PvP.RyuVsRyu"
]

# Define the supported model classes
# TODO: This should be passed in but not hard coded
MODEL_CLASSES = {
    "PPO": PPO,
    "MultiInputPPO": PPO
}

# Define make_env function for creating the gymretro game environment
# Create the gymretro game environment
def make_env(game, state, players = 1):
    def _init():
        env = retro.make(
            game=game, 
            state=state, 
            use_restricted_actions=retro.Actions.FILTERED,
            obs_type=retro.Observations.IMAGE,
            players=players
        )
        # env = StreetFighterCustomWrapper(env, reset_round=True, rendering=True, reverse_env=False, render_interval= 0.1)
        env = SFAIWrapper(env, reset_round= True)
        return env
    return _init


def behavior_detection(info:dict = {}):
    '''
    This function is used to detect the behavior of the agent based on the change of information from the environment.
    The behavior includes the following:
    - Special moves: The agent performs special moves.
    - Projectiles: The agent performs projectiles.
    - Jumps: The agent performs jumps.
    - Regular attacks: The agent performs regular attacks.

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
    # Projectile gone status
    # status_projectile_gone = False
    # if info['agent_projectile_status'] != 34048 and prev_info['agent_projectile_status'] == 34048:
    #     status_projectile_gone = True
        
    # Regular attack status(regular attacks are light~heavy punches, light~heavy kicks)
    status_regular_attack = False
    if agent_status == 522 and prev_agent_status != 522:
        status_regular_attack = True

    # Jumping status
    status_jump = False
    if agent_status == 516 and prev_agent_status != 516:
        status_jump = True


    behavior = {
        "Special Move": status_special_move,
        "Projectile": status_projectile,
        "Regular Attack": status_regular_attack,
        "Jump": status_jump
    }

    return behavior

# Define the run_assessment_round function
# In this function, we run one round of game for assessment.
def run_assessment_round_vanilla(
        mode = "PvP",
        env = None, 
        model= None, 
        enemy_model = None,
        rendering = RENDERING,
        rendering_interval = RENDERING_INTERVAL
    ):
    '''
    This function is used to run one round of game for assessment, and return the statistics of the assessment.
    This is a simple version which only returns the win or not and the accumulated reward of the agent.
    TODO: Add more statistics to be returned.

    Parameters:
    - mode: The mode of the assessment. It should be one of 'PvP' or 'PvE'.
    - env: The environment to run the assessment.
    - model: The model to be assessed.
    - enemy_model: The enemy model for the assessment.
    - rendering: Whether to render the game pixels.
    - rendering_interval: The rendering interval.

    Returns:
    - win or not: Whether the agent wins the game.
    - accumulated_reward: The accumulated reward of the agent.
    '''
    # Check if the mode is one of 'PvP' or 'PvE', if not, raise an error
    if mode not in ["PvP", "PvE"]:
        raise ValueError("The mode should be one of 'PvP' or 'PvE'.")

    # Check if the environment is provided, if not, raise an error
    if env is None:
        raise ValueError("The environment should be provided.")
    
    # Check if the model is provided, if not, raise an error
    if model is None:
        raise ValueError("The model should be provided.")
    
    # Check if the enemy model is provided when the mode is 'PvP', if not, raise an error
    if mode == "PvP" and enemy_model is None:
        raise ValueError("The enemy model should be provided when the mode is 'PvP'.")
    
    # Initialize satistics
    accumulated_reward = 0
    win_or_not = False
    done = False

    # Run the PvP assessment round
    if mode == "PvP":
        # Initialize the action vector for the input of step() function.
        action24 = [0]*24 # The actions for both agent to be assessed(0~11) and the enemy agent(12~23).
        obs = env.reset()
        while not done:
            # Render the game according to the rendering flag
            if rendering:
                env.render()
                if rendering_interval > 0:
                    time.sleep(rendering_interval)
            # Get actions for step() function
            # Get the actions from the agent model
            action, _states = model.predict(obs)
            # Get the actions from the enemy model
            enemy_action, _ = enemy_model.predict(obs)
            # Merge the actions
            action24[0:12] = action
            action24[12:24] = enemy_action

            # Perform the actions
            obs, reward, done, info = env.step(action24)

            # Update the statistics
            accumulated_reward += reward
        
        # Check if the agent wins the game
        if info['enemy_hp'] < 0:
            win_or_not = True
        else:
            win_or_not = False


    

    # Run the PvE assessment round
    if mode == "PvE":
        # Initialize the action vector for the input of step() function.
        action12 = [0]*12 # The actions for the agent to be assessed.
        obs = env.reset()
        while not done:
            # Render the game according to the rendering flag
            if rendering:
                env.render()
                if rendering_interval > 0:
                    time.sleep(rendering_interval)
            # Get actions for step() function
            # Get the actions from the agent model
            action, _states = model.predict(obs)

            # Perform the actions
            obs, reward, done, info = env.step(action)

            # Update the statistics
            accumulated_reward += reward


        # Check if the agent wins the game
        if info['enemy_hp'] < 0:
            win_or_not = True
        else:
            win_or_not = False
        

    
    return win_or_not, accumulated_reward

def run_assessment_round_standard(
        env = None, 
        model= None, 
        rendering = RENDERING,
        rendering_interval = RENDERING_INTERVAL
    ):
    '''
    This function is used to run one round of game for assessment, and return the statistics of the assessment.
    This is the standard version of the assessment round which provides the following features:
    1. The test task is arcade mode in the game, which is the built-in AI. Provides a more stable and consistent assessment.
    2. Provide the fundamental statistics: win or not, accumulated rewar.
    3. Provide the behavior statistics: number of special moves, number of projectiles, number of jumps, number of regular attask, average distance to the enemy.

    Parameters:
    - env: The environment to run the assessment.
    - model: The model to be assessed.
    - rendering: Whether to render the game pixels.
    - rendering_interval: The rendering interval.

    Returns:
    - win or not: Whether the agent wins the game.
    - accumulated_reward: The accumulated reward of the agent.
    - behavior_statistics: The behavior statistics of the agent.
    '''
    # Check iif the required parameters are provided, if not, raise an error
    if env is None:
        raise ValueError("The environment should be provided.")
    if model is None:
        raise ValueError("The model should be provided.")
    
    # Initialize the statistics
    # Fundamental statistics
    accumulated_reward = 0
    win_or_not = False
    # Behavior statistics
    num_special_moves = 0
    num_projectiles = 0
    num_jumps = 0
    num_regular_attacks = 0
    total_distance = 0
    num_steps = 0

    # Run the assessment round
    # Initialization
    done = False
    action12 = [0]*12 # The actions for the agent to be assessed.
    obs = env.reset()
    while not done:
        # Render the game according to the rendering flag
        if rendering:
            env.render()
            if rendering_interval > 0:
                time.sleep(rendering_interval)

        # Get the actions from the agent model and perform the actions
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        num_steps += 1

        # Update the statistics
        # reward:
        accumulated_reward += reward
        # distance:
        total_distance += abs(info['agent_x'] - info['enemy_x'])
        # behavior:
        # Check the returned dictionary 'behavior' and update the behavior statistics
        behavior = behavior_detection(info)
        num_special_moves += behavior["Special Move"]
        num_projectiles += behavior["Projectile"]
        num_jumps += behavior["Jump"]
        num_regular_attacks += behavior["Regular Attack"]

    
    # Calculate the statistics according to the gathered data
    # average distance:
    average_distance = total_distance / num_steps
    # Check if the agent wins the game: 1. The enemy's hp is less than 0. 2. time is up and the agent's hp is larger than the enemy's hp.
    if info['enemy_hp'] < 0:
        win_or_not = True
    elif info['round_countdown'] <= 0 and info['agent_hp'] > info['enemy_hp']:
        win_or_not = True
    else:
        win_or_not = False

    # Construct the behavior statistics
    behavior_statistics = {
        "Special Moves": num_special_moves,
        "Projectiles": num_projectiles,
        "Jumps": num_jumps,
        "Regular Attacks": num_regular_attacks,
        "Average Distance": average_distance
    }

    return win_or_not, accumulated_reward, behavior_statistics


# Define the policy assessment function
def policy_assessment(
        game = GAME,
        make_env: callable = make_env,
        assessed_model = None,
        assessment_type = "Standard",
        assessment_batch_size = 16,
        pvp_states = DEFAULT_PVP_STATES,
        enemies_model_path = ENEMIES_MODEL_PATH,
        num_pvp_tests = 3,
        built_in_ai_states = DEFAULT_BUILT_IN_AI_STATES,
        num_pve_tests = 3,
        test_episodes = TEST_EPISODES,
        rendering = RENDERING,
        rendering_interval = RENDERING_INTERVAL,
        *args, 
        **kwargs
):
    '''
    This function is used to assess the policy of the agent.
    The policy assessment function is used to assess the trained agent's policy by serveral metrics, such as the win rate, the average reward.
    The pipeline of the policy assessment is as follows:
    1. Load the model to be assessed.
    2. Initialize the following test tasks:
       - Model vs Model, initialize the PvP environment using PvP .state file, and load the enemy model.
       - Model vs Built-in AI, initialize the PvE environment using PvE .state file.
    3. Run the 2 types of test tasks for a certain number of episodes.
         - In each episode of PvP task, gather the win rate, average reward for both the agent and the enemy.
         - In each episode of PvE task, gather the average reward for the agent.
    4. Return the gathered data. Including enemy models' statistics.

    Parameters:
    - assessed_model: The model to be assessed.
    - assessment_type: The type of the assessment. It should be one of 'PvP', 'PvE' or 'Hybrid'.
    - assessment_batch_size: The batch size for the PvP mode assessment. Due to the graphic memory limitation, the enemy models are loaded and assessed in batches.
    - pvp_states: The PvP states for Model vs Model test. The state indicates the player 1, player 2, and the stage.
    - enemies_model_path: The path of the enemy models for Model vs Model test.
    - num_pvp_tests: The number of PvP tests (fight against model) to be conducted.
    - built_in_ai_states: The PvE states for Model vs Built-in AI test. The state indicates the player 1, the built-in AI enemy, and the stage.
    - num_pve_tests: The number of PvE tests (fight against built-in AI) to be conducted.
    - test_episodes: The number of episodes to play per test task.
    - rendering: Whether to render the environment.

    Returns:
    - win_rate: The total win rate of the agent.
    - average_reward: The average reward of the agent.
    - enemy_models_win_rate: The win rate of the enemy models.
    '''

    # Check if the model to be assessed is provided, if not, raise an error
    # Also, check if it is a subclass of OnPolicyAlgorithm or OffPolicyAlgorithm of stable_baselines3, if not raise an error.
    if assessed_model is None:
        raise ValueError("The model to be assessed must be provided.")
    if not isinstance(assessed_model, OnPolicyAlgorithm) and not isinstance(assessed_model, OffPolicyAlgorithm):
        raise ValueError("The model to be assessed should be a subclass of OnPolicyAlgorithm or OffPolicyAlgorithm of stable_baselines3.")
    
    # Check if the provided assessment type is valid, if not, raise an error.
    # It should be one of 'PvP', 'PvE' or 'Hybrid'
    if assessment_type not in ["PvP", "PvE", "Hybrid", "Standard"]:
        raise ValueError("The assessment type should be one of 'PvP', 'PvE', 'Hybrid' or 'Standard'.")
    
    # Have to initialize the enemy_models_names, otherwise if the assessment_type is not 'PvP' or 'Hybrid', the enemy_models_names will be referenced before assignment.
    enemy_models_names = []

    # Set the Model vs Model tasks's enemy models and environments
    if assessment_type in ["PvP", "Hybrid"]:
        # Load the enemy models
        # The parameter 'enemies_model_path' provides the path to the pool of enemy models.
        # Select the num_pvp_tests enemy models with the highest win rate under the 'enemies_model_path' directory for the Model vs Model test.
        # The name of the enemy_models should be in the format of 'ww_rr.zip', where 'ww' is the win rate and 'rr' is the reward.
        
        # Select the enemy models based on the win rate
        enemy_models_names, _ , enemy_model_class = policy_selection_vanilla(enemies_model_path, top_n=num_pvp_tests)
       
        # Retrive the enemy model class from the MODEL_CLASSES dictionary
        # TODO: Currently only support one enemy model class. Can be extended to support multiple enemy model classes.
        enemy_model_class = MODEL_CLASSES.get(enemy_model_class[0], None)
        # Check if the enemy_model_class is supported, if not, raise an error
        if enemy_model_class is None:
            raise ValueError("The enemy model class is not supported.")
        if not issubclass(enemy_model_class, BaseAlgorithm) and not issubclass(enemy_model_class, AuxObjPPO):
            raise ValueError("The enemy model class should be a subclass of stable_baselines3.common.policies.BasePolicy or AuxObjPPO.")

        # Load the names of the enemy models
        enemy_model_paths = [os.path.join(enemies_model_path, model) for model in enemy_models_names]

        # Set the PvP environment
        # Randomly select num_pvp_tests PvP states from the default PvP states. 
        # Be aware of that, num_pvp_tests might be larger than the number of default PvP states. If so, just repeat the default PvP states.
        if num_pvp_tests > len(pvp_states):
            # Repeat the default PvP states to make the number of PvP states equal to num_pvp_tests.
            pvp_states = pvp_states * (num_pvp_tests // len(pvp_states)) + pvp_states[:num_pvp_tests % len(pvp_states)]
        else:
            # Randomly select num_pvp_tests PvP states from the default PvP states by shuffling the default PvP states.
            shuffled_pvp_states = pvp_states.copy()
            np.random.shuffle(shuffled_pvp_states)
            pvp_states = shuffled_pvp_states[:num_pvp_tests]

        # Initialize the PvP environments.
        pvp_envs = [make_env(game=game, state=state, players=2) for state in pvp_states]

    # Set the Model vs Built-in AI tasks' environments
    if assessment_type in ["PvE", "Hybrid", "Standard"]:
        # Randomly select num_pve_tests PvE states from the default PvE states.

        # For 'Standard' assessment, set num_pve_tests = len(built_in_ai_states):
        if assessment_type == "Standard":
            num_pve_tests = len(built_in_ai_states)

        # Be aware of that, num_pve_tests might be larger than the number of default PvE states. If so, just repeat the default PvE states.
        if num_pve_tests > len(built_in_ai_states):
            # Repeat the default PvE states to make the number of PvE states equal to num_pve_tests.
            built_in_ai_states = built_in_ai_states * (num_pve_tests // len(built_in_ai_states)) + built_in_ai_states[:num_pve_tests % len(built_in_ai_states)]
        else:
            # Randomly select num_pve_tests PvE states from the default PvE states by shuffling the default PvE states.
            shuffled_built_in_ai_states = built_in_ai_states.copy()
            np.random.shuffle(shuffled_built_in_ai_states)
            built_in_ai_states = shuffled_built_in_ai_states[:num_pve_tests]

        # Initialize the PvE environments.
        pve_envs = [make_env(game = game, state= state, players=1) for state in built_in_ai_states]


    # Initialize the PvP statistics
    pvp_round_results = [] # Win or lose of each PvP round
    pvp_rewards = [] # Accumulated reward of each PvP round
    # Make a dictionary to store the win rate of each enemy model. The key is the enemy model's name, and the value is list of result of each round.
    enemy_models_round_results = {} # The win rate of each enemy model
    # Initialize the dictionary using file names in enemy_models_names
    for model_name in enemy_models_names:
        enemy_models_round_results[model_name] = []

    # Initialize the PvE statistics
    pve_round_results = [] # Win or lose of each PvE round
    pve_rewards = [] # Accumulated reward of each PvE round

    # Initialize the behavior statistics
    average_behavior_statistic = {}
    

    # Run the PvP assessment
    if assessment_type in ["PvP", "Hybrid"]:
        num_batchs = math.ceil(len(enemy_model_paths) / assessment_batch_size)
        for batch_idx in range(num_batchs):
            enemy_models = [enemy_model_class.load(model) for model in enemy_model_paths[batch_idx * assessment_batch_size: (batch_idx + 1) * assessment_batch_size]]
            for i in range(len(enemy_models)):
                # Run one round of PvP assessment
                pvp_env = pvp_envs[i]()
                # Run the assessment round for test_episodes times
                for _ in range(test_episodes):
                    # Run the assessment round
                    pvp_round_result, pvp_reward = run_assessment_round_vanilla(mode="PvP", env=pvp_env, model=assessed_model, enemy_model=enemy_models[i], rendering=rendering, rendering_interval=rendering_interval)
                    # Gather the statistics for the model to be assessed
                    pvp_round_results.append(pvp_round_result)
                    pvp_rewards.append(pvp_reward)
                    # Update the enemy models' statistics, the rould result is opposite to the agent's round result
                    enemy_models_round_results[enemy_models_names[batch_idx * assessment_batch_size + i]].append(not pvp_round_result)
                    # enemy_models_round_results[enemy_models_names[i]].append(not pvp_round_result)
                
                # Close the environment after each assessment task ends
                if rendering:
                    # BE AWARE: This function must be called before the close() function. otherwise, the rendering window will not be closed.
                    pvp_env.render(close = True) 
                pvp_env.close()


    # Run the PvE assessment
    if assessment_type in ["PvE", "Hybrid"]:
        for i in range(num_pve_tests):
            # Run one round of PvE assessment
            pve_env = pve_envs[i]()
            # Run the assessment round for test_episodes times
            for _ in range(test_episodes):
                pve_round_result, pve_reward = run_assessment_round_vanilla(mode="PvE", env=pve_env, model=assessed_model, rendering=rendering, rendering_interval=rendering_interval)
                pve_round_results.append(pve_round_result)
                pve_rewards.append(pve_reward)

            # Close the environment after each assessment task ends
            if rendering:
                # BE AWARE: This function must be called before the close() function. otherwise, the rendering window will not be closed.
                pve_env.render(close = True) 
            pve_env.close()

    elif assessment_type == "Standard":
        for i in range(num_pve_tests):
            pve_env = pve_envs[i]()
            # Run the assessment round for test_episodes times
            for _ in range(test_episodes):
                pve_round_result, pve_reward, behavior_statistics = run_assessment_round_standard(env=pve_env, model=assessed_model, rendering=rendering, rendering_interval=rendering_interval)
                pve_round_results.append(pve_round_result)
                pve_rewards.append(pve_reward)
                
                for key, value in behavior_statistics.items():
                    # if no such a key in the behavior statistics, append it to the behavior statistics
                    if key not in average_behavior_statistic:
                        average_behavior_statistic[key] = 0
                        average_behavior_statistic[key] += value
                    else:
                        average_behavior_statistic[key] += value

            # Close the environment after each assessment task ends
            if rendering:
                # BE AWARE: This function must be called before the close() function. otherwise, the rendering window will not be closed.
                pve_env.render(close = True) 
            pve_env.close()
        
        # Calculate the average behavior statistics
        for key in average_behavior_statistic.keys():
            average_behavior_statistic[key] = average_behavior_statistic[key] / (num_pve_tests * test_episodes)

    # Caclulate the statistics according to the gathered data
    # Calculate the total win rate of PvP and PvE assessments.
    # Check the length, to avoid the zero division error. If the length is zero, set the win rate of that part to 0.
    win_rate = (sum(pvp_round_results) + sum(pve_round_results)) / (len(pvp_round_results) + len(pve_round_results))
    # Calculate the average reward of PvP and PvE assessments.
    average_reward = (sum(pvp_rewards) + sum(pve_rewards)) / (len(pvp_rewards) + len(pve_rewards))
    # Calculate the PvP assessment win rate
    if len(pvp_round_results) == 0:
        pvp_win_rate = 0
    else:
        pvp_win_rate = sum(pvp_round_results) / len(pvp_round_results)
    # Calculate the PvE assessment win rate
    if len(pve_round_results) == 0:
        pve_win_rate = 0
    else:
        pve_win_rate = sum(pve_round_results) / len(pve_round_results)

    # Calculate the win rate of each enemy model by averaging the results of each round for each enemy model.
    enemy_models_win_rate = {}
    for model_name in enemy_models_names:
        enemy_models_win_rate[model_name] = sum(enemy_models_round_results[model_name]) / len(enemy_models_round_results[model_name])
    
    # Return the gathered data
    return win_rate, pvp_win_rate, pve_win_rate, average_reward, enemy_models_win_rate, average_behavior_statistic



    
# # Unit test code 
# # To test the function, uncomment the following code and import the function in the unit_test.py file and run the unit_test.py file.
# # Load the model to be assessed
# model_path = r"agent_models/policy_pool_ppo/600000_steps_%PPO%self_play_PPO_spmv_6layerA2C_entropy$w_0.5333333333333333_r_0.3246496564026966$.zip"
# from stable_baselines3 import PPO
# model = PPO.load(model_path)

# # Run the policy assessment
# start_time = time.time()
# win_rate, win_rate_pvp, win_rate_pve, average_reward, enemy_models = policy_assessment(
#     assessed_model=model,
#     assessment_type="Hybrid",
#     pvp_states=DEFAULT_PVP_STATES,
#     enemies_model_path=r"agent_models/policy_pool_ppo",
#     num_pvp_tests=2,
#     built_in_ai_states=DEFAULT_BUILT_IN_AI_STATES,
#     num_pve_tests=2,
#     test_episodes=2,
#     rendering=True,
#     rendering_interval=0.00
# )
# end_time = time.time()
# # Print the result
# print(f"Win rate: {win_rate}")
# print(f"Average reward: {average_reward}")
# print("Enemy models' win rate:")
# for model, win_rate in enemy_models.items():
#     print(f"Model: {model}, Win rate: {win_rate}")

# print("Time cost: ", end_time - start_time)





