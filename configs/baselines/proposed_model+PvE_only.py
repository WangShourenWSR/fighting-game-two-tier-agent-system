# The config file is used to store constants and default values that are used throughout the self-play training process. 

# The settings can only be modified by changing the values in the config file, which helps to maintain consistency and avoid hardcoding values in the code.
# ==============================================================================================================
from stable_baselines3 import PPO
from models.custom_models import CustomCNN, CustomResNet18,CustomCombinedExtractor_LSTM
import os
from pathlib import Path
import random

# Generate a random seed
SEED = random.randint(0, 10000)
# Name of the task
# For convenience, the TASK_NAME will be set as same name as this configuration file.
TASK_NAME = Path(__file__).stem

# If enable resume training, if set to True, the training will start with checking if there are models in the POLICY_POOL_DIR, 
# if there are, the training will resume from the latest model.
ENABLE_RESUME_TRAINING = True

# Define some constants for the experiment settings
LOG_DIR = r"logs"

# Define some constants for the self-play settings
GAME = "StreetFighterIISpecialChampionEdition-Genesis"
# POLICY_POOL_DIR = r"agent_models/policy_pool/"
POLICY_POOL_DIR = os.path.join(r"agent_models/policy_pool", TASK_NAME)
GAME_STATES_PVE = [
        "PvE.KenVsKen",
        "PvE.KenVsRyu",
        "PvE.KenVsDhalsim",
        "PvE.DhalsimVsKen",
        "PvE.DhalsimVsRyu",
        "PvE.DhalsimVsSagat",
        "PvE.SagatVsKen",
        "PvE.SagatVsDhalsim",
        "PvE.SagatVsSagat",
        "Champion.Level9.RyuVsBalrog",
        "Champion.Level10.RyuVsVega",
        "Champion.Level11.RyuVsSagat",
        "PvE.RyuVsKen",
        "PvE.RyuVsRyu",
    ]

GAME_STATES_PVP = [
        "PvP.KenVsKen",
    ]

GAME_STATES_EVAL =[
        "Champion.Level12.RyuVsBison",
        "Champion.Level5.RyuVsDhalsim",
        "Champion.Level6.RyuVsRyu",
        "Champion.Level4.RyuVsZangief",
        "Champion.Level3.RyuVsChunli",
        "Champion.Level2.RyuVsKen",
        "Champion.Level1.RyuVsGuile",
        "PvE.KenVsHonda",
        "PvE.KenVsSagat",
        "PvE.SagatVsRyu",
        "PvE.DhalsimVsDhalsim",
        "PvE.ChunliVsRyu",
        "PvE.HondaVsChunli",
    ]
# BE AWARE: MODEL CLASS and POLICY CLASS must share the same key.
MODEL_CLASSES = {
    "PPO": PPO,
    "MultiInputPPO": PPO
}
POLICY_CLASSES = {
    "PPO": "CnnPolicy",
    "MultiInputPPO": "MultiInputPolicy"
}

ENEMY_POLICY_SELECTION_METHODS = [
    "All",
    "Random"
]

# Set the logging parameters
LOG2FILE = False
NETWORK_GRAPH_TENSORBOARD = False
CHECKPOINT_INTERVAL = 500000

# Set the training parameters
RENDERING = False
CHARACTER_FLIP_RATE = 0.0
MAX_ITERATIONS = 50
STEPS_PER_ITERATION = 5000000
PVP_RATIO = 0.0
ENEMY_POLICY_SELECTION = 'All' # Select top 7 enemies, can be an integer or 'All'
POLICY_POOL_UPDATE_METHOD = 'All' # Maintain top 15 agents in the policy pool
NUM_ENVS = 12
STICKY_ACTION_MODE = False
STICKINESS = 0.0

# Set the reward function parameters
REWARD_FUNCTION_INDEX = 0  # index for this list: [sfai_reward, default_reward, naruto_reward, reward_with_cost, cost_spmv_preferred_reward, cost_projectile_preferred_reward, projectile_only_reward]
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
# Set the model parameters
AGENT_MODEL_CLASS = "MultiInputPPO"
AGENT_ENT_COEF = 0.01
AGENT_VF_COEF = 1.0
AGENT_FEATURES_EXTRACTOR_KWARGS = {
    # CNN parameters for game pixels
    "cnn_output_dim": 128,
    "normalize_images": False,
    # LSTM(RNN) parameters for action sequence 
    "lstm_input_dim": 12,
    "lstm_hidden_dim": 64,
    "num_layers": 2,
    # Extra parameters
}
AGENT_POLICY_KWARGS ={
    "features_extractor_class": CustomCombinedExtractor_LSTM,
    "features_extractor_kwargs": AGENT_FEATURES_EXTRACTOR_KWARGS,
    "net_arch": dict(pi=[512, 256, 128, 128], vf=[512, 256, 128, 128])
}

# Set evaluation parameters
EVALUATION_KWARGS = {
    'eval_callback_freq': 5000, 
    'assessment_type': 'Standard',
    'test_episodes': 2,
    'rendering': False
}


from stable_baselines3.common.base_class import BaseAlgorithm
from environments.sfai_wrapper import SFAIWrapper
from environments.multi_input_wrapper import MultiInputWrapper
from environments.self_play_wrapper import SelfPlayWrapper
from stable_baselines3.common.monitor import Monitor
from environments.self_play_wrapper import SelfPlayWrapper
from environments.evaluation_wrapper import EvaluationWrapper
import retro

# Define the make_env function
def make_env(
        game: str, 
        state: str, 
        rendering: bool = RENDERING,
        enemy_model_path: str = None, 
        enemy_model_class: BaseAlgorithm = None,
        seed: int = SEED, 
        sticky_action_mode: bool = False, 
        stickiness: float = 0.0,
        players: int = 1
    ):
    def _init():
        # Create the gymretro game environment
        env = retro.make(
            game=game, 
            state=state, 
            use_restricted_actions=retro.Actions.FILTERED, 
            obs_type=retro.Observations.IMAGE,    
            players=players
        )
        # Apply the SFAIWrapper to the environment
        env = SFAIWrapper(env, rendering=rendering, rendering_interval=-0.1, sticky_action_mode=sticky_action_mode, stickiness=stickiness, reward_function_idx=REWARD_FUNCTION_INDEX, reward_kwargs=REWARD_KWARGS, character_flip_rate=CHARACTER_FLIP_RATE)
        env = MultiInputWrapper(env)
        # If making a PvP environment, apply the SelfPlayWrapper to the environment
        if players == 2:
            env = SelfPlayWrapper(env, enemy_model_path=enemy_model_path, enemy_model_class=enemy_model_class)
        env = Monitor(env)
        env.seed(seed)
        return env
    return _init

def make_env_eval(game, state, players = 1):
    def _init():
        env = retro.make(
            game=game, 
            state=state, 
            use_restricted_actions=retro.Actions.FILTERED,
            obs_type=retro.Observations.IMAGE,
            players=players
        )
        # env = StreetFighterCustomWrapper(env, reset_round=True, rendering=True, reverse_env=False, render_interval= 0.1)
        env = SFAIWrapper(env, reset_round= True, rendering = EVALUATION_KWARGS.get('rendering', False))
        env = EvaluationWrapper(env)
        env = MultiInputWrapper(env)
        env = Monitor(env)
        return env
    return _init
