# The config file is used to store constants and default values that are used throughout the self-play training process. 
# The settings can only be modified by changing the values in the config file, which helps to maintain consistency and avoid hardcoding values in the code.
# TODO: ent_coef should be a parameter in config.py
# ==============================================================================================================
from stable_baselines3 import PPO
from models.aux_obj_ppo import AuxObjPPO
from models.custom_models import CustomCNN, CustomResNet18, CustomResNet50, AuxObjActorCriticCnnPolicy, CustomCombinedExtractor_ResNet18
from stable_baselines3.common.policies import MultiInputActorCriticPolicy
from stable_baselines3.common.torch_layers import CombinedExtractor
import os

# Name of the task
TASK_NAME = "default_HP_based_reward"
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
        # "Champion.Level6.RyuVsRyu",
        # "Champion.Level5.RyuVsDhalsim",
        # "Champion.Level3.RyuVsChunli",
        "Champion.Level9.RyuVsBalrog",
        # "Champion.Level11.RyuVsSagat",
        # "Champion.Level2.RyuVsKen",
        "Champion.Level12.RyuVsBison"
    ]

GAME_STATES_PVP = [
        "PvP.RyuVsRyu",
        "PvP.RyuVsChunli",
        "PvP.RyuVsEHonda",
        "PvP.ChunliVsChunli",
        "PvP.ChunliVsEHonda",
        "PvP.ChunliVsRyu",
        "PvP.EHondaVsEHonda",
        "PvP.EHondaVsRyu",
        "PvP.EHondaVsChunli"    
    ]
# BE AWARE: MODEL CLASS and POLICY CLASS must share the same key.
MODEL_CLASSES = {
    "AuxObjPPO": AuxObjPPO,
    "PPO": PPO,
    "MultiInputPPO": PPO
}
POLICY_CLASSES = {
    "AuxObjPPO": AuxObjActorCriticCnnPolicy,
    "PPO": "CnnPolicy",
    "MultiInputPPO": "MultiInputPolicy"
}
FEATURES_EXTRACTOR_CLASSES = {
    "CustomCNN": CustomCNN,
    "CustomResNet18": CustomResNet18,
    "CustomResNet50": CustomResNet50,
    "CombinedExtractor" : CombinedExtractor,
    "CustomCombinedExtractor_ResNet18": CustomCombinedExtractor_ResNet18
}
AC_ARCHITECTURES = {
    "custom_4_layers": dict(pi=[512, 256, 128, 128], vf=[512, 256, 128, 128]),
    "custom_6_layers": dict(pi=[512, 256, 256, 128, 128, 128], vf=[512, 256, 256, 128, 128, 128]),
    "custom_6_layers_2": dict(pi=[512, 1024, 1024, 512, 256, 128], vf=[512, 1024, 1024, 512, 256, 128]),
    "custom_8_layers": dict(pi=[512, 512, 256, 256, 128, 128, 128, 64], vf=[512, 512, 256, 256, 128, 128, 128, 64])

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
CHARACTER_FLIP_RATE = 0.5
MAX_ITERATIONS = 50
STEPS_PER_ITERATION = 500000
PVP_RATIO = 0.7
ENEMY_POLICY_SELECTION = 'All' # Select top 7 enemies, can be an integer or 'All'
POLICY_POOL_UPDATE_METHOD = 'All' # Maintain top 15 agents in the policy pool
NUM_ENVS = 8
STICKY_ACTION_MODE = False
STICKINESS = 0.0

# Set the reward function parameters
REWARD_FUNCTION_INDEX = 0  # index for this list: [sfai_reward, default_reward, naruto_reward, reward_with_cost, cost_spmv_preferred_reward, cost_projectile_preferred_reward, projectile_only_reward]
REWARD_KWARGS = {
    'raw_reward_coef': 1.0, # How much HP change is rewarded, only happens during the fighting (not round over)
    'special_move_reward': 0.0, # Reward for using special moves
    'special_move_bonus': 0.0, # Bonus for dealing damage with special moves. The raw reward will be multiplied by this value.
    'projectile_reward': 0.0, # Reward for using projectiles
    'projectile_bonus': 0.0, # Bonus for dealing damage with projectiles. The raw reward will be multiplied by this value.
    'cost_coef': 0.0, # Ratio between the cost and the reward
    'special_move_cost': 0.0,
    'regular_attack_cost': 0.0,
    'jump_cost': 0.0,
}


# Set the model parameters
AGENT_MODEL_CLASS = "MultiInputPPO"
AGENT_FEATURES_EXTRACTOR_CLASS = "CustomCombinedExtractor_ResNet18"
AGENT_AC_ARCHITECTURE = "custom_6_layers_2"
AGENT_FEATURE_DIM = 128
AGENT_NUM_REGR_AUX_HEADS = 5
AGENT_NUM_CLSF_AUX_HEADS = 8
AGENT_ENT_COEF = 0.00

# Set evaluation parameters
EVALUATION_KWARGS = {
    'eval_callback_freq': 1000, 
    'assessment_type': 'Standard',
    'test_episodes': 2,
    'rendering': False
}


from stable_baselines3.common.base_class import BaseAlgorithm
from environments.sfai_wrapper import SFAIWrapper
from environments.multi_input_wrapper import MultiInputWrapper
from environments.self_play_wrapper import SelfPlayWrapper
from environments.evaluation_wrapper import EvaluationWrapper
from stable_baselines3.common.monitor import Monitor
import retro

# Define the make_env function
def make_env(
        game: str, 
        state: str, 
        rendering: bool = RENDERING,
        enemy_model_path: str = None, 
        enemy_model_class: BaseAlgorithm = None,
        seed: int = 0, 
        sticky_action_mode: bool = False, 
        stickiness: float = 0.0,
        players: int = 2
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
        env = SFAIWrapper(env, reset_round= True, rendering = True, rendering_interval=0)
        env = EvaluationWrapper(env)
        env = MultiInputWrapper(env)
        env = Monitor(env)
        return env
    return _init