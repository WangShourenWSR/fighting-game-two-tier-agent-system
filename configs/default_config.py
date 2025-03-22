# The config file is used to store constants and default values that are used throughout the self-play training process. 
# The settings can only be modified by changing the values in the config file, which helps to maintain consistency and avoid hardcoding values in the code.
# ==============================================================================================================
from stable_baselines3 import PPO
from models.aux_obj_ppo import AuxObjPPO
from models.custom_models import CustomCNN, CustomResNet18, CustomResNet50, AuxObjActorCriticCnnPolicy

# Define some constants for the experiment settings
LOG_DIR = r"logs/self_play"
SAVE_DIR = r"agent_models/self_play"

# Define some constants for the self-play settings
GAME = "StreetFighterIISpecialChampionEdition-Genesis"
POLICY_POOL_DIR = r"agent_models/policy_pool"
GAME_STATES_PVE = [
        "Champion.Level6.RyuVsRyu",
        "Champion.Level5.RyuVsDhalsim",
        "Champion.Level3.RyuVsChunLi",
        "Champion.Level11.RyuVsSagat",
        "Champion.Level2.RyuVsKen"
    ]

GAME_STATES_PVP = [
        "PvP.RyuVsRyue"
    ]

GAME_STATES_EVAL =[
        "Champion.Level6.RyuVsRyu",
        "Champion.Level5.RyuVsDhalsim",
        "Champion.Level12.RyuVsBison"
    ]
FEATURES_EXTRACTOR_CLASSES = {
    "CustomCNN": CustomCNN,
    "CustomResNet18": CustomResNet18,
    "CustomResNet50": CustomResNet50
}
MODEL_CLASSES = {
    "AuxObjPPO": AuxObjPPO,
    "PPO": PPO
}
POLICY_CLASSES = {
    "AuxObjPPO": AuxObjActorCriticCnnPolicy,
    "PPO": "CnnPolicy"
}
ENEMY_POLICY_SELECTION_METHODS = [
    "All",
    "Random"
]
AC_ARCHITECTURES = {
    "custom_4_layers": dict(pi=[512, 256, 128, 128], vf=[512, 256, 128, 128]),
}
EVALUATION_KWARGS = {
    'num_pvp_tests': 5,
    'num_pve_tests': 5,
    'test_episodes': 3,
    'rendering': False
}

# Define some constants as default values
NUM_ENVS = 4
RENDERING = True
CHECKPOINT_INTERVAL = 50000