# Use this file to test the code implementation. 
# Running the code in some directories directly may cause some errors.
# Just import the .py file and run the functions in the file.

# Write the code in the file you want to test, and import it or uncomment the code below to test the code implementation.

import self_play.policy_assessment
import configs.config_projectile_only as config
from self_play.self_play import SelfPlay

from stable_baselines3 import PPO
import os

if __name__ == "__main__":
    self_play = SelfPlay(
        config = config
    )
    ENEMY_MODEL_DIR = r"agent_models/policy_pool/Projectile_Only"
    ENEMY_MODEL_NAME = r'%MultiInputPPO%_initial_enemy_policy$w_0.5_r_0.0$'
    # ENEMY_MODEL_DIR = r"agent_models/trained_models"
    # ENEMY_MODEL_NAME = r'14400000_steps_%PPO%ppo_entCoef0.1_6layersAC$w_0.26666666666666666_r_0.20879597564521116$'
    agent_model = PPO.load(os.path.join(ENEMY_MODEL_DIR, ENEMY_MODEL_NAME))
    # self_play._evaluate_whole_policy_pool(
    #     agent_model,
    #     self_play.evaluation_kwargs
    # )
    win_rate, pvp_win_rate, pve_win_rate, average_reward, enemy_models_win_rate, average_behavior_statistic = self_play._evaluate_agent(
        agent_model = agent_model,
        evaluation_kwargs = config.EVALUATION_KWARGS
    )

    print("Evaluation results:")
    print(f"Win rate: {win_rate}")
    print(f"PvP win rate: {pvp_win_rate}")
    print(f"PvE win rate: {pve_win_rate}")
    print(f"Average reward: {average_reward}")
    for key, value in average_behavior_statistic.items():
        print("Average {} per round: {}".format(key, value))