# Self-play trainning class
# Author: Shouren Wang
# Using config file to initialize the settings is recommended.

#NOTE: If you want ot add new arguments, in addition to where it's defined/used, please also add them to the following positions:
#         1. Add the arguments to the __init__ function arguments.
#         2. Add the arguments to the class attributes set by config file.
#         3. Add the arguments to the class attributes set by passed-in arguments.
#         4. Add the arguments to the check function.
#5. Add the arguments to the print information in the initialization function.

# TODO: Print the policy pool updates after evaluation in each iteration1
# ==============================================================================================================
import os
import sys
import time
import argparse
from typing import Union, Dict, Any, Optional
from types import ModuleType

import numpy as np
import torch as th
from torch.utils.tensorboard import SummaryWriter
import retro
import gym

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.torch_layers import NatureCNN
from stable_baselines3.common.base_class import BaseAlgorithm

from environments.sfai_wrapper import SFAIWrapper
from environments.self_play_wrapper import SelfPlayWrapper
from utils.extended_checkpoint_callback import GraphCheckpointCallback
from models.custom_models import CustomCNN, CustomResNet18, CustomResNet50

from self_play.policy_selection import policy_selection_vanilla
from self_play.policy_pool_update import update_policy_pool_all, update_policy_pool_top_n
from self_play.self_play_environment_manager import SelfPlayEnvironmentManager
from self_play.self_play_agent_manager import SelfPlayAgentManager
from self_play.policy_assessment import policy_assessment
from self_play.evaluation import SFAIEvalCallback

from utils.str2bool import str2bool

# Ignore the UserWarning
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# import the config file
from self_play.default_config import *
def default_make_env(
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
        env = retro.make(
            game=game, 
            state=state, 
            use_restricted_actions=retro.Actions.FILTERED, 
            obs_type=retro.Observations.IMAGE,    
            players=players
        )
        # Apply the SFAIWrapper to the environment
        env = SFAIWrapper(env, rendering=rendering, rendering_interval=-0.1, sticky_action_mode=sticky_action_mode, stickiness=stickiness)
        # If making a PvP environment, apply the SelfPlayWrapper to the environment
        if players == 2:
            env = SelfPlayWrapper(env, enemy_model_path=enemy_model_path, enemy_model_class=enemy_model_class)
        env = Monitor(env)
        env.seed(seed)
        return env
    return _init


class SelfPlay:
    def __init__(
        self,
        task_name: str = None,
        config: ModuleType = None,
    ):
        '''
        Initialize the self-play trainer.
        :param task_name (str): The name of the task
        :param config (ModuleType): RECOMMENDED. The config module, a .py file that contains the task settings.
        :param enable_resume_training (bool): If true, resume the training from the latest model in the policy pool directory. Default is True.
        :return: None       
        '''        
        # Check if the config file is provided. 
        # If not, raise an error:
        if config is None:
            raise ValueError("The config file is not provided. Please provide a config file.")

        # Use the config file to initialize the settings
        # Experiment fundamental settings
        self.task_name = getattr(config, "TASK_NAME", None)
        self.seed = getattr(config, "SEED", 0)
        self.enable_resume_training = getattr(config, "ENABLE_RESUME_TRAINING", True)
        self.game = getattr(config, "GAME", "StreetFighterIISpecialChampionEdition-Genesis")    
        self.game_states_pvp = getattr(config, "GAME_STATES_PVP", None)
        self.game_states_pve = getattr(config, "GAME_STATES_PVE", None)
        self.game_states_eval = getattr(config, "GAME_STATES_EVAL", None)
        self.policy_pool_dir = getattr(config, "POLICY_POOL_DIR", None)
        self.rendering = getattr(config, "RENDERING", False)
        self.make_env = getattr(config, 'make_env', None)
        self.make_env_eval = getattr(config, 'make_env_eval', None)
        # Set the directories:  Make the log and save directories as LOG/SAVE_DIR + task_name
        log_dir = getattr(config, "LOG_DIR", "logs")
        self.log_dir = os.path.join(log_dir, self.task_name)
        os.makedirs(self.log_dir, exist_ok=True)
        # Logging settings
        self.log2file = getattr (config, "LOG2FILE", False)
        # TODO: The following 2 settings are not used yet. Remove it in the future.
        self.network_graph_tensorboard = getattr(config, "NETWORK_GRAPH_TENSORBOARD", False)
        self.checkpoint_interval = getattr(config, "CHECKPOINT_INTERVAL", 10000)
        # Training settings
        self.max_iterations = getattr(config, "MAX_ITERATIONS", 100)
        self.steps_per_iteration = getattr(config, "STEPS_PER_ITERATION", 5000000)
        self.num_envs = getattr(config, "NUM_ENVS", 12)
        self.pvp_ratio = getattr(config, "PVP_RATIO", 0.7)
        self.enemy_policy_selection = getattr(config, "ENEMY_POLICY_SELECTION", 'All')
        self.policy_pool_update_method = getattr(config, "POLICY_POOL_UPDATE_METHOD", "All")
        self.sticky_action_mode = getattr(config, "STICKY_ACTION_MODE", False)
        self.stickiness = getattr(config, "STICKINESS", 0.0)
        # Agent Model settings
        self.agent_model_class = getattr(config, "AGENT_MODEL_CLASS", None)
        self.agent_ent_coef = getattr(config, "AGENT_ENT_COEF", 0.01)
        self.agent_vf_coef = getattr(config, "AGENT_VF_COEF", 1.0)
        self.agent_policy_kwargs = getattr(config, "AGENT_POLICY_KWARGS", None)
   
        # Supported Model Architectures
        # TODO: These 2 parameters can be simplified and removed in the future.
        self.model_classes = getattr(config, "MODEL_CLASSES", None)
        self.policy_classes = getattr(config, "POLICY_CLASSES", None)
        # Evaluation settings
        self.evaluation_kwargs = getattr(config, "EVALUATION_KWARGS", None)     

        # Check if the parameter are legal 
        self.check()

        # Save the seed to file for reproducibility
        with open(os.path.join(self.log_dir, "seed.txt"), "w") as f:
            f.write(str(self.seed))

        # TODO: Remove this once SFAIEvalCallback is implemented.
        # Initialize the tensorboard writer
        # self.tb_writter = SummaryWriter(log_dir=os.path.join(self.log_dir, "evaluation_metrics") )  

        # Check if the policy pool directory exists or if the directory is not empty,
        # if not, create a new policy pool and save the untrained agent model in the policy pool as the initial enemy policy.
        if not os.path.exists(self.policy_pool_dir) or len(os.listdir(self.policy_pool_dir)) == 0:
            print("The policy pool directory does not exist or is empty. Creating a new policy pool.")
            self._create_new_policy_pool()        
            
        # Create the environment manager
        self.environment_manager = SelfPlayEnvironmentManager(
            game=self.game,
            make_env=self.make_env,
            rendering=self.rendering,
            num_envs=self.num_envs,
            sticky_action_mode=self.sticky_action_mode,
            stickiness=self.stickiness,
            pvp_ratio=self.pvp_ratio,
            enemy_policy_selection=self.enemy_policy_selection,
            model_classes=self.model_classes,
            policy_pool_dir=self.policy_pool_dir,
            game_states_pvp=self.game_states_pvp,
            game_states_pve=self.game_states_pve
        )
        # Check if the environment manager is created and initialized successfully
        if self.environment_manager is None:
            raise RuntimeError("The environment manager is not created successfully.")
        if self.environment_manager.envs is None:
            raise RuntimeError("The environments are not created by the environment manager successfully.")

        # Create the agent manager
        self.agent_manager = SelfPlayAgentManager(
            model_classes=self.model_classes,
            policy_classes=self.policy_classes,
        )

        # Add a agent to the agent manager
        # TODO: need to support multiple agents
        # For initialization, use the first element in SubprocVecEnv list to construct the agent
        print("Adding agent......")
        self.agent_manager.add_agent(
            env = self.environment_manager.envs,
            seed = self.seed,
            model_class = self.agent_model_class,
            agent_policy_kwargs=self.agent_policy_kwargs,
            ent_coef = self.agent_ent_coef,
            vf_coef = self.agent_vf_coef,
            log_dir=self.log_dir
        )
        # Check if the agent manager is created and initialized successfully
        if self.agent_manager is None:
            raise RuntimeError("The agent manager is not created successfully.")
        if self.agent_manager.agents is None:
            raise RuntimeError("The agents are not created by the agent manager successfully.")
        
        # Initialization finished, print the initialization information
        print("Initialization finished. The settings are as follows:")
        print("Task name: ", self.task_name)
        print("Resume training: ", self.enable_resume_training)
        print("Game: ", self.game)
        print("Game states for PvP: ", self.game_states_pvp)
        print("Game states for PvE: ", self.game_states_pve)
        print("Game state for evaluation: ", self.game_states_eval)
        print("Policy pool directory: ", self.policy_pool_dir)
        print("Rendering: ", self.rendering)
        print("Log directory: ", self.log_dir)
        print("Log to file: ", self.log2file)
        print("Network graph tensorboard: ", self.network_graph_tensorboard)
        print("Checkpoint interval: ", self.checkpoint_interval)
        print("Max iterations: ", self.max_iterations)
        print("Steps per iteration: ", self.steps_per_iteration)
        print("Number of environments: ", self.num_envs)
        print("PvP ratio: ", self.pvp_ratio)
        print("Enemy policy selection: ", self.enemy_policy_selection)
        print("Policy pool update method: ", self.policy_pool_update_method)
        print("Sticky action mode: ", self.sticky_action_mode)
        print("Stickiness: ", self.stickiness)
        print("Evaluation kwargs: ", self.evaluation_kwargs)
        print("Agent model class: ", self.agent_model_class)
        print("Agent entropy coefficient: ", self.agent_ent_coef)
        print("Initialization finished.")

            
    def check(self):
        '''
        Check if the parameters are legal.
        '''
        if self.task_name is None:
            raise ValueError("The task name should not be None. Double check the config file.")
        
        if self.game_states_pvp is None or len(self.game_states_pvp) == 0:
            raise ValueError("The game states for PvP should not be None or empty. Double check the config file.")
        
        if self.game_states_pve is None or len(self.game_states_pve) == 0:
            raise ValueError("The game states for PvE should not be None or empty. Double check the config file.")
        
        if self.game_states_eval is None or len(self.game_states_eval) == 0:
            raise ValueError("The game state for evaluation should not be None or empty. Double check the config file.")
        
        if self.policy_pool_dir is None:
            raise ValueError("The policy pool directory should not be None. Double check the config file.")
        
        if self.make_env is None or not callable(self.make_env):
            raise ValueError("The make_env function should not be None and should be callable. Double check the config file.")
        
        if self.make_env_eval is None or not callable(self.make_env_eval):
            raise ValueError("The make_env_eval function should not be None and should be callable. Double check the config file.")
        
        if self.agent_model_class is None:
            raise ValueError("The agent model class should not be None. Double check the config file.")
        
        if self.agent_policy_kwargs is None:
            raise ValueError("The agent policy kwargs should not be None. Double check the config file.")
        
        if self.evaluation_kwargs is None:
            raise ValueError("The evaluation kwargs should not be None. Double check the config file.")
        
        if self.model_classes is None:
            raise ValueError("The model classes should not be None. Double check the config file.")
        
        if self.policy_classes is None:
            raise ValueError("The policy classes should not be None. Double check the config file.")
        

    def train(self):
        """
        Train the agent using self-play.

        The training iteration consists of the following steps:
        1. Update the environment using the policy pool in this iteration.
        2. read self.environment_manager.envs which is a list of SubprocVecEnv and update the agent model
        3. Train the agent model for each SubprocVecEnv
        4. Evaluate the agent model
        5. Save the agent model and update the policy pool. Then go to step 1.
        """
        # Resume training if needed
        if self.enable_resume_training:
            # Check if in the policy pool directory, there are more than 2 models(.zip files)
            # If so, initialize the training progress based on the models.
            # If not, it is a new training, no need to resume training, just pass.
            if len(os.listdir(self.policy_pool_dir)) < 2:
                print("No need to resume training. It is a new training.")
                resumed_num_steps = 0
            else:
                # Resume the training process
                print("Currently, only number of steps will be updated after resuming training. Iterations will start from 1.")
                # Retrive the number of steps from the latest model
                resumed_num_steps = int(max(os.listdir(self.policy_pool_dir), key=lambda f: os.path.getmtime(os.path.join(self.policy_pool_dir, f))).split("_")[0])
                
                # Update the agent models by loading the latest model in the policy pool directory
                latest_model_path = os.path.join(
                    self.policy_pool_dir, 
                    max(os.listdir(self.policy_pool_dir), key=lambda f: os.path.getmtime(os.path.join(self.policy_pool_dir, f)))
                )
                for agent_model in self.agent_manager.agents:
                    agent_model_class = type(agent_model)
                    agent_model = agent_model_class.load(latest_model_path, device="cuda")
                print("Agent model loaded successfully. The loaded model is: ", latest_model_path)
                print("Resumed training. The number of steps is: ", resumed_num_steps)

        # Construct the evaluation environment and callback
        eval_envs = [self.make_env_eval(game = self.game, state= state, players=1) for state in self.game_states_eval]
        eval_env = SubprocVecEnv(eval_envs)
        best_model_save_path = os.path.join('agent_models/best_eval_models', self.task_name)
        os.makedirs(best_model_save_path, exist_ok=True)
        # 构造EvalCallback
        eval_callback = SFAIEvalCallback(
            eval_env=eval_env,                 # 评估环境
            best_model_save_path = best_model_save_path,    # 如果出现新最佳分数，模型会自动保存在此路径
            log_path=self.log_dir,                # 用于保存评估结果(.npz等)
            eval_freq=self.evaluation_kwargs.get('eval_callback_freq', 50000),                   # 训练多少步后触发一次评估
            n_eval_episodes=len(eval_envs),                 # 每次评估时跑多少个episode
            deterministic=True,                # 是否使用确定性策略(适用于有随机策略的场景)
            render=False,                      # 是否在评估时渲染环境画面
            verbose=1                          # 输出评估过程的信息
        )

        # Main training loop
        for iteration in range(self.max_iterations):
            print("##############################################################################")
            print(f"Iteration {iteration + 1}/{self.max_iterations}")

            if iteration > 0:
                # have to reconstruct the environments for each iteration, because the policy pool may have been updated
                self.environment_manager.construct_environments(env_idx = 0)
                # TODO: 运行代码确认无误后，删除下面这行代码
                # Update the steps per SubprocVecEnv
                # steps_per_env = self.steps_per_iteration // len(self.environment_manager.envs)
            
            # Calculate the steps per SubprocVecEnv
            steps_per_env = self.steps_per_iteration // self.environment_manager.num_subproc_vecenvs
            # Train the agent model for each SubprocVecEnv
            for env_idx in range(self.environment_manager.num_subproc_vecenvs):
                print("This is the no. ", env_idx + 1, " environment.")
                print("The total number of environments is: ", self.environment_manager.num_subproc_vecenvs)
                print("In this iteration, the total number of steps is: ", steps_per_env)

                # Update the environments
                # The first iteration does not need to update the environments because the environments are already updated by self.environment_manager.construct_environments()
                if env_idx > 0:
                    self.environment_manager.update_environments(env_idx = env_idx)

                # Set the agents models' environment
                # TODO: support multiple agents
                for agent_model in self.agent_manager.agents:
                    agent_model.set_env(self.environment_manager.envs)

                # Train the agent model
                for agent_model in self.agent_manager.agents:
                    agent_model.learn(
                        total_timesteps=steps_per_env,
                        callback=eval_callback,
                        tb_log_name = 'training_curve',
                        reset_num_timesteps = False
                    )

            #     # Set the agents models' environment
            #     # TODO: support multiple agents
            #     for agent_model in self.agent_manager.agents:
            #         agent_model.set_env(env)

            #     # Train the agent model
            #     for agent_model in self.agent_manager.agents:
            #         agent_model.learn(
            #             total_timesteps=steps_per_env,
            #             callback=self.checkpoint_callback,
            #             tb_log_name = self.log_dir
            #         )
            #     # Close the environment that has been used
            #     if self.rendering:
            #         env.env_method('render', close=True)
            #     env.close()

            # Evaluate the agent model
            for agnet_model_idx, agent_model in enumerate(self.agent_manager.agents):
                # Evaluate the agent model                    
                win_rate, pvp_win_rate, pve_win_rate, average_reward, enemy_models_win_rate, average_behavior_statistic = self._evaluate_agent(
                    agent_model = agent_model,
                    evaluation_kwargs=self.evaluation_kwargs
                )
                # For every 10 iterations, evaluate the agent model against all the enemies in the policy pool to update the whole policy pool's win rate
                if (iteration + 1) % 10 == 0:
                    _, _, _, _, enemy_models_win_rate = self._evaluate_whole_policy_pool(
                        agent_model = agent_model,
                        evaluation_kwargs = self.evaluation_kwargs
                    )
                # Update the enemy model in policy pool
                # The enemy_models_win_rate is a dictionary, the key is the enemy model's name, the value is the win rate of the enemy model
                # Very convenient to update the name using self._update_model_name()
                for enemy_model_name, enemy_model_win_rate in enemy_models_win_rate.items():
                    self._update_model_name(enemy_model_name, enemy_model_win_rate)
                # Print the evaluation results
                print("\033[32mIteration{} Evaluation results: \033[0m".format(iteration))
                print(f"Win rate: {win_rate}")
                print(f"PvP win rate: {pvp_win_rate}")
                print(f"PvE win rate: {pve_win_rate}")
                print(f"Average reward: {average_reward}")
                for key, value in average_behavior_statistic.items():
                    print("Average {} per round: {}".format(key, value))

                # # log the results to tensorboard
                # self.tb_writter.add_scalar(f"agent_{agnet_model_idx}/win_rate", win_rate, iteration)
                # self.tb_writter.add_scalar(f"agent_{agnet_model_idx}/pvp_win_rate", pvp_win_rate, iteration)
                # self.tb_writter.add_scalar(f"agent_{agnet_model_idx}/pve_win_rate", pve_win_rate, iteration)
                # self.tb_writter.add_scalar(f"agent_{agnet_model_idx}/average_reward", average_reward, iteration)
                # for key, value in average_behavior_statistic.items():
                #     self.tb_writter.add_scalar(f"agent_{agnet_model_idx}/average_{key}_per_round", value, iteration)


                # Save the agent model and update the policy pool
                # Construct the model's name, the name should be in this format:
                # {num_iterations*steps_per_iteration}_steps_<otherparts>$w_{win rate}_r_{average reward}_otr_{other metrics to be determined}$<otherparts>.zip
                # for example:
                # 200000_steps_%AuxObjPPO%spmv_reward_$w_0.7_r_1$.zip
                # First, calculate the total steps
                name_steps = (iteration + 1) * self.steps_per_iteration + resumed_num_steps
                # Get the agent model's class name like AuxObjPPO
                name_model_class = agent_model.__class__.__name__
                # Get this task's name
                name_task = self.task_name
                # Combine these parts with win rate and average reward
                file_name = f"{name_steps}_steps_%{name_model_class}%{name_task}$w_{win_rate}_r_{average_reward}$"

                if self.policy_pool_update_method == "All":
                    update_policy_pool_all(
                        policy_pool_dir=self.policy_pool_dir,
                        model_to_save=agent_model,
                        model_name=file_name
                    )
                elif isinstance(self.policy_pool_update_method, int):
                    update_policy_pool_top_n(
                        policy_pool_dir=self.policy_pool_dir,
                        model_to_save=agent_model,
                        model_name=file_name,
                        n=self.policy_pool_update_method
                    )
                else:
                    raise ValueError("The policy pool update method should be an integer or a string (currently only support 'All').")

                
                

        print("Training finished.")
                
    def _create_new_policy_pool(self):
            
            def linear_schedule(initial_value, final_value=0.0):
                if isinstance(initial_value, str):
                    initial_value = float(initial_value)
                    final_value = float(final_value)
                    assert (initial_value > 0.0)
                def scheduler(progress):
                    return final_value + progress * (initial_value - final_value)
                return scheduler
            
            # Create the policy pool directory
            os.makedirs(self.policy_pool_dir, exist_ok=True)

            # Construct the policy_kwargs if self.agent_policy_kwargs is None
            if self.agent_policy_kwargs is None:
                # If the policy class is AuxObjPPO class, the auxiliary heads are required
                if self.agent_model_class == "AuxObjPPO":
                    policy_kwargs = dict(
                        features_extractor_class=self.features_extractor_classes[self.agent_features_extractor_class],
                        features_extractor_kwargs=dict(features_dim=self.agent_feature_dimension),
                        net_arch = self.ac_architectures[self.agent_ac_architecture],
                        regressor_aux_heads_num = self.agent_num_regressor_aux_heads,
                        classifier_aux_heads_num = self.agent_num_classifier_aux_heads
                    )
                elif self.agent_model_class == "MultiInputPPO":
                    policy_kwargs = dict(
                        features_extractor_class=self.features_extractor_classes[self.agent_features_extractor_class],
                        net_arch = self.ac_architectures[self.agent_ac_architecture],
                    )
                else:
                    policy_kwargs = dict(
                        features_extractor_class=self.features_extractor_classes[self.agent_features_extractor_class],
                        features_extractor_kwargs=dict(features_dim=self.agent_feature_dimension),
                        net_arch = self.ac_architectures[self.agent_ac_architecture],
                    )
            else:
                policy_kwargs = self.agent_policy_kwargs

            # Set up learning rate and clip range schedules
            lr_schedule = linear_schedule(
                initial_value = 2.5e-4, 
                final_value = 2.5e-6
            )
            clip_range_schedule = linear_schedule(
                initial_value = 0.15, 
                final_value = 0.025
            )

            # Initialize an environment for creating the agent model
            env = self.make_env(
                game=self.game,
                state=self.game_states_pve[0],
                rendering=self.rendering,
                enemy_model_path=None,
                enemy_model_class=None,
                seed=0,
                sticky_action_mode=self.sticky_action_mode,
                stickiness=self.stickiness,
                players=1 # Just for crreating enemy model, no need to set players to 2 as PvP environment.
            )
            _env = SubprocVecEnv([env for _ in range(1)])

            # Construct the agent model
            cls = self.model_classes[self.agent_model_class]
            if self.agent_model_class == "AuxObjPPO":
                agent = cls(
                    policy = self.policy_classes[self.agent_model_class],
                    policy_kwargs = policy_kwargs,
                    env = _env,
                    device = "cuda",
                    verbose = 1,
                    n_steps = 512,
                    batch_size = 256,
                    n_epochs = 4,
                    gamma = 0.94,
                    learning_rate = lr_schedule,
                    clip_range = clip_range_schedule,
                    tensorboard_log = self.log_dir,
                )
            elif self.agent_model_class == "MultiInputPPO" or "PPO":
                agent = cls(
                    policy = self.policy_classes[self.agent_model_class],
                    policy_kwargs = policy_kwargs,
                    env = _env,
                    device = "cuda",
                    verbose = 1,
                    n_steps = 512,
                    batch_size = 256,
                    n_epochs = 4,
                    gamma = 0.94,
                    learning_rate = lr_schedule,
                    clip_range = clip_range_schedule,
                    tensorboard_log = self.log_dir
                )
            else:
                # For models without entropy coefficient, e.g. DQN
                agent = cls(
                    policy = self.policy_classes[self.agent_model_class],
                    policy_kwargs = policy_kwargs,
                    env = _env,
                    device = "cuda",
                    verbose = 1,
                    n_steps = 512,
                    batch_size = 256,
                    n_epochs = 4,
                    gamma = 0.94,
                    learning_rate = lr_schedule,
                    clip_range = clip_range_schedule,
                    tensorboard_log = self.log_dir
                )  
            
            # Create the name of the initial enemy policy
            initial_enemy_policy_name = '%' + self.agent_model_class + '%' + "_initial_enemy_policy" + "$w_0.5_r_0.0$" + ".zip"
            # Save the agent model as the initial enemy policy
            agent.save(os.path.join(self.policy_pool_dir, initial_enemy_policy_name))
                                                                           
    def _evaluate_agent(
            self,
            agent_model: BaseAlgorithm = None,
            evaluation_kwargs: Dict = {}
        ):
        
        """
        Evaluate the agent's performance.

        :param episode: The current episode number.
        :param agent_model: The agent model to be evaluated.
        :param num_pvp_tests: The number of PvP tests.
        :param num_pve_tests: The number of PvE tests.
        :param test_episodes: The number of episodes to test.
        """
        if agent_model is None:
            raise ValueError("The agent model should be provided.")
        if evaluation_kwargs is None:
            raise ValueError("The evaluation_kwargs should be provided.")
            
        print("Evaluating the agent model... ...")
        # Use policy_assessment to evaluate the agent
        win_rate, pvp_win_rate, pve_win_rate, average_reward, enemy_models_win_rate, average_behavior_statistic = policy_assessment(
            make_env=self.make_env_eval,
            assessed_model = agent_model,
            # assessment_type = "Hybrid",
            pvp_states = self.game_states_pvp,
            enemies_model_path =self.policy_pool_dir,
            # num_pvp_tests = num_pvp_tests,
            # built_in_ai_states = self.game_states_pve,
            # num_pve_tests = num_pve_tests,
            # test_episodes = test_episodes,
            # rendering = rendering # NOTE: if the code runs correctly, removed the commented lines
            **evaluation_kwargs
        )

        return win_rate, pvp_win_rate, pve_win_rate, average_reward, enemy_models_win_rate, average_behavior_statistic

    def _evaluate_whole_policy_pool(
            self,
            agent_model: BaseAlgorithm = None,
            evaluation_kwargs: Dict = {}
        ):
        '''
        In this method, we evaluate the agent model against all the enemies in the policy pool.
        This is used to update the whole policy pool's win rate.


        :param agent_model: The agent model to be evaluated.
        '''
        if agent_model is None:
            raise ValueError("The agent model should be provided.")
        
        print("Evaluating the agent model against all the enemies in the policy pool... ...")
        # Calculate how many enemies in the policy pool
        num_enemies = len(os.listdir(self.policy_pool_dir))
        # Use policy_assessment to evaluate the agent
        win_rate, pvp_win_rate, pve_win_rate, average_reward, enemy_models_win_rate, average_behavior_statistic = policy_assessment(
            make_env = self.make_env_eval,
            assessed_model = agent_model,
            assessment_type = evaluation_kwargs.get("assessment_type", "Standard"),
            pvp_states = self.game_states_pvp,
            enemies_model_path =self.policy_pool_dir,
            num_pvp_tests = num_enemies,
            built_in_ai_states = self.game_states_pve,
            num_pve_tests = 0,
            test_episodes = evaluation_kwargs.get("test_episodes", 2),
            rendering = evaluation_kwargs.get("rendering", False),
            assessment_batch_size = evaluation_kwargs.get("assessment_batch_size", 5)
        )
        
        return win_rate, pvp_win_rate, pve_win_rate, average_reward, enemy_models_win_rate

    def _update_model_name(
            self,
            model_name: str = None,
            win_rate: float = None
        ):
        '''
        Update the name of an existed model
        Usually, just for update the enemy model in the policy pool used in evaluating the agent model.

        The model name should be in this format:
        <otherparts>$w_{win rate}_r_{average reward}_otr_{other metrics to be determined}$<otherparts>.zip
        for example:
        %AuxObjPPO%spmv_reward_$w_0.7_r_1$.zip
        '''
        # The model name should be in this format:
        # Read the model' name and replace the win rate with the new win rate
        # The model name should be in this format:
        # <otherparts>$w_{win rate}_r_{average reward}_otr_{other metrics to be determined}$<otherparts>.zip
        # for example:
        # %AuxObjPPO%spmv_reward_$w_0.7_r_1$.zip

        # The win rate should be in range [0,1]
        if win_rate < 0 or win_rate > 1:
            raise ValueError("The win rate should be in range [0,1].")
        # Split the model name by "$"
        model_name_parts = model_name.split("$")
        # Find the index of the win rate in the splited model name
        # The win rate is after 'w_' and before '_r'
        # Local this part in the model name and split it by "_", then replace the win rate with the new win rate
        for i, part in enumerate(model_name_parts):
            if part.startswith("w_"):
                win_rate_part = part.split("_")
                win_rate_part[1] = str(win_rate)
                model_name_parts[i] = "_".join(win_rate_part)

        # Join the model name parts as the updated model name
        new_name = "$".join(model_name_parts)
        # Join the old and new model name with policy pool directory
        new_path = os.path.join(self.policy_pool_dir, new_name)
        model_path = os.path.join(self.policy_pool_dir, model_name)
        # Update the file's name
        os.rename(model_path, new_path)
        return new_name
    




# # Unit test codes
# name = r'%AuxObjPPO%spmv_reward_$w_0.7_r_1$.zip'
# print(SelfPlay._update_model_name(name, 0.8))


