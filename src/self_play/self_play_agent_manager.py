# To specify the arguments for adding agents. import the Env class from gym
# TODO: rename the model_xxx in add_agent to agent_model_xxx
import gym
from typing import Union
from stable_baselines3 import PPO, A2C, SAC
ALGO_WITH_ENT_COEF = [PPO, A2C, SAC]
# TODO: support sb3_contrib models, e.g. Recurrent PPO, TQC, CrossQ


class SelfPlayAgentManager:
    '''
    This class is responsible for managing the agents in the self-play environment.
    
    The Manager class manage one or more agents in the self-play environment.
    Usually, we just train one agent, but the manager provide the flexibility to train multiple agents in the self-play training.

    The Manager class has:
    - The supported model classes (defined during the initialization)

    The Manager class provides:
    - The list of agent/agents

    The Manager class can:
    - Add agent/agents
    '''
    def __init__(
            self,
            model_classes: dict,
            policy_classes: dict,
        ):
        # Must provide the arguments
        # Check if the arguments are provided, if not raise an error
        if not model_classes:
            raise ValueError("The model classes must be provided.")
        if not policy_classes:
            raise ValueError("The policy classes must be provided.")
        self.model_classes = model_classes
        self.policy_classes = policy_classes

        self.agents = []
        print("---------------------------Agent Manager initialized---------------------------")

    def add_agent(
            self,
            env: gym.Env,
            seed: int = 0,
            model_class: str = None,
            agent_policy_kwargs: dict = None,
            ent_coef: float = 0.01,
            vf_coef: float = 0.5,
            log_dir: str = None
        ):
        '''
        Add an agent to the manager.

        Args:
        - model_class: The model class to use for the agent.
        - features_extractor_class: The features extractor class to use for the agent.
        - ac_architecture: The Actor-Critic architecture to use for the agent.
        - ent_coef: The entropy coefficient to use for the agent.
        - feature_dimension: The feature dimension to use for the agent.
        - log_dir: The log directory to use for the agent.
        TODO: log_dir needs to be further customized to specify the model name.
        '''
        # Check if the model class is provided, if not raise an error
        if not model_class:
            raise ValueError("The model class must be provided.")
        if log_dir is None:
            raise ValueError("The log directory must be provided.")        
        
        
        # Extract the information for constructing the agent
        # Check if the model class is in the supported model classes dictionary, if not raise an error
        if model_class not in self.model_classes:
            raise ValueError(f"The model class {model_class} is not supported.")
        # Check if the policy class is in the supported policy classes dictionary, if not raise an error
        # The policy class dictionary shares the same keys as the model class dictionary
        # If it's not in the supported policy classes, raise an error
        if model_class not in self.policy_classes:
            raise ValueError(f"The model class {model_class} is not supported.")
        # Extract the policy class and model class
        policy_class = self.policy_classes[model_class]
        model_class = self.model_classes[model_class]
        

        # Check if the entropy coefficient is a float and non-negative, if not raise an error
        if not isinstance(ent_coef, float) or ent_coef < 0:
            raise ValueError("The entropy coefficient must be a non-negative float.")
        
        # Check if the value function coefficient is a float and non-negative, if not raise an error
        if not isinstance(vf_coef, float) or vf_coef < 0:
            raise ValueError("The value function coefficient must be a non-negative float.")

        policy_kwargs = agent_policy_kwargs
        

        # Set up learning rate and clip range schedules
        lr_schedule = self.linear_schedule(
            initial_value = 2.5e-4, 
            final_value = 2.5e-6
        )
        clip_range_schedule = self.linear_schedule(
            initial_value = 0.15, 
            final_value = 0.025
        )

        # Construct the agent model
        if model_class in ALGO_WITH_ENT_COEF:
            # For other models with entropy coefficient, e.g. PPO, A2C, SAC
            agent = model_class(
                policy = policy_class,
                policy_kwargs = policy_kwargs,
                env = env,
                seed = seed,
                device = "cuda",
                ent_coef = ent_coef,
                vf_coef = vf_coef,
                verbose = 1,
                n_steps = 512,
                batch_size = 256,
                n_epochs = 4,
                gamma = 0.94,
                learning_rate = lr_schedule,
                clip_range = clip_range_schedule,
                tensorboard_log = log_dir
            )
        else:
            # For models without entropy coefficient, e.g. DQN
            agent = model_class(
                policy = policy_class,
                policy_kwargs = policy_kwargs,
                env = env,
                seed = seed,
                device = "cuda",
                verbose = 1,
                n_steps = 512,
                batch_size = 256,
                n_epochs = 4,
                gamma = 0.94,
                learning_rate = lr_schedule,
                clip_range = clip_range_schedule,
                tensorboard_log = log_dir
            )

        self.agents.append(agent)
        print("------------------From AgentManager------------------")
        print("Agent added to the manager, the agent information is as follows:")
        print(f"Model class: {model_class}")
        print(f"Entropy coefficient: {ent_coef}")
        print(f"Value function coefficient: {vf_coef}")
        print(f'agent`s policy_kwargs is: {agent_policy_kwargs}')
        print("--------------------------------------------------------")
    

    def change_env(
            self,
            index: int = None,
            env: gym.Env = None
        ):
        '''
        Change the environment for all agents or the agent at the specified index.

        Args:
        - index: The index of the agent to change the environment.
        - env: The new environment to use.
        '''
        if env is None:
            raise ValueError("The new environment must be provided.")
        if index >= len(self.agents) or index < 0:
            raise ValueError("The agent index is out of range.")
        # update all agents
        if index is None:
            for agent in self.agents:
                agent.set_env(env)
        # update the agent at the specified index  
        elif 0 <= index < len(self.agents):
            self.agents[index].set_env(env)
        else:
            raise IndexError("Agent index out of range")

    def linear_schedule(self, initial_value, final_value=0.0):
        
        if isinstance(initial_value, str):
            initial_value = float(initial_value)
            final_value = float(final_value)
            assert (initial_value > 0.0)
        def scheduler(progress):
            return final_value + progress * (initial_value - final_value)

        return scheduler

    def get_agents(self):
        return self.agents

    def get_agent(self, index):
        if 0 <= index < len(self.agents):
            return self.agents[index]
        else:
            raise IndexError("Agent index out of range")

