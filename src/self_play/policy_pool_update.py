# Policy pool updaing methods
# Currently supported methods are:
# - "All": All agents in the policy pool are updated
# - "TopN": Only the top N agents in the policy pool are updated
import os
from stable_baselines3.common.base_class import BaseAlgorithm

from self_play.policy_selection import policy_selection_vanilla
from self_play.policy_selection import retrieve_performance

def update_policy_pool_all(
        policy_pool_dir,
        model_to_save,
        model_name
    ):
    """
    Update all agents in the policy pool with the new model.

    :param policy_pool_dir: (str) The directory where the policy pool is stored
    :param model_to_save: (BaseAlgorithm) The model to save
    :param model_name: (str) The name of the model
    """
    # Check if the model is a BaseAlgorithm or subclass of BaseAlgorithm
    if not issubclass(model_to_save.__class__, BaseAlgorithm):
        raise ValueError("The model to save must be a subclass of BaseAlgorithm")
    # Save the model to the policy pool directory
    model_to_save.save(os.path.join(policy_pool_dir, model_name + ".zip"))
    print(f"Model {model_name} saved to policy pool directory: {policy_pool_dir}")

def update_policy_pool_top_n(
        policy_pool_dir: str = None,
        model_to_save: BaseAlgorithm = None,
        model_name: str = None,
        n: int = 10
    ):
    """
    Update the top N win rate agents in the policy pool with the new model.

    The model name should be in this format:
        <otherparts>$w_{win rate}_r_{average reward}_otr_{other metrics to be determined}$<otherparts>.zip
        for example:
        %AuxObjPPO%spmv_reward_$w_0.7_r_1$.zip

    :param policy_pool_dir: (str) The directory where the policy pool is stored
    :param model_to_save: (BaseAlgorithm) The model to save
    :param model_name: (str) The name of the model
    :param n: (int) The number of agents to update
    """
    # Check if the model is a BaseAlgorithm or subclass of BaseAlgorithm
    if not issubclass(model_to_save.__class__, BaseAlgorithm):
        raise ValueError("The model to save must be a subclass of BaseAlgorithm")
    # Check if the policy pool directory exists
    if not os.path.exists(policy_pool_dir):
        raise FileNotFoundError(f"The policy pool directory {policy_pool_dir} does not exist")
    # Check if the model name is provided
    if model_name is None:
        raise ValueError("The model name must be provided")
    
    # Check if n is large than the number of models(files) in the policy pool directory
    # If so, simply add the new model to the policy pool directory
    if n > len(os.listdir(policy_pool_dir)):
        model_to_save.save(os.path.join(policy_pool_dir, model_name + ".zip"))
        print(f"Model {model_name} saved to policy pool directory: {policy_pool_dir}")
    else:
        # Select top N agents from the policy pool
        top_n_agents, top_n_agents_performance, _ = policy_selection_vanilla(policy_pool_dir, n)
        # Retrive the agents' win rate from the top_n_agents_performance dictionary into a list
        # top_n_agents_win_rate = [top_n_agents_performance[agent]["win_rate"] for agent in top_n_agents]
        top_n_agents_win_rate = []
        for performance in top_n_agents_performance:
            top_n_agents_win_rate.append(performance["win_rate"])
        # Sort the top N agents by win rate
        top_n_agents_sorted = [agent for _, agent in sorted(zip(top_n_agents_win_rate, top_n_agents), reverse=True)]
        
        # If the new agent' performance is not the weakest, remove the weakest agent by deleting the file
        # Retrive the agent's performance
        model_performance = retrieve_performance(model_name)
        model_win_rate = model_performance["win_rate"]
        if model_win_rate > top_n_agents_win_rate[-1]:
            os.remove(os.path.join(policy_pool_dir, top_n_agents_sorted[-1]))
            print(f"Model {top_n_agents_sorted[-1]} removed from policy pool directory: {policy_pool_dir}")
            # Save the new model to the policy pool directory
            model_to_save.save(os.path.join(policy_pool_dir, model_name + ".zip"))
            print(f"Model {model_name} saved to policy pool directory: {policy_pool_dir}")
        else:
            print(f"Model {model_name} is not in the top {n} win rate agents, not saved to policy pool directory: {policy_pool_dir}")


# # Unit test
# from models.aux_obj_ppo import AuxObjPPO
# model_path = "agent_models/policy_pool/%AuxObjPPO%_projectile_reward3_$w_1.0_r_0.1$.zip"
# model = AuxObjPPO.load(model_path)
# update_policy_pool_top_n(
#     policy_pool_dir="agent_models/policy_pool",
#     model_to_save=model,
#     model_name="AuxObjPPO$w_1.0_r_1$",
#     n=15
# )
