# In this file, we define the policy selection function, which is used to select the policy as the agent's enemy.
# The policy selection function is used to select the policy as the agent's enemy by serveral metrics, such as the win rate, the average reward.
# The pipeline of the policy selection is as follows:
# 1. load the policy pool (This pool can be policy pool for self-play, or can also be a policy pool specific for evaluation, this method is not limited to self-play).
# 2. Retrieve the file names of models in the policy pool.
# 3. Retrieve the information of the models from the files names.
# 4. Sort the models according to the information.
# 5. Return the top N models as the selected policies.

# Be aware that the legel format of the file name is as follows:
# %model_class%_$w_{win rate}_r_{average reward}_otr_{other metrics to be determined}$.zip
# For example, a legal file name is '%AuxObjPPO%_Ryu_Self_Play_PPO$w_0.5_r_0.5.zip$'.


# The peformance information retrieval function is implemented in this way:
# 1. Extract the model's performance from string between $ and $ only.
# 2. Split the string by "_".
# 3. Retrieve the information by the index of the split:
#    - win rate: after the first "w", and should be the 2nd element of the split.
#    - average reward: after the first "r", and should be the 4th element of the split.
#    - other metrics: TBD
# Check if the retrived information match the legal format mentioned above. All other formats are illegal and should be discarded after raising an error.

# The model class retrieval function is implemented in this way:
# 1. Extract the model class from string between % and % only.
# 2. Return the model class.
# ==============================================================================================================

import os
import sys
import numpy as np

# Define the information retrieval function
def retrieve_performance(file_name: str = None): 
    '''
    This function is used to retrieve model's performacne information from the file name of the model.
    The information should be in the format of <otherparts>$w_{win rate}_r_{average reward}_otr_{other metrics to be determined}$<otherparts>.zip.
    The function will extract the information from the file name, and return the win rate and the average reward.

    Parameters:
    - file_name: The file name of the model.

    Returns:
    - model_performance: A dictionary containing the win rate and the average reward (and other information will be implemented in the future).
    '''
    # If no file name is provided, raise an error
    if file_name is None:
        raise ValueError("The file name should be provided.")
    # Extract the string between $ and $ only
    model_performance_info = file_name.split("$")[1]
    # Split the string by "_"
    model_performance_info = model_performance_info.split("_")
    # Retrieve the information by the index of the split
    # win rate: after the first "w", and should be the 2nd element of the split
    # average reward: after the first "r", and should be the 4th element of the split
    # other metrics: TBD
    win_rate = model_performance_info[model_performance_info.index("w") + 1]
    average_reward = model_performance_info[model_performance_info.index("r") + 1]
    # Check if the retrived information match the legal format mentioned above
    # All other formats are illegal and should be discarded after raising an error
    # Check if the win_rate and average_reward string is a number
    if not win_rate.replace(".", "", 1).isdigit():
        raise ValueError("The win rate should be a number.")
    if not (average_reward.replace(".", "", 1).isdigit() or (average_reward[0] == '-' and average_reward[1:].replace(".", "", 1).isdigit() or average_reward.replace(".", "", 1).isdigit())):
        raise ValueError("The average reward should be a number.")
    
    # Store the information in a dictionary
    model_performance = {
        "win_rate": float(win_rate),
        "average_reward": float(average_reward)
    }

    return model_performance


# Define the model class retrieval function
def retrieve_model_class(file_name: str = None):
    '''
    This function is used to retrieve the model class from the file name of the model.
    The model class should be in the format of 'Ryu_Self_Play_%AuxObjPPO%_$w_0.5_r_0.5.zip'.
    The function will extract the model class from the file name, and return the model class.

    Parameters:
    - file_name: The file name of the model.

    Returns:
    - model_class: The model class of the model.
    '''
    # If no file name is provided, raise an error
    if file_name is None:
        raise ValueError("The file name should be provided.")
    # Extract the string between % and % only
    model_class = file_name.split("%")[1]

    return model_class
    

# Vinilla version, only based on the win rate
def policy_selection_vanilla(
        policy_pool_dir: str = None, 
        top_n: int = 1,
        repeat = True,
    ):
    '''
    This function is used to select the policy as the agent's enemy based on the win rate.
    The policy selection function is used to select the policy as the agent's enemy by serveral metrics, such as the win rate, the average reward.
    The pipeline of the policy selection is as follows:
    1. load the policy pool (This pool can be policy pool for self-play, or can also be a policy pool specific for evaluation, this method is not limited to self-play).
    2. Retrieve the file names of models in the policy pool.
    3. Retrieve the information of the models from the files names.
    4. Sort the models according to the information.
    5. Return the top N models as the selected policies.

    Parameters:
    - policy_pool_dir: The directory of the policy pool.
    - top_n: The number of models to be selected.
    - repeat: Whether to allow the same model to be selected when top_n is larger than the number of models in the policy pool.

    Returns:
    - selected_policies: A list of the top N models selected.
    - selected_policies_performance: A list of the performance information of the top N models.
    - selected_policies_model_class: A list of the model class of the top N models.
    '''
    # If no policy pool directory is provided, raise an error
    if policy_pool_dir is None:
        raise ValueError("The policy pool directory should be provided.")


    # Retrieve the file names of models in the policy pool
    policy_pool = os.listdir(policy_pool_dir)
    # Retrieve the performance information of the models from the files names
    policy_performance = [retrieve_performance(file_name) for file_name in policy_pool]
    # Sort the models according to the performance information
    policy_pool = [x for _, x in sorted(zip(policy_performance, policy_pool), key=lambda pair: pair[0]["win_rate"], reverse=True)]

    # Check if top_n is larger than the length of the policy pool, if so, raise an error
    if top_n > len(policy_pool) and not repeat:
        raise ValueError("If not repeat, the 'top_n' should not be larger the number of models in the policy pool. Please set 'repeat' to True or reduce the 'top_n'.")

    # Return the top N models as the selected policies's file names
    if top_n <= len(policy_pool):
        selected_policies = [policy_pool[i] for i in range(top_n)]
    # When repeat is True and top_n is larger than the length of the policy pool.
    # Repeat by random duplicate the element in selected_policies until the length of the policy pool is equal to top_n
    # The duplicated element should be right after the original element to avoid influence the order of the original elements.
    elif repeat and top_n > len(policy_pool):
        selected_policies = policy_pool.copy()
        # Calcuate the number of elements to be duplicated
        num_dup = top_n - len(policy_pool)
        # Randomly select the index of the element to be duplicated
        dup_index = np.random.choice(len(policy_pool), num_dup, replace=True)
        # Duplicate the element by inserting the duplicated element right after the original element
        for i in range(num_dup):
            # The index of the element to be duplicated
            index = dup_index[i]
            # The element to be duplicated
            element = selected_policies[index]
            # Insert the duplicated element right after the original element
            selected_policies.insert(index + 1, element)
        print(f"Currently the policy pool doesn't have enough models, {num_dup} models are duplicated to make the number of models in the policy pool equal to {top_n}.")
    else:
        raise ValueError("The 'top_n' should not be larger the number of models in the policy pool. Please set 'repeat' to True or reduce the 'top_n.")

    # Top N performance information
    # Retrieve the performance information of the top N models from the files names
    selected_policies_performance = [retrieve_performance(file_name) for file_name in selected_policies]

    # Top N model classes
    # Retrieve the model class of the top N models from the files names
    selected_policies_model_class = [retrieve_model_class(file_name) for file_name in selected_policies]

    # Return the top N models as the selected policies and their performance information
    return selected_policies, selected_policies_performance, selected_policies_model_class

# Random version, randomly select n models.
def policy_selection_random(
        policy_pool_dir: str = None, 
        num_models: int = 1,
):
    '''
    This function is used to select the policy as the agent's enemy randomly.

    Parameters:
    - policy_pool_dir: The directory of the policy pool.
    - num_models: The number of models to be selected.

    Returns:
    - selected_policies: A list of the selected models.
    - selected_policies_performance: A list of the performance information of the selected models.
    - selected_policies_model_class: A list of the model class of the selected models.
    '''
    # If no policy pool directory is provided, raise an error
    if policy_pool_dir is None:
        raise ValueError("The policy pool directory should be provided.")
    
    # Retrieve the file names of models in the policy pool
    policy_pool = os.listdir(policy_pool_dir)
    # Randomly select n models
    selected_policies = np.random.choice(policy_pool, num_models, replace=True)
    # Retrieve the performance information of the selected models from the files names
    selected_policies_performance = [retrieve_performance(file_name) for file_name in selected_policies]
    # Retrieve the model class of the selected models from the files names
    selected_policies_model_class = [retrieve_model_class(file_name) for file_name in selected_policies]

    return selected_policies, selected_policies_performance, selected_policies_model_class

def policy_selection_all(
        policy_pool_dir: str = None,
):
    '''
    This function is used to select all the policies as the agent's enemy.

    Parameters:
    - policy_pool_dir: The directory of the policy pool.

    Returns:
    - selected_policies: A list of the selected models.
    - selected_policies_performance: A list of the performance information of the selected models.
    - selected_policies_model_class: A list of the model class of the selected models.
    '''
    # If no policy pool directory is provided, raise an error
    if policy_pool_dir is None:
        raise ValueError("The policy pool directory should be provided.")
    
    # Retrieve the file names of models in the policy pool
    policy_pool = os.listdir(policy_pool_dir)
    # Select all the models
    selected_policies = policy_pool
    # Retrieve the performance information of the selected models from the files names
    selected_policies_performance = [retrieve_performance(file_name) for file_name in policy_pool]
    # Retrieve the model class of the selected models from the files names
    selected_policies_model_class = [retrieve_model_class(file_name) for file_name in policy_pool]

    return selected_policies, selected_policies_performance, selected_policies_model_class



    



# # Unit test code 
# # To test the function, uncomment the following code and import the function in the unit_test.py file and run the unit_test.py file.

# policy_pool_dir = r"agent_models/policy_pool"
# top_n = 3
# selected_policies, selected_policies_performance, selected_policies_model_class = policy_selection_vanilla(policy_pool_dir, top_n)
# print(f"The selected policies are: {selected_policies}")
# print(f"The selected policies' performance information are: {selected_policies_performance}")
# print(f"The selected policies' model class are: {selected_policies_model_class}")
