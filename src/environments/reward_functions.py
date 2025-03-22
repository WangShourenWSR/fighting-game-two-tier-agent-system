import math
import numpy as np

def naruto_reward(
        info: dict = None, 
        round_over_reward_given: bool = None, 
        full_hp = 176, 
        reward_coeff = 1.0
    ):
    '''
    The default reward function for the environment.
    param info: The information returned by the environment. By the implementation of SFAIWrapper, the info also contains the info of previous 20 steps (and it is required for computing reward).
    param round_over_reward_given: A boolean value to indicate whether the reward for the round over has been given (Just need to be given once each round). Default is None, must be given.
    param full_hp: The full health points of the player. Default is 176. Usually don't need to change this value.
    param reward_coeff: The coefficient of the reward. Default is 1.0. Usually don't need to change this value.
    '''
    # Check if the required parameters are given. If not given, raise an error.
    if info is None:
        raise ValueError("The info parameter is not given.")
    if round_over_reward_given is None:
        raise ValueError("The round_over_reward_given parameter is not given.")
    
    # Retrieve the required information from the info.
    # information of the current step
    curr_player_health = info['agent_hp']
    curr_oppont_health = info['enemy_hp']
    round_countdown = info['round_countdown']

    # information of the previous step
    prev_info = info['info_sequence_buffer'][-1]
    prev_player_health = prev_info['agent_hp']
    prev_oppont_health = prev_info['enemy_hp']

    # Initialize the custom reward
    change_of_hp = prev_oppont_health - curr_oppont_health - (prev_player_health - curr_player_health)

    # Game is over and player loses.
    if curr_player_health < 0:
        if not round_over_reward_given:
            custom_reward = change_of_hp
            round_over_reward_given = True
    # Game is over and player wins.
    elif curr_oppont_health < 0:
        if not round_over_reward_given:
            custom_reward = change_of_hp
            round_over_reward_given = True
    
    # Game is over due to time out.
    elif round_countdown == 0 and not round_over_reward_given:
        if not round_over_reward_given:
            custom_reward = change_of_hp * 7
            round_over_reward_given = True

    # While the fighting is still going on
    else:
        # WRITE YOUR CUSTOM REWARD HERE, AND DELETE THE 'pass' BELOW
        custom_reward = change_of_hp

    return custom_reward, round_over_reward_given


def sfai_reward(
        info: dict = None, 
        round_over_reward_given: bool = False, 
        reward_kwargs: dict = {},
        full_hp = 176, 
        max_dist: int = 210,
        min_dist: int = 70,
    ):
    '''
    The reward function for the SFAI environment.
    param info: The information returned by the environment. By the implementation of SFAIWrapper, the info also contains the info of previous 20 steps (and it is required for computing reward).
    param round_over_reward_given: A boolean value to indicate whether the reward for the round over has been given (Just need to be given once each round). Default is None, must be given.
    param reward_kwargs: The dictionary of the reward function parameters. Default is None. Must be given.

    # TODO: Add aggresive_defensive_reward
    # TODO: Add jump_coef (add it in cost term, if encourage jumping, set it to negative value)
    '''
    
    # Check if the required parameters are given. If not given, raise an error.
    if info is None:
        raise ValueError("The info parameter is not given.")
    if round_over_reward_given is None:
        raise ValueError("The round_over_reward_given parameter is not given.")
    

    # Retrieve the required information from the info.
    # information of the current step
    curr_player_health = info['agent_hp']
    curr_oppont_health = info['enemy_hp']
    round_countdown = info['round_countdown']
    agent_status = info['agent_status']
    # information of the previous step
    prev_info = info['info_sequence_buffer'][-1]
    prev_player_health = prev_info['agent_hp']
    prev_oppont_health = prev_info['enemy_hp']
    prev_agent_status = prev_info['agent_status']


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
    status_projectile_gone = False
    if info['agent_projectile_status'] != 34048 and prev_info['agent_projectile_status'] == 34048:
        status_projectile_gone = True
    # Regular attack status(regular attacks are light~heavy punches, light~heavy kicks)
    status_regular_attack = False
    if agent_status == 522 and prev_agent_status != 522:
        status_regular_attack = True
    # Jumping status
    status_jump = False
    if agent_status == 516 and prev_agent_status != 516:
        status_jump = True
    # Vulnerable frame check
    # The vulnerable frames are considered as the following cases:
    # 1. The agent is in a special move status. NOTE: Using invincibility frames in special moves are not considered for now.
    # 2. The agent is in a regular attack status but not using throw.
    # 3. The agent is in a jumping status and has already attacked in the air.
    vulnerable_frame = False
    if agent_status == 524 or (agent_status == 522 and info['enemy_status'] != 532) or info['jump_attack_status'] == 11246:
        vulnerable_frame = True


    # Non-player-status information check
    # Distance check
    distance = abs(info['agent_x'] - info['enemy_x'])
    # Normalize the distance from -1 to 1 according to the max_dist and min_dist
    middle_point = (max_dist + min_dist) / 2
    distance_normalized = 2 * (distance - middle_point) / (max_dist - min_dist)

    # In-Air check (if the agent is in the air)
    in_air = False
    if agent_status == 516:
        in_air = True


    # Initialization 
    # Initialize the reward terms and cost
    total_reward = 0
    special_move_reward = 0
    projectile_reward = 0

    cost = 0
    # Retrieve the reward parameters
    raw_reward_coef = reward_kwargs.get('raw_reward_coef', 1.0) # How much HP change is rewarded, only happens during the fighting (not round over)
    cost_coef = reward_kwargs.get('cost_coef', 0)
    # special_move_coef = reward_kwargs.get('special_move_coef', 0.5)
    # damage_reward_bonus_special_move = reward_kwargs.get('damage_reward_bonus_special_move', 3)
    # projectile_coef = reward_kwargs.get('projectile_coef', 2)
    # damage_reward_bonus_projectile = reward_kwargs.get('damage_reward_bonus_projectile', 5)
    # distance_coef = reward_kwargs.get('distance_coef', 0.01)

    # Cost calculation
    # Special move cost
    if status_special_move:
        cost += reward_kwargs.get('special_move_cost', 2)
    # Regular attack cost
    if status_regular_attack:
        cost += reward_kwargs.get('regular_attack_cost', 1)
    # Jumping cost
    if status_jump:
        cost += reward_kwargs.get('jump_cost', 3)
    # Vulnerable frame cost
    if vulnerable_frame:
        cost += reward_kwargs.get('vulnerable_frame_cost', 0.1)


    # Reward calculation
    # Game is over and player loses.
    custom_reward = 0
    if curr_player_health < 0:
        if not round_over_reward_given:
            custom_reward = -math.pow(full_hp, (curr_oppont_health + 1) / (full_hp + 1))    # Use the remaining health points of opponent as penalty. 
            round_over_reward_given = True
    # Game is over and player wins.
    elif curr_oppont_health < 0:
        # custom_reward = curr_player_health * self.reward_coeff # Use the remaining health points of player as reward.
        # custom_reward = math.pow(self.full_hp, (5940 - self.total_timesteps) / 5940) * self.reward_coeff # Use the remaining time steps as reward.
        if not round_over_reward_given:
            custom_reward = math.pow(full_hp, (curr_player_health + 1) / (full_hp + 1))
            round_over_reward_given = True
    
    # Game is over due to time out.
    elif round_countdown == 0 and not round_over_reward_given:
        if not round_over_reward_given:
            custom_reward = 10 * (prev_oppont_health - curr_oppont_health) - (prev_player_health - curr_player_health)
            round_over_reward_given = True

    # While the fighting is still going on
    else:
        # The basic reward term, calculated based on the health points change
        raw_reward = (prev_oppont_health - curr_oppont_health) - (prev_player_health - curr_player_health)

        # Special move reward
        if status_special_move:
            special_move_reward = reward_kwargs.get('special_move_reward', 0)
        if agent_status == 524 and prev_oppont_health - curr_oppont_health > 0:
            special_move_reward *= reward_kwargs.get('special_move_bonus', 1)
            raw_reward *= reward_kwargs.get('special_move_bonus', 1)

        # Projectile reward
        if status_projectile:
            projectile_reward = reward_kwargs.get('projectile_reward', 0)
        if status_projectile_gone and prev_oppont_health - curr_oppont_health > 0:
            projectile_reward *= reward_kwargs.get('projectile_bonus', 1)
            raw_reward *= reward_kwargs.get('projectile_bonus', 1)

        # Distance reward 
        distance_reward = distance_normalized * reward_kwargs.get('distance_reward', 0)
        # If distance bonus is set, when at a preferred/not-preferred distance, increase/decrease the reward for dealing damage and decrease/increase the penalty for taking damage
        distance_coef = math.exp(distance_normalized * reward_kwargs.get('distance_bonus', 0))
        if prev_oppont_health - curr_oppont_health > 0:
            raw_reward *= distance_coef
        if prev_player_health - curr_player_health > 0:
            raw_reward /= distance_coef

        # In-Air reward.
        in_air_reward = 0
        if in_air:
            in_air_reward = reward_kwargs.get('in_air_reward', 0)
        
        # Time reward
        # Calculate the time reward coefficient
        # NOTE: The absolute value for this coefficient term increases as the round goes on and the maximum value is 3. Round countdown starts from 39194 and ends at 0.
        time_reward_coef = 3 * (1 - round_countdown / 39194)
        time_reward = time_reward_coef * reward_kwargs.get('time_reward_bonus', 0)


        custom_reward = raw_reward * raw_reward_coef + special_move_reward + projectile_reward + distance_reward + in_air_reward + time_reward

    # Apply the cost to the total reward
    total_reward = custom_reward - cost * cost_coef
    reward_scale = reward_kwargs.get('reward_scale', 1.0)
    total_reward *= reward_scale

    return total_reward, round_over_reward_given
