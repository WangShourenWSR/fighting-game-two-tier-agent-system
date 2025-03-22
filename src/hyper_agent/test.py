import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers import StoppingCriteria, StoppingCriteriaList
import re
import json

def extract_model_path_from_llm_response(llm_response: str) -> str:
    """
    From a possibly long text containing multiple lines, messages, or JSON-like
    fragments, extract 'chosen_agent_model_path' from the last valid JSON object
    found. If multiple valid JSON blocks appear, we pick the last one containing
    the required key.

    Args:
        llm_response (str): The entire string returned by the LLM, which may
                            contain additional commentary or multiple JSON blocks.

    Returns:
        str: The value of 'chosen_agent_model_path' in the last valid JSON block.

    Raises:
        ValueError: If no valid JSON with the required key is found.
    """

    # 1) 使用正则找出所有以 '{' 开头、 '}' 结尾的子串（懒惰匹配）
    #    DOTALL使 '.' 能匹配换行符，尽量避免跨越多个对象。
    #    这是一种简单启发式，如果文本里有花括号但不是真正的JSON也会被匹配到。
    potential_jsons = re.findall(r"\{.*?\}", llm_response, flags=re.DOTALL)

    if not potential_jsons:
        raise ValueError("No braces-enclosed text found in LLM response.")

    valid_model_paths = []

    # 2) 逐个尝试解析 JSON
    for fragment in potential_jsons:
        fragment_str = fragment.strip()
        try:
            # 尝试加载
            obj = json.loads(fragment_str)
            # 如果成功解析，并且包含 'chosen_agent_model_path'
            if "chosen_agent_model_path" in obj:
                valid_model_paths.append(obj["chosen_agent_model_path"])
        except json.JSONDecodeError:
            # 解析失败就跳过
            continue

    # 3) 如果没有找到符合要求的 JSON，则报错
    if not valid_model_paths:
        raise ValueError("No valid JSON object containing 'chosen_agent_model_path' was found.")

    # 4) 返回最后一个匹配到的值（如需第一个，可改为 valid_model_paths[0]）
    return valid_model_paths[-1]

class StreamStoppingCriteria(StoppingCriteria):
    def __call__(self, input_ids, scores, **kwargs):
        print(tokenizer.decode(input_ids[0, -1]), end="", flush=True)
        return False  # 不终止，直到 generate 结束

stopping_criteria = StoppingCriteriaList([StreamStoppingCriteria()])


# Set the path of the local model
model_path = "C:/Users/SR.W/LLMs/DeepSeek-R1-Distill-Llama-8B/snapshots/6a6f4aa4197940add57724a7707d069478df56b1"

quantization_config = BitsAndBytesConfig(
    load_in_8bit=True  # 启用 8-bit 量化
)

# Set device to cuda 
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    raise RuntimeError("CUDA is not available.")

# Load the model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    model_path, 
    quantization_config = quantization_config
)
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token


# Run the inference
# load the prompt template
with open("src/hyper_agent/prompt_template.txt", "r") as f:
    prompt_template = f.read()
# load the selection_principles file
with open("src/hyper_agent/selection_principles.txt", "r") as f:
    selection_principles = f.read()
# load the player date example file
with open("src/hyper_agent/example_player_data.txt", "r") as f:
    player_data_example = f.read()
# load the output format requirement file
with open("src/hyper_agent/output_format_requirement.txt", "r") as f:
    output_format_requirement = f.read()
# load the few shot example file
with open("src/hyper_agent/few_shot_examples.txt", "r") as f:
    few_shot_example = f.read()
# load the archive info json file
with open("src/hyper_agent/archive_info.json", "r") as f:
    archive_info = json.load(f)
    archive_info_str = json.dumps(archive_info, indent=2)
    archive_info_str_escaped = archive_info_str.replace("{", "{{").replace("}", "}}")
# apply the contents to the prompt template
prompt = prompt_template.format(
    SELECTION_PRINCIPLES = selection_principles,
    PLAYING_DATA = player_data_example,
    ARCHIVE_INFO = archive_info_str_escaped,
    OUTPUT_FORMAT_REQUIREMENT = output_format_requirement,
    FEW_SHOT_EXAMPLES = few_shot_example
)

inputs = tokenizer(prompt, padding = True, truncation=True, return_tensors="pt")
input_ids = inputs["input_ids"].to(device)
attention_mask = inputs["attention_mask"].to(device)

with torch.no_grad():
    output = model.generate(
        input_ids = input_ids, 
        attention_mask = attention_mask, 
        stopping_criteria = stopping_criteria,
        max_length=50000, 
        do_sample=True, 
        pad_token_id=tokenizer.eos_token_id
    )
    new_tokens = output[0][input_ids.shape[-1]:]
    generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True)

# print the output
# print(f"inputs: {prompt}")
print("\033[33mThe output text are as follows: \033[0m") 
print(f"Output: {generated_text}")

model_path = extract_model_path_from_llm_response(generated_text)


import time
import os
import retro
import pygame
from pygame.locals import *
from environments.sfai_wrapper import SFAIWrapper
from environments.multi_input_wrapper import MultiInputWrapper
from models.aux_obj_ppo import AuxObjPPO
from stable_baselines3 import PPO
from stable_baselines3.common.base_class import BaseAlgorithm

NUM_EPISODES = 3 # The number of episodes to play
MODEL_DIR = r"" # Specify the model directory.
MODEL_NAME ="28800000_steps_%PPO%self_play_ppo$w_0.4_r_0.2853500634809375$"
GAME_STATE = "PvP.RyuvsRyu"
DEFAULT_REWARD_KWARGS = {
    'reward_scale': 0.001,
    'raw_reward_coef': 1.0, # How much HP change is rewarded, only happens during the fighting (not round over)
    'special_move_reward': 0.0, # Reward for using special moves
    'special_move_bonus': 1.0, # Bonus for dealing damage with special moves. The raw reward will be multiplied by this value, so set it to 1.0 if you don't want to use this feature.
    'projectile_reward': 0.0, # Reward for using projectiles
    'projectile_bonus': 3.0, # Bonus for dealing damage with projectiles. The raw reward will be multiplied by this value, so set it to 1.0 if you don't want to use this feature.
    'distance_reward': 0.00, # Set it to positive value to encourage the agent to stay far from the enemy, negative value to encourage the agent to stay close to the enemy
    'distance_bonus': 0.0, # Bonus for dealing damage with projectiles. 0.0 means no bonus, positive/negative value means reward being far/close to the enemy
    'cost_coef': 0.0, # Ratio between the cost and the reward
    'special_move_cost': 2.0,
    'regular_attack_cost': 1.0,
    'jump_cost': 3.0,
    'vulnerable_frame_cost':0.1,
}

def action_translate(a):
    action = ''
    if a[4]==1:
        action = 'jump'
    elif a[5]==1:
        action = 'crouch'
    elif a[6]==1:
        action = 'left'
    elif a[7]==1:
        action = 'right'
    elif a[10]==1:
        action = 'light_punch'
    elif a[1]==1:
        action = 'light_kick'
    elif a[9]==1:
        action = 'medium_punch'
    elif a[0]==1:
        action = 'medium_kick'
    elif a[11]==1:
        action = 'hard_punch'
    elif a[8]==1:
        action = 'hard_kick'
    else:
        action = 'nothing'

    return action

# Create the gymretro game environment
def make_env(game, state, players = 1, reward_kwargs = None, character_flip_rate = 0.0):
    def _init():
        env = retro.make(
            game=game, 
            state=state, 
            use_restricted_actions=retro.Actions.FILTERED,
            obs_type=retro.Observations.IMAGE,
            players=players
        )
        # env = StreetFighterCustomWrapper(env, reset_round=True, rendering=True, reverse_env=False, render_interval= 0.1)
        env = SFAIWrapper(env, reset_round= True, rendering = True, rendering_interval= 0.005, reward_kwargs=reward_kwargs, character_flip_rate=character_flip_rate)
        return env
    return _init

def play_game(
        model_class: BaseAlgorithm = PPO,
        multi_input: bool = False,
        game_state: str = GAME_STATE,
        model_dir: str = MODEL_DIR,
        model_name: str = MODEL_NAME,
        reward_kwargs = DEFAULT_REWARD_KWARGS,
        character_flip_rate = 0.0
):
    game = "StreetFighterIISpecialChampionEdition-Genesis"
    env = make_env(
        game, state = game_state, 
        players = 2, 
        reward_kwargs = reward_kwargs,
        character_flip_rate = character_flip_rate
        )()
    if multi_input:
        env = MultiInputWrapper(env)

    print("loading model")
    # Load the agent's model
    print("model_name: ", model_name)
    model = model_class.load(os.path.join(model_dir, model_name))
    print("model loaded")

    # Initialize pygame
    pygame.init()
    screen = pygame.display.set_mode((320, 224))
    pygame.display.set_caption("Street Fighter II PvP")

    # Main loop for the game
    num_episodes = NUM_EPISODES
    print_info = False

    for _ in range(num_episodes):
        accumulated_reward = 0
        done = False
        running = True
        obs = env.reset()
        action24 = [0]*24 # The actions for both player(0~11) and the agent(12~23).
        # Loop for each episode
        num_steps = 0
        start_time = time.time()
        while running:
            time.sleep(0.003)
            num_steps += 1
            if num_steps % 100 == 0:
                print('FPS: ', num_steps/(time.time()-start_time))
                print('Total time: ', time.time()-start_time)
                start_time = time.time()
                num_steps = 0

            # Get the actions from the agent model
            action_agent, _states = model.predict(obs)
            # action24[12:24] = action_agent
            action24[0:12] = action_agent

            # Get the actions from the player
            keys = pygame.key.get_pressed()
            key_values = [keys[K_w], keys[K_s], keys[K_a], keys[K_d], keys[K_u], keys[K_i], keys[K_o], keys[K_j], keys[K_k], keys[K_l]]
            action_player = [0]*12
            if keys[K_w]:
                action_player[4]=1
            if keys[K_s]:
                action_player[5]=1
            if keys[K_a]:
                action_player[6]=1
            if keys[K_d]:
                action_player[7]=1
            if keys[K_u]:
                action_player[10]=1
            if keys[K_j]:
                action_player[1]=1
            if keys[K_i]:
                action_player[9]=1
            if keys[K_k]:
                action_player[0]=1
            if keys[K_o]:
                action_player[11]=1
            if keys[K_l]:
                action_player[8]=1
            if keys[K_1]:
                action_player[2]=1
            if keys[K_2]:
                action_player[3]=1
            
            # action24[0:12] = action_player
            action24[12:24] = action_player
            
            # Perform the actions
            obs, reward, done, info = env.step(action24) 

            accumulated_reward += reward
            # if reward != 0:
            #     print("reward of current step: ", reward)
            #     print("accumulated_reward: ", accumulated_reward)
            # print("The accumulated reward is : ", accumulated_reward)
            # print('Eneny HP: ', info['enemy_hp'])
            # print("Enemy HP previous: ",  info['info_sequence_buffer'][-1]['enemy_hp'])
            # print("Enemy HP changed: ", info['info_sequence_buffer'][-1]['enemy_hp'] != info['enemy_hp'])

            if info['round_countdown'] >= 30000 and info['round_countdown']<= 36000 and not print_info:
                # print('The info sequence buffer is : ', info['info_sequence_buffer'])
                # Save info to a file
                # Write the element of info in one line, and write the elements of info_sequence_buffer line by line.
                with open('info_sequence_buffer.txt', 'w') as f:
                    # First, write the elements of info except info_sequence_buffer.
                    for key in info:
                        if key != 'info_sequence_buffer':
                            f.write(key+':'+str(info[key])+'\n')
                    for i in info['info_sequence_buffer']:
                        f.write(str(i)+'\n')
                print_info = True

    


            
play_game(
    model_class = PPO,
    multi_input = False,
    game_state = GAME_STATE,
    model_dir = MODEL_DIR,
    model_name = model_path,
    reward_kwargs = DEFAULT_REWARD_KWARGS,
    character_flip_rate = 0.0
)

        
        




