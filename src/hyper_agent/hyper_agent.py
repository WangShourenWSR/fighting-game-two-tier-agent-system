import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers import StoppingCriteria, StoppingCriteriaList
import re
import json
import os

class StreamStoppingCriteria(StoppingCriteria):
    """
    A custom stopping criteria that prints each newly generated token as it is produced.
    """
    def __init__(self, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs):
        # input_ids shape: [batch_size, current_sequence_length]
        # we take the last generated token from the first batch
        last_token_id = input_ids[0, -1]
        # decode it to string
        token_str = self.tokenizer.decode(last_token_id, skip_special_tokens=False)
        print(token_str, end="", flush=True)
        # return False => don't stop
        return False

class HyperAgent:
    def __init__(
            self, 
            model_path = "C:/Users/SR.W/LLMs/DeepSeek-R1-Distill-Llama-8B/snapshots/6a6f4aa4197940add57724a7707d069478df56b1", 
            quantization_config = BitsAndBytesConfig(load_in_8bit=True),  # 启用 8-bit 量化
            agents_archive: dict = None,
            state_files: dict = None,
        ):
        # Set device to cuda 
        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            raise RuntimeError("CUDA is not available.")
        
        # Load the model and tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config = quantization_config
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        if state_files is None:
            raise ValueError("state_files must be provided.")
        self.state_files = state_files

        # Initialize the game-playing information from the game_manager
        if agents_archive is None:
            raise ValueError("agents_archive must be provided.")
        agents_archive_str = json.dumps(agents_archive, indent=2)
        self.agents_archive_str = agents_archive_str.replace("{", "{{").replace("}", "}}")
        # Load the templates for prompt
        with open("src/hyper_agent/prompt_template.txt", "r") as f:
            self.prompt_template = f.read()
        # load the selection_principles file
        with open("src/hyper_agent/selection_principles.txt", "r") as f:
            self.selection_principles = f.read()
        # load the output format requirement file
        with open("src/hyper_agent/output_format_requirement.txt", "r") as f:
            self.output_format_requirement = f.read()
        # load the few shot example file
        with open("src/hyper_agent/few_shot_examples.txt", "r") as f:
            self.few_shot_examples = f.read()

        # Set the stopping criteria
        self.stream_stopping_criteria = StreamStoppingCriteria(self.tokenizer)
        self.stopping_criteria_list = StoppingCriteriaList([self.stream_stopping_criteria])

    def select_agent(
            self,
            playing_data: dict = None,
    ):
        """
        Selects an agent for the player based on the playing_data.
        
        Args:
            playing_data (dict): The playing data for the player. Should be passed from the game_manager.

        Returns:
            agent_type (str): The type of the selected agent.
            agent_model_path (str): The path to the agent model selected by the HyperAgent.
            character_name (str): The name of the character for the selected agent.
            state_file_name (str): The name of the state file for the selected agent.
        """
        def extract_agent_info_from_llm_output(llm_output: str):
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
            # potential_jsons = re.findall(r"\{.*?\}", llm_output, re.DOTALL)
            try:    
                potential_jsons = re.findall(r"\{(?:[^{}]*|\{.*?\})*\}", llm_output, re.DOTALL)
                
                # Reorganize the potential_jsons(str) to a dictionary. The origianl potential_jsons contains '\n' and '\t' characters.
                # output_dict = [json.loads(json_str) for json_str in potential_jsons]
                output_dict = json.loads(potential_jsons[-1])
            except:
                print("Hyper-agent did not successfully generate a response.")
                return None, None, None

            if not output_dict:
                print("Hyper-agent did not successfully generate a response.")
                return None, None, None
            
            # The loop to find the last valid JSON block containing the required key
            valid_agent_model_paths = []
            valid_agent_character = []
            valid_agent_type = []

            for key, value in output_dict.items():
                if key == "chosen_agent_model_path":
                    valid_agent_model_paths.append(value)
                elif key == "chosen_agent_character":
                    valid_agent_character.append(value)
                elif key == "chosen_agent_type":
                    valid_agent_type.append(value)
            
            if not valid_agent_model_paths:
                print("Hyper-agent did not successfully generate a response of valid model path.")
                return None, None, None
            if not valid_agent_character:
                print("Hyper-agent did not successfully generate a response of valid character.")
                return None, None, None
            if not valid_agent_type:
                print("Hyper-agent did not successfully generate a response of valid agent type.")
                return None, None, None
            
            # Return the last valid Json block
            return valid_agent_type[-1], valid_agent_model_paths[-1], valid_agent_character[-1]
        
        if playing_data is None:
            raise ValueError("playing_data must be provided.")
        
        # Apply the contents to the prompt template
        prompt = self.prompt_template.format(
            SELECTION_PRINCIPLES = self.selection_principles,
            PLAYING_DATA = json.dumps(playing_data, indent=2),
            ARCHIVE_INFO = self.agents_archive_str,
            OUTPUT_FORMAT_REQUIREMENT = self.output_format_requirement,
            FEW_SHOT_EXAMPLES = self.few_shot_examples,
        )
        
        input = self.tokenizer(
            prompt,
            padding = True,
            truncation = True,
            return_tensors = "pt" 
        )
        input_ids = input["input_ids"].to(self.device)
        attention_mask = input["attention_mask"].to(self.device)

        # print("\033[3;36mPrompt: ")
        # print(prompt)
        # print("\033[0m")


        # Inference
        while True:
            with torch.no_grad():
                print("\033[3;36mFrom hyper-agent: \033[0m")
                output = self.model.generate(
                    input_ids = input_ids,
                    attention_mask = attention_mask,
                    do_sample = True,
                    max_length = 4000,
                    pad_token_id = self.tokenizer.eos_token_id,
                    stopping_criteria = self.stopping_criteria_list
                )
                new_tokens = output[0][input_ids.shape[-1]:]
                generated_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

                

            # Extract the agent information from the generated text
            agent_type, agent_model_path, character_name = extract_agent_info_from_llm_output(generated_text)

            # Get the state file name based on the character name and the playing_data
            state_file_name = "PvP.{}Vs{}".format(playing_data["current_character"], character_name)
            # Check if the agent_model_path, character_name, and state_file_name are valid
            if self.check_llm_output(agent_type, agent_model_path, character_name, state_file_name):
                break
            else:
                print("Re-generating the agent information...")

        return agent_type, agent_model_path, character_name, state_file_name, 
    

    def check_llm_output(
            self,
            agent_type: str,
            agent_model_path: str,
            character_name: str,
            state_file_name: str,
    ):
        if agent_type is not None and agent_model_path is not None and character_name is not None and state_file_name is not None:
                # Check if the agent_model_path, character_name, and state_file_name are valid (means they exist)
                # Agent type:
                agent_type_exists = False
                agent_types = ['special_move_type', 'projectile_type', 'defensive_type', 'aggressive_type','coward_type', 'newbie_type', 'air_type']
                if agent_type in agent_types:
                    agent_type_exists = True
                else:
                    print("The generated agent type is invalid.")
                    
                # State files:
                state_file_exists = False
                for player1_character_name, player2_dict in self.state_files.items():
                    for player2_character_name, state_file in player2_dict.items():
                        if state_file_name == state_file:
                            state_file_exists = True
                            break
                    if state_file_exists:
                        break
                if not state_file_exists:
                    print("The generated state file name is invalid.")

                # Agent model path:
                # Agent model path the path of the .zip agent model file, just check if it does exist
                agent_model_path_exists = False
                if os.path.exists(agent_model_path+'.zip'):
                    agent_model_path_exists = True
                else:
                    print("The generated agent model path is invalid.")

                # Character name:
                # 2 steps checking: 1, if the character name is in all 12 available characters; 2, if the character name is in the state files
                character_name_exists = False
                available_characters = ['Ryu', 'Ken', 'Chunli', 'EHonda', 'Blanka', 'Guile', 'Zangief', 'Dhalsim', 'Balrog', 'Vega', 'Sagat', 'Bison']
                if character_name in available_characters:
                    # Check if the character name is in the state file's name (e.g., "PvP.RyuVsRyu"). 
                    # Just check sub-str after 'Vs'.
                    if character_name == state_file_name.split("Vs")[1]:
                        character_name_exists = True
                else:
                    print("The character name is invalid.")

                if agent_type_exists and state_file_exists and agent_model_path_exists and character_name_exists:
                    return True
                else:
                    print("The generated agent information is invalid.")
                    return False
        
        else:
            print("Missing arguments in the generated agent information.")
            return False

   

