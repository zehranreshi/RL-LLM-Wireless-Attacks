import torch
import json
import re
import random
from collections import deque
from transformers import AutoTokenizer
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead
from simulation_environment import mock_run_simulation_and_get_reward
# This import is needed to modify the global variable
import simulation_environment

class LLMAgent:
    def __init__(self, model_name="gpt2"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"--- Using device: {self.device} ---")
        
        self.ppo_config = PPOConfig(
            batch_size=1,
            mini_batch_size=1,
            learning_rate=5e-6,
            kl_penalty="kl",
            ppo_epochs=4, # Make bigger leaps in learning
        )
        
        self.model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model.to(self.device)

        self.ppo_trainer = PPOTrainer(
            model=self.model,
            config=self.ppo_config,
            tokenizer=self.tokenizer
        )

    def _parse_llm_output(self, text_output: str):
        try:
            json_match = re.search(r'\{.*\}', text_output, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                if json_str.count('{') == json_str.count('}'):
                    return json.loads(json_str)
            return {}
        except json.JSONDecodeError:
            return {}

    def generate_action(self, prompt_text: str):
        query_tensor = self.tokenizer.encode(prompt_text, return_tensors="pt").to(self.device)
        
        generation_kwargs = {
            "min_length": -1,
            "top_k": 0.0,
            "top_p": 1.0,
            "do_sample": True,
            "pad_token_id": self.tokenizer.eos_token_id,
            "max_new_tokens": 70,
            "temperature": 1.2,
        }
        
        response_tensor = self.ppo_trainer.generate(query_tensor.squeeze(0), **generation_kwargs)
        response_text = self.tokenizer.decode(response_tensor[0], skip_special_tokens=True)
        
        parsed_config = self._parse_llm_output(response_text)
        return parsed_config, query_tensor, response_tensor

    def update(self, query_tensor, response_tensor, reward):
        reward_tensor = torch.tensor([reward]).to(self.device)
        self.ppo_trainer.step([query_tensor.squeeze(0)], [response_tensor.squeeze(0)], [reward_tensor])

def format_history(history: deque):
    if not history:
        return ""
    history_str = "### Previous Attempts:\n"
    for config, reward in history:
        history_str += f"Attempt: {json.dumps(config)}, Reward: {reward:.3f}\n"
    return history_str

def run_training_loop():
    agent = LLMAgent()
    num_episodes = 50
    
    history = deque(maxlen=3)
    
    base_prompt = (
        "Generate a JSON config for a wireless jammer with keys 'center_frequency', 'bandwidth', and 'tx_gain'. "
        "The goal is to maximize the reward.\n"
    )

    print("--- Starting Advanced Training with Dynamic Scenarios and Memory ---")
    successful_episodes = 0
    total_reward = 0

    for episode in range(num_episodes):
        target_freq_ghz = round(random.uniform(3.5, 3.7), 3)
        dynamic_prompt_part = f"### Current Goal:\nJam a target at {target_freq_ghz} GHz.\n"
        
        history_str = format_history(history)
        prompt = base_prompt + dynamic_prompt_part + history_str + "\n### Response:\n"

        print(f"--- Episode {episode+1}/{num_episodes} ---")
        print(f"Goal: Jam {target_freq_ghz} GHz")

        config, query, response = agent.generate_action(prompt)
        
        if not config:
            print("Failed to generate valid config. Skipping episode.")
            reward = -0.1
        else:
            successful_episodes += 1
            # This line updates the global TARGET_FREQ in the other file for this episode
            simulation_environment.TARGET_FREQ = target_freq_ghz * 1e9
            
            reward = mock_run_simulation_and_get_reward(config)
        
        if config:
             history.append((config, reward))

        total_reward += reward
        agent.update(query, response, reward)

    print(f"\n--- Training Complete ---")
    print(f"Successfully generated valid configs in {successful_episodes}/{num_episodes} episodes.")
    print(f"Final Average Reward: {total_reward / num_episodes:.3f}")

if __name__ == '__main__':
    run_training_loop()