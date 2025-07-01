import torch
import json
import re
import random
from collections import deque
from transformers import AutoTokenizer
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead
from simulation_environment import mock_run_simulation_and_get_reward
import simulation_environment
 
class LLMAgent:
    def __init__(self, model_name):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"--- Using device: {self.device} ---")
        self.ppo_config = PPOConfig(
            batch_size=1,
            mini_batch_size=1,
            learning_rate=1e-6, # Keep a low learning rate for this large model
            kl_penalty="kl",
            ppo_epochs=4,
        )
        print(f"--- Loading model: {model_name}. This may take a while... ---")
        # --- MODEL CHANGE 1: Loading DeepSeek Coder V2 ---
        self.model = AutoModelForCausalLMWithValueHead.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16, # Optimized for modern GPUs like the 4090
            trust_remote_code=True,
            device_map="auto" # Automatically handles GPU memory distribution
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # --------------------------------------------------
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
 
        self.ppo_trainer = PPOTrainer(
            model=self.model,
            config=self.ppo_config,
            tokenizer=self.tokenizer
        )
 
    def _parse_llm_output(self, text_output: str):
        try:
            # This regex is robust and should work for any model's JSON output
            json_match = re.search(r'\{.*\}', text_output, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                if json_str.count('{') == json_str.count('}'):
                    return json.loads(json_str)
            print(f"DEBUG: Failed to parse JSON from output: {text_output}")
            return {}
        except json.JSONDecodeError:
            print(f"DEBUG: JSONDecodeError from output: {text_output}")
            return {}
 
    def generate_action(self, prompt_text: str):
        # --- PROMPT FORMAT CHANGE 2: Use the model's specific chat template ---
        # This is the standard and most reliable way to prompt instruction-tuned models.
        messages = [{"role": "user", "content": prompt_text}]
        tokenized_prompt = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(self.device)
        # ----------------------------------------------------------------------
        generation_kwargs = {
            "min_length": -1, "top_p": 0.9, "do_sample": True,
            "pad_token_id": self.tokenizer.eos_token_id, "max_new_tokens": 100,
            "temperature": 1.1,
        }
        response_tensor = self.ppo_trainer.generate(tokenized_prompt.squeeze(0), **generation_kwargs)
        response_text = self.tokenizer.decode(response_tensor[0][tokenized_prompt.shape[1]:], skip_special_tokens=True)
        parsed_config = self._parse_llm_output(response_text)
        return parsed_config, tokenized_prompt, response_tensor
 
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
    # --- MODEL CHANGE 3: The new model name ---
    agent = LLMAgent(model_name="deepseek-ai/DeepSeek-Coder-V2-Instruct")
    # ------------------------------------------
    num_episodes = 100
    history = deque(maxlen=3)
    # Tailoring the prompt slightly to leverage the "coder" identity
    base_prompt = "You are a world-class network security expert specializing in radio frequency analysis. Your task is to generate a precise JSON configuration for a radio jammer. The keys must be 'center_frequency', 'bandwidth', and 'tx_gain'. Adhere strictly to the JSON format."
 
    print("--- Starting STATE-OF-THE-ART Training with DeepSeek Coder V2 ---")
    successful_episodes = 0
    total_reward = 0
    high_score = 0
 
    for episode in range(num_episodes):
        target_freq_ghz = round(random.uniform(3.5, 3.7), 3)
        dynamic_prompt_part = f"Your current mission is to generate a config to jam a target at {target_freq_ghz} GHz."
        history_str = format_history(history)
        prompt = f"{base_prompt}\n{dynamic_prompt_part}\n{history_str}"
 
        print(f"--- Episode {episode+1}/{num_episodes} | Goal: {target_freq_ghz} GHz | High Score: {high_score:.3f} ---")
        config, query, response = agent.generate_action(prompt)
        if not config:
            print("Failed to generate valid config. Skipping episode.")
            reward = -0.1
        else:
            successful_episodes += 1
            simulation_environment.TARGET_FREQ = target_freq_ghz * 1e9
            reward = mock_run_simulation_and_get_reward(config)
        if reward > high_score:
            high_score = reward
        if config:
             history.append((config, reward))
 
        total_reward += reward
        agent.update(query, response, reward)
 
    print(f"\n--- Training Complete ---")
    print(f"Successfully generated valid configs in {successful_episodes}/{num_episodes} episodes.")
    print(f"Final Average Reward: {total_reward / num_episodes:.3f}")
    print(f"\n--- Saving trained model to disk ---")
    agent.model.save_pretrained("jammer_agent_deepseek_v1")
    agent.tokenizer.save_pretrained("jammer_agent_deepseek_v1")
    print("Model saved successfully!")
 
if __name__ == '__main__':
    run_training_loop()