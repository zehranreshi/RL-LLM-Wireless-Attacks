# llm_rl_agent.py (Final Version - Bypassing PPOConfig)
import torch
import json
import re
import random
import argparse
from collections import deque
from tqdm import tqdm

from transformers import AutoTokenizer
from trl import PPOTrainer, AutoModelForCausalLMWithValueHead

from simulation_environment import mock_run_simulation_and_get_reward
import simulation_environment

class LLMAgent:
    def __init__(self, model_name: str, ppo_params: dict):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"--- Using device: {self.device} ---")
        
        print(f"--- Loading model: {model_name}. This may take a while... ---")
        self.model = AutoModelForCausalLMWithValueHead.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # THE FINAL CORRECTION: Pass the manually-built dictionary directly
        self.ppo_trainer = PPOTrainer(
            model=self.model,
            **ppo_params
        )
        print(f"--- Model {model_name} loaded successfully! ---")

    def _parse_llm_output(self, text_output: str) -> dict:
        try:
            json_match = re.search(r'\{[^{}]*\}', text_output, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                if json_str.count('{') == json_str.count('}'):
                    return json.loads(json_str)
            print(f"DEBUG: Failed to parse JSON from output: {text_output}")
            return None
        except json.JSONDecodeError:
            print(f"DEBUG: JSONDecodeError from output: {text_output}")
            return None

    def generate_action(self, prompt_text: str):
        messages = [{"role": "user", "content": prompt_text}]
        tokenized_prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to(self.device)

        generation_kwargs = {
            "min_length": -1, "top_p": 0.9, "do_sample": True,
            "pad_token_id": self.tokenizer.eos_token_id, "max_new_tokens": 150,
            "temperature": 1.1,
        }
        
        response_tensor = self.model.generate(tokenized_prompt, **generation_kwargs)
        prompt_len = tokenized_prompt.shape[1]
        response_text = self.tokenizer.decode(response_tensor[0][prompt_len:], skip_special_tokens=True)
        
        parsed_config = self._parse_llm_output(response_text)
        return parsed_config, tokenized_prompt, response_tensor

    def update(self, query_tensor, response_tensor, reward: float):
        reward_tensor = torch.tensor([reward], device=self.device)
        self.ppo_trainer.step([query_tensor.squeeze(0)], [response_tensor.squeeze(0)], [reward_tensor])

def format_history(history: deque) -> str:
    if not history:
        return "You have no previous attempts for this target."
    history_str = "### Previous Attempts and Rewards:\n"
    for config, reward in history:
        history_str += f"- Attempt: {json.dumps(config)}, Reward: {reward:.3f}\n"
    return history_str

def run_training_loop(args):
    """The main training loop."""
    
    print("--- Building PPO configuration dictionary manually ---")
    
    # Manually create the dictionary of parameters for the PPOTrainer
    ppo_params = {
        "learning_rate": args.learning_rate,
        "batch_size": 1,
        "mini_batch_size": 1,
        "adap_kl_ctrl": True,
        "init_kl_coef": 0.2,
        "target_kl": 0.1,
    }

    agent = LLMAgent(
        model_name=args.model,
        ppo_params=ppo_params
    )
    
    history = deque(maxlen=3)
    base_prompt = (
        "You are a world-class network security expert specializing in radio frequency analysis. "
        "Your task is to generate a precise JSON configuration for a radio jammer to neutralize a target signal. "
        "The JSON object must contain three keys: 'center_frequency' (float, in Hz), 'bandwidth' (float, in Hz), and 'tx_gain' (float, from 0-90). "
        "Adhere strictly to the JSON format. Do not provide any other text, explanation, or markdown. Your entire output must be only the JSON object."
    )
    print(f"\n--- Starting RL Training with {args.model} ---")
    total_reward = 0
    high_score = 0
    pbar = tqdm(range(args.num_episodes), desc="Training Episodes")
    for episode in pbar:
        target_freq_ghz = round(random.uniform(3.5, 3.7), 4)
        simulation_environment.TARGET_FREQ = target_freq_ghz * 1e9
        dynamic_prompt_part = f"Your current mission is to generate a config to jam a target at {target_freq_ghz:.4f} GHz."
        history_str = format_history(history)
        prompt = (f"{base_prompt}\n\n### Current Mission\n{dynamic_prompt_part}\n\n{history_str}\n\n### JSON Output:\n")
        config, query, response = agent.generate_action(prompt)
        if config is None:
            reward = -1.0
            history.append(({"error": "invalid JSON"}, reward))
        else:
            reward = mock_run_simulation_and_get_reward(config)
            history.append((config, reward))
        if reward > high_score:
            high_score = reward
        total_reward += reward
        agent.update(query, response, reward)
        pbar.set_description(f"Episode {episode+1}/{args.num_episodes} | Goal: {target_freq_ghz:.4f}GHz | Reward: {reward:.3f} | High Score: {high_score:.3f}")

    print("\n--- Training Complete ---")
    avg_reward = total_reward / args.num_episodes
    print(f"Final Average Reward: {avg_reward:.3f}")
    print(f"Highest Reward Achieved: {high_score:.3f}")
    print("\n--- Saving trained model to disk ---")
    output_dir = f"jammer_agent_{args.model.split('/')[-1]}"
    agent.model.save_pretrained(output_dir)
    agent.tokenizer.save_pretrained(output_dir)
    print(f"Model saved successfully to ./{output_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train an LLM Agent for Radio Jammer Configuration")
    parser.add_argument(
        "--model", type=str, default="deepseek-ai/DeepSeek-Coder-V2-Instruct",
        help="Hugging Face model name to train.",
        choices=["deepseek-ai/DeepSeek-Coder-V2-Instruct", "meta-llama/Meta-Llama-3-8B-Instruct"]
    )
    parser.add_argument("--num_episodes", type=int, default=50, help="Number of training episodes.")
    parser.add_argument("--learning_rate", type=float, default=1e-6, help="PPO learning rate.")
    args = parser.parse_args()
    run_training_loop(args)