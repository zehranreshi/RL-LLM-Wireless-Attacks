# llm_rl_agent.py — fixed for TRL v0.19  ────────────────────────────────────────
import torch
import json
import re
import random
import argparse
from collections import deque
from tqdm import tqdm
import torch.nn as nn


from transformers import (                   # ### PATCH: import GenerationConfig
    AutoTokenizer,
    GenerationConfig,
    BitsAndBytesConfig,
    AutoModelForCausalLM,
    TrainingArguments
)
from trl import (
    PPOTrainer as TRL_PPOTrainer,
    AutoModelForCausalLMWithValueHead,
    PPOConfig,
)

from simulation_environment import mock_run_simulation_and_get_reward
import simulation_environment

class DummyRewardModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy = nn.Linear(1, 1)

    def forward(self, *args, **kwargs):
        return torch.tensor([0.0], device=self.dummy.weight.device)

dummy_reward_model = DummyRewardModel()

class LLMAgent:
    def __init__(self, model_name: str, ppo_cfg: PPOConfig):       # ### PATCH
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"--- Using device: {self.device} ---")


        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )


        print(f"--- Loading model: {model_name}. This may take a while... ---")
        self.model = AutoModelForCausalLMWithValueHead.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.gradient_checkpointing_enable()


        self.value_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        self.value_model.gradient_checkpointing_enable()

        self.ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        self.ref_model.gradient_checkpointing_enable()

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # ### PATCH: ensure model has generation_config so PPOTrainer won’t crash
        if not hasattr(self.model, "generation_config"):
            try:
                gen_cfg = GenerationConfig.from_pretrained(
                    model_name, local_files_only=True
                )
            except (OSError, ValueError):
                gen_cfg = GenerationConfig()
            gen_cfg.eos_token_id = self.tokenizer.eos_token_id
            self.model.generation_config = gen_cfg

        # Build PPOTrainer with the dataclass config
        dummy_ds=[""]
        self.ppo_trainer = TRL_PPOTrainer(
            ppo_cfg,  
            model=self.model,                    
            processing_class=self.tokenizer,
            ref_model=self.ref_model,
            reward_model=dummy_reward_model,
            train_dataset=dummy_ds,
            value_model=self.value_model
        )
        print(f"--- Model {model_name} loaded successfully! ---")
        print("DEBUG: PPOTrainer type →", type(self.ppo_trainer))


    # -------------------------------------------------------------------------
    # unchanged helper methods
    # -------------------------------------------------------------------------
    def _parse_llm_output(self, text_output: str) -> dict:
        try:
            json_match = re.search(r'\{[^{}]*\}', text_output, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                if json_str.count("{") == json_str.count("}"):
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
            "min_length": -1,
            "top_p": 0.9,
            "do_sample": True,
            "pad_token_id": self.tokenizer.eos_token_id,
            "max_new_tokens": 150,
            "temperature": 1.1,
        }

        response_tensor = self.model.generate(tokenized_prompt, **generation_kwargs)
        prompt_len = tokenized_prompt.shape[1]
        response_text = self.tokenizer.decode(
            response_tensor[0][prompt_len:], skip_special_tokens=True
        )

        parsed_config = self._parse_llm_output(response_text)

        
        # Ensure the returned tensors are 1D for the .step() method
        return parsed_config, tokenized_prompt.squeeze(0), response_tensor.squeeze(0)
        # return parsed_config, tokenized_prompt, response_tensor
    



# In the LLMAgent class

# In the LLMAgent class

    def update(self, query_tensor, response_tensor, reward: float):
        """
        Performs a PPO update step using the correct 'step' method
        for online reinforcement learning. This method is specific to the
        TRL PPOTrainer and will NOT call the generic transformers.Trainer logic.
        """
        # The .step() method expects lists of 1D tensors.
        queries = [query_tensor]
        responses = [response_tensor]
        rewards = [torch.tensor(reward, device=self.device)]

        # This is the correct method call for the TRL PPO loop.
        stats = self.ppo_trainer.step(queries, responses, rewards)
        
        # Optionally log the statistics returned by the step.
        self.ppo_trainer.log_stats(stats, {}, rewards)




    # def update(self, query_tensor, response_tensor, reward: float):
    #     # 1. Create fake input/output pair — just echoing the prompt for now
    #     inputs = {
    #         "input_ids": query_tensor,         # what you fed into the model
    #         "labels": response_tensor,         # same shape, acts as target
    #         "attention_mask": torch.ones_like(query_tensor)  # to avoid warnings
    #     }

    #     # 2. Call training_step using model + input dict
    #     self.ppo_trainer.training_step(self.model, inputs)


# ──────────────────────────────────────────────────────────────────────────────
def format_history(history: deque) -> str:
    if not history:
        return "You have no previous attempts for this target."
    history_str = "### Previous Attempts and Rewards:\n"
    for config, reward in history:
        history_str += f"- Attempt: {json.dumps(config)}, Reward: {reward:.3f}\n"
    return history_str


def run_training_loop(args):
    print("--- Building PPOConfig ---")

    # ### PATCH: create a proper PPOConfig instead of a raw dict
    ppo_cfg = PPOConfig(
        learning_rate=args.learning_rate,
        batch_size=1,
        mini_batch_size=1,
        kl_coef=0.05,                  # static KL penalty for TRL v0.19
        stop_token_id=None,            # field expected by PPOTrainer
        output_dir="./results_ppo",
        bf16=True,
    )

    agent = LLMAgent(model_name=args.model, ppo_cfg=ppo_cfg)

    history = deque(maxlen=3)
    base_prompt = (
        "You are a world-class network security expert specializing in radio "
        "frequency analysis. Your task is to generate a precise JSON "
        "configuration for a radio jammer to neutralize a target signal. "
        "The JSON object must contain three keys: 'center_frequency' (float, in Hz), "
        "'bandwidth' (float, in Hz), and 'tx_gain' (float, from 0-90). "
        "Adhere strictly to the JSON format. Do not provide any other text, "
        "explanation, or markdown. Your entire output must be only the JSON object."
    )

    print(f"\n--- Starting RL Training with {args.model} ---")
    total_reward = 0
    high_score = 0
    pbar = tqdm(range(args.num_episodes), desc="Training Episodes")

    for episode in pbar:
        target_freq_ghz = round(random.uniform(3.5, 3.7), 4)
        simulation_environment.TARGET_FREQ = target_freq_ghz * 1e9
        prompt = (
            f"{base_prompt}\n\n### Current Mission\n"
            f"Your current mission is to generate a config to jam a target at "
            f"{target_freq_ghz:.4f} GHz.\n\n{format_history(history)}\n\n### JSON Output:\n"
        )

        config, query, response = agent.generate_action(prompt)
        if config is None:
            reward = -1.0
            history.append(({"error": "invalid JSON"}, reward))
        else:
            reward = mock_run_simulation_and_get_reward(config)
            history.append((config, reward))

        high_score = max(high_score, reward)
        total_reward += reward

        agent.update(query, response, reward)
        pbar.set_description(
            f"Ep {episode+1}/{args.num_episodes} | "
            f"Goal {target_freq_ghz:.4f} GHz | "
            f"Reward {reward:.3f} | High {high_score:.3f}"
        )

    print("\n--- Training Complete ---")
    avg_reward = total_reward / args.num_episodes
    print(f"Final Average Reward: {avg_reward:.3f}")
    print(f"Highest Reward Achieved: {high_score:.3f}")

    print("\n--- Saving trained model to disk ---")
    output_dir = f"jammer_agent_{args.model.split('/')[-1]}"
    agent.model.save_pretrained(output_dir)
    agent.tokenizer.save_pretrained(output_dir)
    print(f"Model saved successfully to ./{output_dir}")


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train an LLM Agent for Radio Jammer Configuration"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="codellama/CodeLlama-7b-Instruct-hf",
        choices=[
            "deepseek-ai/DeepSeek-Coder-V2-Instruct",
            # "meta-llama/Meta-Llama-3-8B-Instruct",
            "codellama/CodeLlama-7b-Instruct-hf"
        ],
        help="Hugging Face model name to train.",
    )
    parser.add_argument(
        "--num_episodes", type=int, default=50, help="Number of training episodes."
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-6, help="PPO learning rate."
    )
    run_training_loop(parser.parse_args())
