# llm_rl_agent.py (Corrected for the dataset-based .train() method)
import torch
import json
import re
import random
import argparse
from collections import deque
from tqdm import tqdm
import torch.nn as nn
from datasets import Dataset
from functools import partial
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

from transformers import (
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

# class DummyRewardModel(nn.Module):
#     """
#     A dummy reward model required by the PPOTrainer's API.
#     It always returns 0. The real reward is calculated by the KL penalty.
#     For this to be effective, we rely on the KL divergence to guide the model.
#     A more advanced setup would involve a custom PPOTrainer class.
#     """
#     def __init__(self):
#         super().__init__()
#         self.dummy = nn.Linear(1, 1)

#     def forward(self, *args, **kwargs):
#         # Return a tensor of zeros with the correct batch size and device
#         input_ids = args[0]
#         batch_size = input_ids.shape[0]
#         return torch.zeros(batch_size, device=self.dummy.weight.device)

class DummyRewardModel(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, *args, **kwargs):
        return torch.zeros(args[0].size(0), device=args[0].device)
    def score(self, hidden_states):
        return torch.zeros(hidden_states.size(0), device=hidden_states.device)

dummy_reward_model = DummyRewardModel()

class LLMAgent:
    """
    This agent class now primarily serves to initialize and hold all the
    necessary components for the PPOTrainer. The training logic itself
    is handled by the PPOTrainer's .train() method.
    """
    def __init__(self, model_name: str, ppo_cfg: PPOConfig, tokenizer):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"--- Using device: {self.device} ---")

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        print(f"--- Loading model components for {model_name}. This may take a while... ---")
        self.model = AutoModelForCausalLMWithValueHead.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        # self.model.gradient_checkpointing_enable()
        self.model.config.return_dict = True


        self.value_model = AutoModelForCausalLMWithValueHead.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        # self.value_model.gradient_checkpointing_enable()
        self.value_model.config.return_dict = True

        self.ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        # self.ref_model.gradient_checkpointing_enable()
        self.ref_model.config.return_dict = True


        def enforce_return_dict(model):
            class WrappedModel(model.__class__):
                def forward(self_inner, *args, **kwargs):
                    output = super(WrappedModel, self_inner).forward(*args, **kwargs)
                    if isinstance(output, tuple):
                        return CausalLMOutputWithCrossAttentions(logits=output[0])
                    return output
            model.__class__ = WrappedModel

        enforce_return_dict(self.model)
        enforce_return_dict(self.ref_model)
        enforce_return_dict(self.value_model)


        self.tokenizer = AutoTokenizer.from_pretrained(model_name)    
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if not hasattr(self.model, "generation_config"):
            try:
                gen_cfg = GenerationConfig.from_pretrained(model_name, local_files_only=True)
            except (OSError, ValueError):
                gen_cfg = GenerationConfig()
            gen_cfg.eos_token_id = self.tokenizer.eos_token_id
            self.model.generation_config = gen_cfg
        
        # This dataset will be used by the .train() method's internal loop
        # We create a dummy dataset of prompts.
        base_prompt = (
            "You are a world-class network security expert. Generate a precise JSON "
            "configuration for a radio jammer. The keys must be 'center_frequency', "
            "'bandwidth', and 'tx_gain'. Adhere strictly to the JSON format."
        )
        prompts = [base_prompt] * 12 # A dataset of 200 prompts
        raw_dataset = Dataset.from_dict({"query": prompts})

        def tokenize_fn(example):
            return self.tokenizer(
                example["query"],
                truncation=True,
                padding="max_length",
                max_length=512,
            )

        tokenized_ds = raw_dataset.map(tokenize_fn, batched=True)
        tokenized_ds = tokenized_ds.remove_columns(["query"]) 

        # The PPOTrainer from the provided source code requires all these components.
        self.ppo_trainer = TRL_PPOTrainer(
            args=ppo_cfg,
            model=self.model,
            ref_model=self.ref_model,
            processing_class=self.tokenizer,
            reward_model=dummy_reward_model,
            train_dataset=tokenized_ds,
            value_model=self.value_model,
        )

        print(f"--- LLM Agent and PPOTrainer Initialized Successfully ---")

# In your script, replace the existing run_training_loop with this corrected version.

def run_training_loop(args):
    """
    This function sets up the configuration and dataset, then calls the
    PPOTrainer's main .train() method to handle the entire RL loop.
    """
    print("--- Building PPOConfig ---")

    # The PPOConfig holds all settings for the trainer.
    ppo_cfg = PPOConfig(
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        mini_batch_size=args.mini_batch_size,
        kl_coef=0.05,
        output_dir="./results_ppo",
        bf16=True, # Assumes modern GPU for performance
        total_episodes=args.num_episodes,
    )

    # STEP 1: Define the custom reward function
    def custom_get_reward(model, query_responses, pad_token_id, context_length, tokenizer_ref):
        full_value, score, _ = original_get_reward(model, query_responses, pad_token_id, context_length)
        decoded_responses = tokenizer_ref.batch_decode(query_responses)
        custom_rewards = []
        for response_text in decoded_responses:
            reward = -1.0
            try:
                json_match = re.search(r'\{[^{}]*\}', response_text, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                    config = json.loads(json_str)
                    target_freq_ghz = round(random.uniform(3.5, 3.7), 4)
                    simulation_environment.TARGET_FREQ = target_freq_ghz * 1e9
                    reward = mock_run_simulation_and_get_reward(config)
            except Exception:
                reward = -1.0
            custom_rewards.append(reward)
        final_scores = torch.tensor(custom_rewards, device=full_value.device, dtype=full_value.dtype)
        return full_value, final_scores, None

    # STEP 2: Initialize the tokenizer FIRST
    print(f"--- Initializing tokenizer for {args.model} ---")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # STEP 3: Now that the tokenizer exists, we can patch the reward function
    from trl import trainer
    from functools import partial
    original_get_reward = trainer.utils.get_reward
    trainer.utils.get_reward = partial(custom_get_reward, tokenizer_ref=tokenizer)
    print("--- Custom reward function has been patched into TRL ---")

    # STEP 4: Now, initialize the agent, which will create all the models
    # and the PPOTrainer. The PPOTrainer will now use our patched reward function.
    agent = LLMAgent(model_name=args.model, ppo_cfg=ppo_cfg, tokenizer=tokenizer)
    
    print(f"\n--- Starting RL Training with {args.model} ---")
    
    # Call .train() ONCE. It handles the entire training loop internally.
    agent.ppo_trainer.train()
    
    print("\n--- Training Complete ---")

    print("\n--- Saving trained model to disk ---")
    output_dir = f"jammer_agent_{args.model.split('/')[-1]}"
    agent.ppo_trainer.save_model(output_dir)
    agent.tokenizer.save_pretrained(output_dir)
    print(f"Model saved successfully to ./{output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an LLM Agent with TRL PPOTrainer")
    parser.add_argument("--model", type=str, default="codellama/CodeLlama-7b-Instruct-hf", help="Hugging Face model name to train.")
    parser.add_argument("--num_episodes", type=int, default=1000, help="Total training steps (episodes).")
    parser.add_argument("--learning_rate", type=float, default=1.41e-5, help="PPO learning rate.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for PPO.")
    parser.add_argument("--mini_batch_size", type=int, default=1, help="Mini-batch size for PPO.")
    args = parser.parse_args()
    run_training_loop(args)