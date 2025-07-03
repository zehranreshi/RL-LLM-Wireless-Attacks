# llm_rl_agent.py — Corrected for the provided vintage TRL library
import torch
import json
import re
import random
import argparse
from tqdm import tqdm
import torch.nn as nn
from datasets import Dataset
# No need for copy, as deepcopy is unsafe for quantized models
# import copy

# All necessary imports from transformers and trl
from transformers import (
    AutoTokenizer,
    GenerationConfig,
    BitsAndBytesConfig,
    AutoModelForCausalLM,
    LlamaConfig,
    LlamaForSequenceClassification
)
from transformers.modeling_outputs import SequenceClassifierOutput
# This is the trainer class you are actually using
from trl import PPOTrainer as TRL_PPOTrainer
# This class is no longer needed, as we will build the models separately
# from trl import AutoModelForCausalLMWithValueHead
from trl import PPOConfig


# Your simulation environment is still needed for the reward logic
from simulation_environment import mock_run_simulation_and_get_reward
import simulation_environment


# ======================================================================================
# 1. THE COMPLIANT REWARD MODEL (This part was correct and remains unchanged)
# ======================================================================================
class SimulationRewardModel(LlamaForSequenceClassification):
    def __init__(self, tokenizer, model_name):
        config = LlamaConfig.from_pretrained(model_name)
        config.num_labels = 1  # Output is a single reward score
        super().__init__(config)
        self.tokenizer = tokenizer

    def _parse_llm_output(self, text_output: str) -> dict:
        try:
            json_match = re.search(r'\{[^{}]*\}', text_output, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                if json_str.count("{") == json_str.count("}"):
                    return json.loads(json_str)
            return None
        except json.JSONDecodeError:
            return None

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        rewards = []

        for i in range(input_ids.shape[0]):
            try:
                text = self.tokenizer.decode(input_ids[i], skip_special_tokens=True)
            except Exception:
                rewards.append(-1.0)
                continue

            if not text.strip():
                rewards.append(-1.0)
                continue

            json_part_start = text.rfind("### JSON Output:")
            response_text = text[json_part_start:] if json_part_start != -1 else text
            config = self._parse_llm_output(response_text)

            if config:
                try:
                    match = re.search(r'jam a target at ([\d.]+) GHz', text)
                    if match:
                        target_freq_ghz = float(match.group(1))
                        simulation_environment.TARGET_FREQ = target_freq_ghz * 1e9
                    else:
                        simulation_environment.TARGET_FREQ = 3.6e9
                except Exception:
                    simulation_environment.TARGET_FREQ = 3.6e9

                try:
                    reward = mock_run_simulation_and_get_reward(config)
                except Exception:
                    reward = -1.0
            else:
                reward = -1.0

            rewards.append(reward)

        reward_tensor = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        reward_tensor = torch.nan_to_num(reward_tensor, nan=-1.0, posinf=100.0, neginf=-100.0)
        reward_tensor = torch.clamp(reward_tensor, min=-100.0, max=100.0)

        # The older trainer version might expect a different output shape.
        # Let's ensure it's [batch_size, 1] as logits.
        return SequenceClassifierOutput(
            loss=None,
            logits=reward_tensor.unsqueeze(-1),
            hidden_states=None,
            attentions=None,
        )


class LLMAgent:
    def __init__(self, model_name: str, ppo_cfg: PPOConfig, train_dataset: Dataset):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"--- Using device: {self.device} ---")

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            # Use bfloat16 for better numerical stability with 4-bit models
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        # ============================= START: CORRECTED MODEL LOADING (for your TRL version) =============================
        # The key is to load each model independently from the hub to avoid the unstable `deepcopy`.

        # Step 1: Load the ACTIVE POLICY model. This is the one that will be trained.
        print(f"--- Loading policy model: {model_name} ---")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            device_map={"": self.device},
            trust_remote_code=True,
        )
        self.model.config.use_cache = False
        
        # Step 2: Load the REFERENCE model. This is a frozen copy of the original policy.
        print(f"--- Loading reference model: {model_name} ---")
        self.ref_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            device_map={"": self.device},
            trust_remote_code=True,
        )
        self.ref_model.eval() # Set to evaluation mode

        # Step 3: Load the VALUE model (critic). It has the same backbone but will have a custom head.
        print(f"--- Loading value model backbone: {model_name} ---")
        self.value_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            device_map={"": self.device},
            trust_remote_code=True,
        )
        
        # Step 4: Create and attach the `.score` head to the value model.
        # This is what `PolicyAndValueWrapper` in your ppo_trainer.py expects.
        hidden_size = self.value_model.config.hidden_size
        self.value_model.score = nn.Linear(hidden_size, 1, bias=False).to(self.device).to(torch.bfloat16)

        # ============================= END: CORRECTED MODEL LOADING =============================

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # The reward model setup was correct.
        self.reward_model = SimulationRewardModel(self.tokenizer, model_name)

        # Attach generation config to the main policy model.
        if not hasattr(self.model, "generation_config"):
            self.model.generation_config = GenerationConfig.from_pretrained(model_name)
        # Your ppo_trainer.py sets generation params inside the .train() loop,
        # but setting them here is good practice.
        self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id
        self.model.generation_config.eos_token_id = self.tokenizer.eos_token_id
        self.model.generation_config.max_new_tokens = ppo_cfg.response_length
        self.model.generation_config.temperature = ppo_cfg.temperature
        self.model.generation_config.do_sample = True
        
        # Initialize the trainer with all four separate models, as required by your library version.
        self.ppo_trainer = TRL_PPOTrainer(
            args=ppo_cfg,
            model=self.model,
            ref_model=self.ref_model,
            reward_model=self.reward_model,
            value_model=self.value_model,
            processing_class=self.tokenizer,
            train_dataset=train_dataset,
        )
        print(f"--- All models for PPO loaded successfully! ---")
        print("DEBUG: PPOTrainer type →", type(self.ppo_trainer))
        
def clear_cuda_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        print("[INFO] Cleared CUDA cache.")

def run_training_loop(args):
    print("--- Building PPOConfig ---")

    # This config is tailored to the arguments your ppo_trainer.py expects.
    # I have kept the parameters from your original script as they are likely compatible.
    ppo_cfg = PPOConfig(
        # Trainer/Loop settings
        exp_name="ppo_jammer_agent", # Your trainer uses this for run_name
        seed=42,
        total_episodes=args.num_episodes,
        num_ppo_epochs=4,
        # Batching settings
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        num_mini_batches=1, # Adjust if batch_size is large
        # PPO algorithm settings
        learning_rate=args.learning_rate,
        kl_coef=0.05,
        kl_estimator="k3", # As seen in your trainer code
        temperature=0.95,
        cliprange=0.2, # for pg_loss
        cliprange_value=0.2, # for vf_loss
        gamma=0.99,
        lam=0.95,
        vf_coef=0.1,
        # Generation settings
        response_length=150,
        # EOS/Stop token settings
        stop_token_id=None,
        missing_eos_penalty=None, # Set to a float like 1.0 to penalize
        # Logging & Saving
        output_dir="./results_ppo",
        logging_steps=1,
        # Performance
        num_sample_generations=0,
        bf16=True, # Use bfloat16
        fp16=False,
    )


    base_prompt = (
        "You are a world-class network security expert specializing in radio "
        "frequency analysis. Your task is to generate a precise JSON "
        "configuration for a radio jammer to neutralize a target signal. "
        "The JSON object must contain three keys: 'center_frequency' (float, in Hz), "
        "'bandwidth' (float, in Hz), and 'tx_gain' (float, from 0-90). "
        "Adhere strictly to the JSON format. Do not provide any other text, "
        "explanation, or markdown. Your entire output must be only the JSON object."
    )

    print("--- Creating and Tokenizing Dataset ---")
    raw_prompts = []
    # Your trainer calculates total batches based on total_episodes, so this size is reasonable.
    for _ in range(args.num_episodes):
        target_freq_ghz = round(random.uniform(3.5, 3.7), 4)
        prompt = (
            f"{base_prompt}\n\n### Current Mission\n"
            f"Your current mission is to generate a config to jam a target at "
            f"{target_freq_ghz:.4f} GHz.\n\n### JSON Output:\n"
        )
        raw_prompts.append({"query": prompt})

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset = Dataset.from_list(raw_prompts)

    # This tokenization step is mostly for creating the dataset object.
    # The trainer's internal collator will handle padding.
    def tokenize_function(examples):
        return tokenizer(examples["query"], truncation=True, padding=False, max_length=256)

    tokenized_dataset = train_dataset.map(
        tokenize_function, batched=True, remove_columns=["query"]
    )
    tokenized_dataset.set_format("torch")

    agent = LLMAgent(model_name=args.model, ppo_cfg=ppo_cfg, train_dataset=tokenized_dataset)

    print(f"\n--- Starting RL Training with {args.model} ---")
    # This is the correct call for your version of the trainer. It contains the full training loop.
    agent.ppo_trainer.train()

    print("\n--- Training Complete ---")

    print("\n--- Saving trained model to disk ---")
    # Your trainer saves the policy model automatically via its callbacks if configured.
    # But manual saving is also fine.
    output_dir = f"jammer_agent_{args.model.split('/')[-1]}_ppo_trained"
    agent.model.save_pretrained(output_dir)
    agent.tokenizer.save_pretrained(output_dir)
    print(f"Model saved successfully to ./{output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train an LLM Agent for Radio Jammer Configuration"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        choices=[
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "deepseek-ai/DeepSeek-Coder-6.7B-Instruct",
            "codellama/CodeLlama-7b-Instruct-hf"
        ],
        help="Hugging Face model name to train.",
    )
    parser.add_argument(
        "--num_episodes", type=int, default=50, help="Total number of training rollouts (episodes)."
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1.41e-6, help="PPO learning rate."
    )
    run_training_loop(parser.parse_args())