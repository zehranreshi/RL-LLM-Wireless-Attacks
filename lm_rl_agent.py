# llm_rl_agent.py — Corrected Version for TRL v0.19+
# ======================================================================================
# IMPORTS
# ======================================================================================
import torch
import torch.nn as nn
import json
import re
import random
import argparse
from tqdm import tqdm
from datasets import Dataset
from datetime import datetime

# Transformers and TRL Imports
from transformers import (
    AutoTokenizer,
    GenerationConfig,
    BitsAndBytesConfig,
    AutoModelForCausalLM,
    LlamaConfig,
    LlamaForSequenceClassification,
    LogitsProcessorList, 
    TemperatureLogitsWarper, 
    TopPLogitsWarper,
    AutoConfig,
    AutoModelForSequenceClassification
)
from transformers.modeling_outputs import SequenceClassifierOutput
from trl import (
    PPOTrainer as TRL_PPOTrainer,
    AutoModelForCausalLMWithValueHead,
    PPOConfig,
)

# Simulation Environment
from simulation_environment import mock_run_simulation_and_get_reward
import simulation_environment


# ======================================================================================
# REWARD MODEL DEFINITION
# ======================================================================================

class SimulationRewardModel(LlamaForSequenceClassification):
    """
    Custom Reward Model inheriting from LlamaForSequenceClassification to comply 
    with TRL's API expectations. It overrides the forward pass to use custom 
    simulation logic for calculating rewards.
    """
    def __init__(self, tokenizer, model_name):
        # Initialize with a dummy configuration to satisfy inheritance requirements.
        config = AutoConfig.from_pretrained(model_name)
        config.num_labels = 1  # Output is a single reward score
        super().__init__(config)
        self.tokenizer = tokenizer

    def _parse_llm_output(self, text_output: str) -> dict:
        """Helper function to extract JSON configuration from generated text."""
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
        """Overrides the standard forward pass to calculate rewards using the simulation."""
        rewards = []

        for i in range(input_ids.shape[0]):
            # Safe decoding checks
            if input_ids[i].nelement() == 0:
                rewards.append(-1.0)
                continue

            try:
                text = self.tokenizer.decode(input_ids[i], skip_special_tokens=True)
            except Exception:
                rewards.append(-1.0)
                continue

            if not text.strip():
                rewards.append(-1.0)
                continue

            # Extract JSON config from the generated response
            json_part_start = text.rfind("### JSON Output:")
            response_text = text[json_part_start:] if json_part_start != -1 else text
            config = self._parse_llm_output(response_text)

            if config:
                # Determine the target frequency from the prompt context
                try:
                    match = re.search(r'jam a target at ([\d.]+) GHz', text)
                    if match:
                        target_freq_ghz = float(match.group(1))
                        simulation_environment.TARGET_FREQ = target_freq_ghz * 1e9
                except Exception:
                    simulation_environment.TARGET_FREQ = 3.6e9 # Fallback target frequency

                # Calculate reward using the external simulation environment
                try:
                    reward = mock_run_simulation_and_get_reward(config)
                except Exception:
                    reward = -1.0
            else:
                # Penalize invalid JSON generation
                reward = -1.0

            rewards.append(reward)

        # Format the rewards into the structure expected by TRL
        reward_tensor = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        reward_tensor = torch.clamp(reward_tensor, min=-100.0, max=100.0)

        return SequenceClassifierOutput(
            loss=None,
            logits=reward_tensor.unsqueeze(-1), # Ensure correct tensor dimensions
            hidden_states=None,
            attentions=None,
        )


# ======================================================================================
# LLM AGENT DEFINITION
# ======================================================================================

class LLMAgent:
    def __init__(self, model_name: str, ppo_cfg: PPOConfig, train_dataset: Dataset):
        # Memory Management Configuration
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        torch.cuda.empty_cache()
        torch.cuda.set_per_process_memory_fraction(1.0, torch.cuda.current_device())

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"--- Using device: {self.device} ---")

        # Configuration for 4-bit quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            llm_int8_enable_fp32_cpu_offload=True,
            offload_buffers = True
        )

        print(f"--- Loading model: {model_name}. This may take a while... ---")

        # --- Model Initialization Strategy ---
        # Due to complex requirements in TRL's PPOTrainer regarding model structure
        # (PolicyAndValueWrapper and AutoModelForCausalLMWithValueHead), we employ
        # a strategy of loading plain models and manually configuring the value head.

        device_map="auto"
        # Step 1: Load a temporary model to initialize and extract the value head (v_head).
        temp_model_with_head = AutoModelForCausalLMWithValueHead.from_pretrained(
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0", 
            quantization_config=bnb_config, 
            torch_dtype=torch.bfloat16,
            device_map={"": self.device}, 
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        value_head_to_steal = temp_model_with_head.v_head
        
        # Step 2: Define the ACTIVE POLICY model. Must be a plain model for compatibility.
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            quantization_config=bnb_config, 
            torch_dtype=torch.float16,
            device_map=device_map, 
            trust_remote_code=True
        )
        self.model.gradient_checkpointing_enable()
        self.model.config.use_cache = False


        # Step 3: Define the CRITIC model. Load as plain, then attach the extracted value head.
        self.value_model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            quantization_config=bnb_config, 
            torch_dtype=torch.float16,
            device_map=device_map, 
            trust_remote_code=True
        )
        self.value_model.config.use_cache = False
        self.value_model.score = value_head_to_steal
        # self.value_model.gradient_checkpointing_enable()

        # Step 4: Define the FROZEN REFERENCE model. Must be a plain model.
        self.ref_model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            quantization_config=bnb_config, 
            torch_dtype=torch.float16,
            device_map=device_map, 
            trust_remote_code=True
        )

        # --- Tokenizer and Reward Model Setup ---
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.reward_model = SimulationRewardModel(self.tokenizer, model_name)

        # Ensure the policy model has a generation configuration for the trainer.
        if not hasattr(self.model, "generation_config"):
            self.model.generation_config = GenerationConfig.from_pretrained(model_name)
        self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id

        # --- PPOTrainer Initialization ---
        self.ppo_trainer = TRL_PPOTrainer(
            ppo_cfg, model=self.model, processing_class=self.tokenizer,
            ref_model=self.ref_model, reward_model=self.reward_model,
            train_dataset=train_dataset, value_model=self.value_model
        )
        print(f"--- Model {model_name} loaded successfully! ---")
        print("DEBUG: PPOTrainer type →", type(self.ppo_trainer))

        # --- Custom Generate Method (For Logging/Debugging) ---
        # This section patches the generate method to log outputs during training.
        
        def patched_generate(self, input_ids, **kwargs):
            # Setup standard logits processors for generation quality
            logits_processor = LogitsProcessorList([
                TemperatureLogitsWarper(temperature=0.7),
                TopPLogitsWarper(top_p=0.9),
            ])

            # Call the original generate method
            outputs = super(type(self), self).generate(
                input_ids=input_ids,
                logits_processor=logits_processor,
                max_new_tokens=150,
                **kwargs
            )

            # Decode the outputs for logging
            try:
                tokenizer = self.generation_config._tokenizer
            except:
                # Fallback tokenizer if not linked
                tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

            decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)

            # Print to terminal
            print("\n===================== GENERATED OUTPUT =====================")
            for i, output in enumerate(decoded):
                print(f"[Sample {i}]\n{output}\n")
            print("==============================================================\n")

            # Save to log file
            with open("generated_outputs.log", "a", encoding="utf-8") as f:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"\n=== Output generated at {timestamp} ===\n")
                for i, output in enumerate(decoded):
                    f.write(f"[Sample {i}]\n{output}\n\n")

            return outputs
        
        # Bind the patched method to the model instance
        self.ppo_trainer.model.generate = patched_generate.__get__(self.ppo_trainer.model, type(self.ppo_trainer.model))


# ======================================================================================
# TRAINING LOOP EXECUTION
# ======================================================================================

def run_training_loop(args):
    print("--- Building PPOConfig ---")

    # PPO Configuration settings
    ppo_cfg = PPOConfig(
        learning_rate=1.41e-6,
        warmup_steps=10,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        num_ppo_epochs=4,
        kl_coef=0.05,
        temperature=0.95,
        cliprange_value=0.2,
        max_grad_norm=0.5,
        vf_coef=0.1,
        stop_token_id=None,
        output_dir="./results_ppo",
        bf16=False,
        fp16=False,
        response_length=150,
        total_episodes=args.num_episodes,
        num_sample_generations=0 # Disable mid-training sampling/evaluation
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
    
    # Generate training prompts based on randomized target frequencies
    raw_prompts = []
    for _ in range(args.num_episodes * 2):
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

    # Convert prompts to a Hugging Face Dataset and tokenize
    train_dataset = Dataset.from_list(raw_prompts)

    def tokenize_function(examples):
        return tokenizer(examples["query"], truncation=True, padding=False)

    tokenized_dataset = train_dataset.map(
        tokenize_function, batched=True, remove_columns=["query"]
    )
    tokenized_dataset.set_format("torch")

    # Initialize the agent
    agent = LLMAgent(model_name=args.model, ppo_cfg=ppo_cfg, train_dataset=tokenized_dataset)

    print(f"\n--- Starting RL Training with {args.model} ---")
    agent.ppo_trainer.train()

    print("\n--- Training Complete ---")

    # Save the final trained model and tokenizer
    print("\n--- Saving trained model to disk ---")
    output_dir = f"jammer_agent_{args.model.split('/')[-1]}"
    agent.model.save_pretrained(output_dir)
    agent.tokenizer.save_pretrained(output_dir)
    print(f"Model saved successfully to ./{output_dir}")


# ======================================================================================
# MAIN EXECUTION BLOCK
# ======================================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train an LLM Agent for Radio Jammer Configuration"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="deepseek-ai/DeepSeek-Coder-V2-Instruct",
        choices=[
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "deepseek-ai/DeepSeek-Coder-V2-Instruct",
            "codellama/CodeLlama-7b-Instruct-hf"
        ],
        help="Hugging Face model name to train.",
    )
    parser.add_argument(
        "--num_episodes", type=int, default=50, help="Total number of training rollouts (episodes)."
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-6, help="PPO learning rate."
    )
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    run_training_loop(parser.parse_args())