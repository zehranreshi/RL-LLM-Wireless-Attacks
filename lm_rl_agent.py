# llm_rl_agent.py — Final Corrected Version for TRL v0.19+
import torch
import json
import re
import random
import argparse
from tqdm import tqdm
import torch.nn as nn
from datasets import Dataset

# All necessary imports from transformers and trl
from transformers import (
    AutoTokenizer,
    GenerationConfig,
    BitsAndBytesConfig,
    AutoModelForCausalLM,
    LlamaConfig, # Needed for the custom reward model
    LlamaForSequenceClassification # Needed for the custom reward model
)
from transformers.modeling_outputs import SequenceClassifierOutput # Needed for the custom reward model
from trl import (
    PPOTrainer as TRL_PPOTrainer,
    AutoModelForCausalLMWithValueHead,
    PPOConfig,
)

# Your simulation environment is still needed for the reward logic
from simulation_environment import mock_run_simulation_and_get_reward
import simulation_environment


# ======================================================================================
# 1. THE COMPLIANT REWARD MODEL
# This class inherits from a standard HF model to satisfy the trainer's expectations,
# but uses our custom Python logic in its forward pass.
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
            # === Safe decoding ===
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

            # === Extract JSON config and target frequency ===
            json_part_start = text.rfind("### JSON Output:")
            response_text = text[json_part_start:] if json_part_start != -1 else text
            config = self._parse_llm_output(response_text)

            if config:
                try:
                    match = re.search(r'jam a target at ([\d.]+) GHz', text)
                    if match:
                        target_freq_ghz = float(match.group(1))
                        simulation_environment.TARGET_FREQ = target_freq_ghz * 1e9
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
        reward_tensor = torch.clamp(reward_tensor, min=-100.0, max=100.0)

        return SequenceClassifierOutput(
            loss=None,
            logits=reward_tensor.unsqueeze(-1), # <--- ADD .unsqueeze(-1) HERE
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
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        print(f"--- Loading model: {model_name}. This may take a while... ---")

        # ============================ START: THE FINAL, UNIFIED SOLUTION ============================
        #
        # Step 1: Load a temporary model with a value head JUST to steal its `v_head`.
        # This is the key to breaking the configuration cycle.
        #
        temp_model_with_head = AutoModelForCausalLMWithValueHead.from_pretrained(
            model_name, quantization_config=bnb_config, torch_dtype=torch.float16,
            device_map="auto", trust_remote_code=True
        )
        value_head_to_steal = temp_model_with_head.v_head
        # We can now let this temporary model be garbage collected.

        #
        # Step 2: The ACTIVE POLICY model. It MUST be a PLAIN model to avoid the nested tuple error.
        # The trainer's internal PolicyAndValueWrapper will handle everything else.
        #
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=bnb_config, torch_dtype=torch.float16,
            device_map="auto", trust_remote_code=True
        )

        self.model.config.use_cache = False
        self.model.gradient_checkpointing_enable()
        #
        # Step 3: The CRITIC model. Load as a PLAIN model, then attach the stolen value head.
        # This creates our perfect hybrid critic that has both `.base_model_prefix` and `.score`.
        #
        self.value_model = AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=bnb_config, torch_dtype=torch.float16,
            device_map="auto", trust_remote_code=True
        )
        self.value_model.config.use_cache = False
        self.value_model.score = value_head_to_steal # Attach the stolen head
        self.value_model.gradient_checkpointing_enable()

        #
        # Step 4: The FROZEN REFERENCE model. This was already correct. It's a PLAIN model.
        #
        self.ref_model = AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=bnb_config, torch_dtype=torch.float16,
            device_map="auto", trust_remote_code=True
        )
        #


        # ============================= END: THE FINAL, UNIFIED SOLUTION =============================


        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Use our compliant SimulationRewardModel. This was also correct.
        self.reward_model = SimulationRewardModel(self.tokenizer, model_name)

        # Attach generation config to the main policy model.
        if not hasattr(self.model, "generation_config"):
            self.model.generation_config = GenerationConfig.from_pretrained(model_name)
        self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id
        
        # All models are now correctly configured. This will initialize and run.
        self.ppo_trainer = TRL_PPOTrainer(
            ppo_cfg, model=self.model, processing_class=self.tokenizer,
            ref_model=self.ref_model, reward_model=self.reward_model,
            train_dataset=train_dataset, value_model=self.value_model
        )
        print(f"--- Model {model_name} loaded successfully! ---")
        print("DEBUG: PPOTrainer type →", type(self.ppo_trainer))
        

def run_training_loop(args):
    print("--- Building PPOConfig ---")

    # This config uses ONLY valid arguments for your version of TRL.
    # The fix is to use a much smaller learning rate and add gradient clipping
    # to prevent the KL divergence from exploding.
    ppo_cfg = PPOConfig(
        learning_rate=args.learning_rate, # This will now be a much smaller value
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        num_ppo_epochs=4,
        
        # We must use the static kl_coef, as it's the valid parameter.
        # The explosion is controlled by the learning rate.
        kl_coef=0.05, 
        
        # Add a gradient clipping safety rail.
        max_grad_norm=1.0,

        stop_token_id=None,
        output_dir="./results_ppo",
        bf16=False,
        fp16=False,
        response_length=150,
        total_episodes=args.num_episodes,
        num_sample_generations=0
    )

    # The rest of this function is correct and unchanged.
    print("--- Creating and Tokenizing Dataset ---")
    base_prompt = (
        "You are a world-class network security expert specializing in radio "
        "frequency analysis. Your task is to generate a precise JSON "
        "configuration for a radio jammer to neutralize a target signal. "
        "The JSON object must contain three keys: 'center_frequency' (float, in Hz), "
        "'bandwidth' (float, in Hz), and 'tx_gain' (float, from 0-90). "
        "Adhere strictly to the JSON format. Do not provide any other text, "
        "explanation, or markdown. Your entire output must be only the JSON object."
    )
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

    train_dataset = Dataset.from_list(raw_prompts)
    def tokenize_function(examples):
        return tokenizer(examples["query"], truncation=True, padding=False)
    tokenized_dataset = train_dataset.map(
        tokenize_function, batched=True, remove_columns=["query"]
    )
    tokenized_dataset.set_format("torch")

    agent = LLMAgent(model_name=args.model, ppo_cfg=ppo_cfg, train_dataset=tokenized_dataset)

    print(f"\n--- Starting RL Training with {args.model} ---")
    agent.ppo_trainer.train()

    print("\n--- Training Complete ---")
    output_dir = f"jammer_agent_{args.model.split('/')[-1]}"
    agent.model.save_pretrained(output_dir)
    agent.tokenizer.save_pretrained(output_dir)
    print(f"Model saved successfully to ./{output_dir}")


# Now, also update the default learning rate in your main execution block
# to reflect this much more stable value.
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--num_episodes", type=int, default=50)
    
    # Use a much smaller, more stable learning rate suitable for PPO.
    parser.add_argument("--learning_rate", type=float, default=1.4e-6) # Changed from 1.4e-5
    
    run_training_loop(parser.parse_args())