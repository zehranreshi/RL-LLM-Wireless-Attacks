# inference.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import argparse
import re
import json

def parse_llm_output(text_output: str) -> dict:
    """A simple function to extract the JSON from the model's raw output."""
    try:
        json_match = re.search(r'\{[^{}]*\}', text_output, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            return json.loads(json_str)
        print("\n[Warning] No JSON object found in the output.")
        return None
    except json.JSONDecodeError:
        print(f"\n[Warning] Could not decode the JSON string: {json_str}")
        return None


def main(args):
    # --- 1. Load the a an and tokenizer ---
    print(f"--- Loading fine-tuned model from: {args.model_path} ---")
    
    # Use the same quantization config as in training for consistency
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # Load your specialized model
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval() # Set the model to evaluation mode

    # Load the tokenizer that was saved with the model
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

    # --- 2. Create a prompt ---
    # This is the same prompt structure you used in training
    base_prompt_template = (
        "You are a world-class network security expert specializing in radio "
        "frequency analysis. Your task is to generate a precise JSON "
        "configuration for a radio jammer to neutralize a target signal. "
        "The JSON object must contain three keys: 'center_frequency' (float, in Hz), "
        "'bandwidth' (float, in Hz), and 'tx_gain' (float, from 0-90). "
        "Adhere strictly to the JSON format. Do not provide any other text, "
        "explanation, or markdown. Your entire output must be only the JSON object."
        "\n\n### Current Mission\n"
        "Your current mission is to generate a config to jam a target at "
        "{target_freq_ghz:.4f} GHz.\n\n### JSON Output:\n"
    )
    
    prompt = base_prompt_template.format(target_freq_ghz=args.target_freq)
    print(f"\n--- Sending Prompt to Model ---\n{prompt}")

    # --- 3. Generate the configuration ---
    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Generate the output from the model
    # We use different settings here for more deterministic output (less randomness)
    output_tokens = model.generate(
        **inputs,
        max_new_tokens=150,
        temperature=0.1,  # Lower temperature for less random, more confident output
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id # Use eos_token_id for padding in generation
    )
    
    # Decode the generated tokens into text, skipping the prompt part
    # Note: output_tokens[0] contains the full sequence (prompt + response)
    response_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
    
    # Isolate just the newly generated part
    generated_part = response_text[len(prompt):]
    
    print("\n--- Model's Raw Output ---")
    print(generated_part)

    # --- 4. Parse and display the clean JSON config ---
    json_config = parse_llm_output(generated_part)

    if json_config:
        print("\n--- Parsed JSON Configuration ---")
        # Pretty print the JSON
        print(json.dumps(json_config, indent=2))
    else:
        print("\n--- Failed to get a valid configuration ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a jammer config using the fine-tuned LLM agent.")
    parser.add_argument(
        "--model_path",
        type=str,
        default="./jammer_agent_TinyLlama-1.1B-Chat-v1.0_ppo_trained",
        help="Path to the directory containing the saved PPO-trained model."
    )
    parser.add_argument(
        "--target_freq",
        type=float,
        default=3.65,
        help="The target frequency to jam, in GHz (e.g., 3.65)."
    )
    main(parser.parse_args())
