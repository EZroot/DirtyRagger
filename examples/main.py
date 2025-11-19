import torch
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

# --- Configuration & One-Time Setup ---
model_name = "Qwen/Qwen3-4B-Instruct-2507"
DEFAULT_MAX_NEW_TOKENS = 512
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)

# History settings
ENABLE_HISTORY = False       # Set to False to disable history entirely
MAX_HISTORY_MESSAGES = 6    # Keep last N messages if history is enabled
MAX_HISTORY_TOKENS = 1024   # Optional: truncate history by tokens

# --- Load Model & Tokenizer ---
print(f"Loading Model: {model_name}...")
try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16,
        device_map="cuda"
    ).to(DEVICE)
    model.eval()
except Exception as e:
    print(f"Error loading model: {e}")
    sys.exit(1)

# Generation Configuration
generation_config = GenerationConfig.from_pretrained(
    model_name,
    max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
    temperature=0.7,
    do_sample=True,
    top_p=0.8,
    pad_token_id=tokenizer.eos_token_id
)

# Chat History
history = []

print("--- Qwen3 CLI Chat Loaded ---")
print("Enter your prompt, or type 'quit' or 'exit' to stop.")
print("-" * 30)


def build_prompt(history, new_input):
    """
    Builds the model input prompt using history (if enabled) and the new user input.
    Truncates history to prevent long prompts.
    """
    if ENABLE_HISTORY:
        # Include only last N messages
        relevant_history = history[-MAX_HISTORY_MESSAGES:]

        # Apply chat template
        text = tokenizer.apply_chat_template(
            relevant_history,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )

        # Optional: truncate by token count
        tokens = tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) > MAX_HISTORY_TOKENS:
            tokens = tokens[-MAX_HISTORY_TOKENS:]
            text = tokenizer.decode(tokens)
    else:
        # Only use current message, ignore history
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": new_input}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
    return text


# --- Interactive Loop with Token Streaming ---
while True:
    try:
        user_input = input("User: ").strip()
        if user_input.lower() in ["quit", "exit"]:
            print("Exiting chat. Goodbye!")
            break
        if not user_input:
            continue

        # Append user message to history
        if ENABLE_HISTORY:
            history.append({"role": "user", "content": user_input})

        # Build prompt
        prompt = build_prompt(history, user_input)

        # Tokenize & move to device
        model_inputs = tokenizer([prompt], return_tensors="pt").to(DEVICE)

        # Prepare streaming generation
        generated_ids = model_inputs.input_ids.clone()
        print("Qwen: ", end="", flush=True)

        with torch.no_grad():
            for _ in range(generation_config.max_new_tokens):
                outputs = model(generated_ids)
                next_token_logits = outputs.logits[:, -1, :]
                next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)

                # Decode new token and stream output
                next_token_text = tokenizer.decode(next_token_id[0], skip_special_tokens=True)
                print(next_token_text, end="", flush=True)

                # Stop if EOS token
                if next_token_id.item() == tokenizer.eos_token_id:
                    break

        print()  # Newline after response

        # Add assistant response to history
        output_ids = generated_ids[0][model_inputs.input_ids.shape[1]:]
        content = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        if ENABLE_HISTORY:
            history.append({"role": "assistant", "content": content})

    except KeyboardInterrupt:
        print("\nExiting chat. Goodbye!")
        break
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        if ENABLE_HISTORY:
            history = []
            print("History cleared. Please start a new query.")
