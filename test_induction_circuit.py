# test_induction_circuit.py

import torch
import torch.nn.functional as F
from transformer_lens import HookedTransformer

# --- Configuration ---
MODEL_NAME = "gpt2-small"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LAYER_TO_PATCH = 0
HEAD_TO_PATCH = 5

# --- Prompts ---
# The original prompt where the pattern exists
CLEAN_PROMPT = "The first man on the moon was Neil Armstrong. The second man on the moon"
# A prompt where the key pattern is broken
CORRUPTED_PROMPT = "The first dog in space was Laika. The second man on the moon"

# --- 1. Load Model and Tokenize ---
print("Loading model and tokenizer...")
model = HookedTransformer.from_pretrained(MODEL_NAME, device=DEVICE)
model.eval()

clean_tokens = model.to_tokens(CLEAN_PROMPT)
corrupted_tokens = model.to_tokens(CORRUPTED_PROMPT)
# We care about the prediction at the very last position
position_to_predict = -1 
# The token ID for " was" (with a space)
was_token_id = model.to_single_token(" was")

# --- 2. The Clean Run: Get Baseline Performance and Activation to Patch ---

print("\n--- Running Clean Run ---")
# Run the model and get the logits and the cache
clean_logits, clean_cache = model.run_with_cache(clean_tokens)

# Get the probability of the target token " was"
clean_log_probs = F.log_softmax(clean_logits[0, position_to_predict, :], dim=-1)
clean_prob = torch.exp(clean_log_probs[was_token_id]).item()
print(f"Probability of ' was' in Clean Run: {clean_prob:.4f}")

# Save the activation we want to patch in later
# The hook name for an attention head's output is 'z'
head_output_hook_name = f"blocks.{LAYER_TO_PATCH}.attn.hook_z"
activation_to_patch = clean_cache[head_output_hook_name][0, position_to_predict, HEAD_TO_PATCH, :]


# --- 3. The Corrupted Run: Show the Behavior is Broken ---

print("\n--- Running Corrupted Run ---")
corrupted_logits = model(corrupted_tokens)
corrupted_log_probs = F.log_softmax(corrupted_logits[0, position_to_predict, :], dim=-1)
corrupted_prob = torch.exp(corrupted_log_probs[was_token_id]).item()
print(f"Probability of ' was' in Corrupted Run: {corrupted_prob:.4f}")


# --- 4. The Patching Run: Test Our Hypothesis ---

def patch_head_activation(activation, hook):
    """
    This is our intervention hook. It will overwrite the activation
    of our target head with the cached value from the clean run.
    """
    print(f"Patching activation at hook: {hook.name}")
    # Overwrite the specific head's activation at the target position
    activation[0, position_to_predict, HEAD_TO_PATCH, :] = activation_to_patch
    return activation


print("\n--- Running Patching Run ---")
# Run the model on the CORRUPTED prompt, but with the hook enabled
with model.hooks(fwd_hooks=[(head_output_hook_name, patch_head_activation)]):
    patched_logits = model(corrupted_tokens)

patched_log_probs = F.log_softmax(patched_logits[0, position_to_predict, :], dim=-1)
patched_prob = torch.exp(patched_log_probs[was_token_id]).item()
print(f"Probability of ' was' in Patched Run: {patched_prob:.4f}")


# --- 5. Analyze the Results ---
print("\n--- Analysis ---")
print(f"Baseline (Clean) Probability: {clean_prob:.4f}")
print(f"Ablated (Corrupted) Probability: {corrupted_prob:.4f}")
print(f"Restored (Patched) Probability: {patched_prob:.4f}")

# Calculate how much of the drop was restored
try:
    percentage_restored = (patched_prob - corrupted_prob) / (clean_prob - corrupted_prob) * 100
    print(f"\nPercentage of performance restored by patching Head {LAYER_TO_PATCH}.{HEAD_TO_PATCH}: {percentage_restored:.2f}%")
except ZeroDivisionError:
    print("\nCannot calculate restoration percentage (clean and corrupted probs are the same).")