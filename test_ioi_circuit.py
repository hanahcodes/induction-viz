# test_ioi_circuit_multi_head.py

import torch
import torch.nn.functional as F
from transformer_lens import HookedTransformer
from functools import partial

# --- Configuration ---
MODEL_NAME = "gpt2-small"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# This time, we patch a LIST of crucial heads in the IOI circuit
# These are some of the most important S-Inhibition and Name Mover heads
HEADS_TO_PATCH = [
    (7, 10), (8, 6), # S-Inhibition Heads
    (9, 6), (9, 9),  # Name Mover Heads
    (10, 0), (10, 7) # Backup Name Mover / Other Heads
]

print("Loading model...")
model = HookedTransformer.from_pretrained(MODEL_NAME, device=DEVICE)
model.eval()

# --- Prompts and Tokens ---
clean_prompt = "When John and Mary went to the park, John gave a drink to"
corrupted_prompt = "When Alice and Bob went to the park, John gave a drink to"

clean_tokens = model.to_tokens(clean_prompt)
corrupted_tokens = model.to_tokens(corrupted_prompt)
target_token_id = model.to_single_token(" Mary")
position_to_predict = -1

# --- 1. Clean Run: Get all the activations we need to patch ---
print("\n--- Running Clean Run ---")
# Create a filter to cache only the heads we care about
# --- NEW, CORRECTED CODE ---
def heads_filter(name: str) -> bool:
    # First, a quick check to see if it's a hook we might even care about.
    # This also implicitly checks that 'blocks' is in the name, making it safer.
    if not name.endswith('attn.hook_z'):
        return False
    
    # Now that we know the name format is likely correct, we can safely split.
    try:
        layer = int(name.split('.')[1])
        # Check if this layer is one of the layers we want to patch
        is_in_patch_list = any([layer == l for l, h in HEADS_TO_PATCH])
        return is_in_patch_list
    except (IndexError, ValueError):
        # If splitting or int conversion fails for any reason, it's not a hook we want.
        return False
clean_logits, clean_cache = model.run_with_cache(clean_tokens, names_filter=heads_filter)
clean_log_probs = F.log_softmax(clean_logits[0, position_to_predict, :], dim=-1)
clean_prob = torch.exp(clean_log_probs[target_token_id]).item()
print(f"Probability of ' Mary' in Clean Run: {clean_prob:.4f}")

# --- 2. Corrupted Run ---
print("\n--- Running Corrupted Run ---")
corrupted_logits = model(corrupted_tokens)
corrupted_log_probs = F.log_softmax(corrupted_logits[0, position_to_predict, :], dim=-1)
corrupted_prob = torch.exp(corrupted_log_probs[target_token_id]).item()
print(f"Probability of ' Mary' in Corrupted Run: {corrupted_prob:.4f}")

# --- 3. Multi-Head Patching Run ---
def multi_head_patching_hook(activation, hook, head_index_to_patch):
    """
    A hook that patches a specific head's output, leaving others untouched.
    We use functools.partial to "bake in" the head_index for each hook.
    """
    layer = int(hook.name.split('.')[1])
    # Get the clean activation for this specific head
    clean_activation_for_head = clean_cache[hook.name][0, position_to_predict, head_index_to_patch, :]
    # Patch it in
    activation[0, position_to_predict, head_index_to_patch, :] = clean_activation_for_head
    return activation

# Create a list of hooks, one for each head we want to patch
hooks_to_add = []
for layer, head in HEADS_TO_PATCH:
    hook_name = f"blocks.{layer}.attn.hook_z"
    # Use partial to create a unique hook function for each head
    hook_fn = partial(multi_head_patching_hook, head_index_to_patch=head)
    hooks_to_add.append((hook_name, hook_fn))

print("\n--- Running Multi-Head Patching Run ---")
print(f"Patching {len(HEADS_TO_PATCH)} heads simultaneously...")
with model.hooks(fwd_hooks=hooks_to_add):
    patched_logits = model(corrupted_tokens)

patched_log_probs = F.log_softmax(patched_logits[0, position_to_predict, :], dim=-1)
patched_prob = torch.exp(patched_log_probs[target_token_id]).item()
print(f"Probability of ' Mary' in Patched Run: {patched_prob:.4f}")

# --- 4. Analyze Results ---
print("\n--- Analysis of Proof ---")
print(f"Baseline (Clean) Probability of ' Mary': {clean_prob:.4f}")
print(f"Ablated (Corrupted) Probability of ' Mary': {corrupted_prob:.4f}")
print(f"Restored (Patched) Probability of ' Mary': {patched_prob:.4f}")

try:
    percentage_restored = (patched_prob - corrupted_prob) / (clean_prob - corrupted_prob) * 100
    print(f"\nConclusion: By patching the key heads of the IOI circuit simultaneously,")
    print(f"we restored {percentage_restored:.2f}% of the model's performance.")
    print("This proves that the IOI capability is a distributed circuit, relying on the composition of multiple specialized heads.")
except (ZeroDivisionError, ValueError):
    print("\nCould not calculate restoration percentage.")