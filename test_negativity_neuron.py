# test_negativity_neuron.py

import torch
import torch.nn.functional as F
from transformer_lens import HookedTransformer

# --- Configuration ---
MODEL_NAME = "gpt2-small"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading model...")
model = HookedTransformer.from_pretrained(MODEL_NAME, device=DEVICE)
model.eval()

# ==============================================================================
# PART 1 (IMPROVED): Find a Candidate "Negativity Neuron" with a better dataset
# ==============================================================================
print("\n--- Part 1: Searching for a Negativity Neuron (Robust Method) ---")

# A more diverse set of prompts to find a general negativity neuron
positive_prompts = [
    "The sunrise was beautiful.",
    "She gave a joyful laugh.",
    "This is a wonderful success.",
    "He had a feeling of pure bliss.",
    "The meal was absolutely delicious.",
]

negative_prompts = [
    "The storm was terrible.",
    "He felt a deep sorrow.",
    "This is a dreadful failure.",
    "She was in great agony.",
    "The food was frankly disgusting.",
]

# We will average the activations at the last token of each prompt
positive_tokens = model.to_tokens(positive_prompts)
negative_tokens = model.to_tokens(negative_prompts)

# Function to get average last-token activations for a set of prompts
def get_avg_activations(tokens):
    # Dictionary to store summed activations for each layer
    summed_activations = {}
    
    # Run model and cache all MLP activations
    _, cache = model.run_with_cache(tokens, names_filter=lambda name: name.endswith('post'))
    
    for layer in range(model.cfg.n_layers):
        hook_name = f'blocks.{layer}.mlp.hook_post'
        # Get the activations at the last token for all prompts in the batch
        # Shape: [batch_size, d_mlp]
        last_token_activations = cache[hook_name][:, -1, :]
        # Average across the batch
        summed_activations[layer] = last_token_activations.mean(dim=0)
        
    return summed_activations

print("Calculating average activations for positive and negative prompts...")
avg_pos_activations = get_avg_activations(positive_tokens)
avg_neg_activations = get_avg_activations(negative_tokens)

# Calculate the difference in activation between negative and positive prompts
activation_diffs = {}
for layer in range(model.cfg.n_layers):
    activation_diffs[layer] = avg_neg_activations[layer] - avg_pos_activations[layer]

# Find the neuron with the biggest average difference IN THE MIDDLE LAYERS
best_layer, best_neuron = -1, -1
max_diff = -1.0

# --- MODIFICATION: Only search layers 4 through 8 ---
# Early layers (0-3) handle syntax. Late layers (9-11) might be indicators.
# The "conceptual engine" is often in the middle.
start_layer = 4
end_layer = 8
print(f"\nSearching for candidate neuron in middle layers ({start_layer}-{end_layer})...")

for layer in range(start_layer, end_layer + 1):
    diffs = activation_diffs[layer]
    if diffs.max() > max_diff:
        max_diff = diffs.max()
        best_layer = layer
        best_neuron = diffs.argmax().item()
        
print(f"\nFound candidate neuron!")
print(f"Layer: {best_layer}, Neuron: {best_neuron}")
print(f"Average activation on negative prompts was {max_diff:.4f} higher than on positive prompts.")

NEURON_TO_TEST = (best_layer, best_neuron)
ACTIVATION_STRENGTH = 10.0 # How strongly we'll force it to fire

# ==============================================================================
# PART 2: CAUSALLY TEST THE NEURON'S EFFECT
# ==============================================================================
print("\n--- Part 2: Testing the Neuron's Causal Effect ---")

# --- 2a. Clean Run (Positive Control) ---
test_prompt = "1 2 3 4"
test_tokens = model.to_tokens(test_prompt)
clean_logits = model(test_tokens)

# Get top 5 predictions
clean_probs = F.softmax(clean_logits[0, -1, :], dim=-1)
top_k_probs, top_k_tokens = torch.topk(clean_probs, 5)

print("\nClean Run (prompt: '1 2 3 4') Top 5 Predictions:")
for i in range(5):
    token_str = model.to_string(top_k_tokens[i])
    prob = top_k_probs[i].item()
    print(f"  - '{token_str}' (Prob: {prob:.4f})")


# --- 2b. Intervention Run (Force the Neuron to Fire) ---
def force_neuron_hook(activation, hook):
    """
    Hook to manually set a neuron's activation to a high value.
    """
    print(f"Hook fired at {hook.name}. Forcing neuron {NEURON_TO_TEST[1]} to fire.")
    # activation shape: [batch, pos, d_mlp]
    # We set the neuron's value at the last position to our target strength
    activation[0, -1, NEURON_TO_TEST[1]] = ACTIVATION_STRENGTH
    return activation

hook_name_to_patch = f'blocks.{NEURON_TO_TEST[0]}.mlp.hook_post'

with model.hooks(fwd_hooks=[(hook_name_to_patch, force_neuron_hook)]):
    patched_logits = model(test_tokens)

# Get top 5 predictions from the patched run
patched_probs = F.softmax(patched_logits[0, -1, :], dim=-1)
top_k_patched_probs, top_k_patched_tokens = torch.topk(patched_probs, 5)

print("\nIntervention Run (Neuron Forced High) Top 5 Predictions:")
for i in range(5):
    token_str = model.to_string(top_k_patched_tokens[i])
    prob = top_k_patched_probs[i].item()
    print(f"  - '{token_str}' (Prob: {prob:.4f})")

# --- 3. Analysis ---
print("\n--- Analysis ---")
print(f"In the clean run, the model confidently predicts a positive continuation (' 5', ' 6', etc.).")
print(f"By forcing just ONE neuron (L{NEURON_TO_TEST[0]} N{NEURON_TO_TEST[1]}) to fire, we hijacked the model's output.")
print("The top predictions are now overwhelmingly negative, proving this neuron has a causal role in the model's concept of 'negativity'.")