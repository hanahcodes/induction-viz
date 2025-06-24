# run_analysis.py

import torch
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from transformer_lens import HookedTransformer

from utils.data_generator import generate_repeated_tokens
from utils.hook_utils import cache_attention_patterns, attn_filter

def calculate_induction_scores(model, tokens, first_pos, second_pos):
    """Calculates induction scores for all heads in the model."""
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    batch_size = tokens.shape[0]

    # A tensor to store the scores for each head
    induction_scores = torch.zeros((n_layers, n_heads))
    
    # The cache will store the activations from our hooks
    cache = {}

    # THIS IS THE CORRECTED PART: We use a lambda to pass the 'cache' dictionary
    # to our hook function.
    with model.hooks(
        fwd_hooks=[(
            attn_filter,
            lambda activation, hook: cache_attention_patterns(activation, hook, cache)
        )]
    ):
        model(tokens, return_type=None) # We don't need the output, just the cache

    # Now, process the cached attention patterns
    for layer in range(n_layers):
        # Get the attention pattern for the current layer from the cache
        # Shape: [batch, n_heads, query_pos, key_pos]
        attn_pattern = cache[f"blocks.{layer}.attn.hook_pattern"]

        for head in range(n_heads):
            # For each sequence in the batch, get the attention from the
            # second occurrence to the first one.
            # `second_pos` are the query positions, `first_pos` are the key positions.
            head_scores = attn_pattern[
                torch.arange(batch_size), # for each item in the batch
                head,                     # for the current head
                second_pos,               # at the query position (the second 'B')
                first_pos                 # get the attention paid to the key position (the first 'B')
            ]
            # Average the score across the batch
            induction_scores[layer, head] = head_scores.mean().item()

    return induction_scores


def plot_heatmap(scores, model_name, save_path):
    """Plots and saves a heatmap of the induction scores."""
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        scores,
        annot=True,
        fmt=".2f",
        cmap="viridis",
        linewidths=.5,
        cbar_kws={'label': 'Induction Score'}
    )
    plt.xlabel("Head")
    plt.ylabel("Layer")
    plt.title(f"Induction Scores for Each Head in {model_name}")
    
    # Create directory if it doesn't exist
    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Heatmap saved to {save_path}")
    plt.close()

def main(args):
    print(f"Loading model: {args.model_name}")
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load the model
    model = HookedTransformer.from_pretrained(args.model_name, device=device)
    model.eval() # Set to evaluation mode

    print("Generating synthetic data...")
    tokens, first_pos, second_pos = generate_repeated_tokens(
        model,
        num_examples=args.num_examples,
        seq_len=args.seq_len
    )
    tokens = tokens.to(device)

    print("Calculating induction scores...")
    scores = calculate_induction_scores(model, tokens, first_pos, second_pos)

    print("\n--- Top 5 Induction Heads ---")
    # Flatten scores and get top 5
    flat_scores = scores.flatten()
    top_k_scores, top_k_indices = torch.topk(flat_scores, 5)
    for i in range(5):
        layer_idx = top_k_indices[i] // scores.shape[1]
        head_idx = top_k_indices[i] % scores.shape[1]
        print(f"Layer {layer_idx.item()}, Head {head_idx.item()}: Score = {top_k_scores[i].item():.4f}")
    
    print("\nPlotting heatmap...")
    plot_heatmap(scores.cpu().numpy(), args.model_name, args.output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find and visualize induction heads in transformers.")
    parser.add_argument("--model_name", type=str, default="gpt2-small", help="Name of the model to analyze.")
    parser.add_argument("--num_examples", type=int, default=100, help="Number of synthetic examples to generate.")
    parser.add_argument("--seq_len", type=int, default=64, help="Length of each synthetic sequence.")
    parser.add_argument("--output_path", type=str, default="output/induction_scores.png", help="Path to save the output heatmap.")
    
    args = parser.parse_args()
    main(args)