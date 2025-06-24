# utils/data_generator.py

import torch

def generate_repeated_tokens(
    model,
    num_examples=100,
    seq_len=64,
    prefix_len=2,
):
    """
    Generates synthetic data with repeated subsequences.

    Args:
        model: The TransformerLens model object to get vocab size from.
        num_examples (int): Number of sequences to generate.
        seq_len (int): The total length of each sequence.
        prefix_len (int): The length of the repeating subsequence (e.g., 2 for 'A B').

    Returns:
        A tuple containing:
        - tokens (torch.Tensor): The generated token sequences [batch, seq_len].
        - first_occurrence_positions (torch.Tensor): Positions of the first 'B'.
        - second_occurrence_positions (torch.Tensor): Positions of the second 'B'.
    """
    final_tokens = []
    first_occurrence_positions = []
    second_occurrence_positions = []

    for _ in range(num_examples):
        # Generate a sequence of random tokens
        tokens = torch.randint(0, model.cfg.d_vocab, (seq_len,))

        # Choose the repeating prefix (our 'A B')
        prefix = torch.randint(0, model.cfg.d_vocab, (prefix_len,))

        # Choose two random positions to insert the prefix, ensuring space
        pos1 = torch.randint(0, seq_len // 2 - prefix_len, (1,)).item()
        pos2 = torch.randint(seq_len // 2, seq_len - prefix_len, (1,)).item()

        # Insert the prefix at the chosen positions
        tokens[pos1 : pos1 + prefix_len] = prefix
        tokens[pos2 : pos2 + prefix_len] = prefix

        final_tokens.append(tokens)
        # We care about the attention from the *last token* of the second prefix
        # to the *last token* of the first prefix.
        first_occurrence_positions.append(pos1 + prefix_len - 1)
        second_occurrence_positions.append(pos2 + prefix_len - 1)

    return (
        torch.stack(final_tokens),
        torch.tensor(first_occurrence_positions),
        torch.tensor(second_occurrence_positions),
    )