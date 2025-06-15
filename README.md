# 🔎 induction-viz: Induction Head Scorer and Visualizer

A tool for identifying, scoring, and visualizing **Induction Heads** in transformer-based language models using mechanistic interpretability techniques. This project replicates one of the foundational findings in transformer circuit analysis.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/)

## 🌟 Overview

Induction heads are specialized components within transformer models that detect and continue repeating patterns in text. For example, in a sequence like `A B C ... A B`, an induction head in the model will cause the attention mechanism at the final `B` to look back at the *first* `B`, increasing the model's probability of predicting `C` next.

This project provides a simple, extensible script to:
1.  Generate synthetic data with repeating patterns.
2.  Run a model (e.g., `gpt2-small`) on this data.
3.  Use hooks to access and cache attention patterns from every head.
4.  Calculate an "induction score" for each head.
5.  Generate a heatmap visualizing which heads are responsible for this behavior.

### Example Visualization

This heatmap shows the induction score for every attention head in `gpt2-small`. Brighter cells indicate a stronger induction capability. We can clearly see strong induction heads appearing in the middle layers (e.g., Layer 5, Head 5 and Layer 6, Head 9), which aligns with published research.

## 🧠 The "Why": Mechanistic Interpretability

Understanding circuits like induction heads is a core goal of Mechanistic Interpretability (MI). Instead of treating models as black boxes, we aim to reverse-engineer the specific algorithms they have learned. This project serves as a "Hello, World!" for MI, demonstrating a concrete, well-understood mechanism inside a real-world model.
This work is heavily inspired by the original discovery of induction heads, detailed in [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html).

## ⚙️ How It Works

The logic is straightforward:

1.  **Synthetic Data:** We generate a batch of sequences of random tokens with a repeating prefix, for example: `[rand_1, rand_2, ..., rand_k, **A, B,** rand_x, ..., rand_y, **A, B**]`.
2.  **Model and Hooks:** We load a pre-trained model using Neel Nanda's `TransformerLens` library. We add a "hook" to the model's attention layers to save the attention pattern (`z.pattern`) for every head on every forward pass.
3.  **Scoring Logic:** We run the model on our synthetic data. For each head, we calculate its **induction score** by measuring how much attention the token `B` in the *second* sequence `A B` pays to the token `B` in the *first* sequence. A high score means the head is consistently performing this pattern-matching task.
4.  **Visualization:** We aggregate the scores across the batch and plot them on a 2D heatmap, with layers on the y-axis and heads on the x-axis.

## 🚀 Getting Started

### Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/YOUR_USERNAME/induction-viz.git
    cd induction-viz
    ```
2.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### Usage

You can run the analysis and generate the visualization with a single command.

```bash
python run_analysis.py --model_name "gpt2-small"
```

The script will print the top 5 induction heads and their scores to the console and save the heatmap visualization as `output/induction_scores.png`.

#### Command-line Arguments

You can customize the run with several arguments:
*   `--model_name`: The Hugging Face model to analyze (e.g., `"gpt2-small"`, `"pythia-70m"`). Defaults to `"gpt2-small"`.
*   `--num_examples`: The number of synthetic examples to generate. Defaults to `100`.
*   `--seq_len`: The length of each synthetic sequence. Defaults to `64`.
*   `--output_path`: The path to save the output visualization. Defaults to `output/induction_scores.png`.

Example with custom arguments:
```bash
python run_analysis.py --model_name "pythia-160m" --num_examples 200
```

## 📂 Project Structure

```
induction-viz/
├── run_analysis.py         # Main script to run the scoring and visualization
├── utils/
│   ├── data_generator.py   # Logic for generating synthetic data
│   └── hook_utils.py       # Hook functions for caching activations
├── requirements.txt        # Python dependencies
├── output/                   # Directory for saved plots (created on first run)
└── README.md               # This file
```

## 🛠️ Future Improvements

This is a foundational project. Here are some ideas for extending it:
*   [ ] **Support More Models:** Add easy support for other model families like LLaMA, Mistral, or Mamba.
*   [ ] **Interactive Dashboard:** Use `Streamlit` or `Gradio` to create a web UI where you can choose a model and see the results interactively.
*   [ ] **Advanced Scoring:** Implement alternative scoring metrics, such as the impact on the logit output for the correct next token.
*   [ ] **Cross-Model Comparison:** Create a script to run the analysis across an entire family of models (e.g., all Pythia models) and plot how induction scores evolve with scale.

## 🙏 Acknowledgments

*   This project relies heavily on the fantastic **`TransformerLens`** library by **Neel Nanda**.
*   The conceptual framework is based on the work of the **Anthropic interpretability team** and their "Circuits" thread.

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
