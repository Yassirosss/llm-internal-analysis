# LLM Internal Analysis

Empirical study of the internal mechanisms of **LLaMA 3.2 1B Instruct**, combining attention visualization and geometric analysis of token representations across layers.

> Project by group 7 — CentraleSupélec Data Science, January 2026  
> Supervisor: Emmanuel Vazquez

---

## Overview

This project investigates how a Transformer-based LLM organizes and propagates information across its layers when processing long sequences. Two main questions are addressed:

- How do attention patterns evolve across heads and depth in long-context settings?
- How do token embeddings become geometrically structured as they pass through the network?

The analysis is conducted on [LLaMA 3.2 1B Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) using *Alice in Wonderland* as the primary input text.

---

## Repository Structure

```
llm-internal-analysis/
│
├── 01_long-context-modeling/
│   └── long_context_analysis.py
│
├── 02_embedding-vector-correlation/
│   ├── pca_attention_outputs.py
│   ├── pca_embeddings.py
│   ├── svd_attention_outputs.py
│   └── svd_embeddings.py
│
├── 03_data/
│   ├── alice_wonderland.txt          # main input (full text)
│   ├── alice_vs_quantum.txt          # experiment: two contrasting topics
│   └── recurrent_motif.txt           # experiment: repetitive structure
│
├── 04_docs/
│   ├── report.pdf
│   └── slides.pdf
│
├── README.md
└── requirements.txt
```

---

## Chapters

### Chapter 3 — Long Context Modeling (`01_long-context-modeling/`)

A single script covering:

- **RoPE (Rotary Positional Embeddings)** — mathematical formalism, geometric properties, and implementation in LLaMA 3.2
- **Attention pattern analysis** — head diversity, depth-wise evolution, attention sink phenomenon
- **Information propagation** — quantifying how information flows across layers over long sequences

### Chapter 4 — Embedding Vectors Correlation Analysis (`02_embedding-vector-correlation/`)

Four scripts, each generating figures per layer:

| Script | Method | Output figures |
|---|---|---|
| `pca_embeddings.py` | PCA on hidden states (centered) | Eigenvalue spectrum, centered cosine similarity, projected cosine similarity (seq_len denoising), projected cosine similarity (PCA — 90% variance) |
| `pca_attention_outputs.py` | PCA on attention outputs (centered) | Same 4 figure types as above |
| `svd_embeddings.py` | SVD on hidden states (non-centered) | Singular values, cosine similarity after SVD projection (90% variance) |
| `svd_attention_outputs.py` | SVD on attention outputs (non-centered) | Singular values, cosine similarity after SVD projection (90% variance) |

**Dimension selection rule:** for both PCA and SVD, `k` is chosen as the smallest integer such that the retained components explain at least **90% of the total variance**.

---

## Getting Started

### Requirements

```bash
pip install -r requirements.txt
```

Main dependencies: `torch`, `transformers`, `numpy`, `matplotlib`, `scikit-learn`.
### Running the Scripts

#### Chapter 3

To generate the attention heatmaps and RoPE visualizations:

```bash
python 01_long-context-modeling/long_context_analysis.py
```

---

#### Chapter 4

All scripts in `02_embedding-vector-correlation/` share a standardized Command Line Interface (CLI).

**General Arguments:**

* `--input` (required): Path to the text file to analyze.
* `--p` (optional): Variance/energy retention threshold (Default: `0.9`).
* `--max_length` (optional): Maximum sequence length to tokenize (Default: `256`).
* `--model` (optional): HuggingFace model name (Default: `meta-llama/Llama-3.2-1B-Instruct`).

---

### Examples of usage

1. Run standard PCA on hidden states with default settings (p=0.9):

```bash
python 02_embedding-vector-correlation/pca_embeddings.py --input 03_data/alice_wonderland.txt
```

2. Run SVD on attention outputs with a stricter threshold (99% energy retention):

```bash
python 02_embedding-vector-correlation/svd_attention_outputs.py --input 03_data/alice_vs_quantum.txt --p 0.99
```

3. Run SVD on hidden states with high stability threshold:
   *Note: The residual stream contains a strong global directional bias. A high threshold (e.g. 0.999) is recommended to avoid overly uniform cosine similarity matrices.*

```bash
python 02_embedding-vector-correlation/svd_embeddings.py --input 03_data/alice_wonderland.txt --p 0.999
```

---

## Data

| File | Description |
|---|---|
| `alice_wonderland.txt` | Full text of *Alice's Adventures in Wonderland* — main corpus |
| `alice_vs_quantum.txt` | Short excerpt juxtaposing literary and scientific language |
| `recurrent_motif.txt` | Text with repetitive syntactic structure |

---

## Notes

- Figures are not tracked by git. Run the scripts to regenerate them locally in the `figures/` directory.
- The model is loaded from HuggingFace (`meta-llama/Llama-3.2-1B-Instruct`). A HuggingFace token with access to the model is required.
- All experiments were run on Google Colab (GPU).