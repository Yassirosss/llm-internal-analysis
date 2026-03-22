# LLM Internal Analysis  
**Analysis of Representation Geometry and Attention (LLaMA 3.2 1B)**

This repository contains the code and data associated with the project report on the **LLaMA 3.2 1B Instruct** model, conducted at **CentraleSupélec** in January 2026. The study focuses on understanding the internal mechanisms of attention and the geometric structure of embeddings when processing long contexts.

---

## 📖 Project Summary

The project explores two main analysis axes:

1. **Exploiting Long Contexts**  
   Analysis of the efficiency of **Rotary Positional Embeddings (RoPE)** and the evolution of attention patterns across model depth.

2. **Representation Geometry**  
   Study of directional alignment of tokens and semantic domain separation in the latent space.

---

## 🏗️ Repository Structure

The project is organized to reflect the progression of the technical analysis:

### 📥 Input Data (`embd test/texts/`)

This folder contains source texts used for inference experiments:

- `text.txt` – Excerpts from *Alice in Wonderland* for baseline tests.  
- `Two_texts.txt` – Combination of literature and quantum mechanics for domain separation analysis.  
- `text_test_3.txt` – Scientific motifs injected into a narrative to test recurring pattern recognition.

### 🔍 Attention Analysis (`attention/`)

Responsible for the first analysis phase (Chapter 3 of the report).

- **Scripts** – Generation of attention heatmaps.  
- **Objectives** – Visualize head diversity, receptive field extension, and the attention sink phenomenon in deep layers.

### 📐 Vector Analysis (`SVD/`, `Covariance matrix/`, `embd test/`)

Contains tools for geometric analysis of hidden states (Chapter 4).

- `embd test/` – Studies on raw cosine similarity and the evolution of semantic alignment between tokens.  
- `Covariance matrix/` – Centered vector analysis and spectral decomposition (PCA) to isolate directions of maximal variance.  
- `SVD/` – Use of uncentered singular value decomposition to filter numerical noise while preserving absolute directional structure.

---

## 🛠️ Technical Configuration

Analyses are based on the **LLaMA 3.2 1B** hyperparameters:

- Layers: 16  
- Hidden dimension: 2048  
- Max context: 131,072 tokens  
- RoPE: Base 500,000 with scale factor 32.0

---

## 📊 Key Findings

- **Attention:** Attention heads evolve from **local focus (diagonal)** to **global aggregation** on specific anchor points (initial tokens).  
- **Geometry:** The model builds a **geometric firewall**, pushing distinct semantic domains into orthogonal subspaces.  
- **Filtering:** Using **SVD** is crucial to isolate semantic signal, as hidden states are dominated by a shared global direction related to residual flow.

> **Note:** To reproduce the report figures, run the notebooks in `SVD/attention outputs/svd.ipynb` or `attention/Attention_script.ipynb`.

---

**Project conducted by Group 7:** Y. Ouhammou, H. Jlibina, A. Hanid, Y. A. Lei, J. Faddil