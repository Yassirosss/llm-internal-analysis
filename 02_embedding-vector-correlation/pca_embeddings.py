import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from transformers import AutoTokenizer, AutoModelForCausalLM

def select_pca_dim(eigvals, p=0.9):
    total = np.sum(eigvals)
    cum = np.cumsum(eigvals)
    return np.searchsorted(cum / total, p) + 1

def main(args):
    # ============================================================
    # 1. SETUP DYNAMIC DIRECTORIES
    # ============================================================
    # Extract the filename without extension (e.g., 'alice_in_wonderland')
    input_basename = os.path.splitext(os.path.basename(args.input))[0]
    
    # Create a unique output directory for this specific text
    EXP_DIR = os.path.join(args.outdir, input_basename)
    
    EIG_DIR = os.path.join(EXP_DIR, "eigvals")
    COS_DIR = os.path.join(EXP_DIR, "cosine")
    COS_PROJ_SEQ_DIR = os.path.join(EXP_DIR, "cosine_proj_seq_len")
    COS_PROJ_DIR = os.path.join(EXP_DIR, f"cosine_proj_p{args.p}") # includes p in folder name
    
    for d in [EIG_DIR, COS_DIR, COS_PROJ_SEQ_DIR, COS_PROJ_DIR]:
        os.makedirs(d, exist_ok=True)
        
    print(f"[*] Starting analysis for: {args.input}")
    print(f"[*] Results will be saved in: {EXP_DIR}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ============================================================
    # 2. LOAD MODEL & TOKENIZER
    # ============================================================
    print(f"[*] Loading model {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float32,
        output_hidden_states=True,
        device_map="auto"
    )
    model.eval()

    # ============================================================
    # 3. PROCESS INPUT TEXT
    # ============================================================
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file not found: {args.input}")

    with open(args.input, "r", encoding="utf-8") as f:
        text = f.read().strip()

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=args.max_length)

    with torch.no_grad():
        outputs = model(**inputs)

    hidden_states = {i: h.squeeze(0).cpu().numpy() for i, h in enumerate(outputs.hidden_states)}
    seq_len = next(iter(hidden_states.values())).shape[0]
    
    print(f"[*] Extracted hidden states for {seq_len} tokens across {len(hidden_states)} layers.")

    # ============================================================
    # STEP 1: Covariance + Eigenvalues
    # ============================================================
    print("[*] Computing Covariance and Eigenvalues...")
    eigvals = {}
    eigvecs = {}

    for i, X in hidden_states.items():
        Xc = X - X.mean(axis=0, keepdims=True)
        C = np.cov(Xc, rowvar=False)

        vals, vecs = np.linalg.eigh(C)
        idx = np.argsort(vals)[::-1]

        eigvals[i] = vals[idx]
        eigvecs[i] = vecs[:, idx]

        plt.figure()
        plt.plot(eigvals[i][:seq_len-1], marker='o', markersize=3)
        plt.yscale("log")
        plt.title(f"Layer {i} Eigenvalues")
        plt.grid(True)
        plt.savefig(os.path.join(EIG_DIR, f"layer_{i}.png"))
        plt.close()

    # ============================================================
    # STEP 2: Cosine similarity (raw)
    # ============================================================
    print("[*] Computing Raw Cosine Similarities...")
    for i, X in hidden_states.items():
        Xc = X - X.mean(axis=0, keepdims=True)
        Xn = Xc / np.linalg.norm(Xc, axis=1, keepdims=True)
        cos = Xn @ Xn.T

        plt.figure()
        plt.imshow(cos, cmap="coolwarm", vmin=-1, vmax=1)
        plt.colorbar()
        plt.title(f"Layer {i} Cosine")
        plt.savefig(os.path.join(COS_DIR, f"layer_{i}.png"))
        plt.close()

    # ============================================================
    # STEP 3: Projection (seq_len) → denoising
    # ============================================================
    print("[*] Computing Denoised Cosine Similarities (proj seq_len)...")
    for i, X in hidden_states.items():
        Xc = X - X.mean(axis=0, keepdims=True)
        V = eigvecs[i][:, :seq_len-1]  # full dimension ≈ denoising
        W = Xc @ V

        Wn = W / np.linalg.norm(W, axis=1, keepdims=True)
        cos = Wn @ Wn.T

        plt.figure()
        plt.imshow(cos, cmap="coolwarm", vmin=-1, vmax=1)
        plt.colorbar()
        plt.title(f"Layer {i} Cosine (proj seq_len)")
        plt.savefig(os.path.join(COS_PROJ_SEQ_DIR, f"layer_{i}.png"))
        plt.close()

    # ============================================================
    # STEP 4: PCA projection (k) based on variance 'p'
    # ============================================================
    print(f"[*] Computing PCA Projected Cosine Similarities (p={args.p})...")
    k_vals = {}

    for i, X in hidden_states.items():
        k = select_pca_dim(eigvals[i], p=args.p)
        k_vals[i] = k

        Xc = X - X.mean(axis=0, keepdims=True)
        V = eigvecs[i][:, :k]
        Z = Xc @ V

        Zn = Z / np.linalg.norm(Z, axis=1, keepdims=True)
        cos = Zn @ Zn.T

        plt.figure()
        plt.imshow(cos, cmap="coolwarm", vmin=-1, vmax=1)
        plt.colorbar()
        plt.title(f"Layer {i} Cosine (PCA k={k})")
        plt.savefig(os.path.join(COS_PROJ_DIR, f"layer_{i}.png"))
        plt.close()

    print("[*] Analysis complete! K dimensions per layer:")
    print(k_vals)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PCA and Cosine Similarity Analysis of Hidden States")
    
    # Required arguments
    parser.add_argument("--input", type=str, required=True, help="Path to the input text file (e.g., 03_data/alice.txt)")
    
    # Optional arguments with defaults
    parser.add_argument("--outdir", type=str, default="figures/pca_embeddings", help="Base directory to save figures")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-1B-Instruct", help="HuggingFace model name")
    parser.add_argument("--p", type=float, default=0.9, help="Variance threshold for PCA (e.g., 0.9, 0.99)")
    parser.add_argument("--max_length", type=int, default=256, help="Maximum sequence length to tokenize")
    
    args = parser.parse_args()
    main(args)