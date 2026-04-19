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
    input_basename = os.path.splitext(os.path.basename(args.input))[0]
    EXP_DIR = os.path.join(args.outdir, input_basename)
    
    EIG_DIR = os.path.join(EXP_DIR, "eigvals")
    COS_DIR = os.path.join(EXP_DIR, "cosine")
    COS_PROJ_SEQ_DIR = os.path.join(EXP_DIR, "cosine_proj_seq_len")
    COS_PROJ_DIR = os.path.join(EXP_DIR, f"cosine_proj_p{args.p}")
    
    for d in [EIG_DIR, COS_DIR, COS_PROJ_SEQ_DIR, COS_PROJ_DIR]:
        os.makedirs(d, exist_ok=True)
        
    print(f"[*] Starting Attention Outputs analysis for: {args.input}")
    print(f"[*] Results will be saved in: {EXP_DIR}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ============================================================
    # 2. LOAD MODEL & REGISTER HOOKS
    # ============================================================
    print(f"[*] Loading model {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float32,
        device_map="auto"
    )
    model.eval()

    # Hook mechanism to capture attention outputs
    attn_outputs = {}

    def make_attn_hook(layer_id):
        def hook(module, inp, out):
            if isinstance(out, tuple):
                attn_outputs[layer_id] = out[0].detach().cpu()
            else:
                attn_outputs[layer_id] = out.detach().cpu()
        return hook

    # Registering hooks for all layers
    for layer_id, layer in enumerate(model.model.layers):
        layer.self_attn.register_forward_hook(make_attn_hook(layer_id))

    # ============================================================
    # 3. PROCESS INPUT TEXT
    # ============================================================
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file not found: {args.input}")

    with open(args.input, "r", encoding="utf-8") as f:
        text = f.read().strip()

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=args.max_length)

    print("[*] Running inference to capture attention outputs...")
    with torch.no_grad():
        _ = model(**inputs)

    # Prepare dictionary of numpy arrays
    hidden_states = {layer_id: t.squeeze(0).numpy() for layer_id, t in attn_outputs.items()}
    seq_len = hidden_states[0].shape[0]
    
    print(f"[*] Extracted attention outputs for {seq_len} tokens across {len(hidden_states)} layers.")

    # ============================================================
    # STEP 1: Covariance + Eigenvalues
    # ============================================================
    print("[*] Computing Covariance and Eigenvalues...")
    eigvals = {}
    eigvecs = {}

    for i, X in hidden_states.items():
        # np.cov expects variables as rows by default, rowvar=False changes this
        C = np.cov(X, rowvar=False, bias=False)

        vals, vecs = np.linalg.eigh(C)
        idx = np.argsort(vals)[::-1]

        eigvals[i] = vals[idx]
        eigvecs[i] = vecs[:, idx]

        plt.figure(figsize=(6, 4))
        plt.plot(eigvals[i][:seq_len], marker='o', linestyle='-', markersize=3)
        plt.yscale("log")
        plt.title(f"Layer {i} Eigenvalues (Attention Output)")
        plt.xlabel("Eigenvalue index")
        plt.ylabel("Eigenvalue (log scale)")
        plt.grid(True, which="both", ls="--", alpha=0.5)
        plt.savefig(os.path.join(EIG_DIR, f"layer_{i}_eigvals.png"), bbox_inches='tight')
        plt.close()

    # ============================================================
    # STEP 2: Cosine similarity (raw)
    # ============================================================
    print("[*] Computing Raw Cosine Similarities...")
    for i, X in hidden_states.items():
        Xc = X - X.mean(axis=0, keepdims=True)
        Xn = Xc / np.linalg.norm(Xc, axis=1, keepdims=True)
        cos = Xn @ Xn.T

        plt.figure(figsize=(6, 5))
        plt.imshow(cos, cmap="coolwarm", vmin=-1, vmax=1)
        plt.colorbar(label="Cosine similarity")
        plt.title(f"Layer {i} - Cosine Similarity (Attn Output)")
        plt.xlabel("Token index")
        plt.ylabel("Token index")
        plt.tight_layout()
        plt.savefig(os.path.join(COS_DIR, f"layer_{i}_cosine.png"), bbox_inches='tight')
        plt.close()

    # ============================================================
    # STEP 3: Projection (seq_len) → denoising
    # ============================================================
    print("[*] Computing Denoised Cosine Similarities (proj seq_len)...")
    for i, X in hidden_states.items():
        Xc = X - X.mean(axis=0, keepdims=True)
        
        max_dim = min(seq_len - 1, Xc.shape[1])
        V = eigvecs[i][:, :max_dim]
        W = Xc @ V

        Wn = W / np.linalg.norm(W, axis=1, keepdims=True)
        cos = Wn @ Wn.T

        plt.figure(figsize=(6, 5))
        plt.imshow(cos, cmap="coolwarm", vmin=-1, vmax=1)
        plt.colorbar(label="Cosine similarity")
        plt.title(f"Layer {i} - Cosine Similarity (Proj seq_len)")
        plt.xlabel("Token index")
        plt.ylabel("Token index")
        plt.tight_layout()
        plt.savefig(os.path.join(COS_PROJ_SEQ_DIR, f"layer_{i}_cosine_proj_seq_len.png"), bbox_inches='tight')
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

        plt.figure(figsize=(6, 5))
        plt.imshow(cos, cmap="coolwarm", vmin=-1, vmax=1)
        plt.colorbar(label="Cosine similarity")
        plt.title(f"Layer {i} - Cosine (PCA k={k})")
        plt.xlabel("Token index")
        plt.ylabel("Token index")
        plt.tight_layout()
        plt.savefig(os.path.join(COS_PROJ_DIR, f"layer_{i}_cosine_proj.png"), bbox_inches='tight')
        plt.close()

    print("[*] Analysis complete! K dimensions per layer:")
    print(k_vals)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PCA and Cosine Similarity Analysis of Attention Outputs")
    
    # Required arguments
    parser.add_argument("--input", type=str, required=True, help="Path to the input text file")
    
    # Optional arguments
    # Note: Default outdir is specific to attention outputs!
    parser.add_argument("--outdir", type=str, default="figures/pca_attention", help="Base directory to save figures")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-1B-Instruct", help="HuggingFace model name")
    parser.add_argument("--p", type=float, default=0.9, help="Variance threshold for PCA (e.g., 0.9, 0.99)")
    parser.add_argument("--max_length", type=int, default=256, help="Maximum sequence length to tokenize")
    
    args = parser.parse_args()
    main(args)