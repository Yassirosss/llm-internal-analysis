import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login

def main(args):
    # ============================================================
    # 1. SETUP DYNAMIC DIRECTORIES
    # ============================================================
    input_basename = os.path.splitext(os.path.basename(args.input))[0]
    EXP_DIR = os.path.join(args.outdir, input_basename)
    
    SVD_DIR = os.path.join(EXP_DIR, "svd_vals")
    COS_RAW_DIR = os.path.join(EXP_DIR, "cosine_raw")
    COS_SVD_DIR = os.path.join(EXP_DIR, f"cosine_svd_proj_p{args.p}")
    
    for d in [SVD_DIR, COS_RAW_DIR, COS_SVD_DIR]:
        os.makedirs(d, exist_ok=True)
        
    print(f"[*] Starting SVD Analysis on Attention Outputs for: {args.input}")
    print(f"[*] Variance/Energy threshold (p): {args.p}")
    print(f"[*] Results will be saved in: {EXP_DIR}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ============================================================
    # 2. LOAD MODEL & REGISTER HOOKS
    # ============================================================
    print(f"[*] Loading model {args.model}...")
    # login() # Uncomment if you need to authenticate directly in the script
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float32,
        device_map="auto"
    )
    model.eval()

    attn_outputs = {}

    def make_attn_hook(layer_id):
        def hook(module, inp, out):
            if isinstance(out, tuple):
                attn_outputs[layer_id] = out[0].detach().cpu()
            else:
                attn_outputs[layer_id] = out.detach().cpu()
        return hook

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

    first_layer = list(attn_outputs.keys())[0]
    seq_len = attn_outputs[first_layer].shape[1]
    
    print(f"[*] Extracted attention outputs for {seq_len} tokens across {len(attn_outputs)} layers.")

    # ============================================================
    # 4. SVD COMPUTATION & COSINE SIMILARITY
    # ============================================================
    print("[*] Computing SVD and Cosine Similarities...")
    svd_vals = {}      
    svd_vecs = {}      
    k_svd = {}         
    cosine_raw = {}    
    cosine_svd = {}    

    for layer_id, X in attn_outputs.items():
        # Data prep
        X = X.squeeze(0).numpy()  # (T, d)

        # -------------------------
        # Raw Cosine Similarity
        # -------------------------
        X_norm = X / np.linalg.norm(X, axis=1, keepdims=True)
        cosine_raw[layer_id] = X_norm @ X_norm.T

        # -------------------------
        # Non-centered SVD
        # -------------------------
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        V = Vt.T

        svd_vals[layer_id] = S
        svd_vecs[layer_id] = V

        # -------------------------
        # Automatic k selection (energy p%)
        # -------------------------
        energy = S**2
        cum_energy = np.cumsum(energy) / np.sum(energy)
        k = np.searchsorted(cum_energy, args.p) + 1
        k_svd[layer_id] = k

        # -------------------------
        # SVD Projection
        # -------------------------
        X_proj = X @ V[:, :k]

        # -------------------------
        # Projected Cosine Similarity
        # -------------------------
        X_proj_norm = X_proj / np.linalg.norm(X_proj, axis=1, keepdims=True)
        cosine_svd[layer_id] = X_proj_norm @ X_proj_norm.T

    print(f"[*] K dimensions retained per layer for p={args.p}:")
    print(k_svd)

    # ============================================================
    # 5. GENERATE PLOTS
    # ============================================================
    print("[*] Generating and saving plots...")

    # Singular Values
    for layer_id, S in svd_vals.items():
        plt.figure(figsize=(6,4))
        plt.plot(S, marker='o', linestyle='-', markersize=3)
        plt.yscale("log")
        plt.xlabel("Singular value index")
        plt.ylabel("Singular value (log scale)")
        plt.title(f"Layer {layer_id} – Singular values (Attention Output)")
        plt.grid(True, which="both", ls="--", alpha=0.5)
        plt.savefig(os.path.join(SVD_DIR, f"layer_{layer_id}_svd_vals.png"), bbox_inches="tight")
        plt.close()

    # Raw Cosine
    for layer_id, cos_sim in cosine_raw.items():
        plt.figure(figsize=(6,5))
        plt.imshow(cos_sim, cmap="coolwarm", vmin=-1, vmax=1)
        plt.colorbar(label="Cosine similarity")
        plt.title(f"Layer {layer_id} – Cosine similarity (raw)")
        plt.xlabel("Token index")
        plt.ylabel("Token index")
        plt.tight_layout()
        plt.savefig(os.path.join(COS_RAW_DIR, f"layer_{layer_id}_cosine_raw.png"), bbox_inches="tight")
        plt.close()

    # SVD Projected Cosine
    for layer_id, cos_sim in cosine_svd.items():
        plt.figure(figsize=(6,5))
        plt.imshow(cos_sim, cmap="coolwarm", vmin=-1, vmax=1)
        plt.colorbar(label="Cosine similarity")
        plt.title(f"Layer {layer_id} – Cosine similarity (SVD p={args.p})")
        plt.xlabel("Token index")
        plt.ylabel("Token index")
        plt.tight_layout()
        plt.savefig(os.path.join(COS_SVD_DIR, f"layer_{layer_id}_cosine_svd_proj.png"), bbox_inches="tight")
        plt.close()

    print("[*] All tasks completed successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SVD and Cosine Similarity Analysis of Attention Outputs")
    
    # Required arguments
    parser.add_argument("--input", type=str, required=True, help="Path to the input text file")
    
    # Optional arguments
    parser.add_argument("--outdir", type=str, default="figures/svd_attention", help="Base directory to save figures")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-1B-Instruct", help="HuggingFace model name")
    parser.add_argument("--p", type=float, default=0.9, help="Energy retention threshold for SVD (e.g., 0.9, 0.999)")
    parser.add_argument("--max_length", type=int, default=256, help="Maximum sequence length to tokenize")
    
    args = parser.parse_args()
    main(args)