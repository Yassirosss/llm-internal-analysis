import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from transformers import AutoTokenizer, AutoModelForCausalLM

# ============================================================
# CONFIG
# ============================================================
MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
INPUT_FILE = "text.txt"
FIG_DIR = "figures/hidden_states"

os.makedirs(FIG_DIR, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================
# LOAD MODEL
# ============================================================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32,
    output_hidden_states=True,
    device_map="auto"
)

model.eval()

# ============================================================
# INPUT
# ============================================================
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    text = f.read().strip()

inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256).to(device)

with torch.no_grad():
    outputs = model(**inputs)

hidden_states = {i: h.squeeze(0).cpu().numpy() for i, h in enumerate(outputs.hidden_states)}
seq_len = next(iter(hidden_states.values())).shape[0]

# ============================================================
# UTILS
# ============================================================
def select_pca_dim(eigvals, p=0.9):
    total = np.sum(eigvals)
    cum = np.cumsum(eigvals)
    return np.searchsorted(cum / total, p) + 1

# ============================================================
# STEP 1: Covariance + Eigenvalues
# ============================================================
eigvals = {}
eigvecs = {}

EIG_DIR = os.path.join(FIG_DIR, "eigvals")
os.makedirs(EIG_DIR, exist_ok=True)

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
COS_DIR = os.path.join(FIG_DIR, "cosine")
os.makedirs(COS_DIR, exist_ok=True)

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
COS_PROJ_SEQ_DIR = os.path.join(FIG_DIR, "cosine_proj_seq_len")
os.makedirs(COS_PROJ_SEQ_DIR, exist_ok=True)

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
# STEP 4: PCA projection (k)
# ============================================================
COS_PROJ_DIR = os.path.join(FIG_DIR, "cosine_proj")
os.makedirs(COS_PROJ_DIR, exist_ok=True)

k_vals = {}

for i, X in hidden_states.items():
    k = select_pca_dim(eigvals[i], p=0.9)
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

print("k per layer:", k_vals)