import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
from torch.nn.functional import cosine_similarity
from sklearn.decomposition import PCA

MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
FIG_DIR = "figures"
INPUT_FILE = "text.txt"

os.makedirs(FIG_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================
# Load model & tokenizer
# ============================================================
login()
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32,
    device_map="auto"
)

model.eval()

attn_outputs = {}

def make_attn_hook(layer_id):
    def hook(module, inp, out):
        # out can be a tensor or a tuple
        if isinstance(out, tuple):
            attn_outputs[layer_id] = out[0].detach().cpu()
        else:
            attn_outputs[layer_id] = out.detach().cpu()
    return hook

for layer_id, layer in enumerate(model.model.layers):
    layer.self_attn.register_forward_hook(make_attn_hook(layer_id))

INPUT_FILE = "text.txt"

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    text = f.read().strip()

max_len = 256
inputs = tokenizer(text, return_tensors="pt",truncation=True,max_length=max_len).to(device)

with torch.no_grad():
    outputs = model(**inputs)

first_layer = list(attn_outputs.keys())[0]
seq_len = attn_outputs[first_layer].shape[1]

var = {}
for i in range(16):
    var[i] = np.cov(attn_outputs[i].squeeze(0).numpy(), rowvar=False, bias=False)

eigvals = {}
eigvecs = {}
for i in range(16):
    eigvals[i], eigvecs[i] = np.linalg.eigh(var[i])
    idx = np.argsort(eigvals[i])[::-1]
    eigvals[i] = eigvals[i][idx]
    eigvecs[i]= eigvecs[i][:, idx]

EIG_DIR = os.path.join(FIG_DIR, "eigvals")

os.makedirs(EIG_DIR, exist_ok=True)

for layer_id, eig in eigvals.items():
    plt.figure(figsize=(6,4))

    eig_to_plot = eig[:]

    plt.plot(eig_to_plot, marker='o', linestyle='-', markersize=3)
    plt.yscale('log')  # log scale for better visualization
    plt.xlabel("Eigenvalue index")
    plt.ylabel("Eigenvalue (log scale)")
    plt.title(f"Layer {layer_id} Eigenvalues (top 256)")
    plt.grid(True, which="both", ls="--", alpha=0.5)

    fname = os.path.join(EIG_DIR, f"layer_{layer_id}_eigvals.png")
    plt.savefig(fname, bbox_inches='tight')
    plt.close()


import os

FIG_DIR = "figures" # Defining FIG_DIR here to ensure it's always available

print("Figures saved in the following directories:")
for root, dirs, files in os.walk(FIG_DIR):
    for file in files:
        print(os.path.join(root, file))

COS_DIR = os.path.join(FIG_DIR, "cosine")
os.makedirs(COS_DIR, exist_ok=True)

for layer_id, X in attn_outputs.items():

    X = X.squeeze(0).numpy()
    X = X - X.mean(axis=0, keepdims=True)
    X_norm = X / np.linalg.norm(X, axis=1, keepdims=True)

    cos_sim = X_norm @ X_norm.T

    plt.figure(figsize=(6,5))
    plt.imshow(cos_sim, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(label="Cosine similarity")
    plt.title(f"Layer {layer_id} – Cosine Similarity (Attention Output)")
    plt.xlabel("Token index")
    plt.ylabel("Token index")
    plt.tight_layout()

    fname = os.path.join(COS_DIR, f"layer_{layer_id}_cosine.png")
    plt.savefig(fname, bbox_inches="tight")
    plt.close()

def select_pca_dim(eigvals_layer, p=0.9):
    total_var = np.sum(eigvals_layer)
    cum_var = np.cumsum(eigvals_layer)
    k_layer = np.searchsorted(cum_var / total_var, p) + 1
    return k_layer

k = {}
for layer_id, vals in eigvals.items():
    k[layer_id] = select_pca_dim(vals, p=0.9)

W = {}
for layer_id in range(16):  # Utiliser layer_id directement
    if layer_id not in attn_outputs:
        continue
    X = attn_outputs[layer_id].squeeze(0).numpy()  # ICI layer_id est correct
    mu = X.mean(axis=0, keepdims=True)
    X_centered = X - mu
    # Sélection dimension correcte
    max_dim = min(seq_len-1, X.shape[1])
    V_k = eigvecs[layer_id][:, :max_dim]
    W[layer_id] = X_centered @ V_k

COS_PROJ_DIR_SEQ_LEN = os.path.join(FIG_DIR, "cosine_proj_seq_len")
os.makedirs(COS_PROJ_DIR_SEQ_LEN, exist_ok=True)

for layer_id, W_layer in W.items():

    W_norm = W_layer / np.linalg.norm(W_layer, axis=1, keepdims=True)
    cos_sim = W_norm @ W_norm.T

    plt.figure(figsize=(6,5))
    plt.imshow(cos_sim, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(label="Cosine similarity")
    plt.title(f"Layer {layer_id} – Cosine Similarity (Projected for seq_len)")
    plt.xlabel("Token index")
    plt.ylabel("Token index")
    plt.tight_layout()

    fname = os.path.join(COS_PROJ_DIR_SEQ_LEN, f"layer_{layer_id}_cosine_proj_seq_len.png")
    plt.savefig(fname, bbox_inches="tight")
    plt.close()

Z = {}
for i in range(16):
    X = attn_outputs[i].squeeze(0).numpy()
    mu = X.mean(axis=0, keepdims=True)
    X_centered = X - mu
    V_k = eigvecs[i][:, :k[i]]
    Z[i] = X_centered @ V_k

COS_PROJ_DIR = os.path.join(FIG_DIR, "cosine_proj")
os.makedirs(COS_PROJ_DIR, exist_ok=True)

for layer_id, Z_layer in Z.items():

    Z_norm = Z_layer / np.linalg.norm(Z_layer, axis=1, keepdims=True)
    cos_sim = Z_norm @ Z_norm.T

    plt.figure(figsize=(6,5))
    plt.imshow(cos_sim, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(label="Cosine similarity")
    plt.title(f"Layer {layer_id} – Cosine Similarity (Projected)")
    plt.xlabel("Token index")
    plt.ylabel("Token index")
    plt.tight_layout()

    fname = os.path.join(COS_PROJ_DIR, f"layer_{layer_id}_cosine_proj.png")
    plt.savefig(fname, bbox_inches="tight")
    plt.close()