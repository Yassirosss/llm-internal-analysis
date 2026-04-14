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
    device_map="auto",
    output_hidden_states=True
)

model.eval()

INPUT_FILE = "text.txt"

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    text = f.read().strip()

max_len = 256
inputs = tokenizer(text, return_tensors="pt",truncation=True,max_length=max_len).to(device)

outputs = model(**inputs, output_hidden_states=True)
hidden_states = outputs.hidden_states

svd_vals = {}      # valeurs singulières
svd_vecs = {}      # vecteurs singuliers droits
k_svd = {}         # dimension retenue
cosine_svd = {}    # cosine similarity après projection
p = 0.999


for layer_id, H in enumerate(hidden_states):

    # -------------------------
    # Données
    # -------------------------
    X = H.squeeze(0).detach().cpu().numpy()  # (T, d)

    # -------------------------
    # SVD non centrée
    # -------------------------
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    V = Vt.T

    svd_vals[layer_id] = S
    svd_vecs[layer_id] = V

    # -------------------------
    # Sélection automatique de k
    # -------------------------
    energy = S**2
    cum_energy = np.cumsum(energy) / np.sum(energy)
    k = np.searchsorted(cum_energy, p) + 1
    k_svd[layer_id] = k

    # -------------------------
    # Projection
    # -------------------------
    X_proj = X @ V[:, :k]

    # -------------------------
    # Cosine similarity projetée
    # -------------------------
    norms = np.linalg.norm(X_proj, axis=1, keepdims=True)
    X_proj_norm = X_proj / (norms + 1e-8)

    cosine_svd[layer_id] = X_proj_norm @ X_proj_norm.T


#Figures — Valeurs singulières
SVD_DIR = os.path.join(FIG_DIR, "svd_vals")
os.makedirs(SVD_DIR, exist_ok=True)

for layer_id, S in svd_vals.items():
    plt.figure(figsize=(6,4))
    plt.plot(S, marker='o', linestyle='-', markersize=3)
    plt.yscale("log")
    plt.xlabel("Singular value index")
    plt.ylabel("Singular value (log scale)")
    plt.title(f"Layer {layer_id} – Singular values")
    plt.grid(True, which="both", ls="--", alpha=0.5)

    fname = os.path.join(SVD_DIR, f"layer_{layer_id}_svd_vals.png")
    plt.savefig(fname, bbox_inches="tight")
    plt.close()


#Figures — Cosine similarity après projection SVD
COS_SVD_DIR = os.path.join(FIG_DIR, "cosine_svd_proj")
os.makedirs(COS_SVD_DIR, exist_ok=True)

for layer_id, cos_sim in cosine_svd.items():
    plt.figure(figsize=(6,5))
    plt.imshow(cos_sim, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(label="Cosine similarity")
    plt.title(f"Layer {layer_id} – Cosine similarity (SVD p={p})")
    plt.xlabel("Token index")
    plt.ylabel("Token index")
    plt.tight_layout()

    fname = os.path.join(COS_SVD_DIR, f"layer_{layer_id}_cosine_svd_proj.png")
    plt.savefig(fname, bbox_inches="tight")
    plt.close()
