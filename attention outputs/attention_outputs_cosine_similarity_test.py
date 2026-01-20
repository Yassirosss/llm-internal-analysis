# ============================================================
# analysis_attn_out_all_layers_colab.py
# ============================================================

import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from transformers import AutoTokenizer, AutoModelForCausalLM

# ============================================================
# 1) Config
# ============================================================
MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
INPUT_FILE = "text.txt"
FIG_DIR = "figures_all_layers"
os.makedirs(FIG_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# 2) Load tokenizer & model
# ============================================================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    output_attentions=False,
    output_hidden_states=False,
).to(device)

model.eval()

# ============================================================
# 3) Hook: capture attention output (before residual)
# ============================================================
attn_outputs = {}  # layer_id -> (1, seq_len, hidden_dim)

def make_attn_hook(layer_id: int):
    def hook(module, inp, out):
        if isinstance(out, tuple):
            out = out[0]
        attn_outputs[layer_id] = out.detach().cpu()
    return hook

for layer_id, layer in enumerate(model.model.layers):
    layer.self_attn.register_forward_hook(make_attn_hook(layer_id))

# ============================================================
# 4) Tokenize input
# ============================================================
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    text = f.read().strip()

inputs = tokenizer(
    text,
    return_tensors="pt",
    truncation=True,
    max_length=256
).to(device)

# ============================================================
# 5) Forward pass
# ============================================================
with torch.no_grad():
    _ = model(**inputs)

num_layers = len(model.model.layers)
seq_len = inputs["input_ids"].shape[1]

print(f"[INFO] num_layers={num_layers}, seq_len={seq_len}")

# ============================================================
# 6) Cosine similarity function
# ============================================================
def cosine_sim_matrix(x: torch.Tensor) -> torch.Tensor:
    x = F.normalize(x, p=2, dim=-1)  # normalisation L2 le long de la dernière dimension
    return x @ x.T

# ============================================================
# 7) Loop over ALL layers
# ============================================================
for layer_id in range(num_layers):

    if layer_id not in attn_outputs:
        print(f"[WARN] Layer {layer_id} missing, skipping.")
        continue

    out = attn_outputs[layer_id].squeeze(0)  # (seq, hidden)
    sim = cosine_sim_matrix(out).numpy()

    plt.figure(figsize=(6, 5))
    plt.imshow(sim, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar()
    plt.title(f"Cosine similarity — Attention output (Layer {layer_id})")
    plt.xlabel("Token index")
    plt.ylabel("Token index")

    out_path = os.path.join(
        FIG_DIR,
        f"attn_out_cosine_layer_{layer_id}.png"
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"[SAVED] Layer {layer_id}: {out_path}")

print("[DONE] All layer figures generated.")
