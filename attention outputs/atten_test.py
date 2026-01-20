# analysis_attn_out_16b.py
import os
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from transformers import AutoTokenizer, AutoModelForCausalLM


# ============================================================
# 0) Config
# ============================================================
MODEL_NAME = "/data/home/lyw/pole_projet/llama/llama-3.2-1b-instruct"
INPUT_FILE = "text.txt"
FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# 1) Load model & tokenizer
# ============================================================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32,
    output_hidden_states=False,  # 本实验不需要 hidden states
    output_attentions=False,     # 本实验不需要 attention matrix
    device_map={"": "cuda:0"},   # 按你的环境；如需改 GPU，改这里
    local_files_only=True
)
model.eval()


# ============================================================
# 2) Hook: capture attention-block output (before residual/MLP)
# ============================================================
attn_outputs = {}  # {layer_id: (1, seq_len, hidden_dim) on CPU}

def make_attn_hook(layer_id: int):
    def hook(module, inp, out):
        # out sometimes is tensor, sometimes tuple(tensor, ...)
        if isinstance(out, tuple):
            out = out[0]
        attn_outputs[layer_id] = out.detach().cpu()
    return hook

# register hook on each layer.self_attn
for layer_id, layer in enumerate(model.model.layers):
    layer.self_attn.register_forward_hook(make_attn_hook(layer_id))


# ============================================================
# 3) Load input text -> tokenize (max 256 tokens, NO stride)
# ============================================================
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    text = f.read().strip()

inputs = tokenizer(
    text,
    return_tensors="pt",
    truncation=True,
    max_length=256
).to(model.device)


# ============================================================
# 4) Forward pass (activates hooks)
# ============================================================
with torch.no_grad():
    _ = model(**inputs)

num_layers = len(model.model.layers)
seq_len = inputs["input_ids"].shape[1]
print(f"[INFO] num_layers={num_layers}, seq_len={seq_len}")


# ============================================================
# 5) 16b: cosine-sim matrices of attention outputs (low vs high)
# ============================================================
# 你原来：low_layer_idx=1, high_layer_idx=num_layers-2
low_layer_idx = 1
high_layer_idx = num_layers - 2

if low_layer_idx not in attn_outputs or high_layer_idx not in attn_outputs:
    missing = [i for i in [low_layer_idx, high_layer_idx] if i not in attn_outputs]
    raise RuntimeError(f"Missing attn_outputs for layers: {missing}. Hook may not have fired.")

# (1, seq, hidden) -> (seq, hidden)
out_low = attn_outputs[low_layer_idx].squeeze(0)
out_high = attn_outputs[high_layer_idx].squeeze(0)

def cosine_sim_matrix(x: torch.Tensor) -> torch.Tensor:
    # x: (seq, hidden)
    x = x / (x.norm(dim=-1, keepdim=True) + 1e-9)
    return x @ x.T  # (seq, seq)

sim_low = cosine_sim_matrix(out_low).numpy()
sim_high = cosine_sim_matrix(out_high).numpy()

# Plot side-by-side heatmaps
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

im1 = axes[0].imshow(sim_low, cmap="viridis", vmin=0, vmax=1)
axes[0].set_title(f"Cosine sim of Attn-Output — Low layer {low_layer_idx}")
axes[0].set_xlabel("Token index")
axes[0].set_ylabel("Token index")

im2 = axes[1].imshow(sim_high, cmap="viridis", vmin=0, vmax=1)
axes[1].set_title(f"Cosine sim of Attn-Output — High layer {high_layer_idx}")
axes[1].set_xlabel("Token index")
axes[1].set_ylabel("Token index")

# One shared colorbar
fig.subplots_adjust(right=0.88)
cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.70])
fig.colorbar(im2, cax=cbar_ax)

plt.suptitle("16b — Attention output structure (low vs high) | max_length=256, no stride", fontsize=14)
out_path = os.path.join(FIG_DIR, f"attn_out_cosine_low{low_layer_idx}_high{high_layer_idx}_L256.png")
plt.savefig(out_path, dpi=200, bbox_inches="tight")
plt.close()

print(f"[DONE] Saved: {out_path}")
