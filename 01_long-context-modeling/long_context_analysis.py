import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')

from transformers import AutoTokenizer, AutoModelForCausalLM

def main(args):
    print("[*] Starting Long Context Analysis (Chapter 3)")
    
    # ============================================================
    # 0. SETUP DYNAMIC DIRECTORIES
    # ============================================================
    input_basename = os.path.splitext(os.path.basename(args.input))[0]
    fig_dir = os.path.join("figures", "long_context", input_basename)
    os.makedirs(fig_dir, exist_ok=True)
    
    print(f"[*] Analyzing text: {args.input}")
    print(f"[*] Results will be saved in: {fig_dir}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ============================================================
    # 1. SETUP ET CHARGEMENT
    # ============================================================
    print(f"[*] Loading model {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float32,
        device_map="auto",
        attn_implementation="eager"
    )
    model.eval()

    # ============================================================
    # DATA PATH HANDLING
    # ============================================================
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Fichier introuvable : {args.input}")

    with open(args.input, "r", encoding="utf-8") as f:
        text = f.read().strip()

    # ============================================================
    # TOKENIZATION
    # ============================================================
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=args.max_length
    )

    # Éviter le conflit device_map="auto" en utilisant model.device
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    tokens_text = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    tokens_text = [t.replace('Ġ', '').replace('Ċ', '\\n') for t in tokens_text]

    print("[*] Running inference to extract attention weights...")
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    attentions = outputs.attentions
    seq_len = attentions[0].shape[-1]
    num_layers = len(attentions)

    print(f"[*] Extracted attention for {seq_len} tokens, {num_layers} layers.")

    # ============================================================
    # ROPE FREQUENCIES (FIX DIM ROBUSTNESS)
    # ============================================================
    print("[*] Generating RoPE frequencies plot...")
    
    dim = model.config.hidden_size // model.config.num_attention_heads
    base = 500000.0

    inv_freq = 1.0 / (base ** (np.arange(0, dim, 2).astype(float) / dim))

    plt.figure(figsize=(8, 4))
    plt.plot(inv_freq)
    plt.yscale('log')
    plt.title("RoPE Inverse Frequencies")
    plt.grid(True, alpha=0.5)
    plt.savefig(os.path.join(fig_dir, "rope_frequencies.png"), bbox_inches="tight")
    plt.close()

    # ============================================================
    # PLOT FUNCTION
    # ============================================================
    def plot_attention_heatmap(attn_matrix, title, filename, max_tokens=64):
        limit = min(seq_len, max_tokens)
        matrix = attn_matrix[:limit, :limit].cpu().numpy()
        labels = tokens_text[:limit]

        plt.figure(figsize=(8, 8))
        plt.imshow(matrix, cmap="viridis", vmin=0, vmax=1)
        plt.colorbar()

        plt.xticks(np.arange(limit), labels, rotation=90, fontsize=6)
        plt.yticks(np.arange(limit), labels, fontsize=6)

        plt.title(title)
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, filename), bbox_inches="tight")
        plt.close()

    # ============================================================
    # HEAD DIVERSITY
    # ============================================================
    layer_idx = 8
    heads_to_plot = [0, 10, 25, 31]

    for h in heads_to_plot:
        plot_attention_heatmap(
            attentions[layer_idx][0, h],
            f"Layer {layer_idx} Head {h}",
            f"attn_L{layer_idx}_H{h}.png"
        )

    # ============================================================
    # DEPTH EFFECT
    # ============================================================
    for l in [0, 8, 14]:
        plot_attention_heatmap(
            attentions[l][0, 0],
            f"Layer {l} Head 0",
            f"attn_depth_L{l}.png"
        )

    # ============================================================
    # MEAN ATTENTION
    # ============================================================
    for l in [0, 4, 10, 15]:
        mean_matrix = attentions[l][0].mean(dim=0)
        plot_attention_heatmap(
            mean_matrix,
            f"Mean Attention Layer {l}",
            f"attn_mean_L{l}.png"
        )

    # ============================================================
    # TOKEN PROPAGATION
    # ============================================================
    target_tokens = [0, 67, 300]
    target_tokens = [t for t in target_tokens if t < seq_len]

    influence_data = {t: [] for t in target_tokens}

    for l in range(num_layers):
        mean_matrix = attentions[l][0].mean(dim=0).cpu().numpy()

        for t in target_tokens:
            influence_data[t].append(mean_matrix[:, t].mean())

    plt.figure(figsize=(12, 6))
    for t in target_tokens:
        plt.plot(influence_data[t], marker='o', label=f"Token {t}")

    plt.title("Token Influence Across Layers")
    plt.xlabel("Layer")
    plt.ylabel("Attention Mass")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(fig_dir, "token_influence.png"), bbox_inches="tight")
    plt.close()

    print(f"[*] Done. Figures saved in {fig_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Long Context Analysis (Chapter 3)")
    
    # Required argument
    parser.add_argument("--input", type=str, required=True, help="Path to the text file to analyze.")
    
    # Optional arguments
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length to tokenize (Default: 512).")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-1B-Instruct", help="HuggingFace model name (Default: meta-llama/Llama-3.2-1B-Instruct).")
    
    args = parser.parse_args()
    main(args)