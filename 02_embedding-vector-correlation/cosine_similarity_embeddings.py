import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
import os



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def cosine_similarity_matrix(z: torch.Tensor) -> torch.Tensor:
    """
    Compute cosine similarity matrix between all token embeddings.

    Args:
        z (Tensor): shape (T, d)

    Returns:
        Tensor: shape (T, T)
    """
    z_norm = F.normalize(z, p=2, dim=1)  # (T, d)
    return z_norm @ z_norm.T


def layerwise_cosine_similarity(hidden_states, layer_indices):
    """
    Compute cosine similarity matrices for selected layers.

    Args:
        hidden_states (tuple of Tensor):
            hidden_states[l] has shape (1, T, d) or (T, d)
        layer_indices (list or iterable of int):
            indices of layers to analyze

    Returns:
        dict: {layer_index: cosine_similarity_matrix}
    """
    similarity_matrices = {}

    for l in layer_indices:
        z = hidden_states[l]

        # Remove batch dimension if present
        if z.dim() == 3:
            z = z.squeeze(0)  # (T, d)

        similarity_matrices[l] = cosine_similarity_matrix(z)

    return similarity_matrices



def plot_similarity_matrices_per_layer(
    similarity_matrices,
    output_dir="results",
    dpi=300
):
    """
    Save one cosine similarity matrix per layer (one figure per layer).

    Args:
        similarity_matrices (dict):
            {layer_index: cosine_similarity_matrix}
        output_dir (str):
            Directory where figures are saved
        dpi (int):
            Resolution of saved figures
    """
    for layer, S in similarity_matrices.items():
        plt.figure(figsize=(5, 5))
        plt.imshow(S.cpu().numpy(), vmin=-1, vmax=1)
        plt.colorbar()
        plt.title(f"Cosine similarity – layer {layer}")
        plt.xlabel("Token index")
        plt.ylabel("Token index")

        save_path = f"{output_dir}/cosine_similarity_layer_{layer}.png"
        plt.tight_layout()
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close()


with open("text1.txt", "r", encoding="utf-8") as f:
    text = f.read().strip()



model_name = "meta-llama/Llama-3.2-1B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    output_hidden_states=True
).to(device)

inputs = tokenizer(
    text,
    return_tensors="pt",
    truncation=True,
    max_length=256
).to(device)



with torch.no_grad():
    outputs = model(**inputs)

hidden_states = outputs.hidden_states  # tuple of length L+1


layer_indices = [i for i in range(len(hidden_states))]

similarity_matrices = layerwise_cosine_similarity(
    hidden_states,
    layer_indices
)

os.makedirs("test1_results", exist_ok=True)
plot_similarity_matrices_per_layer(
    similarity_matrices,
    output_dir="test1_results"
)
