"""
Mechanistic analysis of JEPA representations.

Three complementary lenses, mirroring the CV's existing interpretability
project structure (attention entropy + SAEs on Qwen3):

1. Collapse / effective-rank metrics per layer
   - std of pooled representations across a batch (near-zero = collapse)
   - effective rank of the representation covariance matrix (a single
     number summarizing "how many independent directions of variation
     does this layer actually use" -- low effective rank with high raw
     dimensionality is a softer, partial-collapse signal that plain std
     can miss)

2. Linear probes per layer
   - freeze the target encoder, train a linear classifier on top of each
     layer's pooled representation to predict CIFAR-10 labels
   - accuracy-by-depth curve tells you where semantic information is
     concentrated (JEPA/MAE literature typically finds mid-to-late layers
     peak, unlike raw pixels or very early layers)

3. Attention entropy per layer (same lens as the Qwen3 project)
   - reused directly: high entropy = broad/diffuse attention, low entropy
     = focused/near-deterministic attention
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.no_grad()
def collect_layer_representations(model, images, device, pool="mean"):
    """
    Returns a list of (N_images, D) tensors, one per transformer block,
    from the TARGET encoder (the stable, EMA-averaged one -- this is the
    encoder whose representations actually matter, since it's what the
    predictor is trained to match and what a downstream user would probe).
    """
    model.eval()
    out = model.target_encoder(images.to(device), return_all_layers=True)
    layer_outputs = out["all_layers"]  # list of (B, N, D)
    pooled = []
    for layer_tokens in layer_outputs:
        if pool == "mean":
            pooled.append(layer_tokens.mean(dim=1).cpu())
        elif pool == "cls_like":
            pooled.append(layer_tokens[:, 0].cpu())
    return pooled  # list length = depth, each (B, D)


@torch.no_grad()
def effective_rank(reps, eps=1e-8):
    """
    Effective rank via entropy of the normalized singular value spectrum
    (Roy & Vetterli, 2007): exp(entropy(normalized singular values)).
    A representation that only varies along 1 direction has eff. rank ~1
    (collapse); one that spreads variance evenly across D dims has eff.
    rank ~D (healthy, non-collapsed, non-redundant).
    """
    reps = reps - reps.mean(dim=0, keepdim=True)
    cov = reps.T @ reps / (reps.shape[0] - 1)
    eigvals = torch.linalg.eigvalsh(cov).clamp(min=0)
    p = eigvals / (eigvals.sum() + eps)
    p = p[p > eps]
    entropy = -(p * torch.log(p)).sum()
    return torch.exp(entropy).item()


@torch.no_grad()
def attention_entropy_per_layer(model, images, device):
    """
    Same metric as the Qwen3 mechanistic-interpretability project:
    entropy of each head's attention distribution, averaged over heads
    and queries, per layer. High = broad/exploratory attention, low =
    focused/copy-like attention.
    """
    model.eval()
    out = model.target_encoder(images.to(device), return_attn=True)
    all_attn = out["all_attn"]  # list of (B, num_heads, N, N)
    entropies = []
    for attn in all_attn:
        p = attn.clamp(min=1e-9)
        ent = -(p * torch.log(p)).sum(dim=-1)  # (B, heads, N)
        entropies.append(ent.mean().item())
    return entropies


class LinearProbe(nn.Module):
    def __init__(self, dim, num_classes=10):
        super().__init__()
        self.fc = nn.Linear(dim, num_classes)

    def forward(self, x):
        return self.fc(x)


def train_linear_probe(train_reps, train_labels, test_reps, test_labels,
                        num_classes=10, epochs=50, lr=1e-2, device="cpu"):
    """
    train_reps / test_reps: (N, D) frozen representations from ONE layer.
    Trains a simple linear classifier; returns final test accuracy.
    Used per-layer to trace where class-relevant information lives.
    """
    dim = train_reps.shape[-1]
    probe = LinearProbe(dim, num_classes).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=lr, weight_decay=1e-4)

    train_reps = train_reps.to(device)
    train_labels = train_labels.to(device)
    test_reps = test_reps.to(device)
    test_labels = test_labels.to(device)

    for epoch in range(epochs):
        probe.train()
        logits = probe(train_reps)
        loss = F.cross_entropy(logits, train_labels)
        opt.zero_grad()
        loss.backward()
        opt.step()

    probe.eval()
    with torch.no_grad():
        preds = probe(test_reps).argmax(dim=-1)
        acc = (preds == test_labels).float().mean().item()
    return acc
