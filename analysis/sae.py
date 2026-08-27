"""
Top-k Sparse Autoencoder for probing JEPA representations.

Same recipe as standard LLM-interpretability SAEs (e.g. as used on
transformer residual streams): a single hidden layer with an expansion
factor >1, and hard top-k sparsity instead of an L1 penalty (Gao et al.,
2024 -- avoids the L1-vs-reconstruction tuning problem and gives a
directly interpretable sparsity level).

Applied here to JEPA target-encoder token representations, per layer,
to ask: do JEPA representations decompose into sparse, interpretable
features the way LLM residual streams do? Do early vs late layers differ
in how sparse/dense their decomposition is (mirroring the CV's existing
finding of a dense->sparse gradient in Qwen3 layers)?
"""
import torch
import torch.nn as nn


class TopKSAE(nn.Module):
    def __init__(self, input_dim, expansion=4, k=32):
        super().__init__()
        hidden_dim = input_dim * expansion
        self.k = k
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim, bias=False)
        self.b_pre = nn.Parameter(torch.zeros(input_dim))
        # tie decoder norm init to unit columns -- standard SAE trick,
        # keeps early training from collapsing all features onto one direction
        with torch.no_grad():
            self.decoder.weight.div_(self.decoder.weight.norm(dim=0, keepdim=True) + 1e-8)

    def encode(self, x):
        x = x - self.b_pre
        pre_act = self.encoder(x)
        acts = torch.relu(pre_act)
        topk_vals, topk_idx = torch.topk(acts, self.k, dim=-1)
        sparse_acts = torch.zeros_like(acts)
        sparse_acts.scatter_(-1, topk_idx, topk_vals)
        return sparse_acts

    def decode(self, sparse_acts):
        return self.decoder(sparse_acts) + self.b_pre

    def forward(self, x):
        sparse_acts = self.encode(x)
        recon = self.decode(sparse_acts)
        return recon, sparse_acts


def train_sae(activations, expansion=4, k=32, epochs=50, lr=1e-3, batch_size=256, device="cpu"):
    """
    activations: (N, D) tensor of pooled token representations from one
    layer of the JEPA target encoder, collected over many images.
    """
    input_dim = activations.shape[-1]
    sae = TopKSAE(input_dim, expansion=expansion, k=k).to(device)
    opt = torch.optim.Adam(sae.parameters(), lr=lr)

    activations = activations.to(device)
    n = activations.shape[0]
    losses = []
    for epoch in range(epochs):
        perm = torch.randperm(n)
        running = 0.0
        for i in range(0, n, batch_size):
            batch = activations[perm[i:i + batch_size]]
            recon, acts = sae(batch)
            loss = torch.nn.functional.mse_loss(recon, batch)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item() * batch.shape[0]
        losses.append(running / n)
    return sae, losses


@torch.no_grad()
def feature_sparsity_stats(sae, activations, device="cpu"):
    """
    Returns per-feature activation frequency (fraction of inputs on which
    that SAE feature fires) -- used to distinguish 'dead' features,
    common features, and rare/specific features, and to compare sparsity
    patterns across layers.
    """
    activations = activations.to(device)
    acts = sae.encode(activations)
    fires = (acts > 0).float()
    freq = fires.mean(dim=0)  # (hidden_dim,)
    return freq
