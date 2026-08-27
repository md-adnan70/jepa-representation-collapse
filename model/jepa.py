"""
Minimal I-JEPA implementation.

Core idea (Assran et al., 2023): instead of reconstructing masked pixels
(MAE) or contrastive instance discrimination (SimCLR), JEPA predicts the
*representation* of masked target patches from the representation of
visible context patches, using a separate predictor network. The target
encoder is an EMA (exponential moving average) copy of the context
encoder, and gradients never flow into it (stop-gradient) -- this, plus
the asymmetric predictor, is what prevents representation collapse
(everything mapping to a constant vector), which is the central failure
mode this whole architecture is designed around.

Components:
  - PatchEmbed: image -> patch tokens
  - ViTEncoder: standard pre-norm transformer encoder (used for both
    context and target encoders; target is an EMA copy, never trained
    by gradient descent)
  - Predictor: small transformer that takes context tokens + mask tokens
    (with positional embeddings for the masked positions) and predicts
    target-encoder representations at those positions
  - JEPA: wraps the above, implements masking + EMA update + loss
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_2d_sincos_pos_embed(embed_dim, grid_size):
    """Fixed sin-cos positional embeddings, standard ViT/MAE recipe."""
    grid_h = torch.arange(grid_size, dtype=torch.float32)
    grid_w = torch.arange(grid_size, dtype=torch.float32)
    grid = torch.meshgrid(grid_w, grid_h, indexing="ij")
    grid = torch.stack(grid, dim=0).reshape(2, 1, grid_size, grid_size)

    assert embed_dim % 4 == 0
    dim_half = embed_dim // 2

    def sincos_1d(pos, dim):
        omega = torch.arange(dim // 2, dtype=torch.float32)
        omega = 1.0 / (10000 ** (omega / (dim // 2)))
        pos = pos.reshape(-1)
        out = pos[:, None] * omega[None, :]
        return torch.cat([torch.sin(out), torch.cos(out)], dim=1)

    emb_h = sincos_1d(grid[0], dim_half)
    emb_w = sincos_1d(grid[1], dim_half)
    return torch.cat([emb_h, emb_w], dim=1)  # (grid_size**2, embed_dim)


class PatchEmbed(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_chans=3, embed_dim=128):
        super().__init__()
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)               # (B, C, H', W')
        x = x.flatten(2).transpose(1, 2)  # (B, N, C)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim)
        )

    def forward(self, x, return_attn=False):
        h = self.norm1(x)
        attn_out, attn_weights = self.attn(h, h, h, need_weights=return_attn, average_attn_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        if return_attn:
            return x, attn_weights
        return x


class ViTEncoder(nn.Module):
    """Used for both context encoder (trained) and target encoder (EMA copy)."""

    def __init__(self, img_size=32, patch_size=4, in_chans=3, embed_dim=128,
                 depth=6, num_heads=4, mlp_ratio=4.0):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        grid_size = self.patch_embed.grid_size

        pos_embed = build_2d_sincos_pos_embed(embed_dim, grid_size)
        self.register_buffer("pos_embed", pos_embed.unsqueeze(0))  # (1, N, D)

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.embed_dim = embed_dim
        self.num_patches = num_patches

    def forward(self, x, patch_indices=None, return_all_layers=False, return_attn=False):
        """
        x: (B, C, H, W) images
        patch_indices: optional (B, K) indices of patches to keep (context encoder
            only sees these -- this is what makes it a *context* encoder)
        return_all_layers: if True, also return list of per-block outputs (for
            interpretability probing across depth)
        """
        tokens = self.patch_embed(x)  # (B, N, D)
        tokens = tokens + self.pos_embed

        if patch_indices is not None:
            B, K = patch_indices.shape
            idx = patch_indices.unsqueeze(-1).expand(-1, -1, tokens.shape[-1])
            tokens = torch.gather(tokens, 1, idx)  # (B, K, D)

        all_layers = []
        all_attn = []
        for blk in self.blocks:
            if return_attn:
                tokens, attn = blk(tokens, return_attn=True)
                all_attn.append(attn)
            else:
                tokens = blk(tokens)
            if return_all_layers:
                all_layers.append(tokens)

        tokens = self.norm(tokens)

        out = {"tokens": tokens}
        if return_all_layers:
            out["all_layers"] = all_layers
        if return_attn:
            out["all_attn"] = all_attn
        return out


class Predictor(nn.Module):
    """Narrow transformer: context tokens + mask tokens -> predicted target reps."""

    def __init__(self, embed_dim=128, predictor_dim=64, depth=3, num_heads=4, grid_size=8):
        super().__init__()
        self.embed_in = nn.Linear(embed_dim, predictor_dim)
        self.embed_out = nn.Linear(predictor_dim, embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_dim))
        nn.init.normal_(self.mask_token, std=0.02)

        pos_embed = build_2d_sincos_pos_embed(predictor_dim, grid_size)
        self.register_buffer("pos_embed", pos_embed.unsqueeze(0))

        self.blocks = nn.ModuleList([
            TransformerBlock(predictor_dim, num_heads, mlp_ratio=4.0) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(predictor_dim)

    def forward(self, context_tokens, context_indices, target_indices):
        """
        context_tokens: (B, K_ctx, D) output of context encoder
        context_indices: (B, K_ctx) which patch positions these are
        target_indices: (B, K_tgt) which patch positions to predict
        """
        B = context_tokens.shape[0]
        D = self.mask_token.shape[-1]

        ctx = self.embed_in(context_tokens)
        ctx_pos = torch.gather(
            self.pos_embed.expand(B, -1, -1), 1,
            context_indices.unsqueeze(-1).expand(-1, -1, self.pos_embed.shape[-1])
        )
        ctx = ctx + ctx_pos

        K_tgt = target_indices.shape[1]
        mask_tokens = self.mask_token.expand(B, K_tgt, -1)
        tgt_pos = torch.gather(
            self.pos_embed.expand(B, -1, -1), 1,
            target_indices.unsqueeze(-1).expand(-1, -1, self.pos_embed.shape[-1])
        )
        mask_tokens = mask_tokens + tgt_pos

        x = torch.cat([ctx, mask_tokens], dim=1)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        pred = x[:, ctx.shape[1]:]  # only the (formerly) mask-token positions
        return self.embed_out(pred)  # (B, K_tgt, embed_dim)


def block_mask(num_patches, grid_size, batch_size, device,
                num_target_blocks=4, target_scale=(0.15, 0.2),
                context_scale=(0.85, 1.0)):
    """
    I-JEPA style block masking: sample a handful of rectangular target
    blocks (what the predictor must guess) and a large context region
    (what the context encoder is allowed to see), with target regions
    removed from the context so the task can't be solved by copying.
    Returns index tensors, NOT boolean masks, since we gather() patches.
    """
    def sample_block_indices(scale_range):
        area = grid_size * grid_size
        target_area = area * (scale_range[0] + torch.rand(1).item() * (scale_range[1] - scale_range[0]))
        aspect = 0.75 + torch.rand(1).item() * 0.5
        h = max(1, min(grid_size, round(math.sqrt(target_area * aspect))))
        w = max(1, min(grid_size, round(math.sqrt(target_area / aspect))))
        top = torch.randint(0, grid_size - h + 1, (1,)).item()
        left = torch.randint(0, grid_size - w + 1, (1,)).item()
        idx = []
        for i in range(top, top + h):
            for j in range(left, left + w):
                idx.append(i * grid_size + j)
        return set(idx)

    all_context_idx, all_target_idx = [], []
    for _ in range(batch_size):
        target_set = set()
        for _ in range(num_target_blocks):
            target_set |= sample_block_indices(target_scale)
        context_set = sample_block_indices(context_scale)
        context_set = context_set - target_set  # no leakage

        if len(context_set) == 0:
            context_set = set(range(num_patches)) - target_set
        if len(target_set) == 0:
            target_set = {torch.randint(0, num_patches, (1,)).item()}

        all_context_idx.append(sorted(context_set))
        all_target_idx.append(sorted(target_set))

    min_ctx = min(len(x) for x in all_context_idx)
    min_tgt = min(len(x) for x in all_target_idx)
    all_context_idx = [x[:min_ctx] for x in all_context_idx]
    all_target_idx = [x[:min_tgt] for x in all_target_idx]

    context_indices = torch.tensor(all_context_idx, device=device, dtype=torch.long)
    target_indices = torch.tensor(all_target_idx, device=device, dtype=torch.long)
    return context_indices, target_indices


class JEPA(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_chans=3, embed_dim=128,
                 enc_depth=6, enc_heads=4, pred_dim=64, pred_depth=3, pred_heads=4,
                 ema_momentum=0.996):
        super().__init__()
        grid_size = img_size // patch_size
        self.context_encoder = ViTEncoder(img_size, patch_size, in_chans, embed_dim, enc_depth, enc_heads)
        self.target_encoder = ViTEncoder(img_size, patch_size, in_chans, embed_dim, enc_depth, enc_heads)
        self.predictor = Predictor(embed_dim, pred_dim, pred_depth, pred_heads, grid_size)

        # target encoder starts identical to context encoder, then is EMA-only
        self.target_encoder.load_state_dict(self.context_encoder.state_dict())
        for p in self.target_encoder.parameters():
            p.requires_grad = False

        self.ema_momentum = ema_momentum
        self.num_patches = self.context_encoder.num_patches

    @torch.no_grad()
    def update_target_encoder(self):
        m = self.ema_momentum
        for p_ctx, p_tgt in zip(self.context_encoder.parameters(), self.target_encoder.parameters()):
            p_tgt.data.mul_(m).add_(p_ctx.data, alpha=1 - m)

    def forward(self, images):
        B = images.shape[0]
        device = images.device
        grid_size = self.context_encoder.patch_embed.grid_size

        context_indices, target_indices = block_mask(
            self.num_patches, grid_size, B, device
        )

        ctx_out = self.context_encoder(images, patch_indices=context_indices)
        pred = self.predictor(ctx_out["tokens"], context_indices, target_indices)

        with torch.no_grad():
            tgt_out = self.target_encoder(images)  # full image, no masking
            target_full = tgt_out["tokens"]
            idx = target_indices.unsqueeze(-1).expand(-1, -1, target_full.shape[-1])
            target = torch.gather(target_full, 1, idx)
            target = F.layer_norm(target, (target.shape[-1],))  # stabilizes scale, standard trick

        loss = F.smooth_l1_loss(pred, target)
        return loss, {"pred": pred, "target": target}
