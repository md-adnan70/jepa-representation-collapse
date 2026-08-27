"""
Run the full interpretability analysis on a trained JEPA checkpoint.

Produces:
  - outputs/<run>/effective_rank_by_layer.png
  - outputs/<run>/attention_entropy_by_layer.png
  - outputs/<run>/linear_probe_acc_by_layer.png
  - outputs/<run>/sae_feature_frequency_hist.png
  - outputs/<run>/results.json  (all numbers, for writing up findings)

Usage:
    python analyze.py --run_dir outputs/run1 --n_probe_images 4000
"""
import argparse
import json
import os

import torch
import torchvision
import torchvision.transforms as T
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model.jepa import JEPA
from analysis.probe import (
    collect_layer_representations, effective_rank,
    attention_entropy_per_layer, train_linear_probe,
)
from analysis.sae import train_sae, feature_sparsity_stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, default="./outputs/run1")
    ap.add_argument("--data_dir", type=str, default="./data")
    ap.add_argument("--n_probe_images", type=int, default=4000)
    ap.add_argument("--sae_layer", type=int, default=-1, help="which layer to train the SAE on (-1 = last)")
    ap.add_argument("--sae_expansion", type=int, default=4)
    ap.add_argument("--sae_k", type=int, default=32)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    with open(os.path.join(args.run_dir, "config.json")) as f:
        cfg = json.load(f)

    model = JEPA(
        img_size=32, patch_size=4, embed_dim=cfg["embed_dim"],
        enc_depth=cfg["enc_depth"], enc_heads=cfg["enc_heads"],
        pred_dim=cfg["pred_dim"], pred_depth=cfg["pred_depth"],
    ).to(device)
    state = torch.load(os.path.join(args.run_dir, "jepa_final.pt"), map_location=device)
    model.load_state_dict(state)
    model.eval()

    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]),
    ])
    train_set = torchvision.datasets.CIFAR10(root=args.data_dir, train=True, download=True, transform=transform)
    test_set = torchvision.datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform=transform)

    n = args.n_probe_images
    train_imgs = torch.stack([train_set[i][0] for i in range(n)])
    train_labels = torch.tensor([train_set[i][1] for i in range(n)])
    test_imgs = torch.stack([test_set[i][0] for i in range(min(n, len(test_set)))])
    test_labels = torch.tensor([test_set[i][1] for i in range(min(n, len(test_set)))])

    print("collecting per-layer representations...")
    train_layer_reps = collect_layer_representations(model, train_imgs, device)
    test_layer_reps = collect_layer_representations(model, test_imgs, device)
    depth = len(train_layer_reps)

    results = {"depth": depth}

    # 1. effective rank + std per layer (collapse check)
    print("computing effective rank / std per layer...")
    eff_ranks, stds = [], []
    for reps in train_layer_reps:
        eff_ranks.append(effective_rank(reps))
        stds.append(reps.std(dim=0).mean().item())
    results["effective_rank_by_layer"] = eff_ranks
    results["repr_std_by_layer"] = stds

    plt.figure()
    plt.plot(range(1, depth + 1), eff_ranks, marker="o")
    plt.xlabel("layer"); plt.ylabel("effective rank")
    plt.title("Effective rank of representations by layer")
    plt.savefig(os.path.join(args.run_dir, "effective_rank_by_layer.png"), bbox_inches="tight")
    plt.close()

    # 2. attention entropy per layer
    print("computing attention entropy per layer...")
    entropies = attention_entropy_per_layer(model, train_imgs[:512], device)
    results["attention_entropy_by_layer"] = entropies

    plt.figure()
    plt.plot(range(1, depth + 1), entropies, marker="o", color="darkorange")
    plt.xlabel("layer"); plt.ylabel("mean attention entropy (nats)")
    plt.title("Attention entropy by layer")
    plt.savefig(os.path.join(args.run_dir, "attention_entropy_by_layer.png"), bbox_inches="tight")
    plt.close()

    # 3. linear probe accuracy per layer
    print("training linear probes per layer (this is the slow part)...")
    probe_accs = []
    for i in range(depth):
        acc = train_linear_probe(
            train_layer_reps[i], train_labels, test_layer_reps[i], test_labels,
            epochs=100, lr=1e-2, device=device,
        )
        probe_accs.append(acc)
        print(f"  layer {i+1}/{depth}: probe acc = {acc:.4f}")
    results["linear_probe_acc_by_layer"] = probe_accs

    plt.figure()
    plt.plot(range(1, depth + 1), probe_accs, marker="o", color="green")
    plt.xlabel("layer"); plt.ylabel("linear probe test accuracy")
    plt.title("CIFAR-10 linear probe accuracy by layer (frozen target encoder)")
    plt.savefig(os.path.join(args.run_dir, "linear_probe_acc_by_layer.png"), bbox_inches="tight")
    plt.close()

    # 4. SAE on chosen layer
    print(f"training SAE on layer {args.sae_layer}...")
    sae_layer_reps = train_layer_reps[args.sae_layer]
    sae, sae_losses = train_sae(
        sae_layer_reps, expansion=args.sae_expansion, k=args.sae_k,
        epochs=200, device=device,
    )
    freq = feature_sparsity_stats(sae, sae_layer_reps, device=device)
    results["sae_final_recon_loss"] = sae_losses[-1]
    results["sae_dead_feature_frac"] = (freq == 0).float().mean().item()
    results["sae_mean_firing_freq"] = freq.mean().item()

    plt.figure()
    plt.hist(freq.cpu().numpy(), bins=50)
    plt.xlabel("feature firing frequency"); plt.ylabel("count")
    plt.title(f"SAE feature firing frequency (layer {args.sae_layer}, k={args.sae_k}, exp={args.sae_expansion}x)")
    plt.savefig(os.path.join(args.run_dir, "sae_feature_frequency_hist.png"), bbox_inches="tight")
    plt.close()

    with open(os.path.join(args.run_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("\n=== SUMMARY ===")
    print(f"effective rank: layer1={eff_ranks[0]:.1f} -> layer{depth}={eff_ranks[-1]:.1f}")
    print(f"attention entropy: layer1={entropies[0]:.3f} -> layer{depth}={entropies[-1]:.3f} nats")
    print(f"linear probe acc: layer1={probe_accs[0]:.3f} -> layer{depth}={probe_accs[-1]:.3f} (best={max(probe_accs):.3f} at layer {probe_accs.index(max(probe_accs))+1})")
    print(f"SAE: {results['sae_dead_feature_frac']*100:.1f}% dead features, recon loss={results['sae_final_recon_loss']:.4f}")
    print(f"\nall results saved to {args.run_dir}")


if __name__ == "__main__":
    main()
