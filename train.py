"""
Train the toy I-JEPA on CIFAR-10.

Run on Kaggle/Colab (needs internet access to download CIFAR-10, which
this sandbox does not have). On a T4 this should take well under an hour
for 30-40 epochs at this model size.

Usage:
    python train.py --epochs 40 --batch_size 256 --out_dir outputs/run1
"""
import argparse
import os
import json
import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T

from model.jepa import JEPA


@torch.no_grad()
def representation_std(model, images, device):
    """
    Collapse check #1: per-dimension std of target-encoder output across
    the batch. If this collapses toward 0, the encoder is mapping every
    input to (nearly) the same vector -- the classic JEPA failure mode
    that EMA + stop-gradient + predictor asymmetry is supposed to prevent.
    A healthy, non-collapsed encoder keeps this bounded well above 0
    (rule of thumb popularized by VICReg: compare against ~1/sqrt(D)).
    """
    out = model.target_encoder(images.to(device))
    tokens = out["tokens"]  # (B, N, D)
    pooled = tokens.mean(dim=1)  # (B, D)
    std = pooled.std(dim=0).mean().item()
    return std


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=0.05)
    ap.add_argument("--ema_momentum", type=float, default=0.996)
    ap.add_argument("--embed_dim", type=int, default=128)
    ap.add_argument("--enc_depth", type=int, default=6)
    ap.add_argument("--enc_heads", type=int, default=4)
    ap.add_argument("--pred_dim", type=int, default=64)
    ap.add_argument("--pred_depth", type=int, default=3)
    ap.add_argument("--data_dir", type=str, default="./data")
    ap.add_argument("--out_dir", type=str, default="./outputs/run1")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    transform = T.Compose([
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]),
    ])
    train_set = torchvision.datasets.CIFAR10(root=args.data_dir, train=True, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                               num_workers=2, drop_last=True, pin_memory=(device == "cuda"))

    # small held-out batch purely for tracking collapse metric during training
    probe_images = torch.stack([train_set[i][0] for i in range(256)])

    model = JEPA(
        img_size=32, patch_size=4, embed_dim=args.embed_dim,
        enc_depth=args.enc_depth, enc_heads=args.enc_heads,
        pred_dim=args.pred_dim, pred_depth=args.pred_depth,
        ema_momentum=args.ema_momentum,
    ).to(device)

    opt = torch.optim.AdamW(model.context_encoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.epochs * len(train_loader)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=total_steps, pct_start=0.1)

    history = {"loss": [], "repr_std": [], "epoch": []}

    step = 0
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for images, _ in train_loader:
            images = images.to(device)
            loss, _ = model(images)

            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            model.update_target_encoder()

            running_loss += loss.item()
            step += 1

        avg_loss = running_loss / len(train_loader)
        std = representation_std(model, probe_images, device)
        history["loss"].append(avg_loss)
        history["repr_std"].append(std)
        history["epoch"].append(epoch)

        print(f"epoch {epoch+1:3d}/{args.epochs}  loss={avg_loss:.4f}  repr_std={std:.4f}")

        if std < 1e-3:
            print("WARNING: representation std collapsed near zero. "
                  "Encoder is likely producing near-constant outputs. "
                  "Consider lowering ema_momentum or checking predictor capacity.")

    torch.save(model.state_dict(), os.path.join(args.out_dir, "jepa_final.pt"))
    with open(os.path.join(args.out_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    with open(os.path.join(args.out_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    print(f"done. saved to {args.out_dir}")


if __name__ == "__main__":
    main()
