import argparse
import json
import os
import time
from dataclasses import dataclass, asdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


# ------------------------------
# Model: Simple VGG-style CNN
# ------------------------------
class SimpleCIFAR10CNN(nn.Module):
    """
    Input: 3x32x32 (CIFAR-10)
    Conv layers: 6 conv layers (>=3 required), each 3x3, stride=1, padding=1
    Activations: ReLU
    Pooling: MaxPool 2x2 after each block (3 pools)
    FC: 4096->512->10
    """
    def __init__(self, num_classes=10, dropout=0.25):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1: 32x32 -> 16x16
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),  # conv1
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), # conv2
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2, 2),  # pool1
            nn.Dropout(dropout),

            # Block 2: 16x16 -> 8x8
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), # conv3
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),# conv4
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2, 2),  # pool2
            nn.Dropout(dropout),

            # Block 3: 8x8 -> 4x4
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1), # conv5
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1), # conv6
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2, 2),  # pool3
            nn.Dropout(dropout),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),  # fc1
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)   # fc2
        )

    def forward(self, x):
        return self.classifier(self.features(x))


@dataclass
class TrainConfig:
    seed: int = 42
    batch_size: int = 128
    epochs: int = 35
    lr: float = 3e-3                 # max_lr for OneCycle
    weight_decay: float = 1e-2
    val_split: float = 0.1
    num_workers: int = 2
    dropout: float = 0.25
    label_smoothing: float = 0.1
    amp: bool = True                 # mixed precision on GPU


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)


def accuracy(logits, y):
    preds = logits.argmax(dim=1)
    return (preds == y).float().mean().item()


@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval()
    total_loss, total_acc, n = 0.0, 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item()
        total_acc += accuracy(logits, y)
        n += 1
    return total_loss / max(n, 1), total_acc / max(n, 1)


def plot_curves(hist, out_prefix):
    epochs = np.arange(1, len(hist["train_loss"]) + 1)

    plt.figure()
    plt.plot(epochs, hist["train_loss"], label="Train Loss")
    plt.plot(epochs, hist["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curves")
    plt.legend()
    plt.savefig(out_prefix + "_loss.png", dpi=200, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(epochs, hist["train_acc"], label="Train Acc")
    plt.plot(epochs, hist["val_acc"], label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Curves")
    plt.legend()
    plt.savefig(out_prefix + "_acc.png", dpi=200, bbox_inches="tight")
    plt.close()


@torch.no_grad()
def measure_inference_time_ms_per_image(model, loader, device, warmup_batches=10, timed_batches=50):
    model.eval()

    it = iter(loader)
    # warmup
    for _ in range(warmup_batches):
        x, _ = next(it)
        x = x.to(device)
        _ = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()

    # timing
    total_images = 0
    start = time.perf_counter()
    it = iter(loader)
    for _ in range(timed_batches):
        x, _ = next(it)
        x = x.to(device)
        _ = model(x)
        total_images += x.size(0)
    if device.type == "cuda":
        torch.cuda.synchronize()
    end = time.perf_counter()

    return ((end - start) / total_images) * 1000.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", type=str, default="outputs/cnn")
    p.add_argument("--epochs", type=int, default=35)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--max_lr", type=float, default=3e-3)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    cfg = TrainConfig(seed=args.seed, epochs=args.epochs, batch_size=args.batch_size, lr=args.max_lr)

    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs("data", exist_ok=True)
    os.makedirs("report_assets/figures", exist_ok=True)

    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    # CIFAR-10 normalization
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2470, 0.2435, 0.2616)

    # Strong augmentation for train
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # No augmentation for val/test
    eval_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # IMPORTANT: create two datasets to avoid transform-sharing bug
    train_aug_full = datasets.CIFAR10(root="data", train=True, download=True, transform=train_tf)
    train_eval_full = datasets.CIFAR10(root="data", train=True, download=False, transform=eval_tf)
    test_set = datasets.CIFAR10(root="data", train=False, download=True, transform=eval_tf)

    n_total = len(train_aug_full)
    n_val = int(n_total * cfg.val_split)
    n_train = n_total - n_val

    # deterministic split
    g = torch.Generator().manual_seed(cfg.seed)
    perm = torch.randperm(n_total, generator=g).tolist()
    train_idx = perm[:n_train]
    val_idx = perm[n_train:]

    train_set = Subset(train_aug_full, train_idx)
    val_set = Subset(train_eval_full, val_idx)

    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=cfg.batch_size, shuffle=False,
                             num_workers=cfg.num_workers, pin_memory=True)

    model = SimpleCIFAR10CNN(num_classes=10, dropout=cfg.dropout).to(device)

    print("\n=== Model Architecture (for report) ===")
    print("Input: 3x32x32")
    print("Conv layers: 6 conv (3x3 kernel, stride=1, padding=1) with BatchNorm + ReLU")
    print("Pooling: MaxPool 2x2 after each block (3 pools)")
    print("FC: 4096->512->10 with ReLU + Dropout")
    print(f"Dropout: {cfg.dropout}")

    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)

    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # OneCycleLR must step per batch
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=cfg.lr,
        epochs=cfg.epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.amp and device.type == "cuda"))

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = -1.0
    best_path = os.path.join(args.outdir, "cnn_best.pt")

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running_loss, running_acc, n_batches = 0.0, 0.0, 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(cfg.amp and device.type == "cuda")):
                logits = model(x)
                loss = criterion(logits, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            scheduler.step()

            running_loss += loss.item()
            running_acc += accuracy(logits.detach(), y)
            n_batches += 1

        train_loss = running_loss / n_batches
        train_acc = running_acc / n_batches
        val_loss, val_acc = evaluate(model, val_loader, device, criterion)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch:02d}/{cfg.epochs} | "
              f"train loss {train_loss:.4f} acc {train_acc*100:.2f}% | "
              f"val loss {val_loss:.4f} acc {val_acc*100:.2f}% | "
              f"lr {current_lr:.6f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({"model_state": model.state_dict(), "config": asdict(cfg)}, best_path)

    # Load best model and evaluate on test
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])

    test_loss, test_acc = evaluate(model, test_loader, device, criterion)
    print("\n=== Final Test Results ===")
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_acc*100:.2f}%")

    ms_per_image = measure_inference_time_ms_per_image(model, test_loader, device)
    print(f"Average inference time: {ms_per_image:.4f} ms/image")

    # Save plots
    out_prefix = "report_assets/figures/task1_cnn_curves"
    plot_curves(history, out_prefix)
    print(f"Saved curves: {out_prefix}_loss.png and {out_prefix}_acc.png")

    # Save metrics json
    metrics = {
        "config": asdict(cfg),
        "best_val_acc": float(best_val_acc),
        "test_loss": float(test_loss),
        "test_acc": float(test_acc),
        "ms_per_image": float(ms_per_image),
    }
    metrics_path = os.path.join(args.outdir, "task1_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved: {metrics_path}")
    print(f"Saved: {best_path}")


if __name__ == "__main__":
    main()