import argparse
import json
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

import snntorch as snn
from snntorch import surrogate, spikegen


# -----------------------------
# Helpers
# -----------------------------
def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def accuracy(logits, y):
    return (logits.argmax(dim=1) == y).float().mean().item()


def bytes_to_mb(x: float) -> float:
    return x / (1024.0 * 1024.0)


# -----------------------------
# Task 1 CNN (for initialization only)
# Must match your Task 1 architecture ordering used earlier.
# -----------------------------
class SimpleCIFAR10CNN(nn.Module):
    def __init__(self, num_classes=10, dropout=0.25):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), nn.Dropout(dropout),

            nn.Conv2d(64, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), nn.Dropout(dropout),

            nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), nn.Dropout(dropout),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def init_snn_from_cnn(snn_model, cnn_ckpt_path: str, device):
    """
    Initialize SNN Conv/BN/FC weights from Task-1 CNN checkpoint.
    This DOES NOT replace Task 3 training. It only sets a better starting point.
    """
    if not os.path.exists(cnn_ckpt_path):
        print(f"⚠️ CNN checkpoint not found at {cnn_ckpt_path}. Skipping init_from_cnn.")
        return

    ckpt = torch.load(cnn_ckpt_path, map_location=device)
    cnn = SimpleCIFAR10CNN().to(device)
    cnn.load_state_dict(ckpt["model_state"])
    cnn.eval()

    # Map CNN layers (Sequential indices) -> SNN named layers
    snn_model.conv1.load_state_dict(cnn.features[0].state_dict())
    snn_model.bn1.load_state_dict(cnn.features[1].state_dict())

    snn_model.conv2.load_state_dict(cnn.features[3].state_dict())
    snn_model.bn2.load_state_dict(cnn.features[4].state_dict())

    snn_model.conv3.load_state_dict(cnn.features[8].state_dict())
    snn_model.bn3.load_state_dict(cnn.features[9].state_dict())

    snn_model.conv4.load_state_dict(cnn.features[11].state_dict())
    snn_model.bn4.load_state_dict(cnn.features[12].state_dict())

    snn_model.conv5.load_state_dict(cnn.features[16].state_dict())
    snn_model.bn5.load_state_dict(cnn.features[17].state_dict())

    snn_model.conv6.load_state_dict(cnn.features[19].state_dict())
    snn_model.bn6.load_state_dict(cnn.features[20].state_dict())

    snn_model.fc1.load_state_dict(cnn.classifier[1].state_dict())
    snn_model.fc2.load_state_dict(cnn.classifier[4].state_dict())

    print(f"✅ Initialized SNN weights from CNN checkpoint: {cnn_ckpt_path}")


# -----------------------------
# Surrogate-trained SNN (mirrors Task 1 CNN)
# 6 conv + BN, 3 pools, FC 4096->512->10
# ReLU -> LIF with surrogate gradient; trained via BPTT
# -----------------------------
class SurrogateSNN(nn.Module):
    def __init__(
        self,
        beta=0.95,
        base_threshold=1.0,
        spike_grad=None,
        layerwise_thresholds=True,
    ):
        super().__init__()
        if spike_grad is None:
            spike_grad = surrogate.fast_sigmoid(slope=25)

        # Layer-wise thresholds (helps avoid dead deep layers)
        if layerwise_thresholds:
            th1 = 1.0 * base_threshold
            th2 = 1.0 * base_threshold
            th3 = 0.9 * base_threshold
            th4 = 0.8 * base_threshold
            th5 = 0.6 * base_threshold
            th6 = 0.6 * base_threshold
            th7 = 0.5 * base_threshold
        else:
            th1 = th2 = th3 = th4 = th5 = th6 = th7 = base_threshold

        # Block 1
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(64)
        self.lif1 = snn.Leaky(beta=beta, threshold=th1, spike_grad=spike_grad)

        self.conv2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(64)
        self.lif2 = snn.Leaky(beta=beta, threshold=th2, spike_grad=spike_grad)

        self.pool1 = nn.MaxPool2d(2, 2)

        # Block 2
        self.conv3 = nn.Conv2d(64, 128, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(128)
        self.lif3 = snn.Leaky(beta=beta, threshold=th3, spike_grad=spike_grad)

        self.conv4 = nn.Conv2d(128, 128, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(128)
        self.lif4 = snn.Leaky(beta=beta, threshold=th4, spike_grad=spike_grad)

        self.pool2 = nn.MaxPool2d(2, 2)

        # Block 3
        self.conv5 = nn.Conv2d(128, 256, 3, 1, 1)
        self.bn5 = nn.BatchNorm2d(256)
        self.lif5 = snn.Leaky(beta=beta, threshold=th5, spike_grad=spike_grad)

        self.conv6 = nn.Conv2d(256, 256, 3, 1, 1)
        self.bn6 = nn.BatchNorm2d(256)
        self.lif6 = snn.Leaky(beta=beta, threshold=th6, spike_grad=spike_grad)

        self.pool3 = nn.MaxPool2d(2, 2)

        # FC
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.lif7 = snn.Leaky(beta=beta, threshold=th7, spike_grad=spike_grad)
        self.fc2 = nn.Linear(512, 10)

        # Neuron counts per sample (for firing rates)
        self.neuron_counts = {
            "l1": 64 * 32 * 32,
            "l2": 64 * 32 * 32,
            "l3": 128 * 16 * 16,
            "l4": 128 * 16 * 16,
            "l5": 256 * 8 * 8,
            "l6": 256 * 8 * 8,
            "l7": 512,
        }

    def forward(self, x, T: int):
        """
        x: [B,3,32,32] float in [0,1]
        Returns:
          logits_sum [B,10]
          spike_sums dict layer->total spikes over batch+time
        """
        B = x.size(0)
        device = x.device

        # membranes
        m1 = torch.zeros(B, 64, 32, 32, device=device)
        m2 = torch.zeros(B, 64, 32, 32, device=device)
        m3 = torch.zeros(B, 128, 16, 16, device=device)
        m4 = torch.zeros(B, 128, 16, 16, device=device)
        m5 = torch.zeros(B, 256, 8, 8, device=device)
        m6 = torch.zeros(B, 256, 8, 8, device=device)
        m7 = torch.zeros(B, 512, device=device)

        # Rate coding: [T,B,C,H,W]
        x_spk = spikegen.rate(x, num_steps=T)

        logits_sum = torch.zeros(B, 10, device=device)
        spike_sums = {"l1": 0.0, "l2": 0.0, "l3": 0.0, "l4": 0.0, "l5": 0.0, "l6": 0.0, "l7": 0.0}

        for t in range(T):
            xt = x_spk[t]

            cur = self.bn1(self.conv1(xt))
            spk1, m1 = self.lif1(cur, m1)
            spike_sums["l1"] += spk1.sum()

            cur = self.bn2(self.conv2(spk1))
            spk2, m2 = self.lif2(cur, m2)
            spike_sums["l2"] += spk2.sum()

            spk = self.pool1(spk2)

            cur = self.bn3(self.conv3(spk))
            spk3, m3 = self.lif3(cur, m3)
            spike_sums["l3"] += spk3.sum()

            cur = self.bn4(self.conv4(spk3))
            spk4, m4 = self.lif4(cur, m4)
            spike_sums["l4"] += spk4.sum()

            spk = self.pool2(spk4)

            cur = self.bn5(self.conv5(spk))
            spk5, m5 = self.lif5(cur, m5)
            spike_sums["l5"] += spk5.sum()

            cur = self.bn6(self.conv6(spk5))
            spk6, m6 = self.lif6(cur, m6)
            spike_sums["l6"] += spk6.sum()

            spk = self.pool3(spk6)

            spk = spk.flatten(1)
            cur = self.fc1(spk)
            spk7, m7 = self.lif7(cur, m7)
            spike_sums["l7"] += spk7.sum()

            logits_sum += self.fc2(spk7)

        return logits_sum, spike_sums


@torch.no_grad()
def evaluate(model, loader, device, T, criterion):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    batches = 0

    total_spikes = 0.0
    layer_spikes = {k: 0.0 for k in ["l1","l2","l3","l4","l5","l6","l7"]}
    total_images = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits, spk = model(x, T=T)
        loss = criterion(logits, y)

        total_loss += loss.item()
        total_acc += accuracy(logits, y)
        batches += 1

        B = x.size(0)
        total_images += B

        for k in layer_spikes:
            layer_spikes[k] += float(spk[k].item())
            total_spikes += float(spk[k].item())

    avg_loss = total_loss / max(batches, 1)
    avg_acc = total_acc / max(batches, 1)

    spikes_per_image = total_spikes / max(total_images, 1)

    layer_firing = {}
    for k in layer_spikes:
        neurons = model.neuron_counts[k]
        layer_firing[k] = layer_spikes[k] / (max(total_images, 1) * neurons * T)

    total_neurons = sum(model.neuron_counts.values())
    global_firing = total_spikes / (max(total_images, 1) * total_neurons * T)

    return avg_loss, avg_acc, spikes_per_image, layer_firing, global_firing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--T", type=int, default=10)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--beta", type=float, default=0.95)
    ap.add_argument("--threshold", type=float, default=1.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--outdir", type=str, default="outputs/surrogate_snn")
    ap.add_argument("--init_from_cnn", action="store_true", help="Initialize weights from Task-1 CNN checkpoint")
    ap.add_argument("--cnn_ckpt", type=str, default="outputs/cnn/cnn_best.pt")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    # Data (augmentation like Task 1)
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    test_tf = transforms.Compose([transforms.ToTensor()])

    train_full = datasets.CIFAR10(root="data", train=True, download=True, transform=train_tf)
    val_size = int(0.1 * len(train_full))
    train_size = len(train_full) - val_size
    train_set, val_set = random_split(train_full, [train_size, val_size])

    # Validation should not use augmentation
    val_set.dataset.transform = test_tf

    test_set = datasets.CIFAR10(root="data", train=False, download=True, transform=test_tf)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    spike_grad = surrogate.fast_sigmoid(slope=25)

    model = SurrogateSNN(
        beta=args.beta,
        base_threshold=args.threshold,
        spike_grad=spike_grad,
        layerwise_thresholds=True,
    ).to(device)

    if args.init_from_cnn:
        init_snn_from_cnn(model, args.cnn_ckpt, device)

    params = count_parameters(model)
    print(f"Model parameters: {params:,}")
    print("Surrogate grad: fast_sigmoid(slope=25)")
    print(f"Loss: CrossEntropy, Optimizer: Adam(lr={args.lr})")
    print(f"T={args.T}, epochs={args.epochs}, batch_size={args.batch_size}, beta={args.beta}, threshold={args.threshold}")
    print(f"init_from_cnn: {args.init_from_cnn} (ckpt: {args.cnn_ckpt})")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_val_acc = -1.0
    best_path = os.path.join(args.outdir, "surrogate_snn_best.pt")

    # Rough BPTT activation memory estimate (per sample across time)
    mem_per_sample = sum(model.neuron_counts.values()) * 4.0
    approx_mem_bptt_mb = bytes_to_mb(mem_per_sample * args.T)
    print(f"Approx activation (membrane) memory per sample across T: ~{approx_mem_bptt_mb:.2f} MB (rough)")

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_acc = 0.0
        batches = 0

        start = time.perf_counter()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(x, T=args.T)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_acc += accuracy(logits, y)
            batches += 1

        train_loss = total_loss / max(batches, 1)
        train_acc = total_acc / max(batches, 1)

        val_loss, val_acc, _, _, _ = evaluate(model, val_loader, device, args.T, criterion)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        dur = time.perf_counter() - start
        print(f"Epoch {epoch:02d}: train loss {train_loss:.4f} acc {train_acc*100:.2f}% | "
              f"val loss {val_loss:.4f} acc {val_acc*100:.2f}% | time {dur:.1f}s")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({"model_state": model.state_dict(), "args": vars(args)}, best_path)

    # Evaluate best model on test
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])

    test_loss, test_acc, spikes_per_image, layer_firing, global_firing = evaluate(
        model, test_loader, device, args.T, criterion
    )

    print("\n=== Task 3 Results (Surrogate-trained SNN) ===")
    print(f"T (time steps): {args.T}")
    print(f"Test accuracy: {test_acc*100:.2f}%")
    print(f"Spikes per inference: {spikes_per_image:.2f} spikes/image")
    print(f"Global firing rate: {global_firing*100:.4f}% spikes/neuron/timestep")
    print("Layer-wise firing rates (% spikes/neuron/timestep):")
    for k in ["l1","l2","l3","l4","l5","l6","l7"]:
        print(f"  {k}: {layer_firing[k]*100:.4f}%")

    sparsity = 1.0 - global_firing
    print(f"Spike sparsity (approx): {sparsity*100:.2f}%")

    metrics = {
        "T": args.T,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "beta": args.beta,
        "threshold": args.threshold,
        "init_from_cnn": bool(args.init_from_cnn),
        "cnn_ckpt": args.cnn_ckpt,
        "surrogate": "fast_sigmoid(slope=25)",
        "loss": "CrossEntropy",
        "optimizer": "Adam",
        "model_params": int(params),
        "best_val_acc": float(best_val_acc),
        "test_loss": float(test_loss),
        "test_acc": float(test_acc),
        "spikes_per_image": float(spikes_per_image),
        "global_firing_rate": float(global_firing),
        "layer_firing_rate": {k: float(v) for k, v in layer_firing.items()},
        "approx_membrane_mem_mb_per_sample": float(approx_mem_bptt_mb),
        "spike_sparsity": float(sparsity),
        "history": history,
    }

    out_json = os.path.join(args.outdir, "task3_metrics.json")
    with open(out_json, "w") as f:
        json.dump(metrics, f, indent=2)

    print("Saved best model:", best_path)
    print("Saved metrics:", out_json)


if __name__ == "__main__":
    main()