import argparse
import json
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import snntorch as snn
from snntorch import spikegen


# ------------------------------
# SAME CNN ARCHITECTURE AS TASK 1 (for loading checkpoint)
# ------------------------------
class SimpleCIFAR10CNN(nn.Module):
    def __init__(self, num_classes=10, dropout=0.25):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), nn.Dropout(dropout),

            # Block 2
            nn.Conv2d(64, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), nn.Dropout(dropout),

            # Block 3
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


# ------------------------------
# BN FUSION
# ------------------------------
def fuse_conv_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d):
    """
    Fuse BatchNorm into Conv for inference:
      y = bn(conv(x))  ->  y = conv_fused(x)
    """
    W = conv.weight.data
    if conv.bias is None:
        b = torch.zeros(W.size(0), device=W.device)
    else:
        b = conv.bias.data

    gamma = bn.weight.data
    beta = bn.bias.data
    mean = bn.running_mean.data
    var = bn.running_var.data
    eps = bn.eps

    denom = torch.sqrt(var + eps)
    W_fused = W * (gamma / denom).reshape(-1, 1, 1, 1)
    b_fused = (b - mean) * (gamma / denom) + beta

    conv.weight.data = W_fused
    conv.bias = torch.nn.Parameter(b_fused)


# ------------------------------
# CONVERTED SNN: same conv/fc weights, ReLU -> LIF
# After BN fusion, bn layers are Identity.
# ------------------------------
class ConvertedSNN(nn.Module):
    def __init__(self, beta=0.95, threshold=1.0):
        super().__init__()

        # conv blocks (bn will be Identity after fusion)
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1);  self.bn1 = nn.BatchNorm2d(64);  self.lif1 = snn.Leaky(beta=beta, threshold=threshold)
        self.conv2 = nn.Conv2d(64, 64, 3, 1, 1); self.bn2 = nn.BatchNorm2d(64);  self.lif2 = snn.Leaky(beta=beta, threshold=threshold)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(64, 128, 3, 1, 1); self.bn3 = nn.BatchNorm2d(128); self.lif3 = snn.Leaky(beta=beta, threshold=threshold)
        self.conv4 = nn.Conv2d(128, 128, 3, 1, 1);self.bn4 = nn.BatchNorm2d(128); self.lif4 = snn.Leaky(beta=beta, threshold=threshold)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv5 = nn.Conv2d(128, 256, 3, 1, 1);self.bn5 = nn.BatchNorm2d(256); self.lif5 = snn.Leaky(beta=beta, threshold=threshold)
        self.conv6 = nn.Conv2d(256, 256, 3, 1, 1);self.bn6 = nn.BatchNorm2d(256); self.lif6 = snn.Leaky(beta=beta, threshold=threshold)
        self.pool3 = nn.MaxPool2d(2, 2)

        # classifier
        self.fc1 = nn.Linear(256 * 4 * 4, 512);  self.lif7 = snn.Leaky(beta=beta, threshold=threshold)
        self.fc2 = nn.Linear(512, 10)

    def init_mem(self, B, device):
        return {
            "m1": torch.zeros(B, 64, 32, 32, device=device),
            "m2": torch.zeros(B, 64, 32, 32, device=device),
            "m3": torch.zeros(B, 128, 16, 16, device=device),
            "m4": torch.zeros(B, 128, 16, 16, device=device),
            "m5": torch.zeros(B, 256, 8, 8, device=device),
            "m6": torch.zeros(B, 256, 8, 8, device=device),
            "m7": torch.zeros(B, 512, device=device),
        }

    def forward_step(self, x_t, mem):
        # spike statistics for this timestep
        spike_ct = 0.0
        neuron_ct = 0

        cur = self.bn1(self.conv1(x_t)); spk1, mem["m1"] = self.lif1(cur, mem["m1"])
        spike_ct += float(spk1.sum().item()); neuron_ct += spk1.numel()

        cur = self.bn2(self.conv2(spk1)); spk2, mem["m2"] = self.lif2(cur, mem["m2"])
        spike_ct += float(spk2.sum().item()); neuron_ct += spk2.numel()

        cur = self.pool1(spk2)

        cur = self.bn3(self.conv3(cur)); spk3, mem["m3"] = self.lif3(cur, mem["m3"])
        spike_ct += float(spk3.sum().item()); neuron_ct += spk3.numel()

        cur = self.bn4(self.conv4(spk3)); spk4, mem["m4"] = self.lif4(cur, mem["m4"])
        spike_ct += float(spk4.sum().item()); neuron_ct += spk4.numel()

        cur = self.pool2(spk4)

        cur = self.bn5(self.conv5(cur)); spk5, mem["m5"] = self.lif5(cur, mem["m5"])
        spike_ct += float(spk5.sum().item()); neuron_ct += spk5.numel()

        cur = self.bn6(self.conv6(spk5)); spk6, mem["m6"] = self.lif6(cur, mem["m6"])
        spike_ct += float(spk6.sum().item()); neuron_ct += spk6.numel()

        cur = self.pool3(spk6)

        cur = cur.flatten(1)
        cur = self.fc1(cur); spk7, mem["m7"] = self.lif7(cur, mem["m7"])
        spike_ct += float(spk7.sum().item()); neuron_ct += spk7.numel()

        logits_t = self.fc2(spk7)
        return logits_t, mem, spike_ct, neuron_ct


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="outputs/cnn/cnn_best.pt")
    ap.add_argument("--outdir", type=str, default="outputs/ann_snn")
    ap.add_argument("--T", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--beta", type=float, default=0.95)
    ap.add_argument("--threshold", type=float, default=1.0)
    ap.add_argument("--num_workers", type=int, default=2)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    # IMPORTANT for rate coding:
    # Use raw pixels in [0,1]. Do NOT Normalize() before spikegen.rate.
    tf = transforms.Compose([transforms.ToTensor()])
    test_set = datasets.CIFAR10(root="data", train=False, download=True, transform=tf)
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Load CNN checkpoint
    ckpt = torch.load(args.ckpt, map_location=device)
    cnn = SimpleCIFAR10CNN().to(device)
    cnn.load_state_dict(ckpt["model_state"])
    cnn.eval()

    # Build SNN and transfer weights
    snn_model = ConvertedSNN(beta=args.beta, threshold=args.threshold).to(device)
    snn_model.eval()

    # transfer weights (same ordering as CNN Sequential)
    snn_model.conv1.load_state_dict(cnn.features[0].state_dict());  snn_model.bn1.load_state_dict(cnn.features[1].state_dict())
    snn_model.conv2.load_state_dict(cnn.features[3].state_dict());  snn_model.bn2.load_state_dict(cnn.features[4].state_dict())
    snn_model.conv3.load_state_dict(cnn.features[8].state_dict());  snn_model.bn3.load_state_dict(cnn.features[9].state_dict())
    snn_model.conv4.load_state_dict(cnn.features[11].state_dict()); snn_model.bn4.load_state_dict(cnn.features[12].state_dict())
    snn_model.conv5.load_state_dict(cnn.features[16].state_dict()); snn_model.bn5.load_state_dict(cnn.features[17].state_dict())
    snn_model.conv6.load_state_dict(cnn.features[19].state_dict()); snn_model.bn6.load_state_dict(cnn.features[20].state_dict())
    snn_model.fc1.load_state_dict(cnn.classifier[1].state_dict())
    snn_model.fc2.load_state_dict(cnn.classifier[4].state_dict())

    # ---- BN fusion (conv + bn -> conv_fused; bn -> Identity) ----
    snn_model.bn1.eval(); snn_model.bn2.eval(); snn_model.bn3.eval(); snn_model.bn4.eval(); snn_model.bn5.eval(); snn_model.bn6.eval()

    fuse_conv_bn(snn_model.conv1, snn_model.bn1); snn_model.bn1 = nn.Identity()
    fuse_conv_bn(snn_model.conv2, snn_model.bn2); snn_model.bn2 = nn.Identity()
    fuse_conv_bn(snn_model.conv3, snn_model.bn3); snn_model.bn3 = nn.Identity()
    fuse_conv_bn(snn_model.conv4, snn_model.bn4); snn_model.bn4 = nn.Identity()
    fuse_conv_bn(snn_model.conv5, snn_model.bn5); snn_model.bn5 = nn.Identity()
    fuse_conv_bn(snn_model.conv6, snn_model.bn6); snn_model.bn6 = nn.Identity()

    # ---- Fold ANN input normalization into conv1 ----
    # Task 1 used x_norm = (x - mean) / std with CIFAR-10 mean/std:
    mean = torch.tensor([0.4914, 0.4822, 0.4465], device=device).view(1, 3, 1, 1)
    std  = torch.tensor([0.2470, 0.2435, 0.2616], device=device).view(1, 3, 1, 1)

    W = snn_model.conv1.weight.data
    b = snn_model.conv1.bias.data if snn_model.conv1.bias is not None else torch.zeros(W.size(0), device=device)

    W_fused = W / std
    b_fused = b - (W_fused * mean).sum(dim=(1, 2, 3))

    snn_model.conv1.weight.data = W_fused
    snn_model.conv1.bias = torch.nn.Parameter(b_fused)

    print("Weights transferred (CNN->SNN), BN fused, and input normalization folded into conv1.")
    print(f"Rate coding: T={args.T}, beta={args.beta}, threshold={args.threshold}")

    # Metrics accumulators
    correct = 0
    total = 0

    total_spikes = 0.0
    total_neurons = 0.0
    total_images = 0

    stable_steps_sum = 0.0
    stable_count = 0

    start = time.perf_counter()

    for x, y in test_loader:
        x = x.to(device)
        y = y.to(device)
        B = x.size(0)

        # rate coding expects values in [0,1]
        x_rate = (x.clamp(0.0, 1.0) * 2.0).clamp(0.0, 1.0)

        # Spike train: [T, B, C, H, W]
        x_spk = spikegen.rate(x_rate, num_steps=args.T)

        mem = snn_model.init_mem(B, device)
        logits_sum = torch.zeros(B, 10, device=device)
        preds_over_time = []

        batch_spikes = 0.0
        batch_neurons = 0.0

        for t in range(args.T):
            logits_t, mem, spk_ct, neu_ct = snn_model.forward_step(x_spk[t], mem)
            logits_sum += logits_t

            batch_spikes += float(spk_ct)
            batch_neurons += float(neu_ct)

            preds_over_time.append(logits_sum.argmax(dim=1).detach().cpu())

        pred = logits_sum.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += B

        # stable prediction: earliest timestep after which prediction never changes
        preds_over_time = torch.stack(preds_over_time, dim=0)  # [T,B]
        final_pred = preds_over_time[-1]
        for i in range(B):
            fp = final_pred[i].item()
            stable_t = args.T
            for t in range(args.T):
                if torch.all(preds_over_time[t:, i] == fp):
                    stable_t = t + 1
                    break
            stable_steps_sum += stable_t
            stable_count += 1

        total_spikes += batch_spikes
        total_neurons += batch_neurons
        total_images += B

    if device.type == "cuda":
        torch.cuda.synchronize()
    end = time.perf_counter()

    test_acc = correct / total
    avg_stable_steps = stable_steps_sum / stable_count
    spikes_per_image = total_spikes / total_images
    avg_firing_rate = total_spikes / (total_neurons * args.T)  # spikes/neuron/timestep

    print("\n=== Task 2 Results (ANN -> SNN Conversion) ===")
    print(f"T (time steps): {args.T}")
    print(f"Test accuracy: {test_acc*100:.2f}%")
    print(f"Avg time steps to stable prediction: {avg_stable_steps:.2f}")
    print(f"Total spike count per inference: {spikes_per_image:.2f} spikes/image")
    print(f"Average firing rate: {avg_firing_rate*100:.4f}% spikes/neuron/timestep")
    print(f"Wall time for test loop: {end-start:.2f} sec")

    metrics = {
        "T": args.T,
        "beta": args.beta,
        "threshold": args.threshold,
        "test_acc": float(test_acc),
        "avg_stable_steps": float(avg_stable_steps),
        "spikes_per_inference": float(spikes_per_image),
        "avg_firing_rate": float(avg_firing_rate),
        "elapsed_test_seconds": float(end - start),
    }
    out_path = os.path.join(args.outdir, "task2_metrics.json")
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print("Saved:", out_path)


if __name__ == "__main__":
    main()