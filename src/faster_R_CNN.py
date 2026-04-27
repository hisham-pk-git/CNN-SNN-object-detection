import time
import argparse
from typing import Dict, List, Tuple

import torch
from torchvision import transforms
from torchvision.datasets import VOCDetection
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights

from torchmetrics.detection.mean_ap import MeanAveragePrecision


VOC_CLASSES = [
    "aeroplane","bicycle","bird","boat","bottle",
    "bus","car","cat","chair","cow",
    "diningtable","dog","horse","motorbike","person",
    "pottedplant","sheep","sofa","train","tvmonitor"
]

# VOC->COCO-ish name normalization (for pretrained COCO detectors)
VOC_TO_COCO_NAME = {
    "aeroplane": "airplane",
    "diningtable": "dining table",
    "motorbike": "motorcycle",
    "pottedplant": "potted plant",
    "sofa": "couch",
    "tvmonitor": "tv",
}

def parse_voc_target(target: Dict, name_to_id: Dict[str, int]) -> Tuple[torch.Tensor, torch.Tensor]:
    ann = target["annotation"]
    objs = ann.get("object", [])
    if isinstance(objs, dict):
        objs = [objs]

    boxes = []
    labels = []

    for obj in objs:
        cls = obj["name"]
        cls_norm = VOC_TO_COCO_NAME.get(cls, cls)
        if cls_norm not in name_to_id:
            continue
        b = obj["bndbox"]
        xmin = float(b["xmin"])
        ymin = float(b["ymin"])
        xmax = float(b["xmax"])
        ymax = float(b["ymax"])
        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(name_to_id[cls_norm])

    if len(boxes) == 0:
        return torch.zeros((0, 4), dtype=torch.float32), torch.zeros((0,), dtype=torch.int64)

    return torch.tensor(boxes, dtype=torch.float32), torch.tensor(labels, dtype=torch.int64)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", default="2007")
    ap.add_argument("--image_set", default="test")  # test or val
    ap.add_argument("--n", type=int, default=200, help="subset size")
    ap.add_argument("--batch_size", type=int, default=1)  # keep 1 for timing simplicity
    ap.add_argument("--warmup", type=int, default=10)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn(weights=weights).to(device).eval()

    # Build class name->id from model metadata (COCO category names)
    categories = weights.meta["categories"]  # list where index = label id
    name_to_id = {name: i for i, name in enumerate(categories)}

    tf = transforms.Compose([transforms.ToTensor()])

    ds = VOCDetection(root="data_voc", year=args.year, image_set=args.image_set, download=True, transform=tf)
    n = min(args.n, len(ds))

    metric = MeanAveragePrecision(iou_type="bbox")  # default mAP@[.50:.95] like COCO
    times_ms: List[float] = []

    # Warmup + eval loop
    for i in range(n):
        img, target = ds[i]
        img = img.to(device)
        gt_boxes, gt_labels = parse_voc_target(target, name_to_id)

        # Prepare metric dicts
        gts = [{
            "boxes": gt_boxes.to(device),
            "labels": gt_labels.to(device),
        }]

        # Timing (exclude first few warmup iterations)
        if i < args.warmup:
            _ = model([img])
            continue

        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        preds_raw = model([img])[0]

        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000.0)

        preds = [{
            "boxes": preds_raw["boxes"].detach(),
            "scores": preds_raw["scores"].detach(),
            "labels": preds_raw["labels"].detach(),
        }]

        metric.update(preds, gts)

    result = metric.compute()
    map_5095 = float(result["map"])
    map_50 = float(result["map_50"])
    avg_ms = sum(times_ms) / max(len(times_ms), 1)

    print("\n=== Faster R-CNN (fasterrcnn_resnet50_fpn) on Pascal VOC subset ===")
    print(f"Subset: VOC{args.year} {args.image_set}, n={n}, warmup={args.warmup}")
    print(f"mAP@[.50:.95] (COCO-style): {map_5095:.4f}")
    print(f"mAP@0.50: {map_50:.4f}")
    print(f"Inference time: {avg_ms:.2f} ms/image")

if __name__ == "__main__":
    main()