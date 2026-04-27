import time
import argparse
from typing import Dict, List, Tuple

import torch
from torchvision import transforms
from torchvision.datasets import VOCDetection
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from ultralytics import YOLO


VOC_CLASSES = [
    "aeroplane","bicycle","bird","boat","bottle",
    "bus","car","cat","chair","cow",
    "diningtable","dog","horse","motorbike","person",
    "pottedplant","sheep","sofa","train","tvmonitor"
]

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
    ap.add_argument("--image_set", default="test")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    # YOLOv8 nano pretrained on COCO
    yolo = YOLO("yolov8n.pt")

    # build YOLO name->id (normalize names similar to COCO)
    # yolo.names: dict {id: name}
    yolo_name_to_id = {name: int(i) for i, name in yolo.names.items()}

    tf = transforms.Compose([transforms.ToTensor()])
    ds = VOCDetection(root="data_voc", year=args.year, image_set=args.image_set, download=True, transform=tf)
    n = min(args.n, len(ds))

    metric = MeanAveragePrecision(iou_type="bbox")
    times_ms: List[float] = []

    for i in range(n):
        img_t, target = ds[i]          # tensor CHW in [0,1]
        gt_boxes, gt_labels = parse_voc_target(target, yolo_name_to_id)
        gts = [{
            "boxes": gt_boxes.to(device),
            "labels": gt_labels.to(device),
        }]

        # YOLO expects numpy or path; easiest: convert tensor -> numpy HWC uint8
        img = (img_t.permute(1,2,0).numpy() * 255).astype("uint8")

        if i < args.warmup:
            _ = yolo.predict(source=img, imgsz=args.imgsz, conf=args.conf, verbose=False)
            continue

        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        r = yolo.predict(source=img, imgsz=args.imgsz, conf=args.conf, verbose=False)[0]

        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000.0)

        if r.boxes is None or len(r.boxes) == 0:
            preds = [{"boxes": torch.zeros((0,4), device=device),
                      "scores": torch.zeros((0,), device=device),
                      "labels": torch.zeros((0,), dtype=torch.long, device=device)}]
        else:
            boxes_xyxy = r.boxes.xyxy.to(device)
            scores = r.boxes.conf.to(device)
            labels = r.boxes.cls.to(device).long()
            preds = [{"boxes": boxes_xyxy, "scores": scores, "labels": labels}]

        metric.update(preds, gts)

    result = metric.compute()
    map_5095 = float(result["map"])
    map_50 = float(result["map_50"])
    avg_ms = sum(times_ms) / max(len(times_ms), 1)

    print("\n=== YOLOv8-nano (ultralytics) on Pascal VOC subset ===")
    print(f"Subset: VOC{args.year} {args.image_set}, n={n}, warmup={args.warmup}")
    print(f"mAP@[.50:.95] (COCO-style): {map_5095:.4f}")
    print(f"mAP@0.50: {map_50:.4f}")
    print(f"Inference time: {avg_ms:.2f} ms/image")

if __name__ == "__main__":
    main()