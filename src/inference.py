import os
import argparse
from typing import Tuple, List

import cv2
import numpy as np
import torch
from torchvision import transforms
from torchvision.datasets import VOCDetection
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights


def to_bgr_uint8(img_tensor: torch.Tensor) -> np.ndarray:
    """
    img_tensor: CHW float [0,1]
    returns: HWC uint8 BGR for OpenCV
    """
    img = (img_tensor.permute(1, 2, 0).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img_bgr


def draw_detections(
    img_bgr: np.ndarray,
    boxes: np.ndarray,
    labels: List[str],
    scores: np.ndarray,
    color: Tuple[int, int, int] = (0, 255, 0),
) -> np.ndarray:
    out = img_bgr.copy()
    h, w = out.shape[:2]

    for box, lab, sc in zip(boxes, labels, scores):
        x1, y1, x2, y2 = box.astype(int)

        # Clamp to image bounds
        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w - 1))
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h - 1))

        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        text = f"{lab} {sc:.2f}"
        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        # Draw filled label background
        y_text = max(0, y1 - th - baseline - 4)
        cv2.rectangle(out, (x1, y_text), (x1 + tw + 6, y_text + th + baseline + 4), color, -1)
        cv2.putText(out, text, (x1 + 3, y_text + th + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    return out


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", default="2007")
    ap.add_argument("--image_set", default="test")
    ap.add_argument("--idx1", type=int, default=0, help="First VOC index")
    ap.add_argument("--idx2", type=int, default=1, help="Second VOC index")
    ap.add_argument("--conf", type=float, default=0.50, help="Confidence threshold")
    ap.add_argument("--topk", type=int, default=20, help="Max detections to draw per image")
    ap.add_argument("--outdir", default="outputs/task5b", help="Where to save annotated images")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    # Load pretrained Faster R-CNN
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn(weights=weights).to(device).eval()
    categories = weights.meta["categories"]  # index -> class name (COCO)

    # Load VOC dataset (downloads if missing)
    tf = transforms.Compose([transforms.ToTensor()])
    ds = VOCDetection(root="data_voc", year=args.year, image_set=args.image_set, download=True, transform=tf)

    for idx in [args.idx1, args.idx2]:
        img_t, target = ds[idx]  # img_t is CHW float [0,1]
        img_in = img_t.to(device)

        preds = model([img_in])[0]
        boxes = preds["boxes"].detach().cpu().numpy()
        scores = preds["scores"].detach().cpu().numpy()
        labels_id = preds["labels"].detach().cpu().numpy()

        # Filter by confidence
        keep = scores >= args.conf
        boxes = boxes[keep]
        scores = scores[keep]
        labels_id = labels_id[keep]

        # Top-K
        if len(scores) > args.topk:
            order = np.argsort(-scores)[: args.topk]
            boxes = boxes[order]
            scores = scores[order]
            labels_id = labels_id[order]

        labels_txt = [categories[int(i)] if int(i) < len(categories) else str(int(i)) for i in labels_id]

        img_bgr = to_bgr_uint8(img_t)
        vis = draw_detections(img_bgr, boxes, labels_txt, scores)

        out_path = os.path.join(args.outdir, f"fasterrcnn_voc_{args.image_set}_idx{idx}.png")
        cv2.imwrite(out_path, vis)

        print(f"Saved: {out_path} | detections drawn: {len(scores)} (conf>={args.conf})")

    print("Done.")


if __name__ == "__main__":
    main()