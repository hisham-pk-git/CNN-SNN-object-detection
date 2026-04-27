import os, random, shutil
from pathlib import Path
import xml.etree.ElementTree as ET

VOC_CLASSES = [
    "aeroplane","bicycle","bird","boat","bottle",
    "bus","car","cat","chair","cow",
    "diningtable","dog","horse","motorbike","person",
    "pottedplant","sheep","sofa","train","tvmonitor"
]
cls_to_id = {c:i for i,c in enumerate(VOC_CLASSES)}

def voc_xml_to_yolo(xml_path: Path, img_w: int, img_h: int):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    lines = []
    for obj in root.findall("object"):
        name = obj.find("name").text
        if name not in cls_to_id:
            continue
        b = obj.find("bndbox")
        xmin = float(b.find("xmin").text)
        ymin = float(b.find("ymin").text)
        xmax = float(b.find("xmax").text)
        ymax = float(b.find("ymax").text)

        # YOLO: class cx cy w h normalized
        cx = ((xmin + xmax) / 2.0) / img_w
        cy = ((ymin + ymax) / 2.0) / img_h
        bw = (xmax - xmin) / img_w
        bh = (ymax - ymin) / img_h
        lines.append(f"{cls_to_id[name]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    return lines

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--voc_root", default="data_voc/VOCdevkit/VOC2007")
    ap.add_argument("--n_train", type=int, default=1500)
    ap.add_argument("--n_val", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data_yolo_voc")
    args = ap.parse_args()

    random.seed(args.seed)
    voc = Path(args.voc_root)
    ann_dir = voc / "Annotations"
    img_dir = voc / "JPEGImages"

    out = Path(args.out)
    for split in ["train", "val"]:
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Use trainval.txt list
    ids = (voc / "ImageSets" / "Main" / "trainval.txt").read_text().strip().split()
    random.shuffle(ids)

    train_ids = ids[:args.n_train]
    val_ids = ids[args.n_train:args.n_train + args.n_val]

    def process(split_ids, split_name):
        for img_id in split_ids:
            xml_path = ann_dir / f"{img_id}.xml"
            img_path = img_dir / f"{img_id}.jpg"

            # read size from xml
            root = ET.parse(xml_path).getroot()
            size = root.find("size")
            w = int(size.find("width").text)
            h = int(size.find("height").text)

            yolo_lines = voc_xml_to_yolo(xml_path, w, h)

            # copy image
            shutil.copy2(img_path, out / "images" / split_name / f"{img_id}.jpg")

            # write label (even if empty file; YOLO expects label file)
            label_path = out / "labels" / split_name / f"{img_id}.txt"
            label_path.write_text("\n".join(yolo_lines) + ("\n" if yolo_lines else ""))

    process(train_ids, "train")
    process(val_ids, "val")

    # dataset yaml
    yaml_text = f"""path: {out.resolve()}
train: images/train
val: images/val
names:
"""
    for i, c in enumerate(VOC_CLASSES):
        yaml_text += f"  {i}: {c}\n"

    (out / "voc07_subset.yaml").write_text(yaml_text)
    print("✅ Done")
    print("YAML:", out / "voc07_subset.yaml")
    print("Train images:", len(train_ids))
    print("Val images:", len(val_ids))

if __name__ == "__main__":
    main()