import os
import json
import yaml
import cv2
from tqdm import tqdm

import kagglehub

def yolo_to_coco(data_yaml_path, save_dir):
    # === 1️⃣ 加载 data.yaml ===
    with open(data_yaml_path, 'r') as f:
        data_cfg = yaml.safe_load(f)
    names = data_cfg['names']
    nc = data_cfg['nc']
    print(f"✅ Loaded YAML with {nc} classes: {names}")

    # === 2️⃣ 准备输出目录 ===
    # os.makedirs(os.path.join(save_dir, "annotations"), exist_ok=True)

    # === 3️⃣ 处理每个子集 ===
    subsets = {
        'train': data_cfg['train'].replace('../', ''),
        'val': data_cfg['val'].replace('../', ''),
        'test': data_cfg['test'].replace('../', '')
    }

    for split, img_dir_rel in subsets.items():
        img_dir = os.path.join(os.path.dirname(data_yaml_path), img_dir_rel)
        label_dir = img_dir.replace('images', 'labels')

        print(f"\n🚀 Processing {split} set:")
        image_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        images, annotations = [], []
        ann_id = 1

        for img_id, img_name in enumerate(tqdm(image_files, desc=f"Converting {split}")):
            img_path = os.path.join(img_dir, img_name)
            label_path = os.path.join(label_dir, os.path.splitext(img_name)[0] + ".txt")

            # 读取图像尺寸
            img = cv2.imread(img_path)
            if img is None:
                print(f"⚠️ Warning: cannot read image {img_path}, skipping.")
                continue
            height, width = img.shape[:2]

            # 因为kagglehub的实际文件目录是 valid，所以此处进行处理，val的json文件名，但是路径是valid。
            if split=="val":
                file_name =  os.path.join("valid", 'images', img_name)
            else:
                file_name = os.path.join(split, 'images', img_name)
            images.append({
                'id': img_id + 1,
                'file_name': file_name,
                'width': width,
                'height': height
            })

            if not os.path.exists(label_path):
                continue

            # 读取标签
            with open(label_path, 'r') as lf:
                for line in lf.readlines():
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    cls, x, y, w, h = map(float, parts)
                    cls = int(cls)
                    # YOLO: cx, cy, w, h (normalized) -> COCO: x_min, y_min, w, h (pixels)
                    x_min = (x - w / 2) * width
                    y_min = (y - h / 2) * height
                    w_box = w * width
                    h_box = h * height
                    annotations.append({
                        'id': ann_id,
                        'image_id': img_id + 1,
                        'category_id': cls + 1,
                        'bbox': [x_min, y_min, w_box, h_box],
                        'area': w_box * h_box,
                        'iscrowd': 0
                    })
                    ann_id += 1

        # === 4️⃣ 生成 COCO 格式 JSON ===
        coco_dict = {
            "info": {
                "description": "crop pest dataset",
                "version": "1.0",
                "year": 2025
            },
            "licenses": [],
            'images': images,
            'annotations': annotations,
            'categories': [{'id': i + 1, 'name': name} for i, name in enumerate(names)]
        }

        ann_path = os.path.join(save_dir, f"instances_{split}.json")
        with open(ann_path, 'w', encoding='utf-8') as f:
            json.dump(coco_dict, f, indent=2)
        print(f"✅ Saved COCO annotation to: {ann_path}")
        print(f"📊 {len(images)} images, {len(annotations)} annotations converted.")

    print("\n🎯 Conversion complete!")

# === 🔧 主程序入口 ===
if __name__ == "__main__":
    # 你的数据路径
    path = kagglehub.dataset_download("rupankarmajumdar/crop-pests-dataset")
    print("Dataset downloaded to:", path)

    data_yaml = os.path.join(path, "data.yaml")
    save_dir = path
    yolo_to_coco(data_yaml, save_dir)
