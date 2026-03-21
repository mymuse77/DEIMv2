"""
YOLO 格式数据集转换为 COCO 格式

用于将 YOLO 格式的标注文件转换为 DEIMv2 可使用的 COCO JSON 格式

用法:
    python yolo_to_coco.py --data_dir <YOLO数据集目录> --output_dir <输出目录>

示例:
    python yolo_to_coco.py --data_dir "D:/AI/Datasets/Dataset of Personal Protective Equipment (PPE)/20250731-ppe2286y" --output_dir "./ppe_coco"
"""

import os
import json
import argparse
from pathlib import Path
from datetime import datetime
from PIL import Image
from tqdm import tqdm


def yolo_to_coco(yolo_data_dir, output_dir, class_names=None):
    """
    将 YOLO 格式数据集转换为 COCO 格式

    Args:
        yolo_data_dir: YOLO 数据集根目录 (包含 train/, valid/, data.yaml)
        output_dir: 输出 COCO 格式数据的目录
        class_names: 类别名称列表，如果为 None 则从 data.yaml 读取
    """
    yolo_path = Path(yolo_data_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 尝试从 data.yaml 读取类别
    if class_names is None:
        import yaml
        yaml_path = yolo_path / "data.yaml"
        if yaml_path.exists():
            with open(yaml_path, 'r', encoding='utf-8') as f:
                data_yaml = yaml.safe_load(f)
            class_names = data_yaml.get('names', [])
            print(f"从 data.yaml 读取类别: {class_names}")
        else:
            raise ValueError("未找到 data.yaml，请手动提供 class_names")

    # 创建类别列表
    categories = []
    for i, name in enumerate(class_names):
        categories.append({
            "id": i,
            "name": name,
            "supercategory": name
        })

    # 处理训练集和验证集
    for split in ['train', 'valid']:
        images_dir = yolo_path / split / 'images'
        labels_dir = yolo_path / split / 'labels'

        if not images_dir.exists():
            print(f"跳过 {split}，目录不存在: {images_dir}")
            continue

        print(f"\n处理 {split} 集...")

        # COCO 格式结构
        coco_output = {
            "info": {
                "description": f"Converted from YOLO format",
                "version": "1.0",
                "year": datetime.now().year,
                "date_created": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            },
            "licenses": [],
            "categories": categories,
            "images": [],
            "annotations": []
        }

        image_id = 0
        annotation_id = 0

        # 获取所有图像文件
        image_files = list(images_dir.glob('*.jpg')) + \
                      list(images_dir.glob('*.jpeg')) + \
                      list(images_dir.glob('*.png')) + \
                      list(images_dir.glob('*.bmp'))

        for img_file in tqdm(image_files, desc=f"Converting {split}"):
            # 获取图像尺寸
            try:
                with Image.open(img_file) as img:
                    width, height = img.size
            except Exception as e:
                print(f"无法读取图像 {img_file}: {e}")
                continue

            # 添加图像信息
            image_info = {
                "id": image_id,
                "file_name": img_file.name,
                "width": width,
                "height": height
            }
            coco_output["images"].append(image_info)

            # 读取对应的标注文件
            label_file = labels_dir / (img_file.stem + '.txt')

            if label_file.exists():
                with open(label_file, 'r') as f:
                    lines = f.readlines()

                for line in lines:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue

                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    box_width = float(parts[3])
                    box_height = float(parts[4])

                    # YOLO 格式 (归一化的中心点坐标和宽高) 转换为 COCO 格式 (左上角坐标和宽高)
                    x_min = (x_center - box_width / 2) * width
                    y_min = (y_center - box_height / 2) * height
                    bbox_width = box_width * width
                    bbox_height = box_height * height

                    # 确保坐标有效
                    x_min = max(0, x_min)
                    y_min = max(0, y_min)
                    bbox_width = min(bbox_width, width - x_min)
                    bbox_height = min(bbox_height, height - y_min)

                    annotation = {
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": class_id,
                        "bbox": [x_min, y_min, bbox_width, bbox_height],
                        "area": bbox_width * bbox_height,
                        "iscrowd": 0
                    }
                    coco_output["annotations"].append(annotation)
                    annotation_id += 1

            image_id += 1

        # 保存 COCO JSON 文件
        output_json = output_path / f"{split}.json"
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(coco_output, f, ensure_ascii=False, indent=2)

        print(f"  图像数量: {len(coco_output['images'])}")
        print(f"  标注数量: {len(coco_output['annotations'])}")
        print(f"  已保存到: {output_json}")

    # 复制图像到输出目录（可选）
    print("\n转换完成!")
    print(f"\n输出目录结构:")
    print(f"  {output_path}/")
    print(f"    train.json")
    print(f"    valid.json")
    print(f"\n图像路径保持原位置，请在配置文件中指定正确的 img_folder")

    return output_path


def main():
    parser = argparse.ArgumentParser(description='YOLO to COCO format converter')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='YOLO 数据集根目录')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='输出 COCO JSON 文件的目录')
    parser.add_argument('--class_names', type=str, nargs='+', default=None,
                        help='类别名称列表 (可选，默认从 data.yaml 读取)')

    args = parser.parse_args()

    yolo_to_coco(args.data_dir, args.output_dir, args.class_names)


if __name__ == '__main__':
    main()