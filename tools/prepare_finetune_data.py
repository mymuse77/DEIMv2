"""
将 YOLO 格式标注数据转换为 COCO JSON 格式，并与现有 COCO JSON 数据集合并。

使用方式（在 DEIMv2 项目根目录下执行）：
    python tools/prepare_finetune_data.py

脚本流程：
  1. 读取 NEW_IMG_DIR 下的图片和 NEW_LABEL_DIR 下的 YOLO .txt 标注
  2. 将新数据转换为 COCO JSON 格式
  3. 将新数据合并到 BASE_COCO_JSON（原有训练集 COCO JSON）
  4. 将合并后的 JSON 写到 OUTPUT_JSON
"""

import json
from pathlib import Path
from PIL import Image

# ──────────────────── 配置区 —— 按实际情况修改 ────────────────────

# 新增图片所在目录
NEW_IMG_DIR = Path(r"D:\AI\Datasets\syperson2")

# 新增 YOLO 标注所在目录（.txt 文件，与图片同名）
NEW_LABEL_DIR = Path(
    r"D:\AI\Datasets\syperson2\labels_my-project-name_2026-03-24-10-38-45"
)

# 原有训练集 COCO JSON（作为合并基础；若不想合并可设为 None）
BASE_COCO_JSON = Path(r"D:\AI\Datasets\20250731-ppe2286y\train\_annotations.coco.json")

# 合并后 JSON 的输出路径（建议与原训练集放在一起，方便配置）
OUTPUT_JSON = Path(
    r"D:\AI\Datasets\20250731-ppe2286y\train\_annotations_merged.coco.json"
)

# 类别映射：YOLO class_id（0-based）→ 名称
# 必须与原训练集 categories 名称完全一致！通过检查原 COCO JSON 确认：
# {'id': 0, 'name': 'Helmet'}, {'id': 1, 'name': 'NoHelmet'},
# {'id': 2, 'name': 'NoVest'}, {'id': 3, 'name': 'Vest'}
CLASS_NAMES = {
    0: "Helmet",  # 戴头盔
    1: "NoHelmet",  # 未戴头盔
    2: "NoVest",  # 未穿马甲
    3: "Vest",  # 穿马甲
}

# 支持的图片后缀
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

# ─────────────────────────────────────────────────────────────────


def load_base_coco(json_path):
    """加载已有 COCO JSON，返回 (data_dict, max_image_id, max_ann_id, cat_id_map)"""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    max_img_id = max((img["id"] for img in data.get("images", [])), default=0)
    max_ann_id = max((ann["id"] for ann in data.get("annotations", [])), default=0)

    # 建立 category name → id 映射
    cat_name2id = {cat["name"]: cat["id"] for cat in data.get("categories", [])}

    return data, max_img_id, max_ann_id, cat_name2id


def yolo_to_abs(cx, cy, w, h, img_w, img_h):
    """YOLO 归一化中心坐标转 COCO [x_min, y_min, width, height]"""
    abs_cx, abs_cy = cx * img_w, cy * img_h
    abs_w, abs_h = w * img_w, h * img_h
    x_min = abs_cx - abs_w / 2
    y_min = abs_cy - abs_h / 2
    return [round(x_min, 2), round(y_min, 2), round(abs_w, 2), round(abs_h, 2)]


def convert_and_merge():
    # 1. 加载已有训练集 JSON
    if BASE_COCO_JSON and BASE_COCO_JSON.exists():
        print(f"[INFO] 加载原始训练集 COCO JSON: {BASE_COCO_JSON}")
        coco, img_id_offset, ann_id_offset, cat_name2id = load_base_coco(BASE_COCO_JSON)
        print(
            f"       已有图片: {len(coco['images'])} 张，标注: {len(coco['annotations'])} 条"
        )
    else:
        print("[INFO] 未找到原始 COCO JSON，将从头创建新的 COCO JSON")
        coco = {"images": [], "annotations": [], "categories": []}
        img_id_offset = 0
        ann_id_offset = 0
        cat_name2id = {}

    # 2. 确保 categories 与 CLASS_NAMES 对齐
    # 注意：原数据集使用 0-based category_id，与 YOLO class_id 直接对应，不做 +1 偏移
    existing_cat_ids = {cat["id"] for cat in coco["categories"]}
    for cls_id, cls_name in CLASS_NAMES.items():
        coco_cat_id = cls_id  # 与原数据集保持一致：0-based
        if coco_cat_id not in existing_cat_ids:
            coco["categories"].append(
                {"id": coco_cat_id, "name": cls_name, "supercategory": cls_name}
            )
            existing_cat_ids.add(coco_cat_id)
            cat_name2id[cls_name] = coco_cat_id
        else:
            cat_name2id[cls_name] = coco_cat_id

    # 3. 收集新图片
    img_files = [
        p for p in NEW_IMG_DIR.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS
    ]
    img_files.sort()
    print(f"\n[INFO] 在 {NEW_IMG_DIR} 下找到 {len(img_files)} 张图片")

    new_img_count = 0
    new_ann_count = 0
    skipped = 0

    for img_path in img_files:
        label_path = NEW_LABEL_DIR / (img_path.stem + ".txt")
        if not label_path.exists():
            print(f"  [WARN] 找不到标注文件，跳过: {label_path.name}")
            skipped += 1
            continue

        # 读取图片尺寸
        try:
            with Image.open(img_path) as im:
                img_w, img_h = im.size
        except Exception as e:
            print(f"  [WARN] 无法打开图片，跳过: {img_path.name}  ({e})")
            skipped += 1
            continue

        img_id_offset += 1
        new_img_id = img_id_offset

        # 添加到 images 列表
        coco["images"].append(
            {
                "id": new_img_id,
                "file_name": img_path.name,
                "width": img_w,
                "height": img_h,
            }
        )
        new_img_count += 1

        # 读取 YOLO 标注
        with open(label_path, "r") as f:
            lines = [line.strip() for line in f if line.strip()]

        for line in lines:
            parts = line.split()
            if len(parts) < 5:
                continue
            cls_yolo = int(parts[0])
            cx, cy, bw, bh = (
                float(parts[1]),
                float(parts[2]),
                float(parts[3]),
                float(parts[4]),
            )

            bbox = yolo_to_abs(cx, cy, bw, bh, img_w, img_h)
            area = round(bbox[2] * bbox[3], 2)

            coco_cat_id = cls_yolo  # 与原数据集保持一致：0-based，不做 +1 偏移

            ann_id_offset += 1
            coco["annotations"].append(
                {
                    "id": ann_id_offset,
                    "image_id": new_img_id,
                    "category_id": coco_cat_id,
                    "bbox": bbox,
                    "area": area,
                    "iscrowd": 0,
                    "segmentation": [],
                }
            )
            new_ann_count += 1

    # 4. 写出合并后的 JSON
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(coco, f, ensure_ascii=False, indent=2)

    print(f"\n{'=' * 60}")
    print(f"  转换完成！")
    print(
        f"  新增图片: {new_img_count} 张，新增标注: {new_ann_count} 条，跳过: {skipped} 张"
    )
    print(
        f"  合并后总图片: {len(coco['images'])} 张，总标注: {len(coco['annotations'])} 条"
    )
    print(f"  输出 JSON: {OUTPUT_JSON}")
    print(f"{'=' * 60}")
    print(
        "\n[\u4e0b\u4e00\u6b65] \u4fee\u6539\u6570\u636e\u96c6\u914d\u7f6e\u4e2d\u7684 ann_file \u6307\u5411\u5408\u5e76\u540e\u7684 JSON\uff0c\u7136\u540e\u8fd0\u884c\uff1a"
    )
    print(
        "  python train.py -c configs/deimv2/deimv2_hgnetv2_atto_helmet_cpu_highprec.yml \\"
    )
    print(
        "                  -t D:/AI/Git/DEIMv2/outputs/deimv2_hgnetv2_atto_helmet_cpu_highprec/best_stg2.pth"
    )


if __name__ == "__main__":
    convert_and_merge()
