import os
import json
import random
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from collections import defaultdict

def main():
    img_dir = Path(r"D:\AI\Datasets\20250731-ppe2286y\train\images")
    json_path = Path(r"D:\AI\Datasets\20250731-ppe2286y\train\_annotations_merged.coco.json")
    
    # 结果输出目录 (与images平级)
    output_dir = img_dir.parent / "vis_results_coco"
    output_dir.mkdir(exist_ok=True)
    
    if not json_path.exists():
        print(f"[ERROR] 找不到COCO标注文件: {json_path}")
        return
        
    print(f"[INFO] 正在读取COCO标注文件: {json_path.name} ...")
    with open(json_path, "r", encoding="utf-8") as f:
        coco_data = json.load(f)
        
    # 解析 categories
    categories = {cat["id"]: cat["name"] for cat in coco_data.get("categories", [])}
    print(f"[INFO] 类别信息: {categories}")
    
    # 建立 colors 映射
    colors_list = ["#00CC44", "#FF3333", "#FF8C00", "#3399FF", "#9933FF", "#33FFFF", "#FF3399", "#FFFF33"]
    colors = {}
    for i, cat_id in enumerate(categories.keys()):
        colors[cat_id] = colors_list[i % len(colors_list)]
        
    # 解析 images 和 annotations
    images_info = {img["id"]: img for img in coco_data.get("images", [])}
    
    annotations_map = defaultdict(list)
    for ann in coco_data.get("annotations", []):
        annotations_map[ann["image_id"]].append(ann)
        
    print(f"[INFO] 共有 {len(images_info)} 张图片, {len(coco_data.get('annotations', []))} 个标注框")
    
    # 尝试加载中文字体
    fonts = [
        r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\simhei.ttf",
        r"C:\Windows\Fonts\simsun.ttc",
    ]
    font = None
    for f in fonts:
        if os.path.exists(f):
            try:
                font = ImageFont.truetype(f, 20)
                break
            except Exception:
                pass
    if font is None:
        font = ImageFont.load_default()

    # 设置为 0 表示处理所有图片（不抽样）
    SAMPLE_NUM = 0 
    
    # 获取所有的图片 ID (包括没有任何标注框的背景负样本)
    all_image_ids = list(images_info.keys())
    if SAMPLE_NUM > 0 and len(all_image_ids) > SAMPLE_NUM:
        sample_ids = random.sample(all_image_ids, SAMPLE_NUM)
    else:
        sample_ids = all_image_ids
        
    print(f"[INFO] 准备对全部 {len(sample_ids)} 张图片进行可视化测试...\n")
    
    count = 0
    for img_id in sample_ids:
        img_info = images_info.get(img_id)
        if not img_info:
            continue
            
        img_name = img_info["file_name"]
        
        # 兼容处理：防 file_name 带有路径或者多余前缀
        img_name = Path(img_name).name
        
        img_path = img_dir / img_name
        if not img_path.exists():
            print(f"[WARN] 找不到图片文件，跳过: {img_path}")
            continue
            
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"[WARN] 无法读取图片 {img_path.name}: {e}")
            continue
            
        draw = ImageDraw.Draw(img)
        anns = annotations_map[img_id]
        
        for ann in anns:
            # COCO 格式: [x_min, y_min, width, height] (绝对像素)
            bbox = ann["bbox"]
            if len(bbox) != 4:
                continue
                
            x1, y1, w, h = bbox
            x2 = x1 + w
            y2 = y1 + h
            
            cat_id = ann["category_id"]
            cat_name = categories.get(cat_id, f"类别{cat_id}")
            color = colors.get(cat_id, "yellow")
            
            # 画框
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            
            # 文字背景和文字
            text = f"{cat_name}"
            try:
                tw = draw.textlength(text, font=font)
                th = 20
            except Exception:
                tw, th = 80, 20
            
            draw.rectangle([x1, y1 - th - 2, x1 + tw + 4, y1], fill=color)
            draw.text((x1 + 2, y1 - th - 1), text, fill="white", font=font)
                
        # 保存可视化结果
        out_path = output_dir / f"coco_vis_{img_path.name}"
        img.save(out_path)
        print(f"  已保存: {out_path.name}")
        count += 1
        
    print(f"\n[DONE] 测试完毕！共处理并保存了 {count} 张图片。")
    print(f"可视化结果查看路径: {output_dir}")

if __name__ == "__main__":
    main()
