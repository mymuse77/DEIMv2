import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

def main():
    img_dir = Path(r"D:\AI\Datasets\syperson2")
    
    # 自动查找里面以 labels 开头的文件夹
    label_dirs = [d for d in img_dir.iterdir() if d.is_dir() and d.name.startswith("labels")]
    
    if not label_dirs:
        print(f"[ERROR] 在 {img_dir} 中找不到标签文件夹。")
        return
    
    label_dir = label_dirs[0]
    print(f"[INFO] 找到标签目录: {label_dir}")
    
    output_dir = img_dir / "vis_results"
    output_dir.mkdir(exist_ok=True)
    
    # 尝试加载中文字体，以便在图上正确显示中文
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

    # 根据你的标注类别进行映射（如果 1 代表未戴头盔，2 代表未穿马甲，请根据需要调整）
    classes = {
        0: "戴头盔",
        1: "未戴头盔",
        2: "未穿马甲",
        3: "穿马甲"
    }
    colors = {
        0: "#00CC44", # 绿色
        1: "#FF3333", # 红色
        2: "#FF8C00", # 橙色
        3: "#3399FF"  # 蓝色
    }

    # 读取所有图片
    img_paths = list(img_dir.glob("*.png")) + list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.jpeg"))
    print(f"[INFO] 找到 {len(img_paths)} 张图片\n")
    
    count = 0
    for img_path in img_paths:
        # 在标签目录中寻找同名的 .txt 文件
        label_file = label_dir / f"{img_path.stem}.txt"
        if not label_file.exists():
            continue
            
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"[WARN] 无法读取图片 {img_path.name}: {e}")
            continue
            
        draw = ImageDraw.Draw(img)
        w, h = img.size
        
        with open(label_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:
                # YOLO 格式: class_id x_center y_center width height (归一化到 0-1 之间)
                cls_id = int(parts[0])
                cx = float(parts[1])
                cy = float(parts[2])
                bw = float(parts[3])
                bh = float(parts[4])
                
                # 将归一化坐标转换为绝对像素坐标
                x1 = (cx - bw / 2) * w
                y1 = (cy - bh / 2) * h
                x2 = (cx + bw / 2) * w
                y2 = (cy + bh / 2) * h
                
                cls_name = classes.get(cls_id, f"类别{cls_id}")
                color = colors.get(cls_id, "yellow")
                
                # 画框
                draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
                
                # 文字背景和文字
                text = f"{cls_name}"
                try:
                    tw = draw.textlength(text, font=font)
                    th = 20
                except Exception:
                    tw, th = 80, 20
                
                draw.rectangle([x1, y1 - th - 2, x1 + tw + 4, y1], fill=color)
                draw.text((x1 + 2, y1 - th - 1), text, fill="white", font=font)
                
        # 保存可视化结果
        out_path = output_dir / f"vis_{img_path.name}"
        img.save(out_path)
        print(f"  已可视化并保存: {out_path.name} (目标数: {len(lines)})")
        count += 1
        
    print(f"\n[DONE] 测试完毕！共处理了 {count} 张图片。")
    print(f"可视化结果查看路径: {output_dir}")

if __name__ == "__main__":
    main()
