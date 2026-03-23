"""
多图测试功能 - 用于批量测试指定图像
"""
import os
import sys
from pathlib import Path
import random

# 添加DEIMv2路径
sys.path.append(r'D:\AI\Git\DEIMv2')

import torch
import matplotlib.pyplot as plt
from PIL import Image

# 导入notebook中定义的函数
# 假设这些函数已经在notebook中定义

def test_multiple_images(image_paths, model, preprocessor, postprocessor, device,
                         threshold=0.05, max_images=6, save_dir=None,
                         CLASS_NAMES=None, COLORS=None, TARGET_CLASSES=None):
    """
    批量测试多张图像

    Args:
        image_paths: 图像路径列表
        model: 检测模型
        preprocessor: 预处理器
        postprocessor: 后处理器
        device: 计算设备
        threshold: 置信度阈值
        max_images: 最多显示的图像数量
        save_dir: 保存结果的目录（可选）
        CLASS_NAMES: 类别名称映射
        COLORS: 颜色映射
        TARGET_CLASSES: 目标类别列表

    Returns:
        results: 每张图像的检测结果
    """
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    results = []
    num_to_show = min(len(image_paths), max_images)

    # 创建子图
    cols = 2
    rows = (num_to_show + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(16, rows * 6))
    if rows == 1:
        axes = [axes] if num_to_show == 1 else list(axes)
    else:
        axes = axes.flatten()

    for idx, img_path in enumerate(image_paths[:num_to_show]):
        try:
            # 这里需要调用detect_helmets函数
            # 但为了独立性，我们可以直接在函数内部实现检测逻辑
            from PIL import ImageDraw, ImageFont

            # 加载图像
            orig_image = Image.open(img_path).convert('RGB')
            image_tensor, (orig_w, orig_h) = preprocessor(orig_image)
            image_tensor = image_tensor.unsqueeze(0).to(device)

            # 推理
            with torch.no_grad():
                outputs = model(image_tensor)

            # 后处理
            orig_target_sizes = torch.tensor([[orig_h, orig_w]]).to(device)
            results_post = postprocessor(outputs, orig_target_sizes)[0]

            # 解析结果
            detections = []
            boxes = results_post['boxes'].cpu().numpy()
            scores = results_post['scores'].cpu().numpy()
            labels = results_post['labels'].cpu().numpy()

            for box, score, label in zip(boxes, scores, labels):
                if score >= threshold:
                    class_id = int(label)
                    if class_id in (TARGET_CLASSES or [0, 1]):
                        detections.append({
                            'class_id': class_id,
                            'class_name': (CLASS_NAMES or {0: 'Helmet', 1: 'NoHelmet'}).get(class_id, f'Class_{class_id}'),
                            'bbox': box.tolist(),
                            'confidence': float(score)
                        })

            # 绘制结果
            result_image = orig_image.copy()
            draw = ImageDraw.Draw(result_image)

            try:
                font = ImageFont.truetype("C:/Windows/Fonts/simhei.ttf", 20)
            except:
                try:
                    font = ImageFont.truetype("arial.ttf", 16)
                except:
                    font = ImageFont.load_default()

            for det in detections:
                class_id = det['class_id']
                bbox = det['bbox']
                conf = det['confidence']
                class_name = det['class_name']

                x1, y1, x2, y2 = [int(x) for x in bbox]
                color = (COLORS or {0: '#00FF00', 1: '#FF0000'}).get(class_id, '#0000FF')

                draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
                label_text = f"{class_name} {conf:.2f}"
                text_bbox = draw.textbbox((x1, y1 - 25), label_text, font=font)
                draw.rectangle(text_bbox, fill=color)
                draw.text((x1, y1 - 25), label_text, fill="white", font=font)

            # 统计
            helmet_count = sum(1 for d in detections if d['class_id'] == 0)
            no_helmet_count = sum(1 for d in detections if d['class_id'] == 1)

            results.append({
                'path': str(img_path),
                'total': len(detections),
                'helmet': helmet_count,
                'no_helmet': no_helmet_count
            })

            # 显示
            axes[idx].imshow(result_image)
            axes[idx].set_title(
                f'{Path(img_path).name}\nHelmet:{helmet_count} NoHelmet:{no_helmet_count}',
                fontsize=9
            )
            axes[idx].axis('off')

            # 保存
            if save_dir:
                save_path = os.path.join(save_dir, f'test_{Path(img_path).name}')
                result_image.save(save_path)

        except Exception as e:
            print(f'处理 {img_path} 时出错: {e}')
            import traceback
            traceback.print_exc()
            axes[idx].text(0.5, 0.5, f'Error: {e}', ha='center')
            axes[idx].axis('off')

    # 隐藏多余的子图
    for idx in range(num_to_show, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    plt.show()

    # 打印统计
    print('\n=== 测试结果统计 ===')
    for r in results:
        print(f"{Path(r['path']).name}: "
              f"总检测数={r['total']}, "
              f"Helmet={r['helmet']}, "
              f"NoHelmet={r['no_helmet']}")

    return results


if __name__ == '__main__':
    print("多图测试功能模块已加载")
    print("使用方法：")
    print("1. 在notebook中导入: from test_multiple_images import test_multiple_images")
    print("2. 或者将test_multiple_images函数复制到notebook中")
