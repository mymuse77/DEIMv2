"""
DEIMv2 HGNetv2-Atto 安全帽检测模型 — 准确性验证 Demo
使用训练集图像随机抽样测试，可视化检测结果并输出统计汇总。

使用方式（在 DEIMv2 项目根目录下执行）：
    python test_helmet_accuracy.py

依赖：
    pip install pillow torchvision

脚本默认使用 CUDA（若可用），否则回退到 CPU。
"""

import os
import sys
import random
import json
import time
import argparse
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont

# ──────────────────────────────────── 路径配置 ────────────────────────────────────
# 项目根目录（本脚本放在 DEIMv2 根目录下）
DEIMV2_ROOT = Path(__file__).resolve().parent

# 模型权重（优先使用 best_stg1.pth）
# MODEL_PATH = Path(r"C:\Users\farben\Desktop\fsdownload\deimv2_hgnetv2_atto_helmet_gpu_3060\best_stg1.pth")
# # 配置文件（helmet_detection2.yml 对应的模型配置）
# CONFIG_PATH = DEIMV2_ROOT / "configs" / "deimv2" / "deimv2_hgnetv2_atto_helmet_cpu2.yml"

# # 模型权重（优先使用 best_stg1.pth）
MODEL_PATH = Path(
    r"D:\AI\Git\DEIMv2\outputs\deimv2_hgnetv2_b1_helmet_finetune\best_stg1.pth"
)
# 配置文件（helmet_detection2.yml 对应的模型配置）
CONFIG_PATH = (
    DEIMV2_ROOT / "configs" / "deimv2" / "deimv2_hgnetv2_b1_helmet_finetune.yml"
)

# 验证集图片目录 & COCO 标注文件（用于计算精度）
VAL_IMG_DIR = Path(r"D:\AI\Datasets\20250731-ppe2286y\valid\images")
VAL_ANN_FILE = Path(r"D:\AI\Datasets\20250731-ppe2286y\valid\_annotations.coco.json")

# 训练集图片目录（随机抽样可视化用）
TRAIN_IMG_DIR = Path(r"D:\AI\Datasets\syperson")

# 结果输出目录
OUTPUT_DIR = DEIMV2_ROOT / "test_results"

# ──────────────────────────────────── 超参数 ──────────────────────────────────────
IOU_THRESHOLD = 0.00  # IOU 阈值（TP 计算用）
NUM_SAMPLE_IMGS = 18  # 从训练集随机抽取用于可视化的图片数量
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ──────────────────────── 分类检测配置（按类别独立开关和阈值）─────────────────────────
#   enabled:   是否启用该类别的检测
#   threshold: 该类别的置信度阈值
#
# 头盔检测
HELMET_ENABLED = True
HELMET_THRESHOLD = 0.40
# 马甲检测
VEST_ENABLED = True
VEST_THRESHOLD = 0.40

# 类别名称 / 颜色 / 检测配置（自动从上面的开关和阈值生成）
CLASS_CONFIG = {
    0: {
        "name": "戴头盔",
        "color": "#00CC44",
        "enabled": HELMET_ENABLED,
        "threshold": HELMET_THRESHOLD,
    },
    1: {
        "name": "未戴头盔",
        "color": "#FF3333",
        "enabled": HELMET_ENABLED,
        "threshold": HELMET_THRESHOLD,
    },
    2: {
        "name": "未穿马甲",
        "color": "#FF8C00",
        "enabled": VEST_ENABLED,
        "threshold": VEST_THRESHOLD,
    },
    3: {
        "name": "穿马甲",
        "color": "#3399FF",
        "enabled": VEST_ENABLED,
        "threshold": VEST_THRESHOLD,
    },
}

# 向后兼容：保留 CLASSES / COLORS 方便其他地方引用
CLASSES = {k: v["name"] for k, v in CLASS_CONFIG.items()}
COLORS = {k: v["color"] for k, v in CLASS_CONFIG.items()}

# ─────────────────────────────────── 辅助工具 ─────────────────────────────────────


def is_class_enabled(cls_id):
    """判断某个类别是否启用检测"""
    cfg = CLASS_CONFIG.get(int(cls_id))
    return cfg is not None and cfg["enabled"]


def get_class_threshold(cls_id):
    """获取某个类别的置信度阈值"""
    cfg = CLASS_CONFIG.get(int(cls_id))
    return cfg["threshold"] if cfg else 0.5


def filter_by_class_config(labels, boxes, scores):
    """按各类别的 enabled 和 threshold 过滤检测结果"""
    keep = []
    for i in range(len(labels)):
        cls_id = int(labels[i].item())
        if is_class_enabled(cls_id) and float(scores[i]) >= get_class_threshold(cls_id):
            keep.append(i)
    if not keep:
        return labels[:0], boxes[:0], scores[:0]
    import torch as _t

    idx = _t.tensor(keep, dtype=_t.long)
    return labels[idx], boxes[idx], scores[idx]


def iou(box_a, box_b):
    """计算两个 [x1,y1,x2,y2] 框的 IoU"""
    xa1 = max(box_a[0], box_b[0])
    ya1 = max(box_a[1], box_b[1])
    xa2 = min(box_a[2], box_b[2])
    ya2 = min(box_a[3], box_b[3])
    inter = max(0, xa2 - xa1) * max(0, ya2 - ya1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def load_model(config_path, model_path, device):
    """加载 DEIMv2 模型（参考 torch_inf.py）"""
    sys.path.insert(0, str(DEIMV2_ROOT))
    from engine.core import YAMLConfig

    print(f"[INFO] 加载配置: {config_path}")
    cfg = YAMLConfig(str(config_path))

    # 禁用预训练权重下载
    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

    print(f"[INFO] 加载模型权重: {model_path}")
    checkpoint = torch.load(str(model_path), map_location="cpu", weights_only=False)

    if "ema" in checkpoint:
        state = checkpoint["ema"]["module"]
    elif "model" in checkpoint:
        state = checkpoint["model"]
    else:
        state = checkpoint

    # 去除 'module.' 前缀（多卡训练时可能存在）
    state = {k.replace("module.", "", 1): v for k, v in state.items()}
    cfg.model.load_state_dict(state, strict=False)

    class DeployModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()

        def forward(self, images, orig_sizes):
            out = self.model(images)
            return self.postprocessor(out, orig_sizes)

    model = DeployModel().to(device)
    model.eval()

    eval_size = cfg.yaml_cfg.get("eval_spatial_size", [640, 640])
    vit_backbone = bool(cfg.yaml_cfg.get("DINOv3STAs", False))
    print(f"[INFO] 评估尺寸: {eval_size}, 使用设备: {device.upper()}")
    return model, eval_size, vit_backbone


def preprocess(img_pil, eval_size, vit_backbone):
    """图像预处理"""
    transforms = T.Compose(
        [
            T.Resize(eval_size),
            T.ToTensor(),
            *(
                [T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
                if vit_backbone
                else []
            ),
        ]
    )
    return transforms(img_pil).unsqueeze(0)


def infer(model, img_pil, eval_size, vit_backbone, device):
    """对一张 PIL 图推理，返回 (labels, boxes, scores)"""
    w, h = img_pil.size
    orig_size = torch.tensor([[w, h]], dtype=torch.float32).to(device)
    tensor = preprocess(img_pil, eval_size, vit_backbone).to(device)
    with torch.no_grad():
        labels, boxes, scores = model(tensor, orig_size)
    return labels[0].cpu(), boxes[0].cpu(), scores[0].cpu()


def draw_result(img_pil, labels, boxes, scores):
    """在图上绘制检测框，返回新 PIL Image"""
    img = img_pil.copy()
    draw = ImageDraw.Draw(img)

    # 按优先级查找支持中文的 Windows 系统字体
    _CN_FONTS = [
        r"C:\Windows\Fonts\msyh.ttc",  # 微软雅黑
        r"C:\Windows\Fonts\simhei.ttf",  # 黑体
        r"C:\Windows\Fonts\simsun.ttc",  # 宋体
    ]
    font = None
    for _fp in _CN_FONTS:
        if os.path.exists(_fp):
            try:
                font = ImageFont.truetype(_fp, 18)
                break
            except Exception:
                pass
    if font is None:
        font = ImageFont.load_default()

    # 先按分类配置过滤
    labels, boxes, scores = filter_by_class_config(labels, boxes, scores)

    for label, box, score in zip(labels, boxes, scores):
        cls_id = int(label.item())
        cls_name = CLASSES.get(cls_id, f"cls{cls_id}")
        color = COLORS.get(cls_id, "#FFFFFF")
        x1, y1, x2, y2 = [float(v) for v in box]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        text = f"{cls_name} {score:.2f}"
        # 绘制文字背景
        try:
            tw = draw.textlength(text, font=font)
            th = 18
        except Exception:
            tw, th = 80, 16
        draw.rectangle([x1, y1 - th - 2, x1 + tw + 4, y1], fill=color)
        draw.text((x1 + 2, y1 - th - 1), text, fill="white", font=font)

    return img


# ──────────────────────────────── 可视化测试（训练集随机抽样）────────────────────────────


def run_visual_test(model, eval_size, vit_backbone):
    """从训练集随机抽取图片，推理后保存可视化结果"""
    OUTPUT_DIR.mkdir(exist_ok=True)

    all_imgs = (
        list(TRAIN_IMG_DIR.glob("*.jpg"))
        + list(TRAIN_IMG_DIR.glob("*.jpeg"))
        + list(TRAIN_IMG_DIR.glob("*.png"))
    )

    if not all_imgs:
        print(f"[WARN] 未在 {TRAIN_IMG_DIR} 找到任何图片")
        return

    samples = random.sample(all_imgs, min(NUM_SAMPLE_IMGS, len(all_imgs)))
    print(f"\n[INFO] 随机抽取 {len(samples)} 张训练集图片进行可视化推理…\n")

    total_dets = 0
    cls_count = defaultdict(int)

    for i, img_path in enumerate(samples, 1):
        t0 = time.time()
        img_pil = Image.open(img_path).convert("RGB")
        labels, boxes, scores = infer(model, img_pil, eval_size, vit_backbone, DEVICE)
        elapsed = time.time() - t0

        # 按分类配置过滤
        det_labels, det_boxes, det_scores = filter_by_class_config(
            labels, boxes, scores
        )
        n_det = len(det_labels)
        total_dets += n_det

        for lbl, scr in zip(det_labels, det_scores):
            cls_count[CLASSES.get(int(lbl.item()), f"cls{int(lbl)}")] += 1

        result_img = draw_result(img_pil, labels, boxes, scores)
        out_path = OUTPUT_DIR / f"result_{i:02d}_{img_path.name}"
        result_img.save(str(out_path))

        print(f"  [{i:02d}/{len(samples)}] {img_path.name}")
        print(f"         检测目标: {n_det} 个  |  耗时: {elapsed * 1000:.1f}ms")
        for lbl, box, scr in zip(det_labels, det_boxes, det_scores):
            cls_name = CLASSES.get(int(lbl.item()), f"cls{int(lbl)}")
            print(
                f"           → {cls_name:10s}  置信度={scr:.3f}  "
                f"框=[{box[0]:.0f},{box[1]:.0f},{box[2]:.0f},{box[3]:.0f}]"
            )
        print(f"         结果保存: {out_path.name}\n")

    print("=" * 60)
    print(f"  可视化汇总  |  共检测到 {total_dets} 个目标")
    for cls_name, cnt in sorted(cls_count.items()):
        print(f"    {cls_name:12s}: {cnt} 个")
    print(f"  结果图片已保存至: {OUTPUT_DIR}")
    print("=" * 60)


def run_custom_images_test(model, eval_size, vit_backbone, image_paths):
    """对用户指定的图片列表进行推理并可视化"""
    OUTPUT_DIR.mkdir(exist_ok=True)
    print(f"\n[INFO] 开始对指定的 {len(image_paths)} 张图片进行推理…\n")

    total_dets = 0
    cls_count = defaultdict(int)

    # 过滤掉不存在的路径
    valid_paths = []
    for p in image_paths:
        path = Path(p.strip())
        if path.exists():
            valid_paths.append(path)
        else:
            print(f"[WARN] 图片不存在，跳过: {path}")

    if not valid_paths:
        print("[ERROR] 没有有效的图片路径可供测试")
        return

    for i, img_path in enumerate(valid_paths, 1):
        t0 = time.time()
        try:
            img_pil = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"[ERROR] 无法打开图片 {img_path}: {e}")
            continue

        labels, boxes, scores = infer(model, img_pil, eval_size, vit_backbone, DEVICE)
        elapsed = time.time() - t0

        # 按分类配置过滤
        det_labels, det_boxes, det_scores = filter_by_class_config(
            labels, boxes, scores
        )
        n_det = len(det_labels)
        total_dets += n_det

        for lbl, scr in zip(det_labels, det_scores):
            cls_count[CLASSES.get(int(lbl.item()), f"cls{int(lbl)}")] += 1

        result_img = draw_result(img_pil, labels, boxes, scores)
        out_path = OUTPUT_DIR / f"custom_{i:02d}_{img_path.name}"
        result_img.save(str(out_path))

        print(f"  [{i:02d}/{len(valid_paths)}] {img_path.name}")
        print(f"         检测目标: {n_det} 个  |  耗时: {elapsed * 1000:.1f}ms")
        for lbl, box, scr in zip(det_labels, det_boxes, det_scores):
            cls_name = CLASSES.get(int(lbl.item()), f"cls{int(lbl)}")
            print(
                f"           → {cls_name:10s}  置信度={scr:.3f}  "
                f"框=[{box[0]:.0f},{box[1]:.0f},{box[2]:.0f},{box[3]:.0f}]"
            )
        print(f"         结果保存: {out_path.name}\n")

    print("=" * 60)
    print(f"  自定义图片测试汇总  |  共检测到 {total_dets} 个目标")
    for cls_name, cnt in sorted(cls_count.items()):
        print(f"    {cls_name:12s}: {cnt} 个")
    print(f"  结果图片已保存至: {OUTPUT_DIR}")
    print("=" * 60)


# ──────────────────────────────── 精度验证（验证集 COCO） ────────────────────────────


def run_accuracy_eval(model, eval_size, vit_backbone):
    """在验证集上逐图推理，计算每类 Precision / Recall / F1 及 mAP@0.5"""
    if not VAL_ANN_FILE.exists():
        print(f"\n[WARN] 未找到验证集标注 {VAL_ANN_FILE}，跳过精度评估")
        return

    print(f"\n[INFO] 开始在验证集上评估精度…")
    print(f"       标注文件: {VAL_ANN_FILE}")
    print(f"       图片目录: {VAL_IMG_DIR}")
    enabled_classes = [
        f"{v['name']}(≥{v['threshold']})" for v in CLASS_CONFIG.values() if v["enabled"]
    ]
    disabled_classes = [v["name"] for v in CLASS_CONFIG.values() if not v["enabled"]]
    print(f"       IoU阈值={IOU_THRESHOLD}")
    print(f"       启用检测: {', '.join(enabled_classes)}")
    if disabled_classes:
        print(f"       已禁用:   {', '.join(disabled_classes)}")
    print()

    with open(VAL_ANN_FILE, "r", encoding="utf-8") as f:
        coco = json.load(f)

    # 建立映射：image_id → filename
    id2file = {img["id"]: img["file_name"] for img in coco["images"]}

    # 建立 GT：image_id → list of {"category_id": int, "bbox": [x,y,w,h]}
    gt_map = defaultdict(list)
    for ann in coco["annotations"]:
        gt_map[ann["image_id"]].append(
            {
                "category_id": ann["category_id"],
                "bbox": ann["bbox"],  # [x, y, w, h]
            }
        )

    # COCO category_id → 0-indexed class id
    cat_ids = sorted({ann["category_id"] for ann in coco["annotations"]})
    cat2cls = {cid: i for i, cid in enumerate(cat_ids)}

    # 逐类统计
    tp_count = defaultdict(int)
    fp_count = defaultdict(int)
    fn_count = defaultdict(int)
    total_imgs = len(coco["images"])

    for idx, img_info in enumerate(coco["images"], 1):
        img_id = img_info["id"]
        filename = img_info["file_name"]
        img_path = VAL_IMG_DIR / Path(filename).name

        if not img_path.exists():
            continue

        if idx % 50 == 0 or idx == 1:
            print(f"  评估进度: {idx}/{total_imgs} …")

        img_pil = Image.open(img_path).convert("RGB")
        w, h = img_pil.size
        pred_labels, pred_boxes, pred_scores = infer(
            model, img_pil, eval_size, vit_backbone, DEVICE
        )

        # 按分类配置过滤
        pred_labels, pred_boxes, pred_scores = filter_by_class_config(
            pred_labels, pred_boxes, pred_scores
        )

        # 整理 GT
        gts = gt_map.get(img_id, [])
        gt_boxes_by_cls = defaultdict(list)
        for ann in gts:
            cls_id = cat2cls.get(ann["category_id"], -1)
            if cls_id < 0 or not is_class_enabled(cls_id):
                continue
            bx, by, bw, bh = ann["bbox"]
            gt_boxes_by_cls[cls_id].append([bx, by, bx + bw, by + bh])

        # 整理预测
        pred_by_cls = defaultdict(list)
        for lbl, box, scr in zip(pred_labels, pred_boxes, pred_scores):
            pred_by_cls[int(lbl.item())].append((float(scr), [float(v) for v in box]))

        # 对每个类别匹配 TP/FP/FN
        all_cls = set(gt_boxes_by_cls.keys()) | set(pred_by_cls.keys())
        for cls_id in all_cls:
            gts_cls = gt_boxes_by_cls[cls_id]
            preds_cls = sorted(pred_by_cls[cls_id], key=lambda x: -x[0])

            matched_gt = [False] * len(gts_cls)
            for scr, pbox in preds_cls:
                best_iou = 0.0
                best_j = -1
                for j, gbox in enumerate(gts_cls):
                    if matched_gt[j]:
                        continue
                    v = iou(pbox, gbox)
                    if v > best_iou:
                        best_iou = v
                        best_j = j
                if best_iou >= IOU_THRESHOLD and best_j >= 0:
                    tp_count[cls_id] += 1
                    matched_gt[best_j] = True
                else:
                    fp_count[cls_id] += 1

            fn_count[cls_id] += matched_gt.count(False)

    # 汇总打印
    print("\n" + "=" * 60)
    print(f"  验证集精度报告  (IoU@{IOU_THRESHOLD})")
    print("=" * 60)
    fmt = "{:12s}  {:6s}  {:6s}  {:6s}  {:5s}  {:5s}  {:5s}  {:5s}"
    print(fmt.format("类别", "TP", "FP", "FN", "Prec", "Rec", "F1", "阈值"))
    print("-" * 60)

    all_ap = []
    for cls_id, cfg in CLASS_CONFIG.items():
        if not cfg["enabled"]:
            continue
        cls_name = cfg["name"]
        tp = tp_count.get(cls_id, 0)
        fp = fp_count.get(cls_id, 0)
        fn = fn_count.get(cls_id, 0)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        all_ap.append(f1)
        print(
            fmt.format(
                cls_name,
                str(tp),
                str(fp),
                str(fn),
                f"{prec:.3f}",
                f"{rec:.3f}",
                f"{f1:.3f}",
                f"{cfg['threshold']}",
            )
        )

    map50 = sum(all_ap) / len(all_ap) if all_ap else 0.0
    print("-" * 60)
    print(f"  mAP@0.5 (approx. by mean-F1): {map50:.4f}")
    print("=" * 60)


# ────────────────────────────────────── 主程序 ───────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="DEIMv2 安全帽检测模型测试脚本")
    parser.add_argument(
        "--images", type=str, help="指定要测试的图片路径，多个路径用逗号分割"
    )
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  DEIMv2 HGNetv2-Atto 安全帽/马甲 检测模型 — 准确性验证")
    print("=" * 60)
    print(f"  模型: {MODEL_PATH}")
    print(f"  配置: {CONFIG_PATH}")
    print(f"  设备: {DEVICE.upper()}")
    print()

    if not MODEL_PATH.exists():
        print(f"[ERROR] 模型文件不存在: {MODEL_PATH}")
        sys.exit(1)
    if not CONFIG_PATH.exists():
        print(f"[ERROR] 配置文件不存在: {CONFIG_PATH}")
        sys.exit(1)

    model, eval_size, vit_backbone = load_model(CONFIG_PATH, MODEL_PATH, DEVICE)

    if args.images:
        # 如果指定了图片路径，则只对指定图片进行推理
        image_paths = args.images.split(",")
        run_custom_images_test(model, eval_size, vit_backbone, image_paths)
    else:
        # 否则运行默认的流程：随机抽插可视化 + 精度评估
        # 第一步：可视化（训练集随机抽样）
        run_visual_test(model, eval_size, vit_backbone)

        # 第二步：精度评估（验证集）
        run_accuracy_eval(model, eval_size, vit_backbone)

    print("\n[DONE] 全部测试完成！")


if __name__ == "__main__":
    main()
