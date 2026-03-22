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
MODEL_PATH = Path(r"C:\Users\farben\Desktop\fsdownload\deimv2_hgnetv2_atto_helmet_gpu_3060\best_stg1.pth")

# 配置文件（helmet_detection2.yml 对应的模型配置）
CONFIG_PATH = DEIMV2_ROOT / "configs" / "deimv2" / "deimv2_hgnetv2_atto_helmet_cpu2.yml"

# 验证集图片目录 & COCO 标注文件（用于计算精度）
VAL_IMG_DIR  = Path(r"D:\AI\Dataset\20250731-ppe2286y\valid\images")
VAL_ANN_FILE = Path(r"D:\AI\Dataset\20250731-ppe2286y\valid\_annotations.coco.json")

# 训练集图片目录（随机抽样可视化用）
TRAIN_IMG_DIR = Path(r"D:\AI\Dataset\syperson")

# 结果输出目录
OUTPUT_DIR = DEIMV2_ROOT / "test_results"

# ──────────────────────────────────── 超参数 ──────────────────────────────────────
CONF_THRESHOLD  = 0.40   # 置信度阈值（可视化 & TP 判断）
IOU_THRESHOLD   = 0.50   # IOU 阈值（TP 计算用）
NUM_SAMPLE_IMGS = 18      # 从训练集随机抽取用于可视化的图片数量
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 类别名称 & 颜色
CLASSES = {0: "戴头盔", 1: "未戴头盔", 2: "未穿马甲", 3: "穿马甲"}
COLORS  = {0: "#00CC44", 1: "#FF3333", 2: "#FF8C00", 3: "#3399FF"}

# ─────────────────────────────────── 辅助工具 ─────────────────────────────────────

def iou(box_a, box_b):
    """计算两个 [x1,y1,x2,y2] 框的 IoU"""
    xa1 = max(box_a[0], box_b[0]); ya1 = max(box_a[1], box_b[1])
    xa2 = min(box_a[2], box_b[2]); ya2 = min(box_a[3], box_b[3])
    inter = max(0, xa2 - xa1) * max(0, ya2 - ya1)
    area_a = (box_a[2]-box_a[0]) * (box_a[3]-box_a[1])
    area_b = (box_b[2]-box_b[0]) * (box_b[3]-box_b[1])
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
    transforms = T.Compose([
        T.Resize(eval_size),
        T.ToTensor(),
        *(  [T.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])]
            if vit_backbone else []
        )
    ])
    return transforms(img_pil).unsqueeze(0)


def infer(model, img_pil, eval_size, vit_backbone, device):
    """对一张 PIL 图推理，返回 (labels, boxes, scores)"""
    w, h = img_pil.size
    orig_size = torch.tensor([[w, h]], dtype=torch.float32).to(device)
    tensor = preprocess(img_pil, eval_size, vit_backbone).to(device)
    with torch.no_grad():
        labels, boxes, scores = model(tensor, orig_size)
    return labels[0].cpu(), boxes[0].cpu(), scores[0].cpu()


def draw_result(img_pil, labels, boxes, scores, threshold=CONF_THRESHOLD):
    """在图上绘制检测框，返回新 PIL Image"""
    img = img_pil.copy()
    draw = ImageDraw.Draw(img)

    # 按优先级查找支持中文的 Windows 系统字体
    _CN_FONTS = [
        r"C:\Windows\Fonts\msyh.ttc",    # 微软雅黑
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

    for label, box, score in zip(labels, boxes, scores):
        if score < threshold:
            continue
        cls_id   = int(label.item())
        cls_name = CLASSES.get(cls_id, f"cls{cls_id}")
        color    = COLORS.get(cls_id, "#FFFFFF")
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

    all_imgs = list(TRAIN_IMG_DIR.glob("*.jpg")) + \
               list(TRAIN_IMG_DIR.glob("*.jpeg")) + \
               list(TRAIN_IMG_DIR.glob("*.png"))

    if not all_imgs:
        print(f"[WARN] 未在 {TRAIN_IMG_DIR} 找到任何图片")
        return

    samples = random.sample(all_imgs, min(NUM_SAMPLE_IMGS, len(all_imgs)))
    print(f"\n[INFO] 随机抽取 {len(samples)} 张训练集图片进行可视化推理…\n")

    total_dets = 0
    cls_count  = defaultdict(int)

    for i, img_path in enumerate(samples, 1):
        t0 = time.time()
        img_pil = Image.open(img_path).convert("RGB")
        labels, boxes, scores = infer(model, img_pil, eval_size, vit_backbone, DEVICE)
        elapsed = time.time() - t0

        # 过滤低置信度
        mask = scores >= CONF_THRESHOLD
        det_labels = labels[mask]
        det_boxes  = boxes[mask]
        det_scores = scores[mask]
        n_det = int(mask.sum())
        total_dets += n_det

        for lbl, scr in zip(det_labels, det_scores):
            cls_count[CLASSES.get(int(lbl.item()), f"cls{int(lbl)}")] += 1

        result_img = draw_result(img_pil, labels, boxes, scores)
        out_path = OUTPUT_DIR / f"result_{i:02d}_{img_path.name}"
        result_img.save(str(out_path))

        print(f"  [{i:02d}/{len(samples)}] {img_path.name}")
        print(f"         检测目标: {n_det} 个  |  耗时: {elapsed*1000:.1f}ms")
        for lbl, box, scr in zip(det_labels, det_boxes, det_scores):
            cls_name = CLASSES.get(int(lbl.item()), f"cls{int(lbl)}")
            print(f"           → {cls_name:10s}  置信度={scr:.3f}  "
                  f"框=[{box[0]:.0f},{box[1]:.0f},{box[2]:.0f},{box[3]:.0f}]")
        print(f"         结果保存: {out_path.name}\n")

    print("=" * 60)
    print(f"  可视化汇总  |  共检测到 {total_dets} 个目标")
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
    print(f"       IoU阈值={IOU_THRESHOLD}  置信度阈值={CONF_THRESHOLD}\n")

    with open(VAL_ANN_FILE, "r", encoding="utf-8") as f:
        coco = json.load(f)

    # 建立映射：image_id → filename
    id2file = {img["id"]: img["file_name"] for img in coco["images"]}

    # 建立 GT：image_id → list of {"category_id": int, "bbox": [x,y,w,h]}
    gt_map = defaultdict(list)
    for ann in coco["annotations"]:
        gt_map[ann["image_id"]].append({
            "category_id": ann["category_id"],
            "bbox": ann["bbox"]   # [x, y, w, h]
        })

    # COCO category_id → 0-indexed class id
    cat_ids = sorted({ann["category_id"] for ann in coco["annotations"]})
    cat2cls = {cid: i for i, cid in enumerate(cat_ids)}

    # 逐类统计
    tp_count  = defaultdict(int)
    fp_count  = defaultdict(int)
    fn_count  = defaultdict(int)
    total_imgs = len(coco["images"])

    for idx, img_info in enumerate(coco["images"], 1):
        img_id   = img_info["id"]
        filename = img_info["file_name"]
        img_path = VAL_IMG_DIR / Path(filename).name

        if not img_path.exists():
            continue

        if idx % 50 == 0 or idx == 1:
            print(f"  评估进度: {idx}/{total_imgs} …")

        img_pil = Image.open(img_path).convert("RGB")
        w, h = img_pil.size
        pred_labels, pred_boxes, pred_scores = infer(
            model, img_pil, eval_size, vit_backbone, DEVICE)

        # 过滤低置信度预测
        mask = pred_scores >= CONF_THRESHOLD
        pred_labels = pred_labels[mask]
        pred_boxes  = pred_boxes[mask]   # [x1,y1,x2,y2]
        pred_scores = pred_scores[mask]

        # 整理 GT
        gts = gt_map.get(img_id, [])
        gt_boxes_by_cls = defaultdict(list)
        for ann in gts:
            cls_id = cat2cls.get(ann["category_id"], -1)
            if cls_id < 0:
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
            gts_cls  = gt_boxes_by_cls[cls_id]
            preds_cls = sorted(pred_by_cls[cls_id], key=lambda x: -x[0])

            matched_gt = [False] * len(gts_cls)
            for scr, pbox in preds_cls:
                best_iou = 0.0
                best_j   = -1
                for j, gbox in enumerate(gts_cls):
                    if matched_gt[j]:
                        continue
                    v = iou(pbox, gbox)
                    if v > best_iou:
                        best_iou = v
                        best_j   = j
                if best_iou >= IOU_THRESHOLD and best_j >= 0:
                    tp_count[cls_id] += 1
                    matched_gt[best_j] = True
                else:
                    fp_count[cls_id] += 1

            fn_count[cls_id] += matched_gt.count(False)

    # 汇总打印
    print("\n" + "=" * 60)
    print(f"  验证集精度报告  (IoU@{IOU_THRESHOLD}, conf≥{CONF_THRESHOLD})")
    print("=" * 60)
    fmt = "{:12s}  {:6s}  {:6s}  {:6s}  {:5s}  {:5s}  {:5s}"
    print(fmt.format("类别", "TP", "FP", "FN", "Prec", "Rec", "F1"))
    print("-" * 60)

    all_ap = []
    for cls_id, cls_name in CLASSES.items():
        tp = tp_count.get(cls_id, 0)
        fp = fp_count.get(cls_id, 0)
        fn = fn_count.get(cls_id, 0)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2*prec*rec / (prec+rec) if (prec+rec) > 0 else 0.0
        all_ap.append(f1)   # 简易近似 AP = F1
        print(fmt.format(
            cls_name,
            str(tp), str(fp), str(fn),
            f"{prec:.3f}", f"{rec:.3f}", f"{f1:.3f}"
        ))

    map50 = sum(all_ap) / len(all_ap) if all_ap else 0.0
    print("-" * 60)
    print(f"  mAP@0.5 (approx. by mean-F1): {map50:.4f}")
    print("=" * 60)


# ────────────────────────────────────── 主程序 ───────────────────────────────────

def main():
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

    # 第一步：可视化（训练集随机抽样）
    run_visual_test(model, eval_size, vit_backbone)

    # 第二步：精度评估（验证集）
    run_accuracy_eval(model, eval_size, vit_backbone)

    print("\n[DONE] 全部测试完成！")


if __name__ == "__main__":
    main()
