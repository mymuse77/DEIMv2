import sys
import torch
from PIL import Image
from test_helmet_accuracy import load_model, infer

model, eval_size, vit_backbone = load_model('configs/deimv2/deimv2_hgnetv2_b1_helmet_finetune.yml', r'outputs\deimv2_hgnetv2_b1_helmet_finetune\best_stg1.pth', 'cpu')
img = Image.open(r'D:\AI\Datasets\syperson2\41.png').convert('RGB')
labels, boxes, scores = infer(model, img, eval_size, vit_backbone, 'cpu')
with open('outputs/test_pred.txt', 'w') as f:
    f.write('LABELS:\n' + str(labels[:10]) + '\nSCORES:\n' + str(scores[:10]))
