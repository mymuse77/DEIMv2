# DeimV2 自动化研究架构 (AutoResearch Sandbox)

## 背景

借鉴 Karpathy 的 `autoresearch` 项目理念，为 `d:\AI\Git\deimv2` 项目设计一套"睡一觉自动涨 mAP"的自动化研究沙盒。该沙盒通过 AI Agent 自主修改训练配置和部分代码 → 运行短时训练 → 评估 mAP → 保留/回滚的死循环，实现无人值守的持续优化。

### 核心挑战与设计应对

| 挑战 | autoresearch (NLP) | 本方案 (CV 目标检测) |
|------|----|----|
| 训练时长 | 固定 5 分钟 | **固定 N 个 epoch**（用户可配置，建议 3~8 epoch） |
| 评估指标 | val_bpb | **COCO mAP@0.5:0.95** (已有完善评估管线) |
| AI 改的文件 | 单文件 [train.py](file:///d:/AI/Git/deimv2/train.py) | **单文件 YAML 配置 + 有限代码区域** |
| 数据集 | 在线文本 | **用户本地 Helmet/PPE 数据集** (已配好路径) |

---

## User Review Required

> [!IMPORTANT]
> **实验时间预算**：每轮实验的 epoch 数量直接决定了单次实验耗时和一夜能跑多少轮。建议从 **3~5 epoch** 起步测试，单轮约 15~30 分钟，一晚上可跑 16~30 轮。请确认您的 GPU 型号（看到配置中是 4090）和期望的单轮时间。

> [!IMPORTANT]
> **AI Agent 的修改范围**：本方案设计了两种模式供选择：
> 1. **纯配置调参模式 (安全模式)** — AI 只修改 YAML 配置文件中的超参数（学习率、Loss 权重、增强概率等），不碰 Python 代码。风险极低，适合夜间无人值守。
> 2. **代码+配置混合模式 (激进模式)** — AI 还可以修改 [deim_criterion.py](file:///d:/AI/Git/deimv2/engine/deim/deim_criterion.py)、[hybrid_encoder.py](file:///d:/AI/Git/deimv2/engine/deim/hybrid_encoder.py) 等核心模块。收益上限更高，但有代码崩溃风险。
>
> **建议先从模式 1 开始**，待积累经验后再尝试模式 2。

> [!WARNING]
> **DeimV2 原始代码不会被修改**。所有自动化脚本将放在新建的 `autoresearch/` 目录中，并在独立的 git 分支上操作。主分支代码完全安全。

---

## Proposed Changes

### 组件 1: Git 分支管理

在 deimv2 仓库中创建一个专用分支：
```bash
git checkout -b autoresearch/experiment
```

所有 AI 实验改动都在这个分支上进行，主分支不受任何影响。

---

### 组件 2: 自动化沙盒目录 `autoresearch/`

在项目根目录下新建 `autoresearch/` 目录，包含以下文件：

#### [NEW] [micro_config.yml](file:///d:/AI/Git/deimv2/autoresearch/micro_config.yml)

**微型训练配置文件**，继承 [deimv2_hgnetv2_x_helmet_4090.yml](file:///d:/AI/Git/deimv2/configs/deimv2/deimv2_hgnetv2_x_helmet_4090.yml)，但做以下关键修改：
- `epoches`: 降低至 5 (约 15~30 分钟完成)
- `flat_epoch` / `no_aug_epoch` / `stop_epoch`: 按比例缩减
- `checkpoint_freq`: 设为 1 (每 epoch 保存)
- 数据集路径: 复用现有 Helmet 数据集路径
- 输出目录: 指向 `autoresearch/outputs/`

该文件是 **AI Agent 唯一允许修改**的配置文件。AI 可以在其中调整：
- 优化器参数 (`lr`, `weight_decay`, `betas`)
- Loss 权重 (`loss_mal`, `loss_bbox`, `loss_giou`, `loss_fgl`, `loss_ddf`)
- Backbone 参数 (`freeze_at`, `freeze_stem_only`)
- Encoder 参数 (`hidden_dim`, `nhead`, `dim_feedforward`, `dropout`)
- Decoder 参数 (`num_queries`, `num_denoising`, `reg_max`, `reg_scale`)
- 数据增强概率 (`mosaic_prob`, `mixup_prob`, `copyblend_prob`)
- EMA 参数 (`decay`, `warmups`)
- Matcher 参数 (`cost_class`, `cost_bbox`, `cost_giou`)

---

#### [NEW] [run_experiment.py](file:///d:/AI/Git/deimv2/autoresearch/run_experiment.py)

**实验执行器**，封装单次训练+评估流程：
1. 读取 `micro_config.yml`
2. 执行 `python train.py -c autoresearch/micro_config.yml` (支持 `--tuning` 从 checkpoint 继续)
3. 捕获训练日志输出到 `autoresearch/run.log`
4. 训练结束后，从日志中解析出 COCO mAP 评估结果
5. 将结构化结果（mAP、显存、训练时间等）输出到 stdout 供上层脚本消费

核心设计：
```python
# 超时保护: 设置最大运行时间（如 45 分钟），超过则强制杀进程
# 结果解析: 使用正则匹配 COCO eval 输出中的 AP 指标
# 错误处理: 捕获 OOM / CUDA Error / Python 异常
```

---

#### [NEW] [experiment_loop.py](file:///d:/AI/Git/deimv2/autoresearch/experiment_loop.py)

**自动化实验主循环** (宿主脚本)，这是整个系统的大脑：

```
┌────────────────────────────────────────────────────────┐
│                  experiment_loop.py                     │
│                                                        │
│  LOOP FOREVER:                                         │
│  ┌──────────────────────────────────────────────────┐  │
│  │ 1. 读取当前 micro_config.yml + 历史实验记录      │  │
│  │ 2. 调用 LLM API，让 AI 提出下一组修改方案       │  │
│  │ 3. AI 输出修改后的 micro_config.yml 内容         │  │
│  │ 4. 写入文件 + git commit                         │  │
│  │ 5. 调用 run_experiment.py 执行训练               │  │
│  │ 6. 获取 mAP 结果                                │  │
│  │ 7. 如果 mAP > best_mAP: 保留 commit + 更新基线  │  │
│  │    否则: git reset --hard 回滚                    │  │
│  │ 8. 记录到 results.tsv                            │  │
│  └──────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────┘
```

关键特性：
- **可选 LLM 后端**：支持通过环境变量配置 OpenAI / Anthropic / 本地大模型 API
- **上下文窗口管理**：每次调用 LLM 时，附上最近 N 次实验的结果摘要，让 AI 能从历史中学习
- **异常恢复**：如果训练崩溃，自动回滚并继续下一轮
- **优雅停止**：捕获 Ctrl+C 信号，输出最终实验报告后退出

---

#### [NEW] [program_deimv2.md](file:///d:/AI/Git/deimv2/autoresearch/program_deimv2.md)

**AI Agent 的系统提示词**，告诉大模型：
- DeimV2 的架构概述（Backbone + Encoder + Decoder 三大组件）
- 哪些参数可以动，哪些绝对不能动
- 优化目标：最大化 COCO mAP@0.5:0.95
- 历史实验结果如何解读
- 鼓励怎样的探索策略（先调 LR/Loss 权重等低风险项，再尝试结构改动）

---

#### [NEW] [results.tsv](file:///d:/AI/Git/deimv2/autoresearch/results.tsv)

实验日志表格（TSV 格式），记录每轮实验：
```
commit	mAP	mAP50	mAP75	vram_gb	status	description
a1b2c3d	0.4500	0.6800	0.4900	22.1	keep	baseline
b2c3d4e	0.4620	0.6950	0.5050	22.3	keep	increase decoder lr to 0.0003
c3d4e5f	0.4480	0.6700	0.4800	22.0	discard	reduce loss_giou weight to 1
```

---

### 组件 3: 不修改的文件（裁判区）

以下文件 **严格禁止 AI 修改**，以保证评估公平性：
- [prepare.py](file:///d:/AI/Git/autoresearch/prepare.py) / [train.py](file:///d:/AI/Git/deimv2/train.py) — 训练入口
- `engine/solver/` — 训练循环和评估逻辑
- `engine/data/` — 数据加载和评估器
- [engine/deim/postprocessor.py](file:///d:/AI/Git/deimv2/engine/deim/postprocessor.py) — 后处理
- `configs/dataset/` — 数据集路径配置

---

## 文件变更汇总

| 操作 | 文件 | 说明 |
|------|------|------|
| NEW | `autoresearch/micro_config.yml` | 微型训练配置（AI 可修改区） |
| NEW | `autoresearch/run_experiment.py` | 单次实验执行器 |
| NEW | `autoresearch/experiment_loop.py` | 自动化主循环 |
| NEW | `autoresearch/program_deimv2.md` | Agent 提示词 |
| NEW | `autoresearch/results.tsv` | 实验日志 |

> **原有代码零修改**。所有新文件均在 `autoresearch/` 目录中。

---

## Verification Plan

### 自动化测试

1. **Baseline 微型训练验证**：
   ```bash
   cd d:\AI\Git\deimv2
   python train.py -c autoresearch/micro_config.yml --output-dir autoresearch/outputs/baseline
   ```
   - 验证训练能正常启动、完成 5 个 epoch、输出 COCO eval 结果
   - 验证 `run_experiment.py` 能正确解析日志中的 mAP

2. **实验循环 Dry Run**：
   ```bash
   python autoresearch/experiment_loop.py --dry-run --max-experiments 1
   ```
   - 验证 git commit → 训练 → 结果判定 → commit/revert 的完整流程

### 手动验证

3. **用户确认**：训练完成后请用户检查：
   - `autoresearch/results.tsv` 是否有正确的记录
   - `autoresearch/outputs/` 下是否有 checkpoint
   - `git log --oneline` 查看是否有正确的 commit 记录
