# DeimV2 自动化研究架构 (AutoResearch for DeimV2)

## 目标
为 DeimV2 项目构建一个类似 Karpathy `autoresearch` 的自动化研究沙盒，让 AI Agent 能够在无人值守的情况下，通过自动化的"修改配置→训练→评估→保留/回滚"循环，持续优化模型的 mAP 和识别成功率。

---

## 任务清单

### Phase 1: 规划与设计
- [/] 深入研究 DeimV2 项目结构和训练流程
- [/] 撰写 `implementation_plan.md` 并等待用户审批
- [ ] 根据用户反馈修改计划

### Phase 2: 基础设施搭建
- [ ] 在 `deimv2` 仓库中创建 `autoresearch/` 分支
- [ ] 创建 `autoresearch/` 目录，存放所有沙盒脚本

### Phase 3: 核心脚本开发
- [ ] **`autoresearch/micro_config.yml`** — 微型训练配置 (缩减 epoch 数)
- [ ] **`autoresearch/run_experiment.py`** — 实验执行脚本 (启动训练 + 捕获结果)
- [ ] **`autoresearch/evaluate_result.py`** — 结果解析与评估 (从 log 提取 mAP)
- [ ] **`autoresearch/experiment_loop.py`** — 自动化实验主循环 (调用 LLM API → 修改→训练→评判→commit/revert)
- [ ] **`autoresearch/program_deimv2.md`** — Agent 指令提示词

### Phase 4: 验证
- [ ] 手动执行一次 baseline 微型训练，验证基础设施可用
- [ ] 运行一轮完整的自动化实验循环（包括 commit 和 revert）
