# KataRank Training Log

持续更新的训练日志，用于学术分析和实验追踪。

---

## 1. 项目背景

KataRank 利用自研 KataGo 分支（`batch_analysis` 子命令）对围棋 SGF 棋谱进行特征提取，生成 KAB2 二进制格式的逐手特征帧，然后通过 PyTorch 模型预测棋手段位（20k–9d，29 类序数分类）和连续强度评分。

### 1.1 模型架构

**DualViewSetTransformer** — 双视角集合 Transformer：
- 按颜色（黑/白）分流，各自通过 ISAB 集合编码器
- 因果交叉注意力（每个棋手只能看到对手已下的棋步）
- 分段注意力池化（开局/中盘/收官分段加权）
- 融合后接 rating head (MSE) + ordinal rank head (29 类)

**KataRankModel** (v3):
- `input_dim`: 1034 (full mode, 10 scalar + 2×512 trunk) 或 10 (lite mode, scalar only)
- `hidden_dim`: 128, `num_heads`: 4, `num_inducing`: 16
- `encoder_depth`: 2, `cross_depth`: 1
- 参数量: ~2.26M (full) / ~540K (lite, hidden_dim=64)

### 1.2 数据管道

KAB2 二进制格式包含 96 字节 header + float32 逐手特征：
- **PlayerSummary[2]**: meanLogPrior — 主训练目标
- **PlayerSummary[10]**: humanRankIdx (0–28, -1 未计算) — 序数分类标签
- **PlayerSummary[11]**: humanLogPrior — HumanSL 置信度权重

数据来源: GoPredict 数据库中的围棋棋谱，通过自研 KataGo 分支分析后生成 KAB2 特征文件，缓存于 `~/.katarank/kab2_cache/`。

### 1.3 损失函数

**KataRankLoss** (v3):
- `RatingMSELoss`: 黑/白 rating 预测 vs 归一化 meanLogPrior
- `BradleyTerry`: 相对强弱排序一致性
- `RankAnchorLoss`: HumanSL 段位校准（置信度加权 NLL，humanRankIdx=-1 时自动归零）

---

## 2. 训练实验记录

### 2.1 Baseline 训练 (2026-06-16, v3.0)

**配置:**
- 数据: 5,134 训练 / 570 验证，input_dim=1034
- 超参: batch_size=16, lr=0.001, dropout=0.1, patience=20, epochs=100
- 设备: CPU (PyTorch CPU-only 版本)

**结果:**
| 指标 | 值 |
|------|-----|
| best_val_loss | 2.7322 |
| best_epoch | 3 (early stop at 23) |
| rank_mae | 2.76 段 |
| rank_acc | 11.5% |
| rank_acc_pm1 | 33.2% |
| rating_corr | 0.449 |
| 训练时间 | 5,049 秒 (~84 分钟) |

**分析:**
- Epoch 3 即过拟合，patience 20 导致 epoch 23 early stop
- 原因: 模型容量 (2.26M 参数) vs 数据量 (5134 局) 严重不匹配
- batch_size=16 梯度不稳定，段位分布不均衡（4k-5k 集中，两端稀疏）

---

### 2.2 策略改进训练 (2026-06-16 ~ 06-17, Phase 1)

**改进措施:**

| 改进项 | 旧值 | 新值 | 理由 |
|--------|------|------|------|
| StratifiedRankSampler | 无 | 5 band 分层采样 | 每 batch 覆盖全段位 |
| dropout | 0.1 | 0.2 | 减少过拟合 |
| weight_decay | 1e-5 | 1e-4 | 更强正则化 |
| learning_rate | 0.001 | 0.0005 | 更温和的起步 |
| warmup_epochs | 0 | 5 | 线性 warmup 防止初期震荡 |
| batch_size | 16 | 32 | 更稳定的梯度估计 |
| patience | 20 | 30 | 给模型更多学习时间 |
| epochs | 100 | 150 | 匹配更大的 patience |
| 设备 | CPU | CUDA (RTX 3060) | 安装 PyTorch cu126 |

**StratifiedRankSampler 设计:**
将 29 个段位分为 5 个 band（每 band ~6 段位），每 batch 从各 band 均匀抽样。避免 batch 被峰值段位（4k-5k）主导，确保 ordinal head 的 threshold 学习覆盖全段位。

**LR Schedule:**
```
Epoch 1-5:   Linear warmup  0.05×lr → 1.0×lr
Epoch 6-150: Cosine annealing  lr → lr_min (1e-5)
```

**训练过程 (精选 epoch):**

| Epoch | Train Loss | Val Loss | LR | 备注 |
|-------|-----------|----------|-----|------|
| 1 | 10.6249 | 2.4061 | 5.00e-05 | warmup 开始 |
| 3 | 3.0409 | 2.0943 | 2.30e-04 | 已超越旧 best |
| 5 | 2.7528 | 2.3868 | 4.10e-04 | warmup 震荡 |
| 8 | 2.5032 | 1.9027 | 5.00e-04 | warmup 结束，首次 <2.0 |
| 14 | 2.1534 | 1.7261 | 4.96e-04 | cosine decay 发力 |
| 20 | 1.9422 | 1.5650 | 4.89e-04 | |
| 33 | 1.6766 | 1.3436 | 4.59e-04 | 突破 1.35 |
| 41 | 1.5867 | 1.2790 | 4.33e-04 | 首次 <1.3 |
| 50 | 1.4934 | 1.2312 | 3.97e-04 | 半程 |
| 60 | 1.4317 | 1.1658 | 3.51e-04 | |
| 73 | 1.3266 | 1.1123 | 2.84e-04 | 首次 <1.12 |
| 81 | 1.3032 | 1.0970 | 2.42e-04 | 首次 <1.10 |
| 88 | 1.2733 | 1.0758 | 2.05e-04 | |
| 96 | 1.2507 | 1.0696 | 1.64e-04 | |
| 104 | 1.2334 | 1.0521 | 1.26e-04 | |
| 121 | 1.2117 | 1.0384 | 6.00e-05 | |
| 133 | 1.2054 | 1.0346 | 2.84e-05 | |
| 139 | 1.2038 | **1.0273** | 1.82e-05 | **best** |
| 150 | 1.1913 | 1.0307 | 1.01e-05 | 训练结束 |

**最终结果对比:**

| 指标 | Baseline | Phase 1 | 改善 |
|------|----------|---------|------|
| val_loss | 2.732 | **1.027** | **-62.4%** |
| rank_mae | 2.76 | **0.542** | **-80.4%** |
| rank_acc | 11.5% | **54.4%** | **+4.7x** |
| rank_acc_pm1 | 33.2% | **92.8%** | **+2.8x** |
| rating_corr | 0.449 | **0.993** | **近乎完美** |
| best_epoch | 3 | 139 | 充分学习 |
| 训练时间 | 5,049s | ~14,700s | GPU 加速但 epoch 更多 |

**关键洞察:**
1. **StratifiedRankSampler 是最大的赢因** — 没有它，batch 被 4k-5k 棋局主导，模型从未学会全段位谱
2. **更高 dropout + 更慢 LR** — 从 epoch 3 过拟合延长到 epoch 139 仍在学习
3. **Warmup** — 避免了训练初期的剧烈震荡，使 loss 曲线更平滑

---

### 2.3 全量 Teacher 训练 (2026-06-22, Phase 2)

**改进措施:**
- 数据量从 9,394 → 27,223 游戏 (3x 扩大)，全部含 HumanSL 段位标签
- Warm-start from Phase 1 best.pt
- 保持 Phase 1 超参不变（验证其可复现性）

**配置:**
- 数据: 27,223 训练 / 3,022 验证，input_dim=1034
- 超参: batch_size=32, lr=0.0005, warmup=5, cosine decay → 1e-5, patience=30, epochs=150
- stratified=true, n_bands=5, num_workers=2
- 设备: CUDA (RTX 3060)

**训练过程 (精选 epoch):**

| Epoch | Train Loss | Val Loss | LR | 备注 |
|-------|-----------|----------|-----|------|
| 1 | 1.0375 | 0.8549 | 5.00e-05 | 起点 ≈ Phase 1 终点 |
| 5 | 1.0622 | 0.9067 | 4.10e-04 | warmup 峰值抖动 |
| 13 | 1.0384 | 0.8539 | 4.97e-04 | 首次突破 Phase 1 best |
| 32 | 0.9984 | 0.8334 | 4.62e-04 | train_loss 破 1.0 |
| 45 | 0.9776 | 0.8170 | 4.18e-04 | decay 甜区开始 |
| 66 | 0.9468 | 0.8107 | 3.21e-04 | |
| 76 | 0.9433 | **0.7998** | 2.68e-04 | **首破 0.80** |
| 100 | 0.9163 | 0.7874 | 1.45e-04 | |
| 113 | 0.9076 | 0.7797 | 8.85e-05 | |
| 128 | 0.8982 | **0.7727** | 3.98e-05 | **best** |
| 150 | 0.8906 | 0.7781 | 1.01e-05 | 训练结束 |

**最终结果对比:**

| 指标 | Phase 1 (9.4k) | **Phase 2 (27.2k)** | 改善 |
|------|---------------|---------------------|------|
| val_loss | 0.8585 | **0.7727** | **-10.0%** |
| rank_mae | 0.381 | **0.350** | **-8.1%** |
| rank_acc | 66.6% | **69.0%** | **+2.4pp** |
| rank_acc_pm1 | 96.1% | **96.4%** | **+0.3pp** |
| rating_corr | 0.9925 | **0.9949** | **+0.24pp** |
| best_epoch | 140 | 128 | 更早收敛 |
| 训练时间 | 25,184s (7.0h) | 47,576s (13.2h) | 数据量 3x |
| 评估样本量 | 1,140 | 6,044 | +430% |

**关键洞察:**
1. **3x 数据量的红利充分兑现** — val_loss 降 10%，所有指标改善
2. **无过拟合** — train-val gap 从 0.183 收窄到 0.118
3. **150 epoch 跑满无 early stop** — 数据量充足，模型从未停止学习
4. **rank_acc_pm1 96.4%** — 96% 以上棋谱段位判定在 ±1 段以内
5. **rating_corr 0.9949** — 与 KataGo meanLogPrior 几乎完美相关

---

## 3. V2 Lite Model — 知识蒸馏

### 3.1 动机

Full 模型 (input_dim=1034) 依赖 KataGo trunk vectors，推理时需要跑完整的 KataGo 神经网络。V2 lite 模型 (input_dim=10) 仅使用标量特征（winrate, score, policy 等），不需要 trunk vectors，推理速度可提升 5-10 倍。

**方法: 知识蒸馏 (Knowledge Distillation)**
- Teacher: Phase 2 full 模型 (1034-dim, 2.26M 参数, val_loss=0.7727)
- Student: lite 模型 (10-dim, 540K 参数, hidden_dim=64)
- Loss: KL-divergence(student_rank_probs, teacher_rank_probs) + MSE(student_rating, teacher_rating) + hard label anchor

Student 学习 teacher 的**软概率分布**，而非硬标签。软分布携带段位之间的相对关系信息（例如 "这个棋手大概率是 5k，小概率是 4k 或 6k"）。

### 3.2 DistillationLoss 设计

```
L_total = w_kl × KL(teacher_soft ‖ student_soft)      [rank 分布蒸馏]
        + w_rating × MSE(student_rating, teacher_rating)  [连续强度蒸馏]
        + w_hard × RankAnchorLoss(student, hard_labels)    [段位校准锚点]
```

Temperature T=2.0 控制 teacher 分布的软度。T>1 使分布更平滑，给 student 更丰富的梯度信号。KL loss 乘以 T² 使梯度量级与 T 无关。

### 3.3 渐进式微量验证实验 (2026-06-17)

目的: 用极小样本验证 distillation pipeline 端到端可跑通，观察数据量与效果的关系。

**配置:**
- Teacher: Phase 1 best.pt (val_loss=1.027, 9.4k 游戏)
- Student: input_dim=10, hidden_dim=64, 540K 参数
- Temperature: 2.0
- 每步 warm-start from 上一步 checkpoint
- 设备: CUDA (RTX 3060)

**结果:**

| 训练局数 | val_loss | rank_mae | rank_acc_pm1 | rating_corr | 训练时间 |
|---------|----------|----------|-------------|-------------|---------|
| 10 | 8.649 | 14.1 | 0.0% | 0.061 | 20s |
| 20 | 8.268 | 12.5 | 0.0% | 0.458 | 25s |
| 50 | 7.250 | 11.5 | 0.0% | 0.894 | 160s |
| 100 | 6.695 | 11.2 | 0.0% | 0.960 | 388s |
| 200 | 5.962 | 11.5 | 0.0% | 0.940 | 738s |

**结论:** Pipeline 跑通。Rating 预测学得快（50 局时 rating_corr=0.89），但 rank 分类需要更多数据。

### 3.4 全量蒸馏 (2026-06-22 ~ 06-23, Phase 2 Lite)

**配置:**
- Teacher: Phase 2 best.pt (val_loss=0.7727, 27.2k 游戏训练)
- Student: input_dim=10, hidden_dim=64, 540K 参数 (4.2x 压缩)
- 数据: 27,223 训练 / 3,022 验证
- Temperature: 2.0, w_kl=1.0, w_rating=1.0, w_hard=0.1
- 超参: batch_size=32, lr=0.0005, warmup=5, cosine decay, patience=30, epochs=100
- Teacher target 生成: 248.6s (一次性，train+val)
- 设备: CUDA (RTX 3060)

**训练过程 (精选 epoch):**

| Epoch | Train Loss | Val Loss | 备注 |
|-------|-----------|----------|------|
| 1 | 19.466 | 11.286 | 起步 |
| 10 | 4.743 | 2.285 | 快速下降 |
| 25 | 2.646 | 0.858 | 首破 1.0 |
| 42 | 2.217 | 0.738 | |
| 56 | 2.078 | 0.643 | |
| 74 | 1.925 | 0.599 | |
| 82 | 1.948 | 0.586 | |
| **96** | **1.917** | **0.577** | **best** |
| 100 | 1.914 | 0.591 | 训练结束 |

**最终结果:**

| 指标 | Teacher (Full 1034-dim) | **Lite (Distilled 10-dim)** | 保留率 |
|------|------------------------|----------------------------|--------|
| rank_acc | 69.0% | **61.9%** | **90%** |
| rank_acc_pm1 | 96.4% | **94.8%** | **98%** |
| rank_mae | 0.350 | **0.438** | — |
| rating_corr | 0.9949 | **0.9918** | **99.7%** |
| 参数量 | 2,263,165 | 539,741 | 4.2x 压缩 |
| 训练时间 | 47,576s (13.2h) | 47,595s (13.2h) | |
| 评估样本量 | 6,044 | 6,044 | |

**关键洞察:**
1. **远超预期** — 预期 rank_acc_pm1 85-92%，实际 94.8%
2. **rating_corr 0.9918** — 仅 10 个标量特征就达到近乎完美的棋力排序
3. **100 epoch 跑满无 early stop** — best epoch 96，student 到最后仍在学习
4. **103x 信息压缩** (1034→10 dim) 只损失了 2% 的 rank_acc_pm1
5. 蒸馏证明了 KataGo trunk vectors 中的段位信息**大部分可以从标量统计中恢复**

---

## 4. 模型对比总览

| | Baseline | Phase 1 | **Phase 2 (Full)** | **Phase 2 (Lite)** |
|---|----------|---------|-------------------|-------------------|
| 数据量 | 5,134 | 9,394 | **27,223** | 27,223 (蒸馏) |
| input_dim | 1034 | 1034 | **1034** | **10** |
| 参数量 | 2.26M | 2.26M | **2.26M** | **540K** |
| val_loss | 2.732 | 1.027 | **0.773** | 0.577 (蒸馏 loss) |
| rank_acc | 11.5% | 54.4% | **69.0%** | **61.9%** |
| rank_acc_pm1 | 33.2% | 92.8% | **96.4%** | **94.8%** |
| rank_mae | 2.76 | 0.542 | **0.350** | **0.438** |
| rating_corr | 0.449 | 0.993 | **0.995** | **0.992** |

---

## 5. 数据管道状态

### 5.1 KAB2 缓存

| 指标 | 数量 |
|------|------|
| 已缓存 (KAB2 .npz) | ~30,245 |
| 训练集 (T split) | 27,223 |
| 验证集 (V split) | 3,022 |
| 全部含 HumanSL 标签 | 30,245 / 30,245 |

---

## 6. 训练路线图

### 已完成
- [x] Phase 1: 策略改进训练 (StratifiedRankSampler, dropout, warmup)
- [x] Phase 2: 全量 Teacher 训练 (27.2k 游戏, val_loss 0.7727)
- [x] Phase 2 Lite: 全量蒸馏 (10-dim, rank_acc_pm1 94.8%)

### 下一步
- **数据扩充** — 数据量收益未饱和，更多游戏预期继续改善
- **模型容量探索** — hidden_dim 128→256, depth+1
- **超参调优** — loss 权重、label smoothing、gradient accumulation
- **Lite 模型部署** — engine_mode=lite 快速推理模式

---

## 附录

### A. 环境配置
- GPU: NVIDIA GeForce RTX 3060 (12GB)
- CUDA: 13.2 (driver), PyTorch 2.12.0+cu126
- KataGo: v1.16.5 (custom fork with batch_analysis)
- Python: 3.12+, uv 包管理

### B. 文件结构
- `nets/katarank/best.pt` — Phase 2 最优 full checkpoint (8.7MB)
- `nets/katarank/training_report.json` — Phase 2 训练报告
- `nets/katarank_lite/best_lite.pt` — Phase 2 最优 lite checkpoint (2.2MB)
- `nets/katarank_lite/training_report.json` — Phase 2 lite 蒸馏报告
- `~/.katarank/kab2_cache/` — KAB2 特征缓存 (增量)
- `src/katarank/train/training.py` — 主训练入口
- `src/katarank/train/distill.py` — 蒸馏训练模块
- `src/katarank/data/datasets/dataset_kab2.py` — KAB2 Dataset + StratifiedRankSampler
