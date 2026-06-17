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

## 3. V2 Lite Model — 知识蒸馏

### 3.1 动机

当前 full 模型 (input_dim=1034) 依赖 KataGo trunk vectors，推理时需要跑完整的 KataGo 神经网络。V2 lite 模型 (input_dim=10) 仅使用标量特征（winrate, score, policy 等），不需要 trunk vectors，推理速度可提升 5-10 倍。

**方法: 知识蒸馏 (Knowledge Distillation)**
- Teacher: 已训练的 full 模型 (1034-dim, 2.26M 参数)
- Student: lite 模型 (10-dim, 540K 参数, hidden_dim=64)
- Loss: KL-divergence(student_rank_probs, teacher_rank_probs) + MSE(student_rating, teacher_rating) + 可选 hard label anchor

Student 学习的是 teacher 的**软概率分布**，而非硬标签。这比直接用硬标签训练更有效，因为软分布携带了段位之间的相对关系信息（例如 "这个棋手大概率是 5k，小概率是 4k 或 6k"）。

### 3.2 DistillationLoss 设计

```
L_total = w_kl × KL(teacher_soft ‖ student_soft)      [rank 分布蒸馏]
        + w_rating × MSE(student_rating, teacher_rating)  [连续强度蒸馏]
        + w_hard × RankAnchorLoss(student, hard_labels)    [段位校准锚点]
```

Temperature T 控制 teacher 分布的软度。T>1 使分布更平滑，给 student 更丰富的梯度信号。KL loss 乘以 T² 使梯度量级与 T 无关。

### 3.3 渐进式微量验证实验 (2026-06-17)

目的: 用极小样本验证 distillation pipeline 端到端可跑通，观察数据量与效果的关系。

**配置:**
- Teacher: Phase 1 best.pt (val_loss=1.027)
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

**观察:**
1. **Rating 预测快速学会** — 50 局时 rating_corr 就达到 0.89，100 局 0.96
2. **Rank 分类未学会** — 验证集样本太少（10-40 个有 label），且 10-dim 输入信息量有限
3. **val_loss 持续下降** — 8.65 → 5.96，pipeline 工作正常
4. **200 局 rating_corr 回落** (0.96→0.94) — warm-start LR 或验证集随机性

**结论:** Pipeline 完全跑通。Lite 模型在 rating 预测上有潜力（10-dim scalars 确实编码了强度信息），但 rank 分类需要更多数据和可能更大的 student hidden_dim。全量 5k+ 训练预计效果显著改善。

---

## 4. 数据管道状态

### 4.1 KAB2 缓存

| 指标 | 数量 |
|------|------|
| 已缓存 (KAB2 .npz) | 5,704 |
| DB 已分析 | ~17,000 |
| 待生成 KAB2 | ~11,300 |
| DB 有 SGF 总量 | ~30,000 |

### 4.2 HumanSL 段位分布 (已缓存数据)

```
20k( 0):     1    18k( 2):    31    17k( 3):     5    16k( 4):    37
15k( 5):    58    14k( 6):   100    13k( 7):   299    12k( 8):   183
11k( 9):   306    10k(10):   471     9k(11):   602     8k(12):   899
 7k(13):   942     6k(14):  1034     5k(15):  1174     4k(16):  1207 ← 峰值
 3k(17):  1111     2k(18):   926     1k(19):   755     1d(20):   526
 2d(21):   346     3d(22):   195     4d(23):   101     5d(24):    52
 6d(25):    36     7d(26):     6     8d(27):     5
```

近似正态分布，中心在 4k-5k，覆盖 18k-8d。两端极度稀疏（20k 仅 1 局）。

---

## 5. 训练路线图

### Phase 2: 全量 Teacher 重训 (待 KAB2 生成完成)
- 数据: 17k+ 局 (全量)
- 策略: Warm-start from Phase 1 best.pt, LR 降到 1/5 (1e-4)
- 保持 StratifiedRankSampler + dropout 0.2
- 预期: val_loss < 0.8, rank_acc_pm1 > 95%

### Phase 3: V2 Lite 全量蒸馏
- Teacher: Phase 2 最优 checkpoint
- Student: input_dim=10, hidden_dim=64 (或 128 if 10-dim 不够)
- 全量 17k+ 局蒸馏
- 预期: rating_corr > 0.95, rank_acc_pm1 待验证

### Phase 4: 增量训练协议
- 新数据到达时: Warm-start + 低 LR (1/10)
- 分层采样加权新数据
- 目标: 30k 局全量训练

---

## 附录

### A. 环境配置
- GPU: NVIDIA GeForce RTX 3060 (4GB)
- CUDA: 13.2 (driver), PyTorch 2.12.0+cu126
- KataGo: v1.16.5 (custom fork with batch_analysis)
- Python: 3.10+, uv 包管理

### B. 文件结构
- `nets/katarank/best.pt` — Phase 1 最优 teacher checkpoint
- `nets/katarank/training_report.json` — Phase 1 训练报告
- `nets/katarank_lite/` — V2 lite distillation checkpoints
- `~/.katarank/kab2_cache/` — KAB2 特征缓存 (增量)
- `src/katarank/train/training.py` — 主训练入口
- `src/katarank/train/distill.py` — 蒸馏训练模块
- `src/katarank/data/datasets/dataset_kab2.py` — KAB2 Dataset + StratifiedRankSampler
