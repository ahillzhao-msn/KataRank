# Kata Native Training — 纯 Katago 训练模式设计

> 基于 DESIGN_v1.1.md 的"监督信号内生性"理念，利用 Katago humanSL 模型
> 建立从 25k 到职业 9d 全覆盖的客观棋力评估体系。

## 一、核心理念

### 问题
go-strength-model 使用人类 Glicko-2 评分作为监督信号。这引入三重偏差：
1. **数据有限** — 只有有人类段位标签的棋谱才能训练
2. **标签噪声** — 不同平台段位标准不一，Glicko-2 自身有统计误差
3. **不可扩展** — 无法主动生成指定强度的训练数据

### 解决方案
**用 Katago 训练 Katago。** 具体来说：

1. 下载不同强度的 **humanSL 权重**（Katago 专门模仿人类棋风的模型）
2. 让不同强度的权重互相**自我对弈**，生成海量棋谱
3. 每局棋的"棋手水平" = 下这盘棋的权重在 humanSL 评分梯上的 Elo
4. 模型学习：**"给定这手棋，它的棋手实力是多少 Elo？"**

### humanSL 的优势

| 特性 | 标准 Katago | humanSL |
|------|-------------|---------|
| 目标 | 最优着法 | 模仿人类 |
| 棋风 | 超人类 | 类人 |
| 实力区间 | ~5000+ Elo | 25k → 9d（全人类范围） |
| 权重分布 | 少量最强权重 | 不同训练阶段的多强度快照 |
| 适用性 | 只能评估"最优度" | 可以直接评估"像几段" |

## 二、三层监督信号

```
权重 Elo（主损失）
    └── 模型预测给定走法集合对应的人类棋手 Elo
        └── 损失函数: MSE (predicted_elo, true_elo)

每手质量（辅助损失/弱监督）
    └── 每手棋的: 胜率损失(winrate_loss)、策略损失(policy_loss)
        └── 来自 Katago 标准权重对每手棋的质量评估
            └── 损失函数: AbilityLoss (弱监督模式)

对局一致性（辅助损失）
    └── Bradley-Terry: P(黑胜) = 1/(1+10^((W_elo - B_elo)/400))
        └── 给定已知 Elo 差，胜负应该符合概率分布
            └── 损失函数: BradleyTerryLoss
```

### 权重 Elo 作为主标签的原理

humanSL 权重在训练过程中会产生一系列中间快照。每个快照有：
- 训练步数（step count）
- 在 humanSL 自我对弈评估梯上的 Elo 评分

这些 Elo 评分是**纯净的**——完全来自权重之间的对弈结果，无人类干预。

所以：
- humanSL 权重 A（Elo=800）vs humanSL 权重 B（Elo=1200）
- A 执黑下的棋 = "800 Elo 水平的棋"
- B 执白下的棋 = "1200 Elo 水平的棋"
- 模型的任务：从棋步特征中反向推断出这个 Elo

## 三、训练数据生成管线

```
                    ┌──────────────────────┐
                    │  katagotraining.org   │
                    │  humanSL 权重列表     │
                    └──────────┬───────────┘
                               │ 下载
                               ▼
                    ┌──────────────────────┐
                    │  本地权重存储          │
                    │  weights/humanSL/     │
                    │  ├── elo_0800.bin.gz  │
                    │  ├── elo_1200.bin.gz  │
                    │  └── ...              │
                    └──────────┬───────────┘
                               │ 配对
                               ▼
                    ┌──────────────────────┐
                    │  自我对弈调度器        │
                    │  Katago selfplay      │
                    │  权重A vs 权重B       │
                    │  → SGF 输出            │
                    └──────────┬───────────┘
                               │ 特征提取
                               ▼
                    ┌──────────────────────┐
                    │  extract_features     │
                    │  → Pick/Trunk 特征    │
                    │  → {game}_BlackRecent │
                    │  → {game}_WhiteRecent │
                    └──────────┬───────────┘
                               │ 标签分配
                               ▼
                    ┌──────────────────────┐
                    │  KataNativeDataset    │
                    │  特征 + Elo 标签      │
                    │  可用于训练            │
                    └──────────────────────┘
```

### Step 1: 权重获取

```python
# data/katago_native/weight_manager.py
weights = download_humanSL_weights(min_elo=500, max_elo=3500, count=20)
# 返回 [(elo, local_path, network_name), ...]
```

从 katagotraining.org 获取 humanSL 权重的 Elo 表。每个权重包含：
- Elo rating
- 下载 URL
- 网络架构（如 b18c384nbt）

### Step 2: 自我对弈

```python
# data/katago_native/selfplay.py
for (elo_a, weight_a), (elo_b, weight_b) in pair_weights(weights):
    sgf = run_selfplay(
        katago_path="./katago.exe",
        weight_a=weight_a, weight_b=weight_b,
        games_per_pair=100,
        output_dir="./data/selfplay/",
    )
    # 记录元数据
    metadata.append({
        "sgf": sgf,
        "black_elo": elo_a,
        "white_elo": elo_b,
        "black_weight": weight_a_name,
        "white_weight": weight_b_name,
    })
```

**配对策略**：
- 均匀覆盖各种实力差
- 相近实力对局（差 < 200 Elo）→ 势均力敌，结果有随机性
- 差距较大对局（差 > 500 Elo）→ 强方几乎必胜，标签更可靠
- 混合比例：70% 相近实力 + 30% 差距较大

### Step 3: 特征提取

使用 go-strength-model fork 的 `extract_features` 命令：

```bash
katago extract_features -model {weight} -config analysis.cfg \
  -list games.csv -featuredir ./featurecache \
  -with-pick -window-size 500 -batch-size 10
```

### Step 4: 标签分配

每个对局生成两条训练样本：
```
样本1: Black_features → predict black_elo
样本2: White_features → predict white_elo
```

## 四、数据集格式

### CSV 游戏列表（兼容现有格式）

```csv
File,Player Black,Player White,Score,BlackRating,WhiteRating,Set
selfplay/001.sgf,humanSL_800,humanSL_1200,0,800,1200,T
selfplay/002.sgf,humanSL_1200,humanSL_800,1,1200,800,T
```

关键区别：
- `BlackRating` / `WhiteRating` 不是 Glicko-2，而是 humanSL 权重的实际 Elo
- `Player Black` / `Player White` 记录的是权重标识符
- `Score` 来自对局结果

### 特征存储

复用 go-strength-model 的 featurecache 目录结构：
```
featurecache/
├── selfplay/001_BlackRecent_pick.npy  # (N_b, 256) Black's moves
├── selfplay/001_WhiteRecent_pick.npy  # (N_w, 256) White's moves
└── selfplay/001_BlackRecent.zip       # (压缩格式)
```

## 五、归一化问题

humanSL Elo 和 Glicko-2 的尺度不同。需要在训练中处理：

1. **Z-score 归一化**：计算训练集 Elo 的 mean/std，归一化到 ~N(0,1)
2. **推理时逆转**：模型输出归一化值 → 乘以 std + mean → 实际 Elo

这与当前 go-strength-model 的做法一致，只是换了一组统计值。

## 六、与现有管线的集成

```
KataNativeDataset(kata_native=True)
    ↓ 相同的 collate_fn
KataRankMultiTaskModel(enable_score=True, enable_abilities=False)
    ↓ 主损失: score_head → MSE(weight_elo)
    ↓ 辅助损失: bradley_terry → 对局一致性
```

训练命令：
```bash
python train/train.py --config train/config_kata_native.yaml
```

**渐进式训练策略**：
1. **Phase 1**: humanSL Elo 回归（纯自对弈数据）
2. **Phase 2**: 加入人类棋谱微调（可选）
3. **Phase 3**: 加入多任务能力头

## 七、关键设计决策

| 决策 | 选择 | 理由 |
|------|------|------|
| 监督信号 | humanSL 权重 Elo | 最纯净、最客观的标签 |
| 特征类型 | Pick (256-dim) | go-strength-model 已验证 |
| 对弈模式 | GTP vs GTP | 最简单可靠，无需修改 Katago |
| 配对策略 | 70% 相近 + 30% 差距大 | 平衡学习难度和标签信噪比 |
| 覆盖范围 | 500-3500 Elo | 对应 30k 到 9d+ 的人类范围 |
| 归一化 | Z-score | 兼容现有管线 |

## 八、后续扩展

### 知识蒸馏（DESIGN_v1.1.md 第 3 步）

用标准 Katago 超强权重作为"教师网络"：
1. 教师网络分析每手棋，给出"最优着法偏差"
2. 这个偏差可以作为额外的弱监督信号
3. 模型不仅学习"这像几段的棋"，还学习"离最优有多远"

### 多权重集成

用多个不同 humanSL 权重分析同一局人类棋谱：
- 每个权重给出一个评估
- 取平均值作为最终评估
- 方差可以作为置信度指标

### 持续学习

新 humanSL 权重发布时，自动加入训练：
- 增量训练（在已有模型上继续）
- 不需要重新训练整个 pipeline
