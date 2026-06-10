# KAB2 训练适配研究报告

> 基于 STRENGTH_MODEL_RESEARCH.md v3 + batch_analysis KAB2 输出分析

## 一、数据结构变化

### 1.1 KAB2 文件格式

```
Header: 96 bytes = [magic:4][n:4][sc:4][tr:4][pk:4][nnX:4][nnY:4][flags:4][PlayerSummary:64B]
Per move: scalars[10] + avgTrunk[C] + pick[C]   C = trunkCh (384 or 512)
                                         总：10 + 2C 个 float

旧数据 (KABN/KABT):  12 + 256 + 256  = 524 floats/move (256 trunk, hardcoded)
新数据 (KAB2):       10 +  C  +  C   = 10 + 2C  floats/move (C 动态, 384/512)

模型 b28c512nbt (本地 default_model):
  trunkCh = 512 → 10 + 1024 = 1034 floats/move
```

### 1.2 Scalar 字段重定义

```
旧 (12 dim):  head[12] — 手工拼凑的 12 维
新 (10 dim):  
  [0] whiteWinProb      [1] whiteLossProb     [2] whiteNoResultProb
  [3] whiteScoreMean/50 [4] shorttermScoreError/10
  [5] policyPrior        [6] policyRank/361    [7] isWhite
  [8] winDelta           [9] scoreDelta/50 ← 延迟回填，t+1 手知道 t 手的 delta
```

### 1.3 PlayerSummary

```
旧: 10 项聚合指标（无人类参考）
新: 12 项 + humanRankIdx(0-28) + humanLogPrior
    humanRankIdx = -1 表示未启用 HumanSL
```

## 二、模型架构变化（与现有 model/ 对比）

现有 `model/trunk_pick_head.py` 与 RESEARCH.md 设计之间的差异：

| 维度 | 现有代码 | RESEARCH.md 设计 |
|------|---------|-----------------|
| 输入 | trunk[256] + pick[256] + head[12] | scalars[10] + avgTrunk[C] + pick[C] |
| 通道数 | 硬编码 256 | 动态 C (384/512) |
| 自注意力 | 一个序列（不含对手） | 黑白各一个，独立编码 |
| 交叉注意力 | pick → trunk_enc 交叉 | 因果掩码交叉（B←→W） |
| 池化 | head-weighted sum | SegmentedAttentionPool（开局/中盘/收官） |
| 输出 | score + style + key_moves | 双评分(黑白各一) + 阶段分项(3×2) + 段位辅助分类 |
| 棋局条件 | 无 | FiLM Conditioning (komi, rules, handicap) |

## 三、模型输入数据适配

### 3.1 Per-move 特征 (1034 dim)

```python
C = 512  # trunkCh from model
move_dim = 10 + C + C  # scalars + avgTrunk + pick

# 分组处理
scalars = x[:, :10]        # (N, 10)
avg_trunk = x[:, 10:10+C]  # (N, C)
pick = x[:, 10+C:]         # (N, C)
```

### 3.2 Per-game 元数据

```python
# 从 NPZ header 提取 (96 bytes offset)
summary = np.frombuffer(header_bytes[32:96], dtype=np.float32)  # (16,)
human_rank = summary[10]  # 0-28 或 -1
human_logp = summary[11]
```

## 四、训练标签构造

### 4.1 客观评分（主任务，无外部依赖）

```
综合评分 = aggregate(
    avg(win_delta),        # 每手平均胜率变化
    avg(score_delta),      # 每手平均得分变化  
    avg(policy_prior),     # 平均落子概率
    top1_accuracy,          # 与 KataGo 一致率
    score_variance          # 稳定性
)
```

这些全部可以从 scalars + PlayerSummary 中直接聚合，**无需任何人类标注**。

### 4.2 HumanSL 辅助校准

```
secondary_loss = CE(pred_rank_onehot, humanRankIdx_onehot)
                 × w_rank (建议 0.1)
```

## 五、实施路径

```python
# 修改点清单

# 1. 数据加载
#    train_loader.py: 解析 KAB2 header → PlayerSummary
#                     scalars[10] + avgTrunk[C] + pick[C]

# 2. 模型输入
#    从 input_dim=768→1034 (动态: 10+2C)
#    从单序列→双序列 (B/W 分开)

# 3. 模型架构
#    删除旧的 trunk_pick_head.py
#    新建: dual_transformer.py (RESEARCH.md §9.2)
#    - Dual Self-Attention
#    - Causal Cross-Attention
#    - SegmentedAttentionPool
#    - Multi-task output heads

# 4. 训练管线
#    loss = score_loss(B) + score_loss(W) + phase_loss + rank_loss
```

## 六、旧数据向后兼容

542 个旧 NPZ 文件（KABN 格式，12+256+256）与新 KAB2 格式不兼容。

两种选择：
- A. 用最新 katago 重跑全部 271 局 → 推荐（数据格式统一）
- B. 写兼容层读取旧格式 → 不推荐（增加维护成本）
