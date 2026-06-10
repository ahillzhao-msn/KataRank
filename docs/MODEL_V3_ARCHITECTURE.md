# KataRank Model v3 Architecture

**Version**: 3.0  
**Date**: 2026-06-09  
**Based on**: STRENGTH_MODEL_RESEARCH.md §9–§14, KAB2 format spec

---

## 1. 设计目标

| 目标 | 方案 |
|---|---|
| 置换不变性 | ISAB Set Transformer（对每局手棋顺序不敏感） |
| 双视角评估 | Black / White 独立编码流 + 跨流交叉注意力 |
| 因果防泄漏 | CrossMAB 施加时序掩码：只看对手**已有**落子 |
| 强度信号自给自足 | 训练标签来自 KAB2 header，无需外部 Elo |
| 段位边界自维护 | HumanSL 训练期锚定 threshold，推理期模型自主输出 |
| 动态维度 | `Linear(input_dim, hidden_dim)` 支持任意 trunkCh |

### 1.1 HumanSL 的角色定位：训练期锚定器，非推理期依赖

HumanSL 在本系统中是**一次性的段位空间标定工具**，而非持久依赖：

```
训练期                              推理期
──────────────────────────────      ──────────────────────────
KAB2 batch_analysis（含 -human-model）
  → humanRankIdx / humanLogPrior    [不需要]
  → 以置信度加权的 CE loss
  → 将 OrdinalHead 的 28 个可学习
    threshold 锚定到有意义的段位边界
  → 通过自我嵌入聚合，同段位棋谱在
    z 空间自然聚集
                                    推理时：
                                      b_rating / w_rating  ← 唯一权威强度值
                                      rank_probs           ← 来自已锚定的
                                                             threshold，无需
                                                             HumanSL
```

**推理时模型是完全自洽的**：OrdinalHead 的 threshold 在训练中内化了段位语义，之后由模型自发维护，不再依赖外部评定。

---

## 2. 输入规格

### 2.1 每手特征向量（per-move）

```
move_vec (1, input_dim)  where input_dim = 10 + 2 × trunkCh
```

| 区段 | 索引 | 内容 |
|---|---|---|
| `scalars` | `[0..9]` | winRate, scoreLead, complexity, policyEntropy, priorProb, winDelta, scoreDelta, **isWhite**, turnNumber, boardArea |
| `pick` | `[10 .. 10+trunkCh-1]` | KataGo trunk 在实际落子位置的激活（NCHW → `trunk[ch * HW + rowPos]`） |
| `avgTrunk` | `[10+trunkCh .. input_dim-1]` | KataGo trunk 全盘空间均值（全局棋盘理解） |

> **scalar[7] = isWhite**：0=Black，1=White。模型用此字段在流内分割 Black/White。

### 2.2 游戏级标签（PlayerSummary，from KAB2 header）

| 字段 | 类型 | 范围 | 用途 |
|---|---|---|---|
| `meanLogPrior` | float | -5.9（20k）~ -0.5（9d） | 主任务：强度回归目标 |
| `humanRankIdx` | float→int | 0-28 或 -1（未启用） | 辅任务：段位分类目标 |
| `humanLogPrior` | float | < 0 | 段位标签置信度权重 |

---

## 3. 前向流程（完整标注尺寸）

以 b18 KataGo（trunkCh=512）、hidden_dim=128、batch=1 游戏（N_b+N_w 手）为例：

```
Input:  x  (N_b+N_w, 1034)          ← 拼接 _B.npz + _W.npz 的 moves

─── Step 1: Input Projection ────────────────────────────────────
    h = Linear(1034 → 128)(x)        (N_b+N_w, 128)  共享，B/W 共用
    h = Dropout(h)

─── Step 2: Player Split ────────────────────────────────────────
    is_black = x[:, 7] < 0.5
    h_b = h[is_black]                (N_b, 128)
    h_w = h[~is_black]               (N_w, 128)
    turn_b = x[is_black,  8]         (N_b,)  ← scalar[8]=turnNumber
    turn_w = x[~is_black, 8]         (N_w,)

─── Step 3: Independent Encoding ────────────────────────────────
    h_b = SetEncoder_b(h_b)          (N_b, 128)   ← ISAB × depth
    h_w = SetEncoder_w(h_w)          (N_w, 128)   ← 独立权重

─── Step 4: Causal Cross-Attention ──────────────────────────────
    mask_bw = causal_mask_bw(turn_b, turn_w)  (N_b, N_w)
    mask_wb = causal_mask_wb(turn_w, turn_b)  (N_w, N_b)
    for block in BidirectionalCrossMAB.blocks:
        Δh_b = MAB(h_b, h_w, attn_mask=mask_bw)
        Δh_w = MAB(h_w, h_b, attn_mask=mask_wb)
        h_b  = LayerNorm(h_b + Δh_b)
        h_w  = LayerNorm(h_w + Δh_w)

─── Step 5: Segmented Attention Pooling ────────────────────────
    z_b = SegmentedAttentionPool(h_b)    (3×128 = 384)
    z_w = SegmentedAttentionPool(h_w)    (3×128 = 384)
    z   = Linear(768 → 128)(concat(z_b, z_w))   (128,)

─── Step 6: Output Heads ────────────────────────────────────────
    b_rating    = Linear → scalar         ← MSE vs meanLogPrior_B
    w_rating    = Linear → scalar         ← MSE vs meanLogPrior_W
    b_rank_probs = OrdinalHead(29) → probs ← CE  vs humanRankIdx_B
    w_rank_probs = OrdinalHead(29) → probs ← CE  vs humanRankIdx_W
```

---

## 4. 因果掩码（Causal Cross-Attention Mask）

### 4.1 时序结构

标准围棋交错顺序：B₀ W₀ B₁ W₁ B₂ W₂ ...

- Black move `i` 对应 turn = `turn_b[i]`（通常 = 2i）
- White move `j` 对应 turn = `turn_w[j]`（通常 = 2j+1）

### 4.2 可见性规则

```
B→W 方向（Black 作为 query，White 作为 key）：
    Black move i 只能看到「已经落下的」White 手
    可见条件: turn_w[j] < turn_b[i]
    
    mask_bw[i, j] = 0    if turn_w[j] < turn_b[i]
                  = -inf  otherwise
```

```
W→B 方向（White 作为 query，Black 作为 key）：
    White move j 只能看到「已经落下的」Black 手（含本轮的 Black 先手）
    可见条件: turn_b[i] <= turn_w[j]      ← 含等号，因为 Black 先手
    
    mask_wb[j, i] = 0    if turn_b[i] <= turn_w[j]
                  = -inf  otherwise
```

### 4.3 实现代码

```python
def build_causal_mask_bw(turn_b: torch.Tensor,
                          turn_w: torch.Tensor) -> torch.Tensor:
    """B→W: Black[i] attends to White[j] iff turn_w[j] < turn_b[i]."""
    # (N_b, 1) < (1, N_w) → broadcast
    visible = turn_w.unsqueeze(0) < turn_b.unsqueeze(1)  # (N_b, N_w) bool
    mask = torch.full((len(turn_b), len(turn_w)), float('-inf'),
                      device=turn_b.device)
    mask[visible] = 0.0
    return mask

def build_causal_mask_wb(turn_w: torch.Tensor,
                          turn_b: torch.Tensor) -> torch.Tensor:
    """W→B: White[j] attends to Black[i] iff turn_b[i] <= turn_w[j]."""
    visible = turn_b.unsqueeze(0) <= turn_w.unsqueeze(1)  # (N_w, N_b) bool
    mask = torch.full((len(turn_w), len(turn_b)), float('-inf'),
                      device=turn_w.device)
    mask[visible] = 0.0
    return mask
```

### 4.4 为什么两个方向不对称

| 方向 | 可见条件 | 直觉 |
|---|---|---|
| B→W | `turn_w < turn_b` | 黑棋 i 手时，白棋还没有「回应」，严格小于 |
| W→B | `turn_b <= turn_w` | 白棋 j 手时，黑棋已经在本轮先手落子，含等于 |

若 B→W 方向也用 `<=`，则 Black move 0 可看到 White move 0（白棋的回应），构成未来信息泄漏。

### 4.5 CrossMAB 修改要点

`MultiHeadAttentionBlock.forward` 需要增加 `attn_mask` 参数：

```python
def forward(self, X, Y, X_lens=None, Y_lens=None, attn_mask=None):
    X_3d = X.unsqueeze(1)
    Y_3d = Y.unsqueeze(1)
    attn_out, _ = self.mha(X_3d, Y_3d, Y_3d, attn_mask=attn_mask)
    ...
```

`CrossMAB.forward` 中构建并传入掩码：

```python
def forward(self, h_b, h_w, blens, wlens, turn_b=None, turn_w=None):
    mask_bw = build_causal_mask_bw(turn_b, turn_w) if turn_b is not None else None
    mask_wb = build_causal_mask_wb(turn_w, turn_b) if turn_w is not None else None
    h_b_out = self.mab_bw(h_b, h_w, attn_mask=mask_bw)
    h_w_out = self.mab_wb(h_w, h_b, attn_mask=mask_wb)
    return h_b_out, h_w_out
```

> **Batch 注意**：当批量 > 1 时，不同游戏的 N_b/N_w 不等，需要 padding 或逐游戏构建 mask。推荐实现为"逐游戏处理后 cat"，与 packed batch 的 xlens 机制一致。

---

## 5. 分段注意力池化（Segmented Attention Pooling）

RESEARCH §9.4 的设计：避免 200 手均值稀释关键着法（妙招仅占 ~1% 手数）。

```python
class SegmentedAttentionPool(nn.Module):
    """
    Pool sequence in 3 segments: opening / midgame / endgame.
    Attention weights automatically emphasize high-complexity moves.
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.score = nn.Linear(d_model, 1)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        """
        Args:
            seq: (N, d_model) encoded move sequence for one stream
        Returns:
            (3 * d_model,) concatenated segment poolings
        """
        n = seq.size(0)
        s1, s2 = n // 3, 2 * n // 3
        segs = [seq[:s1], seq[s1:s2], seq[s2:]]
        out = []
        for seg in segs:
            if len(seg) == 0:
                out.append(torch.zeros(seq.size(-1), device=seq.device))
            else:
                w = torch.softmax(self.score(seg), dim=0)  # (k, 1)
                out.append((seg * w).sum(dim=0))            # (d_model,)
        return torch.cat(out)  # (3 * d_model,)
```

`StreamPooling` 更新为使用 `SegmentedAttentionPool`：

```python
# 更新 StreamPooling.forward:
z_b = self.seg_pool(h_b)          # (3 * hidden_dim,)
z_w = self.seg_pool(h_w)          # (3 * hidden_dim,)
z   = self.fusion(cat([z_b, z_w]))  # Linear(6*hidden_dim → hidden_dim)
```

---

## 6. HumanSL 的训练期锚定机制

### 6.1 核心原则

> **HumanSL 仅用于训练期**：它是 OrdinalHead threshold 的**一次性语义锚点**，不是推理期的依赖。  
> **推理期唯一答案**：`b_rating` / `w_rating`（连续强度值）和 `rank_probs`（来自已锚定 threshold 的推断），两者均由模型自主输出。

### 6.2 两个信号的语义

| 信号 | 来源 | 阶段 | 作用 |
|---|---|---|---|
| `meanLogPrior` | KataGo policy | 训练 + 推理 | 主任务回归目标；KataGo 视角的客观棋力 |
| `humanRankIdx` | HumanSL best-fit | **仅训练** | 锚定 OrdinalHead 的 threshold 到有意义段位边界 |
| `humanLogPrior` | HumanSL log-likelihood | **仅训练** | 段位标签的置信度权重 |

### 6.3 OrdinalHead 的 threshold 锚定机制

```
OrdinalHead 内部:
  θ = Linear(z) → (batch,) 一维强度投影
  thresholds (28,) 可学习，初始化 linspace(-2.5, 2.5)

训练期（有 humanRankIdx）:
  rank_confidence = sigmoid(humanLogPrior - (-4.0))   ← 置信度门控
  rank_loss = rank_confidence × CE(rank_probs, humanRankIdx)
                   ↓
  gradient 把 threshold[k] 拉向「meanLogPrior 轴上第 k 段位的边界」
                   ↓
  threshold 收敛：
    τ₀ ≈ θ(20k/19k 边界) ← 对应 meanLogPrior ≈ -5.7
    τ₁ ≈ θ(19k/18k 边界)
    ...
    τ₂₇ ≈ θ(8d/9d 边界)  ← 对应 meanLogPrior ≈ -0.7

训练深入（无 humanRankIdx 的新棋谱）:
  rank_loss = 0（mask 掉）
  主任务 MSE 仍更新 Linear(z) 的方向
  → threshold 依靠 meanLogPrior 分布的内在规律自发维护
  → 同等强度的棋谱在 θ 轴上聚集（自我嵌入聚合）

推理期:
  θ = Linear(z)             ← 纯模型输出
  rank_probs = f(θ, τ)      ← 已锚定的 threshold，无需 HumanSL
```

### 6.4 自我嵌入聚合的正反馈循环

```
① meanLogPrior 差异 → 主任务 loss → z 在 θ 方向分离（强弱有别）
② 训练初期 humanRankIdx → rank loss → threshold 锚定到段位边界
③ threshold 稳定后 → rank_probs 与 b_rating 一致 → 主任务 loss 进一步强化 z 的段位聚集
④ 新数据即使无 humanRankIdx → ③ 的正反馈已足以维持聚集状态
```

越训练越自洽：rank head 和 rating head 相互强化，段位边界无需外力维持。

### 6.5 主任务投影

```
z (hidden_dim) → Linear(hidden_dim → hidden_dim//2) → ReLU → Dropout
              → Linear(hidden_dim//2 → 1) → b_rating   ← MSE vs norm(meanLogPrior_B)
              → Linear(hidden_dim//2 → 1) → w_rating   ← MSE vs norm(meanLogPrior_W)
```

归一化：`norm(x) = (x − μ) / σ`，μ ≈ −3.2，σ ≈ 1.4。

### 6.6 两个信号在段位空间的线性关系

```
meanLogPrior:  -5.9   -4.5   -3.2   -1.9   -0.5
humanRankIdx:   0      7     14     21     28
               20k    13k    6k     1d     9d
```

近似关系：`humanRankIdx ≈ (meanLogPrior + 5.9) / 5.4 × 28`

这是 `batch_analysis` 的 `rankCandidates` 公式的逆，说明两个信号在同一空间内共轭。训练成功后，OrdinalHead 的 threshold 应当近似落在这条线对应的 θ 位置上。

---

## 7. 多任务损失

```python
# 训练期完整 loss（含 HumanSL 锚定项）
loss = (
    1.0 * (mse(b_rating, target_b) + mse(w_rating, target_w))   # 主任务：强度回归
  + 2.0 * BradleyTerry(b_rating, w_rating, stronger)             # 相对一致性
  + 0.1 * (rank_loss_b + rank_loss_w)                            # 段位锚定（仅有 humanRankIdx 时）
  + 1e-5 * L2(model)
)

# rank_loss 实现（置信度门控）
def rank_loss(rank_probs, rank_target, human_log_prior):
    if rank_target < 0:                         # humanRankIdx == -1，未启用 HumanSL
        return torch.tensor(0.0)
    confidence = torch.sigmoid(               # humanLogPrior 置信度门控
        torch.tensor(human_log_prior + 4.0)
    )
    nll = F.nll_loss(torch.log(rank_probs + 1e-8),
                     torch.tensor(rank_target))
    return confidence * nll
```

`BradleyTerry` 的 `stronger` 标签：`1 if meanLogPrior_B > meanLogPrior_W else 0`（KAB2 header 直接给出，无需外部数据）。

### 7.1 推理期 loss（训练后）

推理时**不计算任何 loss**，也不使用 HumanSL。模型直接输出：

```
b_rating  → 连续强度值（归一化，可反归一化到 meanLogPrior 尺度）
w_rating  → 同上
rank_probs_B (29,)  → 段位概率分布（argmax → 段位标签）
rank_probs_W (29,)
```

rank_probs 的段位边界来自训练中锚定的 threshold，不依赖 HumanSL 模型。

### 7.2 训练阶段策略（Curriculum）

| 阶段 | 数据 | 启用的 loss 项 | 目的 |
|---|---|---|---|
| Phase 1（锚定期） | 含 HumanSL 标注的棋谱 | rating + BT + **rank** | threshold 收敛到段位边界 |
| Phase 2（泛化期） | 大量棋谱（可无 HumanSL） | rating + BT（rank 自动 mask） | 依靠分布自发维护段位聚集 |

Phase 1 的 epoch 数不需要多，只要 threshold 收敛即可（通常 20–50 epoch 后段位边界已稳定）。

---

## 8. 完整模型配置

```yaml
model:
  input_dim: auto           # 从 KAB2 header 读取：10 + 2 * trunkCh
  hidden_dim: 128
  num_heads: 4
  num_inducing: 16          # ISAB 诱导点数
  encoder_depth: 2          # 每个流的 ISAB 层数
  cross_depth: 1            # BidirectionalCrossMAB 层数
  dropout: 0.1
  pooling: segmented        # SegmentedAttentionPool (开局/中盘/官子)
  player_dim: 7             # scalar[7] = isWhite
  causal_mask: true         # 因果交叉注意力

  # Heads
  enable_dual_rating: true  # 主任务
  enable_ordinal: true      # 辅任务（29 段位）
  num_ranks: 29             # 20k(0) → 9d(28)
  enable_abilities: false   # 暂不启用（无标签）
  enable_style: false       # 暂不启用（无标签）

training:
  loss_weights:
    rating_mse: 1.0
    bradley_terry: 2.0
    rank_ordinal: 0.1
    l2: 1e-5
  label_normalization:
    meanLogPrior_mu: -3.2
    meanLogPrior_sigma: 1.4
  rank_confidence_threshold: -4.0   # humanLogPrior 低于此值时 rank loss 权重接近 0
```

---

## 9. 与现有代码的对应关系

| v3 设计点 | 现有代码 | 需要的改动 |
|---|---|---|
| 因果掩码 | `CrossMAB`（无 mask） | 新增 `build_causal_mask_bw/wb`；`MAB` 接受 `attn_mask` |
| 分段注意力池化 | `StreamPooling`（注意力种子向量） | 替换为 `SegmentedAttentionPool`；fusion Linear 输入 6×hidden |
| 29 类 Ordinal head | `OrdinalLogisticHead(n_classes=9)` | `n_classes=29`，thresholds 重新初始化 |
| 置信度加权 rank loss | `OrdinalLoss`（无权重） | 增加 `weight` 参数 |
| turn_b/turn_w 传递 | `DualViewSetTransformer.forward` 无 turn | `_split_by_dim` 同时返回 turn arrays；向下传至 `CrossMAB` |
| `input_dim` 动态化 | `train_kata_native.py` 硬编码 `256` | 从 KAB2 header 读取后传入 |
| `DualRatingHead` | 已实现 | 直接使用 |

---

## 10. 数据流完整示意

```
KAB2 文件对
  game_XXXX_B.npz  →  moves_B (N_b, 1034)  +  summary_B
  game_XXXX_W.npz  →  moves_W (N_w, 1034)  +  summary_W

标签提取:
  target_b = normalize(summary_B[2])    ← meanLogPrior_B
  target_w = normalize(summary_W[2])    ← meanLogPrior_W
  rank_b   = int(summary_B[10])        ← humanRankIdx_B  (-1 = mask)
  rank_w   = int(summary_W[10])        ← humanRankIdx_W
  conf_b   = summary_B[11]             ← humanLogPrior_B (confidence)
  conf_w   = summary_W[11]             ← humanLogPrior_W

模型输入:
  x      = cat([moves_B, moves_W])      (N_b+N_w, 1034)
  xlens  = [N_b + N_w]
  turn_b = moves_B[:, 8]               (N_b,) ← scalar[8]
  turn_w = moves_W[:, 8]               (N_w,)

前向：
  x → projection → split(dim=7)
    → encoder_b / encoder_w
    → causal CrossMAB (mask_bw, mask_wb from turn arrays)
    → SegmentedAttentionPool × 2
    → fusion
    → z (128,)
    → DualRatingHead     → (b_rating, w_rating)
    → OrdinalRankHead_B  → rank_probs_B (29,)
    → OrdinalRankHead_W  → rank_probs_W (29,)

损失:
  L = 1.0 × [MSE(b_rating, target_b) + MSE(w_rating, target_w)]
    + 2.0 × BradleyTerry(b_rating, w_rating, target_b > target_w)
    + 0.1 × [conf_weighted_CE(rank_probs_B, rank_b)
            + conf_weighted_CE(rank_probs_W, rank_w)]
    + 1e-5 × L2
```
