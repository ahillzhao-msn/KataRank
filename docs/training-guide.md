# KataRank 训练配置指南

> 对应 Phase 1 → Phase 2 的经验固化。
> 每次训练迭代更新此文档，记录配置变更与效果。

---

## 一、核心原则

### 1.1 验证集质量决定训练质量

验证集必须是**分层随机抽样**，而非简单随机分割。简单随机分割在长尾分布的数据集中容易导致：

| 问题 | 后果 | 例 |
|------|------|-----|
| 高段位样本被排挤出验证集 | 高段位评估不可靠 | Phase 2: 7d-9d 验证集=0 |
| 低段位样本被排挤出验证集 | 低段位评估不可靠 | Phase 2: 20k 验证集=0 |
| 验证 loss 偏向中段位 | 无法反映全段位质量 | — |

**结论：** 验证集应该按段位 band 分层抽样，保证每个 band 都有一定比例进入验证集。

### 1.2 训练集分布应符合自然态

围棋段位分布是**长尾分布**（中段位数万局，高段位数百局）。StratifiedRankSampler 解决的是 batch 内的段位覆盖，不是数据分布本身。

| 手段 | 解决什么 | 启用条件 |
|------|---------|---------|
| StratifiedRankSampler | 每个 batch 覆盖 5 个段位 band | 任何训练 |
| 数据补采 | 补充稀疏段位的样本 | 验证集某 band < 5% 时 |

### 1.3 超参随数据量自适应

| 超参 | 规则 | Phase 1 (5K) | Phase 2 (9K) | Phase 3 (15K) |
|------|------|-------------|-------------|--------------|
| dropout | 数据越多，dropout 越低 | 0.2 | 0.15-0.2 | 0.1-0.15 |
| weight_decay | 数据越多，正则越弱 | 1e-4 | 1e-4 | 5e-5 |
| batch_size | 稳定超过 32 | 32 | 32 | 32-64 |
| warmup_epochs | 固定 5 | 5 | 5 | 5 |
| patience | 数据越多，耐心越大 | 30 | 30 | 40 |
| learning_rate | 固定 5e-4 | 5e-4 | 5e-4 | 5e-4 |

---

## 二、config 配置文件集

### 2.1 基础配置（base.yaml）

所有训练的公共底数：

```yaml
# base.yaml — 跨迭代不变的参数
model:
  num_rank_classes: 29
  num_heads: 4
  num_inducing: 16
  encoder_depth: 2
  cross_depth: 1

training:
  lr_min: 0.00001
  warmup_epochs: 5
  gradient_clip: 1.0
  num_workers: 0
  stratified: true
  n_bands: 5
  max_moves: 400
  min_moves: 5

  loss_weights:
    rating_mse: 1.0
    bradley_terry: 0.5
    rank_anchor: 0.3

  device: "auto"
```

### 2.2 规模自适应配置（phase2.yaml 示例）

```yaml
# phase2.yaml — 继承 base，按数据量覆盖
_base_: "base.yaml"
model:
  hidden_dim: 128
  dropout: 0.2

training:
  data_dir: "data/kab2"
  batch_size: 32
  epochs: 150
  learning_rate: 0.0005
  weight_decay: 0.0001
  patience: 30
```

---

## 三、训练流程

### 3.1 前置：验证集分层检验

训练前必须执行：

```bash
uv run python scripts/validate_split.py --meta data/kab2/_meta.csv
```

检验标准：

| 条件 | 通过 | 不通过 |
|------|------|--------|
| 每个段位 band 在验证集中至少 5 样本 | ✅ | ❌ 需重新分割 |
| 验证集占总样本 5-10% | ✅ | ❌ 需调整 |
| 验证集的段位分布与训练集正相关 (corr>0.9) | ✅ | ❌ 需调查 |

### 3.2 训练命令

```bash
# 从头训练
uv run katarank-train --config src/katarank/train/config_phase2.yaml

# 继续训练（resume）
uv run katarank-train --config src/katarank/train/config_phase2.yaml \
  --resume nets/katarank/best.pt

# 覆盖超参
uv run katarank-train --config config.yaml \
  --epochs 200 --lr 0.0003 --batch-size 64
```

### 3.3 训练后

```bash
# 生成评估报告
uv run python scripts/evaluate.py --checkpoint nets/katarank/best.pt

# 写入训练日志
# 手动更新 docs/training_log.md
```

---

## 四、验证集分割策略

### 4.1 核心原则

验证集采用**三段式分割策略**，按段位样本量分档处理：

| 全局样本 | 策略 | 理由 |
|---------|------|------|
| ≥ 20 | 严格分层，每个段位分 5-10% 到验证集 | 样本足够，能够做有意义的独立评估 |
| 3~19 | 分 1 个到验证集 | 保有覆盖面，但不足以做统计评估 |
| < 3 | 全部用于训练 | 分割会严重削弱训练集 |

主流段位（2k ~ 6d，占总量 ~90%）必须通过分层验证。
两端稀疏段位（20k-3k、7d-9d，占 ~10%）按上述三档处理，不强制验证集存在。

```
段位分布（自然态）：
                    ██
                  ██████
                 █████████
               █████████████
            ███████████████████
  ── 20k-3k ──┼── 2k-6d ──┼─── 7d-9d ──
  样本占比 ~5%    样本占比 ~90%    样本占比 ~5%
  三档处理         严格分层          三档处理
```

理由：
1. 两端样本本来就少，再分割会进一步削弱训练集中极端段的表示
2. 两端段的评估置信度低（样本量决定置信度），独立验证集的收益微乎其微
3. 验证 loss 主要由主流段位驱动，两端段的贡献可以忽略

### 4.2 分层算法

```python
def stratified_split(df, val_ratio=0.1, main_bands=[1, 2, 3]):
    """按段位 band 分层分割训练/验证集。
    
    两端 band（稀疏段位）不拆分，全部用于训练。
    main_bands 中的 band 执行严格分层随机分割。
    """
    df = df.copy()
    df['band'] = pd.cut(df['human_rank_idx'], bins=5, labels=False)
    
    train_list, val_list = [], []
    for band in range(5):
        band_df = df[df['band'] == band]
        if band in main_bands:
            # 主流段位：分层分割
            val_count = max(1, int(len(band_df) * val_ratio))
            val_idx = band_df.sample(n=val_count, random_state=42).index
            train_list.append(band_df.drop(val_idx))
            val_list.append(band_df.loc[val_idx])
        else:
            # 两端段位：全部用于训练
            train_list.append(band_df)
    
    return pd.concat(train_list), pd.concat(val_list)
```

---

## 五、实验记录模板

每次训练后更新 `docs/training_log.md`，追加新的实验记录：

```markdown
### Phase N — 日期

**配置变更：**
| 参数 | 旧值 | 新值 | 理由 |
|------|------|------|------|
| ... | ... | ... | ... |

**数据：**
- 训练集: N 局
- 验证集: N 局（分层抽样）
- 总 KAB2 缓存: N 个 npz

**结果：**
| 指标 | Phase N-1 | Phase N | 改善 |
|------|----------|---------|------|
| val_loss | N.N | N.N | ±X% |
| rank_mae | N.N | N.N | ±X% |
| rank_acc | N% | N% | ±X% |
| rank_acc_pm1 | N% | N% | ±X% |
| rating_corr | N.N | N.N | ±X% |

**分析：**
- ...
```

---

## 六、已知局限

1. **分段位评估置信度不同** — 主流段位（样本充足）评估可信，两端段位（样本稀疏）评估仅作参考
2. **_meta.csv 是派生视图** — `auto_train.py` 训练前自动从 .npz 重建（`rebuild_meta`），无需手动维护
3. **20k, 19k-15k 段位无标签** — 数据中不存在这些段位的棋谱，模型无法学会预测
4. **训练日志手动维护** — 可以考虑自动化记录
