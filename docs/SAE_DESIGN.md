# KataRank SAE Design — Capture, Features, Labeling

**Version**: 0.1 (scaffolding)
**Date**: 2026-06-11
**Status**: 接口框架已落地；SAE 训练循环**延后**，归入未来的 ReviewWorkflow
**Related**: `MODEL_V3_ARCHITECTURE.md`, `src/katarank/model/interpret.py`, `src/katarank/model/sae.py`

---

## 1. 背景与目标

复盘（game review）场景需要回答的不是「这局棋黑棋几段」，而是「**这一手**让模型产生了什么判断」。
KataRankModel 的架构天然支持这件事：

- 模型的 **token 就是一手棋**（`x` 为 `(N_total, input_dim)` packed move features）；
- 残差流在 pooling 之前一直保持 token 粒度 → 每手棋都有一个 `hidden_dim` 维的内部表示；
- SAE（Sparse Autoencoder）把这个稠密表示分解为**稀疏、可命名的特征字典**，
  人工只需给少量高频特征起名（如 "overplay", "joseki deviation"），
  复盘界面即可逐手显示「本手触发了哪些已命名特征」。

本设计的范围：

| 范围内（本次落地） | 延后（未来 ReviewWorkflow） |
|---|---|
| SAE 模块本体（encode/decode/loss/save/load） | SAE 训练循环（优化器、调度、dead-feature 重采样） |
| 激活语料采集接口 `collect_sae_corpus` | 大规模语料管理（分片落盘、shuffle buffer） |
| 复盘期逐手特征提取 `FeatureExtractor` | REST 端点 `/review/features`、`/features/{id}/label` |
| 特征标签注册表 `FeatureRegistry`（JSON） | 跨语料 top-activating-moves 检索（辅助人工命名） |
| `ActivationCapture` 增加 accumulate 模式 | SAE 质量评估（recon loss / L0 / feature 死亡率看板） |

---

## 2. 原理

### 2.1 为什么是 SAE

`hidden_dim`（默认 128）维的残差流处于**叠加态**（superposition）：单个神经元
同时参与多个概念，直接读单维激活不可解释。SAE 做的是**过完备字典学习**：

```
f  = ReLU( W_enc · (x − b_dec) + b_enc )      # (d_model,) → (d_features,)  编码，稀疏
x̂ = W_dec · f + b_dec                         # (d_features,) → (d_model,)  重建
```

- `d_features = d_model × expansion`（默认 expansion=8，即 128 → 1024 个候选特征）；
- 稀疏约束使每个 token 只有少数特征激活，每个特征趋向单一语义（monosemantic）；
- `W_dec` 的每一**行**是一个字典原子（feature direction），训练中保持单位范数，
  防止用缩放 `W_dec` 规避 L1 惩罚。

### 2.2 稀疏化的两种模式（实现均支持）

| 模式 | 机制 | 损失函数 | 取舍 |
|---|---|---|---|
| **L1**（k=None） | ReLU 后靠 L1 惩罚自然稀疏 | `MSE + λ·‖f‖₁` | 经典；λ 需调参，存在 shrinkage 偏差 |
| **Top-k**（k=int） | ReLU 后每 token 只保留最大的 k 个激活 | `MSE`（稀疏度由 k 硬保证） | 免调 λ，L0 精确可控；推理路径与训练一致 |

复盘场景推荐 **top-k**：每手棋恰好报告 k 个特征，输出粒度稳定，便于 UI 呈现。

### 2.3 token ↔ 手数对齐

`DualViewSetTransformer._split` 按 `scalar[7]=isWhite` 把 packed 输入切成 B/W 两条流，
流内顺序保持原局顺序，并假设**严格交替、黑先**（与 causal mask 的 turn 推导同一假设）：

```
Black 流位置 i  →  全局手数 2i + 1   (1-based: 第1、3、5…手)
White 流位置 j  →  全局手数 2j + 2   (第2、4、6…手)
```

因此对任一 capture site 的激活矩阵，行号即流内位置，可无损映射回手数。

> **已知限制**：让子棋/非交替棋谱会打破该假设，move_no 将出现偏移。
> 修正方案（延后）：从 KAB2 的 turnNumber scalar（`scalar[8]`）读真实手数，
> 替代位置推导。当前 `FeatureExtractor` 与 dual_view 保持同一假设，不另立标准。

---

## 3. 架构与数据流

### 3.1 组件图

```
                       ┌──────────────────────────────────────────────┐
 SGF ──► KataGoEngine ─►  x (N, input_dim)                            │
                       │       │                                      │
                       │  KataRankModel.forward          推理结果照常输出
                       │       │            ▲                         │
                       │  ActivationCapture (forward hooks, 旁路只读)  │
                       │       │                                      │
                       │  activations: {site: (N_tokens, hidden_dim)} │
                       └───────┼──────────────────────────────────────┘
                               │
            ┌──────────────────┴───────────────────┐
            │ 训练路径（未来 ReviewWorkflow）          │ 复盘路径（本次落地）
            ▼                                      ▼
   collect_sae_corpus                      FeatureExtractor
   多局激活拼成训练矩阵                        单局激活 → SAE.encode → top-k
            │                                      │
            ▼                                      ▼
   SAE 训练（延后）── checkpoint ──►        per-move features
                                           [{move_no, color, feature_ids,
                                             activations, labels}, …]
                                                   │
                                          FeatureRegistry (JSON)
                                          feature_id → 人工标签
```

### 3.2 Capture sites

`ActivationCapture` 对两类模块挂 forward hook：

| Site（named_modules 前缀） | 模块 | 内容 | 适用 |
|---|---|---|---|
| `encoder.encoder_b.*` / `encoder.encoder_w.*` | ISAB 内 `MultiHeadAttentionBlock` | 流内自注意力后的 token 表示 | 语料量大（每个 ISAB 两次 MAB），适合 SAE 训练基底 |
| `encoder.cross_attn.blocks.{i}.mab_bw` / `.mab_wb` | `CausalMAB` | **看过对手上下文后**的 token 表示 | 语义最丰富，**复盘提取的默认 site** |

默认 site 取**最后一个** cross block 的 `mab_bw`（黑流）与 `mab_wb`（白流）——
此处的 token 已融合自身风格与对局互动，最接近"模型对这手棋的最终看法"。

### 3.3 (site, mode) 维度

`lite` 与 `full` 引擎模式产生不同的输入特征分布，残差流统计量随之不同。
**SAE 必须按 (capture site, engine mode) 二元组分别训练与保存**，checkpoint
命名约定：

```
sae_{site_short}_{mode}.pt        e.g.  sae_xattn_b_lite.pt
```

加载错配的 SAE 不报错但特征无意义，故 checkpoint 内嵌 `config` 供调用方核对
（`d_model` 不符会在 encode 时立即 shape error，作为最后防线）。

### 3.4 两条数据流程

**A. 语料采集（为未来训练准备）**

```python
corpus = collect_sae_corpus(
    model,
    batches=((s['x'], [s['seq_len']]) for s in samples),   # 任意 (x, xlens) 迭代器
    sites=['encoder.cross_attn.blocks.0.mab_bw'],
    max_tokens=2_000_000,
)
# {site: (n_tokens, hidden_dim) cpu tensor} → 落盘 .npy，交给未来的训练脚本
```

**B. 复盘提取（现在可用，SAE 可先用随机权重联调接口）**

```python
sae = SparseAutoencoder.load('sae_xattn_b_lite.pt')
fx  = FeatureExtractor(model, sae_b=sae, sae_w=sae_w,
                       registry=FeatureRegistry('features.json'))
moves = fx.extract(x, top_k=8)
# [{'move_no': 1, 'color': 'B', 'stream_pos': 0,
#   'feature_ids': [412, 7, …], 'activations': [3.1, 2.4, …],
#   'labels': ['overplay', None, …]}, …]   按 move_no 升序
```

---

## 4. 实现技术说明

### 4.1 模块清单（`src/katarank/model/sae.py`）

| 对象 | 签名要点 | 说明 |
|---|---|---|
| `SparseAutoencoder(d_model, expansion=8, k=None, l1_coeff=1e-3)` | `encode(x)→f`, `decode(f)→x̂`, `forward(x)→{'recon','features'}`, `loss(x,out)→{'total','mse','l1','l0'}` | k=None 为 L1 模式，k=int 为 top-k 模式；`normalize_decoder_()` 供训练循环每步调用 |
| `FeatureExtractor(model, sae_b, sae_w=None, site_b=None, site_w=None, registry=None)` | `extract(x, top_k=8) → List[MoveFeature]` | site 缺省自动解析为最后一个 cross block；sae_w 缺省复用 sae_b（联调用，正式需分流训练） |
| `MoveFeature` (TypedDict) | `move_no, color, stream_pos, feature_ids, activations, labels` | 复盘 API 的返回单元，可直接 JSON 化 |
| `FeatureRegistry(path)` | `label(fid, label, author, notes)`, `get(fid)`, `all()` | JSON 落盘，写入即保存（tmp+replace 原子写） |
| `collect_sae_corpus(model, batches, sites, max_tokens=0)` | → `{site: (n_tokens, d) tensor}` | 内部用 accumulate 模式 capture |

### 4.2 `ActivationCapture` 的变更（`interpret.py`）

新增 `accumulate: bool = False`：

- 默认 False，行为与既有完全一致（同名 site 后写覆盖前写）；
- True 时同名 site 的激活沿 token 维 `torch.cat` 累积。

**动机**：`CrossMAB.forward` 对 batch 内**每局单独**调用 `CausalMAB`，即一次
forward 中同一 hook 触发多次。覆盖语义下 batch>1 时 cross site 只剩最后一局
——语料采集必须 accumulate。`attention_maps` 因各局形状不同不参与累积（保持
末次覆盖），已在 docstring 注明。

### 4.3 SAE checkpoint 格式

与 `KataRankModel.save` 同一约定，保证 `weights_only=True` 可加载：

```python
{'version': '0.1', 'type': 'SparseAutoencoder',
 'config': {...}, 'model_state': state_dict}
```

### 4.4 Registry JSON 格式

```json
{ "412": {"label": "overplay", "author": "bzhao",
           "notes": "fires on low-prior aggressive moves", "updated": "2026-06-11"} }
```

key 为 feature id（字符串化整数）。并发写不设锁——标注是单人低频操作，
原子替换已足够（韩非：例外已注明原因）。

### 4.5 约束与边界

1. `FeatureExtractor.extract` **仅接受单局**（`x` 为 2-D，内部 `xlens=[N]`）。
   复盘本就逐局进行；多局批量请在外层循环。
2. capture 旁路为只读 hook + detach to CPU：不改变模型输出，单局内存开销
   ≈ `N_moves × hidden_dim × 4B × site数`，可忽略；延迟开销主要是 D2H 拷贝。
3. 空流（如只有黑方落子的残局片段）：该流无激活，extract 自然跳过，不报错。
4. SAE 未训练前 `FeatureExtractor` 即可跑通（随机特征），用于接口联调与
   端到端测试；特征**语义**以训练后为准。

---

## 5. 后续路线（不在本次范围）

1. **ReviewWorkflow**（`workflow.py` 第三个 workflow）：语料采集 → SAE 训练
   （Adam、decoder 归一化、dead feature 重采样）→ 质量评估 → checkpoint 发布。
2. **API 端点**：`POST /review/features`（SGF → per-move features + labels）、
   `POST /features/{id}/label`、`GET /features/{id}/top-moves`。
3. **真实手数对齐**：从 `scalar[8]` turnNumber 读取，替代交替假设（解决让子棋）。

---

## 6. 验收标准

`tests/test_model.py` 新增（全部可离线运行，无需 KataGo / 已训练 SAE）：

- SAE forward / encode / decode 形状正确；top-k 模式下每 token L0 ≤ k；
- save/load 往返后输出逐位一致；
- `FeatureExtractor.extract` 在合成单局上返回全部 N 手、move_no 严格为 1..N、
  B/W 颜色与 isWhite 列一致；
- `FeatureRegistry` label → get 往返、落盘重载一致；
- `ActivationCapture(accumulate=True)` 两次 forward 后 token 数翻倍。
