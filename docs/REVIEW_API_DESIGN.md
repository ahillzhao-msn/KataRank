# KataRank Review API Design — Per-Move Output for Game Review

**Version**: 0.1
**Date**: 2026-06-11
**Status**: design approved → implementation in this change
**Related**: `SAE_DESIGN.md`（同一端点家族的未来扩展）, `MODEL_V3_ARCHITECTURE.md`,
gopredict integration（消费方契约见 §5）

---

## 1. 背景与定位

KataRank 已确定为 **gopredict 的唯一后台分析引擎**。现有 `/rank/*` 端点只输出
整局结论（rating / rank / confidence），而 gopredict 的复盘功能（`MoveAnalysis`
表、blunder 标记、逐手胜率曲线）需要**逐手**数据。

本设计新增 `/review/*` 端点家族：

- **现在**：从 KAB2 per-move scalars 直接导出逐手指标（零引擎改动——数据本就在流里）；
- **将来**：同一响应结构追加 SAE 特征字段（`feature_ids` / `labels`，见
  `SAE_DESIGN.md` §3.4-B），端点不另起炉灶。

设计原则（道）：review 是 rank 的**超集视图**——同一次引擎分析、同一条流，
多暴露 token 级数据，不增加第二次 KataGo 运行。

## 2. 数据原理

### 2.1 KAB2 scalar 真实布局（以 batch_analysis.cpp 为准）

> ⚠️ 勘误：`MODEL_V3_ARCHITECTURE.md` §2.1 此前记载的 scalar 表
> （winRate/scoreLead/complexity/policyEntropy/turnNumber/boardArea…）与
> 实现不符，本次一并修正。权威来源：`katago-fork/cpp/command/batch_analysis.cpp`
> `appendMoveRecord()`。

每手 10 维 scalars，**全部白方视角**：

| # | 字段 | 说明 |
|---|------|------|
| 0 | `whiteWinProb` | 该手局面下白方胜率 |
| 1 | `whiteLossProb` | 白方败率（≈黑方胜率；三值含 noResult） |
| 2 | `whiteNoResultProb` | 无结果概率 |
| 3 | `whiteScoreMean / 50` | 白方期望领先目数（÷50 归一化） |
| 4 | `shorttermScoreError / 10` | 短期目数误差（局面不确定度/复杂度代理） |
| 5 | `policyPrior` | 实际落点的策略先验 |
| 6 | `policyRank / 361` | 实际落点在策略中的排名（0 = 引擎首选） |
| 7 | `isWhite` | 0=黑 1=白 |
| 8 | `winDelta` | `whiteWinProb[t+1] − whiteWinProb[t]`（本手造成的胜率变化） |
| 9 | `scoreDelta / 50` | `whiteScoreMean[t+1] − whiteScoreMean[t]`（目数变化） |

`full` 模式行尾追加 `pick(C) + avgTrunk(C)`；**前 10 列在两种模式下相同**，
故 review 在 lite 模式即可全功能工作。

### 2.2 视角归一化（white → mover）

复盘语义要求"这手棋对**落子方**好不好"。转换规则（黑方取反）：

```
sign        = +1 if color == 'W' else −1
winrate     = scalar[0] if W else scalar[1]      # mover 胜率
score_lead  = sign × scalar[3] × 50              # mover 领先目数
win_delta   = sign × scalar[8]                   # >0 = 本手提升 mover 胜率
score_delta = sign × scalar[9] × 50              # >0 = 本手为 mover 赢目
```

消费方派生指标（gopredict 侧，不在本 API 内做）：
`points_lost = max(0, −score_delta)`；`is_blunder = points_lost > 阈值`；
`is_top1 = (policy_rank == 0)`；`is_top5 = (policy_rank < 5)`。

### 2.3 手数对齐

KAB2 不携带绝对手数；B/W 流内行序即落子序。沿用 `dual_view` 的严格交替假设：

```
B 流第 i 行 → 全局第 2i+1 手；W 流第 j 行 → 全局第 2j+2 手（1-based）
```

> 限制（与 SAE_DESIGN.md §2.3 同源）：让子棋会偏移。修正属未来工作
> （batch_analysis 输出真实 turn number），两处文档同时记账。

## 3. API 设计

### 3.1 端点

| 端点 | 输入 | 说明 |
|---|---|---|
| `POST /review/string` | `{sgf, mode?, min_moves?}` | 单局 SGF 字符串 → ReviewOutput |
| `POST /review/file` | `{path, mode?, min_moves?}` | 单局文件路径（受 `--sgf-root` 白名单约束） |
| `POST /review/batch` | `{items, item_type?, mode?, min_moves?}` | 批量（gopredict 回溯全库用） |

请求 schema 与 `/rank/*` 完全同形（mode 默认 `lite`，min_moves 默认 10），
并发控制走同一 `engine_sem`。

### 3.2 响应：ReviewOutput = KAB2Output + moves

```jsonc
{
  // ── KAB2Output 全部字段（整局结论，与 /rank/* 一致）──
  "game_id": "game001",
  "metadata": { "PB": "...", "PW": "...", ... },
  "b_rating": -2.31, "w_rating": -2.05,
  "b_rank": 18, "w_rank": 20,
  "b_confidence": 0.41, "w_confidence": 0.38,
  "b_rank_probs": [...], "w_rank_probs": [...],

  // ── 新增：逐手记录，按 move_no 升序 ──
  "moves": [
    { "move_no": 1, "color": "B",
      "winrate": 0.48,        // mover 视角
      "score_lead": -0.7,     // mover 视角，目
      "score_stdev": 13.2,    // shorttermScoreError，局面复杂度代理
      "policy_prior": 0.21,
      "policy_rank": 3,       // 0 = 引擎首选
      "win_delta": -0.01,     // mover 视角
      "score_delta": -0.4 },  // mover 视角，目
    ...
  ]
}
```

预留（本次不实现，字段名已定）：每个 move 对象将来追加
`"features": [{"id": 412, "activation": 3.1, "label": "overplay"}]`（SAE）。

### 3.3 Schema 契约：OpenAPI，而非引库

消费方（gopredict 等）**不引用 katarank Python 库**（会拖入 torch 依赖树），
也不提供自建 `/schema` 端点。所有端点声明 Pydantic `response_model`
（`KAB2OutputModel` / `MoveRecordModel` / `ReviewOutputModel`，与 schema.py
的 TypedDict 镜像），**`GET /openapi.json` 即权威契约**——机器可读、随服务
版本走。消费方可在 CI 里拉取比对关键字段做契约测试。

错误语义随之收紧：空结果不再返回 `{'error': ...}` 200，改为 HTTP 422
（too few moves）/ 404（文件或目录缺失），响应体恒符合声明 schema。

### 3.4 与 /rank 的关系

`/rank/*` 不变（gopredict 聚合只要整局结论时用，响应小）。
`/review/*` 同一次流式分析多回传 `moves` 数组——一局 250 手 ≈ 250 × 8 字段
JSON，约 30 KB，可接受；不做分页。

## 4. 实现技术说明

| 改动点 | 内容 |
|---|---|
| `schema.py` | 新增 `MoveRecord`、`ReviewOutput` TypedDict（`ReviewOutput` = KAB2Output 字段 + `moves`） |
| `workflow.py` | `run_review_files()` / `run_review_strings()`：单次 `engine.stream_games()`，按 `game_id` 聚合 B/W 帧 → `_move_records()` 做视角归一化与手数对齐；整局结论复用现有逻辑——有 checkpoint 走模型推理（`kab2_make_sample` + forward），无则 `meanLogPrior` 启发式（与 `/rank` 等价） |
| `api/server.py` | 三个 `/review/*` 路由，纯薄壳（与 `/rank/*` 同模式：`def` + worker 线程 + `engine_sem` + `_check_path`） |
| 元数据 | 复用 `_attach_metadata_from_files/strings` |

**关键不变量（韩非：可验证）**

1. 单次引擎分析：review 不比 rank 多跑 KataGo；
2. `moves` 长度 = `N_b + N_w`，`move_no` 严格为 `1..N` 无空洞（交替假设下）；
3. 视角符号：同一手在 B/W 流的 `win_delta` 与白方原始值关系正确（黑取反）；
4. lite/full 两种模式输出相同的 10 个 scalar 派生字段。

测试：合成 B/W 移动矩阵直测 `_move_records`（符号、排序、字段域）；
stub engine（仅 `stream_games` 方法）端到端测 `run_review_files`，无需 KataGo。

## 5. gopredict 消费契约（前瞻，实现属 gopredict 侧任务）

| ReviewOutput 字段 | gopredict 落库 |
|---|---|
| `b_rating`/`b_rank`/`b_confidence`（及 w_*） | `GameFeature.vector` 替代物 / `PlayerProfile` 聚合输入 |
| `moves[].winrate / score_lead` | `MoveAnalysis.winrate / score_lead` |
| `moves[].score_delta` | `MoveAnalysis.points_lost = max(0, −score_delta)`；`is_blunder` 由 gopredict 阈值判定 |
| `moves[].policy_rank / policy_prior` | top1/top5 命中、`MoveAnalysis.policy_prior` |
| `moves[].score_stdev` | `MoveAnalysis.complexity` |
| `moves[].move_no / color` | `MoveAnalysis.move_number / player` |
| —（无候选着法） | `MoveAnalysis.top_moves` 置空——KAB2 不携带 top_moves，功能降级已确认接受 |
| —（无落点坐标） | `MoveAnalysis.board_position` 由 gopredict 从自家 SGF 解析回填（它已有 SGFParser） |

调用方式：gopredict 容器 → `http://host.docker.internal:8765/review/string`
（katarank 留守 Windows 宿主机，Linux 支持未做——已确认）。

## 6. 范围外（记录在案）

- SAE 特征字段注入（等 SAE 训练落地，见 SAE_DESIGN.md §5）；
- top_moves 候选列表（需改 batch_analysis.cpp，功能降级已被接受）；
- 真实手数（让子棋）修正；
- 分页/流式 HTTP 响应。
