# KataGo Trunk Feature Extraction — Customization Guide

This document describes the customizations made to vanilla [lightvector/KataGo](https://github.com/lightvector/KataGo) for trunk feature extraction. Engine intrusion vs upstream (2026-10): `nninputs.h/.cpp` +18, `nneval.cpp/.h` +7, `openclbackend.cpp` +46, plus the self-contained `command/batch_analysis.cpp` and its registration.

> **History (2026-10).** From fork commit 15a5bced (2026-06-09) until KataGo#1, every KAB2 trunk/pick
> column was zero: `includeTrunk` was not carried from the request to the backend, and the OpenCL readback
> ignored the FP16 tensor-core padding (`paddedNNXYLen` 368, not 361), scrambling channels across batch rows.
> Any KAB2 file written before KataGo#1 has an all-zero trunk.

## What Was Added

### 1. Trunk/Pick Output in NNOutput (`nninputs.h`, +4 lines)

```cpp
bool includeTrunk = false;    // request trunk features from backend
bool includePick = false;     // request pick features from backend
float* trunk = NULL;          // trunk: trunkNumChannels × nnXLen × nnYLen
float* pick = NULL;           // pick: trunkNumChannels (at move position)
```

These fields are initialized to `false`/`NULL`. A caller requests trunk per evaluation with `NNResultBuf::includeTrunk`; `nneval.cpp` copies that flag onto the output it hands the backend, and the backend only reads trunk back for rows that asked — no cost for normal KataGo usage.

### 2. OpenCL Trunk Buffer Read (`openclbackend.cpp`, +20 lines)

In `NeuralNet::getOutput()`, after the GPU forward pass:

- `InputBuffers` gains a `trunkResults` host buffer
- After policy/value results are read from GPU, the trunk buffer is read to host only if some row requested it
- The readback strips the per-row padding (`paddedNNXYLen`, e.g. 368 with FP16 tensor cores) before
  copying each row to `output->trunk` with symmetry correction

Other backends (CUDA, Eigen) follow the same pattern but weren't modified since OpenCL is the target deployment backend.

### 3. `batch_analysis` Command (`command/batch_analysis.cpp`, new file)

A self-contained module (~900 lines) that:

- Accepts `-list games.csv` or `-sgf-dir ./sgfs/`
- For each SGF: loads game, extracts main line, skips < `-min-moves` moves
- Replays each game into position snapshots and evaluates them concurrently
  (`-threads`, default 16; results folded in move order, output identical to 1 thread);
  collects scalars(10) + pick(C) + avgTrunk(C) features (KAB2 format)
- The NN cache is disabled: every position is evaluated once, and NNEvaluator would
  otherwise retain every trunk-carrying result (~0.74 MB each)
- Optional HumanSL pass (`-human-model`): the rank anchor v3 — per player, moves whose
  main-model point loss is in that player's 10–90% band vote; HumanSL log-likelihood under
  7 coarse profiles (20k, 15k, 10k, 5k, 1d, 6d, 9d), then the skipped profiles within ±2 of
  the coarse estimate → natural spline → posterior over 29 profiles →
  `humanRankExp`/`humanRankSd` (summary[12]/[13]), rounded into `humanRankIdx` (summary[10])
- File mode: one combined compressed `<stem>.npz` per game + `_meta.csv`
- Stream mode (`-stream`): per-player frames on stdout tagged with the
  game id; `-no-trunk` for scalars-only lite output
- Daemon mode (`-daemon`): persistent process, jobs fed via stdin
  (`reset` clears NN caches, `quit` exits) — models loaded once
- Handles errors per-file and continues

### 4. Command Registration (`main.h/cpp`, +3 lines)

- `int batch_analysis(const std::vector<std::string>& subArgs);` in `MainCmds` namespace
- Registered in `main.cpp` with help text and command handler

### 5. CMakeLists.txt (+1 line)

Added `command/batch_analysis.cpp` to the `COMMAND_SOURCES` list.

## Upgrade Strategy

When upgrading to a newer KataGo version:

```
1. Port these from old fork to new:
   ├── neuralnet/nninputs.h/.cpp   — trunk/pick fields + ownership in operator=/dtor
   ├── neuralnet/nneval.cpp/.h     — includeTrunk on NNResultBuf → output; trunk channel getter
   ├── neuralnet/openclbackend.cpp — padded, on-request trunk readback
   ├── command/batch_analysis.cpp  [new file] — the module
   └── CMakeLists.txt              [+1 line] — add to build

2. Update main.h and main.cpp:
   ├── main.h   [+1 line]  — add function declaration
   └── main.cpp [+2 lines] — add help text + command handler
```

The engine-side merge is ~70 lines. Conflicts are unlikely because:
- `nninputs.h` additions are at the end of the struct, near other bool fields
- `openclbackend.cpp` additions are in the batch loop, a stable function
- `batch_analysis.cpp` is a self-contained new file
- `main.h/cpp` additions are at the end of existing lists

## Output Format

### KAB2 combined file (file mode)

```
<sgf-stem>.npz = [4B B_size][B KAB2 payload][4B W_size][W KAB2 payload]
KAB2 payload:    96-byte header (magic 'KAB2', dims, flags, PlayerSummary)
                 + numMoves × (scalars[10] + pick[C] + avgTrunk[C]) float32
                 (zlib compressed in file mode)
```

### Stream protocol (stream/daemon mode)

```
[1B side 'B'/'W'][4B idLen][game id][4B size][KAB2 payload]   per player
0x01 = end of daemon job        0x00 = end of stream / daemon exit
```

See the KataRank README for the full KAB2 header layout.

### CSV Metadata

`_meta.csv` accompanies each file-mode batch run with game-level metadata
(player names, move counts, PlayerSummary fields, HumanSL ranks). Stream
and daemon modes write nothing to disk.

## References

- Original KataGo: [lightvector/KataGo](https://github.com/lightvector/KataGo)
- Strength model concept: [Animiral/go-strength-model](https://github.com/Animiral/go-strength-model)
- Preprocessing patterns: [ahillzhao-msn/go-analyzer](https://github.com/ahillzhao-msn/go-analyzer)
- This fork: [ahillzhao-msn/KataGo](https://github.com/ahillzhao-msn/KataGo)
