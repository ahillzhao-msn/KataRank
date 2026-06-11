# KataGo Trunk Feature Extraction — Customization Guide

This document describes the customizations made to vanilla [lightvector/KataGo](https://github.com/lightvector/KataGo) for trunk feature extraction. Total intrusion: **28 lines across 5 files**.

## What Was Added

### 1. Trunk/Pick Output in NNOutput (`nninputs.h`, +4 lines)

```cpp
bool includeTrunk = false;    // request trunk features from backend
bool includePick = false;     // request pick features from backend
float* trunk = NULL;          // trunk: trunkNumChannels × nnXLen × nnYLen
float* pick = NULL;           // pick: trunkNumChannels (at move position)
```

These fields are initialized to `false`/`NULL` — the neural net backend will only compute trunk if explicitly requested, so there is zero performance impact on normal Katago usage.

### 2. OpenCL Trunk Buffer Read (`openclbackend.cpp`, +20 lines)

In `NeuralNet::getOutput()`, after the GPU forward pass:

- `InputBuffers` gains a `trunkResults` host buffer
- After policy/value results are read from GPU, a `clEnqueueReadBuffer` copies `buffers->trunk` to host
- In the batch loop, trunk is copied to `output->trunk`/`output->pick` with symmetry correction

Other backends (CUDA, Eigen) follow the same pattern but weren't modified since OpenCL is the target deployment backend.

### 3. `batch_analysis` Command (`command/batch_analysis.cpp`, new file)

A self-contained module (~900 lines) that:

- Accepts `-list games.csv` or `-sgf-dir ./sgfs/`
- For each SGF: loads game, extracts main line, skips < `-min-moves` moves
- For each position: evaluates with NNEvaluator, collects
  scalars(10) + pick(C) + avgTrunk(C) features (KAB2 format)
- Optional HumanSL second pass (`-human-model`) annotating each player
  with a rank estimate (20k–9d) and confidence
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
1. Copy these 4 files from old fork to new:
   ├── neuralnet/nninputs.h        [+4 lines] — trunk/pick fields
   ├── neuralnet/openclbackend.cpp [+20 lines] — trunk buffer read
   ├── command/batch_analysis.cpp     [new file] — the module
   └── CMakeLists.txt              [+1 line] — add to build

2. Update main.h and main.cpp:
   ├── main.h   [+1 line]  — add function declaration
   └── main.cpp [+2 lines] — add help text + command handler
```

The total merge effort is **~30 lines**. Conflicts are unlikely because:
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
