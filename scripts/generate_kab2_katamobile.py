"""用 katamobile 引擎为 KAB2 缓存里的棋局生成学生特征（P3：蒸馏移动端 KataRank）。

教师（KataRank Full）吃的是 KataGo b18 产出的 KAB2；端上只有 katamobile 的 b6c96，产不出那些
特征。本脚本对教师缓存里的每一盘棋，从数据库取 SGF，用 katamobile 的 `kata-rank-features`
（GTP v0.4.0+，语义对齐 batch_analysis.cpp 的 10 维标量）重新生成逐手特征，写成同格式的
combined KAB2 文件：着法矩阵换成 katamobile 的，summary（标签、锚点）原样从教师文件复制；
`_meta.csv` 按产出的棋局过滤后复制。于是数据划分、标签与教师完全一致，KAB2Dataset 可直接读。

跳过：让子局（HA>0 / AB / AW，katamobile 从空盘重放）、非 19 路。没有 KM 的棋按 7.5 贴目
（KataGo 规则默认）。着法序列取主线；同色连走处截断（同 batch_analysis 的 replayGame）。
katamobile 以 --int8 运行，与 App 端推理一致。可断点续跑（已存在的输出跳过）。

用法：
  uv run python scripts/generate_kab2_katamobile.py \\
      --katamobile C:/Users/xiaoj/katamobile/build/Release/katamobile.exe \\
      --model C:/Users/xiaoj/katamobile/models/b6c96-sft-v3.ktg --workers 16
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import re
import struct
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from katarank.data.preprocess import build_kab2_payload, read_kab2_combined

log = logging.getLogger("kab2_km")

DB_DSN = "host=localhost port=5432 dbname=gopredict user=gopredict password=gopredict_dev"
TEACHER_CACHE = Path.home() / ".katarank" / "kab2_cache"
STUDENT_CACHE = Path.home() / ".katarank" / "kab2_katamobile"
GTP_LETTERS = "ABCDEFGHJKLMNOPQRST"  # GTP 跳过 I


# ─── SGF ─────────────────────────────────────────────────────────────────────

_PROP = re.compile(r"([A-Z]+)((?:\[(?:[^\]\\]|\\.)*\])+)")
_VAL = re.compile(r"\[((?:[^\]\\]|\\.)*)\]")


def sgf_main_line(sgf: str) -> list[dict[str, list[str]]]:
    """主线节点列表（每个节点 {属性: [值...]}）：递归下降解析，分支处只取第一个子树。"""
    n = len(sgf)

    def skip_ws(i: int) -> int:
        while i < n and sgf[i] not in "();[" and not sgf[i].isupper():
            i += 1
        return i

    def node(i: int) -> tuple[dict[str, list[str]], int]:  # sgf[i] == ';'
        props: dict[str, list[str]] = {}
        i += 1
        while True:
            i = skip_ws(i)
            m = _PROP.match(sgf, i)
            if not m:
                return props, i
            props[m.group(1)] = _VAL.findall(m.group(2))
            i = m.end()

    def tree(i: int) -> tuple[list[dict[str, list[str]]], int]:  # sgf[i] == '('
        nodes: list[dict[str, list[str]]] = []
        i = skip_ws(i + 1)
        while i < n and sgf[i] == ";":
            nd, i = node(i)
            nodes.append(nd)
            i = skip_ws(i)
        first = True
        while i < n and sgf[i] == "(":
            sub, i = tree(i)
            if first:
                nodes.extend(sub)
                first = False
            i = skip_ws(i)
        return nodes, i + 1  # 跳过 ')'

    i = sgf.find("(")
    return tree(i)[0] if i >= 0 else []


def sgf_to_moves(sgf: str) -> tuple[list[tuple[str, str]], float] | None:
    """返回 ([(color, gtp_coord)], komi)；不支持的棋局返回 None。"""
    nodes = sgf_main_line(sgf)
    if not nodes:
        return None
    root = nodes[0]
    if root.get("SZ", ["19"])[0].strip() != "19":
        return None
    if "AB" in root or "AW" in root or any("AB" in nd or "AW" in nd for nd in nodes[1:]):
        return None
    try:
        if int(root.get("HA", ["0"])[0] or 0) > 0:
            return None
        komi = float(root.get("KM", ["7.5"])[0] or 7.5)
    except ValueError:
        return None
    moves, last = [], None
    first = next((c for nd in nodes[1:] for c in ("B", "W") if c in nd), None)
    if first != "B":  # katamobile 从空盘按黑先交替重放，白先的棋（非让子）不支持
        return None
    for nd in nodes[1:]:
        for color in ("B", "W"):
            if color not in nd:
                continue
            if last == color:  # 同色连走：终局标注等，截断
                return moves, komi
            last = color
            v = nd[color][0].strip().lower()
            if v in ("", "tt"):
                moves.append((color, "pass"))
            else:
                x, y = ord(v[0]) - 97, ord(v[1]) - 97
                if not (0 <= x < 19 and 0 <= y < 19):
                    return moves, komi
                moves.append((color, f"{GTP_LETTERS[x]}{19 - y}"))
    return moves, komi


# ─── katamobile GTP ──────────────────────────────────────────────────────────

class KatamobileWorker:
    """一个常驻的 katamobile GTP 进程（单线程：OMP_NUM_THREADS=1，靠多进程并行）。"""

    def __init__(self, exe: str, model: str):
        env = dict(os.environ, OMP_NUM_THREADS="1")
        self.p = subprocess.Popen([exe, "--model", model, "--int8"], stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
                                  bufsize=1, env=env)

    def _responses(self, commands: list[str]) -> list[str]:
        self.p.stdin.write("".join(c + "\n" for c in commands))
        self.p.stdin.flush()
        out, cur = [], []
        while len(out) < len(commands):
            line = self.p.stdout.readline()
            if line == "":
                raise RuntimeError("katamobile exited")
            line = line.rstrip("\n")
            if line == "" and cur:
                out.append("\n".join(cur))
                cur = []
            elif line != "":
                cur.append(line)
        return out

    def features(self, moves: list[tuple[str, str]], komi: float) -> tuple[np.ndarray, np.ndarray]:
        plays = [f"play {c} {m}" for c, m in moves]
        resp = self._responses(["clear_board", f"komi {komi}", *plays, "kata-rank-features"])
        bad = next((i for i, r in enumerate(resp[2:-1]) if r.startswith("?")), None)
        if bad is not None:  # 非法着：截断到它之前重来（同 KataGo replayGame）
            resp = self._responses(["clear_board", f"komi {komi}", *plays[:bad], "kata-rank-features"])
        lines = resp[-1].removeprefix("= ").splitlines()
        nb, nw, dim = map(int, lines[0].split())
        rows = np.array([list(map(float, ln.split())) for ln in lines[1:]], dtype=np.float32)
        rows = rows.reshape(nb + nw, dim) if nb + nw else np.zeros((0, dim), np.float32)
        return rows[:nb], rows[nb:]

    def net_id(self) -> str:
        return self._responses(["kata-net-id"])[0].removeprefix("= ").strip()

    def close(self) -> None:
        try:
            self.p.stdin.write("quit\n")
            self.p.stdin.flush()
            self.p.wait(timeout=10)
        except (OSError, subprocess.TimeoutExpired):
            self.p.kill()


def write_combined(path: Path, xb: np.ndarray, xw: np.ndarray, info_b: dict, info_w: dict) -> None:
    parts = []
    for x, info in ((xb, info_b), (xw, info_w)):
        info = dict(info, input_dim=x.shape[1], scalar_dim=x.shape[1], trunk_dim=0)
        payload = build_kab2_payload(x, info)
        parts.append(struct.pack("<I", len(payload)) + payload)
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(b"".join(parts))
    tmp.replace(path)


# ─── main ────────────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--katamobile", required=True, help="katamobile GTP 可执行文件（v0.4.0+）")
    ap.add_argument("--model", required=True, help="与 App 相同的 b6c96 网络（.ktg / .kenc）")
    ap.add_argument("--teacher-cache", type=Path, default=TEACHER_CACHE)
    ap.add_argument("--out-dir", type=Path, default=STUDENT_CACHE)
    ap.add_argument("--db-dsn", default=DB_DSN)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 4))
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 盘（试跑）")
    args = ap.parse_args()

    import psycopg2

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with open(args.teacher_cache / "_meta.csv", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields, meta = reader.fieldnames, list(reader)
    todo = [r["file"].strip() for r in meta if not (args.out_dir / f"{r['file'].strip()}.npz").exists()]
    if args.limit:
        todo = todo[: args.limit]
    log.info("teacher games %d, to generate %d, workers %d", len(meta), len(todo), args.workers)

    workers = [KatamobileWorker(args.katamobile, args.model) for _ in range(args.workers)]
    net_id = workers[0].net_id()
    (args.out_dir / "_feature_net.txt").write_text(f"katamobile:{net_id}\n", encoding="utf-8")
    log.info("feature net katamobile:%s", net_id)

    free = list(workers)
    lock = threading.Lock()
    stats = {"done": 0, "skipped": 0, "failed": 0}
    t0 = time.time()

    def run(gid: str, sgf: str) -> None:
        parsed = sgf_to_moves(sgf)
        if parsed is None or len(parsed[0]) < 2:
            with lock:
                stats["skipped"] += 1
            return
        _, _, info_b, info_w = read_kab2_combined(str(args.teacher_cache / f"{gid}.npz"))
        with lock:
            w = free.pop()
        try:
            xb, xw = w.features(*parsed)
            write_combined(args.out_dir / f"{gid}.npz", xb, xw, info_b, info_w)
            key = "done"
        except Exception as e:  # noqa: BLE001 —— 单盘失败不影响整批，记数即可
            log.warning("game %s failed: %s", gid, e)
            key = "failed"
        finally:
            with lock:
                free.append(w)
        with lock:
            stats[key] += 1
            n = sum(stats.values())
            if n % 500 == 0:
                rate = n / (time.time() - t0)
                log.info("%d/%d  %s  %.1f games/s  eta %.0f min", n, len(todo), stats, rate,
                         (len(todo) - n) / rate / 60)

    conn = psycopg2.connect(args.db_dsn)
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for i in range(0, len(todo), 500):
                chunk = todo[i: i + 500]
                with conn.cursor() as cur:
                    cur.execute("SELECT id::text, sgf_content FROM games WHERE id::text = ANY(%s)", (chunk,))
                    rows = cur.fetchall()
                list(pool.map(lambda r: run(*r), rows))
    finally:
        conn.close()
        for w in workers:
            w.close()

    produced = {p.stem for p in args.out_dir.glob("*.npz")}
    with open(args.out_dir / "_meta.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=fields)
        wr.writeheader()
        wr.writerows(r for r in meta if r["file"].strip() in produced)
    log.info("finished %s in %.1f min; student cache has %d games", stats, (time.time() - t0) / 60,
             len(produced))


if __name__ == "__main__":
    main()
