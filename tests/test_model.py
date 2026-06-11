"""
KataRank — Unit Tests
======================
Tests for the current public API: engine KAB2 parsing, stream protocol,
schema/collate, models, and losses.

Run with:
    uv run pytest tests/ -v
"""

import io
import os
import struct
import tempfile
import unittest
import zlib

import numpy as np
import torch

from katarank.engine import KataGoEngine, parse_kab2_buffer
from katarank.schema import (
    KAB2Sample, kab2_make_sample, kab2_collate, rank_idx_to_str, RANK_NAMES,
)
from katarank.data.preprocess import (
    read_kab2, read_kab2_combined, probe_kab2_dim,
)
from katarank.model import (
    KataRankModel, OrdinalLogisticHead, KataRankLoss,
    DualViewSetTransformer, SetTransformer, pack_batch, unbatch,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def make_kab2_bytes(
    num_moves: int = 8,
    scalar_dim: int = 10,
    trunk_dim: int = 0,
    mean_lp: float = -2.5,
    human_rank: float = 5.0,
    human_lp: float = -3.0,
    compress: bool = False,
    seed: int = 0,
):
    """Synthesize a KAB2 payload matching batch_analysis.cpp output."""
    rng = np.random.default_rng(seed)
    move_dim = scalar_dim + 2 * trunk_dim
    moves = rng.standard_normal((num_moves, move_dim)).astype(np.float32)

    summary = [0.0] * 16
    summary[2]  = mean_lp
    summary[10] = human_rank
    summary[11] = human_lp

    header = struct.pack(
        '<4s7i', b'KAB2', num_moves, scalar_dim, trunk_dim,
        trunk_dim, 19, 19, 1 if compress else 0,
    )
    header += struct.pack('<16f', *summary)

    payload = moves.tobytes()
    if compress:
        comp = zlib.compress(payload)
        payload = struct.pack('<I', len(comp)) + comp
    return header + payload, moves


def make_combined_bytes(buf_b: bytes, buf_w: bytes) -> bytes:
    """[4B B_size][B][4B W_size][W] as written by file mode."""
    return (struct.pack('<I', len(buf_b)) + buf_b
            + struct.pack('<I', len(buf_w)) + buf_w)


def make_stream_frame(side: bytes, game_id: str, payload: bytes) -> bytes:
    """[1B side][4B idLen][id][4B size][payload]"""
    gid = game_id.encode('utf-8')
    return (side + struct.pack('<I', len(gid)) + gid
            + struct.pack('<I', len(payload)) + payload)


def make_game_x(n_b: int = 12, n_w: int = 12, input_dim: int = 10,
                seed: int = 0) -> torch.Tensor:
    """Packed (n_b + n_w, input_dim) with isWhite flag at column 7."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n_b + n_w, input_dim, generator=g)
    x[:n_b, 7] = 0.0
    x[n_b:, 7] = 1.0
    return x


# ─── KAB2 parsing ─────────────────────────────────────────────────────────────

class TestKAB2Parsing(unittest.TestCase):

    def test_parse_buffer_uncompressed(self):
        buf, moves = make_kab2_bytes(num_moves=10)
        parsed, info = parse_kab2_buffer(buf)
        np.testing.assert_array_almost_equal(parsed, moves)
        self.assertEqual(info['num_moves'], 10)
        self.assertEqual(info['input_dim'], 10)
        self.assertAlmostEqual(info['mean_log_prior'], -2.5, places=5)
        self.assertEqual(info['human_rank_idx'], 5)

    def test_parse_buffer_compressed(self):
        buf, moves = make_kab2_bytes(num_moves=10, compress=True)
        parsed, info = parse_kab2_buffer(buf)
        np.testing.assert_array_almost_equal(parsed, moves)

    def test_parse_buffer_with_trunk(self):
        buf, moves = make_kab2_bytes(num_moves=6, trunk_dim=4)
        parsed, info = parse_kab2_buffer(buf)
        self.assertEqual(info['input_dim'], 10 + 2 * 4)
        self.assertEqual(parsed.shape, (6, 18))

    def test_human_rank_unset(self):
        buf, _ = make_kab2_bytes(human_rank=-1.0)
        _, info = parse_kab2_buffer(buf)
        self.assertEqual(info['human_rank_idx'], -1)

    def test_bad_magic(self):
        with self.assertRaises(ValueError):
            parse_kab2_buffer(b'XXXX' + b'\x00' * 100)


class TestCombinedFormat(unittest.TestCase):

    def _write_tmp(self, data: bytes) -> str:
        fd, path = tempfile.mkstemp(suffix='.npz')
        with os.fdopen(fd, 'wb') as f:
            f.write(data)
        self.addCleanup(os.remove, path)
        return path

    def test_combined_roundtrip(self):
        buf_b, moves_b = make_kab2_bytes(num_moves=9, compress=True, seed=1)
        buf_w, moves_w = make_kab2_bytes(num_moves=8, compress=True, seed=2)
        path = self._write_tmp(make_combined_bytes(buf_b, buf_w))

        b, w, bi, wi = read_kab2_combined(path)
        np.testing.assert_array_almost_equal(b, moves_b)
        np.testing.assert_array_almost_equal(w, moves_w)
        self.assertEqual(bi['num_moves'], 9)
        self.assertEqual(wi['num_moves'], 8)

    def test_combined_missing_black(self):
        buf_w, moves_w = make_kab2_bytes(num_moves=7, seed=3)
        path = self._write_tmp(make_combined_bytes(b'', buf_w))

        b, w, bi, wi = read_kab2_combined(path)
        self.assertIsNone(b)
        np.testing.assert_array_almost_equal(w, moves_w)

        # read_kab2 falls back to White when Black is empty
        moves, info = read_kab2(path)
        np.testing.assert_array_almost_equal(moves, moves_w)

    def test_probe_dim_combined(self):
        buf_b, _ = make_kab2_bytes(trunk_dim=4, seed=4)
        buf_w, _ = make_kab2_bytes(trunk_dim=4, seed=5)
        path = self._write_tmp(make_combined_bytes(buf_b, buf_w))
        self.assertEqual(probe_kab2_dim(path), 18)

    def test_probe_dim_combined_empty_black(self):
        buf_w, _ = make_kab2_bytes(trunk_dim=2, seed=6)
        path = self._write_tmp(make_combined_bytes(b'', buf_w))
        self.assertEqual(probe_kab2_dim(path), 14)

    def test_truncated_raises(self):
        buf_b, _ = make_kab2_bytes(seed=7)
        data = make_combined_bytes(buf_b, b'')[:-2]
        path = self._write_tmp(data)
        with self.assertRaises(ValueError):
            read_kab2_combined(path)


class TestStreamProtocol(unittest.TestCase):

    def _engine(self) -> KataGoEngine:
        return KataGoEngine(model='dummy.bin.gz', katago_bin='dummy')

    def test_read_stream_frames(self):
        buf1, _ = make_kab2_bytes(num_moves=6, seed=8)
        buf2, _ = make_kab2_bytes(num_moves=7, seed=9)
        raw = (make_stream_frame(b'B', 'game_alpha', buf1)
               + make_stream_frame(b'W', 'game_alpha', buf2)
               + b'\x00')

        frames = list(self._engine()._read_stream(io.BytesIO(raw)))
        self.assertEqual(len(frames), 2)
        (s1, m1, i1), (s2, m2, i2) = frames
        self.assertEqual((s1, s2), ('B', 'W'))
        self.assertEqual(i1['game_id'], 'game_alpha')
        self.assertEqual(i2['game_id'], 'game_alpha')
        self.assertEqual(m1.shape, (6, 10))
        self.assertEqual(m2.shape, (7, 10))

    def test_read_stream_terminator_only(self):
        frames = list(self._engine()._read_stream(io.BytesIO(b'\x00')))
        self.assertEqual(frames, [])

    def test_read_stream_truncated(self):
        buf, _ = make_kab2_bytes(seed=10)
        raw = make_stream_frame(b'B', 'g', buf)[:-5]   # cut payload short
        frames = list(self._engine()._read_stream(io.BytesIO(raw)))
        self.assertEqual(frames, [])


# ─── SGF metadata ─────────────────────────────────────────────────────────────

class TestSgfMetadata(unittest.TestCase):

    SGF = ("(;GM[1]FF[4]SZ[19]PB[Lee Sedol]PW[Gu Li]BR[9d]WR[9d]"
           "DT[2014-02-23]RU[Chinese]KM[7.5]RE[B+R]EV[MLily Jubango]"
           ";B[pd];W[dd];B[qp])")

    def test_parse_header(self):
        from katarank.sgf_meta import parse_sgf_metadata
        meta = parse_sgf_metadata(self.SGF)
        self.assertEqual(meta['player_black'], 'Lee Sedol')
        self.assertEqual(meta['player_white'], 'Gu Li')
        self.assertEqual(meta['date'], '2014-02-23')
        self.assertEqual(meta['komi'], '7.5')
        self.assertEqual(meta['result'], 'B+R')
        self.assertEqual(meta['board_size'], '19')

    def test_moves_not_mistaken_for_props(self):
        from katarank.sgf_meta import parse_sgf_metadata
        # No RE/PB props here; move B[re] must not leak into metadata
        meta = parse_sgf_metadata("(;GM[1]SZ[19];B[re];W[pb])")
        self.assertNotIn('result', meta)
        self.assertNotIn('player_black', meta)

    def test_escaped_bracket(self):
        from katarank.sgf_meta import parse_sgf_metadata
        meta = parse_sgf_metadata(r"(;PB[a\]b];B[dd])")
        self.assertEqual(meta['player_black'], 'a]b')

    def test_read_from_file(self):
        from katarank.sgf_meta import read_sgf_metadata
        meta = read_sgf_metadata('tests/test.sgf')
        self.assertIsInstance(meta, dict)

    def test_missing_file_returns_empty(self):
        from katarank.sgf_meta import read_sgf_metadata
        self.assertEqual(read_sgf_metadata('no/such/file.sgf'), {})


# ─── Schema / collate ─────────────────────────────────────────────────────────

class TestSchema(unittest.TestCase):

    def test_make_sample_layout(self):
        b = np.random.randn(5, 10).astype(np.float32)
        w = np.random.randn(4, 10).astype(np.float32)
        s = kab2_make_sample(b, w, game_id='g1', target_b=-1.0, rank_b=3)
        self.assertEqual(s['seq_len'], 9)
        self.assertEqual(s['x'].shape, (9, 10))
        self.assertEqual(s['rank_b'].item(), 3)
        np.testing.assert_array_almost_equal(s['x'][:5].numpy(), b)

    def test_collate_packs_lengths(self):
        samples = [
            kab2_make_sample(np.zeros((3, 10), np.float32),
                             np.zeros((2, 10), np.float32), 'a'),
            kab2_make_sample(np.zeros((4, 10), np.float32),
                             np.zeros((4, 10), np.float32), 'b'),
        ]
        batch = kab2_collate(samples)
        self.assertEqual(batch['xlens'], [5, 8])
        self.assertEqual(batch['x'].shape, (13, 10))
        self.assertEqual(batch['game_ids'], ['a', 'b'])

    def test_collate_filters_empty(self):
        empty = kab2_make_sample(np.zeros((0, 10), np.float32),
                                 np.zeros((0, 10), np.float32), 'e')
        batch = kab2_collate([empty])
        self.assertEqual(batch['xlens'], [])

    def test_rank_idx_to_str(self):
        self.assertEqual(rank_idx_to_str(0), '20k')
        self.assertEqual(rank_idx_to_str(28), '9d')
        self.assertEqual(rank_idx_to_str(-1), '?')
        self.assertEqual(len(RANK_NAMES), 29)


# ─── Models ───────────────────────────────────────────────────────────────────

class TestOrdinalHead(unittest.TestCase):

    def test_probs_sum_to_one(self):
        head = OrdinalLogisticHead(16)
        z = torch.randn(4, 16)
        probs = head(z)
        self.assertEqual(probs.shape, (4, 29))
        self.assertTrue(torch.allclose(probs.sum(-1), torch.ones(4), atol=1e-4))
        self.assertTrue((probs >= 0).all())

    def test_thresholds_monotonic_after_update(self):
        head = OrdinalLogisticHead(16)
        # Perturb parameters as a training step would
        with torch.no_grad():
            head.thresh_deltas.add_(torch.randn_like(head.thresh_deltas) * 5)
            head.thresh_base.add_(1.3)
        t = head.thresholds
        self.assertTrue((t[1:] > t[:-1]).all(), "thresholds must stay increasing")


class TestDualView(unittest.TestCase):

    def test_forward_single(self):
        model = DualViewSetTransformer(input_dim=10, hidden_dim=32,
                                       num_heads=2, num_inducing=4,
                                       encoder_depth=1, cross_depth=1)
        x = make_game_x(12, 12)
        out = model(x, [24])
        self.assertEqual(out.shape, (1, 32))

    def test_forward_batched(self):
        model = DualViewSetTransformer(input_dim=10, hidden_dim=32,
                                       num_heads=2, num_inducing=4,
                                       encoder_depth=1, cross_depth=1)
        x1 = make_game_x(10, 10, seed=1)
        x2 = make_game_x(15, 14, seed=2)
        packed, lens = pack_batch([x1, x2])
        out = model(packed, lens)
        self.assertEqual(out.shape, (2, 32))


class TestBatchIndependence(unittest.TestCase):
    """Games in a packed batch must not influence each other.

    Regression test for the ISAB/PMA cross-game attention leak.
    """

    def test_set_transformer_independence(self):
        model = SetTransformer(input_dim=10, hidden_dim=32, num_heads=2,
                               num_inducing=4, depth=1).eval()
        a = torch.randn(20, 10)
        b = torch.randn(30, 10)
        c = torch.randn(25, 10)
        with torch.no_grad():
            z_ab = model(torch.cat([a, b]), [20, 30])
            z_ac = model(torch.cat([a, c]), [20, 25])
        self.assertTrue(
            torch.allclose(z_ab[0], z_ac[0], atol=1e-5),
            "game A's embedding changed when its batch partner changed",
        )

    def test_full_model_independence(self):
        model = KataRankModel(input_dim=10, hidden_dim=32, num_heads=2,
                              num_inducing=4, encoder_depth=1,
                              cross_depth=1).eval()
        a = make_game_x(10, 10, seed=1)
        b = make_game_x(12, 12, seed=2)
        c = make_game_x(14, 13, seed=3)
        with torch.no_grad():
            out_ab = model(torch.cat([a, b]), [20, 24])
            out_ac = model(torch.cat([a, c]), [20, 27])
        self.assertTrue(
            torch.allclose(out_ab['b_rating'][0], out_ac['b_rating'][0], atol=1e-5)
        )
        self.assertTrue(
            torch.allclose(out_ab['rank_probs_b'][0], out_ac['rank_probs_b'][0],
                           atol=1e-5)
        )


class TestKataRankModel(unittest.TestCase):

    def _model(self) -> KataRankModel:
        return KataRankModel(input_dim=10, hidden_dim=32, num_heads=2,
                             num_inducing=4, encoder_depth=1, cross_depth=1)

    def test_forward_keys_and_shapes(self):
        model = self._model()
        x1 = make_game_x(8, 8, seed=1)
        x2 = make_game_x(9, 9, seed=2)
        packed, lens = pack_batch([x1, x2])
        out = model(packed, lens)
        self.assertEqual(out['b_rating'].shape, (2,))
        self.assertEqual(out['w_rating'].shape, (2,))
        self.assertEqual(out['rank_probs_b'].shape, (2, 29))
        self.assertEqual(out['rank_probs_w'].shape, (2, 29))

    def test_predict_rank_batched(self):
        model = self._model()
        x1 = make_game_x(8, 8, seed=1)
        x2 = make_game_x(9, 9, seed=2)
        packed, lens = pack_batch([x1, x2])
        pred = model.predict_rank(packed, lens)
        self.assertEqual(len(pred['rank_name_b']), 2)
        self.assertIn(pred['rank_name_b'][0], OrdinalLogisticHead.RANK_NAMES)

    def test_save_load_roundtrip(self):
        model = self._model().eval()
        fd, path = tempfile.mkstemp(suffix='.pt')
        os.close(fd)
        self.addCleanup(os.remove, path)

        model.save(path)
        loaded = KataRankModel.load(path).eval()

        x = make_game_x(8, 8)
        with torch.no_grad():
            o1 = model(x, [16])
            o2 = loaded(x, [16])
        self.assertTrue(torch.allclose(o1['b_rating'], o2['b_rating']))
        self.assertTrue(torch.allclose(o1['rank_probs_b'], o2['rank_probs_b']))


# ─── Interpretability capture ─────────────────────────────────────────────────

class TestActivationCapture(unittest.TestCase):

    def test_capture_blocks_and_attention(self):
        from katarank.model import ActivationCapture
        model = KataRankModel(input_dim=10, hidden_dim=32, num_heads=2,
                              num_inducing=4, encoder_depth=1,
                              cross_depth=1).eval()
        x = make_game_x(10, 10)
        with ActivationCapture(model) as cap:
            with torch.no_grad():
                model(x, [20])

        self.assertGreater(len(cap.activations), 0)
        self.assertGreater(len(cap.attention_maps), 0)
        # Residual-stream activations are token-level (tokens, dim)
        for name, t in cap.activations.items():
            self.assertEqual(t.dim(), 2, name)
            self.assertEqual(t.shape[1], 32, name)
        # Captured tensors are detached
        self.assertFalse(any(t.requires_grad for t in cap.activations.values()))

    def test_hooks_removed_after_exit(self):
        from katarank.model import ActivationCapture
        model = KataRankModel(input_dim=10, hidden_dim=32, num_heads=2,
                              num_inducing=4, encoder_depth=1,
                              cross_depth=1).eval()
        x = make_game_x(8, 8)
        with ActivationCapture(model) as cap:
            with torch.no_grad():
                model(x, [16])
        n = len(cap.activations)
        with torch.no_grad():
            model(x, [16])   # outside the context: no new captures
        self.assertEqual(len(cap.activations), n)

    def test_stack_prefix(self):
        from katarank.model import ActivationCapture
        model = KataRankModel(input_dim=10, hidden_dim=32, num_heads=2,
                              num_inducing=4, encoder_depth=1,
                              cross_depth=1).eval()
        x = make_game_x(10, 10)
        with ActivationCapture(model) as cap:
            with torch.no_grad():
                model(x, [20])
        mat = cap.stack('encoder.encoder_b')
        self.assertEqual(mat.dim(), 2)
        self.assertEqual(mat.shape[1], 32)
        with self.assertRaises(KeyError):
            cap.stack('no_such_prefix')


# ─── Losses ───────────────────────────────────────────────────────────────────

class TestLosses(unittest.TestCase):

    def _predictions(self, batch: int = 4):
        return {
            'b_rating':     torch.randn(batch, requires_grad=True),
            'w_rating':     torch.randn(batch, requires_grad=True),
            'rank_probs_b': torch.softmax(torch.randn(batch, 29), -1),
            'rank_probs_w': torch.softmax(torch.randn(batch, 29), -1),
        }

    def _targets(self, batch: int = 4, with_rank: bool = True):
        return {
            'target_b':   torch.randn(batch),
            'target_w':   torch.randn(batch),
            'rank_b':     torch.randint(0, 29, (batch,)) if with_rank
                          else torch.full((batch,), -1, dtype=torch.long),
            'rank_w':     torch.randint(0, 29, (batch,)) if with_rank
                          else torch.full((batch,), -1, dtype=torch.long),
            'human_lp_b': torch.full((batch,), -3.0),
            'human_lp_w': torch.full((batch,), -3.0),
        }

    def test_loss_returns_dict_with_total(self):
        loss_fn = KataRankLoss()
        losses = loss_fn(self._predictions(), self._targets())
        for key in ('total', 'rating', 'bradley_terry', 'rank', 'l2'):
            self.assertIn(key, losses)
        self.assertEqual(losses['total'].dim(), 0)
        losses['total'].backward()   # gradient flows

    def test_rank_loss_zero_without_labels(self):
        loss_fn = KataRankLoss()
        losses = loss_fn(self._predictions(), self._targets(with_rank=False))
        self.assertAlmostEqual(losses['rank'].item(), 0.0, places=6)

    def test_phase_switch(self):
        loss_fn = KataRankLoss(w_rank=0.3)
        loss_fn.set_rank_weight(0.0)
        self.assertEqual(loss_fn.w_rank, 0.0)


# ─── Per-move review (workflow.py / REVIEW_API_DESIGN.md) ────────────────────

def make_review_streams(n_b: int = 3, n_w: int = 3, dim: int = 10):
    """Synthesize B/W KAB2 move matrices with known scalar values."""
    def rows(n, is_white):
        m = np.zeros((n, dim), dtype=np.float32)
        m[:, 0] = 0.60                  # whiteWinProb
        m[:, 1] = 0.35                  # whiteLossProb
        m[:, 3] = 2.0 / 50.0            # whiteScoreMean/50 → +2.0 pts for W
        m[:, 4] = 1.5 / 10.0            # shorttermScoreError/10 → 1.5
        m[:, 5] = 0.25                  # policyPrior
        m[:, 6] = 5.0 / 361.0           # policyRank/361 → rank 5
        m[:, 7] = 1.0 if is_white else 0.0
        m[:, 8] = 0.10                  # winDelta (white perspective)
        m[:, 9] = -1.0 / 50.0           # scoreDelta/50 → −1.0 pts for W
        return m
    return rows(n_b, False), rows(n_w, True)


class TestMoveRecords(unittest.TestCase):

    def test_alignment_and_perspective(self):
        from katarank.workflow import _move_records
        moves_b, moves_w = make_review_streams(3, 3)
        recs = _move_records(moves_b, moves_w)

        self.assertEqual([r['move_no'] for r in recs], [1, 2, 3, 4, 5, 6])
        self.assertEqual([r['color'] for r in recs],
                         ['B', 'W', 'B', 'W', 'B', 'W'])

        b, w = recs[0], recs[1]
        # White-perspective raws: win 0.60 / loss 0.35, +2 pts, delta +0.1 / −1 pt
        self.assertAlmostEqual(w['winrate'], 0.60, places=5)
        self.assertAlmostEqual(b['winrate'], 0.35, places=5)   # whiteLossProb
        self.assertAlmostEqual(w['score_lead'], 2.0, places=4)
        self.assertAlmostEqual(b['score_lead'], -2.0, places=4)
        self.assertAlmostEqual(w['win_delta'], 0.10, places=5)
        self.assertAlmostEqual(b['win_delta'], -0.10, places=5)
        self.assertAlmostEqual(w['score_delta'], -1.0, places=4)
        self.assertAlmostEqual(b['score_delta'], 1.0, places=4)
        # Perspective-free fields
        for r in (b, w):
            self.assertAlmostEqual(r['score_stdev'], 1.5, places=4)
            self.assertAlmostEqual(r['policy_prior'], 0.25, places=5)
            self.assertEqual(r['policy_rank'], 5)

    def test_empty_stream(self):
        from katarank.workflow import _move_records
        moves_b, _ = make_review_streams(4, 0)
        recs = _move_records(moves_b, np.zeros((0, 10), dtype=np.float32))
        self.assertEqual(len(recs), 4)
        self.assertTrue(all(r['color'] == 'B' for r in recs))


class _StubEngine:
    """Minimal engine stub: replays canned (side, moves, info) frames."""

    def __init__(self, frames):
        self.frames = frames

    def stream_games(self, sgf_paths=None, sgf_strings=None,
                     mode='lite', min_moves=10):
        yield from self.frames


class TestRunReview(unittest.TestCase):

    def _frames(self, gid='g1'):
        moves_b, moves_w = make_review_streams(3, 3)
        info = {'game_id': gid, 'mean_log_prior': -2.5, 'human_rank_idx': 7,
                'human_log_prior': -3.0}
        return [('B', moves_b, dict(info)), ('W', moves_w, dict(info))]

    def test_engine_stats_path(self):
        from katarank.workflow import run_review_files
        engine = _StubEngine(self._frames())
        outs = run_review_files(engine, None, ['g1.sgf'])
        self.assertEqual(len(outs), 1)
        out = outs[0]
        self.assertEqual(out['game_id'], 'g1')
        self.assertEqual(len(out['moves']), 6)
        self.assertAlmostEqual(out['b_rating'], -2.5, places=5)
        self.assertEqual(out['b_rank'], 7)
        self.assertEqual(out['metadata']['source'], 'engine')

    def test_model_inference_path(self):
        from katarank.workflow import run_review_files, InferenceWorkflow
        model = KataRankModel(input_dim=10, hidden_dim=32, num_heads=2,
                              num_inducing=4, encoder_depth=1, cross_depth=1)
        engine = _StubEngine(self._frames())
        wf = InferenceWorkflow(model, engine, device='cpu')
        outs = run_review_files(engine, wf, ['g1.sgf'])
        out = outs[0]
        self.assertEqual(out['metadata']['source'], 'model')
        self.assertEqual(len(out['moves']), 6)
        self.assertIn('b_rating', out)
        self.assertEqual(len(out['b_rank_probs']), 29)

    def test_two_games_grouped(self):
        from katarank.workflow import run_review_files
        engine = _StubEngine(self._frames('g1') + self._frames('g2'))
        outs = run_review_files(engine, None, ['g1.sgf', 'g2.sgf'])
        self.assertEqual([o['game_id'] for o in outs], ['g1', 'g2'])
        self.assertTrue(all(len(o['moves']) == 6 for o in outs))


# ─── SAE interface (sae.py) ───────────────────────────────────────────────────

class TestSparseAutoencoder(unittest.TestCase):

    def test_forward_shapes(self):
        from katarank.model import SparseAutoencoder
        sae = SparseAutoencoder(d_model=32, expansion=4)
        x = torch.randn(20, 32)
        out = sae(x)
        self.assertEqual(out['recon'].shape, (20, 32))
        self.assertEqual(out['features'].shape, (20, 128))
        self.assertTrue((out['features'] >= 0).all())

    def test_topk_sparsity(self):
        from katarank.model import SparseAutoencoder
        sae = SparseAutoencoder(d_model=32, expansion=4, k=5)
        f = sae.encode(torch.randn(20, 32))
        l0 = (f > 0).sum(dim=-1)
        self.assertTrue((l0 <= 5).all())

    def test_loss_keys_and_modes(self):
        from katarank.model import SparseAutoencoder
        x = torch.randn(16, 32)
        for k in (None, 5):
            sae = SparseAutoencoder(d_model=32, expansion=4, k=k)
            losses = sae.loss(x)
            for key in ('total', 'mse', 'l1', 'l0'):
                self.assertIn(key, losses)
            losses['total'].backward()   # gradient flows

    def test_save_load_roundtrip(self):
        from katarank.model import SparseAutoencoder
        sae = SparseAutoencoder(d_model=32, expansion=4, k=5)
        x = torch.randn(8, 32)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, 'sae.pt')
            sae.save(path)
            sae2 = SparseAutoencoder.load(path)
        self.assertEqual(sae2.get_config(), sae.get_config())
        torch.testing.assert_close(sae2.encode(x), sae.encode(x))

    def test_normalize_decoder(self):
        from katarank.model import SparseAutoencoder
        sae = SparseAutoencoder(d_model=32, expansion=4)
        with torch.no_grad():
            sae.W_dec.mul_(3.0)
        sae.normalize_decoder_()
        norms = sae.W_dec.norm(dim=-1)
        torch.testing.assert_close(norms, torch.ones_like(norms))


class TestFeatureExtractor(unittest.TestCase):

    def _model(self):
        return KataRankModel(input_dim=10, hidden_dim=32, num_heads=2,
                             num_inducing=4, encoder_depth=1,
                             cross_depth=1).eval()

    def test_extract_alignment(self):
        from katarank.model import SparseAutoencoder, FeatureExtractor
        model = self._model()
        sae = SparseAutoencoder(d_model=32, expansion=2, k=4)
        fx = FeatureExtractor(model, sae_b=sae)
        x = make_game_x(6, 6)
        moves = fx.extract(x, top_k=4)

        self.assertEqual(len(moves), 12)
        self.assertEqual([m['move_no'] for m in moves], list(range(1, 13)))
        for m in moves:
            self.assertEqual(m['color'], 'B' if m['move_no'] % 2 == 1 else 'W')
            self.assertLessEqual(len(m['feature_ids']), 4)
            self.assertEqual(len(m['feature_ids']), len(m['activations']))
            self.assertEqual(len(m['feature_ids']), len(m['labels']))

    def test_extract_with_registry_labels(self):
        from katarank.model import (
            SparseAutoencoder, FeatureExtractor, FeatureRegistry,
        )
        model = self._model()
        sae = SparseAutoencoder(d_model=32, expansion=2, k=4)
        x = make_game_x(4, 4)
        with tempfile.TemporaryDirectory() as d:
            reg = FeatureRegistry(os.path.join(d, 'features.json'))
            fx = FeatureExtractor(model, sae_b=sae, registry=reg)
            moves = fx.extract(x, top_k=4)
            fid = moves[0]['feature_ids'][0]
            reg.label(fid, 'test-feature', author='unit')
            moves2 = fx.extract(x, top_k=4)
            labeled = [lbl for m in moves2 for i, lbl in zip(m['feature_ids'],
                                                             m['labels'])
                       if i == fid]
            self.assertTrue(all(l == 'test-feature' for l in labeled))
            self.assertGreater(len(labeled), 0)

    def test_default_sites_resolution(self):
        from katarank.model import default_cross_sites
        site_b, site_w = default_cross_sites(self._model())
        self.assertTrue(site_b.endswith('mab_bw'))
        self.assertTrue(site_w.endswith('mab_wb'))


class TestFeatureRegistry(unittest.TestCase):

    def test_label_roundtrip_and_persistence(self):
        from katarank.model import FeatureRegistry
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, 'features.json')
            reg = FeatureRegistry(path)
            reg.label(412, 'overplay', author='bzhao', notes='low-prior aggro')
            self.assertEqual(reg.get(412)['label'], 'overplay')
            self.assertIsNone(reg.get(999))
            # reload from disk
            reg2 = FeatureRegistry(path)
            self.assertEqual(reg2.get(412)['label'], 'overplay')
            self.assertEqual(len(reg2.all()), 1)


class TestSAECorpus(unittest.TestCase):

    def test_accumulate_capture(self):
        from katarank.model import ActivationCapture, default_cross_sites
        model = KataRankModel(input_dim=10, hidden_dim=32, num_heads=2,
                              num_inducing=4, encoder_depth=1,
                              cross_depth=1).eval()
        site_b, _ = default_cross_sites(model)
        x = make_game_x(5, 5)
        with ActivationCapture(model, capture_attn=False,
                               accumulate=True) as cap:
            with torch.no_grad():
                model(x, [10])
                model(x, [10])
        self.assertEqual(cap.activations[site_b].shape, (10, 32))

    def test_collect_corpus(self):
        from katarank.model import collect_sae_corpus, default_cross_sites
        model = KataRankModel(input_dim=10, hidden_dim=32, num_heads=2,
                              num_inducing=4, encoder_depth=1,
                              cross_depth=1).eval()
        site_b, site_w = default_cross_sites(model)
        # Batch of two games: cross sites fire once per game — accumulate
        # mode must collect both (5 + 7 black tokens).
        g1 = make_game_x(5, 5, seed=1)
        g2 = make_game_x(7, 7, seed=2)
        batches = [(torch.cat([g1, g2]), [10, 14])]
        corpus = collect_sae_corpus(model, batches, sites=[site_b, site_w])
        self.assertEqual(corpus[site_b].shape, (12, 32))
        self.assertEqual(corpus[site_w].shape, (12, 32))


if __name__ == '__main__':
    unittest.main(verbosity=2)
