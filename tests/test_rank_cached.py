"""/rank/cached re-ranks from cached KAB2 frames and tags each verdict with
the checkpoint that produced it. No KataGo binary needed."""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from katarank.model import KataRankModel
from katarank.workflow import InferenceWorkflow


def _app(checkpoint: Path, cache: Path):
    from katarank.api.server import create_app
    with mock.patch("katarank.analysis_daemon.AnalysisDaemon"), \
            mock.patch("katarank.engine.KataGoEngine"), \
            mock.patch("katarank.workflow.InferenceWorkflow"), \
            mock.patch("katarank.model.KataRankModel.load"):
        return create_app("model.bin.gz", checkpoint_path=str(checkpoint), persistent=False,
                          kab2_cache=str(cache))


def _endpoint(app, path):
    return next(r.endpoint for r in app.routes if getattr(r, "path", None) == path)


class RankCachedTest(unittest.TestCase):
    def setUp(self):
        tmp = Path(tempfile.mkdtemp())
        self.ckpt = tmp / "best.pt"
        self.ckpt.write_bytes(b"weights")
        self.cache = tmp / "kab2"
        self.cache.mkdir()
        (self.cache / "g1.npz").write_bytes(b"frames")

    def test_cached_ids_are_ranked_and_stamped_missing_ones_listed(self):
        from katarank.api import server
        app = _app(self.ckpt, self.cache)
        frames = np.zeros((30, 10), dtype=np.float32)
        verdict = {"game_id": "g1", "metadata": {"source": "model"}}
        with mock.patch("katarank.data.preprocess.read_kab2_combined",
                        return_value=(frames, frames, {}, {})), \
                mock.patch("katarank.workflow.result_to_output", return_value=verdict):
            out = _endpoint(app, "/rank/cached")(
                server.CachedRankRequest(game_ids=["g1", "nope", "../g1"], mode="full"))
        self.assertEqual(out["missing"], ["nope", "../g1"])
        self.assertEqual(len(out["results"]), 1)
        self.assertEqual(out["results"][0]["metadata"]["model"]["mode"], "full")
        self.assertEqual(len(out["results"][0]["metadata"]["model"]["sha256"]), 64)


class RankFramesTest(unittest.TestCase):
    def test_full_width_frames_feed_a_lite_model(self):
        torch.manual_seed(0)
        model = KataRankModel(input_dim=10)
        wf = InferenceWorkflow(model, engine=None, device="cpu")
        frames = np.random.default_rng(0).normal(size=(40, 1034)).astype(np.float32)
        a = wf.rank_frames(frames, frames, "g")
        b = wf.rank_frames(frames[:, :10], frames[:, :10], "g")
        self.assertAlmostEqual(a["b_rating"], b["b_rating"], places=5)


if __name__ == "__main__":
    unittest.main()
