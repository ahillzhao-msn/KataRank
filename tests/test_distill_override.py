"""distill_session --promote-candidate: refuses while a session runs, and
records the gate's own verdict plus the override reason. No DB or GPU."""

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]


def _distill_session():
    try:
        import psycopg2  # noqa: F401
    except ImportError:
        sys.modules["psycopg2"] = mock.MagicMock()
    sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location("distill_session",
                                                  ROOT / "scripts" / "distill_session.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class PromoteOverrideTest(unittest.TestCase):
    def setUp(self):
        self.ds = _distill_session()
        tmp = Path(tempfile.mkdtemp())
        cand = tmp / "_candidate_lite"
        cand.mkdir()
        (cand / "best_lite.pt").write_bytes(b"candidate")
        self.job = tmp / "distill_job.json"
        self.args = mock.Mock(cache_dir=tmp, db_dsn="x", katarank_url="http://x")
        patches = [
            mock.patch.object(self.ds, "CANDIDATE_LITE_DIR", cand),
            mock.patch.object(self.ds, "LIVE_LITE_CKPT", tmp / "live_lite.pt"),
            mock.patch.object(self.ds, "JOB_FILE", self.job),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        (tmp / "live_lite.pt").write_bytes(b"live")

    def test_refuses_while_a_session_runs(self):
        self.job.write_text(json.dumps({"running": True}), encoding="utf-8")
        with mock.patch.object(self.ds, "_promote") as promote, \
                self.assertRaises(SystemExit):
            self.ds._promote_override(self.args, "why")
        promote.assert_not_called()

    def test_records_gate_verdict_and_reason(self):
        self.job.write_text(json.dumps({"running": False, "teacher_version": "v_t"}),
                            encoding="utf-8")
        cand = {"val_loss": 1.0, "rank_acc_pm1": 0.6, "rating_corr": 0.97}
        live = {"val_loss": 3.0, "rank_acc_pm1": 0.2, "rating_corr": 0.99}
        state = {}
        with mock.patch.object(self.ds, "_evaluate", side_effect=[cand, live]), \
                mock.patch.object(self.ds.psycopg2, "connect"), \
                mock.patch.object(self.ds, "_promote", return_value=("v1", True)) as promote, \
                mock.patch.object(self.ds.at, "get_cached_game_ids", return_value=[1, 2]), \
                mock.patch.object(self.ds.at, "_load_state", return_value=state), \
                mock.patch.object(self.ds.at, "_save_state"):
            self.ds._promote_override(self.args, "structural fix")
        note = promote.call_args.args[5]
        self.assertIn("manual override (structural fix)", note)
        self.assertIn("gate rejected", note)
        self.assertIn("rating_corr regressed", note)
        self.assertEqual(promote.call_args.args[6], "v_t")
        run = state["lite_runs"][-1]
        self.assertEqual(run["outcome"], "promoted_override")
        self.assertEqual(run["live_metrics"], live)
        self.assertTrue(run["served"])


if __name__ == "__main__":
    unittest.main()
