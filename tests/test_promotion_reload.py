"""Promotion reaches the running server: /models/reload swaps the weights
in memory, and auto_train.reload_models only reports success when the
server's sha256 matches the promoted file. No KataGo binary needed."""

import asyncio
import hashlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from fastapi import HTTPException

ROOT = Path(__file__).resolve().parents[1]


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


class _Client:
    """Calls route handlers directly (the server venv has no httpx for TestClient)."""

    def __init__(self, app):
        self.routes = {(m, r.path): r.endpoint for r in app.routes
                       for m in getattr(r, "methods", ())}

    def health(self) -> dict:
        return asyncio.run(self.routes[("GET", "/health")]())

    def reload(self) -> dict:
        return self.routes[("POST", "/models/reload")]()


def _app(checkpoint: Path):
    from katarank.api.server import create_app
    with mock.patch("katarank.analysis_daemon.AnalysisDaemon"), \
            mock.patch("katarank.engine.KataGoEngine"), \
            mock.patch("katarank.workflow.InferenceWorkflow"), \
            mock.patch("katarank.model.KataRankModel.load") as load:
        app = create_app("model.bin.gz", checkpoint_path=str(checkpoint), persistent=False)
    return app, load


class ModelsReloadTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.ckpt = self.tmp / "best.pt"
        self.ckpt.write_bytes(b"old weights")

    def test_reload_serves_the_file_now_on_disk(self):
        client = _Client(_app(self.ckpt)[0])
        self.assertEqual(client.health()["checkpoint_sha256"], {"full": _sha(self.ckpt)})
        self.ckpt.write_bytes(b"promoted weights")
        with mock.patch("katarank.model.KataRankModel.load"):
            r = client.reload()
        self.assertEqual(r["checkpoint_sha256"], {"full": _sha(self.ckpt)})
        self.assertEqual(client.health()["checkpoint_sha256"]["full"], _sha(self.ckpt))

    def test_failed_reload_keeps_the_old_model(self):
        client = _Client(_app(self.ckpt)[0])
        old = _sha(self.ckpt)
        self.ckpt.write_bytes(b"corrupt")
        with mock.patch("katarank.model.KataRankModel.load", side_effect=RuntimeError("bad")), \
                self.assertRaises(HTTPException) as err:
            client.reload()
        self.assertEqual(err.exception.status_code, 500)
        self.assertEqual(client.health()["checkpoint_sha256"]["full"], old)


def _auto_train():
    sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location("auto_train", ROOT / "scripts" / "auto_train.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class ReloadModelsClientTest(unittest.TestCase):
    def setUp(self):
        self.at = _auto_train()
        self.ckpt = Path(tempfile.mkdtemp()) / "best.pt"
        self.ckpt.write_bytes(b"promoted weights")

    def _respond(self, sha: dict):
        return mock.patch("urllib.request.urlopen", return_value=io.BytesIO(
            json.dumps({"status": "reloaded", "checkpoint_sha256": sha}).encode()))

    def test_matching_sha_is_served(self):
        with self._respond({"full": _sha(self.ckpt)}):
            self.assertTrue(self.at.reload_models("http://x", {"full": self.ckpt}))

    def test_other_weights_are_not_served(self):
        with self._respond({"full": "0" * 64}):
            self.assertFalse(self.at.reload_models("http://x", {"full": self.ckpt}))

    def test_unreachable_server_is_not_served(self):
        import urllib.error
        with mock.patch("urllib.request.urlopen", side_effect=urllib.error.URLError("down")):
            self.assertFalse(self.at.reload_models("http://x", {"full": self.ckpt}))


if __name__ == "__main__":
    unittest.main()
