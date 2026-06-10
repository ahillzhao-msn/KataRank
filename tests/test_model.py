"""
KataRank — Unit Tests
======================
Tests for core modules: Set Transformer, Dual-View, Multi-Task, Losses.

Run with:
    python -m pytest tests/ -v
    python tests/test_model.py  (standalone)
"""

import os
import sys
import unittest
import math

import torch
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.set_transformer import (
    SetTransformer, SetEncoder, MultiHeadAttentionBlock,
    ISAB, PMA, KataRankRatingModel,
    bradley_terry_score, scale_rating, unbatch, pack_batch,
)
from model.dual_view import (
    DualViewSetTransformer, CrossMAB, BidirectionalCrossMAB,
)
from model.multi_task import (
    KataRankMultiTaskModel, KataRankScoreHead,
    AbilityDimensionHead, StyleClassificationHead,
    OrdinalLogisticHead, DualRatingHead,
)
from model.losses import (
    KataRankCombinedLoss, BradleyTerryLoss,
    ScoreLoss, RatingMSELoss, AbilityLoss,
)


class TestSetTransformer(unittest.TestCase):
    """Tests for core Set Transformer components."""

    def setUp(self):
        self.batch_size = 4
        self.input_dim = 256
        self.hidden_dim = 128
        self.num_heads = 4
        self.num_inducing = 16

    def test_pack_unbatch(self):
        """Test batch packing/unpacking."""
        seqs = [torch.randn(100, 64), torch.randn(50, 64), torch.randn(200, 64)]
        packed, lens = pack_batch(seqs)
        self.assertEqual(packed.shape[0], sum(lens))
        self.assertEqual(len(lens), 3)

        slices = unbatch(packed, lens)
        self.assertEqual(len(slices), 3)
        for i, sl in enumerate(slices):
            self.assertEqual(sl.stop - sl.start, lens[i])

    def test_multi_head_attention_block(self):
        """MAB forward pass."""
        mab = MultiHeadAttentionBlock(self.hidden_dim, self.num_heads)
        X = torch.randn(100, self.hidden_dim)
        Y = torch.randn(80, self.hidden_dim)
        out = mab(X, Y)
        self.assertEqual(out.shape, (100, self.hidden_dim))

    def test_isab(self):
        """ISAB forward pass."""
        isab = ISAB(self.hidden_dim, self.num_heads, self.num_inducing)
        x = torch.randn(200, self.hidden_dim)
        out = isab(x, [200])
        self.assertEqual(out.shape, (200, self.hidden_dim))

    def test_pma(self):
        """PMA forward pass."""
        pma = PMA(self.hidden_dim, self.num_heads, num_seeds=1)
        x = torch.randn(200, self.hidden_dim)
        out = pma(x, [200])
        # PMA produces (batch * num_seeds, hidden_dim). With 1 seed and batch=1: (1, hidden_dim)
        self.assertEqual(out.shape, (1, self.hidden_dim))

    def test_set_transformer_forward(self):
        """Full SetTransformer forward pass."""
        model = SetTransformer(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            num_inducing=self.num_inducing,
            depth=2,
        )
        x = torch.randn(400, self.input_dim)
        out = model(x, [400])
        self.assertEqual(out.shape, (1, self.hidden_dim))

    def test_set_transformer_batched(self):
        """SetTransformer with batched input."""
        model = SetTransformer(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_heads=4,
            num_inducing=16,
            depth=1,
        )
        # Pack two sequences
        x1 = torch.randn(100, self.input_dim)
        x2 = torch.randn(200, self.input_dim)
        packed, lens = pack_batch([x1, x2])
        out = model(packed, lens)
        self.assertEqual(out.shape[0], 2)  # 2 pooled outputs (one per seed per batch)

    def test_rating_model(self):
        """KataRankRatingModel forward pass."""
        model = KataRankRatingModel(
            input_dim=self.input_dim,
            hidden_dim=64,
            num_heads=4,
            num_inducing=8,
            depth=1,
        )
        x = torch.randn(200, self.input_dim)
        out = model(x, [200])
        self.assertEqual(out.shape, (1,))  # scalar output

    def test_bradley_terry(self):
        """Bradley-Terry score computation."""
        # Equal ratings => 50% win probability
        p = bradley_terry_score(torch.tensor(0.0), torch.tensor(0.0))
        self.assertAlmostEqual(p.item(), 0.5, places=4)

        # Higher rated player should have >50% win prob
        p = bradley_terry_score(torch.tensor(1.0), torch.tensor(0.0))
        self.assertGreater(p.item(), 0.5)

    def test_scale_rating(self):
        """Rating scaling roundtrip."""
        normalized = torch.tensor(0.0)
        scaled = scale_rating(normalized)
        self.assertAlmostEqual(scaled.item(), 1623.19, places=1)

    def test_save_load_rating_model(self):
        """Save and load KataRankRatingModel."""
        model = KataRankRatingModel(
            input_dim=64, hidden_dim=32, num_heads=2,
            num_inducing=4, depth=1,
        )
        path = '/tmp/test_katarank_model.pt'
        model.save(path)
        loaded = KataRankRatingModel.load(path)
        self.assertIsNotNone(loaded)
        os.remove(path)


class TestDualView(unittest.TestCase):
    """Tests for Dual-View architecture."""

    def setUp(self):
        self.input_dim = 256
        self.hidden_dim = 64

    def test_dual_view_forward(self):
        """DualViewSetTransformer forward pass."""
        model = DualViewSetTransformer(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_heads=4,
            num_inducing=8,
            encoder_depth=1,
            cross_depth=1,
        )
        # Interleaved B/W sequence: B0,W0,B1,W1,...
        x = torch.randn(200, self.input_dim)
        out = model(x, [200])
        self.assertEqual(out.shape, (1, self.hidden_dim))

    def test_dual_view_batched(self):
        """DualView with batched input."""
        model = DualViewSetTransformer(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_heads=2,
            num_inducing=4,
            encoder_depth=1,
            cross_depth=1,
        )
        x1 = torch.randn(100, self.input_dim)
        x2 = torch.randn(200, self.input_dim)
        packed, lens = pack_batch([x1, x2])
        out = model(packed, lens)
        # Should produce (batch, hidden_dim) outputs
        self.assertEqual(out.shape[0], 2)

    def test_split_interleaved(self):
        """Test the split_interleaved logic."""
        model = DualViewSetTransformer(
            input_dim=64, hidden_dim=32,
            num_heads=2, num_inducing=4, encoder_depth=1,
        ).eval()

        # Create predictable interleaved sequence
        seq = torch.zeros(10, 64)
        seq[0::2, 0] = 1.0  # Black moves: first dim = 1
        seq[1::2, 0] = 2.0  # White moves: first dim = 2

        # After input projection, split should still work
        h = model.input_proj(seq)
        x_b, x_w, blens, wlens = model._split_interleaved(h, [10])

        self.assertEqual(len(blens), 1)
        self.assertEqual(blens[0], 5)  # ceil(10/2)
        self.assertEqual(wlens[0], 5)


class TestMultiTask(unittest.TestCase):
    """Tests for multi-task learning heads."""

    def setUp(self):
        self.batch_size = 4
        self.hidden_dim = 128

    def test_score_head(self):
        """Score head forward."""
        head = KataRankScoreHead(self.hidden_dim)
        z = torch.randn(self.batch_size, self.hidden_dim)
        out = head(z)
        self.assertEqual(out.shape, (self.batch_size,))

    def test_ability_head(self):
        """Ability dimension head."""
        head = AbilityDimensionHead(self.hidden_dim, num_dimensions=6)
        z = torch.randn(self.batch_size, self.hidden_dim)
        out = head(z)
        self.assertEqual(out.shape, (self.batch_size, 6))
        self.assertTrue(torch.all(out >= 0) and torch.all(out <= 1))  # sigmoid output

    def test_style_head(self):
        """Style classification head."""
        head = StyleClassificationHead(self.hidden_dim, num_styles=5)
        z = torch.randn(self.batch_size, self.hidden_dim)
        out = head(z)
        self.assertEqual(out.shape, (self.batch_size, 5))

        # Test predict method
        pred, conf = head.predict(z)
        self.assertEqual(pred.shape, (self.batch_size,))
        self.assertEqual(conf.shape, (self.batch_size,))

    def test_ordinal_head(self):
        """Ordinal logistic head."""
        head = OrdinalLogisticHead(self.hidden_dim, n_classes=9)
        z = torch.randn(self.batch_size, self.hidden_dim)
        out = head(z)
        self.assertEqual(out.shape, (self.batch_size, 9))
        # Probabilities should sum to 1
        self.assertTrue(torch.allclose(out.sum(dim=-1), torch.ones(self.batch_size)))

    def test_dual_rating_head(self):
        """Dual rating head."""
        head = DualRatingHead(self.hidden_dim)
        z = torch.randn(self.batch_size, self.hidden_dim)
        br, wr = head(z)
        self.assertEqual(br.shape, (self.batch_size,))
        self.assertEqual(wr.shape, (self.batch_size,))

    def test_full_multi_task_model(self):
        """Complete KataRankMultiTaskModel forward pass."""
        model = KataRankMultiTaskModel(
            input_dim=256,
            hidden_dim=64,
            num_heads=2,
            num_inducing=8,
            encoder_depth=1,
            cross_depth=1,
        )
        x = torch.randn(200, 256)
        outputs = model(x, [200])

        self.assertIn('score', outputs)
        self.assertEqual(outputs['score'].shape, (1,))

        self.assertIn('abilities', outputs)
        self.assertEqual(outputs['abilities'].shape, (1, 6))

        self.assertIn('style_logits', outputs)
        self.assertEqual(outputs['style_logits'].shape, (1, 5))

    def test_full_multi_task_batched(self):
        """Multi-task model with batch."""
        model = KataRankMultiTaskModel(
            input_dim=64,
            hidden_dim=32,
            num_heads=2,
            num_inducing=4,
            encoder_depth=1,
        )
        x1 = torch.randn(100, 64)
        x2 = torch.randn(150, 64)
        packed, lens = pack_batch([x1, x2])
        outputs = model(packed, lens)

        self.assertEqual(outputs['score'].shape, (2,))
        self.assertEqual(outputs['abilities'].shape, (2, 6))

    def test_model_save_load(self):
        """Save and load multi-task model."""
        model = KataRankMultiTaskModel(
            input_dim=64, hidden_dim=32,
            num_heads=2, num_inducing=4,
            encoder_depth=1,
        )
        path = '/tmp/test_katarank_mt.pt'
        model.save(path)

        loaded = KataRankMultiTaskModel.load(path)
        self.assertIsNotNone(loaded)

        # Put both in eval mode for deterministic comparison
        model.eval()
        loaded.eval()

        # Compare forward passes
        x = torch.randn(100, 64)
        out1 = model(x, [100])
        out2 = loaded(x, [100])
        self.assertTrue(torch.allclose(out1['score'], out2['score']))

        os.remove(path)


class TestLosses(unittest.TestCase):
    """Tests for loss functions."""

    def test_bradley_terry_loss(self):
        """BradleyTerryLoss."""
        loss_fn = BradleyTerryLoss()
        br = torch.tensor([1.0, 0.0])
        wr = torch.tensor([0.0, 1.0])
        score = torch.tensor([1.0, 0.0])  # B wins 1st, W wins 2nd

        loss = loss_fn(br, wr, score)
        self.assertGreater(loss.item(), 0)  # should be positive

    def test_ability_loss(self):
        """AbilityLoss with weak supervision."""
        loss_fn = AbilityLoss(mode='mse')
        pred = torch.sigmoid(torch.randn(4, 6))
        rating = torch.randn(4)

        loss = loss_fn(pred, rating=rating)
        self.assertGreater(loss.item(), 0)

    def test_combined_loss(self):
        """KataRankCombinedLoss."""
        loss_fn = KataRankCombinedLoss()

        predictions = {
            'black_rating': torch.randn(4),
            'white_rating': torch.randn(4),
            'abilities': torch.sigmoid(torch.randn(4, 6)),
            'style_logits': torch.randn(4, 5),
        }
        targets = {
            'score': torch.tensor([1.0, 0.0, 1.0, 0.0]),
            'target_score': torch.randn(4),
            'target_style': torch.randint(0, 5, (4,)),
        }

        losses = loss_fn(predictions, targets)
        self.assertIn('total', losses)
        self.assertIn('bradley_terry', losses)
        self.assertIsInstance(losses['total'].item(), float)


class TestDataPreprocess(unittest.TestCase):
    """Tests for data preprocessing utilities."""

    def test_ktrk_format(self):
        """Binary KTRK format roundtrip."""
        from data.preprocess import write_ktrk_features, read_ktrk_features
        features = np.random.randn(100, 256).astype(np.float32)
        path = '/tmp/test_ktrk.bin'
        write_ktrk_features(path, features)
        loaded = read_ktrk_features(path)
        np.testing.assert_array_almost_equal(features, loaded)
        os.remove(path)

    def test_compute_game_stats(self):
        """Game statistics computation."""
        from data.preprocess import compute_game_stats
        features = np.random.randn(200, 12)
        stats = compute_game_stats(features)
        self.assertIn('num_moves', stats)
        self.assertEqual(stats['num_moves'], 200)

    def test_normalize_features(self):
        """Feature normalization."""
        from data.preprocess import normalize_features
        features = np.random.randn(100, 64)
        normalized, mean, std = normalize_features(features)
        self.assertAlmostEqual(normalized.mean(), 0.0, places=1)
        self.assertAlmostEqual(normalized.std(), 1.0, places=1)


class TestScoreToRank(unittest.TestCase):
    """Tests for score-to-rank mapping."""

    def test_score_to_rank(self):
        from inference.predictor import score_to_rank, score_to_dan
        # A very high score should map to a high rank
        rank = score_to_rank(30000.0)
        # 30000 gives rank_code ~ 11p (depending on exact log mapping)
        self.assertIn('p', rank)  # should be pro rank

        # A low score should map to kyu
        rank = score_to_rank(100.0)
        self.assertIn('k', rank)

        # Zero should give minimum
        rank = score_to_rank(0.0)
        self.assertEqual(rank, '30k')

    def test_dan_continuous(self):
        """Continuous dan value."""
        from inference.predictor import score_to_dan
        dan = score_to_dan(30000.0)
        self.assertGreater(dan, 30)

        dan = score_to_dan(0.0)
        self.assertEqual(dan, 30.0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
