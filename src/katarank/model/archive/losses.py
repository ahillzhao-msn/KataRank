"""
KataRank — Multi-Task Loss Functions
======================================
Combined loss for all output heads with configurable weighting.

Total loss:
    L = λ_score * L_score + λ_rating * L_rating + λ_ability * L_ability
      + λ_style * L_style + λ_ordinal * L_ordinal + λ_reg * L_reg

Each loss is normalized to be ~1 at initialization for stable training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List, Tuple


class BradleyTerryLoss(nn.Module):
    """
    Score prediction loss using Bradley-Terry model.

    Measures how well the predicted ratings explain the game outcome:
    P(Black wins) = 1 / (1 + 10^((W_rating - B_rating) / 400))

    This is the primary training signal for rating-based training.
    """

    def __init__(self, glicko_scale: float = 315.8088):
        super().__init__()
        self.glicko_scale = glicko_scale

    def forward(self, black_rating: torch.Tensor,
                white_rating: torch.Tensor,
                score: torch.Tensor) -> torch.Tensor:
        """
        Args:
            black_rating: (batch,) predicted Black rating (normalized)
            white_rating: (batch,) predicted White rating (normalized)
            score: (batch,) actual outcome: 1=Black wins, 0=White wins

        Returns:
            scalar negative log-likelihood loss
        """
        # Compute win probability for Black
        win_prob = 1.0 / (1.0 + (
            10 ** ((white_rating - black_rating) * self.glicko_scale / 400.0)
        ))

        # Clamp for numerical stability
        win_prob = torch.clamp(win_prob, 1e-7, 1 - 1e-7)

        # Negative log-likelihood
        loss = -(score * torch.log(win_prob) + (1 - score) * torch.log(1 - win_prob))
        return loss.mean()


class ScoreLoss(nn.Module):
    """
    Direct score prediction loss (MSE on Bradley-Terry scores).

    Alternative to BradleyTerryLoss when we predict score directly.
    """
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, predicted_score: torch.Tensor,
                actual_score: torch.Tensor) -> torch.Tensor:
        return self.mse(predicted_score, actual_score)


class RatingMSELoss(nn.Module):
    """
    MSE loss on rating predictions.

    Used when ground-truth ratings are available (e.g., Glicko-2 labels).
    """
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.mse(predicted, target)


class AbilityLoss(nn.Module):
    """
    Loss for multi-dimensional ability scores.

    Supports:
      - BCE: when binary ability labels are available
      - MSE: when continuous ability scores are available
      - Weak supervision: uses rating as proxy for all abilities
    """

    def __init__(self, mode: str = 'mse'):
        super().__init__()
        assert mode in ('mse', 'bce')
        self.mode = mode

    def forward(self, predicted: torch.Tensor,
                target: Optional[torch.Tensor] = None,
                rating: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            predicted: (batch, num_dims) predicted ability scores in [0,1]
            target: (batch, num_dims) ground-truth ability scores (optional)
            rating: (batch,) rating scores for weak supervision (optional)

        Returns:
            scalar loss
        """
        if target is not None:
            if self.mode == 'mse':
                return F.mse_loss(predicted, target)
            else:
                return F.binary_cross_entropy(predicted, target)

        # Weak supervision: assume ability scores correlate with rating
        if rating is not None:
            # Normalize rating to [0, 1] range for weak supervision
            rating_norm = torch.sigmoid(rating)
            target = rating_norm.unsqueeze(-1).expand_as(predicted)
            return F.mse_loss(predicted, target)

        # No supervision at all — return 0
        return torch.tensor(0.0, device=predicted.device)


class StyleLoss(nn.Module):
    """Cross-entropy loss for style classification."""

    def __init__(self):
        super().__init__()

    def forward(self, logits: torch.Tensor,
                targets: Optional[torch.Tensor] = None) -> torch.Tensor:
        if targets is not None:
            return F.cross_entropy(logits, targets)
        return torch.tensor(0.0, device=logits.device)


class OrdinalLoss(nn.Module):
    """
    Ordinal regression loss (from go-analyzer).

    Uses threshold-based ordinal logistic loss for rank prediction.
    """
    def __init__(self):
        super().__init__()

    def forward(self, probs: torch.Tensor,
                targets: Optional[torch.Tensor] = None) -> torch.Tensor:
        if targets is not None:
            return F.nll_loss(torch.log(torch.clamp(probs, min=1e-8)), targets)
        return torch.tensor(0.0, device=probs.device)


class L2Regularization(nn.Module):
    """L2 weight decay on all model parameters."""

    def __init__(self, weight_decay: float = 1e-5):
        super().__init__()
        self.weight_decay = weight_decay

    def forward(self, model: nn.Module) -> torch.Tensor:
        l2 = sum(p.pow(2).sum() for p in model.parameters() if p.requires_grad)
        return self.weight_decay * l2


# ─── Combined Multi-Task Loss ─────────────────────────────────────────────────

class KataRankCombinedLoss(nn.Module):
    """
    Combined multi-task loss with per-head weighting.

    DESIGN_V1.md Section 5.4:
        L = λ_rank * L_rank + λ_ability * Σ L_ability + λ_style * L_style

    Extended with:
        L = λ_score * L_score + λ_rating * L_rating + λ_ability * L_ability
          + λ_style * L_style + λ_ordinal * L_ordinal + λ_reg * L_reg
    """

    LOSSES = {
        'score': {
            'loss': ScoreLoss(),
            'default_weight': 1.0,
            'requires': ('predicted_score', 'target_score'),
        },
        'bradley_terry': {
            'loss': BradleyTerryLoss(),
            'default_weight': 2.0,
            'requires': ('black_rating', 'white_rating', 'score'),
        },
        'rating_mse': {
            'loss': RatingMSELoss(),
            'default_weight': 1.0,
            'requires': ('predicted', 'target'),
        },
        'abilities': {
            'loss': AbilityLoss(mode='mse'),
            'default_weight': 0.5,
            'requires': ('predicted_abilities', 'target_abilities'),
        },
        'style': {
            'loss': StyleLoss(),
            'default_weight': 0.3,
            'requires': ('style_logits', 'target_style'),
        },
        'ordinal': {
            'loss': OrdinalLoss(),
            'default_weight': 0.5,
            'requires': ('ordinal_probs', 'target_rank'),
        },
    }

    def __init__(self, weights: Optional[Dict[str, float]] = None,
                 enable_l2: bool = True, l2_weight: float = 1e-5):
        super().__init__()
        self.weights = weights or {}
        self.l2 = L2Regularization(l2_weight) if enable_l2 else None

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        model: Optional[nn.Module] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss.

        Args:
            predictions: dict from model forward pass
            targets: dict with ground-truth values
            model: model instance (for L2 regularization)

        Returns:
            dict: {'total': scalar, 'score': ..., 'abilities': ..., etc.}
        """
        losses = {}
        total = 0.0

        # Score loss (Bradley-Terry)
        if ('black_rating' in predictions and 'white_rating' in predictions
                and 'score' in targets):
            weight = self.weights.get('bradley_terry', 2.0)
            loss_val = self.LOSSES['bradley_terry']['loss'](
                predictions['black_rating'],
                predictions['white_rating'],
                targets['score'],
            )
            losses['bradley_terry'] = loss_val
            total += weight * loss_val

        # Rating MSE loss
        if 'score' in predictions and 'target_score' in targets:
            weight = self.weights.get('rating_mse', 1.0)
            loss_val = self.LOSSES['rating_mse']['loss'](
                predictions['score'], targets['target_score']
            )
            losses['rating_mse'] = loss_val
            total += weight * loss_val

        # Ability loss
        if 'abilities' in predictions:
            weight = self.weights.get('abilities', 0.5)
            loss_val = self.LOSSES['abilities']['loss'](
                predictions['abilities'],
                targets.get('target_abilities'),
                targets.get('target_score'),
            )
            losses['abilities'] = loss_val
            total += weight * loss_val

        # Style loss
        if 'style_logits' in predictions and 'target_style' in targets:
            weight = self.weights.get('style', 0.3)
            loss_val = self.LOSSES['style']['loss'](
                predictions['style_logits'], targets['target_style']
            )
            losses['style'] = loss_val
            total += weight * loss_val

        # Ordinal loss
        if 'rank_probs' in predictions and 'target_rank' in targets:
            weight = self.weights.get('ordinal', 0.5)
            loss_val = self.LOSSES['ordinal']['loss'](
                predictions['rank_probs'], targets['target_rank']
            )
            losses['ordinal'] = loss_val
            total += weight * loss_val

        # L2 regularization
        if self.l2 is not None and model is not None:
            l2_loss = self.l2(model)
            losses['l2'] = l2_loss
            total += l2_loss

        losses['total'] = total
        return losses


# ─── Legacy loss (compatible with StrengthNetLoss) ───────────────────────────

class KataRankLegacyLoss(nn.Module):
    """
    Legacy loss compatible with the original StrengthNetLoss interface.

    This allows the new model to be trained with the old training script.
    """

    def __init__(self, model: nn.Module, tau_ratings: float = 1.0,
                 tau_l2: float = 10.0):
        super().__init__()
        self.model = model
        self.tau_ratings = tau_ratings
        self.tau_l2 = tau_l2
        self.mse = nn.MSELoss()

    def trainLoss(self, bpred, wpred, by, wy, score):
        """Legacy interface returning (score_loss, rating_loss, l2_loss)."""
        # Bradley-Terry score loss
        win_prob = 1.0 / (1.0 + (10 ** ((wpred - bpred) * 315.8088 / 400.0)))
        win_prob = torch.clamp(win_prob, 1e-7, 1 - 1e-7)
        l_score = -(score * torch.log(win_prob) + (1 - score) * torch.log(1 - win_prob)).mean()

        # Rating MSE
        l_ratings = self.tau_ratings * (self.mse(bpred, by) + self.mse(wpred, wy))

        # L2
        params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        l_l2 = self.tau_l2 * sum(p.pow(2).sum() for p in self.model.parameters()) / params

        return l_score, l_ratings, l_l2

    def validationLoss(self, bpred, wpred, by, wy, score):
        win_prob = 1.0 / (1.0 + (10 ** ((wpred - bpred) * 315.8088 / 400.0)))
        win_prob = torch.clamp(win_prob, 1e-7, 1 - 1e-7)
        l_score = -(score * torch.log(win_prob) + (1 - score) * torch.log(1 - win_prob)).mean()
        l_ratings = self.tau_ratings * (self.mse(bpred, by) + self.mse(wpred, wy))
        return l_score, l_ratings
