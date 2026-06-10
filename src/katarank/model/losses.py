"""
KataRank — Loss Functions
==========================
Training losses for KataRankModel.

Design:
  - Primary:   RatingMSELoss  (b_rating / w_rating vs normalised meanLogPrior)
  - Secondary: BradleyTerry (relative ordering consistency)
  - Anchor:    RankAnchorLoss (HumanSL rank calibration, training-phase only)

HumanSL is a training-only signal:
  - When humanRankIdx == -1 (not computed), rank loss is automatically zeroed.
  - humanLogPrior acts as per-sample confidence weight for the rank loss.
  - After Phase-1 training, OrdinalHead thresholds are calibrated;
    Phase-2 training can omit HumanSL data entirely.

Curriculum phases (KataRankLoss):
  Phase 1 (anchoring): rating + BT + rank  (rank_weight > 0)
  Phase 2 (generalisation): rating + BT    (rank_weight = 0, auto via mask)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


# ─── Individual Losses ────────────────────────────────────────────────────────

class RatingMSELoss(nn.Module):
    """MSE on both B and W ratings simultaneously."""

    def forward(self, b_rating: torch.Tensor, w_rating: torch.Tensor,
                target_b: torch.Tensor, target_w: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(b_rating, target_b) + F.mse_loss(w_rating, target_w)


class BradleyTerry(nn.Module):
    """
    Bradley-Terry consistency loss.

    Penalises disagreement between predicted relative strength and
    which player has higher meanLogPrior.

    stronger = 1.0 if meanLogPrior_B > meanLogPrior_W, else 0.0
    (derived from KAB2 header; no external label needed)
    """

    def forward(self, b_rating: torch.Tensor, w_rating: torch.Tensor,
                stronger: torch.Tensor) -> torch.Tensor:
        """
        Args:
            b_rating, w_rating: (batch,) normalised predicted ratings
            stronger: (batch,) float 1=Black stronger, 0=White stronger
        """
        win_prob = torch.sigmoid(b_rating - w_rating).clamp(1e-7, 1 - 1e-7)
        return -(stronger * torch.log(win_prob)
                 + (1 - stronger) * torch.log(1 - win_prob)).mean()


class RankAnchorLoss(nn.Module):
    """
    HumanSL rank calibration loss. Training-phase only.

    Confidence-weighted NLL over the 29-class ordinal distribution.
    Samples with humanRankIdx == -1 are masked out automatically.
    Samples with low humanLogPrior (poor HumanSL fit) get low weight.

    After Phase-1 training this loss becomes irrelevant once thresholds
    have converged to meaningful rank boundaries.
    """

    def __init__(self, confidence_threshold: float = -4.0):
        """
        Args:
            confidence_threshold: humanLogPrior value at which weight = 0.5.
                -4.0 covers most valid HumanSL fits (typical range -5 to -0.5).
        """
        super().__init__()
        self.conf_thresh = confidence_threshold

    def forward(
        self,
        rank_probs: torch.Tensor,
        rank_target: torch.Tensor,
        human_log_prior: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            rank_probs:       (batch, 29) ordinal probabilities
            rank_target:      (batch,)    int rank index, -1 = no label
            human_log_prior:  (batch,)    HumanSL log-likelihood (confidence)

        Returns:
            scalar loss (0.0 when all targets are -1)
        """
        valid = rank_target >= 0
        if not valid.any():
            return rank_probs.sum() * 0.0   # keeps gradient graph alive

        rp  = rank_probs[valid]
        rt  = rank_target[valid].long()
        hlp = human_log_prior[valid]

        # confidence ∈ (0, 1): sigmoid gate centred at conf_thresh
        confidence = torch.sigmoid(hlp - self.conf_thresh)    # (n_valid,)

        nll = F.nll_loss(torch.log(rp + 1e-8), rt, reduction='none')  # (n_valid,)
        return (confidence * nll).mean()


# ─── Combined v3 Loss ─────────────────────────────────────────────────────────

class KataRankLoss(nn.Module):
    """
    Combined loss for KataRankV3Model.

    Total:
        L = w_rating × L_rating
          + w_bt     × L_bradley_terry
          + w_rank   × (L_rank_b + L_rank_w)     ← auto-zero when no HumanSL
          + w_l2     × L_l2

    Curriculum usage:
        Phase 1 (anchoring):      KataRankLoss(w_rank=0.1)  + HumanSL data
        Phase 2 (generalisation): same loss object, pass rank_target=-1 for all
                                  → rank loss automatically zero, no code change
    """

    def __init__(
        self,
        w_rating: float = 1.0,
        w_bt:     float = 2.0,
        w_rank:   float = 0.1,
        w_l2:     float = 1e-5,
        confidence_threshold: float = -4.0,
    ):
        super().__init__()
        self.w_rating = w_rating
        self.w_bt     = w_bt
        self.w_rank   = w_rank
        self.w_l2     = w_l2

        self.rating_loss = RatingMSELoss()
        self.bt_loss     = BradleyTerry()
        self.rank_loss_b = RankAnchorLoss(confidence_threshold)
        self.rank_loss_w = RankAnchorLoss(confidence_threshold)

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        model: Optional[nn.Module] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            predictions: output of KataRankV3Model.forward()
                Keys: 'b_rating', 'w_rating', 'rank_probs_b', 'rank_probs_w'

            targets: dict with:
                'target_b'    (batch,) normalised meanLogPrior_B
                'target_w'    (batch,) normalised meanLogPrior_W
                'rank_b'      (batch,) int humanRankIdx_B, -1 if unavailable
                'rank_w'      (batch,) int humanRankIdx_W, -1 if unavailable
                'human_lp_b'  (batch,) humanLogPrior_B  (confidence weight)
                'human_lp_w'  (batch,) humanLogPrior_W

            model: optional, for L2 regularisation

        Returns:
            dict with scalar tensors:
                'total', 'rating', 'bradley_terry', 'rank', 'l2'
        """
        b_rating = predictions['b_rating']
        w_rating = predictions['w_rating']

        # Primary: rating MSE
        l_rating = self.rating_loss(b_rating, w_rating,
                                    targets['target_b'], targets['target_w'])

        # Secondary: relative ordering
        stronger = (targets['target_b'] > targets['target_w']).float()
        l_bt = self.bt_loss(b_rating, w_rating, stronger)

        # Anchor: HumanSL rank calibration (auto-zero when rank_b/w == -1)
        l_rank = (
            self.rank_loss_b(predictions['rank_probs_b'],
                             targets['rank_b'], targets['human_lp_b'])
          + self.rank_loss_w(predictions['rank_probs_w'],
                             targets['rank_w'], targets['human_lp_w'])
        )

        # L2 regularisation
        l_l2 = b_rating.new_zeros(())
        if model is not None and self.w_l2 > 0:
            l_l2 = self.w_l2 * sum(
                p.pow(2).sum() for p in model.parameters() if p.requires_grad
            )

        total = (self.w_rating * l_rating
               + self.w_bt     * l_bt
               + self.w_rank   * l_rank
               + l_l2)

        return {
            'total':         total,
            'rating':        l_rating,
            'bradley_terry': l_bt,
            'rank':          l_rank,
            'l2':            l_l2,
        }

    def set_rank_weight(self, w: float):
        """Switch curriculum phase at runtime without recreating the object."""
        self.w_rank = w
