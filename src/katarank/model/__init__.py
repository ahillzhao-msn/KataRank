"""
KataRank Model Package
======================
Core neural network modules for the KataRank evaluation system.

Modules:
    set_transformer:  Set Transformer primitives (ISAB, SetEncoder, etc.)
    dual_view:        Dual-view encoder — causal cross-attention, segmented pooling
    multi_task:       KataRankModel with dual rating heads + ordinal rank heads
    losses:           KataRankLoss — rating MSE + Bradley-Terry + rank anchor
"""

from katarank.model.set_transformer import (
    SetTransformer,
    SetEncoder,
    MultiHeadAttentionBlock,
    SAB,
    ISAB,
    PMA,
    KataRankRatingModel,
    bradley_terry_score,
    scale_rating,
    unbatch,
    pack_batch,
)

from katarank.model.dual_view import (
    DualViewSetTransformer,
    BidirectionalCrossMAB,
    CrossMAB,
    CausalMAB,
    SegmentedAttentionPool,
    StreamPooling,
    build_causal_mask_bw,
    build_causal_mask_wb,
)

from katarank.model.multi_task import (
    KataRankModel,
    OrdinalLogisticHead,
)

from katarank.model.losses import (
    KataRankLoss,
    RankAnchorLoss,
    BradleyTerry,
    RatingMSELoss,
)

from katarank.model.interpret import ActivationCapture

from katarank.model.sae import (
    SparseAutoencoder,
    FeatureExtractor,
    FeatureRegistry,
    MoveFeature,
    collect_sae_corpus,
    default_cross_sites,
)

__all__ = [
    # Set Transformer core
    'SetTransformer', 'SetEncoder', 'MultiHeadAttentionBlock',
    'SAB', 'ISAB', 'PMA', 'KataRankRatingModel',
    'bradley_terry_score', 'scale_rating', 'unbatch', 'pack_batch',

    # Dual-view encoder
    'DualViewSetTransformer', 'BidirectionalCrossMAB', 'CrossMAB',
    'CausalMAB', 'SegmentedAttentionPool', 'StreamPooling',
    'build_causal_mask_bw', 'build_causal_mask_wb',

    # Model
    'KataRankModel', 'OrdinalLogisticHead',

    # Losses
    'KataRankLoss', 'RankAnchorLoss', 'BradleyTerry', 'RatingMSELoss',

    # Interpretability
    'ActivationCapture',
    'SparseAutoencoder', 'FeatureExtractor', 'FeatureRegistry',
    'MoveFeature', 'collect_sae_corpus', 'default_cross_sites',
]
