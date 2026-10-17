"""DistillationDataset 按 game id（而非数据集下标）取教师目标。"""
import torch

from katarank.train.distill import DistillationDataset, distill_collate


class _FakeKAB2:
    def __init__(self, ids, lens):
        self.games = [{'id': g} for g in ids]
        self._lens = lens

    def __len__(self):
        return len(self.games)

    def __getitem__(self, idx):
        n = self._lens[idx]
        return {'x': torch.full((n, 12), float(idx)), 'seq_len': n, 'game_id': self.games[idx]['id'],
                'target_b': torch.tensor(0.), 'target_w': torch.tensor(0.), 'rank_b': torch.tensor(-1),
                'rank_w': torch.tensor(-1), 'human_lp_b': torch.tensor(0.), 'human_lp_w': torch.tensor(0.)}


def _targets(ids):
    n = len(ids)
    return {'index': {g: i for i, g in enumerate(ids)},
            'teacher_b_rating': torch.arange(n, dtype=torch.float32),
            'teacher_w_rating': -torch.arange(n, dtype=torch.float32),
            'teacher_rank_probs_b': torch.eye(29)[:n], 'teacher_rank_probs_w': torch.eye(29)[:n]}


def test_targets_follow_game_id_not_position():
    # 教师行序与学生数据集不同，且学生里有一盘教师没有的棋
    targets = _targets(['g3', 'g1', 'g2'])
    ds = DistillationDataset(_FakeKAB2(['g1', 'g2', 'gX', 'g3'], [6, 6, 6, 6]), targets, student_dim=10)
    assert ds[0]['teacher_b_rating'] == 1 and ds[1]['teacher_b_rating'] == 2
    assert ds[3]['teacher_b_rating'] == 0
    assert ds[2]['seq_len'] == 0 and ds[0]['x'].shape == (6, 10)
    batch = distill_collate([ds[i] for i in range(4)])
    assert batch['game_ids'] == ['g1', 'g2', 'g3']
    assert batch['teacher_b_rating'].tolist() == [1, 2, 0]
