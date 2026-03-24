"""Tests for game-agnostic batched self-play."""

from test.test_core.test_mcts import DummyNet, TicTacToe

from pymcts.core.config import MCTSConfig
from pymcts.core.self_play import batched_self_play


class TestBatchedSelfPlay:
    def test_completes_games(self):
        net = DummyNet()
        config = MCTSConfig(num_simulations=5, num_parallel_leaves=1)
        records = batched_self_play(
            net=net,
            game_factory=TicTacToe,
            mcts_config=config,
            num_games=2,
            batch_size=2,
            verbose=False,
        )
        assert len(records) == 2
        for r in records:
            assert r.num_moves > 0

    def test_records_have_policies(self):
        net = DummyNet()
        config = MCTSConfig(num_simulations=5)
        records = batched_self_play(
            net=net,
            game_factory=TicTacToe,
            mcts_config=config,
            num_games=1,
            batch_size=1,
            verbose=False,
        )
        has_policy = any(m.policy is not None for m in records[0].moves)
        assert has_policy
