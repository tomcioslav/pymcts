import json
from pathlib import Path

from test.test_core.test_mcts import TicTacToe, DummyNet
from pymcts.core.config import ArenaConfig, MCTSConfig, TrainingConfig, PathsConfig
from pymcts.core.players import MCTSPlayer
from pymcts.core.trainer import train


class TestArenaConfigPlayerSaving:
    def test_accepted_players_saved_to_arena_dir(self, tmp_path):
        """When using ArenaConfig, accepted players are saved to arena/."""
        paths = PathsConfig(
            root=tmp_path,
            trainings=tmp_path / "trainings",
            checkpoints=tmp_path / "checkpoints",
            models=tmp_path / "models",
            data=tmp_path / "data",
        )
        net = DummyNet()
        training_config = TrainingConfig(
            num_iterations=2,
            num_self_play_games=4,
            num_epochs=1,
            batch_size=4,
            self_play_batch_size=2,
            replay_buffer_size=2,
        )
        arena_config = ArenaConfig(num_games=4, threshold=0.0)  # threshold=0 to always accept

        train(
            game_factory=TicTacToe,
            net=net,
            mcts_config=MCTSConfig(num_simulations=5),
            training_config=training_config,
            arena=arena_config,
            paths_config=paths,
            verbose=False,
        )

        # Find the run directory
        run_dirs = list((tmp_path / "trainings").glob("run_*"))
        assert len(run_dirs) == 1
        run_dir = run_dirs[0]

        # Arena directory should exist with saved players
        arena_dir = run_dir / "arena"
        assert arena_dir.exists()
        player_dirs = sorted(arena_dir.glob("iteration_*"))
        assert len(player_dirs) >= 1  # At least one accepted player

        # Each should have player.json
        for player_dir in player_dirs:
            assert (player_dir / "player.json").exists()
