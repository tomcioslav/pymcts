"""Smoke test: run a tiny training iteration with the generic engine + Bridgit."""

from bridgit.core.self_play import batched_self_play
from bridgit.core.arena import Arena
from bridgit.core.players import RandomPlayer, GreedyMCTSPlayer
from bridgit.core.data import examples_from_records
from bridgit.core.config import MCTSConfig
from bridgit.games.bridgit.game import BridgitGame
from bridgit.games.bridgit.neural_net import BridgitNet
from bridgit.games.bridgit.config import BoardConfig


class TestFullPipeline:
    def test_self_play_train_arena_cycle(self):
        board_config = BoardConfig(size=3)
        game_factory = lambda: BridgitGame(board_config)
        mcts_config = MCTSConfig(num_simulations=5, num_parallel_leaves=1)

        net = BridgitNet(board_config=board_config)

        # 1. Self-play
        records = batched_self_play(
            net=net,
            game_factory=game_factory,
            mcts_config=mcts_config,
            num_games=2,
            batch_size=2,
            verbose=False,
        )
        assert len(records) == 2

        # 2. Extract examples and train
        examples = examples_from_records(
            records,
            game_factory=lambda cfg: BridgitGame(BoardConfig(**cfg)),
        )
        assert len(examples) > 0

        metrics = net.train_on_examples(examples, num_epochs=2, batch_size=4)
        assert metrics["loss"] > 0

        # 3. Arena
        new_player = GreedyMCTSPlayer(net, mcts_config, name="new")
        random_player = RandomPlayer(name="random")
        arena = Arena(
            new_player, random_player,
            game_factory=game_factory,
            game_type="bridgit",
            game_config=board_config.model_dump(),
        )
        eval_records = arena.play_games(2, verbose=False)
        assert len(eval_records) == 2
