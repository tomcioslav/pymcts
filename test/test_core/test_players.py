from test.test_core.test_mcts import TicTacToe, DummyNet
from bridgit.core.players import BasePlayer, RandomPlayer, MCTSPlayer
from bridgit.core.arena import Arena
from bridgit.core.config import MCTSConfig


class TestRandomPlayer:
    def test_returns_valid_action(self):
        game = TicTacToe()
        player = RandomPlayer()
        action = player.get_action(game)
        assert action in game.valid_actions()

    def test_last_policy_is_none(self):
        player = RandomPlayer()
        game = TicTacToe()
        player.get_action(game)
        assert player.last_policy is None


class TestMCTSPlayer:
    def test_returns_valid_action(self):
        game = TicTacToe()
        net = DummyNet()
        config = MCTSConfig(num_simulations=5)
        player = MCTSPlayer(net, config)
        action = player.get_action(game)
        assert action in game.valid_actions()

    def test_stores_last_policy(self):
        game = TicTacToe()
        net = DummyNet()
        config = MCTSConfig(num_simulations=5)
        player = MCTSPlayer(net, config)
        player.get_action(game)
        assert player.last_policy is not None
        assert player.last_policy.shape == (9,)


class TestArena:
    def test_play_game_completes(self):
        p1 = RandomPlayer(name="p1")
        p2 = RandomPlayer(name="p2")
        arena = Arena(p1, p2, game_factory=TicTacToe)
        record = arena.play_game()
        assert record.num_moves > 0

    def test_play_games_returns_collection(self):
        p1 = RandomPlayer(name="p1")
        p2 = RandomPlayer(name="p2")
        arena = Arena(p1, p2, game_factory=TicTacToe)
        collection = arena.play_games(4)
        assert len(collection) == 4
