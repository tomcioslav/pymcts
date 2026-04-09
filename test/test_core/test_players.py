from test.test_core.test_mcts import TicTacToe, DummyNet
from pymcts.core.players import BasePlayer, RandomPlayer, MCTSPlayer
from pymcts.core.arena import batched_arena
from pymcts.core.config import MCTSConfig


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


class TestBatchedArenaWithPlayers:
    def test_random_vs_random(self):
        """batched_arena should work with two RandomPlayers."""
        p1 = RandomPlayer(name="p1")
        p2 = RandomPlayer(name="p2")
        result = batched_arena(
            player_a=p1,
            player_b=p2,
            game_factory=TicTacToe,
            num_games=10,
            swap_players=True,
            verbose=False,
        )
        assert len(result) == 10
        scores = result.scores
        assert scores.get("p1", 0) + scores.get("p2", 0) <= 10

    def test_mcts_vs_random(self):
        """batched_arena should work with MCTSPlayer vs RandomPlayer."""
        net = DummyNet()
        config = MCTSConfig(num_simulations=5)
        p1 = MCTSPlayer(net, config, name="mcts")
        p2 = RandomPlayer(name="random")
        result = batched_arena(
            player_a=p1,
            player_b=p2,
            game_factory=TicTacToe,
            num_games=4,
            swap_players=True,
            verbose=False,
        )
        assert len(result) == 4

    def test_mcts_vs_mcts(self):
        """batched_arena should work with two MCTSPlayers using batched inference."""
        net_a = DummyNet()
        net_b = DummyNet()
        config = MCTSConfig(num_simulations=5)
        p1 = MCTSPlayer(net_a, config, name="a")
        p2 = MCTSPlayer(net_b, config, name="b")
        result = batched_arena(
            player_a=p1,
            player_b=p2,
            game_factory=TicTacToe,
            num_games=4,
            swap_players=True,
            verbose=False,
        )
        assert len(result) == 4


class TestMCTSPlayerElo:
    def test_elo_defaults_to_none(self):
        net = DummyNet()
        config = MCTSConfig(num_simulations=5)
        player = MCTSPlayer(net, config)
        assert player.elo is None

    def test_elo_can_be_set(self):
        net = DummyNet()
        config = MCTSConfig(num_simulations=5)
        player = MCTSPlayer(net, config)
        player.elo = 1234.5
        assert player.elo == 1234.5


class TestMCTSPlayerSaveLoadElo:
    def test_save_load_with_elo(self, tmp_path):
        net = DummyNet()
        config = MCTSConfig(num_simulations=5)
        player = MCTSPlayer(net, config, name="test_player", elo=1150.0)
        player.save(tmp_path / "player")
        loaded = MCTSPlayer.load(tmp_path / "player")
        assert loaded.elo == 1150.0
        assert loaded.name == "test_player"

    def test_save_load_without_elo(self, tmp_path):
        net = DummyNet()
        config = MCTSConfig(num_simulations=5)
        player = MCTSPlayer(net, config, name="no_elo")
        player.save(tmp_path / "player")
        loaded = MCTSPlayer.load(tmp_path / "player")
        assert loaded.elo is None
