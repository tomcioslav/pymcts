from pathlib import Path

from test.test_core.test_mcts import TicTacToe, DummyNet

from pymcts.core.config import MCTSConfig
from pymcts.core.players import MCTSPlayer, RandomPlayer
from pymcts.elo.config import TournamentConfig, TournamentResult
from pymcts.elo.tournament import RatedPlayer, run_tournament
from pymcts.elo.rating import compute_elo_ratings


class TestEloIntegration:
    def test_tournament_with_mixed_players(self):
        """Run a tournament with MCTSPlayer and RandomPlayer together."""
        net = DummyNet()
        config = MCTSConfig(num_simulations=5)
        mcts_player = MCTSPlayer(net, config, name="dummy_mcts")

        players = [
            RatedPlayer.from_random(),
            RatedPlayer.from_mcts_player(mcts_player),
            RatedPlayer.from_random(name="rng2"),
        ]
        t_config = TournamentConfig(games_per_matchup=4, num_rounds=2, batch_size=1)
        result = run_tournament(
            players=players,
            game_factory=TicTacToe,
            config=t_config,
        )
        assert len(result.ratings) == 3
        assert result.anchor_player == "random"
        by_name = {r.name: r for r in result.ratings}
        assert by_name["random"].rating == 1000.0

    def test_order_independent_ratings(self):
        """Same match results should produce identical ratings regardless of recomputation."""
        players = [
            RatedPlayer.from_random(name="p0"),
            RatedPlayer.from_random(name="p1"),
            RatedPlayer.from_random(name="p2"),
        ]

        config = TournamentConfig(games_per_matchup=10, num_rounds=2, batch_size=1)
        result_a = run_tournament(players=players, game_factory=TicTacToe, config=config)

        # Recompute from same match_results
        ratings_recomputed = compute_elo_ratings(result_a.match_results, anchor_player="p0")

        by_name_orig = {r.name: r.rating for r in result_a.ratings}
        by_name_recomp = {r.name: r.rating for r in ratings_recomputed}

        for name in by_name_orig:
            assert abs(by_name_orig[name] - by_name_recomp[name]) < 0.01

    def test_json_persistence_roundtrip(self, tmp_path: Path):
        """Save tournament result to JSON, load it, verify ratings match."""
        players = [RatedPlayer.from_random(name=f"p{i}") for i in range(3)]
        config = TournamentConfig(games_per_matchup=4, num_rounds=1, batch_size=1)
        result = run_tournament(players=players, game_factory=TicTacToe, config=config)

        # Save
        path = tmp_path / "elo_results.json"
        path.write_text(result.model_dump_json(indent=2))

        # Load
        loaded = TournamentResult.model_validate_json(path.read_text())
        assert len(loaded.ratings) == len(result.ratings)
        for orig, load in zip(result.ratings, loaded.ratings):
            assert orig.name == load.name
            assert abs(orig.rating - load.rating) < 0.01
