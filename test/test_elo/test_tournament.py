from pymcts.core.players import RandomPlayer
from pymcts.elo.config import TournamentConfig, MatchResult, TournamentResult
from pymcts.elo.tournament import RatedPlayer, run_tournament

# Reuse TicTacToe from test_mcts
from test.test_core.test_mcts import TicTacToe


class TestRatedPlayer:
    def test_from_random(self):
        rp = RatedPlayer.from_random()
        assert rp.name == "random"
        player = rp.player_factory()
        assert isinstance(player, RandomPlayer)

    def test_from_random_custom_name(self):
        rp = RatedPlayer.from_random(name="rng_bot")
        assert rp.name == "rng_bot"


class TestSwissPairing:
    def test_no_repeat_matchups(self):
        """Tournament should not repeat the same matchup."""
        players = [RatedPlayer.from_random(name=f"p{i}") for i in range(4)]
        config = TournamentConfig(games_per_matchup=4, num_rounds=2, batch_size=1)
        result = run_tournament(
            players=players,
            game_factory=TicTacToe,
            config=config,
        )
        seen = set()
        for m in result.match_results:
            pair = tuple(sorted([m.player_a, m.player_b]))
            assert pair not in seen, f"Duplicate matchup: {pair}"
            seen.add(pair)

    def test_ratings_computed(self):
        """Tournament should produce ratings for all players."""
        players = [RatedPlayer.from_random(name=f"p{i}") for i in range(3)]
        config = TournamentConfig(games_per_matchup=4, num_rounds=2, batch_size=1)
        result = run_tournament(
            players=players,
            game_factory=TicTacToe,
            config=config,
        )
        rated_names = {r.name for r in result.ratings}
        player_names = {p.name for p in players}
        assert player_names == rated_names

    def test_anchor_player(self):
        """The first player named 'random' should be the anchor at 1000."""
        players = [
            RatedPlayer.from_random(),
            RatedPlayer.from_random(name="rng2"),
            RatedPlayer.from_random(name="rng3"),
        ]
        config = TournamentConfig(games_per_matchup=4, num_rounds=2, batch_size=1)
        result = run_tournament(
            players=players,
            game_factory=TicTacToe,
            config=config,
        )
        assert result.anchor_player == "random"

    def test_previous_results_avoids_replays(self):
        """Providing previous_results should skip already-played matchups."""
        players = [RatedPlayer.from_random(name=f"p{i}") for i in range(3)]
        config = TournamentConfig(games_per_matchup=4, num_rounds=2, batch_size=1)

        result1 = run_tournament(players=players, game_factory=TicTacToe, config=config)

        result2 = run_tournament(
            players=players, game_factory=TicTacToe, config=config,
            previous_results=result1,
        )
        all_pairs = set()
        for m in result2.match_results:
            pair = tuple(sorted([m.player_a, m.player_b]))
            assert pair not in all_pairs
            all_pairs.add(pair)


class TestRunTournament:
    def test_small_tournament_runs(self):
        """Smoke test: a 4-player tournament completes without error."""
        players = [RatedPlayer.from_random(name=f"rng_{i}") for i in range(4)]
        config = TournamentConfig(games_per_matchup=4, num_rounds=2, batch_size=1)
        result = run_tournament(
            players=players,
            game_factory=TicTacToe,
            config=config,
        )
        assert len(result.ratings) == 4
        assert len(result.match_results) > 0
        assert result.timestamp != ""

    def test_json_roundtrip(self):
        """Tournament result should survive JSON serialization."""
        players = [RatedPlayer.from_random(name=f"rng_{i}") for i in range(3)]
        config = TournamentConfig(games_per_matchup=4, num_rounds=1, batch_size=1)
        result = run_tournament(
            players=players,
            game_factory=TicTacToe,
            config=config,
        )
        data = result.model_dump_json()
        loaded = TournamentResult.model_validate_json(data)
        assert len(loaded.ratings) == len(result.ratings)
