import pytest

from pymcts.elo.config import MatchResult, EloRating
from pymcts.elo.rating import compute_elo_ratings


class TestComputeEloRatings:
    def test_anchor_is_respected(self):
        """The anchor player should be exactly at anchor_rating."""
        results = [
            MatchResult(player_a="random", player_b="alice", wins_a=2, wins_b=8, draws=0),
        ]
        ratings = compute_elo_ratings(results, anchor_player="random", anchor_rating=1000.0)
        by_name = {r.name: r for r in ratings}
        assert by_name["random"].rating == pytest.approx(1000.0, abs=0.1)

    def test_stronger_player_rated_higher(self):
        """A player who wins more should have a higher rating."""
        results = [
            MatchResult(player_a="random", player_b="alice", wins_a=2, wins_b=8, draws=0),
            MatchResult(player_a="random", player_b="bob", wins_a=5, wins_b=5, draws=0),
        ]
        ratings = compute_elo_ratings(results, anchor_player="random", anchor_rating=1000.0)
        by_name = {r.name: r for r in ratings}
        assert by_name["alice"].rating > by_name["bob"].rating
        assert by_name["bob"].rating == pytest.approx(1000.0, abs=50.0)

    def test_symmetric_results_equal_ratings(self):
        """Two players with identical results against the anchor should have equal ratings."""
        results = [
            MatchResult(player_a="random", player_b="alice", wins_a=3, wins_b=7, draws=0),
            MatchResult(player_a="random", player_b="bob", wins_a=3, wins_b=7, draws=0),
        ]
        ratings = compute_elo_ratings(results, anchor_player="random", anchor_rating=1000.0)
        by_name = {r.name: r for r in ratings}
        assert by_name["alice"].rating == pytest.approx(by_name["bob"].rating, abs=1.0)

    def test_ratings_sorted_descending(self):
        """Ratings should be returned sorted by rating descending."""
        results = [
            MatchResult(player_a="random", player_b="strong", wins_a=1, wins_b=9, draws=0),
            MatchResult(player_a="random", player_b="weak", wins_a=8, wins_b=2, draws=0),
        ]
        ratings = compute_elo_ratings(results, anchor_player="random", anchor_rating=1000.0)
        assert ratings[0].name == "strong"
        assert ratings[-1].name == "weak"

    def test_games_played_counted(self):
        """games_played should reflect total games for each player."""
        results = [
            MatchResult(player_a="random", player_b="alice", wins_a=2, wins_b=8, draws=0),
            MatchResult(player_a="alice", player_b="bob", wins_a=6, wins_b=4, draws=0),
        ]
        ratings = compute_elo_ratings(results, anchor_player="random", anchor_rating=1000.0)
        by_name = {r.name: r for r in ratings}
        assert by_name["random"].games_played == 10
        assert by_name["alice"].games_played == 20
        assert by_name["bob"].games_played == 10

    def test_single_player_returns_anchor(self):
        """With no match results, only the anchor should be returned if present."""
        ratings = compute_elo_ratings([], anchor_player="random", anchor_rating=1000.0)
        assert ratings == []

    def test_extreme_winrate(self):
        """Even 100% win rates should produce finite ratings."""
        results = [
            MatchResult(player_a="random", player_b="god", wins_a=0, wins_b=20, draws=0),
        ]
        ratings = compute_elo_ratings(results, anchor_player="random", anchor_rating=1000.0)
        by_name = {r.name: r for r in ratings}
        assert by_name["god"].rating > 1000.0
        assert by_name["god"].rating < 5000.0  # finite, not infinity
