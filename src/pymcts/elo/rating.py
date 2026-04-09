"""Maximum Likelihood Elo rating computation."""

from collections import defaultdict

import numpy as np
from scipy.optimize import minimize

from pymcts.elo.config import EloRating, MatchResult

_OPTIMIZER_OPTIONS = {"maxiter": 1000, "ftol": 1e-12}


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _elo_expected_score(rating_a: float, rating_b: float) -> float:
    """Return expected score for player A given the two ratings."""
    raw = 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))
    return float(np.clip(raw, 1e-10, 1.0 - 1e-10))


def _match_nll(wins_a: int, wins_b: int, exp_a: float) -> float:
    """Negative log-likelihood contribution for one matchup."""
    exp_b = 1.0 - exp_a
    nll = 0.0
    if wins_a > 0:
        nll -= wins_a * np.log(exp_a)
    if wins_b > 0:
        nll -= wins_b * np.log(exp_b)
    return nll


def _run_optimizer(objective, x0: np.ndarray) -> np.ndarray:
    """Minimize *objective* starting from *x0* using L-BFGS-B."""
    result = minimize(objective, x0, method="L-BFGS-B", options=_OPTIMIZER_OPTIONS)
    return result.x


# ---------------------------------------------------------------------------
# compute_elo_ratings helpers
# ---------------------------------------------------------------------------

def _collect_players_and_games(
    match_results: list[MatchResult],
) -> dict[str, int]:
    """Return a mapping of player name -> total games played."""
    games_played: dict[str, int] = defaultdict(int)
    for m in match_results:
        total = m.total_games
        games_played[m.player_a] += total
        games_played[m.player_b] += total
    return games_played


def _build_rating_map(
    free_players: list[str],
    ratings_free: np.ndarray,
    anchor_player: str,
    anchor_rating: float,
) -> dict[str, float]:
    """Build a {player: rating} dict from the free-player array plus the anchor."""
    rating_map: dict[str, float] = {anchor_player: anchor_rating}
    for i, p in enumerate(free_players):
        rating_map[p] = ratings_free[i]
    return rating_map


def _full_nll(
    ratings_free: np.ndarray,
    free_players: list[str],
    anchor_player: str,
    anchor_rating: float,
    match_results: list[MatchResult],
) -> float:
    """Negative log-likelihood over all match results for full rating computation."""
    rating_map = _build_rating_map(free_players, ratings_free, anchor_player, anchor_rating)
    nll = 0.0
    for m in match_results:
        exp_a = _elo_expected_score(rating_map[m.player_a], rating_map[m.player_b])
        nll += _match_nll(m.wins_a, m.wins_b, exp_a)
    return nll


def _build_elo_results(
    free_players: list[str],
    optimized_x: np.ndarray,
    anchor_player: str,
    anchor_rating: float,
    games_played: dict[str, int],
) -> list[EloRating]:
    """Assemble and sort the final EloRating list."""
    ratings = [EloRating(name=anchor_player, rating=anchor_rating, games_played=games_played[anchor_player])]
    for i, p in enumerate(free_players):
        ratings.append(EloRating(name=p, rating=float(optimized_x[i]), games_played=games_played[p]))
    ratings.sort(key=lambda r: r.rating, reverse=True)
    return ratings


# ---------------------------------------------------------------------------
# compute_elo_against_pool helpers
# ---------------------------------------------------------------------------

def _candidate_sides(
    m: MatchResult,
    candidate: str,
    pool_ratings: dict[str, float],
) -> tuple[float, int, int]:
    """Return (opponent_rating, wins_candidate, wins_opponent) for one match."""
    if m.player_a == candidate:
        return pool_ratings[m.player_b], m.wins_a, m.wins_b
    return pool_ratings[m.player_a], m.wins_b, m.wins_a


def _candidate_nll(
    candidate_rating: np.ndarray,
    candidate: str,
    pool_ratings: dict[str, float],
    candidate_matches: list[MatchResult],
) -> float:
    """Negative log-likelihood for a single candidate against a frozen pool."""
    r_cand = candidate_rating[0]
    nll = 0.0
    for m in candidate_matches:
        r_opp, wins_cand, wins_opp = _candidate_sides(m, candidate, pool_ratings)
        exp_cand = _elo_expected_score(r_cand, r_opp)
        nll += _match_nll(wins_cand, wins_opp, exp_cand)
    return nll


def _filter_candidate_matches(
    candidate: str, match_results: list[MatchResult]
) -> list[MatchResult]:
    """Return only matches that involve the candidate player."""
    return [m for m in match_results if m.player_a == candidate or m.player_b == candidate]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_elo_ratings(
    match_results: list[MatchResult],
    anchor_player: str = "random",
    anchor_rating: float = 1000.0,
) -> list[EloRating]:
    """Compute ML Elo ratings from match results.

    Finds ratings that maximize the likelihood of observed results.
    The anchor_player is fixed at anchor_rating; all others are optimized.

    Returns ratings sorted descending by rating.
    """
    if not match_results:
        return []

    games_played = _collect_players_and_games(match_results)
    players = sorted(games_played.keys())

    if anchor_player not in players:
        players.append(anchor_player)
        games_played[anchor_player] = 0

    free_players = [p for p in players if p != anchor_player]
    if not free_players:
        return [EloRating(name=anchor_player, rating=anchor_rating, games_played=games_played[anchor_player])]

    def objective(ratings_free: np.ndarray) -> float:
        return _full_nll(ratings_free, free_players, anchor_player, anchor_rating, match_results)

    x0 = np.full(len(free_players), anchor_rating)
    optimized_x = _run_optimizer(objective, x0)
    return _build_elo_results(free_players, optimized_x, anchor_player, anchor_rating, games_played)


def compute_elo_against_pool(
    candidate: str,
    pool_ratings: dict[str, float],
    match_results: list[MatchResult],
) -> float:
    """Compute Elo for a single candidate against a pool with frozen ratings.

    Only matchups involving the candidate are used. Pool players' ratings
    are treated as fixed constants — the candidate's rating is the one
    that maximizes the likelihood of observed results.

    Args:
        candidate: Name of the candidate player.
        pool_ratings: Mapping of pool player names to their frozen Elo ratings.
        match_results: Match results (only those involving candidate are used).

    Returns:
        The candidate's computed Elo rating.
    """
    candidate_matches = _filter_candidate_matches(candidate, match_results)
    if not candidate_matches:
        return 1000.0

    def objective(candidate_rating: np.ndarray) -> float:
        return _candidate_nll(candidate_rating, candidate, pool_ratings, candidate_matches)

    optimized_x = _run_optimizer(objective, np.array([1000.0]))
    return float(optimized_x[0])
