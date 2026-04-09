"""Maximum Likelihood Elo rating computation."""

from collections import defaultdict

import numpy as np
from scipy.optimize import minimize

from pymcts.elo.config import EloRating, MatchResult


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

    # Collect unique players and count games
    games_played: dict[str, int] = defaultdict(int)
    for m in match_results:
        total = m.total_games
        games_played[m.player_a] += total
        games_played[m.player_b] += total

    players = sorted(games_played.keys())

    if anchor_player not in players:
        players.append(anchor_player)
        games_played[anchor_player] = 0

    # Separate anchor from optimized players
    free_players = [p for p in players if p != anchor_player]
    if not free_players:
        return [EloRating(name=anchor_player, rating=anchor_rating, games_played=games_played[anchor_player])]

    player_to_idx = {p: i for i, p in enumerate(free_players)}

    def _negative_log_likelihood(ratings_free: np.ndarray) -> float:
        # Build full rating dict
        rating_map = {anchor_player: anchor_rating}
        for i, p in enumerate(free_players):
            rating_map[p] = ratings_free[i]

        nll = 0.0
        for m in match_results:
            ra = rating_map[m.player_a]
            rb = rating_map[m.player_b]
            # Expected score for player_a
            exp_a = 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))
            # Clamp to avoid log(0)
            exp_a = np.clip(exp_a, 1e-10, 1.0 - 1e-10)
            exp_b = 1.0 - exp_a

            if m.wins_a > 0:
                nll -= m.wins_a * np.log(exp_a)
            if m.wins_b > 0:
                nll -= m.wins_b * np.log(exp_b)

        return nll

    # Initial guess: all free players at anchor rating
    x0 = np.full(len(free_players), anchor_rating)

    result = minimize(
        _negative_log_likelihood,
        x0,
        method="L-BFGS-B",
        options={"maxiter": 1000, "ftol": 1e-12},
    )

    # Build output
    ratings = [EloRating(name=anchor_player, rating=anchor_rating, games_played=games_played[anchor_player])]
    for i, p in enumerate(free_players):
        ratings.append(EloRating(name=p, rating=float(result.x[i]), games_played=games_played[p]))

    ratings.sort(key=lambda r: r.rating, reverse=True)
    return ratings


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
    # Filter to only candidate matchups
    candidate_matches = [
        m for m in match_results
        if m.player_a == candidate or m.player_b == candidate
    ]

    if not candidate_matches:
        return 1000.0

    def _negative_log_likelihood(candidate_rating: np.ndarray) -> float:
        r_cand = candidate_rating[0]
        nll = 0.0
        for m in candidate_matches:
            if m.player_a == candidate:
                r_opp = pool_ratings[m.player_b]
                wins_cand = m.wins_a
                wins_opp = m.wins_b
            else:
                r_opp = pool_ratings[m.player_a]
                wins_cand = m.wins_b
                wins_opp = m.wins_a

            exp_cand = 1.0 / (1.0 + 10.0 ** ((r_opp - r_cand) / 400.0))
            exp_cand = np.clip(exp_cand, 1e-10, 1.0 - 1e-10)
            exp_opp = 1.0 - exp_cand

            if wins_cand > 0:
                nll -= wins_cand * np.log(exp_cand)
            if wins_opp > 0:
                nll -= wins_opp * np.log(exp_opp)

        return nll

    x0 = np.array([1000.0])
    result = minimize(
        _negative_log_likelihood,
        x0,
        method="L-BFGS-B",
        options={"maxiter": 1000, "ftol": 1e-12},
    )
    return float(result.x[0])
