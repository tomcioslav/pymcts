"""Swiss-style tournament with Elo ratings."""

import math
import random as rng
from datetime import datetime, timezone
from typing import Callable

from pydantic import BaseModel, ConfigDict

from pymcts.core.arena import batched_arena
from pymcts.core.base_game import BaseGame
from pymcts.core.players import BasePlayer, MCTSPlayer, RandomPlayer
from pymcts.elo.config import (
    EloRating,
    MatchResult,
    TournamentConfig,
    TournamentResult,
)
from pymcts.elo.rating import compute_elo_ratings


class RatedPlayer(BaseModel):
    """A player that can participate in rated tournaments."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    player_factory: Callable[[], BasePlayer]

    @classmethod
    def from_mcts_player(cls, player: MCTSPlayer) -> "RatedPlayer":
        """Create a RatedPlayer that produces a copy of the given MCTSPlayer."""
        name = player.name
        def factory(p=player):
            return p
        return cls(name=name, player_factory=factory)

    @classmethod
    def from_random(cls, name: str = "random") -> "RatedPlayer":
        """Create a RatedPlayer wrapping a RandomPlayer."""
        def factory(n=name):
            return RandomPlayer(name=n)
        return cls(name=name, player_factory=factory)


def _swiss_pair(
    players: list[str],
    ratings: dict[str, float],
    played: set[tuple[str, str]],
) -> list[tuple[str, str]]:
    """Pair players using Swiss system: sort by rating, pair adjacent, avoid repeats."""
    sorted_players = sorted(players, key=lambda p: ratings.get(p, 0.0), reverse=True)
    paired: set[str] = set()
    pairs: list[tuple[str, str]] = []

    for i, p in enumerate(sorted_players):
        if p in paired:
            continue
        for j in range(i + 1, len(sorted_players)):
            q = sorted_players[j]
            if q in paired:
                continue
            match_key = tuple(sorted([p, q]))
            if match_key in played:
                continue
            pairs.append((p, q))
            paired.add(p)
            paired.add(q)
            break

    return pairs


def run_tournament(
    players: list[RatedPlayer],
    game_factory: Callable[[], BaseGame],
    config: TournamentConfig,
    previous_results: TournamentResult | None = None,
) -> TournamentResult:
    """Run a Swiss-style tournament.

    If previous_results is provided, its match history is included
    and already-played matchups are skipped.
    """
    player_map = {p.name: p for p in players}
    player_names = [p.name for p in players]

    # Determine anchor
    anchor = "random" if "random" in player_map else player_names[0]

    # Seed from previous results
    match_results: list[MatchResult] = []
    played: set[tuple[str, str]] = set()
    if previous_results is not None:
        match_results = list(previous_results.match_results)
        for m in match_results:
            played.add(tuple(sorted([m.player_a, m.player_b])))

    # Current ratings (for pairing)
    current_ratings: dict[str, float] = {name: 1000.0 for name in player_names}

    # Number of rounds
    num_rounds = config.num_rounds or max(3, math.ceil(math.log2(max(len(players), 2))))

    for round_num in range(num_rounds):
        # First round: shuffle for random pairing
        if round_num == 0 and not previous_results:
            shuffled = list(player_names)
            rng.shuffle(shuffled)
            pairs = []
            for i in range(0, len(shuffled) - 1, 2):
                pair_key = tuple(sorted([shuffled[i], shuffled[i + 1]]))
                if pair_key not in played:
                    pairs.append((shuffled[i], shuffled[i + 1]))
        else:
            pairs = _swiss_pair(player_names, current_ratings, played)

        if not pairs:
            break

        for name_a, name_b in pairs:
            pa = player_map[name_a].player_factory()
            pb = player_map[name_b].player_factory()

            records = batched_arena(
                player_a=pa,
                player_b=pb,
                game_factory=game_factory,
                num_games=config.games_per_matchup,
                batch_size=config.batch_size,
                swap_players=config.swap_players,
                verbose=False,
            )

            scores = records.scores
            wins_a = scores.get(name_a, 0)
            wins_b = scores.get(name_b, 0)
            draws = len(records) - wins_a - wins_b

            match_results.append(MatchResult(
                player_a=name_a,
                player_b=name_b,
                wins_a=wins_a,
                wins_b=wins_b,
                draws=draws,
            ))
            played.add(tuple(sorted([name_a, name_b])))

        # Recompute ratings
        elo_ratings = compute_elo_ratings(match_results, anchor_player=anchor)
        new_ratings = {r.name: r.rating for r in elo_ratings}

        # Check convergence
        if round_num > 0:
            max_change = max(
                abs(new_ratings.get(n, 1000.0) - current_ratings.get(n, 1000.0))
                for n in player_names
            )
            if max_change < config.convergence_threshold:
                current_ratings = new_ratings
                break

        current_ratings = new_ratings

    # Final rating computation
    final_ratings = compute_elo_ratings(match_results, anchor_player=anchor)

    return TournamentResult(
        ratings=final_ratings,
        match_results=match_results,
        anchor_player=anchor,
        anchor_rating=1000.0,
        timestamp=datetime.now(timezone.utc).isoformat(),
        metadata={},
    )
