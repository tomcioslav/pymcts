# Elo Rating System — Design Spec

## Overview

An Elo rating system for pymcts that ranks "players" — where a player is any entity that can play games (neural net + MCTS config pair, RandomPlayer, etc.). Uses Maximum Likelihood Elo (BayesElo-style) for order-independent, statistically optimal ratings. RandomPlayer is anchored at 1000 Elo.

Two modes of operation:
1. **Standalone tournament** — Swiss-style tournament across any set of players
2. **Training integration** — lightweight per-iteration Elo tracking against a reference pool

## Core Concepts

### Player Identity

A "player" in the Elo system is identified by a `name: str`. The name is the stable identity across tournaments and persistence. A player wraps either:
- A `(BaseNeuralNet, MCTSConfig)` pair that produces an `MCTSPlayer` (or `GreedyMCTSPlayer`)
- A `RandomPlayer` or any other `BasePlayer` subclass

The same neural net with different MCTS configs (e.g., 100 sims vs 200 sims) counts as two distinct players.

### Maximum Likelihood Elo

Given all match results, find the set of ratings R that maximizes the log-likelihood:

```
L(R) = Σ over all matches: wins_a * log(E_a) + wins_b * log(E_b)
```

Where:
- `E_a = 1 / (1 + 10^((R_b - R_a) / 400))` — expected win probability for player A
- `E_b = 1 - E_a` — expected win probability for player B

Note: Draws are not possible in Bridgit. The `MatchResult.draws` field exists for future game support but is ignored in the likelihood computation for now. If draws need to be supported later, a draw term `+ draws * log(D)` can be added with an appropriate draw model.

Constraint: `R[anchor_player] = 1000` (anchor is RandomPlayer by default).

Solved via `scipy.optimize.minimize` (L-BFGS-B or similar). With typical tournament sizes (< 100 players), this converges in milliseconds.

## Data Model

All models use Pydantic `BaseModel`.

### `MatchResult`

```python
class MatchResult(BaseModel):
    player_a: str
    player_b: str
    wins_a: int
    wins_b: int
    draws: int
```

Represents the outcome of one matchup (e.g., 40 games between two players). A single tournament round produces one `MatchResult` per pairing.

### `EloRating`

```python
class EloRating(BaseModel):
    name: str
    rating: float
    games_played: int
```

### `TournamentResult`

```python
class TournamentResult(BaseModel):
    ratings: list[EloRating]
    match_results: list[MatchResult]
    anchor_player: str
    anchor_rating: float
    timestamp: str  # ISO 8601
    metadata: dict  # game type, configs, notes
```

Serialized to/from JSON via Pydantic's `.model_dump_json()` / `.model_validate_json()`. Since ML Elo is stateless (ratings are fully determined by match results), loading old results and adding new matches simply means recomputing from the combined set.

### `TournamentConfig`

```python
class TournamentConfig(BaseModel):
    games_per_matchup: int = 40
    swap_players: bool = True
    num_rounds: int | None = None  # None = auto: max(3, ceil(log2(num_players)))
    convergence_threshold: float = 10.0  # stop early if max rating change < this
    batch_size: int = 8  # concurrent games in batched_arena
```

### `RatedPlayer`

```python
class RatedPlayer(BaseModel):
    name: str
    player_factory: Callable[[], BasePlayer]  # excluded from serialization

    model_config = ConfigDict(arbitrary_types_allowed=True)
```

`player_factory` is a callable that creates the `BasePlayer` instance on demand. It is not serialized — when loading tournament results from JSON, only ratings and match history are restored. Factories are provided when running new games.

Convenience constructors:

```python
@classmethod
def from_mcts_player(cls, player: MCTSPlayer) -> "RatedPlayer": ...

@classmethod
def from_random(cls, name: str = "random") -> "RatedPlayer": ...
```

## Swiss-Style Tournament

### Algorithm

The tournament runs in rounds:

1. **Pair players** — sort by current rating, pair adjacent players. No repeat matchups within the tournament. If Swiss pairing would create a repeat, slide to the next closest opponent.
2. **Play matches** — each pairing plays `games_per_matchup` games using the existing `batched_arena()` infrastructure (or a new player-based arena variant, see below). Player swapping is enabled by default.
3. **Recompute ratings** — ML Elo from all accumulated `MatchResult`s.
4. **Check convergence** — if max absolute rating change from previous round < `convergence_threshold`, stop early.
5. **Repeat** — until `num_rounds` is reached or convergence.

### Round Count

Default: `num_rounds = max(3, ceil(log2(num_players)))`. For 8 players → 3 rounds, 16 → 4. This is standard Swiss tournament sizing.

### First Round

Since all players start unrated, first-round pairings are randomized (shuffled).

### Bye Handling

If odd number of players, the lowest-rated player that hasn't had a bye gets paired against RandomPlayer (extra games against the anchor improve calibration). This counts as a normal match.

### Arena Refactor

The current `batched_arena()` takes two `BaseNeuralNet` instances. This is refactored to accept two `BasePlayer` instances instead — after all, it's players who enter the arena.

**How it works:**
- `batched_arena()` signature changes from `(net_a: BaseNeuralNet, net_b: BaseNeuralNet, mcts_config: MCTSConfig, ...)` to `(player_a: BasePlayer, player_b: BasePlayer, ...)`
- For `MCTSPlayer` pairs: extracts the MCTS engines and uses batched inference (same performance as before)
- For `RandomPlayer` or mixed pairs: falls back to sequential `player.get_action()` calls (batch_size=1 effectively)
- `trainer.py` is updated to create `MCTSPlayer`/`GreedyMCTSPlayer` instances and pass them to `batched_arena()`, using `MCTSPlayer.from_training_iteration()` for loading checkpoints

This unifies the arena interface and makes it directly usable by the Elo tournament system with no adapter layer.

## Training Integration

### Configuration

New fields added to `TrainingConfig`:

```python
class TrainingConfig(BaseModel):
    # ... existing fields ...
    elo_tracking: bool = False
    elo_reference_interval: int = 5  # save checkpoint as reference every N iterations
    elo_games_per_matchup: int = 40
```

### Behavior When `elo_tracking=True`

1. A `TournamentResult` is initialized at the start of training with RandomPlayer as the only rated player (1000 Elo).
2. After each training iteration (after the existing arena evaluation):
   - The new checkpoint is wrapped as a `RatedPlayer`
   - It plays `elo_games_per_matchup` games against each player in the reference pool
   - Match results are appended to the `TournamentResult`
   - All ratings are recomputed via ML Elo
   - The current Elo is logged
3. Every `elo_reference_interval` iterations, the checkpoint is added to the reference pool (so future iterations have more opponents).
4. The `TournamentResult` is saved to `{training_run_dir}/elo_results.json` after each iteration.

### Reference Pool Growth

- Iteration 0: pool = [RandomPlayer]
- Iteration 5: pool = [RandomPlayer, checkpoint_5]
- Iteration 10: pool = [RandomPlayer, checkpoint_5, checkpoint_10]
- etc.

This gives increasingly precise ratings as training progresses without making each iteration's evaluation proportionally slower.

## File Structure

```
src/pymcts/elo/
├── __init__.py          # Public API re-exports
├── rating.py            # compute_elo_ratings() — pure math, depends on scipy/numpy
├── tournament.py        # SwissTournament, run_tournament(), play_match(), RatedPlayer
├── config.py            # TournamentConfig, TournamentResult, MatchResult, EloRating
```

### `rating.py` — Public API

```python
def compute_elo_ratings(
    match_results: list[MatchResult],
    anchor_player: str = "random",
    anchor_rating: float = 1000.0,
) -> list[EloRating]:
    """Compute ML Elo ratings from match results.

    Returns ratings sorted descending by rating.
    """
```

Implementation: extract unique player names, set up parameter vector (one rating per player minus the anchor), define negative log-likelihood, minimize with scipy. Return sorted `EloRating` list.

### `tournament.py` — Public API

```python
class RatedPlayer(BaseModel):
    name: str
    player_factory: Callable[[], BasePlayer]

    @classmethod
    def from_mcts_player(cls, player: MCTSPlayer) -> "RatedPlayer": ...

    @classmethod
    def from_random(cls, name: str = "random") -> "RatedPlayer": ...

def run_tournament(
    players: list[RatedPlayer],
    game_factory: Callable[[], BaseGame],
    config: TournamentConfig,
    previous_results: TournamentResult | None = None,
) -> TournamentResult:
    """Run a Swiss-style tournament.

    Uses batched_arena() directly for each matchup (now accepts BasePlayer).
    If previous_results provided, incorporates existing match history
    (avoids replaying known matchups).
    """
```

### `config.py` — All Pydantic Models

Contains: `MatchResult`, `EloRating`, `TournamentConfig`, `TournamentResult` (as defined above).

### `__init__.py` — Re-exports

```python
from pymcts.elo.config import MatchResult, EloRating, TournamentConfig, TournamentResult
from pymcts.elo.rating import compute_elo_ratings
from pymcts.elo.tournament import RatedPlayer, run_tournament
```

## Training Integration Changes

### `core/config.py`

Add three fields to `TrainingConfig`:
- `elo_tracking: bool = False`
- `elo_reference_interval: int = 5`
- `elo_games_per_matchup: int = 40`

### `core/trainer.py`

Two changes:

**1. Refactor to use players instead of raw nets:**
- Create `GreedyMCTSPlayer` instances wrapping the nets
- Pass players to `batched_arena()` instead of `(net_a, net_b, mcts_config)`
- Use `MCTSPlayer.from_training_iteration()` for loading historical checkpoints

**2. Add Elo tracking block** (guarded by `if training_config.elo_tracking`):
1. Wrap current checkpoint as `RatedPlayer`
2. Play against reference pool via `batched_arena()`
3. Recompute ratings
4. Log Elo
5. Optionally add to reference pool
6. Save results to disk

## Dependencies

- `scipy` — for `scipy.optimize.minimize` (ML Elo computation)
- Already in use: `pydantic`, `torch`, `numpy`, `tqdm`

Check if scipy is already a dependency; if not, add it to `pyproject.toml`.

## Testing Strategy

1. **Unit tests for `rating.py`**:
   - Known matchups with hand-computed expected ratings
   - Anchor constraint is respected
   - Edge cases: single player, all wins, all draws

2. **Unit tests for `tournament.py`**:
   - Swiss pairing logic (no repeats, correct ordering)
   - Bye handling with odd players
   - `play_match()` with RandomPlayer vs RandomPlayer (smoke test)
   - `previous_results` warm-start behavior

3. **Integration test**:
   - Small tournament with RandomPlayer + 2-3 simple players
   - Verify ratings are order-independent (run twice, same results)
   - Verify JSON round-trip (save → load → same ratings)

4. **Training integration test**:
   - Mock training run with `elo_tracking=True`
   - Verify elo_results.json is created and grows each iteration
