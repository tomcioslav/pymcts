# Arena Abstraction Design

## Problem

The trainer (`core/trainer.py`) currently contains ~200 lines of arena evaluation logic spread across two strategies (head-to-head and Elo pool), selected via `isinstance` checks on config types. Arena state (`best_checkpoints`, `pool_players`, `pool_current_elo`, `elo_match_results`, `elo_reference_pool`) is scattered across `_TrainingContext` fields. The low-level game-playing engine (`batched_arena`) lives in `core/arena.py` alongside unrelated core infrastructure.

This makes the trainer hard to extend and tightly coupled to evaluation strategies.

## Goal

Extract all player management, game-playing, and evaluation logic into a standalone `arena/` package with a clean `Arena` abstraction. The trainer becomes a thin orchestrator: self-play → train → evaluate → accept/reject.

## Package Structure

```
src/pymcts/arena/
├── __init__.py              # re-exports public API
├── base.py                  # Arena ABC
├── engine.py                # batched_arena + game-playing internals (moved from core/arena.py)
├── config.py                # SinglePlayerArenaConfig, MultiPlayerArenaConfig, EloArenaConfig
├── models.py                # EvaluationResult + re-exports of all arena config models
└── arena_types/
    ├── __init__.py           # re-exports the three arena classes
    ├── single_player.py      # SinglePlayerArena
    ├── multi_player.py       # MultiPlayerArena
    └── elo.py                # EloArena
```

## Files Removed

- `core/arena.py` — engine moves to `arena/engine.py`, deleted entirely
- `ArenaConfig` and `EloArenaConfig` move out of `core/config.py` into `arena/config.py`

## Base Class

```python
# arena/base.py

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable

from pymcts.core.base_game import BaseGame
from pymcts.core.players import BasePlayer
from pymcts.core.game_record import GameRecordCollection
from pymcts.arena.models import EvaluationResult


class Arena(ABC):
    """Base class for all arena types.

    An Arena is a stateful evaluator that:
    - Plays games between players (for training data or evaluation)
    - Manages its own directory (persisting players, game records, history)
    - Decides whether a candidate player is better than the current best
    """

    def __init__(
        self,
        config,
        game_factory: Callable[[], BaseGame],
        arena_dir: Path,
        verbose: bool = True,
    ):
        self.config = config
        self.game_factory = game_factory
        self.arena_dir = arena_dir
        self.verbose = verbose
        self.arena_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def play_games(self, player: BasePlayer, num_games: int) -> GameRecordCollection:
        """Play games for training data generation.

        The Arena decides who the player plays against (itself, historical
        players, pool members) based on the arena type.
        """
        ...

    @abstractmethod
    def is_candidate_better(self, candidate: BasePlayer) -> EvaluationResult:
        """Evaluate whether candidate should replace the current best.

        Returns an EvaluationResult with accepted=True/False and
        arena-specific details. On acceptance, the Arena updates its
        internal state (saves the winner, updates pool, etc.).
        """
        ...
```

Key decisions:
- `Arena` is a plain class (not a Pydantic model) — it's a stateful service, not data.
- `game_factory` and `arena_dir` are set at construction. The Arena is long-lived across training iterations.
- The trainer creates `arena_dir` as a subdirectory of its `run_dir` and passes it in.

## Models

```python
# arena/models.py

class EvaluationResult(BaseModel):
    """Returned by Arena.is_candidate_better()."""
    accepted: bool
    details: dict  # arena-specific data (win_rate, elo, pool_size, etc.)
```

`models.py` also re-exports all config models from `arena/config.py` for consistency with the project's `models.py` pattern.

## Config Models

```python
# arena/config.py

class SinglePlayerArenaConfig(BaseModel):
    """Config for SinglePlayerArena (AlphaZero-style head-to-head)."""
    num_games: int = 40
    threshold: float = 0.55
    swap_players: bool = True
    batch_size: int = 8


class MultiPlayerArenaConfig(BaseModel):
    """Config for MultiPlayerArena (play against top N historical players)."""
    num_games: int = 40
    threshold: float = 0.55
    swap_players: bool = True
    batch_size: int = 8
    top_n: int = 5


class EloArenaConfig(BaseModel):
    """Config for EloArena (Elo pool-based evaluation)."""
    games_per_matchup: int = 40
    elo_threshold: float = 20.0
    pool_growth_interval: int = 5
    max_pool_size: int | None = None
    swap_players: bool = True
    batch_size: int = 8
    initial_pool: list[str] | None = None
```

## Arena Implementations

### SinglePlayerArena

- **`play_games(player, num_games)`** — player plays against itself (current self-play behavior). Uses `batched_arena` from `engine.py` with both sides being the same player.
- **`is_candidate_better(candidate)`** — loads `best_so_far` player from `arena_dir/best_so_far/`. Plays candidate vs best_so_far. Accepts if `win_rate >= threshold` OR `win_rate >= 0.5` with faster average wins. On acceptance, saves candidate as the new `best_so_far` and also saves to `arena_dir/iteration_NNN/`.
- **First iteration**: when no `best_so_far` exists yet, the candidate is automatically accepted and saved as the first best.
- **State on disk**:
  - `arena_dir/best_so_far/` — the current best player (overwritten on each acceptance)
  - `arena_dir/history/iteration_NNN/` — each accepted player preserved

### MultiPlayerArena

- **`play_games(player, num_games)`** — player plays against itself AND against the top N historical players from the pool. Returns all game records combined (more diverse training data).
- **`is_candidate_better(candidate)`** — plays candidate against top N players. Uses the same accept logic as SinglePlayerArena (threshold or faster wins) based on aggregate results.
- **State on disk**: same structure as SinglePlayerArena, but keeps up to `top_n` historical players rather than just the best.

### EloArena

- **`play_games(player, num_games)`** — player plays against the pool players. Returns all game records.
- **`is_candidate_better(candidate)`** — plays candidate against the entire pool, computes Elo via `compute_elo_against_pool`. Accepts if `candidate_elo >= current_elo + elo_threshold`. On acceptance, updates `current_elo`. Grows pool every `pool_growth_interval` iterations (saves candidate to pool). Evicts weakest non-random player if pool exceeds `max_pool_size`.
- **Initialization**: starts with a `RandomPlayer` at Elo 1000. Optionally loads players from `initial_pool` paths.
- **State on disk**:
  - `arena_dir/pool/` — all pool players (random, pool_iteration_NNN, etc.)
  - `arena_dir/elo_ratings.json` — current Elo ratings

## Engine (moved from core/arena.py)

`arena/engine.py` contains the low-level game-playing machinery, unchanged:

- `batched_arena(player_a, player_b, game_factory, num_games, ...)` — public function, plays N games between two players. Auto-selects batched MCTS or sequential mode.
- `_GameSlot`, `_batched_mcts_arena`, `_sequential_arena`, and all helper functions — private internals.

No behavioral changes to the engine, just a file move.

## Trainer Simplification

### New signature

```python
def train(
    game_factory: Callable[[], BaseGame],
    net: BaseNeuralNet,
    mcts_config: MCTSConfig,
    training_config: TrainingConfig,
    self_play_arena: Arena,
    eval_arena: Arena,
    paths_config: PathsConfig | None = None,
    verbose: bool = True,
):
```

`ArenaConfig | EloArenaConfig` parameter removed — the Arena objects carry their own config. `game_type` and `game_config` parameters removed — not needed by the trainer.

### New loop

```python
for iteration in range(1, num_iterations + 1):
    pre_checkpoint = str(iter_dir / "pre_training.pt")
    net.save_checkpoint(pre_checkpoint)

    # 1. Self-play
    player = GreedyMCTSPlayer(net, mcts_config, name="self_play")
    records = self_play_arena.play_games(player, num_games=training_config.num_self_play_games)
    # ... replay buffer, examples ...

    # 2. Train
    train_metrics = net.train_on_examples(examples, ...)
    net.save_checkpoint(str(iter_dir / "post_training.pt"))

    # 3. Evaluate
    candidate = GreedyMCTSPlayer(net, mcts_config, name="candidate")
    result = eval_arena.is_candidate_better(candidate)

    if not result.accepted:
        net.load_checkpoint(pre_checkpoint)
```

### Fields removed from _TrainingContext

All arena-related state is removed:
- `arena: ArenaConfig | EloArenaConfig`
- `arena_dir: Path`
- `best_checkpoints: deque`
- `pool_players: list[tuple[str, BasePlayer, float]]`
- `pool_current_elo: float | None`
- `elo_match_results: list[MatchResult]`
- `elo_reference_pool: list`

### Functions removed from trainer.py

All arena evaluation functions are deleted (they move into Arena implementations):
- `_init_elo_pool`
- `_evaluate_vs_opponent`
- `_evaluate_vs_historical`
- `_save_arena_results`
- `_arena_head_to_head`
- `_play_vs_pool`
- `_compute_baseline_elo`
- `_evict_weakest`
- `_grow_pool`
- `_arena_elo_pool`
- `_track_elo`

### User-facing usage

```python
from pymcts.arena import SinglePlayerArena, EloArena
from pymcts.arena.config import SinglePlayerArenaConfig, EloArenaConfig

# Option A: simple head-to-head
self_play = SinglePlayerArena(SinglePlayerArenaConfig(), game_factory, arena_dir=run_dir / "self_play")
evaluator = SinglePlayerArena(SinglePlayerArenaConfig(), game_factory, arena_dir=run_dir / "eval")
train(..., self_play_arena=self_play, eval_arena=evaluator)

# Option B: Elo-based evaluation
self_play = SinglePlayerArena(SinglePlayerArenaConfig(), game_factory, arena_dir=run_dir / "self_play")
evaluator = EloArena(EloArenaConfig(), game_factory, arena_dir=run_dir / "eval")
train(..., self_play_arena=self_play, eval_arena=evaluator)
```

## Import Updates

### core/__init__.py

Remove: `batched_arena`, `ArenaConfig`, `EloArenaConfig`
(These move to `arena/` package)

### core/config.py

Remove: `ArenaConfig`, `EloArenaConfig` classes
(Replaced by `SinglePlayerArenaConfig`, `MultiPlayerArenaConfig`, `EloArenaConfig` in `arena/config.py`)

### arena/__init__.py

Exports: `Arena`, `SinglePlayerArena`, `MultiPlayerArena`, `EloArena`, `EvaluationResult`, `batched_arena`

## Documentation Updates

The following docs and notebooks reference arena components and need updating:

### docs/content/ (MkDocs site)

- **guide/quickstart.md** — uses `ArenaConfig` and `batched_arena` in code examples. Update to new `SinglePlayerArenaConfig` / `EloArenaConfig` imports and new `train()` signature with `self_play_arena` + `eval_arena` parameters.
- **guide/training.md** — documents both `ArenaConfig` and `EloArenaConfig` extensively. Rewrite the arena evaluation section to explain the three Arena types and how they're passed to the trainer.
- **guide/evaluation.md** — documents `batched_arena` function and arena evaluation. Update import paths, add documentation for the Arena abstraction and its three implementations.
- **reference/config.md** — references configuration classes. Remove `ArenaConfig`/`EloArenaConfig`, add pointer to new arena config classes.
- **reference/engine.md** — references `batched_arena`. Update import path from `pymcts.core.arena` to `pymcts.arena.engine`. Document the Arena ABC and implementations.
- **concepts/architecture.md** — system architecture. Update the architecture description to reflect the new `arena/` package as a standalone component.

### Root files

- **README.md** — quick start example uses `ArenaConfig`. Update to new imports and `train()` signature.

### Notebooks

- **notebooks/training.ipynb** — imports `ArenaConfig` from `pymcts.core.config`. Update to new Arena construction and `train()` call.
- **notebooks/arena.ipynb** — imports `batched_arena` from `pymcts.core.arena`. Update import path.
- **notebooks/players.ipynb** — imports `batched_arena` from core. Update import path.

## Migration Notes

- The `batched_arena` function is still available as a public utility from `pymcts.arena.engine` for standalone use outside the training loop.
- Existing notebooks that import `ArenaConfig` or `EloArenaConfig` from `pymcts.core` will need import path updates.
- The `core/self_play.py` module is no longer needed by the trainer — its functionality is absorbed by `SinglePlayerArena.play_games()`. However, the module can be kept for standalone use or removed in a follow-up.
