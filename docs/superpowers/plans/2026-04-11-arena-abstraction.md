# Arena Abstraction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract all arena logic from the trainer into a standalone `arena/` package with three Arena implementations (SinglePlayer, MultiPlayer, Elo) and simplify the trainer to a thin orchestrator.

**Architecture:** New top-level `arena/` package with an `Arena` ABC, three implementations under `arena/arena_types/`, and the game-playing engine moved from `core/arena.py`. The trainer delegates all player management and evaluation to Arena objects passed at construction. Documentation and notebooks updated to match.

**Tech Stack:** Python 3.10+, Pydantic v2, PyTorch, existing pymcts infrastructure.

**Spec:** `docs/superpowers/specs/2026-04-11-arena-abstraction-design.md`

---

## File Structure

**Create:**
- `src/pymcts/arena/__init__.py` — re-exports public API
- `src/pymcts/arena/base.py` — `Arena` ABC
- `src/pymcts/arena/engine.py` — game-playing engine (moved from `core/arena.py`)
- `src/pymcts/arena/config.py` — `SinglePlayerArenaConfig`, `MultiPlayerArenaConfig`, `EloArenaConfig`
- `src/pymcts/arena/models.py` — `EvaluationResult` + re-exports of config models
- `src/pymcts/arena/arena_types/__init__.py` — re-exports arena classes
- `src/pymcts/arena/arena_types/single_player.py` — `SinglePlayerArena`
- `src/pymcts/arena/arena_types/multi_player.py` — `MultiPlayerArena`
- `src/pymcts/arena/arena_types/elo.py` — `EloArena`

**Delete:**
- `src/pymcts/core/arena.py` — engine moves to `arena/engine.py`

**Modify:**
- `src/pymcts/core/config.py` — remove `ArenaConfig`, `EloArenaConfig`
- `src/pymcts/core/__init__.py` — remove arena-related exports
- `src/pymcts/core/models.py` — remove `ArenaConfig`, `EloArenaConfig`
- `src/pymcts/core/trainer.py` — rewrite to use Arena objects
- `README.md` — update quick start example
- `docs/content/getting-started/quickstart.md` — update code examples
- `docs/content/guide/training.md` — rewrite arena config sections
- `docs/content/guide/evaluation.md` — update import paths
- `docs/content/reference/config.md` — remove arena configs, add pointer
- `docs/content/reference/engine.md` — update arena reference
- `docs/content/concepts/architecture.md` — update architecture diagram
- `notebooks/training.ipynb` — update imports and train() call
- `notebooks/arena.ipynb` — update imports
- `notebooks/players.ipynb` — update imports

---

### Task 1: Create arena package scaffold — config, models, base

**Files:**
- Create: `src/pymcts/arena/__init__.py`
- Create: `src/pymcts/arena/config.py`
- Create: `src/pymcts/arena/models.py`
- Create: `src/pymcts/arena/base.py`
- Create: `src/pymcts/arena/arena_types/__init__.py`

- [ ] **Step 1: Create `arena/config.py` with all three config models**

```python
"""Arena configuration models."""

from pydantic import BaseModel


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

- [ ] **Step 2: Create `arena/models.py` with EvaluationResult + re-exports**

```python
"""All public Pydantic models from the arena package."""

from pydantic import BaseModel

from pymcts.arena.config import (
    EloArenaConfig,
    MultiPlayerArenaConfig,
    SinglePlayerArenaConfig,
)


class EvaluationResult(BaseModel):
    """Returned by Arena.is_candidate_better()."""
    accepted: bool
    details: dict


__all__ = [
    "EloArenaConfig",
    "EvaluationResult",
    "MultiPlayerArenaConfig",
    "SinglePlayerArenaConfig",
]
```

- [ ] **Step 3: Create `arena/base.py` with Arena ABC**

```python
"""Arena base class."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable

from pymcts.arena.models import EvaluationResult
from pymcts.core.base_game import BaseGame
from pymcts.core.game_record import GameRecordCollection
from pymcts.core.players import BasePlayer


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
        """Play games for training data generation."""
        ...

    @abstractmethod
    def is_candidate_better(self, candidate: BasePlayer) -> EvaluationResult:
        """Evaluate whether candidate should replace the current best."""
        ...
```

- [ ] **Step 4: Create empty `arena/arena_types/__init__.py`**

```python
"""Arena type implementations."""
```

This will be populated in later tasks as we create each arena type.

- [ ] **Step 5: Create `arena/__init__.py` with re-exports**

```python
from pymcts.arena.base import Arena
from pymcts.arena.models import EvaluationResult

__all__ = [
    "Arena",
    "EvaluationResult",
]
```

This will be expanded in later tasks as we add engine and arena types.

- [ ] **Step 6: Verify imports work**

Run: `python -c "from pymcts.arena import Arena, EvaluationResult; from pymcts.arena.config import SinglePlayerArenaConfig, MultiPlayerArenaConfig, EloArenaConfig; from pymcts.arena.models import EvaluationResult; print('OK')"`

Expected: `OK`

- [ ] **Step 7: Commit**

```bash
git add src/pymcts/arena/
git commit -m "feat: create arena package scaffold with config, models, and base ABC"
```

---

### Task 2: Move game-playing engine from core/arena.py to arena/engine.py

**Files:**
- Create: `src/pymcts/arena/engine.py`
- Delete: `src/pymcts/core/arena.py`
- Modify: `src/pymcts/arena/__init__.py`
- Modify: `src/pymcts/core/__init__.py:5`
- Modify: `src/pymcts/core/trainer.py:12`

- [ ] **Step 1: Copy `core/arena.py` to `arena/engine.py` and update module docstring**

Copy the entire contents of `src/pymcts/core/arena.py` to `src/pymcts/arena/engine.py`. Change only the module docstring on line 1:

```python
"""Game-playing engine: run batched games between two players."""
```

All other code remains identical — same imports, same functions, same classes.

- [ ] **Step 2: Update `arena/__init__.py` to re-export `batched_arena`**

```python
from pymcts.arena.base import Arena
from pymcts.arena.engine import batched_arena
from pymcts.arena.models import EvaluationResult

__all__ = [
    "Arena",
    "EvaluationResult",
    "batched_arena",
]
```

- [ ] **Step 3: Update `core/trainer.py` import — change line 12**

Change:
```python
from pymcts.core.arena import batched_arena
```
To:
```python
from pymcts.arena.engine import batched_arena
```

- [ ] **Step 4: Update `core/__init__.py` — change line 5**

Change:
```python
from pymcts.core.arena import batched_arena
```
To:
```python
from pymcts.arena.engine import batched_arena
```

- [ ] **Step 5: Delete `src/pymcts/core/arena.py`**

```bash
git rm src/pymcts/core/arena.py
```

- [ ] **Step 6: Verify imports work**

Run: `python -c "from pymcts.arena import batched_arena; from pymcts.arena.engine import batched_arena; from pymcts.core import batched_arena; print('OK')"`

Expected: `OK`

- [ ] **Step 7: Commit**

```bash
git add src/pymcts/arena/engine.py src/pymcts/arena/__init__.py src/pymcts/core/__init__.py src/pymcts/core/trainer.py
git commit -m "refactor: move game-playing engine from core/arena.py to arena/engine.py"
```

---

### Task 3: Implement SinglePlayerArena

**Files:**
- Create: `src/pymcts/arena/arena_types/single_player.py`
- Modify: `src/pymcts/arena/arena_types/__init__.py`
- Modify: `src/pymcts/arena/__init__.py`

- [ ] **Step 1: Create `arena/arena_types/single_player.py`**

```python
"""SinglePlayerArena: AlphaZero-style head-to-head evaluation."""

import json
import logging
from pathlib import Path
from typing import Callable

from pymcts.arena.base import Arena
from pymcts.arena.config import SinglePlayerArenaConfig
from pymcts.arena.engine import batched_arena
from pymcts.arena.models import EvaluationResult
from pymcts.core.base_game import BaseGame
from pymcts.core.game_record import GameRecordCollection
from pymcts.core.players import BasePlayer, MCTSPlayer

logger = logging.getLogger("pymcts.arena.single_player")


class SinglePlayerArena(Arena):
    """Arena that evaluates candidates against the current best player.

    - play_games: player plays against itself (self-play)
    - is_candidate_better: candidate plays against best_so_far
    """

    def __init__(
        self,
        config: SinglePlayerArenaConfig,
        game_factory: Callable[[], BaseGame],
        arena_dir: Path,
        verbose: bool = True,
    ):
        super().__init__(config, game_factory, arena_dir, verbose)
        self._history_dir = self.arena_dir / "history"
        self._history_dir.mkdir(parents=True, exist_ok=True)
        self._best_so_far_dir = self.arena_dir / "best_so_far"
        self._iteration = 0

    def _has_best(self) -> bool:
        return (self._best_so_far_dir / "player.json").exists()

    def _load_best(self) -> BasePlayer:
        return MCTSPlayer.load(self._best_so_far_dir)

    def _save_as_best(self, player: BasePlayer) -> None:
        player.save(self._best_so_far_dir)

    def _save_to_history(self, player: BasePlayer) -> None:
        self._iteration += 1
        player.save(self._history_dir / f"iteration_{self._iteration:03d}")

    def play_games(self, player: BasePlayer, num_games: int) -> GameRecordCollection:
        """Player plays against itself (self-play)."""
        return batched_arena(
            player_a=player,
            player_b=player,
            game_factory=self.game_factory,
            num_games=num_games,
            batch_size=self.config.batch_size,
            swap_players=self.config.swap_players,
            verbose=self.verbose,
        )

    def is_candidate_better(self, candidate: BasePlayer) -> EvaluationResult:
        """Evaluate candidate against best_so_far."""
        if not self._has_best():
            self._save_as_best(candidate)
            self._save_to_history(candidate)
            if self.verbose:
                print("  No previous best — accepting first candidate.")
            return EvaluationResult(accepted=True, details={"first_candidate": True})

        best = self._load_best()
        records = batched_arena(
            player_a=candidate,
            player_b=best,
            game_factory=self.game_factory,
            num_games=self.config.num_games,
            batch_size=self.config.batch_size,
            swap_players=self.config.swap_players,
            verbose=self.verbose,
        )

        accepted = records.is_better(candidate.name, self.config.threshold)
        result = records.evaluate(candidate.name)

        if self.verbose:
            print(f"  {candidate.name}: {result.wins}W/{result.losses}L "
                  f"({result.win_rate:.1%})")

        if accepted:
            self._save_as_best(candidate)
            self._save_to_history(candidate)
            if self.verbose:
                print("  -> ACCEPTED")
        else:
            if self.verbose:
                print("  -> REJECTED")

        details = {
            "wins": result.wins,
            "losses": result.losses,
            "draws": result.draws,
            "total": result.total,
            "win_rate": result.win_rate,
        }

        # Save evaluation games
        eval_dir = self._history_dir / f"eval_{self._iteration:03d}"
        eval_dir.mkdir(parents=True, exist_ok=True)
        (eval_dir / "eval_games.json").write_text(records.model_dump_json(indent=2))
        (eval_dir / "eval_result.json").write_text(json.dumps(details, indent=2))

        return EvaluationResult(accepted=accepted, details=details)
```

- [ ] **Step 2: Update `arena/arena_types/__init__.py`**

```python
"""Arena type implementations."""

from pymcts.arena.arena_types.single_player import SinglePlayerArena

__all__ = ["SinglePlayerArena"]
```

- [ ] **Step 3: Update `arena/__init__.py`**

```python
from pymcts.arena.base import Arena
from pymcts.arena.engine import batched_arena
from pymcts.arena.models import EvaluationResult
from pymcts.arena.arena_types.single_player import SinglePlayerArena

__all__ = [
    "Arena",
    "EvaluationResult",
    "SinglePlayerArena",
    "batched_arena",
]
```

- [ ] **Step 4: Verify import**

Run: `python -c "from pymcts.arena import SinglePlayerArena; print('OK')"`

Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git add src/pymcts/arena/arena_types/single_player.py src/pymcts/arena/arena_types/__init__.py src/pymcts/arena/__init__.py
git commit -m "feat: implement SinglePlayerArena"
```

---

### Task 4: Implement MultiPlayerArena

**Files:**
- Create: `src/pymcts/arena/arena_types/multi_player.py`
- Modify: `src/pymcts/arena/arena_types/__init__.py`
- Modify: `src/pymcts/arena/__init__.py`

- [ ] **Step 1: Create `arena/arena_types/multi_player.py`**

```python
"""MultiPlayerArena: evaluate against top N historical players."""

import json
import logging
from pathlib import Path
from typing import Callable

from pymcts.arena.base import Arena
from pymcts.arena.config import MultiPlayerArenaConfig
from pymcts.arena.engine import batched_arena
from pymcts.arena.models import EvaluationResult
from pymcts.core.base_game import BaseGame
from pymcts.core.game_record import GameRecord, GameRecordCollection
from pymcts.core.players import BasePlayer, MCTSPlayer

logger = logging.getLogger("pymcts.arena.multi_player")


class MultiPlayerArena(Arena):
    """Arena that evaluates candidates against top N historical players.

    - play_games: player plays against itself AND top N players
    - is_candidate_better: candidate plays against top N, aggregate accept logic
    """

    def __init__(
        self,
        config: MultiPlayerArenaConfig,
        game_factory: Callable[[], BaseGame],
        arena_dir: Path,
        verbose: bool = True,
    ):
        super().__init__(config, game_factory, arena_dir, verbose)
        self._history_dir = self.arena_dir / "history"
        self._history_dir.mkdir(parents=True, exist_ok=True)
        self._best_so_far_dir = self.arena_dir / "best_so_far"
        self._iteration = 0

    def _has_best(self) -> bool:
        return (self._best_so_far_dir / "player.json").exists()

    def _load_best(self) -> BasePlayer:
        return MCTSPlayer.load(self._best_so_far_dir)

    def _save_as_best(self, player: BasePlayer) -> None:
        player.save(self._best_so_far_dir)

    def _save_to_history(self, player: BasePlayer) -> None:
        self._iteration += 1
        player.save(self._history_dir / f"iteration_{self._iteration:03d}")

    def _load_top_n(self) -> list[BasePlayer]:
        """Load the most recent top_n players from history."""
        player_dirs = sorted(self._history_dir.glob("iteration_*"))
        top_dirs = player_dirs[-self.config.top_n:]
        players = []
        for d in top_dirs:
            if (d / "player.json").exists():
                players.append(MCTSPlayer.load(d))
        return players

    def play_games(self, player: BasePlayer, num_games: int) -> GameRecordCollection:
        """Player plays against itself AND top N historical players."""
        all_records: list[GameRecord] = []

        # Self-play
        self_play_records = batched_arena(
            player_a=player,
            player_b=player,
            game_factory=self.game_factory,
            num_games=num_games,
            batch_size=self.config.batch_size,
            swap_players=self.config.swap_players,
            verbose=self.verbose,
        )
        all_records.extend(self_play_records.game_records)

        # Play against historical players
        pool = self._load_top_n()
        games_per_opponent = max(1, num_games // max(len(pool), 1))
        for opponent in pool:
            records = batched_arena(
                player_a=player,
                player_b=opponent,
                game_factory=self.game_factory,
                num_games=games_per_opponent,
                batch_size=self.config.batch_size,
                swap_players=self.config.swap_players,
                verbose=self.verbose,
            )
            all_records.extend(records.game_records)

        return GameRecordCollection(game_records=all_records)

    def is_candidate_better(self, candidate: BasePlayer) -> EvaluationResult:
        """Evaluate candidate against top N players (aggregate results)."""
        if not self._has_best():
            self._save_as_best(candidate)
            self._save_to_history(candidate)
            if self.verbose:
                print("  No previous best — accepting first candidate.")
            return EvaluationResult(accepted=True, details={"first_candidate": True})

        pool = self._load_top_n()
        all_records: list[GameRecord] = []

        for opponent in pool:
            records = batched_arena(
                player_a=candidate,
                player_b=opponent,
                game_factory=self.game_factory,
                num_games=self.config.num_games,
                batch_size=self.config.batch_size,
                swap_players=self.config.swap_players,
                verbose=self.verbose,
            )
            all_records.extend(records.game_records)

        combined = GameRecordCollection(game_records=all_records)
        accepted = combined.is_better(candidate.name, self.config.threshold)
        result = combined.evaluate(candidate.name)

        if self.verbose:
            print(f"  {candidate.name} vs {len(pool)} opponents: "
                  f"{result.wins}W/{result.losses}L ({result.win_rate:.1%})")

        if accepted:
            self._save_as_best(candidate)
            self._save_to_history(candidate)
            if self.verbose:
                print("  -> ACCEPTED")
        else:
            if self.verbose:
                print("  -> REJECTED")

        details = {
            "wins": result.wins,
            "losses": result.losses,
            "draws": result.draws,
            "total": result.total,
            "win_rate": result.win_rate,
            "pool_size": len(pool),
        }

        eval_dir = self._history_dir / f"eval_{self._iteration:03d}"
        eval_dir.mkdir(parents=True, exist_ok=True)
        (eval_dir / "eval_games.json").write_text(combined.model_dump_json(indent=2))
        (eval_dir / "eval_result.json").write_text(json.dumps(details, indent=2))

        return EvaluationResult(accepted=accepted, details=details)
```

- [ ] **Step 2: Update `arena/arena_types/__init__.py`**

```python
"""Arena type implementations."""

from pymcts.arena.arena_types.multi_player import MultiPlayerArena
from pymcts.arena.arena_types.single_player import SinglePlayerArena

__all__ = ["MultiPlayerArena", "SinglePlayerArena"]
```

- [ ] **Step 3: Update `arena/__init__.py`**

```python
from pymcts.arena.base import Arena
from pymcts.arena.engine import batched_arena
from pymcts.arena.models import EvaluationResult
from pymcts.arena.arena_types.multi_player import MultiPlayerArena
from pymcts.arena.arena_types.single_player import SinglePlayerArena

__all__ = [
    "Arena",
    "EvaluationResult",
    "MultiPlayerArena",
    "SinglePlayerArena",
    "batched_arena",
]
```

- [ ] **Step 4: Verify import**

Run: `python -c "from pymcts.arena import MultiPlayerArena; print('OK')"`

Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git add src/pymcts/arena/arena_types/multi_player.py src/pymcts/arena/arena_types/__init__.py src/pymcts/arena/__init__.py
git commit -m "feat: implement MultiPlayerArena"
```

---

### Task 5: Implement EloArena

**Files:**
- Create: `src/pymcts/arena/arena_types/elo.py`
- Modify: `src/pymcts/arena/arena_types/__init__.py`
- Modify: `src/pymcts/arena/__init__.py`

- [ ] **Step 1: Create `arena/arena_types/elo.py`**

```python
"""EloArena: Elo pool-based evaluation."""

import json
import logging
from pathlib import Path
from typing import Callable

from pymcts.arena.base import Arena
from pymcts.arena.config import EloArenaConfig
from pymcts.arena.engine import batched_arena
from pymcts.arena.models import EvaluationResult
from pymcts.core.base_game import BaseGame
from pymcts.core.game_record import GameRecord, GameRecordCollection
from pymcts.core.players import BasePlayer, MCTSPlayer, RandomPlayer
from pymcts.elo.config import MatchResult
from pymcts.elo.rating import compute_elo_against_pool

logger = logging.getLogger("pymcts.arena.elo")


class EloArena(Arena):
    """Arena that evaluates candidates using Elo ratings against a player pool.

    - play_games: player plays against pool players
    - is_candidate_better: candidate plays pool, Elo must improve by threshold
    """

    def __init__(
        self,
        config: EloArenaConfig,
        game_factory: Callable[[], BaseGame],
        arena_dir: Path,
        verbose: bool = True,
    ):
        super().__init__(config, game_factory, arena_dir, verbose)
        self._pool_dir = self.arena_dir / "pool"
        self._pool_dir.mkdir(parents=True, exist_ok=True)
        self._iteration = 0
        self._current_elo: float | None = None

        # Pool: list of (name, player, elo)
        self._pool: list[tuple[str, BasePlayer, float]] = []
        self._init_pool()

    def _init_pool(self) -> None:
        """Initialize pool with a RandomPlayer and optional seed players."""
        random_player = RandomPlayer(name="random")
        random_player.elo = 1000.0
        random_player.save(self._pool_dir / "random")
        self._pool.append(("random", random_player, 1000.0))

        if self.config.initial_pool:
            for path in self.config.initial_pool:
                loaded = MCTSPlayer.load(path)
                elo = loaded.elo if loaded.elo is not None else 1000.0
                loaded.save(self._pool_dir / loaded.name)
                self._pool.append((loaded.name, loaded, elo))

    def _pool_ratings(self) -> dict[str, float]:
        return {name: elo for name, _, elo in self._pool}

    def _play_vs_pool(self, player: BasePlayer) -> tuple[list[MatchResult], GameRecordCollection]:
        """Play player against every pool member. Returns match results and all game records."""
        match_results: list[MatchResult] = []
        all_records: list[GameRecord] = []

        for name, opponent, _ in self._pool:
            records = batched_arena(
                player_a=player,
                player_b=opponent,
                game_factory=self.game_factory,
                num_games=self.config.games_per_matchup,
                batch_size=self.config.batch_size,
                swap_players=self.config.swap_players,
                verbose=self.verbose,
            )
            all_records.extend(records.game_records)
            scores = records.scores
            match_results.append(MatchResult(
                player_a=player.name,
                player_b=name,
                wins_a=scores.get(player.name, 0),
                wins_b=scores.get(name, 0),
                draws=len(records) - scores.get(player.name, 0) - scores.get(name, 0),
            ))

        return match_results, GameRecordCollection(game_records=all_records)

    def _evict_weakest(self) -> None:
        weakest_idx, weakest_elo = None, float("inf")
        for idx, (name, _, elo) in enumerate(self._pool):
            if name != "random" and elo < weakest_elo:
                weakest_elo = elo
                weakest_idx = idx
        if weakest_idx is not None:
            evicted = self._pool.pop(weakest_idx)[0]
            if self.verbose:
                print(f"  Pool: evicted {evicted} (Elo {weakest_elo:.0f})")

    def _grow_pool(self, player: BasePlayer, elo: float) -> None:
        self._iteration += 1
        name = f"pool_iteration_{self._iteration:03d}"
        player.save(self._pool_dir / name)
        self._pool.append((name, player, elo))
        if self.config.max_pool_size is not None and len(self._pool) > self.config.max_pool_size:
            self._evict_weakest()

    def _save_elo_ratings(self) -> None:
        ratings = {name: elo for name, _, elo in self._pool}
        if self._current_elo is not None:
            ratings["_current_best"] = self._current_elo
        (self.arena_dir / "elo_ratings.json").write_text(json.dumps(ratings, indent=2))

    def play_games(self, player: BasePlayer, num_games: int) -> GameRecordCollection:
        """Player plays against pool players."""
        _, records = self._play_vs_pool(player)
        return records

    def is_candidate_better(self, candidate: BasePlayer) -> EvaluationResult:
        """Evaluate candidate Elo against the pool."""
        self._iteration += 1
        pool_ratings = self._pool_ratings()
        match_results, records = self._play_vs_pool(candidate)
        post_elo = compute_elo_against_pool(candidate.name, pool_ratings, match_results)

        if self._current_elo is None:
            self._current_elo = post_elo
            accepted = True
        else:
            accepted = post_elo >= self._current_elo + self.config.elo_threshold

        if self.verbose:
            current_str = f"{self._current_elo:.0f}" if self._current_elo is not None else "N/A"
            print(f"  Elo: {post_elo:.0f} | Current: {current_str} | "
                  f"Threshold: +{self.config.elo_threshold:.0f}")

        if accepted:
            self._current_elo = post_elo
            if self.verbose:
                print("  -> ACCEPTED")
        else:
            if self.verbose:
                print("  -> REJECTED")

        if self._iteration % self.config.pool_growth_interval == 0:
            self._grow_pool(candidate, post_elo)

        self._save_elo_ratings()

        details = {
            "post_elo": post_elo,
            "current_elo": self._current_elo,
            "threshold": self.config.elo_threshold,
            "pool_size": len(self._pool),
        }

        # Save evaluation games
        eval_dir = self.arena_dir / f"eval_{self._iteration:03d}"
        eval_dir.mkdir(parents=True, exist_ok=True)
        (eval_dir / "eval_games.json").write_text(records.model_dump_json(indent=2))
        (eval_dir / "eval_result.json").write_text(json.dumps(details, indent=2))

        return EvaluationResult(accepted=accepted, details=details)
```

- [ ] **Step 2: Update `arena/arena_types/__init__.py`**

```python
"""Arena type implementations."""

from pymcts.arena.arena_types.elo import EloArena
from pymcts.arena.arena_types.multi_player import MultiPlayerArena
from pymcts.arena.arena_types.single_player import SinglePlayerArena

__all__ = ["EloArena", "MultiPlayerArena", "SinglePlayerArena"]
```

- [ ] **Step 3: Update `arena/__init__.py`**

```python
from pymcts.arena.base import Arena
from pymcts.arena.engine import batched_arena
from pymcts.arena.models import EvaluationResult
from pymcts.arena.arena_types.elo import EloArena
from pymcts.arena.arena_types.multi_player import MultiPlayerArena
from pymcts.arena.arena_types.single_player import SinglePlayerArena

__all__ = [
    "Arena",
    "EloArena",
    "EvaluationResult",
    "MultiPlayerArena",
    "SinglePlayerArena",
    "batched_arena",
]
```

- [ ] **Step 4: Verify import**

Run: `python -c "from pymcts.arena import EloArena, MultiPlayerArena, SinglePlayerArena, Arena, batched_arena, EvaluationResult; print('OK')"`

Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git add src/pymcts/arena/arena_types/elo.py src/pymcts/arena/arena_types/__init__.py src/pymcts/arena/__init__.py
git commit -m "feat: implement EloArena"
```

---

### Task 6: Rewrite trainer to use Arena objects

**Files:**
- Modify: `src/pymcts/core/trainer.py`

- [ ] **Step 1: Rewrite `src/pymcts/core/trainer.py` entirely**

Replace the entire file with:

```python
"""Main training pipeline: self-play -> train -> evaluate loop."""

import json
import logging
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Callable

from pymcts.arena.base import Arena
from pymcts.core.base_game import BaseGame
from pymcts.core.base_neural_net import BaseNeuralNet
from pymcts.core.config import MCTSConfig, PathsConfig, TrainingConfig
from pymcts.core.data import examples_from_records
from pymcts.core.players import GreedyMCTSPlayer

logger = logging.getLogger("pymcts.core.trainer")


def _create_run_dir(paths: PathsConfig) -> Path:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run_dir = paths.trainings / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _save_run_config(
    run_dir: Path,
    net: BaseNeuralNet,
    mcts_config: MCTSConfig,
    training_config: TrainingConfig,
) -> None:
    net_class = type(net)
    config = {
        "net_class": f"{net_class.__module__}.{net_class.__qualname__}",
        "mcts_config": mcts_config.model_dump(),
        "training_config": training_config.model_dump(),
    }
    (run_dir / "run_config.json").write_text(json.dumps(config, indent=2))


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
    """Run the full AlphaZero training pipeline.

    Each iteration: self-play -> train -> evaluate (accept/reject).
    """
    paths = paths_config or PathsConfig()
    run_dir = _create_run_dir(paths)

    _save_run_config(run_dir, net, mcts_config, training_config)

    if verbose:
        print(f"Training run directory: {run_dir}")
        print(f"Model parameters: {sum(p.numel() for p in net.parameters()):,}\n")

    replay_buffer: deque[list] = deque(maxlen=training_config.replay_buffer_size)

    for iteration in range(1, training_config.num_iterations + 1):
        if verbose:
            print(f"{'=' * 60}\nIteration {iteration}/{training_config.num_iterations}\n{'=' * 60}")

        iter_dir = run_dir / f"iteration_{iteration:03d}"
        iter_dir.mkdir(parents=True, exist_ok=True)
        pre_checkpoint = str(iter_dir / "pre_training.pt")
        net.save_checkpoint(pre_checkpoint)

        # 1. Self-play
        if verbose:
            print("\n[1/3] Self-play...")
        player = GreedyMCTSPlayer(net, mcts_config, name="self_play")
        records = self_play_arena.play_games(player, num_games=training_config.num_self_play_games)
        (iter_dir / "self_play_games.json").write_text(records.model_dump_json(indent=2))

        new_examples = examples_from_records(records, lambda cfg: game_factory())
        replay_buffer.append(new_examples)
        all_examples = [ex for batch in replay_buffer for ex in batch]

        if verbose:
            print(f"  {len(records)} games, {len(new_examples)} examples")
            print(f"  Replay buffer: {len(replay_buffer)} iterations, {len(all_examples)} examples")

        # 2. Train
        if verbose:
            print("\n[2/3] Training...")
        net.train_on_examples(
            all_examples,
            num_epochs=training_config.num_epochs,
            batch_size=training_config.batch_size,
            learning_rate=training_config.learning_rate,
            weight_decay=training_config.weight_decay,
            verbose=verbose,
        )
        net.save_checkpoint(str(iter_dir / "post_training.pt"))

        # 3. Evaluate
        if verbose:
            print("\n[3/3] Arena evaluation...")
        candidate = GreedyMCTSPlayer(net, mcts_config, name="candidate")
        result = eval_arena.is_candidate_better(candidate)

        if not result.accepted:
            net.load_checkpoint(pre_checkpoint)

        # Save iteration data
        iteration_data = {
            "iteration": iteration,
            "training": {"num_examples": len(all_examples)},
            "evaluation": result.details,
            "accepted": result.accepted,
        }
        (iter_dir / "iteration_data.json").write_text(json.dumps(iteration_data, indent=2))

        if verbose:
            print()

    if verbose:
        print(f"Training complete!\nRun directory: {run_dir}")
```

- [ ] **Step 2: Verify the trainer module imports cleanly**

Run: `python -c "from pymcts.core.trainer import train; print('OK')"`

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add src/pymcts/core/trainer.py
git commit -m "refactor: rewrite trainer to use Arena objects, remove all arena logic"
```

---

### Task 7: Clean up core package — remove old arena configs and exports

**Files:**
- Modify: `src/pymcts/core/config.py:45-59`
- Modify: `src/pymcts/core/__init__.py`
- Modify: `src/pymcts/core/models.py`

- [ ] **Step 1: Remove `ArenaConfig` and `EloArenaConfig` from `core/config.py`**

Delete lines 44-59 (the `ArenaConfig` and `EloArenaConfig` classes, including the blank line between them). The file should end after `TrainingConfig` (line 43 currently).

After editing, `core/config.py` should contain only: `PathsConfig`, `MCTSConfig`, `TrainingConfig`.

- [ ] **Step 2: Update `core/__init__.py`**

Replace the entire file with:

```python
from pymcts.core.base_game import BaseGame, Board2DGame, GameState
from pymcts.core.base_neural_net import BaseNeuralNet
from pymcts.core.mcts import MCTS, MCTSNode
from pymcts.core.self_play import batched_self_play
from pymcts.core.players import BasePlayer, RandomPlayer, MCTSPlayer, GreedyMCTSPlayer
from pymcts.core.game_record import MoveRecord, GameRecord, GameRecordCollection, EvalResult
from pymcts.core.data import examples_from_records
from pymcts.core.config import MCTSConfig, TrainingConfig, PathsConfig
from pymcts.core.models import *  # noqa: F401, F403 — re-export all core models
```

Note: `batched_arena`, `ArenaConfig`, `EloArenaConfig` removed.

- [ ] **Step 3: Update `core/models.py`**

Replace the entire file with:

```python
"""All public Pydantic models from the core package."""

from pymcts.core.config import (
    MCTSConfig,
    PathsConfig,
    TrainingConfig,
)
from pymcts.core.game_record import (
    EvalResult,
    GameRecord,
    GameRecordCollection,
    MoveRecord,
)

__all__ = [
    # config
    "MCTSConfig",
    "PathsConfig",
    "TrainingConfig",
    # game_record
    "EvalResult",
    "GameRecord",
    "GameRecordCollection",
    "MoveRecord",
]
```

- [ ] **Step 4: Verify all core imports still work**

Run: `python -c "from pymcts.core import MCTSConfig, TrainingConfig, PathsConfig, BaseGame, BaseNeuralNet, MCTS, batched_self_play, BasePlayer, MoveRecord, GameRecord, GameRecordCollection, EvalResult; print('OK')"`

Expected: `OK`

- [ ] **Step 5: Verify old arena imports are gone**

Run: `python -c "from pymcts.core.config import ArenaConfig" 2>&1`

Expected: ImportError

- [ ] **Step 6: Commit**

```bash
git add src/pymcts/core/config.py src/pymcts/core/__init__.py src/pymcts/core/models.py
git commit -m "refactor: remove ArenaConfig and EloArenaConfig from core package"
```

---

### Task 8: Update README.md

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Update the Quick Start code block (lines 25-44)**

Replace:

```python
from pymcts.core.config import MCTSConfig, TrainingConfig, ArenaConfig
from pymcts.core.trainer import train

board_config = BoardConfig(size=5)
net = BridgitNet(board_config=board_config, net_config=NeuralNetConfig())

train(
    game_factory=lambda: BridgitGame(board_config),
    net=net,
    mcts_config=MCTSConfig(num_simulations=50),
    training_config=TrainingConfig(num_iterations=3, num_self_play_games=10),
    arena_config=ArenaConfig(num_games=10),
    game_type="bridgit",
    game_config=board_config.model_dump(),
)
```

With:

```python
from pymcts.core.config import MCTSConfig, TrainingConfig
from pymcts.core.trainer import train
from pymcts.arena import SinglePlayerArena
from pymcts.arena.config import SinglePlayerArenaConfig

board_config = BoardConfig(size=5)
net = BridgitNet(board_config=board_config, net_config=NeuralNetConfig())
game_factory = lambda: BridgitGame(board_config)

arena_config = SinglePlayerArenaConfig(num_games=10)
self_play_arena = SinglePlayerArena(arena_config, game_factory, arena_dir=Path("trainings/self_play"))
eval_arena = SinglePlayerArena(arena_config, game_factory, arena_dir=Path("trainings/eval"))

train(
    game_factory=game_factory,
    net=net,
    mcts_config=MCTSConfig(num_simulations=50),
    training_config=TrainingConfig(num_iterations=3, num_self_play_games=10),
    self_play_arena=self_play_arena,
    eval_arena=eval_arena,
)
```

Also add `from pathlib import Path` to the imports at the top of the code block.

- [ ] **Step 2: Update Architecture tree (lines 57-76)**

Replace the `core/` section to remove `arena.py` and `config.py` arena entries, and add the `arena/` package:

```
src/pymcts/
├── core/                        # Generic engine (game-agnostic)
│   ├── base_game.py             # BaseGame, Board2DGame, GameState ABCs
│   ├── base_neural_net.py       # BaseNeuralNet(ABC, nn.Module)
│   ├── mcts.py                  # MCTS with integer actions
│   ├── self_play.py             # Batched self-play
│   ├── trainer.py               # AlphaZero training loop
│   ├── players.py               # RandomPlayer, MCTSPlayer
│   ├── game_record.py           # Game recording and evaluation
│   ├── data.py                  # Training data extraction
│   └── config.py                # MCTSConfig, TrainingConfig, PathsConfig
├── arena/                       # Arena evaluation system
│   ├── base.py                  # Arena ABC
│   ├── engine.py                # batched_arena game-playing engine
│   ├── config.py                # Arena configuration models
│   └── arena_types/             # Arena implementations
│       ├── single_player.py     # SinglePlayerArena
│       ├── multi_player.py      # MultiPlayerArena
│       └── elo.py               # EloArena
└── games/
    └── bridgit/                 # Bridgit implementation
        ├── game.py              # BridgitGame(Board2DGame)
        ├── neural_net.py        # BridgitNet(BaseNeuralNet) — ResNet
        ├── config.py            # BoardConfig, NeuralNetConfig
        ├── player.py            # Player enum
        ├── union_find.py        # Win detection
        └── visualizer.py        # Plotly visualization
```

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: update README with new arena package structure and imports"
```

---

### Task 9: Update quickstart.md

**Files:**
- Modify: `docs/content/getting-started/quickstart.md`

- [ ] **Step 1: Update the Train code block (lines 7-26)**

Replace the entire code block with:

```python
from pathlib import Path
from pymcts.games.bridgit.game import BridgitGame
from pymcts.games.bridgit.config import BoardConfig, NeuralNetConfig
from pymcts.games.bridgit.neural_net import BridgitNet
from pymcts.core.config import MCTSConfig, TrainingConfig
from pymcts.core.trainer import train
from pymcts.arena import SinglePlayerArena
from pymcts.arena.config import SinglePlayerArenaConfig

board_config = BoardConfig(size=5)
net = BridgitNet(board_config=board_config, net_config=NeuralNetConfig())
game_factory = lambda: BridgitGame(board_config)

arena_config = SinglePlayerArenaConfig(num_games=10)
self_play_arena = SinglePlayerArena(arena_config, game_factory, arena_dir=Path("trainings/self_play"))
eval_arena = SinglePlayerArena(arena_config, game_factory, arena_dir=Path("trainings/eval"))

train(
    game_factory=game_factory,
    net=net,
    mcts_config=MCTSConfig(num_simulations=50),
    training_config=TrainingConfig(num_iterations=3, num_self_play_games=10),
    self_play_arena=self_play_arena,
    eval_arena=eval_arena,
)
```

- [ ] **Step 2: Update the Watch a game code block (lines 42-62)**

Replace the import line:
```python
from pymcts.core.arena import batched_arena
```
With:
```python
from pymcts.arena import batched_arena
```

- [ ] **Step 3: Commit**

```bash
git add docs/content/getting-started/quickstart.md
git commit -m "docs: update quickstart with new arena imports and train() signature"
```

---

### Task 10: Update training.md

**Files:**
- Modify: `docs/content/guide/training.md`

- [ ] **Step 1: Update the train() function example (lines 7-19)**

Replace with:

```python
from pymcts.core.trainer import train
from pymcts.arena import SinglePlayerArena
from pymcts.arena.config import SinglePlayerArenaConfig

train(
    game_factory=lambda: MyGame(),       # creates fresh game instances
    net=my_net,                          # neural network to train
    mcts_config=mcts_config,             # MCTS settings
    training_config=training_config,     # training loop settings
    self_play_arena=self_play_arena,     # arena for generating training data
    eval_arena=eval_arena,               # arena for accept/reject evaluation
)
```

- [ ] **Step 2: Update the description paragraph (lines 21-23)**

Replace:

```
`game_factory` is a callable that returns a new game instance. This is called for every self-play game and every arena game.

The `arena` parameter accepts either `ArenaConfig` (head-to-head evaluation) or `EloArenaConfig` (Elo pool-based evaluation). See below for details on both.
```

With:

```
`game_factory` is a callable that returns a new game instance.

The `self_play_arena` and `eval_arena` parameters accept any `Arena` subclass. `self_play_arena` generates training data via `play_games()`. `eval_arena` decides whether to accept the new model via `is_candidate_better()`. See below for the three available arena types.
```

- [ ] **Step 3: Replace the ArenaConfig section (lines 67-79) with SinglePlayerArenaConfig**

Replace:

```markdown
### ArenaConfig

Controls head-to-head model comparison. The new model plays against the previous version and must exceed the win rate threshold to be accepted.

\`\`\`python
from pymcts.core.config import ArenaConfig

arena_config = ArenaConfig(
    num_games=40,       # games to play per evaluation
    threshold=0.55,     # win rate needed to accept new model
    swap_players=True,  # play both sides for fairness
)
\`\`\`
```

With:

```markdown
### SinglePlayerArenaConfig

Controls head-to-head model comparison. The candidate plays against the current best and must exceed the win rate threshold to be accepted.

\`\`\`python
from pymcts.arena.config import SinglePlayerArenaConfig

arena_config = SinglePlayerArenaConfig(
    num_games=40,       # games to play per evaluation
    threshold=0.55,     # win rate needed to accept new model
    swap_players=True,  # play both sides for fairness
    batch_size=8,       # concurrent games
)
\`\`\`

### MultiPlayerArenaConfig

Evaluates against the top N historical players instead of just the current best. Also generates richer training data by playing against historical opponents during self-play.

\`\`\`python
from pymcts.arena.config import MultiPlayerArenaConfig

arena_config = MultiPlayerArenaConfig(
    num_games=40,
    threshold=0.55,
    swap_players=True,
    batch_size=8,
    top_n=5,            # number of historical opponents
)
\`\`\`
```

- [ ] **Step 4: Update the EloArenaConfig section (lines 81-107)**

Replace the import line:
```python
from pymcts.core.config import EloArenaConfig
```
With:
```python
from pymcts.arena.config import EloArenaConfig
```

Update the tip box:
```
!!! tip "When to use EloArenaConfig"
    Use `EloArenaConfig` when head-to-head comparison is too noisy or when you want
    to measure improvement against a diverse field rather than just the previous model.
```
To:
```
!!! tip "When to use EloArena"
    Use `EloArena` when head-to-head comparison is too noisy or when you want
    to measure improvement against a diverse field rather than just the previous model.
```

- [ ] **Step 5: Update the Resuming training example (lines 155-169)**

Replace:

```python
train(
    game_factory=lambda: MyGame(),
    net=net,
    mcts_config=mcts_config,
    training_config=TrainingConfig(num_iterations=20),  # 20 more
    arena=arena_config,
    game_type="mygame",
)
```

With:

```python
train(
    game_factory=game_factory,
    net=net,
    mcts_config=mcts_config,
    training_config=TrainingConfig(num_iterations=20),  # 20 more
    self_play_arena=self_play_arena,
    eval_arena=eval_arena,
)
```

- [ ] **Step 6: Update the hyperparameter starting point (lines 188-191)**

Replace:
```python
ArenaConfig(num_games=20, threshold=0.55)
```
With:
```python
SinglePlayerArenaConfig(num_games=20, threshold=0.55)
```

- [ ] **Step 7: Update common issues table (lines 206-211)**

Replace all occurrences of `ArenaConfig` with `SinglePlayerArenaConfig` and `EloArenaConfig` stays the same.

- [ ] **Step 8: Commit**

```bash
git add docs/content/guide/training.md
git commit -m "docs: update training guide with new arena types and imports"
```

---

### Task 11: Update evaluation.md

**Files:**
- Modify: `docs/content/guide/evaluation.md`

- [ ] **Step 1: Update the Arena code example (lines 10-11)**

Replace:
```python
from pymcts.core.arena import batched_arena
from pymcts.core.players import GreedyMCTSPlayer, RandomPlayer
from pymcts.core.config import MCTSConfig
```
With:
```python
from pymcts.arena import batched_arena
from pymcts.core.players import GreedyMCTSPlayer, RandomPlayer
from pymcts.core.config import MCTSConfig
```

- [ ] **Step 2: Update the batched self-play import (line 149)**

Replace:
```python
from pymcts.core.self_play import batched_self_play
```

This import is still valid (self_play stays in core), so no change needed. Just verify it.

- [ ] **Step 3: Commit**

```bash
git add docs/content/guide/evaluation.md
git commit -m "docs: update evaluation guide with new arena import path"
```

---

### Task 12: Update reference docs (config.md and engine.md)

**Files:**
- Modify: `docs/content/reference/config.md`
- Modify: `docs/content/reference/engine.md`

- [ ] **Step 1: Update `config.md`**

Replace the entire file with:

```markdown
# Configuration

All configuration classes used by the engine.

## MCTSConfig

::: pymcts.core.config.MCTSConfig

## TrainingConfig

::: pymcts.core.config.TrainingConfig

## PathsConfig

::: pymcts.core.config.PathsConfig

## Arena Configuration

Arena configs have moved to the `pymcts.arena.config` module:

::: pymcts.arena.config.SinglePlayerArenaConfig

::: pymcts.arena.config.MultiPlayerArenaConfig

::: pymcts.arena.config.EloArenaConfig
```

- [ ] **Step 2: Update `engine.md`**

Replace the Arena section (lines 19-21):
```markdown
## Arena

::: pymcts.core.arena.batched_arena
```
With:
```markdown
## Arena

::: pymcts.arena.base.Arena

::: pymcts.arena.arena_types.single_player.SinglePlayerArena

::: pymcts.arena.arena_types.multi_player.MultiPlayerArena

::: pymcts.arena.arena_types.elo.EloArena

### Game Engine

::: pymcts.arena.engine.batched_arena
```

- [ ] **Step 3: Commit**

```bash
git add docs/content/reference/config.md docs/content/reference/engine.md
git commit -m "docs: update reference docs with new arena module paths"
```

---

### Task 13: Update architecture.md

**Files:**
- Modify: `docs/content/concepts/architecture.md`

- [ ] **Step 1: Update the architecture diagram (lines 7-33)**

Replace:
```
┌─────────────────────────────────────────────────────┐
│                    Engine (core/)                     │
│                                                      │
│  MCTS ──> Self-Play ──> Trainer ──> Arena            │
```

With:
```
┌─────────────────────────────────────────────────────┐
│                    Engine (core/)                     │
│                                                      │
│  MCTS ──> Self-Play ──> Trainer                      │
│                            │                         │
│                            ▼                         │
│                      Arena (arena/)                   │
│           SinglePlayer / MultiPlayer / Elo            │
```

Keep the rest of the diagram (the "Sees only:" section and the game boundary) the same.

- [ ] **Step 2: Commit**

```bash
git add docs/content/concepts/architecture.md
git commit -m "docs: update architecture diagram with arena package"
```

---

### Task 14: Update notebooks

**Files:**
- Modify: `notebooks/training.ipynb`
- Modify: `notebooks/arena.ipynb`
- Modify: `notebooks/players.ipynb`

- [ ] **Step 1: Update `training.ipynb`**

Find the cell containing:
```python
from pymcts.core.config import MCTSConfig, TrainingConfig, ArenaConfig
```

Replace `ArenaConfig` import with:
```python
from pymcts.core.config import MCTSConfig, TrainingConfig
from pymcts.arena import SinglePlayerArena
from pymcts.arena.config import SinglePlayerArenaConfig
```

Find the cell that calls `train(...)` and update it to use the new signature: pass `self_play_arena=` and `eval_arena=` instead of `arena=`. Remove `game_type` and `game_config` parameters if present.

Use the NotebookEdit tool for these changes.

- [ ] **Step 2: Update `arena.ipynb`**

Find the cell containing:
```python
from pymcts.core.arena import batched_arena
```

Replace with:
```python
from pymcts.arena import batched_arena
```

Use the NotebookEdit tool.

- [ ] **Step 3: Update `players.ipynb`**

Find the cell containing:
```python
from pymcts.core.arena import batched_arena
```

Replace with:
```python
from pymcts.arena import batched_arena
```

Use the NotebookEdit tool.

- [ ] **Step 4: Commit**

```bash
git add notebooks/training.ipynb notebooks/arena.ipynb notebooks/players.ipynb
git commit -m "docs: update notebooks with new arena import paths"
```

---

### Task 15: Final verification

**Files:** None (verification only)

- [ ] **Step 1: Verify all arena package imports**

Run:
```bash
python -c "
from pymcts.arena import Arena, SinglePlayerArena, MultiPlayerArena, EloArena, EvaluationResult, batched_arena
from pymcts.arena.config import SinglePlayerArenaConfig, MultiPlayerArenaConfig, EloArenaConfig
from pymcts.arena.models import EvaluationResult
from pymcts.arena.engine import batched_arena
print('arena package: OK')
"
```

Expected: `arena package: OK`

- [ ] **Step 2: Verify core package still works**

Run:
```bash
python -c "
from pymcts.core import MCTSConfig, TrainingConfig, PathsConfig, BaseGame, BaseNeuralNet, MCTS, batched_self_play, BasePlayer, MoveRecord, GameRecord, GameRecordCollection, EvalResult
from pymcts.core.trainer import train
print('core package: OK')
"
```

Expected: `core package: OK`

- [ ] **Step 3: Verify old imports are gone**

Run:
```bash
python -c "
try:
    from pymcts.core.config import ArenaConfig
    print('FAIL: ArenaConfig should not exist in core.config')
except ImportError:
    print('ArenaConfig correctly removed from core.config')

try:
    from pymcts.core.arena import batched_arena
    print('FAIL: core.arena should not exist')
except (ImportError, ModuleNotFoundError):
    print('core.arena correctly removed')
"
```

Expected: Both "correctly removed" messages.

- [ ] **Step 4: Verify no stale imports across codebase**

Run:
```bash
grep -r "from pymcts.core.arena import" src/ --include="*.py"
grep -r "from pymcts.core.config import.*ArenaConfig" src/ --include="*.py"
```

Expected: No matches.

- [ ] **Step 5: Commit (if any fixes needed)**

Only if previous steps revealed issues that needed fixing.
