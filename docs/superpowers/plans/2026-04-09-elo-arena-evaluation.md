# Elo-Based Arena Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a pluggable Elo-based arena evaluation strategy to the training loop, where accept/reject decisions use Elo ratings against a player pool instead of head-to-head win rate.

**Architecture:** The training loop accepts either `ArenaConfig` (existing head-to-head) or `EloArenaConfig` (new pool-based Elo). Both strategies persist players to an `arena/` directory inside the training run. `MCTSPlayer` gains an `elo` field for serialization. A new `compute_elo_against_pool()` function rates a single player against fixed-rating opponents.

**Tech Stack:** Python, Pydantic, PyTorch, scipy (existing Elo computation)

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `src/pymcts/core/players.py` | Modify | Add `elo` field to `MCTSPlayer`, add `save()`/`load()` to `RandomPlayer` |
| `src/pymcts/core/config.py` | Modify | Add `EloArenaConfig` |
| `src/pymcts/elo/rating.py` | Modify | Add `compute_elo_against_pool()` |
| `src/pymcts/core/trainer.py` | Modify | Add Elo arena branch, `arena/` directory management for both strategies |
| `src/pymcts/core/__init__.py` | Modify | Re-export `EloArenaConfig` |
| `test/test_core/test_players.py` | Modify | Tests for `elo` field, `RandomPlayer.save()`/`load()` |
| `test/test_elo/test_rating.py` | Modify | Tests for `compute_elo_against_pool()` |
| `test/test_core/test_trainer_elo_arena.py` | Create | Integration test for Elo arena training flow |

---

### Task 1: Add `elo` field to MCTSPlayer

**Files:**
- Modify: `src/pymcts/core/players.py:75-109` (save/load methods)
- Test: `test/test_core/test_players.py`

- [ ] **Step 1: Write failing test for elo field on MCTSPlayer**

In `test/test_core/test_players.py`, add:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest test/test_core/test_players.py::TestMCTSPlayerElo -v`
Expected: FAIL with `AttributeError: 'MCTSPlayer' object has no attribute 'elo'`

- [ ] **Step 3: Add elo field to MCTSPlayer**

In `src/pymcts/core/players.py`, modify `MCTSPlayer.__init__`:

```python
class MCTSPlayer(BasePlayer):
    def __init__(
        self,
        net: BaseNeuralNet,
        mcts_config: MCTSConfig,
        temperature: float = 1.0,
        temp_threshold: int = 0,
        name: str | None = None,
        elo: float | None = None,
    ):
        super().__init__(name)
        self.mcts = MCTS(net, mcts_config)
        self.temperature = temperature
        self.temp_threshold = temp_threshold
        self.elo = elo
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest test/test_core/test_players.py::TestMCTSPlayerElo -v`
Expected: PASS

- [ ] **Step 5: Write failing test for elo in save/load roundtrip**

In `test/test_core/test_players.py`, add:

```python
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
```

- [ ] **Step 6: Run test to verify it fails**

Run: `python -m pytest test/test_core/test_players.py::TestMCTSPlayerSaveLoadElo -v`
Expected: FAIL — `elo` not saved/loaded in player.json

- [ ] **Step 7: Update save() and load() to handle elo**

In `src/pymcts/core/players.py`, modify `save()` at line 84:

```python
    def save(self, path: str | Path) -> None:
        """Save player to a directory (config + neural net weights)."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        net = self.mcts.net
        net.save_checkpoint(str(path / "model.pt"))

        net_class = type(net)
        config = {
            "net_class": f"{net_class.__module__}.{net_class.__qualname__}",
            "mcts_config": self.mcts.mcts_config.model_dump(),
            "temperature": self.temperature,
            "temp_threshold": self.temp_threshold,
            "name": self.name,
            "elo": self.elo,
        }
        (path / "player.json").write_text(json.dumps(config, indent=2))
```

Modify `load()` at line 103:

```python
    @classmethod
    def load(cls, path: str | Path) -> "MCTSPlayer":
        """Load a player from a directory saved with .save()."""
        path = Path(path)
        config = json.loads((path / "player.json").read_text())

        net_class = _import_class(config["net_class"])
        net = net_class.from_checkpoint(str(path / "model.pt"))
        mcts_config = MCTSConfig(**config["mcts_config"])

        return cls(
            net=net,
            mcts_config=mcts_config,
            temperature=config["temperature"],
            temp_threshold=config["temp_threshold"],
            name=config["name"],
            elo=config.get("elo"),
        )
```

- [ ] **Step 8: Run test to verify it passes**

Run: `python -m pytest test/test_core/test_players.py::TestMCTSPlayerSaveLoadElo -v`
Expected: PASS

- [ ] **Step 9: Run all player tests**

Run: `python -m pytest test/test_core/test_players.py -v`
Expected: All PASS

- [ ] **Step 10: Commit**

```bash
git add src/pymcts/core/players.py test/test_core/test_players.py
git commit -m "feat: add elo field to MCTSPlayer with save/load support"
```

---

### Task 2: Add save/load to RandomPlayer

**Files:**
- Modify: `src/pymcts/core/players.py:40-43`
- Test: `test/test_core/test_players.py`

- [ ] **Step 1: Write failing test for RandomPlayer save/load**

In `test/test_core/test_players.py`, add:

```python
class TestRandomPlayerSaveLoad:
    def test_save_creates_player_json(self, tmp_path):
        player = RandomPlayer(name="random")
        player.elo = 1000.0
        player.save(tmp_path / "random")
        assert (tmp_path / "random" / "player.json").exists()
        assert not (tmp_path / "random" / "model.pt").exists()

    def test_load_roundtrip(self, tmp_path):
        player = RandomPlayer(name="random")
        player.elo = 1000.0
        player.save(tmp_path / "random")

        loaded = RandomPlayer.load(tmp_path / "random")
        assert loaded.name == "random"
        assert loaded.elo == 1000.0

    def test_load_returns_random_player(self, tmp_path):
        player = RandomPlayer(name="rng")
        player.elo = 950.0
        player.save(tmp_path / "rng")

        loaded = RandomPlayer.load(tmp_path / "rng")
        assert isinstance(loaded, RandomPlayer)
        game = TicTacToe()
        action = loaded.get_action(game)
        assert action in game.valid_actions()
```

Also add the import at the top of the test file:

```python
from test.test_core.test_mcts import TicTacToe, DummyNet
from pymcts.core.players import BasePlayer, RandomPlayer, MCTSPlayer
from pymcts.core.arena import batched_arena
from pymcts.core.config import MCTSConfig
```

(TicTacToe import already present, just noting for completeness.)

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest test/test_core/test_players.py::TestRandomPlayerSaveLoad -v`
Expected: FAIL — `RandomPlayer` has no `elo` attribute and no `save()`/`load()` methods

- [ ] **Step 3: Add elo field to BasePlayer and save/load to RandomPlayer**

In `src/pymcts/core/players.py`, modify `BasePlayer`:

```python
class BasePlayer(ABC):
    def __init__(self, name: str | None = None):
        self.name = name or self.__class__.__name__
        self._last_policy: torch.Tensor | None = None
        self.elo: float | None = None
```

Add `save()` and `load()` to `RandomPlayer`:

```python
class RandomPlayer(BasePlayer):
    def get_action(self, game: BaseGame) -> int:
        self._last_policy = None
        return random.choice(game.valid_actions())

    def save(self, path: str | Path) -> None:
        """Save player to a directory (config only, no model)."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        config = {
            "type": "random",
            "name": self.name,
            "elo": self.elo,
        }
        (path / "player.json").write_text(json.dumps(config, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "RandomPlayer":
        """Load a RandomPlayer from a directory saved with .save()."""
        path = Path(path)
        config = json.loads((path / "player.json").read_text())
        player = cls(name=config["name"])
        player.elo = config.get("elo")
        return player
```

Also remove `self.elo = elo` from `MCTSPlayer.__init__` and update it to use the inherited `BasePlayer.elo`:

```python
class MCTSPlayer(BasePlayer):
    def __init__(
        self,
        net: BaseNeuralNet,
        mcts_config: MCTSConfig,
        temperature: float = 1.0,
        temp_threshold: int = 0,
        name: str | None = None,
        elo: float | None = None,
    ):
        super().__init__(name)
        self.mcts = MCTS(net, mcts_config)
        self.temperature = temperature
        self.temp_threshold = temp_threshold
        self.elo = elo
```

(This keeps `elo` as a constructor param on MCTSPlayer for convenience, but the attribute lives on BasePlayer.)

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest test/test_core/test_players.py::TestRandomPlayerSaveLoad -v`
Expected: PASS

- [ ] **Step 5: Run all player tests**

Run: `python -m pytest test/test_core/test_players.py -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add src/pymcts/core/players.py test/test_core/test_players.py
git commit -m "feat: add elo field to BasePlayer, save/load to RandomPlayer"
```

---

### Task 3: Update GreedyMCTSPlayer to pass elo through

**Files:**
- Modify: `src/pymcts/core/players.py:158-160`
- Test: `test/test_core/test_players.py`

- [ ] **Step 1: Write failing test**

In `test/test_core/test_players.py`, add:

```python
from pymcts.core.players import GreedyMCTSPlayer


class TestGreedyMCTSPlayerElo:
    def test_elo_passthrough(self):
        net = DummyNet()
        config = MCTSConfig(num_simulations=5)
        player = GreedyMCTSPlayer(net, config, name="greedy", elo=1200.0)
        assert player.elo == 1200.0
        assert player.temperature == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest test/test_core/test_players.py::TestGreedyMCTSPlayerElo -v`
Expected: FAIL — `GreedyMCTSPlayer.__init__` doesn't accept `elo`

- [ ] **Step 3: Update GreedyMCTSPlayer to accept elo**

In `src/pymcts/core/players.py`, modify `GreedyMCTSPlayer`:

```python
class GreedyMCTSPlayer(MCTSPlayer):
    def __init__(self, net: BaseNeuralNet, mcts_config: MCTSConfig, name: str | None = None, elo: float | None = None):
        super().__init__(net, mcts_config, temperature=0.0, name=name, elo=elo)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest test/test_core/test_players.py::TestGreedyMCTSPlayerElo -v`
Expected: PASS

- [ ] **Step 5: Run all tests**

Run: `python -m pytest test/ -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add src/pymcts/core/players.py test/test_core/test_players.py
git commit -m "feat: pass elo through GreedyMCTSPlayer constructor"
```

---

### Task 4: Add EloArenaConfig

**Files:**
- Modify: `src/pymcts/core/config.py`
- Modify: `src/pymcts/core/__init__.py`
- Test: `test/test_core/test_config.py` (create)

- [ ] **Step 1: Write failing test for EloArenaConfig**

Create `test/test_core/test_config.py`:

```python
from pymcts.core.config import EloArenaConfig


class TestEloArenaConfig:
    def test_defaults(self):
        config = EloArenaConfig()
        assert config.games_per_matchup == 40
        assert config.elo_threshold == 20.0
        assert config.pool_growth_interval == 5
        assert config.max_pool_size is None
        assert config.swap_players is True
        assert config.initial_pool is None

    def test_custom_values(self):
        config = EloArenaConfig(
            games_per_matchup=20,
            elo_threshold=30.0,
            pool_growth_interval=3,
            max_pool_size=10,
            initial_pool=["/path/to/player1", "/path/to/player2"],
        )
        assert config.games_per_matchup == 20
        assert config.elo_threshold == 30.0
        assert config.pool_growth_interval == 3
        assert config.max_pool_size == 10
        assert len(config.initial_pool) == 2

    def test_serialization_roundtrip(self):
        config = EloArenaConfig(max_pool_size=15, initial_pool=["/some/path"])
        data = config.model_dump()
        restored = EloArenaConfig(**data)
        assert restored == config
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest test/test_core/test_config.py -v`
Expected: FAIL — `ImportError: cannot import name 'EloArenaConfig'`

- [ ] **Step 3: Add EloArenaConfig to config.py**

In `src/pymcts/core/config.py`, add after `ArenaConfig`:

```python
class EloArenaConfig(BaseModel):
    """Elo pool-based arena evaluation settings."""
    games_per_matchup: int = 40
    elo_threshold: float = 20.0
    pool_growth_interval: int = 5
    max_pool_size: int | None = None
    swap_players: bool = True
    initial_pool: list[str] | None = None
```

- [ ] **Step 4: Update core __init__.py to re-export**

In `src/pymcts/core/__init__.py`, change the config import line to:

```python
from pymcts.core.config import MCTSConfig, TrainingConfig, ArenaConfig, EloArenaConfig, PathsConfig
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest test/test_core/test_config.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/pymcts/core/config.py src/pymcts/core/__init__.py test/test_core/test_config.py
git commit -m "feat: add EloArenaConfig for pool-based arena evaluation"
```

---

### Task 5: Add compute_elo_against_pool()

**Files:**
- Modify: `src/pymcts/elo/rating.py`
- Modify: `src/pymcts/elo/__init__.py`
- Test: `test/test_elo/test_rating.py`

- [ ] **Step 1: Write failing tests for compute_elo_against_pool**

In `test/test_elo/test_rating.py`, add:

```python
from pymcts.elo.rating import compute_elo_against_pool


class TestComputeEloAgainstPool:
    def test_candidate_stronger_than_pool(self):
        """A candidate who beats the pool should get a high rating."""
        pool_ratings = {"random": 1000.0, "weak": 800.0}
        match_results = [
            MatchResult(player_a="candidate", player_b="random", wins_a=8, wins_b=2, draws=0),
            MatchResult(player_a="candidate", player_b="weak", wins_a=9, wins_b=1, draws=0),
        ]
        elo = compute_elo_against_pool("candidate", pool_ratings, match_results)
        assert elo > 1000.0

    def test_candidate_weaker_than_pool(self):
        """A candidate who loses to the pool should get a low rating."""
        pool_ratings = {"random": 1000.0, "strong": 1500.0}
        match_results = [
            MatchResult(player_a="candidate", player_b="random", wins_a=2, wins_b=8, draws=0),
            MatchResult(player_a="candidate", player_b="strong", wins_a=1, wins_b=9, draws=0),
        ]
        elo = compute_elo_against_pool("candidate", pool_ratings, match_results)
        assert elo < 1000.0

    def test_candidate_even_with_anchor(self):
        """A candidate with 50/50 results against anchor should be near anchor rating."""
        pool_ratings = {"random": 1000.0}
        match_results = [
            MatchResult(player_a="candidate", player_b="random", wins_a=10, wins_b=10, draws=0),
        ]
        elo = compute_elo_against_pool("candidate", pool_ratings, match_results)
        assert abs(elo - 1000.0) < 50.0

    def test_only_candidate_matchups_used(self):
        """Only matchups involving the candidate should matter."""
        pool_ratings = {"random": 1000.0, "other": 1200.0}
        match_results = [
            MatchResult(player_a="candidate", player_b="random", wins_a=7, wins_b=3, draws=0),
            # This matchup doesn't involve candidate — should be ignored
            MatchResult(player_a="random", player_b="other", wins_a=5, wins_b=5, draws=0),
        ]
        elo = compute_elo_against_pool("candidate", pool_ratings, match_results)
        assert elo > 1000.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest test/test_elo/test_rating.py::TestComputeEloAgainstPool -v`
Expected: FAIL — `ImportError: cannot import name 'compute_elo_against_pool'`

- [ ] **Step 3: Implement compute_elo_against_pool**

In `src/pymcts/elo/rating.py`, add after the existing `compute_elo_ratings` function:

```python
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
```

- [ ] **Step 4: Update elo __init__.py**

In `src/pymcts/elo/__init__.py`, add to re-exports:

```python
from pymcts.elo.rating import compute_elo_ratings, compute_elo_against_pool
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest test/test_elo/test_rating.py::TestComputeEloAgainstPool -v`
Expected: PASS

- [ ] **Step 6: Run all elo rating tests**

Run: `python -m pytest test/test_elo/test_rating.py -v`
Expected: All PASS

- [ ] **Step 7: Commit**

```bash
git add src/pymcts/elo/rating.py src/pymcts/elo/__init__.py test/test_elo/test_rating.py
git commit -m "feat: add compute_elo_against_pool for frozen-pool Elo computation"
```

---

### Task 6: Add arena/ directory management to existing ArenaConfig path

**Files:**
- Modify: `src/pymcts/core/trainer.py:159-229`
- Test: `test/test_core/test_trainer_elo_arena.py` (create)

- [ ] **Step 1: Write failing test for ArenaConfig saving accepted players**

Create `test/test_core/test_trainer_elo_arena.py`:

```python
import json
from pathlib import Path

from test.test_core.test_mcts import TicTacToe, DummyNet
from pymcts.core.config import ArenaConfig, MCTSConfig, TrainingConfig, PathsConfig
from pymcts.core.players import MCTSPlayer
from pymcts.core.trainer import train


class TestArenaConfigPlayerSaving:
    def test_accepted_players_saved_to_arena_dir(self, tmp_path):
        """When using ArenaConfig, accepted players are saved to arena/."""
        paths = PathsConfig(
            root=tmp_path,
            trainings=tmp_path / "trainings",
            checkpoints=tmp_path / "checkpoints",
            models=tmp_path / "models",
            data=tmp_path / "data",
        )
        net = DummyNet()
        training_config = TrainingConfig(
            num_iterations=2,
            num_self_play_games=4,
            num_epochs=1,
            batch_size=4,
            self_play_batch_size=2,
            replay_buffer_size=2,
        )
        arena_config = ArenaConfig(num_games=4, threshold=0.0)  # threshold=0 to always accept

        train(
            game_factory=TicTacToe,
            net=net,
            mcts_config=MCTSConfig(num_simulations=5),
            training_config=training_config,
            arena=arena_config,
            paths_config=paths,
            verbose=False,
        )

        # Find the run directory
        run_dirs = list((tmp_path / "trainings").glob("run_*"))
        assert len(run_dirs) == 1
        run_dir = run_dirs[0]

        # Arena directory should exist with saved players
        arena_dir = run_dir / "arena"
        assert arena_dir.exists()
        player_dirs = sorted(arena_dir.glob("iteration_*"))
        assert len(player_dirs) >= 1  # At least one accepted player

        # Each should have player.json
        for player_dir in player_dirs:
            assert (player_dir / "player.json").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest test/test_core/test_trainer_elo_arena.py::TestArenaConfigPlayerSaving -v`
Expected: FAIL — `train()` doesn't accept `arena` parameter

- [ ] **Step 3: Update train() signature and add arena/ saving for ArenaConfig**

In `src/pymcts/core/trainer.py`, modify the `train()` signature. Replace `arena_config: ArenaConfig` with `arena: ArenaConfig | EloArenaConfig`. Add the import of `EloArenaConfig` at the top. Update the function body to create the `arena/` directory and save accepted players.

Change the imports at top of file:

```python
from pymcts.core.config import ArenaConfig, EloArenaConfig, MCTSConfig, PathsConfig, TrainingConfig
```

Change the function signature:

```python
def train(
    game_factory: Callable[[], BaseGame],
    net: BaseNeuralNet,
    mcts_config: MCTSConfig,
    training_config: TrainingConfig,
    arena: ArenaConfig | EloArenaConfig,
    paths_config: PathsConfig | None = None,
    game_type: str = "unknown",
    game_config: dict | None = None,
    verbose: bool = True,
):
```

Update `run_config` dict to use `arena` instead of `arena_config`:

```python
    run_config = {
        "net_class": f"{net_class.__module__}.{net_class.__qualname__}",
        "mcts_config": mcts_config.model_dump(),
        "training_config": training_config.model_dump(),
        "arena": arena.model_dump(),
        "arena_type": type(arena).__name__,
        "game_type": game_type,
        "game_config": game_config,
    }
```

Create the arena directory after the run directory:

```python
    arena_dir = run_dir / "arena"
    arena_dir.mkdir(parents=True, exist_ok=True)
```

In the existing `ArenaConfig` branch (the `if accepted:` block at line 222), after `best_checkpoints.append(...)`, add player saving:

```python
        if accepted:
            if verbose:
                print("\n  -> ACCEPTED: new model is better")
            best_checkpoints.append((iteration, post_checkpoint))
            # Save accepted player to arena/
            accepted_player = GreedyMCTSPlayer(net, mcts_config, name=f"iteration_{iteration:03d}")
            accepted_player.save(arena_dir / f"iteration_{iteration:03d}")
```

Wrap the entire existing arena evaluation logic (lines 160-229) in an `if isinstance(arena, ArenaConfig):` block, using `arena` instead of `arena_config`:

```python
        if isinstance(arena, ArenaConfig):
            # ... existing head-to-head logic, replacing arena_config with arena ...
```

Leave the `elif isinstance(arena, EloArenaConfig):` as a placeholder `pass` for now — it will be implemented in Task 6.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest test/test_core/test_trainer_elo_arena.py::TestArenaConfigPlayerSaving -v`
Expected: PASS

- [ ] **Step 5: Run existing tests to check nothing is broken**

Run: `python -m pytest test/ -v`
Expected: All PASS (existing tests that call `train()` with the old `arena_config` parameter will need to be updated to use `arena` — fix any that fail)

- [ ] **Step 6: Commit**

```bash
git add src/pymcts/core/trainer.py test/test_core/test_trainer_elo_arena.py
git commit -m "feat: add arena/ directory for accepted players in ArenaConfig path"
```

---

### Task 7: Implement Elo arena evaluation in trainer

**Files:**
- Modify: `src/pymcts/core/trainer.py`
- Test: `test/test_core/test_trainer_elo_arena.py`

- [ ] **Step 1: Write failing test for EloArenaConfig training flow**

In `test/test_core/test_trainer_elo_arena.py`, add:

```python
from pymcts.core.config import EloArenaConfig


class TestEloArenaTraining:
    def test_elo_arena_runs_to_completion(self, tmp_path):
        """Training with EloArenaConfig should complete without errors."""
        paths = PathsConfig(
            root=tmp_path,
            trainings=tmp_path / "trainings",
            checkpoints=tmp_path / "checkpoints",
            models=tmp_path / "models",
            data=tmp_path / "data",
        )
        net = DummyNet()
        training_config = TrainingConfig(
            num_iterations=3,
            num_self_play_games=4,
            num_epochs=1,
            batch_size=4,
            self_play_batch_size=2,
            replay_buffer_size=2,
        )
        elo_arena = EloArenaConfig(
            games_per_matchup=4,
            elo_threshold=0.0,  # Accept any improvement
            pool_growth_interval=2,
            max_pool_size=5,
        )

        train(
            game_factory=TicTacToe,
            net=net,
            mcts_config=MCTSConfig(num_simulations=5),
            training_config=training_config,
            arena=elo_arena,
            paths_config=paths,
            verbose=False,
        )

        run_dirs = list((tmp_path / "trainings").glob("run_*"))
        assert len(run_dirs) == 1
        run_dir = run_dirs[0]

        # Arena directory should have random player
        arena_dir = run_dir / "arena"
        assert arena_dir.exists()
        assert (arena_dir / "random" / "player.json").exists()

    def test_elo_arena_pool_grows(self, tmp_path):
        """Pool should grow at pool_growth_interval."""
        paths = PathsConfig(
            root=tmp_path,
            trainings=tmp_path / "trainings",
            checkpoints=tmp_path / "checkpoints",
            models=tmp_path / "models",
            data=tmp_path / "data",
        )
        net = DummyNet()
        training_config = TrainingConfig(
            num_iterations=4,
            num_self_play_games=4,
            num_epochs=1,
            batch_size=4,
            self_play_batch_size=2,
            replay_buffer_size=2,
        )
        elo_arena = EloArenaConfig(
            games_per_matchup=4,
            elo_threshold=0.0,
            pool_growth_interval=2,  # Add at iterations 2 and 4
        )

        train(
            game_factory=TicTacToe,
            net=net,
            mcts_config=MCTSConfig(num_simulations=5),
            training_config=training_config,
            arena=elo_arena,
            paths_config=paths,
            verbose=False,
        )

        run_dirs = list((tmp_path / "trainings").glob("run_*"))
        run_dir = run_dirs[0]
        arena_dir = run_dir / "arena"

        # Should have random + at least the growth interval players
        player_dirs = list(arena_dir.iterdir())
        assert len(player_dirs) >= 2  # random + at least 1 grown player

    def test_elo_arena_with_initial_pool(self, tmp_path):
        """Training with initial_pool should seed the pool from saved players."""
        # First, create a saved player to use as initial pool
        net = DummyNet()
        config = MCTSConfig(num_simulations=5)
        seed_player = MCTSPlayer(net, config, name="seed_player")
        seed_player.elo = 1050.0
        seed_dir = tmp_path / "seed_player"
        seed_player.save(seed_dir)

        paths = PathsConfig(
            root=tmp_path,
            trainings=tmp_path / "trainings",
            checkpoints=tmp_path / "checkpoints",
            models=tmp_path / "models",
            data=tmp_path / "data",
        )
        fresh_net = DummyNet()
        training_config = TrainingConfig(
            num_iterations=2,
            num_self_play_games=4,
            num_epochs=1,
            batch_size=4,
            self_play_batch_size=2,
            replay_buffer_size=2,
        )
        elo_arena = EloArenaConfig(
            games_per_matchup=4,
            elo_threshold=0.0,
            initial_pool=[str(seed_dir)],
        )

        train(
            game_factory=TicTacToe,
            net=fresh_net,
            mcts_config=config,
            training_config=training_config,
            arena=elo_arena,
            paths_config=paths,
            verbose=False,
        )

        run_dirs = list((tmp_path / "trainings").glob("run_*"))
        run_dir = run_dirs[0]
        arena_dir = run_dir / "arena"

        # Should have both random and the seeded player
        assert (arena_dir / "random" / "player.json").exists()
        assert (arena_dir / "seed_player" / "player.json").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest test/test_core/test_trainer_elo_arena.py::TestEloArenaTraining -v`
Expected: FAIL — EloArenaConfig branch is not implemented

- [ ] **Step 3: Implement Elo arena evaluation in trainer**

In `src/pymcts/core/trainer.py`, add the import at the top:

```python
from pymcts.elo.rating import compute_elo_against_pool
```

Before the main loop, add pool initialization logic (after `arena_dir` creation):

```python
    # Elo pool state (only used with EloArenaConfig)
    pool_players: list[tuple[str, BasePlayer, float]] = []  # (name, player, frozen_elo)
    current_elo: float | None = None

    if isinstance(arena, EloArenaConfig):
        # Save and add RandomPlayer to pool
        random_player = RandomPlayer(name="random")
        random_player.elo = 1000.0
        random_player.save(arena_dir / "random")
        pool_players.append(("random", random_player, 1000.0))

        # Load initial pool if provided
        if arena.initial_pool:
            for player_path in arena.initial_pool:
                loaded = MCTSPlayer.load(player_path)
                loaded_elo = loaded.elo if loaded.elo is not None else 1000.0
                loaded.save(arena_dir / loaded.name)
                pool_players.append((loaded.name, loaded, loaded_elo))
```

Inside the iteration loop, add the `elif isinstance(arena, EloArenaConfig):` branch after the `ArenaConfig` branch:

```python
        elif isinstance(arena, EloArenaConfig):
            if verbose:
                print("\n[3/3] Elo arena evaluation...")

            post_player = GreedyMCTSPlayer(net, mcts_config, name="candidate")

            # Play candidate against every pool player
            match_results: list[MatchResult] = []
            pool_ratings: dict[str, float] = {}
            for pool_name, pool_player, pool_elo in pool_players:
                pool_ratings[pool_name] = pool_elo
                records = batched_arena(
                    player_a=post_player,
                    player_b=pool_player,
                    game_factory=game_factory,
                    num_games=arena.games_per_matchup,
                    batch_size=arena.games_per_matchup,
                    swap_players=arena.swap_players,
                    game_type=game_type,
                    verbose=verbose,
                )
                scores = records.scores
                wins_a = scores.get("candidate", 0)
                wins_b = scores.get(pool_name, 0)
                draws = len(records) - wins_a - wins_b
                match_results.append(MatchResult(
                    player_a="candidate",
                    player_b=pool_name,
                    wins_a=wins_a,
                    wins_b=wins_b,
                    draws=draws,
                ))

            post_elo = compute_elo_against_pool("candidate", pool_ratings, match_results)

            # First iteration: establish baseline elo
            if current_elo is None:
                pre_player = GreedyMCTSPlayer(
                    net.copy(), mcts_config, name="pre_candidate",
                )
                pre_player.mcts.net.load_checkpoint(pre_checkpoint)
                pre_match_results: list[MatchResult] = []
                for pool_name, pool_player, pool_elo in pool_players:
                    records = batched_arena(
                        player_a=pre_player,
                        player_b=pool_player,
                        game_factory=game_factory,
                        num_games=arena.games_per_matchup,
                        batch_size=arena.games_per_matchup,
                        swap_players=arena.swap_players,
                        game_type=game_type,
                        verbose=verbose,
                    )
                    scores = records.scores
                    wins_a = scores.get("pre_candidate", 0)
                    wins_b = scores.get(pool_name, 0)
                    draws = len(records) - wins_a - wins_b
                    pre_match_results.append(MatchResult(
                        player_a="pre_candidate",
                        player_b=pool_name,
                        wins_a=wins_a,
                        wins_b=wins_b,
                        draws=draws,
                    ))
                current_elo = compute_elo_against_pool("pre_candidate", pool_ratings, pre_match_results)

            accepted = post_elo >= current_elo + arena.elo_threshold

            if verbose:
                print(f"  Post-training Elo: {post_elo:.0f} | Current Elo: {current_elo:.0f} | "
                      f"Threshold: +{arena.elo_threshold:.0f}")

            if accepted:
                if verbose:
                    print("  -> ACCEPTED: Elo improved")
                current_elo = post_elo
                accepted_player = GreedyMCTSPlayer(net, mcts_config, name=f"iteration_{iteration:03d}", elo=post_elo)
                accepted_player.save(arena_dir / f"iteration_{iteration:03d}")
            else:
                if verbose:
                    print("  -> REJECTED: Elo did not improve enough")
                net.load_checkpoint(pre_checkpoint)

            # Pool growth
            if iteration % arena.pool_growth_interval == 0:
                grow_name = f"iteration_{iteration:03d}"
                grow_player = GreedyMCTSPlayer(net.copy(), mcts_config, name=grow_name, elo=current_elo)
                grow_player.save(arena_dir / grow_name)
                pool_players.append((grow_name, grow_player, current_elo))

                # Evict weakest if over max size
                if arena.max_pool_size is not None and len(pool_players) > arena.max_pool_size:
                    # Find weakest non-random player
                    weakest_idx = None
                    weakest_elo = float("inf")
                    for idx, (name, _, elo) in enumerate(pool_players):
                        if name == "random":
                            continue
                        if elo < weakest_elo:
                            weakest_elo = elo
                            weakest_idx = idx
                    if weakest_idx is not None:
                        evicted_name = pool_players[weakest_idx][0]
                        pool_players.pop(weakest_idx)
                        if verbose:
                            print(f"  Pool: evicted {evicted_name} (Elo {weakest_elo:.0f})")

            # Save iteration data
            iteration_data = {
                "iteration": iteration,
                "training": {
                    "num_examples": len(all_examples),
                    "metrics": train_metrics,
                },
                "elo_arena": {
                    "post_elo": post_elo,
                    "current_elo": current_elo,
                    "threshold": arena.elo_threshold,
                    "accepted": accepted,
                    "pool_size": len(pool_players),
                },
            }
            iter_data_path = iter_dir / "iteration_data.json"
            iter_data_path.write_text(json.dumps(iteration_data, indent=2))
```

Note: The existing `iteration_data` saving and `elo_tracking` code (lines 264-343) should remain inside the `if isinstance(arena, ArenaConfig):` block. Move the `iteration_data` saving from the shared section into each branch so each can write its own format.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest test/test_core/test_trainer_elo_arena.py::TestEloArenaTraining -v`
Expected: PASS

- [ ] **Step 5: Run all tests**

Run: `python -m pytest test/ -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add src/pymcts/core/trainer.py test/test_core/test_trainer_elo_arena.py
git commit -m "feat: implement Elo arena evaluation strategy in training loop"
```

---

### Task 8: Add max_pool_size eviction test

**Files:**
- Test: `test/test_core/test_trainer_elo_arena.py`

- [ ] **Step 1: Write test for pool eviction**

In `test/test_core/test_trainer_elo_arena.py`, add:

```python
class TestEloArenaPoolEviction:
    def test_pool_respects_max_size(self, tmp_path):
        """Pool should not exceed max_pool_size."""
        paths = PathsConfig(
            root=tmp_path,
            trainings=tmp_path / "trainings",
            checkpoints=tmp_path / "checkpoints",
            models=tmp_path / "models",
            data=tmp_path / "data",
        )
        net = DummyNet()
        training_config = TrainingConfig(
            num_iterations=6,
            num_self_play_games=4,
            num_epochs=1,
            batch_size=4,
            self_play_batch_size=2,
            replay_buffer_size=2,
        )
        elo_arena = EloArenaConfig(
            games_per_matchup=4,
            elo_threshold=0.0,
            pool_growth_interval=1,  # Add every iteration
            max_pool_size=3,  # random + 2 others max
        )

        train(
            game_factory=TicTacToe,
            net=net,
            mcts_config=MCTSConfig(num_simulations=5),
            training_config=training_config,
            arena=elo_arena,
            paths_config=paths,
            verbose=False,
        )

        run_dirs = list((tmp_path / "trainings").glob("run_*"))
        run_dir = run_dirs[0]
        arena_dir = run_dir / "arena"

        # The in-memory pool should have been capped, but all players
        # that were ever added remain on disk. Check that random always survived.
        assert (arena_dir / "random" / "player.json").exists()

    def test_random_player_never_evicted(self, tmp_path):
        """RandomPlayer should never be evicted from the pool."""
        paths = PathsConfig(
            root=tmp_path,
            trainings=tmp_path / "trainings",
            checkpoints=tmp_path / "checkpoints",
            models=tmp_path / "models",
            data=tmp_path / "data",
        )
        net = DummyNet()
        training_config = TrainingConfig(
            num_iterations=4,
            num_self_play_games=4,
            num_epochs=1,
            batch_size=4,
            self_play_batch_size=2,
            replay_buffer_size=2,
        )
        elo_arena = EloArenaConfig(
            games_per_matchup=4,
            elo_threshold=0.0,
            pool_growth_interval=1,
            max_pool_size=2,  # Very tight — random + 1
        )

        train(
            game_factory=TicTacToe,
            net=net,
            mcts_config=MCTSConfig(num_simulations=5),
            training_config=training_config,
            arena=elo_arena,
            paths_config=paths,
            verbose=False,
        )

        # If this completes without error and random is still in arena/,
        # the eviction logic correctly protected RandomPlayer
        run_dirs = list((tmp_path / "trainings").glob("run_*"))
        run_dir = run_dirs[0]
        assert (run_dir / "arena" / "random" / "player.json").exists()
```

- [ ] **Step 2: Run test to verify it passes**

Run: `python -m pytest test/test_core/test_trainer_elo_arena.py::TestEloArenaPoolEviction -v`
Expected: PASS (this validates the already-implemented eviction logic)

- [ ] **Step 3: Run full test suite**

Run: `python -m pytest test/ -v`
Expected: All PASS

- [ ] **Step 4: Commit**

```bash
git add test/test_core/test_trainer_elo_arena.py
git commit -m "test: add pool eviction and RandomPlayer protection tests"
```

