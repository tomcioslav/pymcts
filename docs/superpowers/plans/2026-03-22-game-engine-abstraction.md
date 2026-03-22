# Game Engine Abstraction — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Decouple the AI training infrastructure (MCTS, self-play, trainer, arena) from Bridgit, creating abstract base classes so the engine can train on any two-player zero-sum game.

**Architecture:** Create a `core/` package with `BaseGame`, `Board2DGame`, `BaseNeuralNet`, and game-agnostic versions of MCTS, self-play, trainer, arena, and players. Move Bridgit-specific code into `games/bridgit/`. The engine operates on integer actions and opaque `GameState` objects.

**Tech Stack:** Python 3.10+, PyTorch, Pydantic, pytest

**Spec:** `docs/superpowers/specs/2026-03-22-game-engine-abstraction-design.md`

---

## File Structure

### New files to create:
- `src/bridgit/core/__init__.py` — public API for the generic engine
- `src/bridgit/core/base_game.py` — `GameState`, `BaseGame` ABC
- `src/bridgit/core/board_2d_game.py` — `Board2DGame(BaseGame)` convenience layer
- `src/bridgit/core/base_neural_net.py` — `BaseNeuralNet(ABC, nn.Module)`
- `src/bridgit/core/mcts.py` — game-agnostic MCTS (action ints, not Move objects)
- `src/bridgit/core/self_play.py` — batched self-play with game/net factories
- `src/bridgit/core/trainer.py` — AlphaZero training loop with factories
- `src/bridgit/core/arena.py` — generic arena
- `src/bridgit/core/players.py` — `BasePlayer`, `RandomPlayer`, `MCTSPlayer`, `GreedyMCTSPlayer`
- `src/bridgit/core/game_record.py` — generic `MoveRecord`, `GameRecord`, `GameRecordCollection`, `EvalResult`
- `src/bridgit/core/data.py` — training data extraction
- `src/bridgit/core/config.py` — `MCTSConfig`, `TrainingConfig`, `ArenaConfig`
- `src/bridgit/games/__init__.py`
- `src/bridgit/games/bridgit/__init__.py`
- `src/bridgit/games/bridgit/game.py` — `BridgitGameState`, `BridgitGame(Board2DGame)`
- `src/bridgit/games/bridgit/neural_net.py` — `BridgitNet(BaseNeuralNet)`
- `src/bridgit/games/bridgit/config.py` — `BoardConfig`, `NeuralNetConfig`
- `test/test_core/` — tests for generic engine
- `test/test_games/test_bridgit/` — tests for Bridgit implementation

### Existing files to keep (moved/adapted):
- `src/bridgit/game/state.py` → internal to `games/bridgit/game.py`
- `src/bridgit/game/union_find.py` → `src/bridgit/games/bridgit/union_find.py`
- `src/bridgit/schema/player.py` → `src/bridgit/games/bridgit/player.py`
- `src/bridgit/visualizer.py` → `src/bridgit/games/bridgit/visualizer.py`

### Files to remove after migration:
- `src/bridgit/ai/` (replaced by `core/`)
- `src/bridgit/players/` (replaced by `core/`)
- `src/bridgit/schema/` (dissolved into `core/game_record.py` and `games/bridgit/`)
- `src/bridgit/data/` (replaced by `core/data.py`)
- `src/bridgit/game/` (replaced by `games/bridgit/`)

---

## Task 1: Core abstractions — BaseGame, Board2DGame, GameState

**Files:**
- Create: `src/bridgit/core/__init__.py`
- Create: `src/bridgit/core/base_game.py`
- Create: `src/bridgit/core/board_2d_game.py`
- Test: `test/test_core/__init__.py`
- Test: `test/test_core/test_base_game.py`

- [ ] **Step 1: Write failing tests for BaseGame and Board2DGame**

```python
# test/test_core/test_base_game.py
import pytest
import torch

from bridgit.core.base_game import BaseGame, Board2DGame, GameState


class DummyState(GameState):
    def __init__(self, board: list[int]):
        self.board = board


class DummyGame(Board2DGame):
    """Minimal 3x3 game for testing."""

    def __init__(self):
        super().__init__(board_rows=3, board_cols=3)
        self._board = [0] * 9
        self._current_player = 0
        self._is_over = False
        self._winner = None

    @property
    def current_player(self) -> int:
        return self._current_player

    @property
    def is_over(self) -> bool:
        return self._is_over

    @property
    def winner(self) -> int | None:
        return self._winner

    def get_state(self) -> GameState:
        return DummyState(self._board[:])

    def to_mask(self) -> torch.Tensor:
        return torch.tensor([cell == 0 for cell in self._board], dtype=torch.bool)

    def make_action(self, action: int) -> None:
        self._board[action] = self._current_player + 1
        self._current_player = 1 - self._current_player

    def copy(self) -> "DummyGame":
        new = DummyGame()
        new._board = self._board[:]
        new._current_player = self._current_player
        new._is_over = self._is_over
        new._winner = self._winner
        return new

    def get_result(self, player: int) -> float | None:
        if not self._is_over:
            return None
        if self._winner == player:
            return 1.0
        return -1.0


class TestBaseGame:
    def test_action_space_size(self):
        game = DummyGame()
        assert game.action_space_size == 9

    def test_valid_actions_from_mask(self):
        game = DummyGame()
        assert game.valid_actions() == list(range(9))

    def test_valid_actions_after_move(self):
        game = DummyGame()
        game.make_action(4)
        actions = game.valid_actions()
        assert 4 not in actions
        assert len(actions) == 8

    def test_get_state_returns_game_state(self):
        game = DummyGame()
        state = game.get_state()
        assert isinstance(state, GameState)

    def test_copy_is_independent(self):
        game = DummyGame()
        game.make_action(0)
        copy = game.copy()
        copy.make_action(1)
        assert game.valid_actions() != copy.valid_actions()


class TestBoard2DGame:
    def test_action_to_row_col(self):
        game = DummyGame()
        assert game.action_to_row_col(0) == (0, 0)
        assert game.action_to_row_col(4) == (1, 1)
        assert game.action_to_row_col(8) == (2, 2)

    def test_row_col_to_action(self):
        game = DummyGame()
        assert game.row_col_to_action(0, 0) == 0
        assert game.row_col_to_action(1, 1) == 4
        assert game.row_col_to_action(2, 2) == 8

    def test_roundtrip(self):
        game = DummyGame()
        for action in range(9):
            r, c = game.action_to_row_col(action)
            assert game.row_col_to_action(r, c) == action
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/tomaszjuszczyszyn/Projects/pwr/bridge-it && python -m pytest test/test_core/test_base_game.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'bridgit.core'`

- [ ] **Step 3: Implement GameState, BaseGame, Board2DGame**

```python
# src/bridgit/core/__init__.py
from bridgit.core.base_game import BaseGame, Board2DGame, GameState
# BaseNeuralNet will be added in Task 2
```

```python
# src/bridgit/core/base_game.py
"""Abstract base classes for game engine."""

from abc import ABC, abstractmethod

import torch


class GameState:
    """Base class for game states. Each game defines its own subclass.
    The engine passes GameState objects opaquely — only the game and its
    neural net know the concrete type.
    """
    pass


class BaseGame(ABC):
    """Minimal contract for a two-player zero-sum game."""

    @property
    @abstractmethod
    def current_player(self) -> int:
        """Current player id. Always an int (0, 1, ...)."""

    @property
    @abstractmethod
    def is_over(self) -> bool: ...

    @property
    @abstractmethod
    def winner(self) -> int | None:
        """Player id of winner, or None if draw / not over."""

    @property
    @abstractmethod
    def action_space_size(self) -> int:
        """Total number of possible actions. Fixed for a game type."""

    @abstractmethod
    def get_state(self) -> GameState:
        """Return current game state from the current player's perspective.
        Canonicalization is the game's responsibility.
        """

    @abstractmethod
    def to_mask(self) -> torch.Tensor:
        """1D boolean tensor of length action_space_size. True = legal action."""

    def valid_actions(self) -> list[int]:
        """List of currently legal action indices. Derived from to_mask()."""
        return self.to_mask().nonzero(as_tuple=False).squeeze(-1).tolist()

    @abstractmethod
    def make_action(self, action: int) -> None:
        """Apply action in-place. Mutates the game state."""

    @abstractmethod
    def copy(self) -> "BaseGame": ...

    @abstractmethod
    def get_result(self, player: int) -> float | None:
        """1.0 for win, -1.0 for loss, 0.0 for draw, None if not over."""


class Board2DGame(BaseGame):
    """Base for games played on a 2D rectangular grid."""

    def __init__(self, board_rows: int, board_cols: int):
        self._board_rows = board_rows
        self._board_cols = board_cols

    @property
    def board_rows(self) -> int:
        return self._board_rows

    @property
    def board_cols(self) -> int:
        return self._board_cols

    @property
    def action_space_size(self) -> int:
        return self._board_rows * self._board_cols

    def action_to_row_col(self, action: int) -> tuple[int, int]:
        return divmod(action, self._board_cols)

    def row_col_to_action(self, row: int, col: int) -> int:
        return row * self._board_cols + col
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/tomaszjuszczyszyn/Projects/pwr/bridge-it && python -m pytest test/test_core/test_base_game.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/bridgit/core/__init__.py src/bridgit/core/base_game.py test/test_core/__init__.py test/test_core/test_base_game.py
git commit -m "feat: add BaseGame, Board2DGame, GameState core abstractions"
```

---

## Task 2: BaseNeuralNet

**Files:**
- Create: `src/bridgit/core/base_neural_net.py`
- Test: `test/test_core/test_base_neural_net.py`

- [ ] **Step 1: Write failing tests**

```python
# test/test_core/test_base_neural_net.py
import copy

import torch
import torch.nn as nn

from bridgit.core.base_game import GameState
from bridgit.core.base_neural_net import BaseNeuralNet


class SimpleState(GameState):
    def __init__(self, data: list[float]):
        self.data = data


class SimpleNet(BaseNeuralNet):
    """Trivial net for testing: 4-feature input, 4-action output."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 4)
        self.value_fc = nn.Linear(4, 1)

    def encode(self, state: GameState) -> torch.Tensor:
        return torch.tensor(state.data, dtype=torch.float32)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        policy = torch.log_softmax(self.fc(x), dim=-1)
        value = torch.tanh(self.value_fc(x))
        return policy, value

    def save_checkpoint(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    def load_checkpoint(self, path: str) -> None:
        self.load_state_dict(torch.load(path, weights_only=True))

    def copy(self) -> "SimpleNet":
        return copy.deepcopy(self)


class TestBaseNeuralNet:
    def test_predict_returns_policy_and_value(self):
        net = SimpleNet()
        state = SimpleState([1.0, 0.0, 0.0, 0.0])
        policy, value = net.predict(state)
        assert policy.shape == (4,)
        assert isinstance(value, float)
        assert -1.0 <= value <= 1.0

    def test_predict_batch(self):
        net = SimpleNet()
        states = [SimpleState([1.0, 0, 0, 0]), SimpleState([0, 1.0, 0, 0])]
        policies, values = net.predict_batch(states)
        assert policies.shape == (2, 4)
        assert values.shape == (2,)

    def test_is_nn_module(self):
        net = SimpleNet()
        assert isinstance(net, nn.Module)
        assert len(list(net.parameters())) > 0

    def test_train_on_examples(self):
        net = SimpleNet()
        examples = [
            (SimpleState([1, 0, 0, 0]), torch.tensor([0.7, 0.1, 0.1, 0.1]), 1.0),
            (SimpleState([0, 1, 0, 0]), torch.tensor([0.1, 0.7, 0.1, 0.1]), -1.0),
        ]
        metrics = net.train_on_examples(examples)
        assert "loss" in metrics
        assert "policy_loss" in metrics
        assert "value_loss" in metrics
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest test/test_core/test_base_neural_net.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'bridgit.core.base_neural_net'`

- [ ] **Step 3: Implement BaseNeuralNet**

```python
# src/bridgit/core/base_neural_net.py
"""Abstract base class for neural networks in the AlphaZero engine."""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from bridgit.core.base_game import GameState


class BaseNeuralNet(ABC, nn.Module):
    """Base class for all neural networks. Inherits from ABC and nn.Module.

    The developer implements encode() and forward(). predict(), predict_batch(),
    and train_on_examples() have sensible defaults.
    """

    @abstractmethod
    def encode(self, state: GameState) -> torch.Tensor:
        """Convert a GameState into the tensor format this architecture expects."""

    @abstractmethod
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Raw forward pass. Input: (batch, *encoded_shape).
        Returns: (log_policy (batch, action_space_size), value (batch, 1)).
        """

    @abstractmethod
    def save_checkpoint(self, path: str) -> None: ...

    @abstractmethod
    def load_checkpoint(self, path: str) -> None: ...

    @abstractmethod
    def copy(self) -> "BaseNeuralNet": ...

    def predict(self, state: GameState) -> tuple[torch.Tensor, float]:
        """Single state -> (policy_1D, value)."""
        self.eval()
        with torch.no_grad():
            tensor = self.encode(state).unsqueeze(0)
            policy, value = self.forward(tensor)
        return policy.squeeze(0), value.item()

    def predict_batch(self, states: list[GameState]) -> tuple[torch.Tensor, torch.Tensor]:
        """Batch of states -> (policies, values)."""
        self.eval()
        with torch.no_grad():
            tensors = torch.stack([self.encode(s) for s in states])
            policies, values = self.forward(tensors)
        return policies, values.squeeze(-1)

    def train_on_examples(
        self,
        examples: list[tuple[GameState, torch.Tensor, float]],
        num_epochs: int = 20,
        batch_size: int = 64,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-4,
    ) -> dict[str, float]:
        """Default training loop: cross-entropy policy loss + MSE value loss.

        Returns metrics dict with final epoch losses.
        """
        states = torch.stack([self.encode(s) for s, _, _ in examples])
        policies = torch.stack([p for _, p, _ in examples])
        values = torch.tensor([v for _, _, v in examples], dtype=torch.float32).unsqueeze(1)

        dataset = TensorDataset(states, policies, values)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.Adam(
            self.parameters(), lr=learning_rate, weight_decay=weight_decay,
        )

        self.train()
        total_policy_loss = 0.0
        total_value_loss = 0.0
        num_batches = 0

        for epoch in range(num_epochs):
            total_policy_loss = 0.0
            total_value_loss = 0.0
            num_batches = 0

            for batch_states, batch_policies, batch_values in loader:
                log_policy, value = self.forward(batch_states)
                policy_loss = -torch.sum(batch_policies * log_policy) / batch_states.size(0)
                value_loss = F.mse_loss(value, batch_values)
                loss = policy_loss + value_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                num_batches += 1

        avg_pi = total_policy_loss / max(num_batches, 1)
        avg_v = total_value_loss / max(num_batches, 1)
        return {
            "policy_loss": avg_pi,
            "value_loss": avg_v,
            "loss": avg_pi + avg_v,
        }
```

- [ ] **Step 4: Update `core/__init__.py` to export BaseNeuralNet**

The `__init__.py` from Task 1 already imports it. Verify it exists.

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest test/test_core/test_base_neural_net.py -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add src/bridgit/core/base_neural_net.py test/test_core/test_base_neural_net.py
git commit -m "feat: add BaseNeuralNet(ABC, nn.Module) with default predict/train"
```

---

## Task 3: Generic game records, config, and data converter

**Files:**
- Create: `src/bridgit/core/game_record.py`
- Create: `src/bridgit/core/config.py`
- Create: `src/bridgit/core/data.py`
- Test: `test/test_core/test_game_record.py`

- [ ] **Step 1: Write failing tests**

```python
# test/test_core/test_game_record.py
from bridgit.core.game_record import MoveRecord, GameRecord, GameRecordCollection, EvalResult


class TestGameRecord:
    def test_move_record_creation(self):
        rec = MoveRecord(action=5, player=0, policy=None)
        assert rec.action == 5
        assert rec.player == 0

    def test_game_record_creation(self):
        moves = [MoveRecord(action=0, player=0, policy=None)]
        record = GameRecord(
            game_type="test",
            game_config={"size": 3},
            moves=moves,
            winner=0,
            player_names=["p0", "p1"],
        )
        assert record.num_moves == 1
        assert record.winner == 0

    def test_collection_evaluate(self):
        moves = [MoveRecord(action=0, player=0, policy=None)]
        records = [
            GameRecord(game_type="test", game_config={}, moves=moves,
                       winner=0, player_names=["alice", "bob"]),
            GameRecord(game_type="test", game_config={}, moves=moves,
                       winner=1, player_names=["alice", "bob"]),
            GameRecord(game_type="test", game_config={}, moves=moves,
                       winner=0, player_names=["alice", "bob"]),
        ]
        collection = GameRecordCollection(game_records=records)
        result = collection.evaluate("alice")
        assert result.wins == 2
        assert result.losses == 1
        assert result.win_rate == 2 / 3

    def test_collection_is_better(self):
        moves = [MoveRecord(action=0, player=0, policy=None)]
        records = [
            GameRecord(game_type="test", game_config={}, moves=moves,
                       winner=0, player_names=["alice", "bob"])
            for _ in range(6)
        ] + [
            GameRecord(game_type="test", game_config={}, moves=moves,
                       winner=1, player_names=["alice", "bob"])
            for _ in range(4)
        ]
        collection = GameRecordCollection(game_records=records)
        assert collection.is_better("alice", 0.55) is True
        assert collection.is_better("bob", 0.55) is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest test/test_core/test_game_record.py -v`
Expected: FAIL

- [ ] **Step 3: Implement game_record.py**

Port from `src/bridgit/schema/game_record.py` but replace `Move` with `int action`, `Player` enum with `int player`, remove `moves_left_after`. Keep `SerializableTensor`, `EvalResult`, `GameRecordCollection` with the `evaluate()` and `is_better()` logic adapted to use `player_names` list indexed by `winner` int.

```python
# src/bridgit/core/game_record.py
"""Generic game recording types."""

from dataclasses import dataclass
from typing import Annotated, Any

import torch
from pydantic import BaseModel, BeforeValidator, PlainSerializer


def _validate_tensor(v: object) -> torch.Tensor | None:
    if v is None:
        return None
    if isinstance(v, torch.Tensor):
        return v
    return torch.tensor(v, dtype=torch.float32)


def _serialize_tensor(v: Any) -> list | None:
    if v is None:
        return None
    return v.tolist()


SerializableTensor = Annotated[
    Any,
    BeforeValidator(_validate_tensor),
    PlainSerializer(_serialize_tensor, return_type=list | None),
]


class MoveRecord(BaseModel):
    """A single move in a game."""
    action: int
    player: int
    policy: SerializableTensor = None  # 1D, length action_space_size


class GameRecord(BaseModel):
    """Full record of a played game."""
    game_type: str
    game_config: dict
    moves: list[MoveRecord]
    winner: int | None
    player_names: list[str]

    @property
    def num_moves(self) -> int:
        return len(self.moves)

    def winner_name(self) -> str | None:
        if self.winner is None:
            return None
        return self.player_names[self.winner]

    def summary(self) -> str:
        names = " vs ".join(self.player_names)
        winner = self.winner_name() or "draw"
        return f"{names}: {winner} wins in {self.num_moves} moves"


@dataclass
class EvalResult:
    """Evaluation statistics for a player in an arena run."""
    wins: int
    losses: int
    draws: int
    total: int
    win_rate: float
    avg_moves_in_wins: float
    avg_moves_in_losses: float


class GameRecordCollection(BaseModel):
    """A collection of game records."""
    game_records: list[GameRecord]

    @property
    def scores(self) -> dict[str, int]:
        scores: dict[str, int] = {}
        for r in self.game_records:
            name = r.winner_name()
            if name:
                scores[name] = scores.get(name, 0) + 1
        return scores

    def evaluate(self, player_name: str) -> EvalResult:
        win_moves: list[int] = []
        loss_moves: list[int] = []
        draw_count = 0
        for r in self.game_records:
            if player_name not in r.player_names:
                continue
            wn = r.winner_name()
            if wn is None:
                draw_count += 1
            elif wn == player_name:
                win_moves.append(r.num_moves)
            else:
                loss_moves.append(r.num_moves)
        wins = len(win_moves)
        losses = len(loss_moves)
        total = wins + losses + draw_count
        return EvalResult(
            wins=wins, losses=losses, draws=draw_count, total=total,
            win_rate=wins / total if total > 0 else 0.0,
            avg_moves_in_wins=sum(win_moves) / wins if wins > 0 else 0.0,
            avg_moves_in_losses=sum(loss_moves) / losses if losses > 0 else 0.0,
        )

    def is_better(self, player_name: str, win_threshold: float = 0.55) -> bool:
        result = self.evaluate(player_name)
        if result.win_rate >= win_threshold:
            return True
        if (result.win_rate >= 0.5
                and result.avg_moves_in_losses > 0
                and result.avg_moves_in_losses > result.avg_moves_in_wins):
            return True
        return False

    def __len__(self) -> int:
        return len(self.game_records)

    def __iter__(self):
        return iter(self.game_records)

    def __getitem__(self, idx: int) -> GameRecord:
        return self.game_records[idx]
```

- [ ] **Step 4: Implement config.py**

```python
# src/bridgit/core/config.py
"""Generic engine configuration."""

import logging
from pathlib import Path

from pydantic import BaseModel


_PROJECT_ROOT = Path(__file__).parent.parent.parent


class PathsConfig(BaseModel):
    """File system paths for the project."""
    root: Path = _PROJECT_ROOT
    checkpoints: Path = _PROJECT_ROOT / "checkpoints"
    models: Path = _PROJECT_ROOT / "models"
    data: Path = _PROJECT_ROOT / "data"
    trainings: Path = _PROJECT_ROOT / "trainings"


class MCTSConfig(BaseModel):
    """Monte Carlo Tree Search settings."""
    num_simulations: int = 100
    c_puct: float = 1.5
    dirichlet_alpha: float = 1.0
    dirichlet_epsilon: float = 0.25
    num_parallel_leaves: int = 1


class TrainingConfig(BaseModel):
    """Training loop settings."""
    num_iterations: int = 50
    num_self_play_games: int = 50
    num_epochs: int = 20
    batch_size: int = 64
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    replay_buffer_size: int = 5
    self_play_batch_size: int = 8


class ArenaConfig(BaseModel):
    """Arena evaluation settings."""
    num_games: int = 40
    threshold: float = 0.55
    swap_players: bool = True
```

- [ ] **Step 5: Implement data.py**

```python
# src/bridgit/core/data.py
"""Training data extraction from game records."""

from typing import Callable

import torch

from bridgit.core.base_game import BaseGame, GameState
from bridgit.core.game_record import GameRecordCollection


Example = tuple[GameState, torch.Tensor, float]


def examples_from_records(
    collection: GameRecordCollection,
    game_factory: Callable[[dict], BaseGame],
) -> list[Example]:
    """Replay games to extract (state, policy, value) training examples.

    Args:
        collection: Game records to replay.
        game_factory: Creates a new game from game_config dict.
    """
    examples: list[Example] = []
    for record in collection.game_records:
        game = game_factory(record.game_config)
        for move_rec in record.moves:
            if move_rec.policy is not None:
                state = game.get_state()
                value = 1.0 if game.current_player == record.winner else -1.0
                examples.append((state, move_rec.policy, value))
            game.make_action(move_rec.action)
    return examples
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `python -m pytest test/test_core/test_game_record.py -v`
Expected: All PASS

- [ ] **Step 7: Commit**

```bash
git add src/bridgit/core/game_record.py src/bridgit/core/config.py src/bridgit/core/data.py test/test_core/test_game_record.py
git commit -m "feat: add generic game records, config, and data converter"
```

---

## Task 4: Generic MCTS

**Files:**
- Create: `src/bridgit/core/mcts.py`
- Test: `test/test_core/test_mcts.py`

- [ ] **Step 1: Write failing tests**

```python
# test/test_core/test_mcts.py
import copy

import torch

from bridgit.core.base_game import BaseGame, Board2DGame, GameState
from bridgit.core.base_neural_net import BaseNeuralNet
from bridgit.core.mcts import MCTS, MCTSNode
from bridgit.core.config import MCTSConfig

import torch.nn as nn


class TicTacToeState(GameState):
    def __init__(self, board: list[int], current: int):
        self.board = board[:]
        self.current = current


class TicTacToe(Board2DGame):
    """Minimal tic-tac-toe for MCTS testing."""

    def __init__(self):
        super().__init__(3, 3)
        self._board = [0] * 9
        self._current = 0
        self._winner = None
        self._over = False

    @property
    def current_player(self) -> int:
        return self._current

    @property
    def is_over(self) -> bool:
        return self._over

    @property
    def winner(self) -> int | None:
        return self._winner

    def get_state(self) -> GameState:
        return TicTacToeState(self._board, self._current)

    def to_mask(self) -> torch.Tensor:
        return torch.tensor([cell == 0 for cell in self._board], dtype=torch.bool)

    def make_action(self, action: int) -> None:
        self._board[action] = self._current + 1
        lines = [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]
        for a, b, c in lines:
            if self._board[a] == self._board[b] == self._board[c] != 0:
                self._winner = self._board[a] - 1
                self._over = True
                return
        if all(cell != 0 for cell in self._board):
            self._over = True
        self._current = 1 - self._current

    def copy(self) -> "TicTacToe":
        new = TicTacToe()
        new._board = self._board[:]
        new._current = self._current
        new._winner = self._winner
        new._over = self._over
        return new

    def get_result(self, player: int) -> float | None:
        if not self._over:
            return None
        if self._winner is None:
            return 0.0
        return 1.0 if self._winner == player else -1.0


class DummyNet(BaseNeuralNet):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(9, 9)
        self.vfc = nn.Linear(9, 1)

    def encode(self, state: GameState) -> torch.Tensor:
        return torch.tensor(state.board, dtype=torch.float32)

    def forward(self, x):
        p = torch.log_softmax(self.fc(x), dim=-1)
        v = torch.tanh(self.vfc(x))
        return p, v

    def save_checkpoint(self, path): torch.save(self.state_dict(), path)
    def load_checkpoint(self, path): self.load_state_dict(torch.load(path, weights_only=True))
    def copy(self): return copy.deepcopy(self)


class TestMCTS:
    def test_search_returns_root(self):
        game = TicTacToe()
        net = DummyNet()
        config = MCTSConfig(num_simulations=10)
        mcts = MCTS(net, config)
        root = mcts._search(game)
        assert isinstance(root, MCTSNode)
        assert root.visit_count > 0

    def test_get_action_probs_shape(self):
        game = TicTacToe()
        net = DummyNet()
        config = MCTSConfig(num_simulations=10)
        mcts = MCTS(net, config)
        probs = mcts.get_action_probs(game)
        assert probs.shape == (9,)
        assert probs.sum().item() > 0

    def test_visit_counts_1d(self):
        game = TicTacToe()
        net = DummyNet()
        config = MCTSConfig(num_simulations=10)
        mcts = MCTS(net, config)
        root = mcts._search(game)
        counts = root.visit_counts(9)
        assert counts.shape == (9,)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest test/test_core/test_mcts.py -v`
Expected: FAIL

- [ ] **Step 3: Implement generic MCTS**

Port from `src/bridgit/ai/mcts.py`. Key changes:
- `MCTSNode` stores `action: int` instead of `action: Move`
- `children: dict[int, MCTSNode]` instead of `dict[Move, MCTSNode]`
- `unexpanded_moves: dict[int, float]` instead of `dict[Move, float]`
- `visit_counts(action_space_size)` returns 1D tensor
- `_set_priors` iterates `valid_actions()` instead of `torch.nonzero` on 2D mask
- `_predict` calls `net.predict(game.get_state())`
- `best_child_or_expand` calls `game.copy()` then `game.make_action(action)` — no decanonicalize
- Constructor takes `BaseGame` and `BaseNeuralNet` types

Write the full `src/bridgit/core/mcts.py` porting every method from the current `mcts.py` with these substitutions. Keep `continue_search`, `_expand`, `_backpropagate`, `_add_dirichlet_noise`, `visit_counts_to_probs`, `get_action_probs`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest test/test_core/test_mcts.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/bridgit/core/mcts.py test/test_core/test_mcts.py
git commit -m "feat: add game-agnostic MCTS with integer actions"
```

---

## Task 5: Generic players and arena

**Files:**
- Create: `src/bridgit/core/players.py`
- Create: `src/bridgit/core/arena.py`
- Test: `test/test_core/test_players.py`

- [ ] **Step 1: Write failing tests**

```python
# test/test_core/test_players.py
from test.test_core.test_mcts import TicTacToe, DummyNet
from bridgit.core.players import BasePlayer, RandomPlayer, MCTSPlayer
from bridgit.core.arena import Arena
from bridgit.core.config import MCTSConfig


class TestRandomPlayer:
    def test_returns_valid_action(self):
        game = TicTacToe()
        player = RandomPlayer()
        action = player.get_action(game)
        assert action in game.valid_actions()

    def test_last_policy_is_none(self):
        player = RandomPlayer()
        game = TicTacToe()
        player.get_action(game)
        assert player.last_policy is None


class TestMCTSPlayer:
    def test_returns_valid_action(self):
        game = TicTacToe()
        net = DummyNet()
        config = MCTSConfig(num_simulations=5)
        player = MCTSPlayer(net, config)
        action = player.get_action(game)
        assert action in game.valid_actions()

    def test_stores_last_policy(self):
        game = TicTacToe()
        net = DummyNet()
        config = MCTSConfig(num_simulations=5)
        player = MCTSPlayer(net, config)
        player.get_action(game)
        assert player.last_policy is not None
        assert player.last_policy.shape == (9,)


class TestArena:
    def test_play_game_completes(self):
        p1 = RandomPlayer(name="p1")
        p2 = RandomPlayer(name="p2")
        arena = Arena(p1, p2, game_factory=TicTacToe)
        record = arena.play_game()
        assert record.winner is not None or record.winner is None  # draw possible
        assert record.num_moves > 0

    def test_play_games_returns_collection(self):
        p1 = RandomPlayer(name="p1")
        p2 = RandomPlayer(name="p2")
        arena = Arena(p1, p2, game_factory=TicTacToe)
        collection = arena.play_games(4)
        assert len(collection) == 4
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest test/test_core/test_players.py -v`
Expected: FAIL

- [ ] **Step 3: Implement players.py**

Port from `src/bridgit/players/players.py`. Key changes:
- `get_action` returns `int` instead of `Move`
- `MCTSPlayer` uses generic `MCTS` from `core.mcts`
- No `decanonicalize` — action int goes directly
- No `to_spec()` serialization
- `last_policy` is 1D tensor

```python
# src/bridgit/core/players.py
"""Player abstractions for the AlphaZero engine."""

import random
from abc import ABC, abstractmethod

import torch

from bridgit.core.base_game import BaseGame
from bridgit.core.base_neural_net import BaseNeuralNet
from bridgit.core.mcts import MCTS
from bridgit.core.config import MCTSConfig


class BasePlayer(ABC):
    def __init__(self, name: str | None = None):
        self.name = name or self.__class__.__name__
        self._last_policy: torch.Tensor | None = None

    @abstractmethod
    def get_action(self, game: BaseGame) -> int: ...

    @property
    def last_policy(self) -> torch.Tensor | None:
        return self._last_policy

    def __repr__(self) -> str:
        return self.name


class RandomPlayer(BasePlayer):
    def get_action(self, game: BaseGame) -> int:
        self._last_policy = None
        return random.choice(game.valid_actions())


class MCTSPlayer(BasePlayer):
    def __init__(
        self,
        net: BaseNeuralNet,
        mcts_config: MCTSConfig,
        temperature: float = 1.0,
        temp_threshold: int = 0,
        name: str | None = None,
    ):
        super().__init__(name)
        self.mcts = MCTS(net, mcts_config)
        self.temperature = temperature
        self.temp_threshold = temp_threshold

    def get_action(self, game: BaseGame) -> int:
        # Use number of moves already played in this game for temperature decay
        move_count = game.action_space_size - len(game.valid_actions())
        temp = self.temperature if move_count < self.temp_threshold else 0.0
        probs = self.mcts.get_action_probs(game, temperature=temp)
        self._last_policy = probs

        mask = game.to_mask()
        if probs.sum() == 0:
            probs = mask.float()

        flat_idx = torch.multinomial(probs, 1).item()
        return flat_idx


class GreedyMCTSPlayer(MCTSPlayer):
    def __init__(self, net: BaseNeuralNet, mcts_config: MCTSConfig, name: str | None = None):
        super().__init__(net, mcts_config, temperature=0.0, name=name)
```

- [ ] **Step 4: Implement arena.py**

Port from `src/bridgit/players/arena.py`. Key changes:
- Receives `game_factory: type[BaseGame] | Callable[[], BaseGame]` instead of `BoardConfig`
- Records `MoveRecord(action=int, player=int)` instead of `Move`
- No `Player` enum — uses `game.current_player` (int)
- Winner determined by `game.winner` (int)
- Player assignment: player 0 = first player, player 1 = second player

```python
# src/bridgit/core/arena.py
"""Arena for pitting two players against each other."""

import logging
from typing import Callable

from tqdm.auto import tqdm

from bridgit.core.base_game import BaseGame
from bridgit.core.players import BasePlayer
from bridgit.core.game_record import GameRecord, GameRecordCollection, MoveRecord

logger = logging.getLogger("core.arena")


class Arena:
    def __init__(
        self,
        player_a: BasePlayer,
        player_b: BasePlayer,
        game_factory: Callable[[], BaseGame],
        game_type: str = "unknown",
        game_config: dict | None = None,
    ):
        self.player_a = player_a
        self.player_b = player_b
        self.game_factory = game_factory
        self.game_type = game_type
        self.game_config = game_config or {}

    def play_game(self, swapped: bool = False) -> GameRecord:
        game = self.game_factory()

        if swapped:
            players = {0: self.player_b, 1: self.player_a}
            names = [self.player_b.name, self.player_a.name]
        else:
            players = {0: self.player_a, 1: self.player_b}
            names = [self.player_a.name, self.player_b.name]

        move_records: list[MoveRecord] = []
        max_moves = game.action_space_size * 4

        while not game.is_over:
            current = game.current_player
            player = players[current]
            action = player.get_action(game)
            policy = player.last_policy

            game.make_action(action)
            move_records.append(MoveRecord(
                action=action,
                player=current,
                policy=policy,
            ))

            if len(move_records) > max_moves:
                raise RuntimeError(f"Game exceeded {max_moves} moves — likely stuck")

        return GameRecord(
            game_type=self.game_type,
            game_config=self.game_config,
            moves=move_records,
            winner=game.winner,
            player_names=names,
        )

    def play_games(
        self,
        num_games: int,
        verbose: bool = False,
        swap_players: bool = False,
    ) -> GameRecordCollection:
        records: list[GameRecord] = []
        half = num_games // 2 if swap_players else num_games

        iterator = range(num_games)
        if verbose:
            iterator = tqdm(iterator, desc=f"{self.player_a.name} vs {self.player_b.name}")

        for i in iterator:
            swapped = swap_players and i >= half
            record = self.play_game(swapped=swapped)
            records.append(record)

        return GameRecordCollection(game_records=records)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest test/test_core/test_players.py -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add src/bridgit/core/players.py src/bridgit/core/arena.py test/test_core/test_players.py
git commit -m "feat: add generic players and arena"
```

---

## Task 6: Generic batched self-play and trainer

**Files:**
- Create: `src/bridgit/core/self_play.py`
- Create: `src/bridgit/core/trainer.py`
- Test: `test/test_core/test_self_play.py`

- [ ] **Step 1: Write failing tests**

```python
# test/test_core/test_self_play.py
from test.test_core.test_mcts import TicTacToe, DummyNet
from bridgit.core.self_play import batched_self_play
from bridgit.core.config import MCTSConfig


class TestBatchedSelfPlay:
    def test_completes_games(self):
        net = DummyNet()
        config = MCTSConfig(num_simulations=5, num_parallel_leaves=1)
        records = batched_self_play(
            net=net,
            game_factory=TicTacToe,
            mcts_config=config,
            num_games=2,
            batch_size=2,
            verbose=False,
        )
        assert len(records) == 2
        for r in records:
            assert r.num_moves > 0

    def test_records_have_policies(self):
        net = DummyNet()
        config = MCTSConfig(num_simulations=5)
        records = batched_self_play(
            net=net,
            game_factory=TicTacToe,
            mcts_config=config,
            num_games=1,
            batch_size=1,
            verbose=False,
        )
        has_policy = any(m.policy is not None for m in records[0].moves)
        assert has_policy
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest test/test_core/test_self_play.py -v`
Expected: FAIL

- [ ] **Step 3: Implement self_play.py**

Port from `src/bridgit/ai/self_play.py`. Key changes:
- Receives `net: BaseNeuralNet` and `game_factory: Callable[[], BaseGame]`
- `BatchedMCTS._predict_batch` calls `net.predict_batch([g.get_state() for g in games])`
- `_set_priors` iterates `game.valid_actions()` with 1D policy
- Action selection: `torch.multinomial(probs, 1).item()` returns int directly
- No `Move`, no `decanonicalize`
- `MoveRecord(action=int, player=int, policy=1D)`

- [ ] **Step 4: Implement trainer.py**

Port from `src/bridgit/ai/trainer.py`. Key changes:
- `train()` receives `game_factory`, `net: BaseNeuralNet`, and config
- Uses `batched_self_play` from `core.self_play`
- Uses `Arena` from `core.arena`
- Uses `GreedyMCTSPlayer` from `core.players`
- `examples_from_records` from `core.data` receives `game_factory`
- No `BridgitNet`, no `NetWrapper`, no `BoardConfig`

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest test/test_core/test_self_play.py -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add src/bridgit/core/self_play.py src/bridgit/core/trainer.py test/test_core/test_self_play.py
git commit -m "feat: add generic batched self-play and trainer"
```

---

## Task 7: BridgitGame implementing Board2DGame

**Files:**
- Create: `src/bridgit/games/__init__.py`
- Create: `src/bridgit/games/bridgit/__init__.py`
- Create: `src/bridgit/games/bridgit/game.py`
- Create: `src/bridgit/games/bridgit/config.py`
- Move: `src/bridgit/game/union_find.py` → `src/bridgit/games/bridgit/union_find.py`
- Move: `src/bridgit/schema/player.py` → `src/bridgit/games/bridgit/player.py`
- Test: `test/test_games/__init__.py`
- Test: `test/test_games/test_bridgit/__init__.py`
- Test: `test/test_games/test_bridgit/test_bridgit_game.py`

- [ ] **Step 1: Write failing tests**

```python
# test/test_games/test_bridgit/test_bridgit_game.py
import torch

from bridgit.games.bridgit.game import BridgitGame, BridgitGameState
from bridgit.games.bridgit.config import BoardConfig
from bridgit.core.base_game import BaseGame, Board2DGame, GameState


class TestBridgitGame:
    def test_is_base_game(self):
        game = BridgitGame()
        assert isinstance(game, BaseGame)
        assert isinstance(game, Board2DGame)

    def test_action_space_size(self):
        game = BridgitGame(BoardConfig(size=5))
        g = 2 * 5 + 1
        assert game.action_space_size == g * g

    def test_initial_state(self):
        game = BridgitGame()
        assert game.current_player == 0
        assert not game.is_over
        assert game.winner is None

    def test_mask_is_1d(self):
        game = BridgitGame()
        mask = game.to_mask()
        assert mask.dim() == 1
        assert mask.shape[0] == game.action_space_size

    def test_get_state_returns_bridgit_state(self):
        game = BridgitGame()
        state = game.get_state()
        assert isinstance(state, GameState)
        assert isinstance(state, BridgitGameState)

    def test_make_action_and_player_switch(self):
        game = BridgitGame(BoardConfig(size=3))
        actions = game.valid_actions()
        game.make_action(actions[0])
        assert game.current_player in (0, 1)

    def test_copy_independent(self):
        game = BridgitGame()
        actions = game.valid_actions()
        game.make_action(actions[0])
        copy = game.copy()
        copy.make_action(copy.valid_actions()[0])
        assert game.to_mask().sum() != copy.to_mask().sum()

    def test_game_completes(self):
        """Random play should eventually end."""
        import random
        game = BridgitGame(BoardConfig(size=3))
        for _ in range(200):
            if game.is_over:
                break
            actions = game.valid_actions()
            game.make_action(random.choice(actions))
        assert game.is_over
        assert game.winner is not None

    def test_canonical_mask_matches_canonical_state(self):
        """Mask should reflect the canonical perspective."""
        game = BridgitGame(BoardConfig(size=3))
        mask = game.to_mask()
        valid = game.valid_actions()
        for a in valid:
            assert mask[a].item() is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest test/test_games/test_bridgit/test_bridgit_game.py -v`
Expected: FAIL

- [ ] **Step 3: Copy union_find.py and player.py**

Copy `src/bridgit/game/union_find.py` → `src/bridgit/games/bridgit/union_find.py` (no changes needed).
Copy `src/bridgit/schema/player.py` → `src/bridgit/games/bridgit/player.py` (no changes needed).

- [ ] **Step 4: Create Bridgit config**

```python
# src/bridgit/games/bridgit/config.py
"""Bridgit-specific configuration."""

from pydantic import BaseModel


class BoardConfig(BaseModel):
    size: int = 5

    @property
    def grid_size(self) -> int:
        return 2 * self.size + 1


class NeuralNetConfig(BaseModel):
    num_channels: int = 64
    num_res_blocks: int = 4
```

- [ ] **Step 5: Implement BridgitGameState and BridgitGame**

Create `src/bridgit/games/bridgit/game.py`. This merges the current `Bridgit` class and `GameState` class, adapting them to the `Board2DGame` interface:

- `BridgitGameState(GameState)` — holds the canonical board (numpy array), `n`, and `moves_left_in_turn`
- `BridgitGame(Board2DGame)` — implements all abstract methods:
  - `current_player` returns 0 (HORIZONTAL) or 1 (VERTICAL)
  - `make_action(action: int)` converts action to (row, col), handles canonical→actual if VERTICAL, delegates to internal state
  - `to_mask()` returns 1D flattened mask from canonical perspective
  - `get_state()` returns `BridgitGameState` from canonical perspective
  - Keeps `Player` enum, `UnionFind` internally — not exposed

Port all logic from current `bridgit.py` and `state.py`. The key adaptations are:
- `to_mask()` must return a **1D boolean tensor** (current code returns 2D float) — flatten the (g,g) mask and use `dtype=torch.bool`
- `make_action(action: int)` converts the flat action to (row, col) via `action_to_row_col`, then if current player is VERTICAL, transposes the coordinates before placing on the internal board
- `get_state()` returns `BridgitGameState` containing the canonical board, `n`, and `moves_left_in_turn`

- [ ] **Step 6: Run tests to verify they pass**

Run: `python -m pytest test/test_games/test_bridgit/test_bridgit_game.py -v`
Expected: All PASS

- [ ] **Step 7: Commit**

```bash
git add src/bridgit/games/ test/test_games/
git commit -m "feat: implement BridgitGame(Board2DGame) with canonical action handling"
```

---

## Task 8: BridgitNet implementing BaseNeuralNet

**Files:**
- Create: `src/bridgit/games/bridgit/neural_net.py`
- Test: `test/test_games/test_bridgit/test_neural_net.py`

- [ ] **Step 1: Write failing tests**

```python
# test/test_games/test_bridgit/test_neural_net.py
import torch

from bridgit.games.bridgit.game import BridgitGame, BridgitGameState
from bridgit.games.bridgit.neural_net import BridgitNet
from bridgit.games.bridgit.config import BoardConfig, NeuralNetConfig
from bridgit.core.base_neural_net import BaseNeuralNet


class TestBridgitNet:
    def test_is_base_neural_net(self):
        net = BridgitNet()
        assert isinstance(net, BaseNeuralNet)

    def test_encode_shape(self):
        net = BridgitNet()
        game = BridgitGame()
        state = game.get_state()
        tensor = net.encode(state)
        g = BoardConfig().grid_size
        assert tensor.shape == (4, g, g)

    def test_predict_shapes(self):
        net = BridgitNet()
        game = BridgitGame()
        state = game.get_state()
        policy, value = net.predict(state)
        assert policy.shape == (game.action_space_size,)
        assert isinstance(value, float)

    def test_predict_batch(self):
        net = BridgitNet()
        game = BridgitGame()
        states = [game.get_state(), game.get_state()]
        policies, values = net.predict_batch(states)
        assert policies.shape == (2, game.action_space_size)
        assert values.shape == (2,)

    def test_checkpoint_roundtrip(self, tmp_path):
        net = BridgitNet()
        path = str(tmp_path / "test.pt")
        net.save_checkpoint(path)

        net2 = BridgitNet()
        net2.load_checkpoint(path)

        game = BridgitGame()
        state = game.get_state()
        p1, v1 = net.predict(state)
        p2, v2 = net2.predict(state)
        assert torch.allclose(p1, p2)

    def test_policy_is_1d_flattened(self):
        """Policy should be flattened g*g, not (g, g)."""
        net = BridgitNet()
        game = BridgitGame()
        state = game.get_state()
        policy, _ = net.predict(state)
        g = BoardConfig().grid_size
        assert policy.shape == (g * g,)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest test/test_games/test_bridgit/test_neural_net.py -v`
Expected: FAIL

- [ ] **Step 3: Implement BridgitNet**

Port from `src/bridgit/ai/neural_net.py`. Key changes:
- Inherits from `BaseNeuralNet` instead of `nn.Module`
- `encode(state: BridgitGameState) → (4, g, g)` tensor — builds channels from state.board and state.moves_left
- `forward()` returns `(batch, g*g)` policy (flattened) and `(batch, 1)` value
- `save_checkpoint` / `load_checkpoint` / `copy` implemented
- No `NetWrapper` — `BaseNeuralNet` provides `predict`, `predict_batch`, `train_on_examples`

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest test/test_games/test_bridgit/test_neural_net.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/bridgit/games/bridgit/neural_net.py test/test_games/test_bridgit/test_neural_net.py
git commit -m "feat: implement BridgitNet(BaseNeuralNet) with encode/forward"
```

---

## Task 9: Integration test — full training loop with Bridgit

**Files:**
- Test: `test/test_integration/test_full_pipeline.py`

- [ ] **Step 1: Write integration test**

```python
# test/test_integration/test_full_pipeline.py
"""Smoke test: run a tiny training iteration with the generic engine + Bridgit."""

from bridgit.core.self_play import batched_self_play
from bridgit.core.arena import Arena
from bridgit.core.players import RandomPlayer, GreedyMCTSPlayer
from bridgit.core.data import examples_from_records
from bridgit.core.config import MCTSConfig, TrainingConfig
from bridgit.games.bridgit.game import BridgitGame
from bridgit.games.bridgit.neural_net import BridgitNet
from bridgit.games.bridgit.config import BoardConfig


class TestFullPipeline:
    def test_self_play_train_arena_cycle(self):
        board_config = BoardConfig(size=3)
        game_factory = lambda: BridgitGame(board_config)
        mcts_config = MCTSConfig(num_simulations=5, num_parallel_leaves=1)

        net = BridgitNet(board_config=board_config)

        # 1. Self-play
        records = batched_self_play(
            net=net,
            game_factory=game_factory,
            mcts_config=mcts_config,
            num_games=2,
            batch_size=2,
            verbose=False,
        )
        assert len(records) == 2

        # 2. Extract examples and train
        examples = examples_from_records(
            records,
            game_factory=lambda cfg: BridgitGame(BoardConfig(**cfg)),
        )
        assert len(examples) > 0

        metrics = net.train_on_examples(examples, num_epochs=2, batch_size=4)
        assert metrics["loss"] > 0

        # 3. Arena
        new_player = GreedyMCTSPlayer(net, mcts_config, name="new")
        random_player = RandomPlayer(name="random")
        arena = Arena(
            new_player, random_player,
            game_factory=game_factory,
            game_type="bridgit",
            game_config=board_config.model_dump(),
        )
        eval_records = arena.play_games(2, verbose=False)
        assert len(eval_records) == 2
```

- [ ] **Step 2: Run integration test**

Run: `python -m pytest test/test_integration/test_full_pipeline.py -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add test/test_integration/
git commit -m "test: add integration test for full training pipeline with Bridgit"
```

---

## Task 10: Update package __init__.py and clean up old code

**Files:**
- Modify: `src/bridgit/__init__.py`
- Modify: `src/bridgit/core/__init__.py`
- Create: `src/bridgit/games/bridgit/__init__.py` (update exports)

- [ ] **Step 1: Update core/__init__.py exports**

```python
# src/bridgit/core/__init__.py
from bridgit.core.base_game import BaseGame, Board2DGame, GameState
from bridgit.core.base_neural_net import BaseNeuralNet
from bridgit.core.mcts import MCTS, MCTSNode
from bridgit.core.self_play import batched_self_play
from bridgit.core.arena import Arena
from bridgit.core.players import BasePlayer, RandomPlayer, MCTSPlayer, GreedyMCTSPlayer
from bridgit.core.game_record import MoveRecord, GameRecord, GameRecordCollection, EvalResult
from bridgit.core.data import examples_from_records
from bridgit.core.config import MCTSConfig, TrainingConfig, ArenaConfig
```

- [ ] **Step 2: Update games/bridgit/__init__.py exports**

```python
# src/bridgit/games/bridgit/__init__.py
from bridgit.games.bridgit.game import BridgitGame, BridgitGameState
from bridgit.games.bridgit.neural_net import BridgitNet
from bridgit.games.bridgit.config import BoardConfig, NeuralNetConfig
```

- [ ] **Step 3: Update top-level __init__.py**

```python
# src/bridgit/__init__.py
from bridgit.core import BaseGame, Board2DGame, GameState, BaseNeuralNet
from bridgit.games.bridgit import BridgitGame, BridgitNet
```

- [ ] **Step 4: Run full test suite**

Run: `python -m pytest test/ -v`
Expected: All new tests PASS. Old tests may fail due to import changes — that's expected and will be fixed in Task 11.

- [ ] **Step 5: Commit**

```bash
git add src/bridgit/__init__.py src/bridgit/core/__init__.py src/bridgit/games/bridgit/__init__.py
git commit -m "feat: update package exports for new structure"
```

---

## Task 11: Migrate old tests, play.py, visualizer, and notebooks

**Files:**
- Modify: `test/test_game/test_bridgit.py` — update imports
- Modify: `test/test_game/test_state.py` — update imports
- Modify: `test/test_game_completion.py` — update imports
- Modify: `play.py` — update imports
- Move: `src/bridgit/visualizer.py` → `src/bridgit/games/bridgit/visualizer.py` — update imports
- Create: `test/test_integration/__init__.py`

- [ ] **Step 1: Move visualizer.py to games/bridgit/**

Move `src/bridgit/visualizer.py` to `src/bridgit/games/bridgit/visualizer.py`. Update all imports inside it from `bridgit.game`, `bridgit.schema` to `bridgit.games.bridgit`.

- [ ] **Step 2: Update test imports to use new paths**

Change all `from bridgit.game import Bridgit` to `from bridgit.games.bridgit.game import BridgitGame` (or adapt tests to use the new API with int actions instead of Move objects).

- [ ] **Step 3: Update play.py imports**

`play.py` imports from `bridgit.game`, `bridgit.schema` — update to `bridgit.games.bridgit`.

- [ ] **Step 4: Create missing __init__.py files**

Create `test/test_integration/__init__.py` (empty).

- [ ] **Step 5: Update notebook imports**

Notebooks in `notebooks/` that import from `bridgit.game`, `bridgit.ai`, `bridgit.schema`, etc. need updated imports. At minimum update the import cells — full notebook re-run is not required.

- [ ] **Step 6: Run full test suite**

Run: `python -m pytest test/ -v`
Expected: All PASS

- [ ] **Step 7: Commit**

```bash
git add test/ play.py src/bridgit/visualizer.py src/bridgit/games/bridgit/visualizer.py notebooks/
git commit -m "refactor: migrate tests, play.py, visualizer, and notebooks to new import paths"
```

---

## Task 12: Remove old code

**Files:**
- Remove: `src/bridgit/ai/` (entire directory)
- Remove: `src/bridgit/players/` (entire directory)
- Remove: `src/bridgit/schema/` (entire directory)
- Remove: `src/bridgit/data/` (entire directory)
- Remove: `src/bridgit/game/` (entire directory)
- Remove: `src/bridgit/config.py` (replaced by `core/config.py` + `games/bridgit/config.py`)

- [ ] **Step 1: Verify no imports reference old paths**

Run: `grep -r "from bridgit.ai\|from bridgit.players\|from bridgit.schema\|from bridgit.data\|from bridgit.game" src/ test/ play.py`
Expected: No matches (all migrated)

- [ ] **Step 2: Remove old directories**

```bash
rm -rf src/bridgit/ai src/bridgit/players src/bridgit/schema src/bridgit/data src/bridgit/game
rm src/bridgit/config.py
```

- [ ] **Step 3: Run full test suite**

Run: `python -m pytest test/ -v`
Expected: All PASS

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "refactor: remove old Bridgit-coupled code, migration complete"
```
