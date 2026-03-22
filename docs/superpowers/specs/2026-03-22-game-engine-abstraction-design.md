# Game Engine Abstraction — Design Spec

## Goal

Decouple the AI training infrastructure (MCTS, self-play, trainer, arena) from the Bridgit game so the same engine can train on any two-player zero-sum game. The game developer only implements two things: a `Game` class and a `NeuralNet` class. Everything else works automatically.

## Architecture Overview

```
core/                          # Generic engine — game-agnostic
├── base_game.py               # BaseGame ABC
├── board_2d_game.py           # Board2DGame(BaseGame) — 2D grid helper
├── base_neural_net.py         # BaseNeuralNet ABC
├── mcts.py                    # MCTS operating on action ints
├── self_play.py               # Batched self-play using game/net factories
├── trainer.py                 # AlphaZero training loop
├── arena.py                   # Model comparison
├── players.py                 # RandomPlayer, MCTSPlayer
├── game_record.py             # Generic game recording (action ints)
├── data.py                    # Training data extraction
└── config.py                  # Generic configs (MCTS, training, arena, net)

games/
└── bridgit/                   # Bridgit-specific implementation
    ├── game.py                # BridgitGame(Board2DGame)
    ├── state.py               # GameState, UnionFind (internal)
    ├── neural_net.py          # BridgitNet(BaseNeuralNet)
    ├── config.py              # BoardConfig (size, grid_size formula)
    └── visualizer.py          # Plotly visualization
```

## Core Abstractions

### BaseGame

The minimal contract that MCTS, trainer, arena, and self-play depend on. All game-specific logic (canonicalization, win detection, board encoding) is hidden behind this interface.

```python
from abc import ABC, abstractmethod
import torch

class GameState:
    """Base class for game states. Each game defines its own subclass.
    The engine passes GameState objects opaquely — only the game and its
    neural net know the concrete type.
    """
    pass

class BaseGame(ABC):

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
        """Total number of possible actions. Fixed for a game type, not state-dependent."""

    @abstractmethod
    def get_state(self) -> GameState:
        """Return the current game state from the current player's perspective.
        Canonicalization is the game's responsibility — the returned state
        should always be from the current player's POV.
        The engine treats this as opaque. Only the game's neural net
        knows the concrete type and how to encode it into tensors.
        """

    @abstractmethod
    def to_mask(self) -> torch.Tensor:
        """1D boolean tensor of length action_space_size.
        True = legal action. Matches the action index space.
        """

    def valid_actions(self) -> list[int]:
        """List of currently legal action indices.
        Default implementation derives from to_mask(). Override for performance if needed.
        """
        return self.to_mask().nonzero(as_tuple=False).squeeze(-1).tolist()

    @abstractmethod
    def make_action(self, action: int) -> None:
        """Apply action in-place. Mutates the game state.
        The game internally handles any canonicalization
        (e.g., if the current player is VERTICAL in Bridgit,
        the game transposes the action coordinates internally).
        """

    @abstractmethod
    def copy(self) -> "BaseGame": ...

    @abstractmethod
    def get_result(self, player: int) -> float | None:
        """1.0 for win, -1.0 for loss, 0.0 for draw, None if not over."""
```

**Key design decisions:**
- Players are ints (0, 1, ...), not enums, in the generic layer.
- Actions are ints (0 to `action_space_size - 1`). The game maps these to whatever internal representation it uses.
- **Game state is opaque to the engine.** `get_state()` returns a `GameState` subclass that the engine passes to `BaseNeuralNet.predict()` without inspecting. The neural net knows the concrete type and encodes it into tensors internally. This decouples the game from the net architecture — you can try a ResNet and a Transformer on the same game without changing the Game class.
- `to_mask()` stays on `BaseGame` because legality is a game concept, not an architecture concept. It is 1D — even for 2D board games, the mask is flattened.
- `action_space_size` is fixed for a game type — it represents all moves that could ever be legal, not just those legal in the current state. The mask filters.

### Board2DGame

A thin convenience layer for games on rectangular grids. Provides action-to-coordinate mapping. Does not add any game logic.

```python
class Board2DGame(BaseGame):

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

**Note:** `action_space_size` is a concrete property here but can be overridden by subclasses that need a different calculation (e.g., Go with a pass action: `rows * cols + 1`).

**For Bridgit:** `BridgitGame(Board2DGame)` uses `board_rows = board_cols = grid_size` (i.e., `2*n+1`). The action space is `grid_size * grid_size`. Many of those actions are always-illegal (boundary cells, non-crossing cells) — the mask handles this. The wasted policy slots are negligible and this avoids needing a custom index mapping.

### BaseNeuralNet

The contract for neural networks. The engine calls `predict`, `predict_batch`, and `train_on_examples`. It never inspects tensor shapes.

```python
class BaseNeuralNet(ABC, nn.Module):
    """Base class for all neural networks. Inherits from both ABC and nn.Module,
    so subclasses are standard PyTorch modules with forward(), parameters(), etc.
    """

    # --- Abstract: the developer implements these ---

    @abstractmethod
    def encode(self, state: GameState) -> torch.Tensor:
        """Convert a GameState into the tensor format this architecture expects.
        A ResNet might return (4, g, g). A Transformer might return (seq_len, d_model).
        This is where game state meets architecture.
        """

    @abstractmethod
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Raw forward pass on a batch of encoded tensors.
        Input: (batch, *encoded_shape)
        Returns: (log_policy (batch, action_space_size), value (batch, 1))
        """

    @abstractmethod
    def save_checkpoint(self, path: str) -> None: ...

    @abstractmethod
    def load_checkpoint(self, path: str) -> None: ...

    @abstractmethod
    def copy(self) -> "BaseNeuralNet":
        """Deep copy for arena comparison (current vs new)."""

    # --- Concrete defaults: work out of the box ---

    def predict(self, state: GameState) -> tuple[torch.Tensor, float]:
        """Single state -> (policy_1D, value). Encodes, forwards, unwraps batch dim."""
        tensor = self.encode(state).unsqueeze(0)
        policy, value = self.forward(tensor)
        return policy.squeeze(0), value.item()

    def predict_batch(self, states: list[GameState]) -> tuple[torch.Tensor, torch.Tensor]:
        """Batch of states -> (policies, values). Encodes each, stacks, forwards."""
        tensors = torch.stack([self.encode(s) for s in states])
        policies, values = self.forward(tensors)
        return policies, values.squeeze(-1)

    def train_on_examples(
        self, examples: list[tuple[GameState, torch.Tensor, float]]
    ) -> dict:
        """Default training loop: cross-entropy policy loss + MSE value loss.
        Override for custom training (LR schedules, gradient clipping, etc.).
        Returns metrics dict {"loss": ..., "policy_loss": ..., "value_loss": ...}.
        """
        # Default implementation encodes all states, computes standard losses,
        # runs optimizer step. Concrete class provides self.optimizer, self.device, etc.
```

**Key decisions:**
- **The developer implements `encode()` and `forward()`.** Everything else has concrete defaults. `predict`, `predict_batch`, and `train_on_examples` work out of the box.
- **`encode()` is the decoupling point.** A ResNet encodes `BridgitGameState` into (4, g, g) channels. A Transformer encodes the same state into a token sequence. The Game class doesn't change.
- Policy output is always **1D** of length `action_space_size`. If BridgitNet uses a (g,g) conv head internally, `forward()` flattens before returning. The engine never sees 2D.
- `train_on_examples` has a sensible default (cross-entropy + MSE, Adam optimizer) but can be overridden for custom training logic.
- `copy()` is needed for the arena pattern (compare new weights vs old).
- **Device management** (CPU/GPU/MPS) is the `BaseNeuralNet` implementation's responsibility. The engine never calls `.to(device)` — the net handles it internally.

## Generic Engine Components

### MCTS

**Current state:** Calls `game.to_tensor()`, `game.to_mask()`, `game.make_move(Move(row, col))`. Uses `Move.decanonicalize()`. Stores `Move` objects in nodes.

**After refactor:**
- `MCTSNode` stores `action: int`, not `Move`.
- Priors become `dict[int, float]` (action index → prior probability), replacing the current `dict[Move, float]`.
- `visit_counts` becomes a 1D array of length `action_space_size` or `dict[int, int]`.
- Tree expansion: `game.copy()` then `game.make_action(action)`.
- Neural net evaluation: `net.predict(game.get_state())` returns 1D policy. Setting priors is simply `{a: policy[a] for a in game.valid_actions()}`.
- No canonicalize/decanonicalize — the game handles it inside `make_action()` and `get_state()`.
- **Player-switching and Q-value sign flip:** The current UCB logic flips Q-value sign when parent and child have different `current_player`. This carries forward unchanged — it works with `current_player` as `int` and correctly handles multi-move turns (e.g., Bridgit's 2-moves-per-turn) because the comparison is per-node, not per-turn.

### Batched Self-Play

**Current state:** Imports `Bridgit` directly. Uses `moves_left_in_turn`. Does `divmod` to reconstruct `Move(row, col)`.

**After refactor:**
- Receives a `game_factory: Callable[[], BaseGame]` — no game import.
- Actions are ints — `torch.multinomial` on the 1D policy returns an int directly.
- `moves_left_in_turn` disappears from the generic layer. It is a Bridgit-specific turn mechanic — `BridgitGameState` carries it, and `BridgitNet.encode()` includes it as an input channel. Invisible to the engine, which just passes `GameState` to the net.
- Records `(action: int, player: int, policy: Tensor | None)` per move.

### Trainer

**Current state:** Imports `Bridgit`, `GreedyMCTSPlayer`, `Arena`.

**After refactor:**
- Receives `game_factory: Callable[[], BaseGame]` and `net_factory: Callable[[], BaseNeuralNet]`.
- Iteration loop (self-play -> train -> arena evaluate) stays the same structurally.
- `BoardConfig` moves out — the generic trainer only needs `TrainingConfig`, `MCTSConfig`, `ArenaConfig`.

### Arena

**Current state:** Creates `Bridgit()` instances. Reads `move.row`, `move.col`.

**After refactor:**
- Receives `game_factory`.
- Records actions as ints.
- Never inspects what an action means.
- Determines winner via `game.winner` (returns `int | None`).

### Players

**Current state:** `RandomPlayer` calls `game.get_available_moves()` returning `list[Move]`. `MCTSPlayer` calls `Move.decanonicalize()`.

**After refactor:**
```python
class BasePlayer(ABC):
    last_policy: torch.Tensor | None = None  # 1D, set after get_action

    @abstractmethod
    def get_action(self, game: BaseGame) -> int: ...

class RandomPlayer(BasePlayer):
    def get_action(self, game: BaseGame) -> int:
        self.last_policy = None
        return random.choice(game.valid_actions())

class MCTSPlayer(BasePlayer):
    def get_action(self, game: BaseGame) -> int:
        # Run MCTS, store policy in self.last_policy, return action int.
        # No decanonicalize — game.make_action() handles it.

class GreedyMCTSPlayer(MCTSPlayer):
    """MCTSPlayer with temperature=0 — always picks most-visited action."""
```

`BasePlayer.last_policy` is used by the arena to record MCTS policies alongside moves. `player_factory.py` is removed — the factory pattern (`game_factory`, `net_factory`) replaces serialization-based player reconstruction.

### GameRecord (Generic)

**Current state:** Stores `Move(row, col)`, `Player` enum, `moves_left_after`.

**After refactor:**
```python
class MoveRecord(BaseModel):
    action: int
    player: int
    policy: list[float] | None   # 1D, length action_space_size

class GameRecord(BaseModel):
    game_type: str                # "bridgit", "hex", etc.
    game_config: dict             # game-specific params (opaque to engine)
    moves: list[MoveRecord]
    winner: int | None
    player_names: list[str]       # indexed by player id
```

No more `moves_left_after`, no `Player` enum, no `Move` object in the generic layer.

### GameRecordCollection and EvalResult

The current `GameRecordCollection` (aggregates multiple `GameRecord`s) and `EvalResult` (wins, losses, win_rate, avg_moves) move to `core/game_record.py` alongside `MoveRecord` and `GameRecord`. They become generic:

```python
class EvalResult(BaseModel):
    wins: int
    losses: int
    draws: int
    win_rate: float
    avg_moves: float

class GameRecordCollection(BaseModel):
    records: list[GameRecord]

    def evaluate(self, player_name: str) -> EvalResult: ...
    def is_better(self, player_name: str, threshold: float) -> bool: ...
```

These are already mostly game-agnostic — they just count wins/losses. The only change is replacing `Player` enum references with `int` player ids.

### Data Converter

**Current state:** Replays games using `Bridgit()`, calls `game.to_tensor()`.

**After refactor:**
- Receives `game_factory` (constructed from `game_record.game_config`).
- Replays via `game.make_action(move_rec.action)`.
- Extracts `(game.get_state(), move_rec.policy, value)` tuples — the net encodes the `GameState` during training.
- Value determined by comparing `game.current_player` with `record.winner`.

### Config

**Generic configs (stay in `core/`):**
- `MCTSConfig`: num_simulations, c_puct, dirichlet_alpha, epsilon
- `TrainingConfig`: epochs, batch_size, lr, weight_decay, replay_buffer_size, num_iterations, num_self_play_games
- `ArenaConfig`: num_games, threshold, swap_players

**Note:** `NeuralNetConfig` (num_channels, num_res_blocks) moves to `games/bridgit/config.py` since it is ResNet-specific. A game using a transformer or MLP would define its own net config. The generic engine does not need net architecture params — it only calls the `BaseNeuralNet` interface.

**Game-specific configs (move to `games/bridgit/`):**
- `BoardConfig`: size, grid_size formula (`2*n+1`)
- Any Bridgit-specific training params

## Canonicalization Contract

The engine does **not** handle canonicalization. This is entirely the game developer's responsibility. The contract is:

1. `get_state()` returns the state from the current player's perspective.
2. `to_mask()` returns legal actions from the current player's perspective.
3. `make_action(action)` accepts actions in the current player's canonical space.
4. The neural net is trained on canonical states, so it always "thinks" it's the same player.

**How Bridgit implements this:**
- HORIZONTAL is the canonical perspective. When it's VERTICAL's turn, `get_state()` returns a `BridgitGameState` with the board transposed and negated. `to_mask()` transposes. `make_action(action)` un-transposes the action before placing.

**How another game might do it:**
- Chess: always orient from white's perspective, flip for black, or add a "current player" channel to the tensor and never flip. Developer's choice.
- Tic-tac-toe (symmetric): no canonicalization needed.

## What the Game Developer Implements (Checklist)

To add a new game to this engine:

1. **`MyGameState(GameState)`** — a dataclass/class holding the game's state representation (board, metadata, etc.).
2. **`MyGame(Board2DGame)` or `MyGame(BaseGame)`** — implement all abstract methods. `get_state()` returns `MyGameState`.
3. **`MyNet(BaseNeuralNet)`** — implement `encode(state) → tensor` and `forward(tensor) → (policy, value)`. `predict`, `predict_batch`, and `train_on_examples` work automatically. Override `train_on_examples` only if you need custom training logic.
4. **Game config** — whatever params the game needs (board size, etc.).
5. **A factory function** — `def create_game(config) -> MyGame` and `def create_net(config) -> MyNet`.

That's it. MCTS, self-play, trainer, arena, players all work automatically.

## Directory Structure (After Refactor)

```
src/bridgit/
├── core/
│   ├── __init__.py
│   ├── base_game.py            # BaseGame ABC
│   ├── board_2d_game.py        # Board2DGame(BaseGame)
│   ├── base_neural_net.py      # BaseNeuralNet ABC
│   ├── mcts.py                 # Game-agnostic MCTS
│   ├── self_play.py            # Batched self-play with factories
│   ├── trainer.py              # AlphaZero training loop
│   ├── arena.py                # Model comparison
│   ├── players.py              # RandomPlayer, MCTSPlayer
│   ├── game_record.py          # Generic MoveRecord, GameRecord
│   ├── data.py                 # Training data extraction
│   └── config.py               # MCTSConfig, TrainingConfig, ArenaConfig
├── games/
│   └── bridgit/
│       ├── __init__.py
│       ├── game.py             # BridgitGame(Board2DGame)
│       ├── state.py            # GameState (board representation)
│       ├── union_find.py       # UnionFind (win detection)
│       ├── neural_net.py       # BridgitNet(BaseNeuralNet)
│       ├── config.py           # BoardConfig
│       ├── player.py           # Player enum (HORIZONTAL, VERTICAL) — internal to Bridgit
│       └── visualizer.py       # Plotly visualization
├── __init__.py
└── config.py                   # Top-level config aggregating core + game configs
```

## Scope

This engine targets **two-player zero-sum games**. Multi-player games or cooperative games are explicitly out of scope and would require a spec revision.

## Migration Notes

- The `schema/` directory dissolves: `Player` enum and `Move` class become Bridgit-internal (`games/bridgit/`). `GameRecord`, `GameRecordCollection`, `EvalResult` move to `core/game_record.py`.
- The `data/` directory becomes `core/data.py`.
- The `players/` directory splits: `BasePlayer`, `RandomPlayer`, `MCTSPlayer`, `GreedyMCTSPlayer` go to `core/players.py`. `Arena` goes to `core/arena.py`. `player_factory.py` is removed.
- `visualizer.py` moves to `games/bridgit/visualizer.py` (it imports `GameState`, `Move`, `Player` — all Bridgit-specific).
- Existing notebooks and `play.py` will import from `games/bridgit/` instead of top-level.
- Tests need updating to match new import paths.
