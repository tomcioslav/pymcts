# Architecture

pymcts separates the **engine** (game-agnostic) from the **game** (game-specific). The engine handles MCTS, self-play, training, and evaluation. You implement the game logic and neural network architecture.

## The boundary

```
┌─────────────────────────────────────────────────────┐
│                    Engine (core/)                     │
│                                                      │
│  MCTS ──> Self-Play ──> Trainer ──> Arena            │
│                                                      │
│  Sees only:                                          │
│    - Actions as integers (0, 1, 2, ...)              │
│    - GameState as opaque objects                     │
│    - Mask as 1D boolean tensor                       │
│    - Policy as 1D float tensor                       │
│    - Players as integers (0, 1)                      │
│                                                      │
└──────────────────────┬──────────────────────────────┘
                       │ calls BaseGame / BaseNeuralNet
┌──────────────────────┴──────────────────────────────┐
│                 Your Game (games/xyz/)                │
│                                                      │
│  Handles:                                            │
│    - Board representation                            │
│    - Move legality                                   │
│    - Win detection                                   │
│    - Canonicalization                                │
│    - Tensor encoding                                 │
│    - Network architecture                            │
│                                                      │
└─────────────────────────────────────────────────────┘
```

## Core abstractions

### GameState

A base class for game states. The engine passes these around opaquely — it never inspects the contents. Only the game's neural network knows the concrete type and how to encode it into tensors.

```python
class GameState:
    """Subclass this for your game's state."""
    pass
```

### BaseGame

The contract that the engine depends on. Every game implements these methods:

| Method | Returns | Purpose |
|---|---|---|
| `current_player` | `int` | Whose turn it is (0 or 1) |
| `is_over` | `bool` | Has the game ended? |
| `winner` | `int \| None` | Who won (or None for draw/ongoing) |
| `action_space_size` | `int` | Total possible actions (fixed per game type) |
| `get_state()` | `GameState` | Canonical state from current player's perspective (for engine/neural net) |
| `get_display_state()` | `GameState` | Absolute state for visualization (defaults to `get_state()`) |
| `to_mask()` | `1D bool Tensor` | Which actions are legal right now |
| `valid_actions()` | `list[int]` | Legal action indices (derived from mask) |
| `make_action(int)` | `None` | Apply an action (mutates in-place) |
| `copy()` | `BaseGame` | Deep copy for MCTS branching |
| `get_result(int)` | `float \| None` | 1.0=win, -1.0=loss, 0.0=draw |

### Board2DGame

A convenience layer for games on rectangular grids. Provides `action_to_row_col()` and `row_col_to_action()` mappings. The engine never uses these — they're helpers for your game implementation.

### BaseNeuralNet

The contract for neural networks. You implement two methods:

| Method | What it does |
|---|---|
| `encode(state)` | Convert GameState to tensor (your encoding) |
| `forward(tensor)` | Run the network, return (policy, value) |

The base class provides `predict()`, `predict_batch()`, and `train_on_examples()` for free — they call your `encode()` and `forward()` with the right batching and loss computation.

## Key design decisions

### Actions are integers

Every possible action in your game maps to an integer from 0 to `action_space_size - 1`. The engine never interprets what an action means — it just picks indices and passes them to `make_action()`.

For a 3x3 board game, action 0 might be (0,0), action 4 might be (1,1), etc. The game handles the mapping internally.

### Game state is opaque

`get_state()` returns a `GameState` subclass that the engine passes to the neural net without looking at it. The neural net's `encode()` method knows the concrete type and converts it to tensors.

This means you can swap neural network architectures (ResNet, Transformer, MLP) on the same game without changing the game class. Each architecture encodes the same GameState differently.

### Canonicalization is yours

`get_state()` and `to_mask()` must return results from the **current player's perspective**. This way, the neural network always "thinks" it's the same player.

How you achieve this is up to you:

- **Bridgit**: transposes the board and negates values for the second player
- **Chess**: might flip the board for black, or add a "whose turn" channel
- **Tic-tac-toe**: no canonicalization needed (symmetric)

The engine never knows canonicalization exists.

For visualization and external display, override `get_display_state()` to return the **absolute** (non-canonical) state. If your game doesn't canonicalize, you don't need to override it — the default returns `get_state()`.

## Data flow

Here's how a single MCTS simulation flows through the system:

```
1. MCTS calls game.copy()           → branched game state
2. MCTS calls game.make_action(42)  → game applies action 42
3. MCTS calls game.get_state()      → opaque GameState
4. MCTS calls net.predict(state)    → (1D policy, value)
   └─ net.encode(state)             → tensor (architecture-specific)
   └─ net.forward(tensor)           → (log_policy, value)
5. MCTS uses policy as priors, value for backpropagation
6. After N simulations, visit counts → action selection
```

The engine only touches steps 1-2 and 4-6. Steps 3 and the internals of 4 are your code.
