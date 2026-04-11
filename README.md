# pymcts

A generic AlphaZero-style training engine for two-player zero-sum games. Implement a `Game` class and a `NeuralNet` class — MCTS, self-play, training, and arena evaluation work automatically.

Ships with **Bridgit** as the first game implementation.

## Installation

```bash
# Clone the repository
git clone git@github.com:tomcioslav/pymcts.git
cd pymcts

# Create virtual environment and install
uv venv && source .venv/bin/activate
uv pip install -e ".[all]"
```

Requires Python 3.10+ and [uv](https://github.com/astral-sh/uv).

## Quick Start

### Train on Bridgit

```python
from pymcts.games.bridgit.game import BridgitGame
from pymcts.games.bridgit.config import BoardConfig, NeuralNetConfig
from pymcts.games.bridgit.neural_net import BridgitNet
from pymcts.core.config import MCTSConfig, TrainingConfig
from pymcts.arena import SinglePlayerArena
from pymcts.arena.config import SinglePlayerArenaConfig
from pathlib import Path
from pymcts.core.trainer import train

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

### Play Bridgit (GUI)

```bash
python play.py        # default 5x5
python play.py 7      # custom size
```

## Architecture

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

## Adding a New Game

Implement three things:

1. **`MyGameState(GameState)`** — your game's state representation
2. **`MyGame(BaseGame)` or `MyGame(Board2DGame)`** — game logic, actions as integers, canonical state via `get_state()`
3. **`MyNet(BaseNeuralNet)`** — implement `encode(state) -> tensor` and `forward(tensor) -> (policy, value)`

The engine handles everything else: MCTS tree search, batched self-play, training loop, arena evaluation.

Key design principles:
- **Actions are integers** (0 to `action_space_size - 1`). The game maps to/from internal representation.
- **Game state is opaque** to the engine. The neural net's `encode()` converts it to tensors.
- **Canonicalization is the game's responsibility**. `get_state()` and `to_mask()` always return from the current player's perspective.

## Notebooks

- `notebooks/training.ipynb` — full training pipeline
- `notebooks/arena.ipynb` — model comparison and batched self-play benchmarks
- `notebooks/mcts.ipynb` — MCTS visualization and tree inspection
- `notebooks/analysis.ipynb` — training run analysis
- `notebooks/game.ipynb` — game mechanics exploration

## Running Tests

```bash
pytest test/ -v
```

## License

MIT License - see LICENSE file for details
