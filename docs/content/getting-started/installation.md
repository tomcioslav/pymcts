# Installation

## Prerequisites

- Python 3.10 or later
- [uv](https://github.com/astral-sh/uv) (recommended) or pip

## Install from source

```bash
git clone git@github.com:tomcioslav/pymcts.git
cd pymcts
```

### With uv (recommended)

```bash
uv venv
source .venv/bin/activate   # macOS/Linux
# .venv\Scripts\activate    # Windows

uv pip install -e ".[all]"
```

### With pip

```bash
python -m venv .venv
source .venv/bin/activate

pip install -e ".[all]"
```

## Dependency groups

The `[all]` extra installs everything. You can also install selectively:

| Extra | What it includes |
|---|---|
| `ai` | PyTorch, matplotlib, tensorboard |
| `gui` | Pygame (for the Bridgit GUI) |
| `dev` | pytest, ruff, black |
| `docs` | mkdocs, mkdocs-material, mkdocstrings |

Example: install only AI dependencies:

```bash
uv pip install -e ".[ai]"
```

## Verify installation

```python
from pymcts.core import BaseGame, BaseNeuralNet
from pymcts.games.bridgit import BridgitGame, BridgitNet

game = BridgitGame()
print(f"Action space: {game.action_space_size}")
print(f"Legal moves: {len(game.valid_actions())}")
```

Expected output:

```
Action space: 121
Legal moves: 25
```
