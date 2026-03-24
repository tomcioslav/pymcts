# MkDocs Documentation — Design Spec

## Goal

Create comprehensive MkDocs documentation for pymcts that serves two audiences: ML researchers who just need the API, and Python developers who need conceptual background on AlphaZero/MCTS before building their own game.

## Tech Stack

- **mkdocs** with **mkdocs-material** theme
- **mkdocstrings[python]** for auto-generated API reference from docstrings
- Hosted as static site (GitHub Pages or local `mkdocs serve`)

## Site Structure

```
docs/
├── index.md                     # Landing page — what is pymcts, quick links
├── getting-started/
│   ├── installation.md          # uv, pip, deps
│   └── quickstart.md            # Train Bridgit in 10 lines, see results
├── concepts/
│   ├── how-it-works.md          # AlphaZero loop: self-play → train → evaluate
│   ├── mcts.md                  # What is MCTS, how neural net guides it
│   └── architecture.md          # BaseGame / BaseNeuralNet / engine split
├── guide/
│   ├── creating-a-game.md       # Step-by-step: implement BaseGame for Tic-tac-toe
│   ├── creating-a-neural-net.md # Step-by-step: implement BaseNeuralNet
│   ├── training.md              # Config, running training, checkpoints
│   └── evaluation.md            # Arena, comparing models, visualizing games
├── reference/                   # Auto-generated from docstrings via mkdocstrings
│   ├── core.md                  # BaseGame, Board2DGame, GameState, BaseNeuralNet
│   ├── engine.md                # MCTS, self-play, trainer, arena, players
│   ├── config.md                # All config classes
│   └── bridgit.md               # BridgitGame, BridgitNet as reference implementation
└── bridgit/
    └── rules.md                 # Bridgit game rules
```

## Page Content Overview

### index.md
- One-paragraph pitch: generic AlphaZero engine, implement Game + Net, everything works
- Feature highlights (batched MCTS, arena evaluation, game-agnostic)
- Quick links: "I want to train Bridgit" → quickstart, "I want to add my own game" → guide, "I want to understand how it works" → concepts

### getting-started/installation.md
- Prerequisites (Python 3.10+, uv)
- Install from source with `uv pip install -e ".[all]"`
- Verify installation

### getting-started/quickstart.md
- 10-line script: create BridgitGame, BridgitNet, call train()
- Expected output (self-play progress, training loss, arena results)
- What just happened (brief explanation linking to concepts)

### concepts/how-it-works.md
- The AlphaZero loop explained simply: self-play generates training data → neural net learns from it → arena compares new vs old → repeat
- Diagram of the loop
- Why this works (self-improvement through self-play)

### concepts/mcts.md
- What is MCTS (selection, expansion, evaluation, backpropagation)
- How the neural net guides search (policy prior, value estimate)
- Temperature and exploration vs exploitation
- Virtual loss and batching (for performance)

### concepts/architecture.md
- The three-layer design: BaseGame (abstract) → Board2DGame (2D helper) → BridgitGame (concrete)
- BaseNeuralNet: encode() + forward(), predict/train come free
- GameState is opaque to the engine
- Actions are integers, mask is 1D boolean
- Canonicalization is the game's responsibility
- Diagram showing what the engine sees vs what the game handles

### guide/creating-a-game.md
- Full walkthrough using **Tic-tac-toe** as the example (not added to codebase, just in docs)
- Step 1: Define TicTacToeState(GameState)
- Step 2: Define TicTacToeGame(BaseGame) — action_space_size=9, current_player, is_over, winner, get_state, to_mask, make_action, copy, get_result
- Step 3: Test it with RandomPlayer
- Complete working code at the end
- Tips: canonicalization strategies, when to use Board2DGame

### guide/creating-a-neural-net.md
- Full walkthrough: TicTacToeNet(BaseNeuralNet)
- Step 1: encode(state) → tensor (explain the design choice)
- Step 2: __init__ — define layers (simple MLP for tic-tac-toe)
- Step 3: forward(x) → (policy, value)
- Step 4: Test with predict() and predict_batch()
- Tips: ResNet vs MLP vs Transformer, when to override train_on_examples

### guide/training.md
- Config classes: MCTSConfig, TrainingConfig, ArenaConfig
- The train() function — what it does step by step
- Checkpointing and resuming
- Monitoring training (loss curves, arena win rates)
- Hyperparameter guidance

### guide/evaluation.md
- Arena: pitting models against each other
- MCTSPlayer and GreedyMCTSPlayer
- GameRecord and GameRecordCollection
- Visualizing games (Bridgit's Visualizer as example)

### reference/* (auto-generated)
- Each page uses mkdocstrings directives to pull from source
- core.md: `::: pymcts.core.base_game`, `::: pymcts.core.base_neural_net`
- engine.md: `::: pymcts.core.mcts`, `::: pymcts.core.self_play`, `::: pymcts.core.trainer`, `::: pymcts.core.arena`, `::: pymcts.core.players`
- config.md: `::: pymcts.core.config`
- bridgit.md: `::: pymcts.games.bridgit.game`, `::: pymcts.games.bridgit.neural_net`

### bridgit/rules.md
- Bridgit game rules (board, players, win condition)
- The canonical transformation explained
- Board encoding details

## Configuration Files

### mkdocs.yml
```yaml
site_name: pymcts
site_description: A generic AlphaZero-style training engine for two-player zero-sum games
repo_url: https://github.com/tomcioslav/pymcts

theme:
  name: material
  features:
    - content.code.copy
    - content.code.annotate
    - navigation.sections
    - navigation.expand
    - search.suggest
  palette:
    - scheme: default
      primary: indigo
      toggle:
        icon: material/brightness-7
        name: Switch to dark mode
    - scheme: slate
      primary: indigo
      toggle:
        icon: material/brightness-4
        name: Switch to light mode

plugins:
  - search
  - mkdocstrings:
      handlers:
        python:
          options:
            show_source: true
            show_root_heading: true
            members_order: source

nav:
  - Home: index.md
  - Getting Started:
    - Installation: getting-started/installation.md
    - Quick Start: getting-started/quickstart.md
  - Concepts:
    - How It Works: concepts/how-it-works.md
    - MCTS Explained: concepts/mcts.md
    - Architecture: concepts/architecture.md
  - Guide:
    - Creating a Game: guide/creating-a-game.md
    - Creating a Neural Net: guide/creating-a-neural-net.md
    - Training: guide/training.md
    - Evaluation: guide/evaluation.md
  - API Reference:
    - Core Abstractions: reference/core.md
    - Engine Components: reference/engine.md
    - Configuration: reference/config.md
    - Bridgit Implementation: reference/bridgit.md
  - Bridgit:
    - Game Rules: bridgit/rules.md

markdown_extensions:
  - pymdownx.highlight
  - pymdownx.superfences
  - pymdownx.tabbed:
      alternate_style: true
  - admonition
  - pymdownx.details
```

### pyproject.toml additions
```toml
[project.optional-dependencies]
docs = [
    "mkdocs>=1.5.0",
    "mkdocs-material>=9.0.0",
    "mkdocstrings[python]>=0.24.0",
]
```

## Design Decisions

- **Tic-tac-toe as teaching example**: simpler than Bridgit, universally understood, demonstrates all abstractions without Bridgit-specific complexity (union-find, bridge endpoints, canonical transposition). Lives only in docs, not in codebase.
- **Auto-generated API reference**: keeps docs in sync with code. Docstrings are the source of truth for API details; hand-written guides provide narrative and context.
- **Concepts section is optional reading**: researchers can skip to the guide. Developers who need background can read concepts first. Clear links between sections.
- **No generated code in the repo**: the Tic-tac-toe example is documentation-only. If someone wants to run it, they copy-paste from the guide.

## What This Does NOT Include

- Deployment to GitHub Pages (can be added later with a GH Actions workflow)
- API versioning
- Changelog
- Contributing guide
