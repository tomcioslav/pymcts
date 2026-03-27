# pymcts

A generic AlphaZero-style training engine for two-player zero-sum games.

Implement a **Game** class and a **NeuralNet** class — MCTS, self-play, training, and arena evaluation work automatically.

## What you get

- **Game-agnostic MCTS** with batched neural net inference and virtual loss
- **Self-play pipeline** that generates training data from games played against itself
- **AlphaZero training loop** with automatic checkpointing and model comparison
- **Arena evaluation** to compare models and track improvement
- **Bridgit** as a complete reference implementation

## Quick links

| I want to... | Go to... |
|---|---|
| Train a model on Bridgit right now | [Quick Start](getting-started/quickstart.md) |
| Understand how AlphaZero works | [How It Works](concepts/how-it-works.md) |
| Add my own game | [Creating a Game](guide/creating-a-game.md) |
| Look up the API | [API Reference](reference/core.md) |

## How it works (30-second version)

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Self-Play   │────>│   Train     │────>│   Arena     │
│  (MCTS +     │     │  (Neural    │     │  (New vs    │
│   Neural Net)│     │   Net)      │     │   Old)      │
└─────────────┘     └─────────────┘     └──────┬──────┘
       ^                                        │
       └────────────────────────────────────────┘
                    repeat until strong
```

The neural net plays against itself using MCTS to generate training data. It trains on that data, then competes against its previous version. If it wins enough, the new weights are kept. Repeat.

## Install

```bash
git clone git@github.com:tomcioslav/pymcts.git
cd pymcts
uv venv && source .venv/bin/activate
uv pip install -e ".[all]"
```
