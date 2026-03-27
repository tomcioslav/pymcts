# How It Works

pymcts implements the **AlphaZero** algorithm — a self-play reinforcement learning approach where a neural network learns to play a game by playing against itself, guided by Monte Carlo Tree Search (MCTS).

## The training loop

```
┌──────────────────────────────────────────────────────────┐
│                    One Iteration                          │
│                                                          │
│  1. Self-Play        2. Train          3. Arena          │
│  ┌────────────┐    ┌────────────┐    ┌────────────┐     │
│  │ Play N     │    │ Train net  │    │ New model  │     │
│  │ games      │───>│ on the     │───>│ vs old     │     │
│  │ using MCTS │    │ games      │    │ model      │     │
│  └────────────┘    └────────────┘    └─────┬──────┘     │
│                                             │            │
│                                     ┌───────┴───────┐   │
│                                     │  Win rate     │   │
│                                     │  > threshold? │   │
│                                     └───┬───────┬───┘   │
│                                    yes  │       │ no    │
│                                  ┌──────┘       └────┐  │
│                                  │ Keep new    │Revert│  │
│                                  │ weights     │      │  │
│                                  └─────────────┘──────┘  │
└──────────────────────────────────────────────────────────┘
                         repeat
```

### Step 1: Self-play

The current neural network plays games against itself. For each move:

1. MCTS runs many simulations from the current position, using the neural net to evaluate positions
2. The move is selected based on visit counts (with temperature for exploration)
3. The game state, MCTS policy, and eventual outcome are recorded

This produces training data: **(state, policy, value)** tuples, where policy is the MCTS search result and value is +1 (win) or -1 (loss).

### Step 2: Training

The neural network trains on the self-play data:

- **Policy head** learns to predict the MCTS search result (cross-entropy loss)
- **Value head** learns to predict game outcomes (MSE loss)

Over time, the net gets better at both evaluating positions and suggesting good moves — which in turn makes MCTS stronger, which generates better training data.

### Step 3: Arena evaluation

The newly trained network plays against the previous version. If it wins more than a threshold (e.g., 55% of games), the new weights are accepted. Otherwise, the update is rejected and training continues with the old weights.

This prevents catastrophic forgetting — the model only updates when it's demonstrably stronger.

## Why this works

The key insight is the **virtuous cycle** between MCTS and the neural network:

- MCTS uses the neural net to evaluate positions, so a better net → stronger MCTS
- Training data comes from MCTS, so stronger MCTS → better training data → better net

The net starts random. MCTS makes the random net play slightly better than random (because tree search helps even with bad evaluation). The net trains on these slightly-better-than-random games and improves. Now MCTS with the improved net is even stronger, producing better data. And so on.

## What pymcts provides

You don't implement any of this. pymcts provides:

- MCTS with batched neural net inference
- Self-play pipeline (concurrent games for GPU efficiency)
- Training loop with configurable hyperparameters
- Arena evaluation with automatic model comparison
- Checkpointing and resume

You implement:

- **A Game class** — the rules of your game
- **A NeuralNet class** — how to encode game states and the network architecture

See [Creating a Game](../guide/creating-a-game.md) and [Creating a Neural Net](../guide/creating-a-neural-net.md).
