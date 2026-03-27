# MCTS Explained

Monte Carlo Tree Search (MCTS) is the algorithm that decides which move to play. It builds a search tree of possible futures, using the neural network to guide where to look.

## The four phases

Each MCTS simulation has four phases:

### 1. Selection

Starting from the current position (root), walk down the tree by picking the child with the highest **UCB score**:

```
UCB(child) = Q(child) + c_puct * prior(child) * sqrt(parent_visits) / (1 + child_visits)
```

- **Q(child)**: average value of all simulations through this child (exploitation)
- **prior(child)**: neural net's policy probability for this move (guidance)
- **c_puct**: exploration constant — higher values explore more

This balances exploitation (moves that have worked well) with exploration (moves the neural net thinks are promising but haven't been tried much).

### 2. Expansion

When selection reaches a node that hasn't been visited, expand it: create a new tree node for this game state.

### 3. Evaluation

Ask the neural network to evaluate the new position:

- **Policy**: a probability distribution over all actions — which moves look promising?
- **Value**: a scalar in [-1, 1] — how good is this position for the current player?

The policy becomes the prior probabilities for the new node's children. The value is the simulation result.

### 4. Backpropagation

Walk back up to the root, updating each node along the path:

- Increment visit count
- Add the value to the running total (flipping sign when the player changes)

After many simulations (e.g., 200), the root's visit counts tell you which move was explored most — that's the best move.

## Temperature

After MCTS completes, moves are selected based on visit counts:

- **Temperature = 1**: sample proportionally to visit counts (more exploration, used early in training games)
- **Temperature → 0**: always pick the most-visited move (greedy, used for evaluation)

Early moves in a game use higher temperature to ensure diverse training data. Later moves use lower temperature for stronger play.

## Batching and virtual loss

The `MCTS` class supports two levels of batching to maximize GPU throughput:

1. **Game-level batching** — `search_batch(games)` searches multiple games at once, collecting all leaf evaluations into a single neural net call.
2. **Leaf-level batching** — with `num_parallel_leaves=K`, each simulation step selects K different leaves per game before evaluating them.

Together, a batch of N games with K parallel leaves produces up to N×K evaluations per GPU call.

**Virtual loss** makes leaf-level batching work: when a leaf is selected for evaluation, it's temporarily penalized (as if it lost) so that the next selection within the same step picks a different path. After the real evaluation comes back, the virtual loss is removed and the true value is backpropagated. With `num_parallel_leaves=1`, virtual loss has no effect.

## Configuration

Key MCTS parameters in `MCTSConfig`:

| Parameter | Default | What it does |
|---|---|---|
| `num_simulations` | 200 | Simulations per move — more = stronger but slower |
| `c_puct` | 1.5 | Exploration constant — higher = more exploration |
| `dirichlet_alpha` | 0.3 | Noise added to root priors for exploration |
| `dirichlet_epsilon` | 0.25 | How much noise to mix in (0 = none, 1 = all noise) |
| `num_parallel_leaves` | 1 | Leaves per selection step for batching |

See [Configuration](../reference/config.md) for the full reference.
