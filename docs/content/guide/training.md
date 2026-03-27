# Training

This guide covers how to configure and run the AlphaZero training pipeline.

## The train() function

```python
from pymcts.core.trainer import train

train(
    game_factory=lambda: MyGame(),       # creates fresh game instances
    net=my_net,                          # neural network to train
    mcts_config=mcts_config,             # MCTS settings
    training_config=training_config,     # training loop settings
    arena_config=arena_config,           # model comparison settings
    game_type="mygame",                  # label for saved records
    game_config=my_config.model_dump(),  # saved alongside game records
)
```

`game_factory` is a callable that returns a new game instance. This is called for every self-play game and every arena game.

## Configuration

### MCTSConfig

Controls the MCTS search at self-play time.

```python
from pymcts.core.config import MCTSConfig

mcts_config = MCTSConfig(
    num_simulations=200,       # simulations per move (more = stronger)
    c_puct=1.5,                # exploration constant
    dirichlet_alpha=0.3,       # root noise for exploration
    dirichlet_epsilon=0.25,    # noise mixing ratio
    num_parallel_leaves=8,     # leaves per batch (GPU efficiency)
)
```

!!! tip "Simulations vs quality"
    - **25-50**: fast but weak, good for initial debugging
    - **200**: reasonable quality, good default
    - **800+**: strong play, slow training

### TrainingConfig

Controls the training loop.

```python
from pymcts.core.config import TrainingConfig

training_config = TrainingConfig(
    num_iterations=50,           # self-play → train → evaluate cycles
    num_self_play_games=100,     # games per iteration
    self_play_batch_size=16,     # concurrent self-play games
    batch_size=64,               # training batch size
    num_epochs=10,               # epochs per training step
    learning_rate=0.001,
    weight_decay=1e-4,
    temperature=1.0,             # exploration temperature for self-play
    temp_threshold=15,           # switch to greedy after this many moves
)
```

### ArenaConfig

Controls model comparison.

```python
from pymcts.core.config import ArenaConfig

arena_config = ArenaConfig(
    num_games=40,       # games to play per evaluation
    threshold=0.55,     # win rate needed to accept new model
    swap_players=True,  # play both sides for fairness
)
```

## Checkpoints

Training saves to `trainings/run_<timestamp>/` by default:

```
trainings/run_2026-03-24_143000/
├── config.json                    # all configs
├── iteration_001/
│   ├── pre_training.pt            # weights before training
│   ├── post_training.pt           # weights after training
│   ├── self_play_games.json       # games played
│   ├── eval_games.json            # arena games
│   └── iteration_data.json        # losses, win rates, etc.
├── iteration_002/
│   └── ...
└── best.pt                        # best model so far
```

### Resuming training

Load the checkpoint and call `train()` again:

```python
net = MyNet()
net.load_checkpoint("trainings/run_.../iteration_010/post_training.pt")

train(
    game_factory=lambda: MyGame(),
    net=net,
    mcts_config=mcts_config,
    training_config=TrainingConfig(num_iterations=20),  # 20 more
    arena_config=arena_config,
    game_type="mygame",
)
```

## Monitoring

Each iteration logs:

- **Self-play**: number of games, average game length
- **Training**: policy loss, value loss, total loss per epoch
- **Arena**: win rate, games played, accepted/rejected

The `iteration_data.json` files contain this data for analysis. See `notebooks/analysis.ipynb` for a plotting example.

## Hyperparameter guidance

### Starting point

For a new game, start with conservative settings and increase:

```python
MCTSConfig(num_simulations=50)
TrainingConfig(num_iterations=10, num_self_play_games=20, num_epochs=5)
ArenaConfig(num_games=20, threshold=0.55)
```

This runs fast and lets you verify the pipeline works.

### Scaling up

Once you see improvement:

- Increase `num_simulations` (200-800)
- Increase `num_self_play_games` (100-500)
- Increase `num_iterations` (50-200)
- Use `num_parallel_leaves` for GPU efficiency (4-16)

### Common issues

| Symptom | Likely cause | Fix |
|---|---|---|
| Training loss doesn't decrease | Learning rate too low, or too few examples | Increase `num_self_play_games` or `learning_rate` |
| Arena never accepts | Threshold too high, or not enough training | Lower `threshold` to 0.52, increase `num_epochs` |
| Arena always accepts | Threshold too low | Raise `threshold` to 0.55-0.60 |
| Training is slow | Too many MCTS simulations | Reduce `num_simulations`, increase `num_parallel_leaves` |
| Model doesn't improve after many iterations | Network too small, or too few simulations | Increase network size or `num_simulations` |

## Next step

Learn how to [evaluate and compare models](evaluation.md).
