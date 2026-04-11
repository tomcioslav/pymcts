# Quick Start

Train a neural network to play Bridgit in under 20 lines of code.

## Train

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

This will:

1. Play 10 games of self-play using MCTS (50 simulations per move)
2. Train the neural network on the generated games
3. Compare the new model against the old one in an arena (10 games)
4. If the new model wins enough, keep it. Otherwise, revert.
5. Repeat for 3 iterations.

Checkpoints are saved to `trainings/` automatically.

## Watch a game

After training, pit two models against each other:

```python
from pymcts.arena import batched_arena
from pymcts.core.players import GreedyMCTSPlayer
from pymcts.core.config import MCTSConfig
from pymcts.games.bridgit.game import BridgitGame
from pymcts.games.bridgit.config import BoardConfig
from pymcts.games.bridgit.neural_net import BridgitNet

board_config = BoardConfig(size=5)
net = BridgitNet.from_checkpoint("trainings/run_.../iteration_.../post_training.pt")

player = GreedyMCTSPlayer(net=net, mcts_config=MCTSConfig(num_simulations=200), name="trained")
records = batched_arena(
    player_a=player,
    player_b=player,
    game_factory=lambda: BridgitGame(board_config),
    num_games=1,
    verbose=True,
)
print(records[0].summary())
```

## Play interactively (GUI)

```bash
python play.py       # default 5x5 board
python play.py 7     # 7x7 board
```

## What's next?

- [How It Works](../concepts/how-it-works.md) — understand the AlphaZero training loop
- [Creating a Game](../guide/creating-a-game.md) — add your own game to the engine
- [API Reference](../reference/core.md) — detailed API documentation
