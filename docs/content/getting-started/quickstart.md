# Quick Start

Train a neural network to play Bridgit in under 20 lines of code.

## Train

```python
from pymcts.games.bridgit.game import BridgitGame
from pymcts.games.bridgit.config import BoardConfig, NeuralNetConfig
from pymcts.games.bridgit.neural_net import BridgitNet
from pymcts.core.config import MCTSConfig, TrainingConfig, ArenaConfig
from pymcts.core.trainer import train

board_config = BoardConfig(size=5)
net = BridgitNet(board_config=board_config, net_config=NeuralNetConfig())

train(
    game_factory=lambda: BridgitGame(board_config),
    net=net,
    mcts_config=MCTSConfig(num_simulations=50),
    training_config=TrainingConfig(num_iterations=3, num_self_play_games=10),
    arena_config=ArenaConfig(num_games=10),
    game_type="bridgit",
    game_config=board_config.model_dump(),
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
from pymcts.core.arena import Arena
from pymcts.core.players import MCTSPlayer
from pymcts.core.config import MCTSConfig
from pymcts.games.bridgit.game import BridgitGame
from pymcts.games.bridgit.config import BoardConfig
from pymcts.games.bridgit.neural_net import BridgitNet

board_config = BoardConfig(size=5)
net = BridgitNet(board_config)
net.load_checkpoint("trainings/run_.../iteration_.../post_training.pt")

player = MCTSPlayer(net=net, mcts_config=MCTSConfig(num_simulations=200), name="trained")
arena = Arena(player, player, game_factory=lambda: BridgitGame(board_config))

record = arena.play_game(verbose=True)
print(record.summary())
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
