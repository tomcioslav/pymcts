# Evaluation

This guide covers how to compare models, run arena matches, and analyze game records.

## Arena

The `batched_arena` function pits two players against each other over multiple games.

```python
from pymcts.core.arena import batched_arena
from pymcts.core.players import GreedyMCTSPlayer, RandomPlayer
from pymcts.core.config import MCTSConfig

# Load two models
net_v1 = MyNet.from_checkpoint("trainings/.../iteration_005/post_training.pt")
net_v2 = MyNet.from_checkpoint("trainings/.../iteration_010/post_training.pt")

# Create players
config = MCTSConfig(num_simulations=200)
player_v1 = GreedyMCTSPlayer(net=net_v1, mcts_config=config, name="v1")
player_v2 = GreedyMCTSPlayer(net=net_v2, mcts_config=config, name="v2")

# Run arena
collection = batched_arena(
    player_a=player_v1,
    player_b=player_v2,
    game_factory=lambda: MyGame(),
    num_games=40,
    swap_players=True,
)
print(collection.scores)
```

Setting `swap_players=True` ensures each model plays both sides — this eliminates first-player advantage.

## Loading players from training

The easiest way to load a trained model is from the `arena/` directory:

```python
from pymcts.core.players import MCTSPlayer

# Load a player saved during training
player = MCTSPlayer.load("trainings/run_.../arena/iteration_010")
print(f"Player: {player.name}, Elo: {player.elo}")
```

Or load from a training iteration directly:

```python
player = MCTSPlayer.from_training_iteration("trainings/run_.../iteration_010")
```

## Players

pymcts provides several player types:

### RandomPlayer

Picks a random legal action. Useful as a baseline.

```python
from pymcts.core.players import RandomPlayer
random_player = RandomPlayer(name="random")
```

### MCTSPlayer

Uses MCTS + neural net to select moves. Temperature controls exploration:

```python
from pymcts.core.players import MCTSPlayer

# Exploratory (for self-play training)
player = MCTSPlayer(net=net, mcts_config=config, name="explorer", temperature=1.0)

# Greedy (for evaluation)
player = MCTSPlayer(net=net, mcts_config=config, name="greedy", temperature=0.0)
```

### GreedyMCTSPlayer

Shorthand for `MCTSPlayer` with temperature=0:

```python
from pymcts.core.players import GreedyMCTSPlayer
player = GreedyMCTSPlayer(net=net, mcts_config=config, name="greedy")
```

### Player serialization

All players can be saved and loaded:

```python
# Save
player.save("my_player/")

# Load
loaded = MCTSPlayer.load("my_player/")
print(loaded.elo)  # Elo is preserved if set
```

`RandomPlayer` also supports save/load (no model weights, just config).

## Game records

Every game played through `batched_arena` or self-play is recorded as a `GameRecord`:

```python
record = collection[0]
print(record.summary())
print(f"Winner: {record.winner_name()}")
print(f"Moves: {record.num_moves}")
```

### GameRecordCollection

Multiple records are grouped in a `GameRecordCollection`:

```python
# Win/loss counts
print(collection.scores)  # {"v1": 12, "v2": 8}

# Detailed evaluation for a specific player
result = collection.evaluate("v1")
print(f"Win rate: {result.win_rate:.1%}")
print(f"Avg moves in wins: {result.avg_moves_in_wins:.1f}")
```

### Saving and loading

Game records use Pydantic, so they serialize to JSON:

```python
# Save
with open("games.json", "w") as f:
    f.write(collection.model_dump_json(indent=2))

# Load
from pymcts.core.game_record import GameRecordCollection
collection = GameRecordCollection.model_validate_json(open("games.json").read())
```

## Batched self-play

For generating training data efficiently, use batched self-play:

```python
from pymcts.core.self_play import batched_self_play

collection = batched_self_play(
    net=net,
    game_factory=lambda: MyGame(),
    mcts_config=MCTSConfig(num_simulations=200, num_parallel_leaves=8),
    num_games=100,
    batch_size=16,    # 16 games concurrent
    temperature=1.0,
)
```

This runs `batch_size` games concurrently, batching all neural net evaluations into single GPU calls. With `num_parallel_leaves=K`, each simulation step evaluates K leaves per game — giving K * batch_size evaluations per GPU call.

## Visualization (Bridgit)

Bridgit includes a Plotly-based visualizer for game records:

```python
from pymcts.games.bridgit.visualizer import Visualizer

# Interactive game replay with move slider
Visualizer.visualize_game(record)

# Save as standalone HTML
Visualizer.save_game_html(record, "game.html")

# Single board state
Visualizer.visualize_game_state(game.get_state())

# Raw array heatmap (for policies, visit counts, etc.)
Visualizer.visualize_array(tensor.reshape(g, g), colorscale="Blues")
```

For your own game, you can build a similar visualizer — it just needs to read `GameRecord` and display states.
