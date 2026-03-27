# Creating a Neural Net

This guide walks through implementing a neural network for pymcts, continuing the Tic-tac-toe example from [Creating a Game](creating-a-game.md).

## What you need to implement

Inherit from `BaseNeuralNet` and implement:

1. **`encode(state)`** — convert your GameState to a tensor
2. **`forward(x)`** — the neural network architecture
3. **`save_checkpoint(path)`** / **`load_checkpoint(path)`** — serialization
4. **`copy()`** — deep copy for arena comparison

`predict()`, `predict_batch()`, and `train_on_examples()` work automatically.

## Step 1: Encode the game state

`encode()` converts your `GameState` subclass into whatever tensor format your architecture needs. This is the bridge between the game and the network.

```python
import torch
import torch.nn as nn
from pymcts.core.base_neural_net import BaseNeuralNet

class TicTacToeNet(BaseNeuralNet):

    def __init__(self):
        super().__init__()
        # We'll define layers in Step 2

    def encode(self, state: TicTacToeState) -> torch.Tensor:
        """Encode board as 2 channels: current player's marks, opponent's marks."""
        board = state.board  # 3x3, +1=current, -1=opponent, 0=empty
        current = (board == 1).astype(np.float32)
        opponent = (board == -1).astype(np.float32)
        return torch.tensor(np.stack([current, opponent]))  # shape: (2, 3, 3)
```

!!! tip "Encoding design matters"
    The encoding is where you decide what information the network sees. Common choices:

    - **Binary channels**: one channel per player (like above)
    - **Single channel**: raw board values
    - **Extra channels**: valid moves, move count, etc.

    Experiment — different encodings can significantly affect training.

## Step 2: Define the architecture

`forward()` takes a batch of encoded tensors and returns `(log_policy, value)`.

For Tic-tac-toe, a simple MLP is sufficient:

```python
class TicTacToeNet(BaseNeuralNet):

    def __init__(self):
        super().__init__()
        # Input: 2 channels * 3 * 3 = 18 features
        self.shared = nn.Sequential(
            nn.Linear(18, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(64, 9)   # 9 actions
        self.value_head = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, 2, 3, 3) encoded board states

        Returns:
            log_policy: (batch, 9) log-probabilities over actions
            value: (batch, 1) position evaluation in [-1, 1]
        """
        batch = x.shape[0]
        flat = x.view(batch, -1)             # (batch, 18)
        shared = self.shared(flat)            # (batch, 64)
        log_policy = self.policy_head(shared) # (batch, 9)
        log_policy = torch.log_softmax(log_policy, dim=1)
        value = torch.tanh(self.value_head(shared))  # (batch, 1)
        return log_policy, value
```

!!! note "Output requirements"
    - **Policy**: `(batch, action_space_size)` — must be **log-probabilities** (use `log_softmax`)
    - **Value**: `(batch, 1)` — must be in **[-1, 1]** (use `tanh`)

### For larger games: use a ResNet

For games with larger boards (like Bridgit), convolutional residual networks work better:

```python
class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return torch.relu(x + residual)
```

See `pymcts.games.bridgit.neural_net.BridgitNet` for a full ResNet implementation.

## Step 3: Checkpointing and copy

```python
    def save_checkpoint(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    def load_checkpoint(self, path: str) -> None:
        self.load_state_dict(torch.load(path, weights_only=True))

    def copy(self) -> "TicTacToeNet":
        new = TicTacToeNet()
        new.load_state_dict(self.state_dict())
        return new
```

## Step 4: Test it

```python
# Create game and net
game = TicTacToeGame()
net = TicTacToeNet()

# Single prediction
state = game.get_state()
policy, value = net.predict(state)
print(f"Policy shape: {policy.shape}")  # (9,)
print(f"Value: {value:.3f}")            # random initially

# Batch prediction
states = [game.get_state() for _ in range(4)]
policies, values = net.predict_batch(states)
print(f"Batch policy shape: {policies.shape}")  # (4, 9)
print(f"Batch values shape: {values.shape}")     # (4,)
```

## Step 5: Train

```python
from pymcts.core.trainer import train
from pymcts.core.config import MCTSConfig, TrainingConfig, ArenaConfig

train(
    game_factory=TicTacToeGame,
    net=TicTacToeNet(),
    mcts_config=MCTSConfig(num_simulations=25),
    training_config=TrainingConfig(num_iterations=5, num_self_play_games=20),
    arena_config=ArenaConfig(num_games=20),
    game_type="tictactoe",
)
```

Tic-tac-toe is simple enough that the network should learn to play perfectly within a few iterations.

## Device management

`BaseNeuralNet` inherits from `nn.Module`, so standard PyTorch device management works:

```python
net = TicTacToeNet()
net = net.to("cuda")  # or "mps" for Apple Silicon
```

The `encode()` method should place tensors on the same device as the model:

```python
def encode(self, state: TicTacToeState) -> torch.Tensor:
    board = state.board
    current = (board == 1).astype(np.float32)
    opponent = (board == -1).astype(np.float32)
    tensor = torch.tensor(np.stack([current, opponent]))
    return tensor.to(next(self.parameters()).device)
```

## Architecture tips

| Game complexity | Recommended architecture |
|---|---|
| Very small (Tic-tac-toe) | MLP (2-3 layers) |
| Small board (Connect 4, small Bridgit) | Shallow ResNet (2-4 blocks) |
| Medium board (Go 9x9, Bridgit 7x7) | ResNet (4-8 blocks) |
| Large board (Go 19x19) | Deep ResNet or Transformer |

Key hyperparameters to tune:

- **Number of channels** in the ResNet (16-256)
- **Number of residual blocks** (2-20)
- **Encoding channels** — more information helps, but adds parameters

## Complete code

??? example "Full TicTacToeNet implementation (click to expand)"

    ```python
    import numpy as np
    import torch
    import torch.nn as nn
    from pymcts.core.base_neural_net import BaseNeuralNet


    class TicTacToeNet(BaseNeuralNet):

        def __init__(self):
            super().__init__()
            self.shared = nn.Sequential(
                nn.Linear(18, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
            )
            self.policy_head = nn.Linear(64, 9)
            self.value_head = nn.Linear(64, 1)

        def encode(self, state) -> torch.Tensor:
            board = state.board
            current = (board == 1).astype(np.float32)
            opponent = (board == -1).astype(np.float32)
            tensor = torch.tensor(np.stack([current, opponent]))
            return tensor.to(next(self.parameters()).device)

        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            batch = x.shape[0]
            flat = x.view(batch, -1)
            shared = self.shared(flat)
            log_policy = torch.log_softmax(self.policy_head(shared), dim=1)
            value = torch.tanh(self.value_head(shared))
            return log_policy, value

        def save_checkpoint(self, path: str) -> None:
            torch.save(self.state_dict(), path)

        def load_checkpoint(self, path: str) -> None:
            self.load_state_dict(torch.load(path, weights_only=True))

        def copy(self) -> "TicTacToeNet":
            new = TicTacToeNet()
            new.load_state_dict(self.state_dict())
            return new
    ```

## Next step

Now learn how to [configure and run training](training.md).
