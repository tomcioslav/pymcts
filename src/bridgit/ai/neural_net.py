"""ResNet-based neural network for Bridgit (policy + value heads).

Input:  Bridgit.to_tensor() → shape (4, g, g) where g = 2n+1
  - Channel 0: current player's edges
  - Channel 1: opponent's edges
  - Channel 2: playability mask (1 at edge positions)
  - Channel 3: moves left in turn (1 or 2, constant plane)
Output: (log_policy, value)
  - log_policy: shape (g, g) — log probabilities over board positions
  - value: shape (1,) — position evaluation in [-1, 1]
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from bridgit.data.converter import examples_from_records
from bridgit.game import Bridgit
from bridgit.schema import Move, GameRecordCollection
from bridgit.config import BoardConfig, NeuralNetConfig, TrainingConfig


class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual)


class BridgitNet(nn.Module):
    """ResNet with dual policy/value heads for Bridgit.

    Takes a (batch, 3, g, g) tensor where g = 2*board_size - 1.
    Returns (log_policy, value):
      - log_policy: (batch, g, g) log-softmax over board positions
      - value: (batch, 1) tanh-scaled position evaluation
    """

    def __init__(
        self,
        board_config: BoardConfig = BoardConfig(),
        net_config: NeuralNetConfig = NeuralNetConfig(),
    ):
        super().__init__()
        self.board_config = board_config
        self.net_config = net_config
        g = board_config.grid_size
        ch = net_config.num_channels

        # Initial convolution: 4 input channels (mine, theirs, playable, moves_left)
        self.conv_init = nn.Conv2d(4, ch, 3, padding=1, bias=False)
        self.bn_init = nn.BatchNorm2d(ch)

        # Residual tower
        self.res_blocks = nn.Sequential(*[ResBlock(ch) for _ in range(net_config.num_res_blocks)])

        # Policy head: conv down to 1 channel → (batch, g, g)
        self.policy_conv = nn.Conv2d(ch, 1, 1)

        # Value head
        self.value_conv = nn.Conv2d(ch, 1, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(g * g, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (batch, 3, g, g) where g = 2n-1
        out = F.relu(self.bn_init(self.conv_init(x)))
        out = self.res_blocks(out)

        # Policy head → (batch, g, g)
        p = self.policy_conv(out).squeeze(1)
        p = F.log_softmax(p.flatten(1), dim=1).view_as(p)

        # Value head
        v = F.relu(self.value_bn(self.value_conv(out)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))

        return p, v


class NetWrapper:
    """Thin wrapper around BridgitNet: device handling, checkpoints, and inference."""

    def __init__(
        self,
        model_or_path: BridgitNet | str | Path | None = None,
    ):
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        if isinstance(model_or_path, BridgitNet):
            model = model_or_path
        elif model_or_path is not None:
            checkpoint = torch.load(Path(model_or_path), map_location="cpu", weights_only=False)
            board_config = BoardConfig(**checkpoint["board_config"])
            net_config = NeuralNetConfig(**checkpoint["net_config"])
            model = BridgitNet(board_config, net_config)
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model = BridgitNet()

        self.model = model.to(self.device)

    def save_checkpoint(self, path: str | Path) -> None:
        """Save model and config to a checkpoint file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "board_config": self.model.board_config.model_dump(),
            "net_config": self.model.net_config.model_dump(),
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str | Path) -> None:
        """Load model weights from a checkpoint file.

        Raises ValueError if the checkpoint's board or net config
        doesn't match this model's architecture.
        """
        path = Path(path)
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        ckpt_board = checkpoint["board_config"]
        ckpt_net = checkpoint["net_config"]
        cur_board = self.model.board_config.model_dump()
        cur_net = self.model.net_config.model_dump()

        if ckpt_board != cur_board:
            raise ValueError(
                f"Board config mismatch: checkpoint has {ckpt_board}, "
                f"model expects {cur_board}"
            )
        if ckpt_net != cur_net:
            raise ValueError(
                f"Neural net config mismatch: checkpoint has {ckpt_net}, "
                f"model expects {cur_net}"
            )

        self.model.load_state_dict(checkpoint["model_state_dict"])

    def train_on_examples(
        self,
        data: GameRecordCollection | list[tuple[torch.Tensor, torch.Tensor, float]],
        training_config: TrainingConfig = TrainingConfig(),
    ) -> list[dict[str, float]]:
        """Train the model on game records or pre-converted examples.

        Returns a list of per-epoch loss dicts with keys:
        policy_loss, value_loss, total_loss.
        """
        if isinstance(data, GameRecordCollection):
            examples = examples_from_records(data)
        else:
            examples = data
        states = torch.stack([s for s, _, _ in examples])
        policies = torch.stack([p for _, p, _ in examples])
        values = torch.tensor([v for _, _, v in examples], dtype=torch.float32).unsqueeze(1)

        dataset = TensorDataset(states, policies, values)
        loader = DataLoader(dataset, batch_size=training_config.batch_size, shuffle=True)

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=training_config.learning_rate,
            weight_decay=training_config.weight_decay,
        )

        self.model.train()
        history: list[dict[str, float]] = []
        epoch_iter = tqdm(range(training_config.num_epochs), desc="Training", leave=False)
        for epoch in epoch_iter:
            total_policy_loss = 0.0
            total_value_loss = 0.0
            num_batches = 0

            for batch_states, batch_policies, batch_values in loader:
                batch_states = batch_states.to(self.device)
                batch_policies = batch_policies.to(self.device)
                batch_values = batch_values.to(self.device)

                log_policy, value = self.model(batch_states)

                # Policy loss: cross-entropy with MCTS target distribution
                policy_loss = -torch.sum(batch_policies * log_policy) / batch_states.size(0)
                # Value loss: MSE
                value_loss = F.mse_loss(value, batch_values)
                loss = policy_loss + value_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                num_batches += 1

            avg_pi = total_policy_loss / num_batches
            avg_v = total_value_loss / num_batches
            history.append({
                "policy_loss": avg_pi,
                "value_loss": avg_v,
                "total_loss": avg_pi + avg_v,
            })
            epoch_iter.set_postfix_str(
                f"pi={avg_pi:.4f}, v={avg_v:.4f}"
            )

        first = history[0]
        last = history[-1]
        print(f"  Loss: pi={last['policy_loss']:.4f}, v={last['value_loss']:.4f} "
              f"(from pi={first['policy_loss']:.4f}, v={first['value_loss']:.4f})")

        return history

    def make_move(self, game: Bridgit) -> Move:
        """Pick the best legal move."""
        self.model.eval()
        tensor = game.to_tensor().unsqueeze(0).to(self.device)  # (1, 4, g, g)
        mask = game.to_mask()  # (g, g)

        with torch.no_grad():
            log_policy, _ = self.model(tensor)

        # Mask illegal moves and pick the best one
        log_policy = log_policy[0].cpu()  # (g, g)
        log_policy = log_policy.masked_fill(mask == 0, float("-inf"))
        best = torch.argmax(log_policy)
        row = best // log_policy.shape[1]
        col = best % log_policy.shape[1]
        return Move(row=int(row), col=int(col))

