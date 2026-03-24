"""ResNet-based neural network for Bridgit implementing BaseNeuralNet."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pymcts.core.base_neural_net import BaseNeuralNet
from pymcts.games.bridgit.config import BoardConfig, NeuralNetConfig
from pymcts.games.bridgit.game import BridgitGameState


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


class BridgitNet(BaseNeuralNet):
    """ResNet with dual policy/value heads for Bridgit.

    Implements BaseNeuralNet: encode() builds a (4, g, g) tensor from
    BridgitGameState, and forward() returns flattened (batch, g*g) policy
    and (batch, 1) value.
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

        # Initial convolution: 4 input channels
        self.conv_init = nn.Conv2d(4, ch, 3, padding=1, bias=False)
        self.bn_init = nn.BatchNorm2d(ch)

        # Residual tower
        self.res_blocks = nn.Sequential(
            *[ResBlock(ch) for _ in range(net_config.num_res_blocks)]
        )

        # Policy head: conv down to 1 channel -> flatten to g*g -> log_softmax
        self.policy_conv = nn.Conv2d(ch, 1, 1)

        # Value head
        self.value_conv = nn.Conv2d(ch, 1, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(g * g, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def encode(self, state: BridgitGameState) -> torch.Tensor:
        """Convert BridgitGameState to (4, g, g) tensor.

        Channel 0: current player's cells (HORIZONTAL = -1 in canonical form)
        Channel 1: opponent's cells (VERTICAL = 1 in canonical form)
        Channel 2: playability mask (empty interior crossings)
        Channel 3: moves_left constant plane
        """
        board = state.board
        g = board.shape[0]

        # Channel 0: current player (HORIZONTAL = -1)
        ch0 = (board == -1).astype(np.float32)
        # Channel 1: opponent (VERTICAL = 1)
        ch1 = (board == 1).astype(np.float32)

        # Channel 2: playability mask — empty interior crossings
        ch2 = np.zeros((g, g), dtype=np.float32)
        for r in range(1, g - 1):
            for c in range(1, g - 1):
                if (r + c) % 2 == 0 and board[r, c] == 0:
                    ch2[r, c] = 1.0

        # Channel 3: moves left constant plane
        ch3 = np.full((g, g), state.moves_left_in_turn, dtype=np.float32)

        tensor = np.stack([ch0, ch1, ch2, ch3], axis=0)
        return torch.from_numpy(tensor)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass. Returns (log_policy, value).

        log_policy: (batch, g*g) flattened log-softmax
        value: (batch, 1) tanh-scaled evaluation
        """
        out = F.relu(self.bn_init(self.conv_init(x)))
        out = self.res_blocks(out)

        # Policy head -> (batch, 1, g, g) -> (batch, g*g) -> log_softmax
        p = self.policy_conv(out).squeeze(1)  # (batch, g, g)
        p = F.log_softmax(p.flatten(1), dim=1)  # (batch, g*g)

        # Value head
        v = F.relu(self.value_bn(self.value_conv(out)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))

        return p, v

    def save_checkpoint(self, path: str) -> None:
        """Save model weights and config to checkpoint file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "model_state_dict": self.state_dict(),
            "board_config": self.board_config.model_dump(),
            "net_config": self.net_config.model_dump(),
        }
        torch.save(checkpoint, p)

    def load_checkpoint(self, path: str) -> None:
        """Load model weights from checkpoint file."""
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        self.load_state_dict(checkpoint["model_state_dict"])

    def copy(self) -> "BridgitNet":
        """Return a deep copy of this network."""
        return copy.deepcopy(self)
