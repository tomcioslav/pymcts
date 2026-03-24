"""Abstract base class for neural networks in the AlphaZero engine."""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from pymcts.core.base_game import GameState


class BaseNeuralNet(ABC, nn.Module):
    """Base class for all neural networks. Inherits from ABC and nn.Module.

    The developer implements encode() and forward(). predict(), predict_batch(),
    and train_on_examples() have sensible defaults.
    """

    @abstractmethod
    def encode(self, state: GameState) -> torch.Tensor:
        """Convert a GameState into the tensor format this architecture expects."""

    @abstractmethod
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Raw forward pass. Input: (batch, *encoded_shape).
        Returns: (log_policy (batch, action_space_size), value (batch, 1)).
        """

    @abstractmethod
    def save_checkpoint(self, path: str) -> None: ...

    @abstractmethod
    def load_checkpoint(self, path: str) -> None: ...

    @abstractmethod
    def copy(self) -> "BaseNeuralNet": ...

    def predict(self, state: GameState) -> tuple[torch.Tensor, float]:
        """Single state -> (policy_1D, value)."""
        self.eval()
        with torch.no_grad():
            tensor = self.encode(state).unsqueeze(0)
            policy, value = self.forward(tensor)
        return policy.squeeze(0), value.item()

    def predict_batch(self, states: list[GameState]) -> tuple[torch.Tensor, torch.Tensor]:
        """Batch of states -> (policies, values)."""
        self.eval()
        with torch.no_grad():
            tensors = torch.stack([self.encode(s) for s in states])
            policies, values = self.forward(tensors)
        return policies, values.squeeze(-1)

    def train_on_examples(
        self,
        examples: list[tuple[GameState, torch.Tensor, float]],
        num_epochs: int = 20,
        batch_size: int = 64,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-4,
    ) -> dict[str, float]:
        """Default training loop: cross-entropy policy loss + MSE value loss.

        Returns metrics dict with final epoch losses.
        """
        states = torch.stack([self.encode(s) for s, _, _ in examples])
        policies = torch.stack([p for _, p, _ in examples])
        values = torch.tensor([v for _, _, v in examples], dtype=torch.float32).unsqueeze(1)

        dataset = TensorDataset(states, policies, values)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.Adam(
            self.parameters(), lr=learning_rate, weight_decay=weight_decay,
        )

        self.train()
        total_policy_loss = 0.0
        total_value_loss = 0.0
        num_batches = 0

        for epoch in range(num_epochs):
            total_policy_loss = 0.0
            total_value_loss = 0.0
            num_batches = 0

            for batch_states, batch_policies, batch_values in loader:
                log_policy, value = self.forward(batch_states)
                policy_loss = -torch.sum(batch_policies * log_policy) / batch_states.size(0)
                value_loss = F.mse_loss(value, batch_values)
                loss = policy_loss + value_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                num_batches += 1

        avg_pi = total_policy_loss / max(num_batches, 1)
        avg_v = total_value_loss / max(num_batches, 1)
        return {
            "policy_loss": avg_pi,
            "value_loss": avg_v,
            "loss": avg_pi + avg_v,
        }
