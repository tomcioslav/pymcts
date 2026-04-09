"""Abstract base class for neural networks in the AlphaZero engine."""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from pymcts.core.base_game import GameState


def _best_device() -> torch.device:
    """Return the best available device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class BaseNeuralNet(ABC, nn.Module):
    """Base class for all neural networks. Inherits from ABC and nn.Module.

    The developer implements encode() and forward(). predict(), predict_batch(),
    and train_on_examples() have sensible defaults.

    Automatically moves to the best available device (CUDA > MPS > CPU)
    after construction.
    """

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        original_init = cls.__init__

        def _wrapped_init(self, *args, **kw):
            original_init(self, *args, **kw)
            device = getattr(self, '_force_device', None) or _best_device()
            self.to(device)

        cls.__init__ = _wrapped_init

    @abstractmethod
    def encode(self, state: GameState) -> torch.Tensor:
        """Convert a GameState into the tensor format this architecture expects.
        Should return a CPU tensor — the base class handles device transfer.
        """

    def encode_batch(self, states: list[GameState]) -> torch.Tensor:
        """Encode multiple states into a single batched tensor.
        Override in subclasses for more efficient batch encoding.
        """
        return torch.stack([self.encode(s) for s in states])

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

    @property
    def device(self) -> torch.device:
        """Return the device this model's parameters live on."""
        return next(self.parameters()).device

    def to_best_device(self) -> "BaseNeuralNet":
        """Move this model to the best available device."""
        return self.to(_best_device())

    def predict(self, state: GameState) -> tuple[torch.Tensor, float]:
        """Single state -> (policy_1D, value). Returns CPU tensors."""
        self.eval()
        with torch.no_grad():
            tensor = self.encode(state).unsqueeze(0).to(self.device)
            policy, value = self.forward(tensor)
        return policy.squeeze(0).cpu(), value.item()

    def predict_batch(self, states: list[GameState]) -> tuple[torch.Tensor, torch.Tensor]:
        """Batch of states -> (policies, values). Returns CPU tensors."""
        self.eval()
        with torch.no_grad():
            tensors = self.encode_batch(states).to(self.device)
            policies, values = self.forward(tensors)
        return policies.cpu(), values.squeeze(-1).cpu()

    def _prepare_dataset(
        self,
        examples: list[tuple[GameState, torch.Tensor, float]],
        batch_size: int,
    ) -> DataLoader:
        """Encode examples into tensors and wrap them in a shuffled DataLoader."""
        states = self.encode_batch([s for s, _, _ in examples])
        policies = torch.stack([p for _, p, _ in examples])
        values = torch.tensor([v for _, _, v in examples], dtype=torch.float32).unsqueeze(1)
        dataset = TensorDataset(states, policies, values)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)

    def _make_optimizer(self, learning_rate: float, weight_decay: float) -> torch.optim.Optimizer:
        """Create the Adam optimizer for this model's parameters."""
        return torch.optim.Adam(
            self.parameters(), lr=learning_rate, weight_decay=weight_decay,
        )

    def _compute_losses(
        self,
        batch_states: torch.Tensor,
        batch_policies: torch.Tensor,
        batch_values: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run a forward pass and return (policy_loss, value_loss) for the batch."""
        log_policy, value = self.forward(batch_states)
        policy_loss = -torch.sum(batch_policies * log_policy) / batch_states.size(0)
        value_loss = F.mse_loss(value, batch_values)
        return policy_loss, value_loss

    def _train_single_epoch(
        self, loader: DataLoader, optimizer: torch.optim.Optimizer,
    ) -> tuple[float, float]:
        """Train for one epoch; return (total_policy_loss, total_value_loss) summed over batches."""
        device = self.device
        total_policy_loss = 0.0
        total_value_loss = 0.0
        for batch_states, batch_policies, batch_values in loader:
            batch_states = batch_states.to(device)
            batch_policies = batch_policies.to(device)
            batch_values = batch_values.to(device)
            policy_loss, value_loss = self._compute_losses(
                batch_states, batch_policies, batch_values,
            )
            optimizer.zero_grad()
            (policy_loss + value_loss).backward()
            optimizer.step()
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
        return total_policy_loss, total_value_loss

    def _make_epoch_iter(self, num_epochs: int, verbose: bool):
        """Return an iterable over epoch indices, optionally wrapped with tqdm."""
        epoch_iter = range(num_epochs)
        if verbose:
            return tqdm(epoch_iter, desc="Training", leave=True)
        return epoch_iter

    def _update_verbose_postfix(self, epoch_iter, avg_pi: float, avg_v: float) -> None:
        """Update tqdm postfix with current average losses if epoch_iter supports it."""
        if hasattr(epoch_iter, 'set_postfix'):
            epoch_iter.set_postfix(
                pi_loss=f"{avg_pi:.4f}",
                v_loss=f"{avg_v:.4f}",
                loss=f"{avg_pi + avg_v:.4f}",
            )

    def train_on_examples(
        self,
        examples: list[tuple[GameState, torch.Tensor, float]],
        num_epochs: int = 20,
        batch_size: int = 64,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-4,
        verbose: bool = False,
    ) -> dict[str, float]:
        """Default training loop: cross-entropy policy loss + MSE value loss.

        Returns metrics dict with final epoch losses.
        """
        loader = self._prepare_dataset(examples, batch_size)
        optimizer = self._make_optimizer(learning_rate, weight_decay)
        epoch_iter = self._make_epoch_iter(num_epochs, verbose)
        self.train()
        avg_pi, avg_v = 0.0, 0.0
        for _ in epoch_iter:
            total_pi, total_v = self._train_single_epoch(loader, optimizer)
            num_batches = max(len(loader), 1)
            avg_pi = total_pi / num_batches
            avg_v = total_v / num_batches
            self._update_verbose_postfix(epoch_iter, avg_pi, avg_v)
        return {"policy_loss": avg_pi, "value_loss": avg_v, "loss": avg_pi + avg_v}
