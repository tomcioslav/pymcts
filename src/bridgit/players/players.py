"""Player abstractions for different playing strategies."""

from abc import ABC, abstractmethod

import numpy as np
import torch

from bridgit.ai.mcts import MCTS
from bridgit.ai.neural_net import NetWrapper
from bridgit.game import Bridgit
from bridgit.schema import Move, Player
from bridgit.config import MCTSConfig


class BasePlayer(ABC):
    """Abstract base class for all players."""

    def __init__(self, name: str | None = None):
        self.name = name or self.__class__.__name__

    @abstractmethod
    def get_action(self, game: Bridgit) -> Move:
        """Select a move (row, col) for the current game state."""
        pass

    @property
    def last_policy(self) -> torch.Tensor | None:
        """MCTS policy from the last get_action call, if available."""
        return None

    def __repr__(self) -> str:
        return self.name


class RandomPlayer(BasePlayer):
    """Player that selects moves uniformly at random."""

    def __init__(self, name: str | None = None):
        super().__init__(name)

    def get_action(self, game: Bridgit) -> Move:
        """Select a random legal move."""
        moves = game.get_available_moves()
        if not moves:
            raise ValueError("No valid moves available")
        idx = np.random.randint(len(moves))
        return moves[idx]


class MCTSPlayer(BasePlayer):
    """Player that uses MCTS with neural network guidance."""

    def __init__(
        self,
        net_wrapper: NetWrapper,
        mcts_config: MCTSConfig,
        temperature: float = 1.0,
        temp_threshold: int = 0,
        name: str | None = None
    ):
        super().__init__(name)
        self.mcts_search = MCTS(net_wrapper, mcts_config)
        self.temperature = temperature
        self.temp_threshold = temp_threshold
        self._last_policy: torch.Tensor | None = None

    @property
    def last_policy(self) -> torch.Tensor | None:
        return self._last_policy

    def get_action(self, game: Bridgit) -> Move:
        """Select move using MCTS."""
        temp = self.temperature if game.move_count < self.temp_threshold else 0.0
        probs = self.mcts_search.get_action_probs(game, temperature=temp)
        self._last_policy = probs  # already in canonical space

        # Sample move from probability distribution
        if probs.sum() == 0:
            probs = game.to_mask()
        flat_idx = torch.multinomial(probs.flatten(), 1).item()
        row, col = divmod(flat_idx, probs.shape[1])
        canonical_move = Move(row=row, col=col)
        return canonical_move.decanonicalize(game.current_player)


class GreedyMCTSPlayer(MCTSPlayer):
    """MCTS player that always picks the most-visited move (temperature=0)."""

    def __init__(self, net_wrapper: NetWrapper, mcts_config: MCTSConfig, name: str | None = None):
        super().__init__(net_wrapper, mcts_config, temperature=0.0, name=name)


