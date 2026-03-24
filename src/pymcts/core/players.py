"""Player abstractions for the AlphaZero engine."""

import random
from abc import ABC, abstractmethod

import torch

from pymcts.core.base_game import BaseGame
from pymcts.core.base_neural_net import BaseNeuralNet
from pymcts.core.mcts import MCTS
from pymcts.core.config import MCTSConfig


class BasePlayer(ABC):
    def __init__(self, name: str | None = None):
        self.name = name or self.__class__.__name__
        self._last_policy: torch.Tensor | None = None

    @abstractmethod
    def get_action(self, game: BaseGame) -> int: ...

    @property
    def last_policy(self) -> torch.Tensor | None:
        return self._last_policy

    def __repr__(self) -> str:
        return self.name


class RandomPlayer(BasePlayer):
    def get_action(self, game: BaseGame) -> int:
        self._last_policy = None
        return random.choice(game.valid_actions())


class MCTSPlayer(BasePlayer):
    def __init__(
        self,
        net: BaseNeuralNet,
        mcts_config: MCTSConfig,
        temperature: float = 1.0,
        temp_threshold: int = 0,
        name: str | None = None,
    ):
        super().__init__(name)
        self.mcts = MCTS(net, mcts_config)
        self.temperature = temperature
        self.temp_threshold = temp_threshold

    def get_action(self, game: BaseGame) -> int:
        # Use number of moves already played for temperature decay
        move_count = game.action_space_size - len(game.valid_actions())
        temp = self.temperature if move_count < self.temp_threshold else 0.0
        probs = self.mcts.get_action_probs(game, temperature=temp)
        self._last_policy = probs

        mask = game.to_mask()
        if probs.sum() == 0:
            probs = mask.float()

        flat_idx = torch.multinomial(probs, 1).item()
        return flat_idx


class GreedyMCTSPlayer(MCTSPlayer):
    def __init__(self, net: BaseNeuralNet, mcts_config: MCTSConfig, name: str | None = None):
        super().__init__(net, mcts_config, temperature=0.0, name=name)
