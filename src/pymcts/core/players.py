"""Player abstractions for the AlphaZero engine."""

import importlib
import json
import random
from abc import ABC, abstractmethod
from pathlib import Path

import torch

from pymcts.core.base_game import BaseGame
from pymcts.core.base_neural_net import BaseNeuralNet
from pymcts.core.mcts import MCTS
from pymcts.core.config import MCTSConfig


def _import_class(fully_qualified_name: str) -> type:
    """Import a class from its fully qualified name (e.g. 'pymcts.games.bridgit.neural_net.BridgitNet')."""
    module_path, class_name = fully_qualified_name.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


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
        elo: float | None = None,
    ):
        super().__init__(name)
        self.elo = elo
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


    def save(self, path: str | Path) -> None:
        """Save player to a directory (config + neural net weights)."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        net = self.mcts.net
        net.save_checkpoint(str(path / "model.pt"))

        net_class = type(net)
        config = {
            "net_class": f"{net_class.__module__}.{net_class.__qualname__}",
            "mcts_config": self.mcts.mcts_config.model_dump(),
            "temperature": self.temperature,
            "temp_threshold": self.temp_threshold,
            "name": self.name,
            "elo": self.elo,
        }
        (path / "player.json").write_text(json.dumps(config, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "MCTSPlayer":
        """Load a player from a directory saved with .save()."""
        path = Path(path)
        config = json.loads((path / "player.json").read_text())

        net_class = _import_class(config["net_class"])
        net = net_class.from_checkpoint(str(path / "model.pt"))
        mcts_config = MCTSConfig(**config["mcts_config"])

        return cls(
            net=net,
            mcts_config=mcts_config,
            temperature=config["temperature"],
            temp_threshold=config["temp_threshold"],
            name=config["name"],
            elo=config.get("elo"),
        )

    @classmethod
    def from_training_iteration(
        cls,
        iter_path: str | Path,
        mcts_config: MCTSConfig | None = None,
        temperature: float = 0.0,
        name: str | None = None,
    ) -> "MCTSPlayer":
        """Load a player from a training iteration directory.

        Args:
            iter_path: Path to an iteration directory (e.g. run_xxx/iteration_003)
                       or a run directory (loads the last iteration).
            mcts_config: MCTS config override. If None, uses the run's config.
            temperature: Temperature for move selection.
            name: Player name. Defaults to the iteration directory name.
        """
        iter_path = Path(iter_path)

        # If pointing at a run directory, find the last iteration
        if (iter_path / "run_config.json").exists():
            run_dir = iter_path
            iter_dirs = sorted(iter_path.glob("iteration_*"))
            if not iter_dirs:
                raise FileNotFoundError(f"No iteration directories in {iter_path}")
            iter_path = iter_dirs[-1]
        else:
            run_dir = iter_path.parent

        checkpoint = iter_path / "post_training.pt"
        if not checkpoint.exists():
            raise FileNotFoundError(f"No post_training.pt in {iter_path}")

        # Load run config
        run_config_path = run_dir / "run_config.json"
        run_config = json.loads(run_config_path.read_text())

        net_class = _import_class(run_config["net_class"])

        if mcts_config is None:
            mcts_config = MCTSConfig(**run_config["mcts_config"])

        net = net_class.from_checkpoint(str(checkpoint))
        player_name = name or iter_path.name
        return cls(net=net, mcts_config=mcts_config, temperature=temperature, name=player_name)


class GreedyMCTSPlayer(MCTSPlayer):
    def __init__(self, net: BaseNeuralNet, mcts_config: MCTSConfig, name: str | None = None):
        super().__init__(net, mcts_config, temperature=0.0, name=name)
