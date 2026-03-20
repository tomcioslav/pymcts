"""Project configuration using Pydantic."""

import logging
from pathlib import Path

from pydantic import BaseModel


_PROJECT_ROOT = Path(__file__).parent.parent.parent


class PathsConfig(BaseModel):
    """File system paths for the project."""
    root: Path = _PROJECT_ROOT
    checkpoints: Path = _PROJECT_ROOT / "checkpoints"
    models: Path = _PROJECT_ROOT / "models"
    data: Path = _PROJECT_ROOT / "data"
    trainings: Path = _PROJECT_ROOT / "trainings"

paths = PathsConfig()

class BoardConfig(BaseModel):
    """Board / game settings."""
    size: int = 5

    @property
    def grid_size(self) -> int:
        return 2 * self.size + 1



class MCTSConfig(BaseModel):
    """Monte Carlo Tree Search settings."""
    num_simulations: int = 100
    c_puct: float = 1.5
    dirichlet_alpha: float = 1.0
    dirichlet_epsilon: float = 0.25
    num_parallel_leaves: int = 1  # leaves per tree in batched MCTS (virtual loss)


class NeuralNetConfig(BaseModel):
    """Neural network architecture settings."""
    num_channels: int = 64
    num_res_blocks: int = 4


class TrainingConfig(BaseModel):
    """Training loop settings."""
    num_iterations: int = 50
    num_self_play_games: int = 50
    num_epochs: int = 20
    batch_size: int = 64
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    replay_buffer_size: int = 5  # number of iterations to keep
    self_play_batch_size: int = 8  # concurrent games in batched self-play


class ArenaConfig(BaseModel):
    """Arena evaluation settings."""
    num_games: int = 40
    threshold: float = 0.55
    swap_players: bool = True


_LOGGER_NAMES = {
    "mcts": "bridgit.mcts",
    "arena": "bridgit.arena",
    "game": "bridgit.game",
}


class LoggingSettings:
    """Controls log levels for bridgit subsystems.

    Usage:
        from bridgit.config import logging_settings
        logging_settings.mcts = "DEBUG"
        logging_settings.arena = "INFO"
        logging_settings.game = "WARNING"
    """

    def __init__(self):
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(name)s] %(levelname)s: %(message)s",
            datefmt="%H:%M:%S",
        ))
        for logger_name in _LOGGER_NAMES.values():
            logger = logging.getLogger(logger_name)
            logger.addHandler(handler)
            logger.setLevel(logging.WARNING)

    def _set(self, key: str, level: str):
        logging.getLogger(_LOGGER_NAMES[key]).setLevel(level.upper())

    def _get(self, key: str) -> str:
        return logging.getLevelName(
            logging.getLogger(_LOGGER_NAMES[key]).level
        )

    @property
    def mcts(self) -> str:
        return self._get("mcts")

    @mcts.setter
    def mcts(self, level: str):
        self._set("mcts", level)

    @property
    def arena(self) -> str:
        return self._get("arena")

    @arena.setter
    def arena(self, level: str):
        self._set("arena", level)

    @property
    def game(self) -> str:
        return self._get("game")

    @game.setter
    def game(self, level: str):
        self._set("game", level)

    def __repr__(self) -> str:
        return f"LoggingSettings(mcts={self.mcts!r}, arena={self.arena!r}, game={self.game!r})"


logging_settings = LoggingSettings()


class Config(BaseModel):
    """Top-level project configuration."""
    paths: PathsConfig = PathsConfig()
    board: BoardConfig = BoardConfig()
    mcts: MCTSConfig = MCTSConfig()
    neural_net: NeuralNetConfig = NeuralNetConfig()
    training: TrainingConfig = TrainingConfig()
    arena: ArenaConfig = ArenaConfig()
