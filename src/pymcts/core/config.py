"""Generic engine configuration."""

import logging
from pathlib import Path

from pydantic import BaseModel


_PROJECT_ROOT = Path(__file__).parent.parent.parent.parent


class PathsConfig(BaseModel):
    """File system paths for the project."""
    root: Path = _PROJECT_ROOT
    checkpoints: Path = _PROJECT_ROOT / "checkpoints"
    models: Path = _PROJECT_ROOT / "models"
    data: Path = _PROJECT_ROOT / "data"
    trainings: Path = _PROJECT_ROOT / "trainings"


class MCTSConfig(BaseModel):
    """Monte Carlo Tree Search settings."""
    num_simulations: int = 100
    c_puct: float = 1.5
    dirichlet_alpha: float = 1.0
    dirichlet_epsilon: float = 0.25
    num_parallel_leaves: int = 1


class TrainingConfig(BaseModel):
    """Training loop settings."""
    num_iterations: int = 50
    num_self_play_games: int = 50
    num_epochs: int = 20
    batch_size: int = 64
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    replay_buffer_size: int = 5
    self_play_batch_size: int = 8
    elo_tracking: bool = False
    elo_reference_interval: int = 5
    elo_games_per_matchup: int = 40
