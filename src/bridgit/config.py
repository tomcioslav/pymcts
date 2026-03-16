"""Project configuration using Pydantic."""

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
    solve_terminal: bool = True  # treat opponent's winning replies as guaranteed losses


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


class ArenaConfig(BaseModel):
    """Arena evaluation settings."""
    num_games: int = 40
    threshold: float = 0.55
    swap_players: bool = True


class Config(BaseModel):
    """Top-level project configuration."""
    paths: PathsConfig = PathsConfig()
    board: BoardConfig = BoardConfig()
    mcts: MCTSConfig = MCTSConfig()
    neural_net: NeuralNetConfig = NeuralNetConfig()
    training: TrainingConfig = TrainingConfig()
    arena: ArenaConfig = ArenaConfig()
