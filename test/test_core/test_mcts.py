"""Tests for game-agnostic MCTS."""

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from pymcts.core.base_game import Board2DGame, GameState
from pymcts.core.base_neural_net import BaseNeuralNet
from pymcts.core.config import MCTSConfig
from pymcts.core.mcts import MCTS, MCTSNode


# ---------------------------------------------------------------------------
# TicTacToe implementation for testing
# ---------------------------------------------------------------------------

class TicTacToeState(GameState):
    def __init__(self, board: list[int], current_player: int):
        self.board = list(board)
        self.current_player = current_player


class TicTacToe(Board2DGame):
    """Minimal TicTacToe for MCTS testing. Players are 0 and 1."""

    def __init__(self):
        super().__init__(board_rows=3, board_cols=3)
        self._board = [0] * 9  # 0=empty, 1=player0, 2=player1
        self._current_player = 0
        self._winner: int | None = None
        self._is_over = False

    @property
    def current_player(self) -> int:
        return self._current_player

    @property
    def is_over(self) -> bool:
        return self._is_over

    @property
    def winner(self) -> int | None:
        return self._winner

    def get_state(self) -> TicTacToeState:
        return TicTacToeState(self._board, self._current_player)

    def to_mask(self) -> torch.Tensor:
        return torch.tensor([cell == 0 for cell in self._board], dtype=torch.bool)

    def make_action(self, action: int) -> None:
        self._board[action] = self._current_player + 1
        self._check_winner()
        if not self._is_over:
            self._current_player = 1 - self._current_player

    def copy(self) -> "TicTacToe":
        g = TicTacToe()
        g._board = list(self._board)
        g._current_player = self._current_player
        g._winner = self._winner
        g._is_over = self._is_over
        return g

    def get_result(self, player: int) -> float | None:
        if not self._is_over:
            return None
        if self._winner is None:
            return 0.0
        return 1.0 if self._winner == player else -1.0

    def _check_winner(self):
        lines = [
            (0, 1, 2), (3, 4, 5), (6, 7, 8),
            (0, 3, 6), (1, 4, 7), (2, 5, 8),
            (0, 4, 8), (2, 4, 6),
        ]
        for a, b, c in lines:
            if self._board[a] != 0 and self._board[a] == self._board[b] == self._board[c]:
                self._winner = self._board[a] - 1
                self._is_over = True
                return
        if all(cell != 0 for cell in self._board):
            self._is_over = True


# ---------------------------------------------------------------------------
# DummyNet — returns uniform policy and 0 value
# ---------------------------------------------------------------------------

class DummyNet(BaseNeuralNet):
    """Neural net that returns uniform log-policy and zero value."""

    def __init__(self, action_size: int = 9):
        super().__init__()
        self.action_size = action_size
        # Need at least one parameter for nn.Module
        self._dummy = nn.Linear(1, 1)

    def encode(self, state: GameState) -> torch.Tensor:
        return torch.zeros(self.action_size)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch = x.shape[0]
        log_policy = torch.full((batch, self.action_size), -float("inf"))
        # Uniform over all actions (will be masked later anyway)
        uniform = torch.log(torch.tensor(1.0 / self.action_size))
        log_policy = torch.full((batch, self.action_size), uniform.item())
        value = torch.zeros(batch, 1)
        return log_policy, value

    def save_checkpoint(self, path: str) -> None:
        pass

    def load_checkpoint(self, path: str) -> None:
        pass

    @classmethod
    def from_checkpoint(cls, path: str) -> "DummyNet":
        return cls()

    def copy(self) -> "DummyNet":
        return copy.deepcopy(self)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_search_returns_root():
    """MCTS._search returns a root node with visits."""
    game = TicTacToe()
    net = DummyNet()
    config = MCTSConfig(num_simulations=20, c_puct=1.5)
    mcts = MCTS(net, config)

    root = mcts._search(game)
    assert isinstance(root, MCTSNode)
    # root should have visit_count == num_simulations + 1 (initial expand counts)
    assert root.visit_count > 0
    assert root.is_expanded


def test_get_action_probs_shape():
    """get_action_probs returns a 1D tensor of length action_space_size."""
    game = TicTacToe()
    net = DummyNet()
    config = MCTSConfig(num_simulations=20, c_puct=1.5)
    mcts = MCTS(net, config)

    probs = mcts.get_action_probs(game, temperature=1.0)
    assert probs.shape == (9,)
    assert probs.sum().item() > 0.99  # should sum to ~1


def test_visit_counts_1d():
    """root.visit_counts(9) returns shape (9,)."""
    game = TicTacToe()
    net = DummyNet()
    config = MCTSConfig(num_simulations=20, c_puct=1.5)
    mcts = MCTS(net, config)

    root = mcts._search(game)
    vc = root.visit_counts(9)
    assert vc.shape == (9,)
    assert vc.sum().item() > 0
