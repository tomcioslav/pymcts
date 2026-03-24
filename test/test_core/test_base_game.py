import pytest
import torch

from pymcts.core.base_game import BaseGame, Board2DGame, GameState


class DummyState(GameState):
    def __init__(self, board: list[int]):
        self.board = board


class DummyGame(Board2DGame):
    """Minimal 3x3 game for testing."""

    def __init__(self):
        super().__init__(board_rows=3, board_cols=3)
        self._board = [0] * 9
        self._current_player = 0
        self._is_over = False
        self._winner = None

    @property
    def current_player(self) -> int:
        return self._current_player

    @property
    def is_over(self) -> bool:
        return self._is_over

    @property
    def winner(self) -> int | None:
        return self._winner

    def get_state(self) -> GameState:
        return DummyState(self._board[:])

    def to_mask(self) -> torch.Tensor:
        return torch.tensor([cell == 0 for cell in self._board], dtype=torch.bool)

    def make_action(self, action: int) -> None:
        self._board[action] = self._current_player + 1
        self._current_player = 1 - self._current_player

    def copy(self) -> "DummyGame":
        new = DummyGame()
        new._board = self._board[:]
        new._current_player = self._current_player
        new._is_over = self._is_over
        new._winner = self._winner
        return new

    def get_result(self, player: int) -> float | None:
        if not self._is_over:
            return None
        if self._winner == player:
            return 1.0
        return -1.0


class TestBaseGame:
    def test_action_space_size(self):
        game = DummyGame()
        assert game.action_space_size == 9

    def test_valid_actions_from_mask(self):
        game = DummyGame()
        assert game.valid_actions() == list(range(9))

    def test_valid_actions_after_move(self):
        game = DummyGame()
        game.make_action(4)
        actions = game.valid_actions()
        assert 4 not in actions
        assert len(actions) == 8

    def test_get_state_returns_game_state(self):
        game = DummyGame()
        state = game.get_state()
        assert isinstance(state, GameState)

    def test_copy_is_independent(self):
        game = DummyGame()
        game.make_action(0)
        copy = game.copy()
        copy.make_action(1)
        assert game.valid_actions() != copy.valid_actions()


class TestBoard2DGame:
    def test_action_to_row_col(self):
        game = DummyGame()
        assert game.action_to_row_col(0) == (0, 0)
        assert game.action_to_row_col(4) == (1, 1)
        assert game.action_to_row_col(8) == (2, 2)

    def test_row_col_to_action(self):
        game = DummyGame()
        assert game.row_col_to_action(0, 0) == 0
        assert game.row_col_to_action(1, 1) == 4
        assert game.row_col_to_action(2, 2) == 8

    def test_roundtrip(self):
        game = DummyGame()
        for action in range(9):
            r, c = game.action_to_row_col(action)
            assert game.row_col_to_action(r, c) == action
