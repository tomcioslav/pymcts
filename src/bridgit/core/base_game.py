"""Abstract base classes for game engine."""

from abc import ABC, abstractmethod

import torch


class GameState:
    """Base class for game states. Each game defines its own subclass.
    The engine passes GameState objects opaquely — only the game and its
    neural net know the concrete type.
    """
    pass


class BaseGame(ABC):
    """Minimal contract for a two-player zero-sum game."""

    @property
    @abstractmethod
    def current_player(self) -> int:
        """Current player id. Always an int (0, 1, ...)."""

    @property
    @abstractmethod
    def is_over(self) -> bool: ...

    @property
    @abstractmethod
    def winner(self) -> int | None:
        """Player id of winner, or None if draw / not over."""

    @property
    @abstractmethod
    def action_space_size(self) -> int:
        """Total number of possible actions. Fixed for a game type."""

    @abstractmethod
    def get_state(self) -> GameState:
        """Return current game state from the current player's perspective.
        Canonicalization is the game's responsibility.
        """

    @abstractmethod
    def to_mask(self) -> torch.Tensor:
        """1D boolean tensor of length action_space_size. True = legal action."""

    def valid_actions(self) -> list[int]:
        """List of currently legal action indices. Derived from to_mask()."""
        return self.to_mask().nonzero(as_tuple=False).squeeze(-1).tolist()

    @abstractmethod
    def make_action(self, action: int) -> None:
        """Apply action in-place. Mutates the game state."""

    @abstractmethod
    def copy(self) -> "BaseGame": ...

    @abstractmethod
    def get_result(self, player: int) -> float | None:
        """1.0 for win, -1.0 for loss, 0.0 for draw, None if not over."""

    def get_config(self) -> dict:
        """Return a serializable config dict that can recreate this game instance.

        Override in subclasses that carry configuration (board size, etc.).
        The default implementation returns an empty dict.
        """
        return {}


class Board2DGame(BaseGame):
    """Base for games played on a 2D rectangular grid."""

    def __init__(self, board_rows: int, board_cols: int):
        self._board_rows = board_rows
        self._board_cols = board_cols

    @property
    def board_rows(self) -> int:
        return self._board_rows

    @property
    def board_cols(self) -> int:
        return self._board_cols

    @property
    def action_space_size(self) -> int:
        return self._board_rows * self._board_cols

    def action_to_row_col(self, action: int) -> tuple[int, int]:
        return divmod(action, self._board_cols)

    def row_col_to_action(self, row: int, col: int) -> int:
        return row * self._board_cols + col
