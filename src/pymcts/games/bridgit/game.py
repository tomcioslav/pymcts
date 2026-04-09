"""BridgitGame — Board2DGame implementation for Bridgit."""

import logging

import numpy as np
import torch

from pymcts.core.base_game import Board2DGame, GameState
from pymcts.games.bridgit.config import BoardConfig
from pymcts.games.bridgit.player import Player
from pymcts.games.bridgit.union_find import UnionFind

logger = logging.getLogger("bridgit.games.bridgit")


class BridgitGameState(GameState):
    """Holds the canonical board (numpy array), n, and moves_left_in_turn."""

    __slots__ = ["board", "n", "moves_left_in_turn"]

    def __init__(self, board: np.ndarray, n: int, moves_left_in_turn: int):
        self.board = board
        self.n = n
        self.moves_left_in_turn = moves_left_in_turn


class BridgitGame(Board2DGame):
    """Bridgit game implementing the Board2DGame interface."""

    # Map between int player ids and Player enum
    _PLAYER_FROM_ID = {0: Player.HORIZONTAL, 1: Player.VERTICAL}
    _ID_FROM_PLAYER = {Player.HORIZONTAL: 0, Player.VERTICAL: 1}

    def __init__(self, config: BoardConfig = BoardConfig()):
        g = config.grid_size
        super().__init__(board_rows=g, board_cols=g)

        self._n = config.size
        self._config = config
        self._g = g
        self._board = self._make_empty_board(config.size, g)
        self._current_player = Player.HORIZONTAL
        self._winner: Player | None = None
        self._game_over = False
        self._move_count = 0
        self._moves_left_in_turn = 1
        self._sentinel_start = g * g
        self._sentinel_end = g * g + 1
        self._uf = self._init_union_find(g)

    def _init_union_find(self, g: int) -> dict:
        """Create per-player Union-Find structures and connect boundary sentinels."""
        uf = {
            Player.HORIZONTAL: UnionFind(g * g + 2),
            Player.VERTICAL: UnionFind(g * g + 2),
        }
        self._connect_horizontal_sentinels(uf[Player.HORIZONTAL], g)
        self._connect_vertical_sentinels(uf[Player.VERTICAL], g)
        return uf

    def _connect_horizontal_sentinels(self, uf: UnionFind, g: int) -> None:
        """Connect left (col 0) and right (col 2n) boundary cells to sentinels."""
        for r in range(1, g, 2):
            uf.union(r * g + 0, self._sentinel_start)
            uf.union(r * g + (g - 1), self._sentinel_end)

    def _connect_vertical_sentinels(self, uf: UnionFind, g: int) -> None:
        """Connect top (row 0) and bottom (row 2n) boundary cells to sentinels."""
        for c in range(1, g, 2):
            uf.union(0 * g + c, self._sentinel_start)
            uf.union((g - 1) * g + c, self._sentinel_end)

    @staticmethod
    def _make_empty_board(n: int, g: int) -> np.ndarray:
        board = np.zeros((g, g), dtype=int)
        # Green (VERTICAL=1) owns top (row 0) and bottom (row 2n) at odd columns
        for c in range(1, g, 2):
            board[0, c] = Player.VERTICAL.value
            board[g - 1, c] = Player.VERTICAL.value
        # Red (HORIZONTAL=-1) owns left (col 0) and right (col 2n) at odd rows
        for r in range(1, g, 2):
            board[r, 0] = Player.HORIZONTAL.value
            board[r, g - 1] = Player.HORIZONTAL.value
        return board

    @staticmethod
    def _bridge_direction(player: Player, col: int) -> str:
        """Return 'v' or 'h' for this player at this column."""
        if player == Player.VERTICAL:
            return "h" if col % 2 == 0 else "v"
        else:
            return "v" if col % 2 == 0 else "h"

    @staticmethod
    def _endpoints(row: int, col: int, player: Player) -> list[tuple[int, int]]:
        """Return the two endpoint cells for a bridge at (row, col)."""
        d = BridgitGame._bridge_direction(player, col)
        if d == "v":
            return [(row - 1, col), (row + 1, col)]
        else:
            return [(row, col - 1), (row, col + 1)]

    def _cell_idx(self, r: int, c: int) -> int:
        return r * self._g + c

    def _is_crossing(self, row: int, col: int) -> bool:
        g = self._g
        return (
            (row + col) % 2 == 0
            and 0 < row < g - 1
            and 0 < col < g - 1
        )

    # --- BaseGame interface ---

    @property
    def current_player(self) -> int:
        return self._ID_FROM_PLAYER[self._current_player]

    @property
    def is_over(self) -> bool:
        return self._game_over

    @property
    def winner(self) -> int | None:
        if self._winner is None:
            return None
        return self._ID_FROM_PLAYER[self._winner]

    def get_state(self) -> BridgitGameState:
        """Return canonical state from current player's perspective."""
        if self._current_player == Player.HORIZONTAL:
            canonical_board = self._board.copy()
        else:
            canonical_board = -self._board.T.copy()
        return BridgitGameState(canonical_board, self._n, self._moves_left_in_turn)

    def get_display_state(self) -> BridgitGameState:
        """Return absolute board state for visualization."""
        return BridgitGameState(self._board.copy(), self._n, self._moves_left_in_turn)

    def to_mask(self) -> torch.Tensor:
        """1D boolean mask of length action_space_size from canonical perspective."""
        g = self._g
        if self._current_player == Player.HORIZONTAL:
            board = self._board
        else:
            board = -self._board.T

        mask = np.zeros((g, g), dtype=bool)
        inner = board[1:g-1, 1:g-1]
        rows_plus_cols = np.add.outer(np.arange(1, g-1), np.arange(1, g-1))
        mask[1:g-1, 1:g-1] = (rows_plus_cols % 2 == 0) & (inner == 0)

        return torch.from_numpy(mask.flatten())

    def make_action(self, action: int) -> None:
        """Apply action (flat index in canonical space) in-place."""
        if self._game_over:
            raise ValueError("Game is already over")

        row, col = self._canonical_to_absolute(action)
        player = self._current_player

        if not self._is_crossing(row, col):
            raise ValueError(f"Position ({row}, {col}) is not a playable crossing")
        if self._board[row, col] != 0:
            raise ValueError(f"Crossing ({row}, {col}) is already claimed")

        self._place_bridge(row, col, player)
        self._move_count += 1
        self._moves_left_in_turn -= 1
        self._update_union_find(row, col, player)
        self._check_win_or_switch(player)

    def _canonical_to_absolute(self, action: int) -> tuple[int, int]:
        """Convert flat canonical action index to absolute board (row, col)."""
        row, col = self.action_to_row_col(action)
        if self._current_player == Player.VERTICAL:
            row, col = col, row
        return row, col

    def _place_bridge(self, row: int, col: int, player: Player) -> None:
        """Stamp the bridge cell and both its endpoints onto the board."""
        self._board[row, col] = player.value
        for er, ec in self._endpoints(row, col, player):
            self._board[er, ec] = player.value

    def _update_union_find(self, row: int, col: int, player: Player) -> None:
        """Connect the new bridge and its endpoints in Union-Find."""
        uf = self._uf[player]
        bridge_idx = self._cell_idx(row, col)
        endpoints = self._endpoints(row, col, player)

        for er, ec in endpoints:
            uf.union(bridge_idx, self._cell_idx(er, ec))

        self._connect_endpoints_to_neighbors(uf, endpoints, player)

    def _connect_endpoints_to_neighbors(
        self, uf: UnionFind, endpoints: list[tuple[int, int]], player: Player
    ) -> None:
        """Union each endpoint with adjacent same-player cells."""
        g = self._g
        for er, ec in endpoints:
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = er + dr, ec + dc
                if (0 <= nr < g and 0 <= nc < g
                        and self._board[nr, nc] == player.value):
                    uf.union(self._cell_idx(er, ec), self._cell_idx(nr, nc))

    def _check_win_or_switch(self, player: Player) -> None:
        """Declare winner if sentinels connected, else switch player when turn ends."""
        uf = self._uf[player]
        if uf.connected(self._sentinel_start, self._sentinel_end):
            self._winner = player
            self._game_over = True
        elif self._moves_left_in_turn == 0:
            self._current_player = (
                Player.VERTICAL if player == Player.HORIZONTAL else Player.HORIZONTAL
            )
            self._moves_left_in_turn = 2

    def copy(self) -> "BridgitGame":
        new = BridgitGame.__new__(BridgitGame)
        new._board_rows = self._board_rows
        new._board_cols = self._board_cols
        self._copy_scalar_state(new)
        new._uf = self._copy_union_find()
        return new

    def _copy_scalar_state(self, new: "BridgitGame") -> None:
        """Copy all scalar and array fields onto a freshly allocated game instance."""
        new._n = self._n
        new._config = self._config
        new._board = self._board.copy()
        new._current_player = self._current_player
        new._winner = self._winner
        new._game_over = self._game_over
        new._move_count = self._move_count
        new._moves_left_in_turn = self._moves_left_in_turn
        new._g = self._g
        new._sentinel_start = self._sentinel_start
        new._sentinel_end = self._sentinel_end

    def _copy_union_find(self) -> dict:
        """Return a deep copy of the per-player Union-Find structures."""
        return {
            Player.HORIZONTAL: self._uf[Player.HORIZONTAL].copy(),
            Player.VERTICAL: self._uf[Player.VERTICAL].copy(),
        }

    def get_result(self, player: int) -> float | None:
        if not self._game_over:
            return None
        return 1.0 if self.winner == player else -1.0

    def get_config(self) -> dict:
        """Return config dict for recreating this game instance."""
        return self._config.model_dump()
