"""
GameState for Bridgit

Immutable board state for the (2n+1)×(2n+1) grid. The board stores:
  - Bridges on interior crossings where (r+c)%2==0 and 0<r<2n, 0<c<2n
  - Bridge endpoints stamped ±1 step from the bridge in its direction
  - Boundary cells (row 0, row 2n, col 0, col 2n) that serve as goals

Bridge orientation depends on player + column parity:
  Green (VERTICAL):  even col → horizontal,  odd col → vertical
  Red (HORIZONTAL):  even col → vertical,    odd col → horizontal
"""

import numpy as np
import torch

from bridgit.schema import Player


class GameState:
    """Immutable board state for the (2n+1)×(2n+1) Bridgit grid.

    Board cells:
      0  — empty
      -1 — Red (HORIZONTAL) bridge or endpoint
      1  — Green (VERTICAL) bridge or endpoint
      -2 — blocked (crossing claimed by the opponent, endpoints not here)
    Actually we keep it simpler: bridges store player value at the crossing,
    and also stamp the player value at the two endpoint cells.
    """

    __slots__ = ["board", "n"]

    def __init__(self, board: np.ndarray, n: int):
        self.board = board
        self.n = n

    @classmethod
    def empty(cls, n: int = 5) -> "GameState":
        g = 2 * n + 1
        board = np.zeros((g, g), dtype=int)
        # Pre-populate boundary dots:
        # Green (VERTICAL=1) owns top (row 0) and bottom (row 2n) at odd columns
        for c in range(1, g, 2):
            board[0, c] = Player.VERTICAL.value
            board[g - 1, c] = Player.VERTICAL.value
        # Red (HORIZONTAL=-1) owns left (col 0) and right (col 2n) at odd rows
        for r in range(1, g, 2):
            board[r, 0] = Player.HORIZONTAL.value
            board[r, g - 1] = Player.HORIZONTAL.value
        return cls(board, n)

    @staticmethod
    def bridge_direction(player: Player, col: int) -> str:
        """Return 'v' or 'h' for this player at this column.
          Green (VERTICAL):  even col → horizontal,  odd col → vertical
          Red (HORIZONTAL):  even col → vertical,    odd col → horizontal
        """
        if player == Player.VERTICAL:
            return "h" if col % 2 == 0 else "v"
        else:
            return "v" if col % 2 == 0 else "h"

    @staticmethod
    def endpoints(row: int, col: int, player: Player) -> list[tuple[int, int]]:
        """Return the two endpoint cells for a bridge at (row, col)."""
        d = GameState.bridge_direction(player, col)
        if d == "v":
            return [(row - 1, col), (row + 1, col)]
        else:
            return [(row, col - 1), (row, col + 1)]

    def is_crossing(self, row: int, col: int) -> bool:
        """True if (row, col) is a playable interior crossing."""
        g = 2 * self.n + 1
        return (
            (row + col) % 2 == 0
            and 0 < row < g - 1
            and 0 < col < g - 1
        )

    def make_move(self, row: int, col: int, player: Player) -> "GameState":
        """Return a new GameState with the bridge placed and endpoints stamped."""
        g = 2 * self.n + 1
        if row < 0 or row >= g or col < 0 or col >= g:
            raise ValueError(f"Move ({row}, {col}) out of bounds for {g}x{g} grid")
        if not self.is_crossing(row, col):
            raise ValueError(f"Position ({row}, {col}) is not a playable crossing")
        if self.board[row, col] != 0:
            raise ValueError(f"Crossing ({row}, {col}) is already claimed")
        new_board = self.board.copy()
        new_board[row, col] = player.value
        for er, ec in self.endpoints(row, col, player):
            new_board[er, ec] = player.value
        return GameState(new_board, self.n)

    def canonical(self, player: Player) -> "GameState":
        """Return the board from *player*'s perspective.

        HORIZONTAL is the canonical orientation (left → right).
        If *player* is already HORIZONTAL the board is returned as-is.
        If *player* is VERTICAL the board is transposed and values negated,
        so the current player's pieces become HORIZONTAL value (-1) and
        top/bottom boundaries become left/right.
        """
        if player == Player.HORIZONTAL:
            return GameState(self.board.copy(), self.n)
        return GameState(-self.board.T.copy(), self.n)

    def to_tensor(self) -> torch.Tensor:
        """Convert board to float32 tensor of shape (3, 2n+1, 2n+1).

        Assumes the board is already in canonical form (HORIZONTAL perspective).
        Channels:
          0 — current player's cells (HORIZONTAL = -1)
          1 — opponent's cells (VERTICAL = 1)
          2 — playability mask (1 at empty interior crossings)
        """
        mine = (self.board == Player.HORIZONTAL.value).astype(np.float32)
        theirs = (self.board == Player.VERTICAL.value).astype(np.float32)
        g = 2 * self.n + 1
        playable = np.zeros((g, g), dtype=np.float32)
        for r in range(1, g - 1):
            for c in range(1, g - 1):
                if (r + c) % 2 == 0 and self.board[r, c] == 0:
                    playable[r, c] = 1.0
        return torch.from_numpy(np.stack([mine, theirs, playable]))

    def to_mask(self) -> torch.Tensor:
        """Return float32 tensor of shape (2n+1, 2n+1):
        1.0 at empty interior crossings, 0.0 elsewhere.

        Assumes the board is already in canonical form.
        """
        g = 2 * self.n + 1
        mask = np.zeros((g, g), dtype=np.float32)
        for r in range(1, g - 1):
            for c in range(1, g - 1):
                if (r + c) % 2 == 0 and self.board[r, c] == 0:
                    mask[r, c] = 1.0
        return torch.from_numpy(mask)

    def visualize(self):
        """Visualize the board using plotly."""
        from bridgit.visualizer import Visualizer
        Visualizer.visualize_game_state(self).show()

    def __repr__(self) -> str:
        g = 2 * self.n + 1
        rows = []
        for r in range(g):
            cells = []
            for c in range(g):
                val = self.board[r, c]
                on_boundary = r == 0 or r == g - 1 or c == 0 or c == g - 1
                is_cross = (r + c) % 2 == 0 and not on_boundary

                if on_boundary:
                    if val == 1:
                        cells.append("G")
                    elif val == -1:
                        cells.append("R")
                    elif (r + c) % 2 == 1:
                        # Potential endpoint position
                        if r == 0 or r == g - 1:
                            cells.append("g")  # green goal
                        else:
                            cells.append("r")  # red goal
                    else:
                        cells.append("x")
                elif is_cross:
                    if val == 0:
                        cells.append("\u00b7")  # ·
                    elif val == 1:
                        d = self.bridge_direction(Player.VERTICAL, c)
                        cells.append("\u2503" if d == "v" else "\u2501")
                    else:
                        d = self.bridge_direction(Player.HORIZONTAL, c)
                        cells.append("\u2501" if d == "h" else "\u2503")
                else:
                    # Non-crossing interior cell — could be an endpoint
                    if val == 1:
                        cells.append("G")
                    elif val == -1:
                        cells.append("R")
                    else:
                        cells.append(" ")
            rows.append(" ".join(cells))
        return "\n".join(rows)
