"""
Bridgit Game Implementation

Played on a (2n+1)×(2n+1) grid. The board stores three kinds of info:
  - Bridges on interior crossings where (r+c)%2==0 and 0<r<2n, 0<c<2n
  - Bridge endpoints stamped ±1 step from the bridge in its direction
  - Boundary cells (row 0, row 2n, col 0, col 2n) that serve as goals

Bridge orientation depends on player + column parity:
  Green (VERTICAL):  even col → vertical,  odd col → horizontal
  Red (HORIZONTAL):  even col → horizontal, odd col → vertical

When a bridge is placed, its two endpoint cells are marked with the player's
value. Win check: BFS over a player's endpoint marks (4-connected) from
one boundary to the opposite boundary.
"""

from collections import deque
from enum import Enum

import numpy as np
import torch


class Player(Enum):
    """Player representation."""
    HORIZONTAL = -1  # Red: tries to connect left to right
    VERTICAL = 1     # Green: tries to connect top to bottom


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

        On the (2n+1) grid, parity is flipped vs the (2n-1) rule:
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

    def to_tensor(self, player: Player) -> torch.Tensor:
        """Convert board to float32 tensor of shape (3, 2n+1, 2n+1).

        Channels:
          0 — current player's cells (bridges + endpoints)
          1 — opponent's cells (bridges + endpoints)
          2 — playability mask (1 at interior crossings that are empty)
        """
        mine = (self.board == player.value).astype(np.float32)
        theirs = (self.board == -player.value).astype(np.float32)
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
        import plotly.graph_objects as go

        g = 2 * self.n + 1
        fig = go.Figure()

        # Draw potential endpoint dots on boundary
        # Green endpoints: rows 0 and 2n, odd cols
        # Red endpoints: cols 0 and 2n, odd rows
        green_x, green_y = [], []
        red_x, red_y = [], []
        for c in range(1, g - 1, 2):
            green_x.extend([c, c])
            green_y.extend([0, g - 1])
        for r in range(1, g - 1, 2):
            red_x.extend([0, g - 1])
            red_y.extend([r, r])

        fig.add_trace(go.Scatter(
            x=green_x, y=green_y, mode="markers",
            marker=dict(size=8, color="green", opacity=0.3),
            showlegend=False, hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=red_x, y=red_y, mode="markers",
            marker=dict(size=8, color="red", opacity=0.3),
            showlegend=False, hoverinfo="skip",
        ))

        # Draw ghost bridges (potential moves) as very faint lines
        for r in range(1, g - 1):
            for c in range(1, g - 1):
                if (r + c) % 2 != 0 or self.board[r, c] != 0:
                    continue
                for player, color in [(Player.VERTICAL, "green"), (Player.HORIZONTAL, "red")]:
                    eps = self.endpoints(r, c, player)
                    (r0, c0), (r1, c1) = eps
                    fig.add_shape(type="line",
                        x0=c0, x1=c1, y0=r0, y1=r1,
                        line=dict(color=color, width=2),
                        opacity=0.2)

        # Draw bridges and their endpoints
        for r in range(1, g - 1):
            for c in range(1, g - 1):
                if (r + c) % 2 != 0:
                    continue
                val = self.board[r, c]
                if val == 0:
                    continue
                player = Player.VERTICAL if val == 1 else Player.HORIZONTAL
                color = "green" if val == 1 else "red"
                d = self.bridge_direction(player, c)
                eps = self.endpoints(r, c, player)
                (r0, c0), (r1, c1) = eps
                fig.add_shape(type="line",
                    x0=c0, x1=c1, y0=r0, y1=r1,
                    line=dict(color=color, width=5))
                # Draw endpoint dots
                fig.add_trace(go.Scatter(
                    x=[c0, c1], y=[r0, r1], mode="markers",
                    marker=dict(size=8, color=color),
                    showlegend=False, hoverinfo="skip",
                ))

        fig.update_layout(
            width=120 + g * 50, height=120 + g * 50,
            xaxis=dict(
                range=[-0.5, g - 0.5], scaleanchor="y",
                tickmode="array",
                tickvals=list(range(g)),
                ticktext=[str(i) for i in range(g)],
                title="col",
            ),
            yaxis=dict(
                range=[g - 0.5, -0.5], autorange=False,
                tickmode="array",
                tickvals=list(range(g)),
                ticktext=[str(i) for i in range(g)],
                title="row",
            ),
            plot_bgcolor="white",
            margin=dict(l=50, r=30, t=30, b=50),
        )

        fig.show()
        return fig

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


class Bridgit:
    """Bridgit game — manages turns, win detection, and game flow."""

    def __init__(self, n: int = 5):
        self.state = GameState.empty(n)
        self.n = n
        self.current_player = Player.HORIZONTAL
        self.winner: Player | None = None
        self.game_over = False

    @property
    def grid(self) -> np.ndarray:
        """Direct access to the board array."""
        return self.state.board

    def get_available_moves(self) -> list[tuple[int, int]]:
        """Get list of available moves (empty interior crossings)."""
        g = 2 * self.n + 1
        return [
            (r, c)
            for r in range(1, g - 1) for c in range(1, g - 1)
            if (r + c) % 2 == 0 and self.state.board[r, c] == 0
        ]

    def is_valid_move(self, row: int, col: int) -> bool:
        if self.game_over:
            return False
        return self.state.is_crossing(row, col) and self.state.board[row, col] == 0

    def make_move(self, row: int, col: int) -> bool:
        """Make a move on the board. Returns True if successful."""
        if not self.is_valid_move(row, col):
            return False
        self.state = self.state.make_move(row, col, self.current_player)
        if self._check_winner():
            self.winner = self.current_player
            self.game_over = True
        else:
            self.current_player = (
                Player.VERTICAL if self.current_player == Player.HORIZONTAL else Player.HORIZONTAL
            )
        return True

    def _check_winner(self) -> bool:
        """Check if the current player has won."""
        if self.current_player == Player.HORIZONTAL:
            return self._has_path_endpoints("left", "right")
        return self._has_path_endpoints("top", "bottom")

    def _has_path_endpoints(self, start_side: str, end_side: str) -> bool:
        """BFS over the current player's endpoint marks (4-connected).

        Green: top (row=0) to bottom (row=2n)
        Red: left (col=0) to right (col=2n)
        """
        board = self.state.board
        g = 2 * self.n + 1
        player_val = self.current_player.value

        # Find start cells: player's endpoints on the start boundary
        start = set()
        for r in range(g):
            for c in range(g):
                if board[r, c] != player_val:
                    continue
                if start_side == "left" and c == 0:
                    start.add((r, c))
                elif start_side == "top" and r == 0:
                    start.add((r, c))

        if not start:
            return False

        visited = set(start)
        queue = deque(start)

        while queue:
            r, c = queue.popleft()
            if end_side == "right" and c == g - 1:
                return True
            if end_side == "bottom" and r == g - 1:
                return True

            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < g and 0 <= nc < g and (nr, nc) not in visited:
                    if board[nr, nc] == player_val:
                        visited.add((nr, nc))
                        queue.append((nr, nc))

        return False

    def copy(self) -> "Bridgit":
        """Create a deep copy of the game."""
        new_game = Bridgit.__new__(Bridgit)
        new_game.state = GameState(self.state.board.copy(), self.n)
        new_game.n = self.n
        new_game.current_player = self.current_player
        new_game.winner = self.winner
        new_game.game_over = self.game_over
        return new_game

    def visualize(self):
        """Visualize the current game state using plotly."""
        return self.state.visualize()

    def __str__(self) -> str:
        result = [f"Current Player: {self.current_player.name}", ""]
        result.append(repr(self.state))
        if self.game_over:
            result.extend(["", f"Winner: {self.winner.name}"])
        return "\n".join(result)
