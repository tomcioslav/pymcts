"""Bridgit game — manages turns, win detection, and game flow."""

import logging

import numpy as np
import torch

from bridgit.game.state import GameState, Player
from bridgit.game.union_find import UnionFind
from bridgit.schema import Move
from bridgit.config import BoardConfig

logger = logging.getLogger("bridgit.game")


class Bridgit:
    """Bridgit game — manages turns, win detection, and game flow."""

    def __init__(self, board_config: BoardConfig = BoardConfig()):
        self.state = GameState.empty(board_config.size)
        self.n = board_config.size
        self.board_config = board_config
        self.current_player = Player.HORIZONTAL
        self.winner: Player | None = None
        self.game_over = False
        self.move_count = 0
        self.moves_left_in_turn = 1

        g = board_config.grid_size
        # Union-Find per player: g*g cells + 2 sentinels
        # sentinel indices: g*g = boundary_start, g*g+1 = boundary_end
        self._uf = {
            Player.HORIZONTAL: UnionFind(g * g + 2),
            Player.VERTICAL: UnionFind(g * g + 2),
        }
        self._g = g
        self._sentinel_start = g * g      # left/top boundary
        self._sentinel_end = g * g + 1    # right/bottom boundary

        # Connect boundary cells to sentinels
        # HORIZONTAL: left (col 0) and right (col 2n) at odd rows
        uf_h = self._uf[Player.HORIZONTAL]
        for r in range(1, g, 2):
            uf_h.union(r * g + 0, self._sentinel_start)
            uf_h.union(r * g + (g - 1), self._sentinel_end)

        # VERTICAL: top (row 0) and bottom (row 2n) at odd columns
        uf_v = self._uf[Player.VERTICAL]
        for c in range(1, g, 2):
            uf_v.union(0 * g + c, self._sentinel_start)
            uf_v.union((g - 1) * g + c, self._sentinel_end)

    def _cell_idx(self, r: int, c: int) -> int:
        return r * self._g + c

    @property
    def grid(self) -> np.ndarray:
        """Direct access to the board array."""
        return self.state.board

    def get_available_moves(self) -> list[Move]:
        """Get list of available moves (empty interior crossings)."""
        g = self._g
        return [
            Move(row=r, col=c)
            for r in range(1, g - 1) for c in range(1, g - 1)
            if (r + c) % 2 == 0 and self.state.board[r, c] == 0
        ]

    def is_valid_move(self, move: Move) -> bool:
        if self.game_over:
            return False
        return (self.state.is_crossing(move.row, move.col)
                and self.state.board[move.row, move.col] == 0)

    def make_move(self, move: Move) -> bool:
        """Make a move on the board. Returns True if successful."""
        if not self.is_valid_move(move):
            logger.warning("Invalid move (%d,%d) by %s (move_count=%d)",
                           move.row, move.col,
                           self.current_player.name, self.move_count)
            return False

        player = self.current_player
        r, c = move.row, move.col
        self.state = self.state.make_move(r, c, player)
        self.move_count += 1
        self.moves_left_in_turn -= 1
        logger.debug("Move #%d: (%d,%d) by %s, moves_left=%d",
                     self.move_count, r, c, player.name, self.moves_left_in_turn)

        # Update Union-Find: union bridge cell with its endpoints
        uf = self._uf[player]
        bridge_idx = self._cell_idx(r, c)
        for er, ec in GameState.endpoints(r, c, player):
            uf.union(bridge_idx, self._cell_idx(er, ec))

        # Also union endpoints with any adjacent same-player cells
        for er, ec in GameState.endpoints(r, c, player):
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = er + dr, ec + dc
                if (0 <= nr < self._g and 0 <= nc < self._g
                        and self.state.board[nr, nc] == player.value):
                    uf.union(self._cell_idx(er, ec), self._cell_idx(nr, nc))

        # Check win via Union-Find
        if uf.connected(self._sentinel_start, self._sentinel_end):
            self.winner = player
            self.game_over = True
            logger.info("Game over at move #%d: %s wins",
                        self.move_count, self.winner.name)
        elif self.moves_left_in_turn == 0:
            self.current_player = (
                Player.VERTICAL if player == Player.HORIZONTAL else Player.HORIZONTAL
            )
            self.moves_left_in_turn = 2
            logger.debug("Turn switch -> %s", self.current_player.name)
        return True

    def copy(self) -> "Bridgit":
        """Create a deep copy of the game."""
        new_game = Bridgit.__new__(Bridgit)
        new_game.state = GameState(self.state.board.copy(), self.n)
        new_game.n = self.n
        new_game.board_config = self.board_config
        new_game.current_player = self.current_player
        new_game.winner = self.winner
        new_game.game_over = self.game_over
        new_game.move_count = self.move_count
        new_game.moves_left_in_turn = self.moves_left_in_turn
        new_game._g = self._g
        new_game._sentinel_start = self._sentinel_start
        new_game._sentinel_end = self._sentinel_end
        new_game._uf = {
            Player.HORIZONTAL: self._uf[Player.HORIZONTAL].copy(),
            Player.VERTICAL: self._uf[Player.VERTICAL].copy(),
        }
        return new_game

    def get_result(self, player: Player) -> float | None:
        """Return +1 if player won, -1 if lost, None if game ongoing."""
        if not self.game_over:
            return None
        return 1.0 if self.winner == player else -1.0

    def to_tensor(self) -> "torch.Tensor":
        """Canonical state tensor for the current player. Shape (4, g, g)."""
        base = self.state.canonical(self.current_player).to_tensor()
        g = base.shape[1]
        moves_plane = torch.full((1, g, g), self.moves_left_in_turn, dtype=torch.float32)
        return torch.cat([base, moves_plane], dim=0)

    def to_mask(self) -> "torch.Tensor":
        """Valid-moves mask from the current player's perspective. Shape (g, g)."""
        return self.state.canonical(self.current_player).to_mask()

    def visualize(self):
        """Visualize the current game state using plotly."""
        return self.state.visualize()

    def __str__(self) -> str:
        result = [f"Current Player: {self.current_player.name}", ""]
        result.append(repr(self.state))
        if self.game_over:
            result.extend(["", f"Winner: {self.winner.name}"])
        return "\n".join(result)
