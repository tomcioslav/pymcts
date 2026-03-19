"""Bridgit game — manages turns, win detection, and game flow."""

import logging
from collections import deque

import numpy as np
import torch

from bridgit.game.state import GameState, Player
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
        self.move_count = 0          # total moves played
        self.moves_left_in_turn = 1  # first player gets 1 move, then 2 each

    @property
    def grid(self) -> np.ndarray:
        """Direct access to the board array."""
        return self.state.board

    def get_available_moves(self) -> list[Move]:
        """Get list of available moves (empty interior crossings)."""
        g = 2 * self.n + 1
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
        self.state = self.state.make_move(move.row, move.col, self.current_player)
        self.move_count += 1
        self.moves_left_in_turn -= 1
        logger.debug("Move #%d: (%d,%d) by %s, moves_left=%d",
                     self.move_count, move.row, move.col,
                     self.current_player.name, self.moves_left_in_turn)
        if self._check_winner(self.current_player):
            self.winner = self.current_player
            self.game_over = True
            logger.info("Game over at move #%d: %s wins",
                        self.move_count, self.winner.name)
        elif self.moves_left_in_turn == 0:
            self.current_player = (
                Player.VERTICAL if self.current_player == Player.HORIZONTAL else Player.HORIZONTAL
            )
            self.moves_left_in_turn = 2
            logger.debug("Turn switch -> %s",
                         self.current_player.name)
        return True

    def _check_winner(self, player: Player) -> bool:
        """Check if *player* has won.

        Uses the canonical board (always HORIZONTAL orientation) so the
        check is a single left-to-right BFS for HORIZONTAL value (-1).
        """
        canon = self.state.canonical(player)
        board = canon.board
        g = 2 * self.n + 1
        val = Player.HORIZONTAL.value

        # Seed: canonical player's cells on the left boundary (col 0)
        start = {(r, 0) for r in range(g) if board[r, 0] == val}
        if not start:
            return False

        visited = set(start)
        queue = deque(start)

        while queue:
            r, c = queue.popleft()
            if c == g - 1:
                return True
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < g and 0 <= nc < g and (nr, nc) not in visited:
                    if board[nr, nc] == val:
                        visited.add((nr, nc))
                        queue.append((nr, nc))

        return False

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
        return new_game

    def get_result(self, player: Player) -> float | None:
        """Return +1 if player won, -1 if lost, None if game ongoing."""
        if not self.game_over:
            return None
        return 1.0 if self.winner == player else -1.0

    def to_tensor(self) -> "torch.Tensor":
        """Canonical state tensor for the current player. Shape (4, g, g).

        Channels: mine, theirs, playable, moves_left_in_turn (1 or 2).
        """
        base = self.state.canonical(self.current_player).to_tensor()  # (3, g, g)
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
