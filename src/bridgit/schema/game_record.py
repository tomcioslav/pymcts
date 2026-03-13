"""Game recording types for Arena play."""

from __future__ import annotations

import torch
from pydantic import BaseModel, ConfigDict

from bridgit.schema.move import Move
from bridgit.schema.player import Player


class MoveRecord(BaseModel):
    """A single move with optional MCTS policy."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    move: Move
    player: Player
    moves_left_after: int
    policy: torch.Tensor | None = None  # (g, g) MCTS visit-count probs


class GameRecord(BaseModel):
    """Full record of a played game."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    board_size: int
    moves: list[MoveRecord]
    winner: Player
    horizontal_player: str
    vertical_player: str

    @property
    def grid_size(self) -> int:
        return 2 * self.board_size + 1

    @property
    def num_moves(self) -> int:
        return len(self.moves)

    def summary(self) -> str:
        """One-line game summary."""
        h = self.horizontal_player
        v = self.vertical_player
        winner = h if self.winner == Player.HORIZONTAL else v
        return f"{h} vs {v}: {winner} wins in {self.num_moves} moves"
