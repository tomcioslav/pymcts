"""Move type for Bridgit."""

from __future__ import annotations

from pydantic import BaseModel

from bridgit.schema.player import Player


class Move(BaseModel, frozen=True):
    """A move on the Bridgit board."""
    row: int
    col: int

    def canonicalize(self, player: Player) -> Move:
        """Convert from original board coordinates to canonical (HORIZONTAL) space."""
        if player == Player.HORIZONTAL:
            return self
        return Move(row=self.col, col=self.row)

    def decanonicalize(self, player: Player) -> Move:
        """Convert from canonical (HORIZONTAL) space back to original board coordinates."""
        if player == Player.HORIZONTAL:
            return self
        return Move(row=self.col, col=self.row)
