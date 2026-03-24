"""Player enum for Bridgit."""

from enum import Enum


class Player(Enum):
    """Player representation."""
    HORIZONTAL = -1  # Red: tries to connect left to right
    VERTICAL = 1     # Green: tries to connect top to bottom
