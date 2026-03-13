"""Schema definitions for Bridgit."""

from bridgit.schema.move import Move
from bridgit.schema.player import Player
from bridgit.schema.game_record import GameRecord, MoveRecord

__all__ = ["GameRecord", "Move", "MoveRecord", "Player"]
