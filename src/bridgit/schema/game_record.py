"""Game recording types for Arena play."""

from dataclasses import dataclass
from typing import Annotated, Any

import torch
from pydantic import BaseModel, BeforeValidator, PlainSerializer


from bridgit.schema.move import Move
from bridgit.schema.player import Player


def _validate_tensor(v: object) -> torch.Tensor | None:
    if v is None:
        return None
    if isinstance(v, torch.Tensor):
        return v
    return torch.tensor(v, dtype=torch.float32)


def _serialize_tensor(v: Any) -> list | None:
    if v is None:
        return None
    return v.tolist()


SerializableTensor = Annotated[
    Any,
    BeforeValidator(_validate_tensor),
    PlainSerializer(_serialize_tensor, return_type=list | None),
]


class MoveRecord(BaseModel):
    """A single move with optional MCTS policy."""

    move: Move
    player: Player
    moves_left_after: int
    policy: SerializableTensor = None  # (g, g) MCTS visit-count probs


class GameRecord(BaseModel):
    """Full record of a played game."""

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


@dataclass
class EvalResult:
    """Evaluation statistics for a player in an arena run."""

    wins: int
    losses: int
    total: int
    win_rate: float
    avg_moves_in_wins: float
    avg_moves_in_losses: float


class GameRecordCollection(BaseModel):
    """A collection of game records."""

    game_records: list[GameRecord]

    @property
    def scores(self) -> dict[str, int]:
        """Tally wins per player name."""
        scores: dict[str, int] = {}
        for r in self.game_records:
            winner_name = r.horizontal_player if r.winner == Player.HORIZONTAL else r.vertical_player
            scores[winner_name] = scores.get(winner_name, 0) + 1
        return scores

    def evaluate(self, player_name: str) -> EvalResult:
        """Compute evaluation stats for a given player name."""
        win_moves: list[int] = []
        loss_moves: list[int] = []

        for r in self.game_records:
            winner_name = (
                r.horizontal_player if r.winner == Player.HORIZONTAL else r.vertical_player
            )
            if winner_name == player_name:
                win_moves.append(r.num_moves)
            elif r.horizontal_player == player_name or r.vertical_player == player_name:
                loss_moves.append(r.num_moves)

        wins = len(win_moves)
        losses = len(loss_moves)
        total = wins + losses

        return EvalResult(
            wins=wins,
            losses=losses,
            total=total,
            win_rate=wins / total if total > 0 else 0.0,
            avg_moves_in_wins=sum(win_moves) / wins if wins > 0 else 0.0,
            avg_moves_in_losses=sum(loss_moves) / losses if losses > 0 else 0.0,
        )

    def is_better(self, player_name: str, win_threshold: float = 0.55) -> bool:
        """Determine if a player is better using win-rate + game-length heuristic.

        Returns True if:
        - win_rate >= win_threshold, OR
        - win_rate >= 0.5 AND avg_moves_in_losses > avg_moves_in_wins
          (the player is harder to beat than their opponent)
        """
        result = self.evaluate(player_name)
        if result.win_rate >= win_threshold:
            return True
        if (
            result.win_rate >= 0.5
            and result.avg_moves_in_losses > 0
            and result.avg_moves_in_losses > result.avg_moves_in_wins
        ):
            return True
        return False

    def __len__(self) -> int:
        return len(self.game_records)

    def __iter__(self):
        return iter(self.game_records)

    def __getitem__(self, idx: int) -> GameRecord:
        return self.game_records[idx]
