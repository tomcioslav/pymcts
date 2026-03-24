"""Generic game recording types."""

from dataclasses import dataclass
from typing import Annotated, Any

import torch
from pydantic import BaseModel, BeforeValidator, PlainSerializer


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
    """A single move in a game."""
    action: int
    player: int
    policy: SerializableTensor = None  # 1D, length action_space_size


class GameRecord(BaseModel):
    """Full record of a played game."""
    game_type: str
    game_config: dict
    moves: list[MoveRecord]
    winner: int | None
    player_names: list[str]

    @property
    def num_moves(self) -> int:
        return len(self.moves)

    def winner_name(self) -> str | None:
        if self.winner is None:
            return None
        return self.player_names[self.winner]

    def summary(self) -> str:
        names = " vs ".join(self.player_names)
        winner = self.winner_name() or "draw"
        return f"{names}: {winner} wins in {self.num_moves} moves"


@dataclass
class EvalResult:
    """Evaluation statistics for a player in an arena run."""
    wins: int
    losses: int
    draws: int
    total: int
    win_rate: float
    avg_moves_in_wins: float
    avg_moves_in_losses: float


class GameRecordCollection(BaseModel):
    """A collection of game records."""
    game_records: list[GameRecord]

    @property
    def scores(self) -> dict[str, int]:
        scores: dict[str, int] = {}
        for r in self.game_records:
            name = r.winner_name()
            if name:
                scores[name] = scores.get(name, 0) + 1
        return scores

    def evaluate(self, player_name: str) -> EvalResult:
        win_moves: list[int] = []
        loss_moves: list[int] = []
        draw_count = 0
        for r in self.game_records:
            if player_name not in r.player_names:
                continue
            wn = r.winner_name()
            if wn is None:
                draw_count += 1
            elif wn == player_name:
                win_moves.append(r.num_moves)
            else:
                loss_moves.append(r.num_moves)
        wins = len(win_moves)
        losses = len(loss_moves)
        total = wins + losses + draw_count
        return EvalResult(
            wins=wins, losses=losses, draws=draw_count, total=total,
            win_rate=wins / total if total > 0 else 0.0,
            avg_moves_in_wins=sum(win_moves) / wins if wins > 0 else 0.0,
            avg_moves_in_losses=sum(loss_moves) / losses if losses > 0 else 0.0,
        )

    def is_better(self, player_name: str, win_threshold: float = 0.55) -> bool:
        result = self.evaluate(player_name)
        if result.win_rate >= win_threshold:
            return True
        if (result.win_rate >= 0.5
                and result.avg_moves_in_losses > 0
                and result.avg_moves_in_losses > result.avg_moves_in_wins):
            return True
        return False

    def __len__(self) -> int:
        return len(self.game_records)

    def __iter__(self):
        return iter(self.game_records)

    def __getitem__(self, idx: int) -> GameRecord:
        return self.game_records[idx]
