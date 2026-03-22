"""Training data extraction from game records."""

from typing import Callable

import torch

from bridgit.core.base_game import BaseGame, GameState
from bridgit.core.game_record import GameRecordCollection


Example = tuple[GameState, torch.Tensor, float]


def examples_from_records(
    collection: GameRecordCollection,
    game_factory: Callable[[dict], BaseGame],
) -> list[Example]:
    """Replay games to extract (state, policy, value) training examples.

    Args:
        collection: Game records to replay.
        game_factory: Creates a new game from game_config dict.
    """
    examples: list[Example] = []
    for record in collection.game_records:
        game = game_factory(record.game_config)
        for move_rec in record.moves:
            if move_rec.policy is not None:
                state = game.get_state()
                value = 1.0 if game.current_player == record.winner else -1.0
                examples.append((state, move_rec.policy, value))
            game.make_action(move_rec.action)
    return examples
