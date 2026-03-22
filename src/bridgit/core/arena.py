"""Arena for pitting two players against each other."""

import logging
from typing import Callable

from tqdm.auto import tqdm

from bridgit.core.base_game import BaseGame
from bridgit.core.players import BasePlayer
from bridgit.core.game_record import GameRecord, GameRecordCollection, MoveRecord

logger = logging.getLogger("core.arena")


class Arena:
    def __init__(
        self,
        player_a: BasePlayer,
        player_b: BasePlayer,
        game_factory: Callable[[], BaseGame],
        game_type: str = "unknown",
        game_config: dict | None = None,
    ):
        self.player_a = player_a
        self.player_b = player_b
        self.game_factory = game_factory
        self.game_type = game_type
        self.game_config = game_config or {}

    def play_game(self, swapped: bool = False) -> GameRecord:
        game = self.game_factory()

        if swapped:
            players = {0: self.player_b, 1: self.player_a}
            names = [self.player_b.name, self.player_a.name]
        else:
            players = {0: self.player_a, 1: self.player_b}
            names = [self.player_a.name, self.player_b.name]

        move_records: list[MoveRecord] = []
        max_moves = game.action_space_size * 4

        while not game.is_over:
            current = game.current_player
            player = players[current]
            action = player.get_action(game)
            policy = player.last_policy

            game.make_action(action)
            move_records.append(MoveRecord(
                action=action,
                player=current,
                policy=policy,
            ))

            if len(move_records) > max_moves:
                raise RuntimeError(f"Game exceeded {max_moves} moves — likely stuck")

        return GameRecord(
            game_type=self.game_type,
            game_config=self.game_config,
            moves=move_records,
            winner=game.winner,
            player_names=names,
        )

    def play_games(
        self,
        num_games: int,
        verbose: bool = False,
        swap_players: bool = False,
    ) -> GameRecordCollection:
        records: list[GameRecord] = []
        half = num_games // 2 if swap_players else num_games

        iterator = range(num_games)
        if verbose:
            iterator = tqdm(iterator, desc=f"{self.player_a.name} vs {self.player_b.name}")

        for i in iterator:
            swapped = swap_players and i >= half
            record = self.play_game(swapped=swapped)
            records.append(record)

        return GameRecordCollection(game_records=records)
