"""Arena for pitting two players against each other with full game recording."""

import json
import logging
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from bridgit.players.players import BasePlayer
from bridgit.game import Bridgit
from bridgit.schema import GameRecord, GameRecordCollection, MoveRecord
from bridgit.schema.player import Player
from bridgit.config import BoardConfig

logger = logging.getLogger("bridgit.arena")


class Arena:
    """Play games between two players and record results."""

    def __init__(
        self,
        horizontal_player: BasePlayer,
        vertical_player: BasePlayer,
        board_config: BoardConfig,
    ):
        self.horizontal_player = horizontal_player
        self.vertical_player = vertical_player
        self.board_config = board_config

    def play_game(
        self,
        verbose: bool = False,
        swapped: bool = False,
        trace_path: str | Path | None = None,
    ) -> GameRecord:
        """Play a single game and return the full record.

        Args:
            verbose: Print move-by-move output.
            swapped: If True, swap player positions (H plays V and vice versa).
            trace_path: If set, save a partial GameRecord to this file after
                each move (for debugging stuck games).
        """
        game = Bridgit(self.board_config)

        if swapped:
            h_player = self.vertical_player
            v_player = self.horizontal_player
        else:
            h_player = self.horizontal_player
            v_player = self.vertical_player

        players = {
            Player.HORIZONTAL: h_player,
            Player.VERTICAL: v_player,
        }

        logger.info("Game start: %s (H) vs %s (V), swapped=%s",
                     h_player.name, v_player.name, swapped)

        if verbose:
            print(f"{h_player.name} (H) vs {v_player.name} (V)")

        if trace_path is not None:
            trace_path = Path(trace_path)
            trace_path.parent.mkdir(parents=True, exist_ok=True)

        move_records: list[MoveRecord] = []
        max_moves = (self.board_config.grid_size ** 2) * 2

        while not game.game_over:
            current = game.current_player
            player = players[current]
            move = player.get_action(game)
            policy = player.last_policy
            if not game.make_move(move):
                raise RuntimeError(
                    f"Invalid move ({move.row},{move.col}) by {player.name} "
                    f"at move_count={game.move_count}"
                )

            logger.debug("Move %d: (%d,%d) by %s (%s)",
                         game.move_count, move.row, move.col,
                         player.name, current.name)

            move_records.append(MoveRecord(
                move=move,
                player=current,
                moves_left_after=game.moves_left_in_turn,
                policy=policy,
            ))

            if verbose:
                print(f"  Move {game.move_count}: ({move.row},{move.col}) by {player.name}")

            if trace_path is not None:
                trace_data = {
                    "status": "in_progress",
                    "board_size": self.board_config.size,
                    "horizontal_player": h_player.name,
                    "vertical_player": v_player.name,
                    "current_player": current.name,
                    "move_count": game.move_count,
                    "game_over": game.game_over,
                    "moves": [
                        {
                            "move": {"row": m.move.row, "col": m.move.col},
                            "player": m.player.name,
                            "moves_left_after": m.moves_left_after,
                        }
                        for m in move_records
                    ],
                }
                trace_path.write_text(json.dumps(trace_data, indent=2))

            if len(move_records) > max_moves:
                raise RuntimeError(
                    f"Game exceeded {max_moves} moves — likely stuck"
                )

        winner_player = players[game.winner]
        logger.info("Game over: winner=%s (%s), %d moves",
                     winner_player.name, game.winner.name,
                     game.move_count)

        if verbose:
            print(f"  Winner: {winner_player.name}\n")

        record = GameRecord(
            board_size=self.board_config.size,
            moves=move_records,
            winner=game.winner,
            horizontal_player=h_player.name,
            vertical_player=v_player.name,
        )

        if trace_path is not None:
            trace_data = {
                "status": "finished",
                "board_size": self.board_config.size,
                "horizontal_player": h_player.name,
                "vertical_player": v_player.name,
                "winner": game.winner.name,
                "move_count": game.move_count,
                "game_over": True,
                "moves": [
                    {
                        "move": {"row": m.move.row, "col": m.move.col},
                        "player": m.player.name,
                        "moves_left_after": m.moves_left_after,
                    }
                    for m in move_records
                ],
            }
            trace_path.write_text(json.dumps(trace_data, indent=2))

        return record

    def play_games(
        self,
        num_games: int,
        verbose: bool = False,
        swap_players: bool = False,
        seed: int | None = None,
        trace_dir: str | Path | None = None,
    ) -> GameRecordCollection:
        """Play multiple games and return all records.

        Args:
            num_games: Total number of games to play.
            verbose: Print progress.
            swap_players: If True, play half the games with players swapped
                (so each player gets to play both H and V positions).
            seed: Random seed for reproducibility.
            trace_dir: If set, save per-move trace files for each game
                (game_000.json, game_001.json, ...) into this directory.

        Returns:
            GameRecordCollection with all game records.
        """
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

        if trace_dir is not None:
            trace_dir = Path(trace_dir)
            trace_dir.mkdir(parents=True, exist_ok=True)

        records: list[GameRecord] = []
        h_name = self.horizontal_player.name
        v_name = self.vertical_player.name
        wins: dict[str, int] = {h_name: 0, v_name: 0}

        logger.info("Arena: %d games, swap=%s, %s vs %s",
                     num_games, swap_players, h_name, v_name)

        half = num_games // 2 if swap_players else num_games

        iterator = range(num_games)
        if verbose:
            iterator = tqdm(iterator, desc=f"{h_name} vs {v_name}")

        for i in iterator:
            swapped = swap_players and i >= half
            trace_path = trace_dir / f"game_{i:03d}.json" if trace_dir else None
            record = self.play_game(
                verbose=False, swapped=swapped, trace_path=trace_path,
            )
            records.append(record)

            winner_name = (
                record.horizontal_player if record.winner == Player.HORIZONTAL
                else record.vertical_player
            )
            if winner_name in wins:
                wins[winner_name] += 1

            if verbose:
                iterator.set_postfix_str(
                    f"{h_name}: {wins[h_name]}, {v_name}: {wins[v_name]}"
                )

        if verbose:
            print(f"Result: {h_name}: {wins[h_name]}, {v_name}: {wins[v_name]}")

        return GameRecordCollection(game_records=records)
