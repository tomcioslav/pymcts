"""Arena for pitting two players against each other with full game recording."""

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

from tqdm.auto import tqdm

from bridgit.players.players import BasePlayer
from bridgit.players.player_factory import create_player
from bridgit.game import Bridgit
from bridgit.schema import GameRecord, GameRecordCollection, MoveRecord
from bridgit.schema.player import Player
from bridgit.config import BoardConfig


def _play_one_game(player_specs: tuple, board_config_dict: dict) -> GameRecord:
    """Top-level function for process pool — plays a single game.

    Each worker reconstructs its own players from specs to avoid
    sharing state (especially neural net models) across processes.
    """
    board_config = BoardConfig(**board_config_dict)
    h_spec, v_spec = player_specs
    h_player = create_player(h_spec)
    v_player = create_player(v_spec)

    game = Bridgit(board_config)
    players = {
        Player.HORIZONTAL: h_player,
        Player.VERTICAL: v_player,
    }

    move_records: list[MoveRecord] = []

    while not game.game_over:
        current = game.current_player
        player = players[current]
        move = player.get_action(game)
        policy = player.last_policy
        game.make_move(move)

        move_records.append(MoveRecord(
            move=move,
            player=current,
            moves_left_after=game.moves_left_in_turn,
            policy=policy,
        ))

    return GameRecord(
        board_size=board_config.size,
        moves=move_records,
        winner=game.winner,
        horizontal_player=h_player.name,
        vertical_player=v_player.name,
    )


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

    def play_game(self, verbose: bool = False, swapped: bool = False) -> GameRecord:
        """Play a single game and return the full record.

        Args:
            verbose: Print move-by-move output.
            swapped: If True, swap player positions (H plays V and vice versa).
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

        if verbose:
            print(f"{h_player.name} (H) vs {v_player.name} (V)")

        move_records: list[MoveRecord] = []

        while not game.game_over:
            current = game.current_player
            player = players[current]
            move = player.get_action(game)
            policy = player.last_policy
            game.make_move(move)

            move_records.append(MoveRecord(
                move=move,
                player=current,
                moves_left_after=game.moves_left_in_turn,
                policy=policy,
            ))

            if verbose:
                print(f"  Move {game.move_count}: ({move.row},{move.col}) by {player.name}")

        if verbose:
            winner_player = players[game.winner]
            print(f"  Winner: {winner_player.name}\n")

        return GameRecord(
            board_size=self.board_config.size,
            moves=move_records,
            winner=game.winner,
            horizontal_player=h_player.name,
            vertical_player=v_player.name,
        )

    def play_games(
        self,
        num_games: int,
        verbose: bool = False,
        num_workers: int = 1,
        swap_players: bool = False,
    ) -> GameRecordCollection:
        """Play multiple games and return all records.

        Args:
            num_games: Total number of games to play.
            verbose: Print progress.
            num_workers: Number of parallel processes (1 = sequential).
            swap_players: If True, play half the games with players swapped
                (so each player gets to play both H and V positions).

        Returns:
            GameRecordCollection with all game records.
        """
        if num_workers <= 1:
            records = self._play_sequential(num_games, verbose, swap_players)
        else:
            records = self._play_parallel(num_games, num_workers, verbose, swap_players)
        return GameRecordCollection(game_records=records)

    def _play_sequential(
        self, num_games: int, verbose: bool, swap_players: bool
    ) -> list[GameRecord]:
        records: list[GameRecord] = []
        h_name = self.horizontal_player.name
        v_name = self.vertical_player.name
        wins: dict[str, int] = {h_name: 0, v_name: 0}

        half = num_games // 2 if swap_players else num_games

        iterator = range(num_games)
        if verbose:
            iterator = tqdm(iterator, desc=f"{h_name} vs {v_name}")

        for i in iterator:
            swapped = swap_players and i >= half
            record = self.play_game(verbose=False, swapped=swapped)
            records.append(record)

            winner_name = (
                record.horizontal_player if record.winner == Player.HORIZONTAL
                else record.vertical_player
            )
            if winner_name in wins:
                wins[winner_name] += 1

            if verbose:
                print(
                    f"  Game {i + 1}: H={record.horizontal_player}, "
                    f"V={record.vertical_player} → {winner_name} wins "
                    f"({record.num_moves} moves)"
                )
                iterator.set_postfix_str(
                    f"{h_name}: {wins[h_name]}, {v_name}: {wins[v_name]}"
                )

        if verbose:
            print(f"Result: {h_name}: {wins[h_name]}, {v_name}: {wins[v_name]}")

        return records

    def _play_parallel(
        self, num_games: int, num_workers: int, verbose: bool, swap_players: bool
    ) -> list[GameRecord]:
        h_name = self.horizontal_player.name
        v_name = self.vertical_player.name
        wins: dict[str, int] = {h_name: 0, v_name: 0}

        h_spec = self.horizontal_player.to_spec()
        v_spec = self.vertical_player.to_spec()
        board_dict = self.board_config.model_dump()

        half = num_games // 2 if swap_players else num_games

        pbar = None
        if verbose:
            pbar = tqdm(total=num_games, desc=f"{h_name} vs {v_name}")

        records: list[GameRecord] = []
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx) as executor:
            futures = []
            for i in range(num_games):
                if swap_players and i >= half:
                    specs = (v_spec, h_spec)
                else:
                    specs = (h_spec, v_spec)
                futures.append(
                    executor.submit(_play_one_game, specs, board_dict)
                )
            for future in as_completed(futures):
                record = future.result()
                records.append(record)
                winner_name = (
                    record.horizontal_player if record.winner == Player.HORIZONTAL
                    else record.vertical_player
                )
                if winner_name in wins:
                    wins[winner_name] += 1
                if pbar is not None:
                    print(
                        f"  Game {len(records)}: H={record.horizontal_player}, "
                        f"V={record.vertical_player} → {winner_name} wins "
                        f"({record.num_moves} moves)"
                    )
                    pbar.update(1)
                    pbar.set_postfix_str(
                        f"{h_name}: {wins[h_name]}, {v_name}: {wins[v_name]}"
                    )

        if pbar is not None:
            pbar.close()
            print(f"Result: {h_name}: {wins[h_name]}, {v_name}: {wins[v_name]}")

        return records
