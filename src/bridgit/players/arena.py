"""Arena for pitting two players against each other with full game recording."""

from bridgit.players.players import BasePlayer
from bridgit.game import Bridgit
from bridgit.schema import GameRecord, MoveRecord
from bridgit.schema.player import Player
from bridgit.config import BoardConfig


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

    def play_game(self, verbose: bool = False) -> GameRecord:
        """Play a single game and return the full record."""
        game = Bridgit(self.board_config)
        players = {
            Player.HORIZONTAL: self.horizontal_player,
            Player.VERTICAL: self.vertical_player,
        }

        if verbose:
            print(f"{self.horizontal_player.name} (H) vs {self.vertical_player.name} (V)")

        move_records: list[MoveRecord] = []

        while not game.game_over:
            current = game.current_player
            player = players[current]
            move = player.get_action(game)
            game.make_move(move)

            move_records.append(MoveRecord(
                move=move,
                player=current,
                moves_left_after=game.moves_left_in_turn,
                policy=player.last_policy,
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
            horizontal_player=self.horizontal_player.name,
            vertical_player=self.vertical_player.name,
        )

    def play_games(self, num_games: int, verbose: bool = False) -> list[GameRecord]:
        """Play multiple games and return all records.

        Args:
            num_games: Number of games to play.
            verbose: Print progress.

        Returns:
            List of GameRecord objects.
        """
        records: list[GameRecord] = []
        h_name = self.horizontal_player.name
        v_name = self.vertical_player.name
        h_wins = 0

        iterator = range(num_games)
        if verbose:
            from tqdm import tqdm
            iterator = tqdm(iterator, desc=f"{h_name} vs {v_name}")

        for i in iterator:
            record = self.play_game(verbose=False)
            records.append(record)
            if record.winner == Player.HORIZONTAL:
                h_wins += 1

            if verbose:
                v_wins = len(records) - h_wins
                iterator.set_postfix_str(f"{h_name}: {h_wins}, {v_name}: {v_wins}")

        if verbose:
            v_wins = len(records) - h_wins
            print(f"Result: {h_name}: {h_wins}, {v_name}: {v_wins}")

        return records

    @staticmethod
    def score(records: list[GameRecord]) -> dict[str, int]:
        """Tally wins per player name from a list of game records."""
        scores: dict[str, int] = {}
        for r in records:
            winner_name = r.horizontal_player if r.winner == Player.HORIZONTAL else r.vertical_player
            scores[winner_name] = scores.get(winner_name, 0) + 1
        return scores
