"""Stress test: ensure all games complete within a reasonable number of moves."""

import signal

import pytest

from bridgit.ai.neural_net import BridgitNet, NetWrapper
from bridgit.config import BoardConfig, MCTSConfig, NeuralNetConfig
from bridgit.game import Bridgit
from bridgit.players.players import MCTSPlayer


class TimeoutError(Exception):
    pass


def _timeout_handler(signum, frame):
    raise TimeoutError("Game timed out")


def _play_one_game(board_config, mcts_config, net_wrapper, game_idx):
    """Play a single game between two MCTS players, return move count or raise."""
    game = Bridgit(board_config)
    player = MCTSPlayer(net_wrapper, mcts_config, temperature=1.0, name="test")

    max_moves = board_config.grid_size ** 2

    move_count = 0
    while not game.game_over:
        move = player.get_action(game)
        success = game.make_move(move)
        if not success:
            pytest.fail(
                f"Game {game_idx}: invalid move ({move.row},{move.col}) "
                f"by {game.current_player.name} at move_count={game.move_count}"
            )
        move_count += 1
        if move_count > max_moves:
            pytest.fail(
                f"Game {game_idx}: exceeded {max_moves} moves — stuck"
            )

    return move_count


@pytest.mark.parametrize("game_idx", range(200))
def test_game_completes(game_idx):
    """Each game must complete within timeout and max moves."""
    board_config = BoardConfig(size=4)
    mcts_config = MCTSConfig(
        num_simulations=1000,
        c_puct=1.5,
        dirichlet_alpha=1.0,
        dirichlet_epsilon=0.25,
    )
    net_config = NeuralNetConfig(num_channels=32, num_res_blocks=2)
    net = BridgitNet(board_config, net_config)
    net_wrapper = NetWrapper(net)

    # 30 second timeout per game
    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(30)
    try:
        moves = _play_one_game(board_config, mcts_config, net_wrapper, game_idx)
    except TimeoutError:
        pytest.fail(f"Game {game_idx}: timed out after 30 seconds")
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
