"""Test for MCTS solved value bug with 1-2-2 turn structure."""

from bridgit import Bridgit, Player
from bridgit.schema import Move
from bridgit.ai import BridgitNet, NetWrapper, MCTS
from bridgit.config import BoardConfig, MCTSConfig, NeuralNetConfig


def test_mcts_solved_value_vertical_winning():
    """Reproduce bug: MCTS says H wins but V should be winning.

    Board state after 11 moves (H's turn, 2 moves left):
    V has col 3 nearly connected: (1,3), (3,3), (5,3) — needs (7,3)
    Even if H blocks (7,3), V has two alternative winning paths.
    So V should be winning — root.solved_value should NOT be +1.
    """
    board_config = BoardConfig(size=4)
    game = Bridgit(board_config)

    moves = [
        Move(row=1, col=1),  # 1. H
        Move(row=1, col=3),  # 2. V
        Move(row=3, col=3),  # 3. V
        Move(row=1, col=5),  # 4. H
        Move(row=7, col=1),  # 5. H
        Move(row=1, col=7),  # 6. V
        Move(row=5, col=3),  # 7. V
        Move(row=5, col=1),  # 8. H
        Move(row=7, col=5),  # 9. H
        Move(row=7, col=7),  # 10. V
        Move(row=4, col=4),  # 11. V
    ]
    for m in moves:
        assert game.make_move(m), f"Failed to make move {m}"

    print(f"Current player: {game.current_player.name}")
    print(f"Moves left in turn: {game.moves_left_in_turn}")
    print(f"Move count: {game.move_count}")

    # Verify it's H's turn with 2 moves
    assert game.current_player == Player.HORIZONTAL
    assert game.moves_left_in_turn == 2

    # Manually verify V wins: even if H blocks (7,3), V wins
    # Path 1: H blocks (7,3), then V plays (5,5) and (6,6) — but let's check
    test_game = game.copy()

    # H blocks column 3
    test_game.make_move(Move(row=7, col=3))  # H move 1 of 2
    print(f"\nAfter H blocks (7,3): player={test_game.current_player.name}, "
          f"moves_left={test_game.moves_left_in_turn}")

    # H makes another move (anything)
    available = test_game.get_available_moves()
    test_game.make_move(available[0])  # H move 2 of 2
    print(f"After H's 2nd move: player={test_game.current_player.name}, "
          f"moves_left={test_game.moves_left_in_turn}")

    # Now V's turn with 2 moves — check if V can win
    # V needs to connect top to bottom via columns 5 or 7
    print(f"\nAvailable moves for V: {test_game.get_available_moves()}")

    # Run MCTS
    mcts_config = MCTSConfig(num_simulations=5000, solve_terminal=True)
    net_config = NeuralNetConfig(num_channels=32, num_res_blocks=2)
    net = BridgitNet(board_config, net_config)
    wrapper = NetWrapper(net)
    mcts = MCTS(wrapper, mcts_config)

    root = mcts._search(game, verbose=False)

    print(f"\nRoot solved_value: {root.solved_value}")
    print(f"Root q_value: {root.q_value:.3f}")
    print(f"Root visits: {root.visit_count}")

    # Print children info
    for move, child in root.children.items():
        print(f"  Child ({move.row},{move.col}): solved={child.solved_value}, "
              f"visits={child.visit_count}, q={child.q_value:.3f}, "
              f"game_over={child.game.game_over}, "
              f"winner={child.game.winner}")

    # The root should NOT be solved as a win for H (current player)
    # V has a winning strategy here
    if root.solved_value == 1.0:
        print("\nBUG: Root is solved as WIN for H, but V should be winning!")
    elif root.solved_value == -1.0:
        print("\nCORRECT: Root is solved as LOSS for H (V is winning)")
    else:
        print(f"\nRoot not fully solved (solved_value={root.solved_value})")


if __name__ == "__main__":
    test_mcts_solved_value_vertical_winning()
