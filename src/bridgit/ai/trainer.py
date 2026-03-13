"""Main training pipeline: self-play → train → evaluate loop."""

import argparse
from collections import deque

import torch

from bridgit.ai.neural_net import BridgitNet, NetWrapper
from bridgit.players.arena import Arena
from bridgit.players.players import MCTSPlayer, GreedyMCTSPlayer
from bridgit.schema import GameRecord
from bridgit.config import Config


# Training example: (state_tensor, target_policy, target_value)
Example = tuple[torch.Tensor, torch.Tensor, float]


def examples_from_records(records: list[GameRecord]) -> list[Example]:
    """Extract training examples from game records.

    Only includes moves that have an MCTS policy attached.
    """
    from bridgit.game import Bridgit
    from bridgit.config import BoardConfig
    from bridgit.schema.player import Player

    examples: list[Example] = []
    for record in records:
        board_config = BoardConfig(size=record.board_size)
        game = Bridgit(board_config)

        for move_rec in record.moves:
            if move_rec.policy is not None:
                state_tensor = game.to_tensor()
                player_value = game.current_player.value
                winner_value = record.winner.value
                value = 1.0 if player_value == winner_value else -1.0
                examples.append((state_tensor, move_rec.policy, value))
            game.make_move(move_rec.move)

    return examples


def train(config: Config):
    """Run the full AlphaZero training pipeline."""
    print(f"Training Bridgit AI on {config.board.size}x{config.board.size} board")
    print(f"Config: {config}")
    print()

    model = BridgitNet(config.board, config.neural_net)
    net_wrapper = NetWrapper(model)
    print(f"Device: {net_wrapper.device}")
    print(f"Model parameters: {sum(p.numel() for p in net_wrapper.model.parameters()):,}")
    print()

    # Replay buffer: stores examples from last N iterations
    replay_buffer: deque[list[Example]] = deque(maxlen=config.training.replay_buffer_size)

    best_checkpoint = config.paths.checkpoints / "best.pt"

    for iteration in range(1, config.training.num_iterations + 1):
        print(f"{'=' * 60}")
        print(f"Iteration {iteration}/{config.training.num_iterations}")
        print(f"{'=' * 60}")

        # 1. Self-play
        print("\n[1/3] Self-play...")
        self_play_player = MCTSPlayer(
            net_wrapper, config.mcts,
            temperature=1.0, name="self-play",
        )
        arena = Arena(self_play_player, self_play_player, config.board)
        records = arena.play_games(config.training.num_self_play_games)

        new_examples = examples_from_records(records)
        replay_buffer.append(new_examples)

        all_examples = [ex for batch in replay_buffer for ex in batch]
        print(f"  Self-play: {len(records)} games, {len(new_examples)} examples")
        print(f"  Replay buffer: {len(replay_buffer)} iterations, {len(all_examples)} examples")

        # 2. Train
        print("\n[2/3] Training...")
        temp_checkpoint = config.paths.checkpoints / "temp.pt"
        net_wrapper.save_checkpoint(str(temp_checkpoint))

        net_wrapper.train_on_examples(all_examples)

        # 3. Evaluate
        print("\n[3/3] Arena evaluation...")
        new_player = GreedyMCTSPlayer(net_wrapper, config.mcts, name="new")

        prev_model = BridgitNet(config.board, config.neural_net)
        prev_net = NetWrapper(prev_model)
        if best_checkpoint.exists():
            prev_net.load_checkpoint(str(best_checkpoint))
        else:
            prev_net.load_checkpoint(str(temp_checkpoint))

        prev_player = GreedyMCTSPlayer(prev_net, config.mcts, name="prev")

        eval_arena = Arena(new_player, prev_player, config.board)
        eval_records = eval_arena.play_games(config.arena.num_games)
        scores = Arena.score(eval_records)
        new_wins = scores.get("new", 0)
        prev_wins = scores.get("prev", 0)
        total = new_wins + prev_wins
        win_rate = new_wins / total if total > 0 else 0

        print(f"  New model: {new_wins} wins | Previous: {prev_wins} wins | "
              f"Win rate: {win_rate:.1%}")

        if win_rate >= config.arena.threshold:
            print("  -> ACCEPTED: new model is better, saving checkpoint")
            net_wrapper.save_checkpoint(str(best_checkpoint))
        else:
            print("  -> REJECTED: keeping previous model")
            net_wrapper.load_checkpoint(str(temp_checkpoint))

        if temp_checkpoint.exists():
            temp_checkpoint.unlink()

        iter_checkpoint = config.paths.checkpoints / f"iter_{iteration:04d}.pt"
        net_wrapper.save_checkpoint(str(iter_checkpoint))
        print(f"  Saved iteration checkpoint: {iter_checkpoint}")
        print()

    print("Training complete!")
    print(f"Best model: {best_checkpoint}")


def main():
    parser = argparse.ArgumentParser(description="Train Bridgit AI")
    parser.add_argument("--iterations", type=int, default=None,
                        help="Number of training iterations")
    parser.add_argument("--games", type=int, default=None,
                        help="Self-play games per iteration")
    parser.add_argument("--sims", type=int, default=None,
                        help="MCTS simulations per move")
    parser.add_argument("--arena-games", type=int, default=None,
                        help="Arena games for evaluation")
    args = parser.parse_args()

    config = Config()
    if args.iterations is not None:
        config.training.num_iterations = args.iterations
    if args.games is not None:
        config.training.num_self_play_games = args.games
    if args.sims is not None:
        config.mcts.num_simulations = args.sims
    if args.arena_games is not None:
        config.arena.num_games = args.arena_games

    train(config)


if __name__ == "__main__":
    main()
