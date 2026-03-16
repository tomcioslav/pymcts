"""Main training pipeline: self-play → train → evaluate loop."""

import argparse
from collections import deque
from datetime import datetime
from pathlib import Path

import torch

from bridgit.ai.neural_net import BridgitNet, NetWrapper
from bridgit.data.converter import Example, examples_from_records
from bridgit.players.arena import Arena
from bridgit.players.players import MCTSPlayer, GreedyMCTSPlayer
from bridgit.config import Config


def _create_run_dir(config: Config) -> Path:
    """Create a timestamped directory for this training run."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run_dir = config.paths.trainings / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config_path = run_dir / "config.json"
    config_path.write_text(config.model_dump_json(indent=2))

    return run_dir


def _create_iter_dir(run_dir: Path, iteration: int) -> Path:
    """Create a directory for a single iteration."""
    iter_dir = run_dir / f"iteration_{iteration:03d}"
    iter_dir.mkdir(parents=True, exist_ok=True)
    return iter_dir


def train(config: Config, checkpoint_path: str | None = None):
    """Run the full AlphaZero training pipeline.

    Args:
        config: Full training configuration.
        checkpoint_path: Optional path to a checkpoint to resume from.
            The checkpoint's board and net config must match the config.
    """
    run_dir = _create_run_dir(config)

    print(f"Training Bridgit AI on {config.board.size}x{config.board.size} board")
    print(f"Run directory: {run_dir}")
    print()

    if checkpoint_path is not None:
        net_wrapper = NetWrapper(checkpoint_path)
        loaded_board = net_wrapper.model.board_config
        loaded_net = net_wrapper.model.net_config
        if loaded_board.model_dump() != config.board.model_dump():
            raise ValueError(
                f"Checkpoint board config {loaded_board.model_dump()} "
                f"doesn't match training config {config.board.model_dump()}"
            )
        if loaded_net.model_dump() != config.neural_net.model_dump():
            raise ValueError(
                f"Checkpoint net config {loaded_net.model_dump()} "
                f"doesn't match training config {config.neural_net.model_dump()}"
            )
        print(f"Loaded checkpoint: {checkpoint_path}")
    else:
        model = BridgitNet(config.board, config.neural_net)
        net_wrapper = NetWrapper(model)
        print("Starting from random initialization")

    print(f"Device: {net_wrapper.device}")
    print(f"Model parameters: {sum(p.numel() for p in net_wrapper.model.parameters()):,}")
    print()

    # Replay buffer: stores examples from last N iterations
    replay_buffer: deque[list[Example]] = deque(maxlen=config.training.replay_buffer_size)

    for iteration in range(1, config.training.num_iterations + 1):
        print(f"{'=' * 60}")
        print(f"Iteration {iteration}/{config.training.num_iterations}")
        print(f"{'=' * 60}")

        iter_dir = _create_iter_dir(run_dir, iteration)

        # Save pre-training checkpoint
        pre_checkpoint = iter_dir / "pre_training.pt"
        net_wrapper.save_checkpoint(str(pre_checkpoint))

        # 1. Self-play
        print("\n[1/3] Self-play...")
        self_play_player = MCTSPlayer(
            net_wrapper, config.mcts,
            temperature=1.0, name="self-play",
        )
        arena = Arena(self_play_player, self_play_player, config.board)
        self_play_records = arena.play_games(
            config.training.num_self_play_games, verbose=True,
        )

        # Save self-play games
        self_play_path = iter_dir / "self_play_games.json"
        self_play_path.write_text(self_play_records.model_dump_json(indent=2))

        new_examples = examples_from_records(self_play_records)
        replay_buffer.append(new_examples)

        all_examples = [ex for batch in replay_buffer for ex in batch]
        print(f"  Self-play: {len(self_play_records)} games, {len(new_examples)} examples")
        print(f"  Replay buffer: {len(replay_buffer)} iterations, {len(all_examples)} examples")

        # 2. Train
        print("\n[2/3] Training...")
        net_wrapper.train_on_examples(all_examples)

        # Save post-training checkpoint
        post_checkpoint = iter_dir / "post_training.pt"
        net_wrapper.save_checkpoint(str(post_checkpoint))

        # 3. Evaluate
        print("\n[3/3] Arena evaluation...")
        new_player = GreedyMCTSPlayer(net_wrapper, config.mcts, name="new")

        prev_net = NetWrapper(str(pre_checkpoint))
        prev_player = GreedyMCTSPlayer(prev_net, config.mcts, name="prev")

        eval_arena = Arena(new_player, prev_player, config.board)
        eval_records = eval_arena.play_games(
            config.arena.num_games, verbose=True,
            swap_players=config.arena.swap_players,
        )

        # Save eval games
        eval_path = iter_dir / "eval_games.json"
        eval_path.write_text(eval_records.model_dump_json(indent=2))

        result = eval_records.evaluate("new")
        accepted = eval_records.is_better("new", config.arena.threshold)

        print(f"  New model: {result.wins} wins | Previous: {result.losses} wins | "
              f"Win rate: {result.win_rate:.1%}")
        print(f"  Avg moves — wins: {result.avg_moves_in_wins:.1f}, "
              f"losses: {result.avg_moves_in_losses:.1f}")

        if accepted:
            print("  -> ACCEPTED: new model is better")
        else:
            print("  -> REJECTED: reverting to pre-training weights")
            net_wrapper.load_checkpoint(str(pre_checkpoint))

        print()

    print("Training complete!")
    print(f"Run directory: {run_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train Bridgit AI")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint to resume from")
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

    train(config, checkpoint_path=args.checkpoint)


if __name__ == "__main__":
    main()
