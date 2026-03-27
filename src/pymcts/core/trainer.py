"""Main training pipeline: self-play -> train -> evaluate loop."""

import json
import logging
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Callable

from pymcts.core.arena import Arena
from pymcts.core.base_game import BaseGame
from pymcts.core.base_neural_net import BaseNeuralNet
from pymcts.core.config import ArenaConfig, MCTSConfig, PathsConfig, TrainingConfig
from pymcts.core.data import Example, examples_from_records
from pymcts.core.players import GreedyMCTSPlayer
from pymcts.core.self_play import batched_self_play

logger = logging.getLogger("bridgit.core.trainer")


def _create_run_dir(paths: PathsConfig) -> Path:
    """Create a timestamped directory for this training run."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run_dir = paths.trainings / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _create_iter_dir(run_dir: Path, iteration: int) -> Path:
    """Create a directory for a single iteration."""
    iter_dir = run_dir / f"iteration_{iteration:03d}"
    iter_dir.mkdir(parents=True, exist_ok=True)
    return iter_dir


def train(
    game_factory: Callable[[], BaseGame],
    net: BaseNeuralNet,
    mcts_config: MCTSConfig,
    training_config: TrainingConfig,
    arena_config: ArenaConfig,
    paths_config: PathsConfig | None = None,
    game_type: str = "unknown",
    game_config: dict | None = None,
    verbose: bool = True,
):
    """Run the full AlphaZero training pipeline.

    Args:
        game_factory: Callable that creates a new game instance.
        net: Neural network to train.
        mcts_config: MCTS configuration.
        training_config: Training loop configuration.
        arena_config: Arena evaluation configuration.
        paths_config: File system paths. If None, uses defaults.
        game_type: Game type string for records.
        game_config: Game config dict for records.
        verbose: Whether to print progress.
    """
    paths = paths_config or PathsConfig()
    game_config = game_config or {}

    run_dir = _create_run_dir(paths)

    if verbose:
        print(f"Training run directory: {run_dir}")
        param_count = sum(p.numel() for p in net.parameters())
        print(f"Model parameters: {param_count:,}")
        print()

    # Replay buffer: stores examples from last N iterations
    replay_buffer: deque[list[Example]] = deque(
        maxlen=training_config.replay_buffer_size,
    )

    # game_factory wrapper for examples_from_records (takes config dict)
    def _game_from_config(cfg: dict) -> BaseGame:
        return game_factory()

    for iteration in range(1, training_config.num_iterations + 1):
        if verbose:
            print(f"{'=' * 60}")
            print(f"Iteration {iteration}/{training_config.num_iterations}")
            print(f"{'=' * 60}")

        iter_dir = _create_iter_dir(run_dir, iteration)

        # Save pre-training checkpoint
        pre_checkpoint = str(iter_dir / "pre_training.pt")
        net.save_checkpoint(pre_checkpoint)

        # 1. Self-play
        if verbose:
            print("\n[1/3] Self-play...")
        self_play_records = batched_self_play(
            net=net,
            game_factory=game_factory,
            mcts_config=mcts_config,
            num_games=training_config.num_self_play_games,
            batch_size=training_config.self_play_batch_size,
            temperature=1.0,
            verbose=verbose,
        )

        # Save self-play games
        self_play_path = iter_dir / "self_play_games.json"
        self_play_path.write_text(self_play_records.model_dump_json(indent=2))

        new_examples = examples_from_records(self_play_records, _game_from_config)
        replay_buffer.append(new_examples)

        all_examples = [ex for batch in replay_buffer for ex in batch]
        if verbose:
            print(f"  Self-play: {len(self_play_records)} games, "
                  f"{len(new_examples)} examples")
            print(f"  Replay buffer: {len(replay_buffer)} iterations, "
                  f"{len(all_examples)} examples")

        # 2. Train
        if verbose:
            print("\n[2/3] Training...")
        train_metrics = net.train_on_examples(
            all_examples,
            num_epochs=training_config.num_epochs,
            batch_size=training_config.batch_size,
            learning_rate=training_config.learning_rate,
            weight_decay=training_config.weight_decay,
        )

        # Save post-training checkpoint
        post_checkpoint = str(iter_dir / "post_training.pt")
        net.save_checkpoint(post_checkpoint)

        # 3. Evaluate
        if verbose:
            print("\n[3/3] Arena evaluation...")
        new_player = GreedyMCTSPlayer(net, mcts_config, name="new")

        prev_net = net.copy()
        prev_net.load_checkpoint(pre_checkpoint)
        prev_player = GreedyMCTSPlayer(prev_net, mcts_config, name="prev")

        eval_arena = Arena(
            new_player, prev_player, game_factory,
            game_type=game_type,
        )
        eval_records = eval_arena.play_games(
            arena_config.num_games, verbose=verbose,
            swap_players=arena_config.swap_players,
        )

        # Save eval games
        eval_path = iter_dir / "eval_games.json"
        eval_path.write_text(eval_records.model_dump_json(indent=2))

        result = eval_records.evaluate("new")
        accepted = eval_records.is_better("new", arena_config.threshold)

        if verbose:
            print(f"  New model: {result.wins} wins | Previous: {result.losses} wins | "
                  f"Win rate: {result.win_rate:.1%}")
            print(f"  Avg moves - wins: {result.avg_moves_in_wins:.1f}, "
                  f"losses: {result.avg_moves_in_losses:.1f}")

        if accepted:
            if verbose:
                print("  -> ACCEPTED: new model is better")
        else:
            if verbose:
                print("  -> REJECTED: reverting to pre-training weights")
            net.load_checkpoint(pre_checkpoint)

        # Save iteration data
        iteration_data = {
            "iteration": iteration,
            "training": {
                "num_examples": len(all_examples),
                "metrics": train_metrics,
            },
            "arena": {
                "new_wins": result.wins,
                "prev_wins": result.losses,
                "total_games": result.total,
                "win_rate": result.win_rate,
                "accepted": accepted,
            },
        }
        iter_data_path = iter_dir / "iteration_data.json"
        iter_data_path.write_text(json.dumps(iteration_data, indent=2))

        if verbose:
            print()

    if verbose:
        print("Training complete!")
        print(f"Run directory: {run_dir}")
