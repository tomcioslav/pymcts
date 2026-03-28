"""Main training pipeline: self-play -> train -> evaluate loop."""

import json
import logging
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Callable

from pymcts.core.arena import batched_arena
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

    # Save run config so models can be loaded later
    run_config = {
        "mcts_config": mcts_config.model_dump(),
        "training_config": training_config.model_dump(),
        "arena_config": arena_config.model_dump(),
        "game_type": game_type,
        "game_config": game_config,
    }
    run_config_path = run_dir / "run_config.json"
    run_config_path.write_text(json.dumps(run_config, indent=2))

    if verbose:
        print(f"Training run directory: {run_dir}")
        param_count = sum(p.numel() for p in net.parameters())
        print(f"Model parameters: {param_count:,}")
        print()

    # Replay buffer: stores examples from last N iterations
    replay_buffer: deque[list[Example]] = deque(
        maxlen=training_config.replay_buffer_size,
    )

    # Best accepted model checkpoints (most recent last), up to 10
    best_checkpoints: deque[tuple[int, str]] = deque(maxlen=10)

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
            verbose=verbose,
        )

        # Save post-training checkpoint
        post_checkpoint = str(iter_dir / "post_training.pt")
        net.save_checkpoint(post_checkpoint)

        # 3. Evaluate — new model vs previous
        if verbose:
            print("\n[3/3] Arena evaluation...")
            print("  vs previous model:")

        new_player = GreedyMCTSPlayer(net, mcts_config, name="new")

        prev_net = net.copy()
        prev_net.load_checkpoint(pre_checkpoint)
        prev_player = GreedyMCTSPlayer(prev_net, mcts_config, name="prev")

        eval_records = batched_arena(
            player_a=new_player,
            player_b=prev_player,
            game_factory=game_factory,
            num_games=arena_config.num_games,
            batch_size=arena_config.num_games,
            swap_players=arena_config.swap_players,
            game_type=game_type,
            verbose=verbose,
        )

        result = eval_records.evaluate("new")
        accepted = eval_records.is_better("new", arena_config.threshold)

        if verbose:
            print(f"  New: {result.wins} wins | Prev: {result.losses} wins | "
                  f"Win rate: {result.win_rate:.1%}")

        # Evaluate vs past best models
        historical_results = {}
        if best_checkpoints:
            if verbose:
                print(f"\n  vs past best models ({len(best_checkpoints)}):")
            for past_iter, past_path in best_checkpoints:
                past_net = net.copy()
                past_net.load_checkpoint(past_path)
                past_player = GreedyMCTSPlayer(past_net, mcts_config, name=f"iter_{past_iter:03d}")

                past_records = batched_arena(
                    player_a=new_player,
                    player_b=past_player,
                    game_factory=game_factory,
                    num_games=arena_config.num_games,
                    batch_size=arena_config.num_games,
                    swap_players=arena_config.swap_players,
                    game_type=game_type,
                    verbose=verbose,
                )

                past_result = past_records.evaluate("new")
                historical_results[past_iter] = {
                    "wins": past_result.wins,
                    "losses": past_result.losses,
                    "win_rate": past_result.win_rate,
                }
                if verbose:
                    print(f"    vs iter {past_iter:03d}: "
                          f"{past_result.wins}W/{past_result.losses}L "
                          f"({past_result.win_rate:.0%})")

        if accepted:
            if verbose:
                print("\n  -> ACCEPTED: new model is better")
            best_checkpoints.append((iteration, post_checkpoint))
        else:
            if verbose:
                print("\n  -> REJECTED: reverting to pre-training weights")
            net.load_checkpoint(pre_checkpoint)

        # Save eval games
        eval_path = iter_dir / "eval_games.json"
        eval_path.write_text(eval_records.model_dump_json(indent=2))

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
            "historical_arena": historical_results,
        }
        iter_data_path = iter_dir / "iteration_data.json"
        iter_data_path.write_text(json.dumps(iteration_data, indent=2))

        if verbose:
            print()

    if verbose:
        print("Training complete!")
        print(f"Run directory: {run_dir}")
