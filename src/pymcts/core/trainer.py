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
from pymcts.core.config import ArenaConfig, EloArenaConfig, MCTSConfig, PathsConfig, TrainingConfig
from pymcts.core.data import Example, examples_from_records
from pymcts.core.players import BasePlayer, GreedyMCTSPlayer, MCTSPlayer, RandomPlayer
from pymcts.elo.config import EloRating, MatchResult, TournamentResult
from pymcts.elo.rating import compute_elo_against_pool, compute_elo_ratings
from pymcts.core.self_play import batched_self_play

logger = logging.getLogger("pymcts.core.trainer")


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


def _save_run_config(run_dir, net, mcts_config, training_config, arena, game_type, game_config):
    """Save run configuration so models can be loaded later."""
    net_class = type(net)
    run_config = {
        "net_class": f"{net_class.__module__}.{net_class.__qualname__}",
        "mcts_config": mcts_config.model_dump(),
        "training_config": training_config.model_dump(),
        "arena": arena.model_dump(),
        "arena_type": type(arena).__name__,
        "game_type": game_type,
        "game_config": game_config,
    }
    (run_dir / "run_config.json").write_text(json.dumps(run_config, indent=2))


def _run_self_play(net, game_factory, mcts_config, training_config, verbose):
    """Generate games via self-play with the current network."""
    return batched_self_play(
        net=net,
        game_factory=game_factory,
        mcts_config=mcts_config,
        num_games=training_config.num_self_play_games,
        batch_size=training_config.self_play_batch_size,
        temperature=1.0,
        verbose=verbose,
    )


def _train_network(net, all_examples, training_config, verbose):
    """Train the network on accumulated examples."""
    return net.train_on_examples(
        all_examples,
        num_epochs=training_config.num_epochs,
        batch_size=training_config.batch_size,
        learning_rate=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
        verbose=verbose,
    )


def _evaluate_vs_previous(net, pre_checkpoint, mcts_config, arena, game_factory, game_type, verbose):
    """Play the new model against the pre-training version."""
    new_player = GreedyMCTSPlayer(net, mcts_config, name="new")
    prev_net = net.copy()
    prev_net.load_checkpoint(pre_checkpoint)
    prev_player = GreedyMCTSPlayer(prev_net, mcts_config, name="prev")

    eval_records = batched_arena(
        player_a=new_player,
        player_b=prev_player,
        game_factory=game_factory,
        num_games=arena.num_games,
        batch_size=arena.num_games,
        swap_players=arena.swap_players,
        game_type=game_type,
        verbose=verbose,
    )
    return new_player, eval_records


def _evaluate_vs_historical(new_player, net, best_checkpoints, mcts_config, arena, game_factory, game_type, verbose):
    """Play the new model against past best models."""
    historical_results = {}
    historical_game_records = {}

    if verbose and best_checkpoints:
        print(f"\n  vs past best models ({len(best_checkpoints)}):")

    for past_iter, past_path in best_checkpoints:
        past_net = net.copy()
        past_net.load_checkpoint(past_path)
        past_player = GreedyMCTSPlayer(past_net, mcts_config, name=f"iter_{past_iter:03d}")

        past_records = batched_arena(
            player_a=new_player,
            player_b=past_player,
            game_factory=game_factory,
            num_games=arena.num_games,
            batch_size=arena.num_games,
            swap_players=arena.swap_players,
            game_type=game_type,
            verbose=verbose,
        )

        past_result = past_records.evaluate("new")
        historical_results[past_iter] = {
            "wins": past_result.wins,
            "losses": past_result.losses,
            "win_rate": past_result.win_rate,
        }
        historical_game_records[past_iter] = past_records

        if verbose:
            print(f"    vs iter {past_iter:03d}: "
                  f"{past_result.wins}W/{past_result.losses}L "
                  f"({past_result.win_rate:.0%})")

    return historical_results, historical_game_records


def _save_arena_results(iter_dir, eval_records, result, accepted, historical_results, historical_game_records):
    """Save all arena evaluation results to disk."""
    eval_records.model_dump_json(indent=2)
    (iter_dir / "eval_games.json").write_text(eval_records.model_dump_json(indent=2))

    arena_results = {
        "vs_previous": {
            "wins": result.wins, "losses": result.losses,
            "draws": result.draws, "total": result.total,
            "win_rate": result.win_rate,
            "avg_moves_in_wins": result.avg_moves_in_wins,
            "avg_moves_in_losses": result.avg_moves_in_losses,
            "accepted": accepted,
        },
        "vs_historical": {
            f"iter_{past_iter:03d}": hist
            for past_iter, hist in historical_results.items()
        },
    }
    (iter_dir / "arena_results.json").write_text(json.dumps(arena_results, indent=2))

    for past_iter, past_records in historical_game_records.items():
        (iter_dir / f"eval_games_vs_iter_{past_iter:03d}.json").write_text(
            past_records.model_dump_json(indent=2)
        )


def _save_iteration_data(iter_dir, iteration, all_examples, train_metrics, arena_data):
    """Save iteration summary data to disk."""
    iteration_data = {
        "iteration": iteration,
        "training": {
            "num_examples": len(all_examples),
            "metrics": train_metrics,
        },
        **arena_data,
    }
    (iter_dir / "iteration_data.json").write_text(json.dumps(iteration_data, indent=2))
    return iteration_data


def _arena_head_to_head(
    net, pre_checkpoint, mcts_config, arena, game_factory, game_type, verbose,
    best_checkpoints, arena_dir, iteration, iter_dir, all_examples, train_metrics,
):
    """Evaluate using head-to-head ArenaConfig strategy. Returns whether accepted."""
    if verbose:
        print("\n[3/3] Arena evaluation...")
        print("  vs previous model:")

    new_player, eval_records = _evaluate_vs_previous(
        net, pre_checkpoint, mcts_config, arena, game_factory, game_type, verbose,
    )
    result = eval_records.evaluate("new")
    accepted = eval_records.is_better("new", arena.threshold)

    if verbose:
        print(f"  New: {result.wins} wins | Prev: {result.losses} wins | "
              f"Win rate: {result.win_rate:.1%}")

    historical_results, historical_game_records = _evaluate_vs_historical(
        new_player, net, best_checkpoints, mcts_config, arena, game_factory, game_type, verbose,
    )

    if accepted:
        if verbose:
            print("\n  -> ACCEPTED: new model is better")
        best_checkpoints.append((iteration, str(iter_dir / "post_training.pt")))
        GreedyMCTSPlayer(net, mcts_config, name=f"iteration_{iteration:03d}").save(
            arena_dir / f"iteration_{iteration:03d}"
        )
    else:
        if verbose:
            print("\n  -> REJECTED: reverting to pre-training weights")
        net.load_checkpoint(pre_checkpoint)

    _save_arena_results(iter_dir, eval_records, result, accepted, historical_results, historical_game_records)
    _save_iteration_data(iter_dir, iteration, all_examples, train_metrics, {
        "arena": {
            "new_wins": result.wins, "prev_wins": result.losses,
            "total_games": result.total, "win_rate": result.win_rate,
            "accepted": accepted,
        },
        "historical_arena": historical_results,
    })
    return accepted


def _play_candidate_vs_pool(candidate, pool_players, arena, game_factory, game_type, verbose):
    """Play a candidate player against every pool player. Returns match results and pool ratings."""
    match_results = []
    pool_ratings = {}

    for pool_name, pool_player, pool_elo in pool_players:
        pool_ratings[pool_name] = pool_elo
        records = batched_arena(
            player_a=candidate,
            player_b=pool_player,
            game_factory=game_factory,
            num_games=arena.games_per_matchup,
            batch_size=arena.games_per_matchup,
            swap_players=arena.swap_players,
            game_type=game_type,
            verbose=verbose,
        )
        scores = records.scores
        wins_a = scores.get(candidate.name, 0)
        wins_b = scores.get(pool_name, 0)
        draws = len(records) - wins_a - wins_b
        match_results.append(MatchResult(
            player_a=candidate.name, player_b=pool_name,
            wins_a=wins_a, wins_b=wins_b, draws=draws,
        ))

    return match_results, pool_ratings


def _evict_weakest_from_pool(pool_players, verbose):
    """Remove the weakest non-random player from the pool."""
    weakest_idx = None
    weakest_elo = float("inf")

    for idx, (name, _, elo) in enumerate(pool_players):
        if name == "random":
            continue
        if elo < weakest_elo:
            weakest_elo = elo
            weakest_idx = idx

    if weakest_idx is not None:
        evicted_name = pool_players[weakest_idx][0]
        pool_players.pop(weakest_idx)
        if verbose:
            print(f"  Pool: evicted {evicted_name} (Elo {weakest_elo:.0f})")


def _grow_pool(pool_players, net, mcts_config, arena, arena_dir, iteration, pool_current_elo, verbose):
    """Add current model to pool and evict weakest if over max size."""
    grow_name = f"pool_iteration_{iteration:03d}"
    grow_player = GreedyMCTSPlayer(net.copy(), mcts_config, name=grow_name, elo=pool_current_elo)
    grow_player.save(arena_dir / grow_name)
    pool_players.append((grow_name, grow_player, pool_current_elo))

    if arena.max_pool_size is not None and len(pool_players) > arena.max_pool_size:
        _evict_weakest_from_pool(pool_players, verbose)


def _arena_elo_pool(
    net, pre_checkpoint, mcts_config, arena, game_factory, game_type, verbose,
    pool_players, pool_current_elo, arena_dir, iteration, iter_dir, all_examples, train_metrics,
):
    """Evaluate using Elo pool strategy. Returns (accepted, updated pool_current_elo)."""
    if verbose:
        print("\n[3/3] Elo arena evaluation...")

    post_player = GreedyMCTSPlayer(net, mcts_config, name="candidate")
    match_results, pool_ratings = _play_candidate_vs_pool(
        post_player, pool_players, arena, game_factory, game_type, verbose,
    )
    post_elo = compute_elo_against_pool("candidate", pool_ratings, match_results)

    if pool_current_elo is None:
        pool_current_elo = _compute_baseline_elo(
            net, pre_checkpoint, mcts_config, pool_players, arena, game_factory, game_type, verbose, pool_ratings,
        )

    accepted = post_elo >= pool_current_elo + arena.elo_threshold

    if verbose:
        print(f"  Post-training Elo: {post_elo:.0f} | Current Elo: {pool_current_elo:.0f} | "
              f"Threshold: +{arena.elo_threshold:.0f}")

    if accepted:
        if verbose:
            print("  -> ACCEPTED: Elo improved")
        pool_current_elo = post_elo
        GreedyMCTSPlayer(net, mcts_config, name=f"iteration_{iteration:03d}", elo=post_elo).save(
            arena_dir / f"iteration_{iteration:03d}"
        )
    else:
        if verbose:
            print("  -> REJECTED: Elo did not improve enough")
        net.load_checkpoint(pre_checkpoint)

    if iteration % arena.pool_growth_interval == 0:
        _grow_pool(pool_players, net, mcts_config, arena, arena_dir, iteration, pool_current_elo, verbose)

    _save_iteration_data(iter_dir, iteration, all_examples, train_metrics, {
        "elo_arena": {
            "post_elo": post_elo, "current_elo": pool_current_elo,
            "threshold": arena.elo_threshold, "accepted": accepted,
            "pool_size": len(pool_players),
        },
    })
    return accepted, pool_current_elo


def _compute_baseline_elo(net, pre_checkpoint, mcts_config, pool_players, arena, game_factory, game_type, verbose, pool_ratings):
    """Compute baseline Elo for the pre-training model on the first iteration."""
    pre_net = net.copy()
    pre_net.load_checkpoint(pre_checkpoint)
    pre_player = GreedyMCTSPlayer(pre_net, mcts_config, name="pre_candidate")

    pre_match_results, _ = _play_candidate_vs_pool(
        pre_player, pool_players, arena, game_factory, game_type, verbose,
    )
    return compute_elo_against_pool("pre_candidate", pool_ratings, pre_match_results)


def _track_elo(
    net, mcts_config, game_factory, training_config, verbose,
    iteration, elo_match_results, elo_reference_pool, run_dir, iteration_data, iter_data_path,
):
    """Run Elo tracking against reference pool (optional monitoring feature)."""
    if verbose:
        print("  Elo evaluation...")

    current_player = GreedyMCTSPlayer(net, mcts_config, name=f"iter_{iteration:03d}")

    for ref_name, ref_player in elo_reference_pool:
        ref_records = batched_arena(
            player_a=current_player, player_b=ref_player,
            game_factory=game_factory,
            num_games=training_config.elo_games_per_matchup,
            batch_size=training_config.elo_games_per_matchup,
            swap_players=True, game_type="elo", verbose=verbose,
        )
        scores = ref_records.scores
        elo_match_results.append(MatchResult(
            player_a=current_player.name, player_b=ref_name,
            wins_a=scores.get(current_player.name, 0),
            wins_b=scores.get(ref_name, 0),
            draws=len(ref_records) - scores.get(current_player.name, 0) - scores.get(ref_name, 0),
        ))

    elo_ratings = compute_elo_ratings(elo_match_results, anchor_player="random")
    current_elo = next(
        (r.rating for r in elo_ratings if r.name == current_player.name), 1000.0,
    )

    if verbose:
        print(f"  Elo: {current_elo:.0f}")

    _maybe_grow_elo_reference_pool(net, mcts_config, iteration, training_config, elo_reference_pool)
    _save_elo_results(elo_ratings, elo_match_results, run_dir, iteration)

    iteration_data["elo"] = current_elo
    iter_data_path.write_text(json.dumps(iteration_data, indent=2))


def _maybe_grow_elo_reference_pool(net, mcts_config, iteration, training_config, elo_reference_pool):
    """Add current model to the Elo tracking reference pool at configured intervals."""
    if iteration % training_config.elo_reference_interval == 0:
        ref_net = net.copy()
        ref_player = GreedyMCTSPlayer(ref_net, mcts_config, name=f"ref_iter_{iteration:03d}")
        elo_reference_pool.append((ref_player.name, ref_player))


def _save_elo_results(elo_ratings, elo_match_results, run_dir, iteration):
    """Save Elo tournament results to disk."""
    elo_result = TournamentResult(
        ratings=elo_ratings,
        match_results=elo_match_results,
        anchor_player="random",
        anchor_rating=1000.0,
        timestamp=datetime.now().isoformat(),
        metadata={"training_run": str(run_dir), "iteration": iteration},
    )
    (run_dir / "elo_results.json").write_text(elo_result.model_dump_json(indent=2))


def _init_elo_pool(arena, arena_dir):
    """Initialize the Elo player pool with RandomPlayer and optional seeded players."""
    pool_players = []

    random_player = RandomPlayer(name="random")
    random_player.elo = 1000.0
    random_player.save(arena_dir / "random")
    pool_players.append(("random", random_player, 1000.0))

    if arena.initial_pool:
        for player_path in arena.initial_pool:
            loaded = MCTSPlayer.load(player_path)
            loaded_elo = loaded.elo if loaded.elo is not None else 1000.0
            loaded.save(arena_dir / loaded.name)
            pool_players.append((loaded.name, loaded, loaded_elo))

    return pool_players


def train(
    game_factory: Callable[[], BaseGame],
    net: BaseNeuralNet,
    mcts_config: MCTSConfig,
    training_config: TrainingConfig,
    arena: ArenaConfig | EloArenaConfig,
    paths_config: PathsConfig | None = None,
    game_type: str = "unknown",
    game_config: dict | None = None,
    verbose: bool = True,
):
    """Run the full AlphaZero training pipeline.

    Each iteration: self-play -> train -> evaluate (accept/reject).

    Args:
        game_factory: Callable that creates a new game instance.
        net: Neural network to train.
        mcts_config: MCTS configuration.
        training_config: Training loop configuration.
        arena: Arena evaluation configuration (ArenaConfig or EloArenaConfig).
        paths_config: File system paths. If None, uses defaults.
        game_type: Game type string for records.
        game_config: Game config dict for records.
        verbose: Whether to print progress.
    """
    paths = paths_config or PathsConfig()
    game_config = game_config or {}

    run_dir = _create_run_dir(paths)
    arena_dir = run_dir / "arena"
    arena_dir.mkdir(parents=True, exist_ok=True)

    _save_run_config(run_dir, net, mcts_config, training_config, arena, game_type, game_config)

    if verbose:
        param_count = sum(p.numel() for p in net.parameters())
        print(f"Training run directory: {run_dir}")
        print(f"Model parameters: {param_count:,}\n")

    replay_buffer: deque[list[Example]] = deque(maxlen=training_config.replay_buffer_size)
    best_checkpoints: deque[tuple[int, str]] = deque(maxlen=10)

    elo_match_results: list[MatchResult] = []
    elo_reference_pool: list[tuple[str, GreedyMCTSPlayer | RandomPlayer]] = []
    if training_config.elo_tracking:
        elo_reference_pool.append(("random", RandomPlayer(name="random")))

    pool_players: list[tuple[str, BasePlayer, float]] = []
    pool_current_elo: float | None = None
    if isinstance(arena, EloArenaConfig):
        pool_players = _init_elo_pool(arena, arena_dir)

    def _game_from_config(cfg: dict) -> BaseGame:
        return game_factory()

    for iteration in range(1, training_config.num_iterations + 1):
        if verbose:
            print(f"{'=' * 60}")
            print(f"Iteration {iteration}/{training_config.num_iterations}")
            print(f"{'=' * 60}")

        iter_dir = _create_iter_dir(run_dir, iteration)
        pre_checkpoint = str(iter_dir / "pre_training.pt")
        net.save_checkpoint(pre_checkpoint)

        # 1. Self-play
        if verbose:
            print("\n[1/3] Self-play...")
        self_play_records = _run_self_play(net, game_factory, mcts_config, training_config, verbose)
        (iter_dir / "self_play_games.json").write_text(self_play_records.model_dump_json(indent=2))

        new_examples = examples_from_records(self_play_records, _game_from_config)
        replay_buffer.append(new_examples)
        all_examples = [ex for batch in replay_buffer for ex in batch]

        if verbose:
            print(f"  Self-play: {len(self_play_records)} games, {len(new_examples)} examples")
            print(f"  Replay buffer: {len(replay_buffer)} iterations, {len(all_examples)} examples")

        # 2. Train
        if verbose:
            print("\n[2/3] Training...")
        train_metrics = _train_network(net, all_examples, training_config, verbose)
        post_checkpoint = str(iter_dir / "post_training.pt")
        net.save_checkpoint(post_checkpoint)

        # 3. Evaluate
        if isinstance(arena, ArenaConfig):
            accepted = _arena_head_to_head(
                net, pre_checkpoint, mcts_config, arena, game_factory, game_type, verbose,
                best_checkpoints, arena_dir, iteration, iter_dir, all_examples, train_metrics,
            )

            if training_config.elo_tracking:
                iter_data_path = iter_dir / "iteration_data.json"
                iteration_data = json.loads(iter_data_path.read_text())
                _track_elo(
                    net, mcts_config, game_factory, training_config, verbose,
                    iteration, elo_match_results, elo_reference_pool, run_dir,
                    iteration_data, iter_data_path,
                )

        elif isinstance(arena, EloArenaConfig):
            accepted, pool_current_elo = _arena_elo_pool(
                net, pre_checkpoint, mcts_config, arena, game_factory, game_type, verbose,
                pool_players, pool_current_elo, arena_dir, iteration, iter_dir,
                all_examples, train_metrics,
            )

        if verbose:
            print()

    if verbose:
        print("Training complete!")
        print(f"Run directory: {run_dir}")
