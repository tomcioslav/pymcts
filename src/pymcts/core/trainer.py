"""Main training pipeline: self-play -> train -> evaluate loop."""

import json
import logging
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Callable

from pydantic import BaseModel, ConfigDict, Field

from pymcts.arena.engine import batched_arena
from pymcts.core.base_game import BaseGame
from pymcts.core.base_neural_net import BaseNeuralNet
from pymcts.core.config import ArenaConfig, EloArenaConfig, MCTSConfig, PathsConfig, TrainingConfig
from pymcts.core.data import examples_from_records
from pymcts.core.players import BasePlayer, GreedyMCTSPlayer, MCTSPlayer, RandomPlayer
from pymcts.elo.config import MatchResult, TournamentResult
from pymcts.elo.rating import compute_elo_against_pool, compute_elo_ratings
from pymcts.core.self_play import batched_self_play

logger = logging.getLogger("pymcts.core.trainer")


class _TrainingContext(BaseModel):
    """All shared state for a training run — replaces long parameter lists."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    net: BaseNeuralNet
    mcts_config: MCTSConfig
    training_config: TrainingConfig
    arena: ArenaConfig | EloArenaConfig
    game_factory: Callable[[], BaseGame]
    game_type: str
    game_config: dict
    verbose: bool
    run_dir: Path
    arena_dir: Path
    # ArenaConfig state
    best_checkpoints: deque = Field(default_factory=lambda: deque(maxlen=10))
    # EloArenaConfig state
    pool_players: list[tuple[str, BasePlayer, float]] = []
    pool_current_elo: float | None = None
    # Elo tracking state (optional monitoring)
    elo_match_results: list[MatchResult] = []
    elo_reference_pool: list = []


def _create_run_dir(paths: PathsConfig) -> Path:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run_dir = paths.trainings / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _save_run_config(ctx: _TrainingContext):
    net_class = type(ctx.net)
    config = {
        "net_class": f"{net_class.__module__}.{net_class.__qualname__}",
        "mcts_config": ctx.mcts_config.model_dump(),
        "training_config": ctx.training_config.model_dump(),
        "arena": ctx.arena.model_dump(),
        "arena_type": type(ctx.arena).__name__,
        "game_type": ctx.game_type,
        "game_config": ctx.game_config,
    }
    (ctx.run_dir / "run_config.json").write_text(json.dumps(config, indent=2))


def _init_elo_pool(ctx: _TrainingContext):
    random_player = RandomPlayer(name="random")
    random_player.elo = 1000.0
    random_player.save(ctx.arena_dir / "random")
    ctx.pool_players.append(("random", random_player, 1000.0))

    if ctx.arena.initial_pool:
        for path in ctx.arena.initial_pool:
            loaded = MCTSPlayer.load(path)
            elo = loaded.elo if loaded.elo is not None else 1000.0
            loaded.save(ctx.arena_dir / loaded.name)
            ctx.pool_players.append((loaded.name, loaded, elo))


# ---------------------------------------------------------------------------
# Self-play & training
# ---------------------------------------------------------------------------

def _run_self_play(ctx: _TrainingContext):
    return batched_self_play(
        net=ctx.net, game_factory=ctx.game_factory, mcts_config=ctx.mcts_config,
        num_games=ctx.training_config.num_self_play_games,
        batch_size=ctx.training_config.self_play_batch_size,
        temperature=1.0, verbose=ctx.verbose,
    )


def _train_network(ctx: _TrainingContext, examples):
    return ctx.net.train_on_examples(
        examples,
        num_epochs=ctx.training_config.num_epochs,
        batch_size=ctx.training_config.batch_size,
        learning_rate=ctx.training_config.learning_rate,
        weight_decay=ctx.training_config.weight_decay,
        verbose=ctx.verbose,
    )


# ---------------------------------------------------------------------------
# Head-to-head arena evaluation (ArenaConfig)
# ---------------------------------------------------------------------------

def _evaluate_vs_opponent(ctx: _TrainingContext, new_player, opponent, name: str):
    """Play new_player against a single opponent. Returns game records."""
    return batched_arena(
        player_a=new_player, player_b=opponent, game_factory=ctx.game_factory,
        num_games=ctx.arena.num_games, batch_size=ctx.arena.num_games,
        swap_players=ctx.arena.swap_players, game_type=ctx.game_type, verbose=ctx.verbose,
    )


def _evaluate_vs_historical(ctx: _TrainingContext, new_player):
    results = {}
    records = {}
    if ctx.verbose and ctx.best_checkpoints:
        print(f"\n  vs past best models ({len(ctx.best_checkpoints)}):")

    for past_iter, past_path in ctx.best_checkpoints:
        past_net = ctx.net.copy()
        past_net.load_checkpoint(past_path)
        past_player = GreedyMCTSPlayer(past_net, ctx.mcts_config, name=f"iter_{past_iter:03d}")
        past_records = _evaluate_vs_opponent(ctx, new_player, past_player, past_player.name)
        r = past_records.evaluate("new")
        results[past_iter] = {"wins": r.wins, "losses": r.losses, "win_rate": r.win_rate}
        records[past_iter] = past_records
        if ctx.verbose:
            print(f"    vs iter {past_iter:03d}: {r.wins}W/{r.losses}L ({r.win_rate:.0%})")

    return results, records


def _save_arena_results(iter_dir, eval_records, result, accepted, hist_results, hist_records):
    (iter_dir / "eval_games.json").write_text(eval_records.model_dump_json(indent=2))
    arena_results = {
        "vs_previous": {
            "wins": result.wins, "losses": result.losses, "draws": result.draws,
            "total": result.total, "win_rate": result.win_rate,
            "avg_moves_in_wins": result.avg_moves_in_wins,
            "avg_moves_in_losses": result.avg_moves_in_losses, "accepted": accepted,
        },
        "vs_historical": {f"iter_{k:03d}": v for k, v in hist_results.items()},
    }
    (iter_dir / "arena_results.json").write_text(json.dumps(arena_results, indent=2))
    for past_iter, past_records in hist_records.items():
        (iter_dir / f"eval_games_vs_iter_{past_iter:03d}.json").write_text(
            past_records.model_dump_json(indent=2)
        )


def _save_iteration_data(iter_dir, iteration, num_examples, train_metrics, arena_data):
    data = {"iteration": iteration, "training": {"num_examples": num_examples, "metrics": train_metrics}, **arena_data}
    path = iter_dir / "iteration_data.json"
    path.write_text(json.dumps(data, indent=2))
    return data, path


def _arena_head_to_head(ctx: _TrainingContext, pre_checkpoint, iteration, iter_dir, num_examples, train_metrics):
    """Evaluate using head-to-head strategy. Returns whether model was accepted."""
    if ctx.verbose:
        print("\n[3/3] Arena evaluation...\n  vs previous model:")

    new_player = GreedyMCTSPlayer(ctx.net, ctx.mcts_config, name="new")
    prev_net = ctx.net.copy()
    prev_net.load_checkpoint(pre_checkpoint)
    prev_player = GreedyMCTSPlayer(prev_net, ctx.mcts_config, name="prev")

    eval_records = _evaluate_vs_opponent(ctx, new_player, prev_player, "prev")
    result = eval_records.evaluate("new")
    accepted = eval_records.is_better("new", ctx.arena.threshold)

    if ctx.verbose:
        print(f"  New: {result.wins} wins | Prev: {result.losses} wins | Win rate: {result.win_rate:.1%}")

    hist_results, hist_records = _evaluate_vs_historical(ctx, new_player)

    if accepted:
        if ctx.verbose:
            print("\n  -> ACCEPTED: new model is better")
        ctx.best_checkpoints.append((iteration, str(iter_dir / "post_training.pt")))
        GreedyMCTSPlayer(ctx.net, ctx.mcts_config, name=f"iteration_{iteration:03d}").save(
            ctx.arena_dir / f"iteration_{iteration:03d}"
        )
    else:
        if ctx.verbose:
            print("\n  -> REJECTED: reverting to pre-training weights")
        ctx.net.load_checkpoint(pre_checkpoint)

    _save_arena_results(iter_dir, eval_records, result, accepted, hist_results, hist_records)
    _save_iteration_data(iter_dir, iteration, num_examples, train_metrics, {
        "arena": {"new_wins": result.wins, "prev_wins": result.losses,
                  "total_games": result.total, "win_rate": result.win_rate, "accepted": accepted},
        "historical_arena": hist_results,
    })
    return accepted


# ---------------------------------------------------------------------------
# Elo pool arena evaluation (EloArenaConfig)
# ---------------------------------------------------------------------------

def _play_vs_pool(ctx: _TrainingContext, candidate):
    """Play candidate against every pool player. Returns (match_results, pool_ratings)."""
    match_results = []
    pool_ratings = {}
    for name, player, elo in ctx.pool_players:
        pool_ratings[name] = elo
        records = batched_arena(
            player_a=candidate, player_b=player, game_factory=ctx.game_factory,
            num_games=ctx.arena.games_per_matchup, batch_size=ctx.arena.games_per_matchup,
            swap_players=ctx.arena.swap_players, game_type=ctx.game_type, verbose=ctx.verbose,
        )
        scores = records.scores
        match_results.append(MatchResult(
            player_a=candidate.name, player_b=name,
            wins_a=scores.get(candidate.name, 0), wins_b=scores.get(name, 0),
            draws=len(records) - scores.get(candidate.name, 0) - scores.get(name, 0),
        ))
    return match_results, pool_ratings


def _compute_baseline_elo(ctx: _TrainingContext, pre_checkpoint, pool_ratings):
    pre_net = ctx.net.copy()
    pre_net.load_checkpoint(pre_checkpoint)
    pre_player = GreedyMCTSPlayer(pre_net, ctx.mcts_config, name="pre_candidate")
    results, _ = _play_vs_pool(ctx, pre_player)
    return compute_elo_against_pool("pre_candidate", pool_ratings, results)


def _evict_weakest(ctx: _TrainingContext):
    weakest_idx, weakest_elo = None, float("inf")
    for idx, (name, _, elo) in enumerate(ctx.pool_players):
        if name != "random" and elo < weakest_elo:
            weakest_elo = elo
            weakest_idx = idx
    if weakest_idx is not None:
        evicted = ctx.pool_players.pop(weakest_idx)[0]
        if ctx.verbose:
            print(f"  Pool: evicted {evicted} (Elo {weakest_elo:.0f})")


def _grow_pool(ctx: _TrainingContext, iteration):
    name = f"pool_iteration_{iteration:03d}"
    player = GreedyMCTSPlayer(ctx.net.copy(), ctx.mcts_config, name=name, elo=ctx.pool_current_elo)
    player.save(ctx.arena_dir / name)
    ctx.pool_players.append((name, player, ctx.pool_current_elo))
    if ctx.arena.max_pool_size is not None and len(ctx.pool_players) > ctx.arena.max_pool_size:
        _evict_weakest(ctx)


def _arena_elo_pool(ctx: _TrainingContext, pre_checkpoint, iteration, iter_dir, num_examples, train_metrics):
    """Evaluate using Elo pool strategy. Returns whether model was accepted."""
    if ctx.verbose:
        print("\n[3/3] Elo arena evaluation...")

    candidate = GreedyMCTSPlayer(ctx.net, ctx.mcts_config, name="candidate")
    match_results, pool_ratings = _play_vs_pool(ctx, candidate)
    post_elo = compute_elo_against_pool("candidate", pool_ratings, match_results)

    if ctx.pool_current_elo is None:
        ctx.pool_current_elo = _compute_baseline_elo(ctx, pre_checkpoint, pool_ratings)

    accepted = post_elo >= ctx.pool_current_elo + ctx.arena.elo_threshold

    if ctx.verbose:
        print(f"  Post-training Elo: {post_elo:.0f} | Current Elo: {ctx.pool_current_elo:.0f} | "
              f"Threshold: +{ctx.arena.elo_threshold:.0f}")

    if accepted:
        if ctx.verbose:
            print("  -> ACCEPTED: Elo improved")
        ctx.pool_current_elo = post_elo
        GreedyMCTSPlayer(ctx.net, ctx.mcts_config, name=f"iteration_{iteration:03d}", elo=post_elo).save(
            ctx.arena_dir / f"iteration_{iteration:03d}"
        )
    else:
        if ctx.verbose:
            print("  -> REJECTED: Elo did not improve enough")
        ctx.net.load_checkpoint(pre_checkpoint)

    if iteration % ctx.arena.pool_growth_interval == 0:
        _grow_pool(ctx, iteration)

    _save_iteration_data(iter_dir, iteration, num_examples, train_metrics, {
        "elo_arena": {"post_elo": post_elo, "current_elo": ctx.pool_current_elo,
                      "threshold": ctx.arena.elo_threshold, "accepted": accepted,
                      "pool_size": len(ctx.pool_players)},
    })
    return accepted


# ---------------------------------------------------------------------------
# Elo tracking (optional monitoring, independent of arena strategy)
# ---------------------------------------------------------------------------

def _track_elo(ctx: _TrainingContext, iteration, iteration_data, iter_data_path):
    if ctx.verbose:
        print("  Elo evaluation...")

    player = GreedyMCTSPlayer(ctx.net, ctx.mcts_config, name=f"iter_{iteration:03d}")

    for ref_name, ref_player in ctx.elo_reference_pool:
        records = batched_arena(
            player_a=player, player_b=ref_player, game_factory=ctx.game_factory,
            num_games=ctx.training_config.elo_games_per_matchup,
            batch_size=ctx.training_config.elo_games_per_matchup,
            swap_players=True, game_type="elo", verbose=ctx.verbose,
        )
        scores = records.scores
        ctx.elo_match_results.append(MatchResult(
            player_a=player.name, player_b=ref_name,
            wins_a=scores.get(player.name, 0), wins_b=scores.get(ref_name, 0),
            draws=len(records) - scores.get(player.name, 0) - scores.get(ref_name, 0),
        ))

    ratings = compute_elo_ratings(ctx.elo_match_results, anchor_player="random")
    current_elo = next((r.rating for r in ratings if r.name == player.name), 1000.0)

    if ctx.verbose:
        print(f"  Elo: {current_elo:.0f}")

    if iteration % ctx.training_config.elo_reference_interval == 0:
        ref_player = GreedyMCTSPlayer(ctx.net.copy(), ctx.mcts_config, name=f"ref_iter_{iteration:03d}")
        ctx.elo_reference_pool.append((ref_player.name, ref_player))

    elo_result = TournamentResult(
        ratings=ratings, match_results=ctx.elo_match_results,
        anchor_player="random", anchor_rating=1000.0,
        timestamp=datetime.now().isoformat(),
        metadata={"training_run": str(ctx.run_dir), "iteration": iteration},
    )
    (ctx.run_dir / "elo_results.json").write_text(elo_result.model_dump_json(indent=2))

    iteration_data["elo"] = current_elo
    iter_data_path.write_text(json.dumps(iteration_data, indent=2))


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

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
    """
    paths = paths_config or PathsConfig()
    run_dir = _create_run_dir(paths)
    arena_dir = run_dir / "arena"
    arena_dir.mkdir(parents=True, exist_ok=True)

    ctx = _TrainingContext(
        net=net, mcts_config=mcts_config, training_config=training_config,
        arena=arena, game_factory=game_factory, game_type=game_type,
        game_config=game_config or {}, verbose=verbose,
        run_dir=run_dir, arena_dir=arena_dir,
    )
    _save_run_config(ctx)

    if verbose:
        print(f"Training run directory: {run_dir}")
        print(f"Model parameters: {sum(p.numel() for p in net.parameters()):,}\n")

    if training_config.elo_tracking:
        ctx.elo_reference_pool.append(("random", RandomPlayer(name="random")))
    if isinstance(arena, EloArenaConfig):
        _init_elo_pool(ctx)

    replay_buffer: deque[list] = deque(maxlen=training_config.replay_buffer_size)

    for iteration in range(1, training_config.num_iterations + 1):
        if verbose:
            print(f"{'=' * 60}\nIteration {iteration}/{training_config.num_iterations}\n{'=' * 60}")

        iter_dir = run_dir / f"iteration_{iteration:03d}"
        iter_dir.mkdir(parents=True, exist_ok=True)
        pre_checkpoint = str(iter_dir / "pre_training.pt")
        net.save_checkpoint(pre_checkpoint)

        # 1. Self-play
        if verbose:
            print("\n[1/3] Self-play...")
        records = _run_self_play(ctx)
        (iter_dir / "self_play_games.json").write_text(records.model_dump_json(indent=2))

        new_examples = examples_from_records(records, lambda cfg: game_factory())
        replay_buffer.append(new_examples)
        all_examples = [ex for batch in replay_buffer for ex in batch]

        if verbose:
            print(f"  {len(records)} games, {len(new_examples)} examples")
            print(f"  Replay buffer: {len(replay_buffer)} iterations, {len(all_examples)} examples")

        # 2. Train
        if verbose:
            print("\n[2/3] Training...")
        train_metrics = _train_network(ctx, all_examples)
        net.save_checkpoint(str(iter_dir / "post_training.pt"))

        # 3. Evaluate
        if isinstance(arena, ArenaConfig):
            _arena_head_to_head(ctx, pre_checkpoint, iteration, iter_dir, len(all_examples), train_metrics)
            if training_config.elo_tracking:
                iter_data_path = iter_dir / "iteration_data.json"
                iteration_data = json.loads(iter_data_path.read_text())
                _track_elo(ctx, iteration, iteration_data, iter_data_path)
        elif isinstance(arena, EloArenaConfig):
            _arena_elo_pool(ctx, pre_checkpoint, iteration, iter_dir, len(all_examples), train_metrics)

        if verbose:
            print()

    if verbose:
        print(f"Training complete!\nRun directory: {run_dir}")
