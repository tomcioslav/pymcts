"""Batched self-play: run N games concurrently with batched neural net inference."""

import logging
from typing import Callable

import torch
from tqdm.auto import tqdm

from pymcts.core.base_game import BaseGame
from pymcts.core.base_neural_net import BaseNeuralNet
from pymcts.core.config import MCTSConfig
from pymcts.core.game_record import GameRecord, GameRecordCollection, MoveRecord
from pymcts.core.mcts import MCTS

logger = logging.getLogger("bridgit.core.self_play")


def _initialize_slots(
    game_factory: Callable[[], BaseGame],
    active_size: int,
) -> tuple[list[BaseGame], list[list[MoveRecord]]]:
    games = [game_factory() for _ in range(active_size)]
    move_histories: list[list[MoveRecord]] = [[] for _ in range(active_size)]
    return games, move_histories


def _get_active_indices(
    games: list[BaseGame],
    recorded: set[int],
) -> list[int]:
    return [i for i in range(len(games)) if not games[i].is_over and i not in recorded]


def _select_action(root, action_space: int, temperature: float, mask) -> int:
    visit_counts = root.visit_counts(action_space)
    probs = MCTS.visit_counts_to_probs(visit_counts, temperature)
    if probs.sum() == 0:
        probs = mask.float()
    return int(torch.multinomial(probs, 1).item()), probs


def _apply_action_to_slot(
    games: list[BaseGame],
    move_histories: list[list[MoveRecord]],
    slot: int,
    root,
    temperature: float,
) -> None:
    game = games[slot]
    action, probs = _select_action(root, game.action_space_size, temperature, game.to_mask())
    current_player = game.current_player
    game.make_action(action)
    move_histories[slot].append(MoveRecord(action=action, player=current_player, policy=probs))


def _process_search_results(
    games: list[BaseGame],
    move_histories: list[list[MoveRecord]],
    active_idx: list[int],
    roots: list,
    temperature: float,
) -> None:
    for j, i in enumerate(active_idx):
        _apply_action_to_slot(games, move_histories, i, roots[j], temperature)


def _make_progress_bar(verbose: bool, num_games: int):
    if verbose:
        return tqdm(total=num_games, desc="Self-play")
    return None


def _record_completed_game(
    games: list[BaseGame],
    move_histories: list[list[MoveRecord]],
    slot: int,
    completed: list[GameRecord],
    game_type: str,
    pbar,
) -> None:
    record = GameRecord(
        game_type=game_type,
        game_config=games[slot].get_config(),
        moves=move_histories[slot],
        winner=games[slot].winner,
        player_names=["self-play-0", "self-play-1"],
    )
    completed.append(record)
    if pbar is not None:
        pbar.update(1)


def _replace_or_mark_slot(
    games: list[BaseGame],
    move_histories: list[list[MoveRecord]],
    slot: int,
    recorded: set[int],
    game_factory: Callable[[], BaseGame],
    games_started: int,
    num_games: int,
) -> int:
    if games_started < num_games:
        games[slot] = game_factory()
        move_histories[slot] = []
        return games_started + 1
    recorded.add(slot)
    return games_started


def _process_finished_slots(
    games: list[BaseGame],
    move_histories: list[list[MoveRecord]],
    completed: list[GameRecord],
    recorded: set[int],
    game_factory: Callable[[], BaseGame],
    games_started: int,
    num_games: int,
    game_type: str,
    pbar,
) -> int:
    for i in range(len(games)):
        if games[i].is_over and i not in recorded:
            _record_completed_game(games, move_histories, i, completed, game_type, pbar)
            games_started = _replace_or_mark_slot(
                games, move_histories, i, recorded, game_factory, games_started, num_games
            )
    return games_started


def batched_self_play(
    net: BaseNeuralNet,
    game_factory: Callable[[], BaseGame],
    mcts_config: MCTSConfig,
    num_games: int,
    batch_size: int = 8,
    temperature: float = 1.0,
    verbose: bool = True,
    game_type: str = "self-play",
) -> GameRecordCollection:
    """Play self-play games with batched MCTS inference.

    Runs `batch_size` games concurrently with virtual loss
    (mcts_config.num_parallel_leaves) for maximum GPU throughput.
    """
    mcts = MCTS(net, mcts_config)
    active_size = min(batch_size, num_games)
    games, move_histories = _initialize_slots(game_factory, active_size)
    completed: list[GameRecord] = []
    games_started = active_size
    recorded: set[int] = set()
    pbar = _make_progress_bar(verbose, num_games)

    while len(completed) < num_games:
        active_idx = _get_active_indices(games, recorded)
        if not active_idx:
            break
        active_games = [games[i] for i in active_idx]
        roots = mcts.search_batch(active_games)
        _process_search_results(games, move_histories, active_idx, roots, temperature)
        games_started = _process_finished_slots(
            games, move_histories, completed, recorded,
            game_factory, games_started, num_games, game_type, pbar,
        )

    if pbar is not None:
        pbar.close()
    logger.info("Batched self-play: %d games completed", len(completed))
    return GameRecordCollection(game_records=completed)
