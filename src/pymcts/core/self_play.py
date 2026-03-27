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
    games = [game_factory() for _ in range(active_size)]
    move_histories: list[list[MoveRecord]] = [[] for _ in range(active_size)]

    completed: list[GameRecord] = []
    games_started = active_size
    # Track which slots have already been recorded (finished but not replaced)
    recorded: set[int] = set()

    pbar = None
    if verbose:
        pbar = tqdm(total=num_games, desc="Self-play")

    while len(completed) < num_games:
        active_idx = [
            i for i in range(len(games))
            if not games[i].is_over and i not in recorded
        ]
        if not active_idx:
            break

        active_games = [games[i] for i in active_idx]
        roots = mcts.search_batch(active_games)

        for j, i in enumerate(active_idx):
            root = roots[j]
            action_space = games[i].action_space_size
            visit_counts = root.visit_counts(action_space)
            probs = MCTS.visit_counts_to_probs(visit_counts, temperature)

            if probs.sum() == 0:
                probs = games[i].to_mask().float()

            action = torch.multinomial(probs, 1).item()

            current_player = games[i].current_player
            games[i].make_action(action)

            move_histories[i].append(MoveRecord(
                action=action,
                player=current_player,
                policy=probs,
            ))

        for i in range(len(games)):
            if games[i].is_over and i not in recorded:
                record = GameRecord(
                    game_type=game_type,
                    game_config=games[i].get_config(),
                    moves=move_histories[i],
                    winner=games[i].winner,
                    player_names=["self-play-0", "self-play-1"],
                )
                completed.append(record)

                if pbar is not None:
                    pbar.update(1)

                if games_started < num_games:
                    games[i] = game_factory()
                    move_histories[i] = []
                    games_started += 1
                else:
                    recorded.add(i)

    if pbar is not None:
        pbar.close()

    logger.info("Batched self-play: %d games completed", len(completed))
    return GameRecordCollection(game_records=completed)
