"""Batched self-play: run N games concurrently with batched neural net inference."""

import logging
from dataclasses import dataclass, field
from typing import Callable

import torch
from tqdm.auto import tqdm

from pymcts.core.base_game import BaseGame
from pymcts.core.base_neural_net import BaseNeuralNet
from pymcts.core.config import MCTSConfig
from pymcts.core.game_record import GameRecord, GameRecordCollection, MoveRecord
from pymcts.core.mcts import MCTS

logger = logging.getLogger("pymcts.core.self_play")


@dataclass
class _GameSlot:
    """A single concurrent game being played during self-play."""
    game: BaseGame
    history: list[MoveRecord] = field(default_factory=list)


def _make_slots(game_factory: Callable[[], BaseGame], count: int) -> list[_GameSlot]:
    return [_GameSlot(game=game_factory()) for _ in range(count)]


def _active_slots(slots: list[_GameSlot], retired: set[int]) -> list[int]:
    return [i for i, s in enumerate(slots) if not s.game.is_over and i not in retired]


def _select_action(root, game: BaseGame, temperature: float) -> tuple[int, torch.Tensor]:
    visit_counts = root.visit_counts(game.action_space_size)
    probs = MCTS.visit_counts_to_probs(visit_counts, temperature)
    if probs.sum() == 0:
        probs = game.to_mask().float()
    return int(torch.multinomial(probs, 1).item()), probs


def _apply_actions(slots: list[_GameSlot], active: list[int], roots: list, temperature: float):
    for j, i in enumerate(active):
        slot = slots[i]
        action, probs = _select_action(roots[j], slot.game, temperature)
        player = slot.game.current_player
        slot.game.make_action(action)
        slot.history.append(MoveRecord(action=action, player=player, policy=probs))


def _to_record(slot: _GameSlot, game_type: str) -> GameRecord:
    return GameRecord(
        game_type=game_type,
        game_config=slot.game.get_config(),
        moves=slot.history,
        winner=slot.game.winner,
        player_names=["self-play-0", "self-play-1"],
    )


def _collect_finished(
    slots: list[_GameSlot], retired: set[int], completed: list[GameRecord],
    game_factory: Callable[[], BaseGame], games_started: int, num_games: int,
    game_type: str, pbar,
) -> int:
    for i, slot in enumerate(slots):
        if not slot.game.is_over or i in retired:
            continue
        completed.append(_to_record(slot, game_type))
        if pbar is not None:
            pbar.update(1)
        if games_started < num_games:
            slots[i] = _GameSlot(game=game_factory())
            games_started += 1
        else:
            retired.add(i)
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
    """Play self-play games with batched MCTS inference."""
    mcts = MCTS(net, mcts_config)
    active_size = min(batch_size, num_games)
    slots = _make_slots(game_factory, active_size)
    completed: list[GameRecord] = []
    retired: set[int] = set()
    games_started = active_size
    pbar = tqdm(total=num_games, desc="Self-play") if verbose else None

    while len(completed) < num_games:
        active = _active_slots(slots, retired)
        if not active:
            break
        roots = mcts.search_batch([slots[i].game for i in active])
        _apply_actions(slots, active, roots, temperature)
        games_started = _collect_finished(
            slots, retired, completed, game_factory, games_started, num_games, game_type, pbar,
        )

    if pbar is not None:
        pbar.close()
    return GameRecordCollection(game_records=completed)
