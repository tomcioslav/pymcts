"""Arena: run evaluation games between two players."""

import logging
from dataclasses import dataclass, field
from typing import Callable

import torch
from tqdm.auto import tqdm

from pymcts.core.base_game import BaseGame
from pymcts.core.mcts import MCTS
from pymcts.core.players import BasePlayer, MCTSPlayer
from pymcts.core.game_record import GameRecord, GameRecordCollection, MoveRecord

logger = logging.getLogger("pymcts.core.arena")


# ---------------------------------------------------------------------------
# Shared types
# ---------------------------------------------------------------------------

@dataclass
class _GameSlot:
    """A single concurrent game in the arena."""
    game: BaseGame
    history: list[MoveRecord] = field(default_factory=list)
    names: list[str] = field(default_factory=list)
    mcts_for_player: dict[int, MCTS] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Sequential arena (non-MCTS players)
# ---------------------------------------------------------------------------

def _play_one_game(game: BaseGame, players: list[BasePlayer]) -> list[MoveRecord]:
    """Play a single game to completion. Returns move history."""
    moves: list[MoveRecord] = []
    while not game.is_over:
        p = game.current_player
        action = players[p].get_action(game)
        moves.append(MoveRecord(action=action, player=p, policy=players[p].last_policy))
        game.make_action(action)
    return moves


def _player_order(player_a, player_b, swapped: bool):
    """Return (players, names) in the correct order for this game."""
    if swapped:
        return [player_b, player_a], [player_b.name, player_a.name]
    return [player_a, player_b], [player_a.name, player_b.name]


def _sequential_arena(
    player_a: BasePlayer,
    player_b: BasePlayer,
    game_factory: Callable[[], BaseGame],
    num_games: int,
    swap_players: bool,
    game_type: str,
    verbose: bool,
) -> GameRecordCollection:
    """Play games sequentially using player.get_action()."""
    half = num_games // 2 if swap_players else num_games
    completed: list[GameRecord] = []
    pbar = tqdm(total=num_games, desc=f"{player_a.name} vs {player_b.name}", leave=False) if verbose else None

    for idx in range(num_games):
        players, names = _player_order(player_a, player_b, swap_players and idx >= half)
        game = game_factory()
        moves = _play_one_game(game, players)
        completed.append(GameRecord(
            game_type=game_type, game_config=game.get_config(),
            moves=moves, winner=game.winner, player_names=names,
        ))
        if pbar is not None:
            pbar.update(1)

    if pbar is not None:
        pbar.close()
    return GameRecordCollection(game_records=completed)


# ---------------------------------------------------------------------------
# Batched MCTS arena
# ---------------------------------------------------------------------------

def _make_slot(game_idx: int, player_a: MCTSPlayer, player_b: MCTSPlayer,
               game_factory: Callable[[], BaseGame], half: int, swap_players: bool) -> _GameSlot:
    """Create a fresh game slot with correct player ordering."""
    swapped = swap_players and game_idx >= half
    game = game_factory()
    if swapped:
        return _GameSlot(game=game, names=[player_b.name, player_a.name],
                         mcts_for_player={0: player_b.mcts, 1: player_a.mcts})
    return _GameSlot(game=game, names=[player_a.name, player_b.name],
                     mcts_for_player={0: player_a.mcts, 1: player_b.mcts})


def _select_action(probs: torch.Tensor, game: BaseGame, temperature: float) -> int:
    """Pick an action from visit-count probabilities."""
    if probs.sum() == 0:
        probs = game.to_mask().float()
    if temperature > 0:
        return torch.multinomial(probs, 1).item()
    return torch.argmax(probs).item()


def _group_by_mcts(slots: list[_GameSlot], active: list[int]) -> dict[int, tuple[MCTS, list[int]]]:
    """Group active slot indices by which MCTS object handles the current player."""
    groups: dict[int, tuple[MCTS, list[int]]] = {}
    for i in active:
        mcts_obj = slots[i].mcts_for_player[slots[i].game.current_player]
        key = id(mcts_obj)
        if key not in groups:
            groups[key] = (mcts_obj, [])
        groups[key][1].append(i)
    return groups


def _run_mcts_and_apply(slots: list[_GameSlot], active: list[int], temperature: float):
    """Run batched MCTS search for all active slots, then apply one action each."""
    for mcts_obj, indices in _group_by_mcts(slots, active).values():
        roots = mcts_obj.search_batch([slots[i].game for i in indices])
        for j, i in enumerate(indices):
            slot = slots[i]
            vc = roots[j].visit_counts(slot.game.action_space_size)
            probs = MCTS.visit_counts_to_probs(vc, temperature)
            action = _select_action(probs, slot.game, temperature)
            player = slot.game.current_player
            slot.game.make_action(action)
            slot.history.append(MoveRecord(action=action, player=player, policy=probs))


def _to_record(slot: _GameSlot, game_type: str) -> GameRecord:
    return GameRecord(
        game_type=game_type, game_config=slot.game.get_config(),
        moves=slot.history, winner=slot.game.winner, player_names=slot.names,
    )


def _collect_finished(
    slots: list[_GameSlot], retired: set[int], completed: list[GameRecord],
    player_a: MCTSPlayer, player_b: MCTSPlayer, game_factory: Callable[[], BaseGame],
    games_started: int, num_games: int, half: int, swap_players: bool,
    game_type: str, pbar, wins: dict[str, int],
) -> int:
    for i, slot in enumerate(slots):
        if not slot.game.is_over or i in retired:
            continue
        record = _to_record(slot, game_type)
        completed.append(record)
        if record.winner is not None:
            winner_name = record.player_names[record.winner]
            wins[winner_name] = wins.get(winner_name, 0) + 1
        if pbar is not None:
            pbar.update(1)
            pbar.set_postfix_str(
                f"{player_a.name}={wins.get(player_a.name, 0)} | "
                f"{player_b.name}={wins.get(player_b.name, 0)}"
            )
        if games_started < num_games:
            slots[i] = _make_slot(games_started, player_a, player_b, game_factory, half, swap_players)
            games_started += 1
        else:
            retired.add(i)
    return games_started


def _batched_mcts_arena(
    player_a: MCTSPlayer,
    player_b: MCTSPlayer,
    game_factory: Callable[[], BaseGame],
    num_games: int,
    batch_size: int,
    swap_players: bool,
    temperature: float,
    game_type: str,
    verbose: bool,
) -> GameRecordCollection:
    """Play arena games with batched MCTS inference."""
    half = num_games // 2 if swap_players else num_games
    active_size = min(batch_size, num_games)
    slots = [_make_slot(i, player_a, player_b, game_factory, half, swap_players) for i in range(active_size)]
    completed: list[GameRecord] = []
    retired: set[int] = set()
    games_started = active_size
    wins: dict[str, int] = {}
    pbar = tqdm(total=num_games, desc=f"{player_a.name} vs {player_b.name}", leave=False) if verbose else None

    while len(completed) < num_games:
        active = [i for i, s in enumerate(slots) if not s.game.is_over and i not in retired]
        if not active:
            break
        _run_mcts_and_apply(slots, active, temperature)
        games_started = _collect_finished(
            slots, retired, completed, player_a, player_b, game_factory,
            games_started, num_games, half, swap_players, game_type, pbar, wins,
        )

    if pbar is not None:
        pbar.close()
    return GameRecordCollection(game_records=completed)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def batched_arena(
    player_a: BasePlayer,
    player_b: BasePlayer,
    game_factory: Callable[[], BaseGame],
    num_games: int,
    batch_size: int = 8,
    swap_players: bool = False,
    temperature: float = 0.0,
    game_type: str = "arena",
    verbose: bool = True,
) -> GameRecordCollection:
    """Play arena games between two players.

    If both players are MCTSPlayer, uses batched MCTS inference for speed.
    Otherwise, falls back to sequential get_action() calls.
    """
    if isinstance(player_a, MCTSPlayer) and isinstance(player_b, MCTSPlayer):
        return _batched_mcts_arena(
            player_a, player_b, game_factory, num_games,
            batch_size, swap_players, temperature, game_type, verbose,
        )
    return _sequential_arena(
        player_a, player_b, game_factory, num_games,
        swap_players, game_type, verbose,
    )
