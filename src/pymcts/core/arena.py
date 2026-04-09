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

logger = logging.getLogger("core.arena")


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
    both_mcts = isinstance(player_a, MCTSPlayer) and isinstance(player_b, MCTSPlayer)
    if both_mcts:
        return _batched_mcts_arena(
            player_a, player_b, game_factory, num_games,
            batch_size, swap_players, temperature, game_type, verbose,
        )
    return _sequential_arena(
        player_a, player_b, game_factory, num_games,
        swap_players, game_type, verbose,
    )


# ---------------------------------------------------------------------------
# Shared context dataclass for batched arena
# ---------------------------------------------------------------------------

@dataclass
class _ArenaContext:
    """Immutable configuration shared across all slots in a batched arena run."""
    name_a: str
    name_b: str
    mcts_a: MCTS
    mcts_b: MCTS
    half: int
    swap_players: bool
    num_games: int
    game_factory: Callable[[], BaseGame]
    game_type: str
    temperature: float


@dataclass
class _ArenaState:
    """Mutable runtime state for an in-progress batched arena run."""
    games: list[BaseGame] = field(default_factory=list)
    move_histories: list[list[MoveRecord]] = field(default_factory=list)
    slot_names: list[list[str]] = field(default_factory=list)
    slot_mcts: list[dict[int, MCTS]] = field(default_factory=list)
    completed: list[GameRecord] = field(default_factory=list)
    recorded: set[int] = field(default_factory=set)
    games_started: int = 0
    wins_a: int = 0
    wins_b: int = 0
    first_player_wins: int = 0
    second_player_wins: int = 0
    pbar: object = None


# ---------------------------------------------------------------------------
# Sequential arena helpers
# ---------------------------------------------------------------------------

def _play_sequential_game(
    game: BaseGame,
    players: list[BasePlayer],
    game_type: str,
) -> tuple[list[MoveRecord], GameRecord]:
    """Play a single game to completion; return (moves, GameRecord)."""
    moves: list[MoveRecord] = []
    while not game.is_over:
        current = game.current_player
        action = players[current].get_action(game)
        moves.append(MoveRecord(
            action=action,
            player=current,
            policy=players[current].last_policy,
        ))
        game.make_action(action)
    return moves, game


def _build_sequential_record(
    game: BaseGame,
    moves: list[MoveRecord],
    names: list[str],
    game_type: str,
) -> GameRecord:
    """Wrap a finished game into a GameRecord."""
    return GameRecord(
        game_type=game_type,
        game_config=game.get_config(),
        moves=moves,
        winner=game.winner,
        player_names=names,
    )


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
    name_a, name_b = player_a.name, player_b.name
    half = num_games // 2 if swap_players else num_games
    completed: list[GameRecord] = []
    pbar = _make_progress_bar(name_a, name_b, num_games, verbose)

    for game_idx in range(num_games):
        swapped = swap_players and game_idx >= half
        players, names = _ordered_players(player_a, player_b, name_a, name_b, swapped)
        moves, finished = _play_sequential_game(game_factory(), players, game_type)
        completed.append(_build_sequential_record(finished, moves, names, game_type))
        if pbar is not None:
            pbar.update(1)

    if pbar is not None:
        pbar.close()
    return GameRecordCollection(game_records=completed)


# ---------------------------------------------------------------------------
# Batched MCTS arena helpers
# ---------------------------------------------------------------------------

def _make_progress_bar(name_a: str, name_b: str, num_games: int, verbose: bool):
    """Create a tqdm progress bar if verbose, otherwise return None."""
    if not verbose:
        return None
    return tqdm(total=num_games, desc=f"{name_a} vs {name_b}", leave=False)


def _ordered_players(
    player_a: BasePlayer,
    player_b: BasePlayer,
    name_a: str,
    name_b: str,
    swapped: bool,
) -> tuple[list[BasePlayer], list[str]]:
    """Return (players, names) ordered for a slot (swapped or normal)."""
    if swapped:
        return [player_b, player_a], [name_b, name_a]
    return [player_a, player_b], [name_a, name_b]


def _make_slot(
    game_idx: int,
    ctx: _ArenaContext,
) -> tuple[BaseGame, list[MoveRecord], list[str], dict[int, MCTS]]:
    """Create a fresh game slot with player ordering determined by game index."""
    swapped = ctx.swap_players and game_idx >= ctx.half
    game = ctx.game_factory()
    if swapped:
        return game, [], [ctx.name_b, ctx.name_a], {0: ctx.mcts_b, 1: ctx.mcts_a}
    return game, [], [ctx.name_a, ctx.name_b], {0: ctx.mcts_a, 1: ctx.mcts_b}


def _initialize_slots(ctx: _ArenaContext, active_size: int) -> _ArenaState:
    """Create the initial batch of active game slots and return a fresh state."""
    state = _ArenaState(games_started=active_size)
    for i in range(active_size):
        game, hist, names, mcts_map = _make_slot(i, ctx)
        state.games.append(game)
        state.move_histories.append(hist)
        state.slot_names.append(names)
        state.slot_mcts.append(mcts_map)
    return state


def _group_slots_by_mcts(
    active_idx: list[int],
    state: _ArenaState,
) -> tuple[dict[int, list[int]], dict[int, MCTS]]:
    """Group active slot indices by the MCTS object responsible for them."""
    groups: dict[int, list[int]] = {}
    mcts_by_id: dict[int, MCTS] = {}
    for i in active_idx:
        mcts_obj = state.slot_mcts[i][state.games[i].current_player]
        key = id(mcts_obj)
        groups.setdefault(key, []).append(i)
        mcts_by_id[key] = mcts_obj
    return groups, mcts_by_id


def _select_action(probs: torch.Tensor, game: BaseGame, temperature: float) -> int:
    """Select an action from visit-count probabilities using temperature."""
    if probs.sum() == 0:
        probs = game.to_mask().float()
    if temperature > 0:
        return torch.multinomial(probs, 1).item()
    return torch.argmax(probs).item()


def _execute_mcts_batch(
    mcts_obj: MCTS,
    slot_indices: list[int],
    state: _ArenaState,
    temperature: float,
) -> None:
    """Run batched MCTS search and apply one action per slot."""
    roots = mcts_obj.search_batch([state.games[i] for i in slot_indices])
    for j, i in enumerate(slot_indices):
        visit_counts = roots[j].visit_counts(state.games[i].action_space_size)
        probs = MCTS.visit_counts_to_probs(visit_counts, temperature)
        action = _select_action(probs, state.games[i], temperature)
        current_player = state.games[i].current_player
        state.games[i].make_action(action)
        state.move_histories[i].append(
            MoveRecord(action=action, player=current_player, policy=probs)
        )


def _update_pbar(state: _ArenaState, ctx: _ArenaContext) -> None:
    """Refresh progress bar postfix with current win counts."""
    if state.pbar is None:
        return
    state.pbar.update(1)
    state.pbar.set_postfix_str(
        f"{ctx.name_a}={state.wins_a} | {ctx.name_b}={state.wins_b} | "
        f"1st={state.first_player_wins} 2nd={state.second_player_wins}"
    )


def _refill_or_retire_slot(i: int, state: _ArenaState, ctx: _ArenaContext) -> None:
    """Replace slot i with a new game, or mark it retired when all games are started."""
    if state.games_started < ctx.num_games:
        game, hist, names, mcts_map = _make_slot(state.games_started, ctx)
        state.games[i] = game
        state.move_histories[i] = hist
        state.slot_names[i] = names
        state.slot_mcts[i] = mcts_map
        state.games_started += 1
    else:
        state.recorded.add(i)


def _record_finished_slot(i: int, state: _ArenaState, ctx: _ArenaContext) -> None:
    """Build and store a GameRecord for a completed slot; refill or retire it."""
    record = GameRecord(
        game_type=ctx.game_type,
        game_config=state.games[i].get_config(),
        moves=state.move_histories[i],
        winner=state.games[i].winner,
        player_names=state.slot_names[i],
    )
    state.completed.append(record)
    _tally_wins(record, state, ctx)
    _update_pbar(state, ctx)
    _refill_or_retire_slot(i, state, ctx)


def _tally_wins(record: GameRecord, state: _ArenaState, ctx: _ArenaContext) -> None:
    """Update win counters on state for a completed game record."""
    if record.winner is None:
        return
    winner_name = record.player_names[record.winner]
    if winner_name == ctx.name_a:
        state.wins_a += 1
    else:
        state.wins_b += 1
    if record.winner == 0:
        state.first_player_wins += 1
    else:
        state.second_player_wins += 1


def _process_completed_slots(state: _ArenaState, ctx: _ArenaContext) -> None:
    """Scan all slots; handle any that finished since the last step."""
    for i in range(len(state.games)):
        if state.games[i].is_over and i not in state.recorded:
            _record_finished_slot(i, state, ctx)


def _active_slot_indices(state: _ArenaState) -> list[int]:
    """Return indices of slots that are still in progress."""
    return [
        i for i in range(len(state.games))
        if not state.games[i].is_over and i not in state.recorded
    ]


def _run_one_step(state: _ArenaState, ctx: _ArenaContext) -> bool:
    """Run one MCTS step across all active slots. Return False if nothing to do."""
    active_idx = _active_slot_indices(state)
    if not active_idx:
        return False
    groups, mcts_by_id = _group_slots_by_mcts(active_idx, state)
    for mcts_id, slot_indices in groups.items():
        _execute_mcts_batch(mcts_by_id[mcts_id], slot_indices, state, ctx.temperature)
    _process_completed_slots(state, ctx)
    return True


def _make_arena_context(
    player_a: MCTSPlayer,
    player_b: MCTSPlayer,
    game_factory: Callable[[], BaseGame],
    num_games: int,
    swap_players: bool,
    temperature: float,
    game_type: str,
) -> _ArenaContext:
    """Build an _ArenaContext from individual arena parameters."""
    return _ArenaContext(
        name_a=player_a.name, name_b=player_b.name,
        mcts_a=player_a.mcts, mcts_b=player_b.mcts,
        half=num_games // 2 if swap_players else num_games,
        swap_players=swap_players, num_games=num_games,
        game_factory=game_factory, game_type=game_type, temperature=temperature,
    )


def _run_batched_arena(ctx: _ArenaContext, batch_size: int, verbose: bool) -> GameRecordCollection:
    """Drive a batched MCTS arena run to completion and return all game records."""
    state = _initialize_slots(ctx, min(batch_size, ctx.num_games))
    state.pbar = _make_progress_bar(ctx.name_a, ctx.name_b, ctx.num_games, verbose)
    while len(state.completed) < ctx.num_games:
        if not _run_one_step(state, ctx):
            break
    if state.pbar is not None:
        state.pbar.close()
    return GameRecordCollection(game_records=state.completed)


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
    """Build context and delegate to the batched arena runner."""
    ctx = _make_arena_context(
        player_a, player_b, game_factory, num_games, swap_players, temperature, game_type
    )
    return _run_batched_arena(ctx, batch_size, verbose)
