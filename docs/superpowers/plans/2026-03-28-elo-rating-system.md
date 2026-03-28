# Elo Rating System Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Maximum Likelihood Elo rating system with Swiss-style tournaments and training integration.

**Architecture:** New `src/pymcts/elo/` package with three modules (config, rating, tournament). Refactor `batched_arena()` to accept `BasePlayer` instead of `BaseNeuralNet`. Update `trainer.py` to use players and optionally track Elo.

**Tech Stack:** Pydantic, scipy (new dep), pytest, existing pymcts core

---

### Task 1: Add scipy dependency

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add scipy to dependencies**

In `pyproject.toml`, add `"scipy>=1.10.0"` to the `dependencies` list:

```toml
dependencies = [
    "jupyter>=1.1.1",
    "numpy>=1.24.0",
    "plotly>=6.6.0",
    "pydantic>=2.12.5",
    "pygame>=2.5.0",
    "pytest>=9.0.2",
    "scipy>=1.10.0",
    "torch>=2.10.0",
    "tqdm>=4.60.0",
]
```

- [ ] **Step 2: Install**

Run: `uv sync`
Expected: scipy installed successfully

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "add scipy dependency for ML Elo computation"
```

---

### Task 2: Elo config models (`src/pymcts/elo/config.py`)

**Files:**
- Create: `src/pymcts/elo/__init__.py`
- Create: `src/pymcts/elo/config.py`
- Test: `test/test_elo/test_config.py`

- [ ] **Step 1: Create test directory and write failing tests**

Create `test/test_elo/__init__.py` (empty) and `test/test_elo/test_config.py`:

```python
from pymcts.elo.config import MatchResult, EloRating, TournamentConfig, TournamentResult


class TestMatchResult:
    def test_create(self):
        m = MatchResult(player_a="alice", player_b="bob", wins_a=7, wins_b=3, draws=0)
        assert m.player_a == "alice"
        assert m.wins_a == 7
        assert m.total_games == 10

    def test_json_roundtrip(self):
        m = MatchResult(player_a="alice", player_b="bob", wins_a=7, wins_b=3, draws=0)
        data = m.model_dump_json()
        m2 = MatchResult.model_validate_json(data)
        assert m == m2


class TestEloRating:
    def test_create(self):
        r = EloRating(name="alice", rating=1500.0, games_played=10)
        assert r.name == "alice"
        assert r.rating == 1500.0


class TestTournamentConfig:
    def test_defaults(self):
        c = TournamentConfig()
        assert c.games_per_matchup == 40
        assert c.swap_players is True
        assert c.num_rounds is None
        assert c.convergence_threshold == 10.0
        assert c.batch_size == 8


class TestTournamentResult:
    def test_json_roundtrip(self):
        result = TournamentResult(
            ratings=[EloRating(name="random", rating=1000.0, games_played=10)],
            match_results=[
                MatchResult(player_a="random", player_b="alice", wins_a=3, wins_b=7, draws=0)
            ],
            anchor_player="random",
            anchor_rating=1000.0,
            timestamp="2026-03-28T12:00:00",
            metadata={"game": "bridgit"},
        )
        data = result.model_dump_json()
        result2 = TournamentResult.model_validate_json(data)
        assert result == result2
        assert result2.ratings[0].rating == 1000.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest test/test_elo/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'pymcts.elo'`

- [ ] **Step 3: Create the elo package and config module**

Create `src/pymcts/elo/__init__.py` (empty for now) and `src/pymcts/elo/config.py`:

```python
"""Pydantic models for the Elo rating system."""

from pydantic import BaseModel


class MatchResult(BaseModel):
    """Outcome of a matchup between two players (multiple games)."""
    player_a: str
    player_b: str
    wins_a: int
    wins_b: int
    draws: int

    @property
    def total_games(self) -> int:
        return self.wins_a + self.wins_b + self.draws


class EloRating(BaseModel):
    """Elo rating for a single player."""
    name: str
    rating: float
    games_played: int


class TournamentConfig(BaseModel):
    """Configuration for a Swiss-style tournament."""
    games_per_matchup: int = 40
    swap_players: bool = True
    num_rounds: int | None = None
    convergence_threshold: float = 10.0
    batch_size: int = 8


class TournamentResult(BaseModel):
    """Complete result of a tournament: ratings + match history."""
    ratings: list[EloRating]
    match_results: list[MatchResult]
    anchor_player: str
    anchor_rating: float
    timestamp: str
    metadata: dict
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest test/test_elo/test_config.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/pymcts/elo/ test/test_elo/
git commit -m "add elo config models (MatchResult, EloRating, TournamentConfig, TournamentResult)"
```

---

### Task 3: ML Elo rating computation (`src/pymcts/elo/rating.py`)

**Files:**
- Create: `src/pymcts/elo/rating.py`
- Test: `test/test_elo/test_rating.py`

- [ ] **Step 1: Write failing tests**

Create `test/test_elo/test_rating.py`:

```python
import pytest

from pymcts.elo.config import MatchResult, EloRating
from pymcts.elo.rating import compute_elo_ratings


class TestComputeEloRatings:
    def test_anchor_is_respected(self):
        """The anchor player should be exactly at anchor_rating."""
        results = [
            MatchResult(player_a="random", player_b="alice", wins_a=2, wins_b=8, draws=0),
        ]
        ratings = compute_elo_ratings(results, anchor_player="random", anchor_rating=1000.0)
        by_name = {r.name: r for r in ratings}
        assert by_name["random"].rating == pytest.approx(1000.0, abs=0.1)

    def test_stronger_player_rated_higher(self):
        """A player who wins more should have a higher rating."""
        results = [
            MatchResult(player_a="random", player_b="alice", wins_a=2, wins_b=8, draws=0),
            MatchResult(player_a="random", player_b="bob", wins_a=5, wins_b=5, draws=0),
        ]
        ratings = compute_elo_ratings(results, anchor_player="random", anchor_rating=1000.0)
        by_name = {r.name: r for r in ratings}
        assert by_name["alice"].rating > by_name["bob"].rating
        assert by_name["bob"].rating == pytest.approx(1000.0, abs=50.0)

    def test_symmetric_results_equal_ratings(self):
        """Two players with identical results against the anchor should have equal ratings."""
        results = [
            MatchResult(player_a="random", player_b="alice", wins_a=3, wins_b=7, draws=0),
            MatchResult(player_a="random", player_b="bob", wins_a=3, wins_b=7, draws=0),
        ]
        ratings = compute_elo_ratings(results, anchor_player="random", anchor_rating=1000.0)
        by_name = {r.name: r for r in ratings}
        assert by_name["alice"].rating == pytest.approx(by_name["bob"].rating, abs=1.0)

    def test_ratings_sorted_descending(self):
        """Ratings should be returned sorted by rating descending."""
        results = [
            MatchResult(player_a="random", player_b="strong", wins_a=1, wins_b=9, draws=0),
            MatchResult(player_a="random", player_b="weak", wins_a=8, wins_b=2, draws=0),
        ]
        ratings = compute_elo_ratings(results, anchor_player="random", anchor_rating=1000.0)
        assert ratings[0].name == "strong"
        assert ratings[-1].name == "weak"

    def test_games_played_counted(self):
        """games_played should reflect total games for each player."""
        results = [
            MatchResult(player_a="random", player_b="alice", wins_a=2, wins_b=8, draws=0),
            MatchResult(player_a="alice", player_b="bob", wins_a=6, wins_b=4, draws=0),
        ]
        ratings = compute_elo_ratings(results, anchor_player="random", anchor_rating=1000.0)
        by_name = {r.name: r for r in ratings}
        assert by_name["random"].games_played == 10
        assert by_name["alice"].games_played == 20
        assert by_name["bob"].games_played == 10

    def test_single_player_returns_anchor(self):
        """With no match results, only the anchor should be returned if present."""
        ratings = compute_elo_ratings([], anchor_player="random", anchor_rating=1000.0)
        assert ratings == []

    def test_extreme_winrate(self):
        """Even 100% win rates should produce finite ratings."""
        results = [
            MatchResult(player_a="random", player_b="god", wins_a=0, wins_b=20, draws=0),
        ]
        ratings = compute_elo_ratings(results, anchor_player="random", anchor_rating=1000.0)
        by_name = {r.name: r for r in ratings}
        assert by_name["god"].rating > 1000.0
        assert by_name["god"].rating < 5000.0  # finite, not infinity
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest test/test_elo/test_rating.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'pymcts.elo.rating'`

- [ ] **Step 3: Implement compute_elo_ratings**

Create `src/pymcts/elo/rating.py`:

```python
"""Maximum Likelihood Elo rating computation."""

from collections import defaultdict

import numpy as np
from scipy.optimize import minimize

from pymcts.elo.config import EloRating, MatchResult


def compute_elo_ratings(
    match_results: list[MatchResult],
    anchor_player: str = "random",
    anchor_rating: float = 1000.0,
) -> list[EloRating]:
    """Compute ML Elo ratings from match results.

    Finds ratings that maximize the likelihood of observed results.
    The anchor_player is fixed at anchor_rating; all others are optimized.

    Returns ratings sorted descending by rating.
    """
    if not match_results:
        return []

    # Collect unique players and count games
    games_played: dict[str, int] = defaultdict(int)
    for m in match_results:
        total = m.total_games
        games_played[m.player_a] += total
        games_played[m.player_b] += total

    players = sorted(games_played.keys())

    if anchor_player not in players:
        players.append(anchor_player)
        games_played[anchor_player] = 0

    # Separate anchor from optimized players
    free_players = [p for p in players if p != anchor_player]
    if not free_players:
        return [EloRating(name=anchor_player, rating=anchor_rating, games_played=games_played[anchor_player])]

    player_to_idx = {p: i for i, p in enumerate(free_players)}

    def _negative_log_likelihood(ratings_free: np.ndarray) -> float:
        # Build full rating dict
        rating_map = {anchor_player: anchor_rating}
        for i, p in enumerate(free_players):
            rating_map[p] = ratings_free[i]

        nll = 0.0
        for m in match_results:
            ra = rating_map[m.player_a]
            rb = rating_map[m.player_b]
            # Expected score for player_a
            exp_a = 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))
            # Clamp to avoid log(0)
            exp_a = np.clip(exp_a, 1e-10, 1.0 - 1e-10)
            exp_b = 1.0 - exp_a

            if m.wins_a > 0:
                nll -= m.wins_a * np.log(exp_a)
            if m.wins_b > 0:
                nll -= m.wins_b * np.log(exp_b)

        return nll

    # Initial guess: all free players at anchor rating
    x0 = np.full(len(free_players), anchor_rating)

    result = minimize(
        _negative_log_likelihood,
        x0,
        method="L-BFGS-B",
        options={"maxiter": 1000, "ftol": 1e-12},
    )

    # Build output
    ratings = [EloRating(name=anchor_player, rating=anchor_rating, games_played=games_played[anchor_player])]
    for i, p in enumerate(free_players):
        ratings.append(EloRating(name=p, rating=float(result.x[i]), games_played=games_played[p]))

    ratings.sort(key=lambda r: r.rating, reverse=True)
    return ratings
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest test/test_elo/test_rating.py -v`
Expected: All 7 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/pymcts/elo/rating.py test/test_elo/test_rating.py
git commit -m "add ML Elo rating computation via scipy optimization"
```

---

### Task 4: Refactor `batched_arena()` to accept `BasePlayer`

**Files:**
- Modify: `src/pymcts/core/arena.py`
- Test: `test/test_core/test_players.py` (existing arena tests)

- [ ] **Step 1: Write failing test for new player-based signature**

Add to `test/test_core/test_players.py`:

```python
from pymcts.core.arena import batched_arena


class TestBatchedArenaWithPlayers:
    def test_random_vs_random(self):
        """batched_arena should work with two RandomPlayers."""
        p1 = RandomPlayer(name="p1")
        p2 = RandomPlayer(name="p2")
        result = batched_arena(
            player_a=p1,
            player_b=p2,
            game_factory=TicTacToe,
            num_games=10,
            swap_players=True,
        )
        assert len(result) == 10
        scores = result.scores
        assert scores.get("p1", 0) + scores.get("p2", 0) <= 10

    def test_mcts_vs_random(self):
        """batched_arena should work with MCTSPlayer vs RandomPlayer."""
        net = DummyNet()
        config = MCTSConfig(num_simulations=5)
        p1 = MCTSPlayer(net, config, name="mcts")
        p2 = RandomPlayer(name="random")
        result = batched_arena(
            player_a=p1,
            player_b=p2,
            game_factory=TicTacToe,
            num_games=4,
            swap_players=True,
        )
        assert len(result) == 4

    def test_mcts_vs_mcts(self):
        """batched_arena should work with two MCTSPlayers using batched inference."""
        net_a = DummyNet()
        net_b = DummyNet()
        config = MCTSConfig(num_simulations=5)
        p1 = MCTSPlayer(net_a, config, name="a")
        p2 = MCTSPlayer(net_b, config, name="b")
        result = batched_arena(
            player_a=p1,
            player_b=p2,
            game_factory=TicTacToe,
            num_games=4,
            swap_players=True,
        )
        assert len(result) == 4
```

Also add the missing imports at the top of `test/test_core/test_players.py`:

```python
from test.test_core.test_mcts import TicTacToe, DummyNet
from pymcts.core.players import BasePlayer, RandomPlayer, MCTSPlayer
from pymcts.core.arena import Arena, batched_arena
from pymcts.core.config import MCTSConfig
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest test/test_core/test_players.py::TestBatchedArenaWithPlayers -v`
Expected: FAIL — `batched_arena()` doesn't accept `BasePlayer` arguments

- [ ] **Step 3: Rewrite `batched_arena()` to accept `BasePlayer`**

Rewrite `src/pymcts/core/arena.py`. The key change: the function accepts two `BasePlayer` instances. If both are `MCTSPlayer`, it extracts their `.mcts` engines for batched inference. Otherwise, it falls back to calling `player.get_action()` sequentially.

```python
"""Arena: run evaluation games between two players."""

import logging
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
    name_a = player_a.name
    name_b = player_b.name

    # Check if we can use batched MCTS inference
    both_mcts = isinstance(player_a, MCTSPlayer) and isinstance(player_b, MCTSPlayer)

    if both_mcts:
        return _batched_mcts_arena(
            player_a, player_b, game_factory, num_games,
            batch_size, swap_players, temperature, game_type, verbose,
        )
    else:
        return _sequential_arena(
            player_a, player_b, game_factory, num_games,
            swap_players, game_type, verbose,
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
    name_a = player_a.name
    name_b = player_b.name
    half = num_games // 2 if swap_players else num_games
    completed: list[GameRecord] = []

    pbar = tqdm(total=num_games, desc=f"{name_a} vs {name_b}", leave=False) if verbose else None

    for game_idx in range(num_games):
        swapped = swap_players and game_idx >= half
        game = game_factory()

        if swapped:
            players = [player_b, player_a]
            names = [name_b, name_a]
        else:
            players = [player_a, player_b]
            names = [name_a, name_b]

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

        completed.append(GameRecord(
            game_type=game_type,
            game_config=game.get_config(),
            moves=moves,
            winner=game.winner,
            player_names=names,
        ))

        if pbar is not None:
            pbar.update(1)

    if pbar is not None:
        pbar.close()

    return GameRecordCollection(game_records=completed)


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
    """Play arena games with batched MCTS inference (both players must be MCTSPlayer)."""
    name_a = player_a.name
    name_b = player_b.name
    mcts_a = player_a.mcts
    mcts_b = player_b.mcts

    half = num_games // 2 if swap_players else num_games
    active_size = min(batch_size, num_games)

    games: list[BaseGame] = []
    move_histories: list[list[MoveRecord]] = []
    slot_names: list[list[str]] = []
    slot_mcts: list[dict[int, MCTS]] = []

    def _make_slot(game_idx: int) -> tuple[BaseGame, list, list[str], dict[int, MCTS]]:
        swapped = swap_players and game_idx >= half
        game = game_factory()
        if swapped:
            names = [name_b, name_a]
            mcts_map = {0: mcts_b, 1: mcts_a}
        else:
            names = [name_a, name_b]
            mcts_map = {0: mcts_a, 1: mcts_b}
        return game, [], names, mcts_map

    for i in range(active_size):
        game, hist, names, mcts_map = _make_slot(i)
        games.append(game)
        move_histories.append(hist)
        slot_names.append(names)
        slot_mcts.append(mcts_map)

    completed: list[GameRecord] = []
    games_started = active_size
    recorded: set[int] = set()

    pbar = None
    if verbose:
        pbar = tqdm(total=num_games, desc=f"{name_a} vs {name_b}", leave=False)

    wins_a = 0
    wins_b = 0
    first_player_wins = 0
    second_player_wins = 0

    while len(completed) < num_games:
        active_idx = [
            i for i in range(len(games))
            if not games[i].is_over and i not in recorded
        ]
        if not active_idx:
            break

        groups: dict[int, list[int]] = {}
        for i in active_idx:
            current = games[i].current_player
            mcts_obj = slot_mcts[i][current]
            key = id(mcts_obj)
            if key not in groups:
                groups[key] = []
            groups[key].append(i)

        mcts_by_id = {}
        for i in active_idx:
            current = games[i].current_player
            mcts_obj = slot_mcts[i][current]
            mcts_by_id[id(mcts_obj)] = mcts_obj

        for mcts_id, slot_indices in groups.items():
            mcts_obj = mcts_by_id[mcts_id]
            batch_games = [games[i] for i in slot_indices]
            roots = mcts_obj.search_batch(batch_games)

            for j, i in enumerate(slot_indices):
                root = roots[j]
                action_space = games[i].action_space_size
                visit_counts = root.visit_counts(action_space)
                probs = MCTS.visit_counts_to_probs(visit_counts, temperature)

                if probs.sum() == 0:
                    probs = games[i].to_mask().float()

                if temperature > 0:
                    action = torch.multinomial(probs, 1).item()
                else:
                    action = torch.argmax(probs).item()

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
                    player_names=slot_names[i],
                )
                completed.append(record)

                if record.winner is not None:
                    winner_name = record.player_names[record.winner]
                    if winner_name == name_a:
                        wins_a += 1
                    else:
                        wins_b += 1
                    if record.winner == 0:
                        first_player_wins += 1
                    else:
                        second_player_wins += 1

                if pbar is not None:
                    pbar.update(1)
                    pbar.set_postfix_str(
                        f"{name_a}={wins_a} | {name_b}={wins_b} | "
                        f"1st={first_player_wins} 2nd={second_player_wins}"
                    )

                if games_started < num_games:
                    game, hist, names, mcts_map = _make_slot(games_started)
                    games[i] = game
                    move_histories[i] = hist
                    slot_names[i] = names
                    slot_mcts[i] = mcts_map
                    games_started += 1
                else:
                    recorded.add(i)

    if pbar is not None:
        pbar.close()

    return GameRecordCollection(game_records=completed)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest test/test_core/test_players.py -v`
Expected: All tests PASS (both old TestArena and new TestBatchedArenaWithPlayers)

- [ ] **Step 5: Commit**

```bash
git add src/pymcts/core/arena.py test/test_core/test_players.py
git commit -m "refactor batched_arena to accept BasePlayer instead of BaseNeuralNet"
```

---

### Task 5: Update `trainer.py` to use players

**Files:**
- Modify: `src/pymcts/core/trainer.py`

- [ ] **Step 1: Run existing tests to establish baseline**

Run: `pytest test/ -v`
Expected: All tests PASS (baseline before modifying trainer)

- [ ] **Step 2: Update trainer.py to create players and pass them to batched_arena**

In `src/pymcts/core/trainer.py`, update the imports to add:

```python
from pymcts.core.players import GreedyMCTSPlayer
```

Then replace the arena evaluation section. The current code at lines 148-168 creates `prev_net` and calls `batched_arena(net_a=net, net_b=prev_net, mcts_config=...)`. Replace with player creation:

Replace lines 153-168 (the `prev_net` and `batched_arena` call):

```python
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
```

Similarly, replace lines 182-198 (the historical checkpoint loop). Replace the body of the `for past_iter, past_path in best_checkpoints:` loop:

```python
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
```

- [ ] **Step 3: Run tests to verify nothing broke**

Run: `pytest test/ -v`
Expected: All tests PASS

- [ ] **Step 4: Commit**

```bash
git add src/pymcts/core/trainer.py
git commit -m "update trainer to create players for arena evaluation"
```

---

### Task 6: Update `core/__init__.py` re-exports

**Files:**
- Modify: `src/pymcts/core/__init__.py`

- [ ] **Step 1: Update re-exports**

The `batched_arena` signature changed. The re-export in `src/pymcts/core/__init__.py` is already correct (it re-exports the function, not its types). No change needed to the import line itself.

Verify by running: `python -c "from pymcts.core import batched_arena; print('OK')"`
Expected: `OK`

- [ ] **Step 2: Commit (skip if no changes)**

No commit needed if re-exports are already correct.

---

### Task 7: Swiss tournament (`src/pymcts/elo/tournament.py`)

**Files:**
- Create: `src/pymcts/elo/tournament.py`
- Test: `test/test_elo/test_tournament.py`

- [ ] **Step 1: Write failing tests**

Create `test/test_elo/test_tournament.py`:

```python
from pymcts.core.players import RandomPlayer
from pymcts.elo.config import TournamentConfig, MatchResult
from pymcts.elo.tournament import RatedPlayer, run_tournament

# Reuse TicTacToe from test_mcts
from test.test_core.test_mcts import TicTacToe


class TestRatedPlayer:
    def test_from_random(self):
        rp = RatedPlayer.from_random()
        assert rp.name == "random"
        player = rp.player_factory()
        assert isinstance(player, RandomPlayer)

    def test_from_random_custom_name(self):
        rp = RatedPlayer.from_random(name="rng_bot")
        assert rp.name == "rng_bot"


class TestSwissPairing:
    def test_no_repeat_matchups(self):
        """Tournament should not repeat the same matchup."""
        players = [RatedPlayer.from_random(name=f"p{i}") for i in range(4)]
        config = TournamentConfig(games_per_matchup=4, num_rounds=2, batch_size=1)
        result = run_tournament(
            players=players,
            game_factory=TicTacToe,
            config=config,
        )
        # Check no duplicate matchups
        seen = set()
        for m in result.match_results:
            pair = tuple(sorted([m.player_a, m.player_b]))
            assert pair not in seen, f"Duplicate matchup: {pair}"
            seen.add(pair)

    def test_ratings_computed(self):
        """Tournament should produce ratings for all players."""
        players = [RatedPlayer.from_random(name=f"p{i}") for i in range(3)]
        config = TournamentConfig(games_per_matchup=4, num_rounds=2, batch_size=1)
        result = run_tournament(
            players=players,
            game_factory=TicTacToe,
            config=config,
        )
        rated_names = {r.name for r in result.ratings}
        player_names = {p.name for p in players}
        assert player_names == rated_names

    def test_anchor_player(self):
        """The first player named 'random' should be the anchor at 1000."""
        players = [
            RatedPlayer.from_random(),
            RatedPlayer.from_random(name="rng2"),
            RatedPlayer.from_random(name="rng3"),
        ]
        config = TournamentConfig(games_per_matchup=4, num_rounds=2, batch_size=1)
        result = run_tournament(
            players=players,
            game_factory=TicTacToe,
            config=config,
        )
        assert result.anchor_player == "random"

    def test_previous_results_avoids_replays(self):
        """Providing previous_results should skip already-played matchups."""
        players = [RatedPlayer.from_random(name=f"p{i}") for i in range(3)]
        config = TournamentConfig(games_per_matchup=4, num_rounds=2, batch_size=1)

        result1 = run_tournament(players=players, game_factory=TicTacToe, config=config)
        played_before = len(result1.match_results)

        result2 = run_tournament(
            players=players, game_factory=TicTacToe, config=config,
            previous_results=result1,
        )
        # Should have the old matches plus potentially new ones (but no duplicates)
        all_pairs = set()
        for m in result2.match_results:
            pair = tuple(sorted([m.player_a, m.player_b]))
            assert pair not in all_pairs
            all_pairs.add(pair)


class TestRunTournament:
    def test_small_tournament_runs(self):
        """Smoke test: a 4-player tournament completes without error."""
        players = [RatedPlayer.from_random(name=f"rng_{i}") for i in range(4)]
        config = TournamentConfig(games_per_matchup=4, num_rounds=2, batch_size=1)
        result = run_tournament(
            players=players,
            game_factory=TicTacToe,
            config=config,
        )
        assert len(result.ratings) == 4
        assert len(result.match_results) > 0
        assert result.timestamp != ""

    def test_json_roundtrip(self):
        """Tournament result should survive JSON serialization."""
        players = [RatedPlayer.from_random(name=f"rng_{i}") for i in range(3)]
        config = TournamentConfig(games_per_matchup=4, num_rounds=1, batch_size=1)
        result = run_tournament(
            players=players,
            game_factory=TicTacToe,
            config=config,
        )
        from pymcts.elo.config import TournamentResult
        data = result.model_dump_json()
        loaded = TournamentResult.model_validate_json(data)
        assert len(loaded.ratings) == len(result.ratings)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest test/test_elo/test_tournament.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'pymcts.elo.tournament'`

- [ ] **Step 3: Implement tournament.py**

Create `src/pymcts/elo/tournament.py`:

```python
"""Swiss-style tournament with Elo ratings."""

import math
import random as rng
from datetime import datetime, timezone
from typing import Callable

from pydantic import BaseModel, ConfigDict

from pymcts.core.arena import batched_arena
from pymcts.core.base_game import BaseGame
from pymcts.core.players import BasePlayer, MCTSPlayer, RandomPlayer
from pymcts.elo.config import (
    EloRating,
    MatchResult,
    TournamentConfig,
    TournamentResult,
)
from pymcts.elo.rating import compute_elo_ratings


class RatedPlayer(BaseModel):
    """A player that can participate in rated tournaments."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    player_factory: Callable[[], BasePlayer]

    @classmethod
    def from_mcts_player(cls, player: MCTSPlayer) -> "RatedPlayer":
        """Create a RatedPlayer that produces a copy of the given MCTSPlayer."""
        name = player.name
        def factory(p=player):
            return p
        return cls(name=name, player_factory=factory)

    @classmethod
    def from_random(cls, name: str = "random") -> "RatedPlayer":
        """Create a RatedPlayer wrapping a RandomPlayer."""
        def factory(n=name):
            return RandomPlayer(name=n)
        return cls(name=name, player_factory=factory)


def _swiss_pair(
    players: list[str],
    ratings: dict[str, float],
    played: set[tuple[str, str]],
) -> list[tuple[str, str]]:
    """Pair players using Swiss system: sort by rating, pair adjacent, avoid repeats."""
    sorted_players = sorted(players, key=lambda p: ratings.get(p, 0.0), reverse=True)
    paired: set[str] = set()
    pairs: list[tuple[str, str]] = []

    for i, p in enumerate(sorted_players):
        if p in paired:
            continue
        # Find closest-rated unpaired opponent without repeat
        for j in range(i + 1, len(sorted_players)):
            q = sorted_players[j]
            if q in paired:
                continue
            match_key = tuple(sorted([p, q]))
            if match_key in played:
                continue
            pairs.append((p, q))
            paired.add(p)
            paired.add(q)
            break

    return pairs


def _collect_games_played(match_results: list[MatchResult]) -> dict[str, int]:
    """Count total games played per player from match results."""
    counts: dict[str, int] = {}
    for m in match_results:
        total = m.total_games
        counts[m.player_a] = counts.get(m.player_a, 0) + total
        counts[m.player_b] = counts.get(m.player_b, 0) + total
    return counts


def run_tournament(
    players: list[RatedPlayer],
    game_factory: Callable[[], BaseGame],
    config: TournamentConfig,
    previous_results: TournamentResult | None = None,
) -> TournamentResult:
    """Run a Swiss-style tournament.

    If previous_results is provided, its match history is included
    and already-played matchups are skipped.
    """
    player_map = {p.name: p for p in players}
    player_names = [p.name for p in players]

    # Determine anchor
    anchor = "random" if "random" in player_map else player_names[0]

    # Seed from previous results
    match_results: list[MatchResult] = []
    played: set[tuple[str, str]] = set()
    if previous_results is not None:
        match_results = list(previous_results.match_results)
        for m in match_results:
            played.add(tuple(sorted([m.player_a, m.player_b])))

    # Current ratings (for pairing)
    current_ratings: dict[str, float] = {name: 1000.0 for name in player_names}

    # Number of rounds
    num_rounds = config.num_rounds or max(3, math.ceil(math.log2(max(len(players), 2))))

    for round_num in range(num_rounds):
        # First round: shuffle for random pairing
        if round_num == 0 and not previous_results:
            shuffled = list(player_names)
            rng.shuffle(shuffled)
            pairs = []
            for i in range(0, len(shuffled) - 1, 2):
                pair_key = tuple(sorted([shuffled[i], shuffled[i + 1]]))
                if pair_key not in played:
                    pairs.append((shuffled[i], shuffled[i + 1]))
        else:
            pairs = _swiss_pair(player_names, current_ratings, played)

        if not pairs:
            break  # No more unpaired matchups possible

        # Play each pair
        for name_a, name_b in pairs:
            pa = player_map[name_a].player_factory()
            pb = player_map[name_b].player_factory()

            records = batched_arena(
                player_a=pa,
                player_b=pb,
                game_factory=game_factory,
                num_games=config.games_per_matchup,
                batch_size=config.batch_size,
                swap_players=config.swap_players,
                verbose=False,
            )

            scores = records.scores
            wins_a = scores.get(name_a, 0)
            wins_b = scores.get(name_b, 0)
            draws = len(records) - wins_a - wins_b

            match_results.append(MatchResult(
                player_a=name_a,
                player_b=name_b,
                wins_a=wins_a,
                wins_b=wins_b,
                draws=draws,
            ))
            played.add(tuple(sorted([name_a, name_b])))

        # Recompute ratings
        elo_ratings = compute_elo_ratings(match_results, anchor_player=anchor)
        new_ratings = {r.name: r.rating for r in elo_ratings}

        # Check convergence
        if round_num > 0:
            max_change = max(
                abs(new_ratings.get(n, 1000.0) - current_ratings.get(n, 1000.0))
                for n in player_names
            )
            if max_change < config.convergence_threshold:
                current_ratings = new_ratings
                break

        current_ratings = new_ratings

    # Final rating computation
    final_ratings = compute_elo_ratings(match_results, anchor_player=anchor)

    return TournamentResult(
        ratings=final_ratings,
        match_results=match_results,
        anchor_player=anchor,
        anchor_rating=1000.0,
        timestamp=datetime.now(timezone.utc).isoformat(),
        metadata={},
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest test/test_elo/test_tournament.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/pymcts/elo/tournament.py test/test_elo/test_tournament.py
git commit -m "add Swiss-style tournament with RatedPlayer and run_tournament"
```

---

### Task 8: Elo package `__init__.py` re-exports

**Files:**
- Modify: `src/pymcts/elo/__init__.py`

- [ ] **Step 1: Add re-exports**

Write `src/pymcts/elo/__init__.py`:

```python
from pymcts.elo.config import EloRating, MatchResult, TournamentConfig, TournamentResult
from pymcts.elo.rating import compute_elo_ratings
from pymcts.elo.tournament import RatedPlayer, run_tournament
```

- [ ] **Step 2: Verify imports work**

Run: `python -c "from pymcts.elo import compute_elo_ratings, run_tournament, RatedPlayer; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add src/pymcts/elo/__init__.py
git commit -m "add elo package re-exports"
```

---

### Task 9: Training integration — config + Elo tracking in trainer

**Files:**
- Modify: `src/pymcts/core/config.py`
- Modify: `src/pymcts/core/trainer.py`
- Test: `test/test_elo/test_training_integration.py`

- [ ] **Step 1: Write failing test**

Create `test/test_elo/test_training_integration.py`:

```python
from pymcts.core.config import TrainingConfig


class TestTrainingConfigElo:
    def test_elo_fields_exist(self):
        config = TrainingConfig()
        assert config.elo_tracking is False
        assert config.elo_reference_interval == 5
        assert config.elo_games_per_matchup == 40

    def test_elo_tracking_enabled(self):
        config = TrainingConfig(elo_tracking=True, elo_reference_interval=3)
        assert config.elo_tracking is True
        assert config.elo_reference_interval == 3
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest test/test_elo/test_training_integration.py -v`
Expected: FAIL — `TrainingConfig` doesn't have `elo_tracking` field

- [ ] **Step 3: Add Elo fields to TrainingConfig**

In `src/pymcts/core/config.py`, add three fields to `TrainingConfig`:

```python
class TrainingConfig(BaseModel):
    """Training loop settings."""
    num_iterations: int = 50
    num_self_play_games: int = 50
    num_epochs: int = 20
    batch_size: int = 64
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    replay_buffer_size: int = 5
    self_play_batch_size: int = 8
    elo_tracking: bool = False
    elo_reference_interval: int = 5
    elo_games_per_matchup: int = 40
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest test/test_elo/test_training_integration.py -v`
Expected: All 2 tests PASS

- [ ] **Step 5: Add Elo tracking to trainer.py**

In `src/pymcts/core/trainer.py`, add these imports at the top:

```python
from pymcts.core.players import GreedyMCTSPlayer, RandomPlayer
from pymcts.elo.config import MatchResult, TournamentResult, EloRating
from pymcts.elo.rating import compute_elo_ratings
```

Add Elo initialization after the `replay_buffer` and `best_checkpoints` setup (after line 87):

```python
    # Elo tracking state
    elo_match_results: list[MatchResult] = []
    elo_reference_pool: list[tuple[str, GreedyMCTSPlayer | RandomPlayer]] = []
    if training_config.elo_tracking:
        elo_reference_pool.append(("random", RandomPlayer(name="random")))
```

Add Elo evaluation block after the `iter_data_path.write_text(...)` line (after line 241), inside the iteration loop:

```python
        # Elo tracking
        if training_config.elo_tracking:
            if verbose:
                print("  Elo evaluation...")

            current_player = GreedyMCTSPlayer(net, mcts_config, name=f"iter_{iteration:03d}")

            for ref_name, ref_player in elo_reference_pool:
                ref_records = batched_arena(
                    player_a=current_player,
                    player_b=ref_player,
                    game_factory=game_factory,
                    num_games=training_config.elo_games_per_matchup,
                    batch_size=training_config.elo_games_per_matchup,
                    swap_players=True,
                    game_type="elo",
                    verbose=verbose,
                )
                scores = ref_records.scores
                wins_a = scores.get(current_player.name, 0)
                wins_b = scores.get(ref_name, 0)
                draws = len(ref_records) - wins_a - wins_b

                elo_match_results.append(MatchResult(
                    player_a=current_player.name,
                    player_b=ref_name,
                    wins_a=wins_a,
                    wins_b=wins_b,
                    draws=draws,
                ))

            elo_ratings = compute_elo_ratings(elo_match_results, anchor_player="random")
            current_elo = next(
                (r.rating for r in elo_ratings if r.name == current_player.name),
                1000.0,
            )

            if verbose:
                print(f"  Elo: {current_elo:.0f}")

            # Add to reference pool at interval
            if iteration % training_config.elo_reference_interval == 0:
                ref_net = net.copy()
                ref_player_new = GreedyMCTSPlayer(ref_net, mcts_config, name=f"ref_iter_{iteration:03d}")
                elo_reference_pool.append((ref_player_new.name, ref_player_new))

            # Save Elo results
            elo_result = TournamentResult(
                ratings=elo_ratings,
                match_results=elo_match_results,
                anchor_player="random",
                anchor_rating=1000.0,
                timestamp=datetime.now().isoformat(),
                metadata={"training_run": str(run_dir), "iteration": iteration},
            )
            elo_path = run_dir / "elo_results.json"
            elo_path.write_text(elo_result.model_dump_json(indent=2))

            # Add elo to iteration data
            iteration_data["elo"] = current_elo
            iter_data_path.write_text(json.dumps(iteration_data, indent=2))
```

- [ ] **Step 6: Run all tests**

Run: `pytest test/ -v`
Expected: All tests PASS

- [ ] **Step 7: Commit**

```bash
git add src/pymcts/core/config.py src/pymcts/core/trainer.py test/test_elo/test_training_integration.py
git commit -m "add Elo tracking integration to training loop"
```

---

### Task 10: Full integration test

**Files:**
- Create: `test/test_elo/test_integration.py`

- [ ] **Step 1: Write integration test**

Create `test/test_elo/test_integration.py`:

```python
import json
from pathlib import Path

from test.test_core.test_mcts import TicTacToe, DummyNet

from pymcts.core.config import MCTSConfig
from pymcts.core.players import MCTSPlayer, RandomPlayer
from pymcts.elo.config import TournamentConfig, TournamentResult
from pymcts.elo.tournament import RatedPlayer, run_tournament


class TestEloIntegration:
    def test_tournament_with_mixed_players(self):
        """Run a tournament with MCTSPlayer and RandomPlayer together."""
        net = DummyNet()
        config = MCTSConfig(num_simulations=5)
        mcts_player = MCTSPlayer(net, config, name="dummy_mcts")

        players = [
            RatedPlayer.from_random(),
            RatedPlayer.from_mcts_player(mcts_player),
            RatedPlayer.from_random(name="rng2"),
        ]
        t_config = TournamentConfig(games_per_matchup=4, num_rounds=2, batch_size=1)
        result = run_tournament(
            players=players,
            game_factory=TicTacToe,
            config=t_config,
        )
        assert len(result.ratings) == 3
        assert result.anchor_player == "random"
        by_name = {r.name: r for r in result.ratings}
        assert by_name["random"].rating == 1000.0

    def test_order_independent_ratings(self):
        """Same matchups should produce identical ratings regardless of input order."""
        players_a = [
            RatedPlayer.from_random(name="p0"),
            RatedPlayer.from_random(name="p1"),
            RatedPlayer.from_random(name="p2"),
        ]
        players_b = [
            RatedPlayer.from_random(name="p2"),
            RatedPlayer.from_random(name="p0"),
            RatedPlayer.from_random(name="p1"),
        ]

        # Use same seed for reproducible random games
        import random
        random.seed(42)
        config = TournamentConfig(games_per_matchup=10, num_rounds=2, batch_size=1)
        result_a = run_tournament(players=players_a, game_factory=TicTacToe, config=config)

        # Re-run with same match_results (not re-playing, just recomputing)
        from pymcts.elo.rating import compute_elo_ratings
        ratings_recomputed = compute_elo_ratings(result_a.match_results, anchor_player="p0")

        by_name_orig = {r.name: r.rating for r in result_a.ratings}
        by_name_recomp = {r.name: r.rating for r in ratings_recomputed}

        for name in by_name_orig:
            assert abs(by_name_orig[name] - by_name_recomp[name]) < 0.01

    def test_json_persistence_roundtrip(self, tmp_path: Path):
        """Save tournament result to JSON, load it, verify ratings match."""
        players = [RatedPlayer.from_random(name=f"p{i}") for i in range(3)]
        config = TournamentConfig(games_per_matchup=4, num_rounds=1, batch_size=1)
        result = run_tournament(players=players, game_factory=TicTacToe, config=config)

        # Save
        path = tmp_path / "elo_results.json"
        path.write_text(result.model_dump_json(indent=2))

        # Load
        loaded = TournamentResult.model_validate_json(path.read_text())
        assert len(loaded.ratings) == len(result.ratings)
        for orig, load in zip(result.ratings, loaded.ratings):
            assert orig.name == load.name
            assert abs(orig.rating - load.rating) < 0.01
```

- [ ] **Step 2: Run integration tests**

Run: `pytest test/test_elo/test_integration.py -v`
Expected: All 3 tests PASS

- [ ] **Step 3: Run full test suite**

Run: `pytest test/ -v`
Expected: All tests PASS

- [ ] **Step 4: Commit**

```bash
git add test/test_elo/test_integration.py
git commit -m "add elo integration tests"
```

---

### Task 11: Final cleanup — run full suite and verify

**Files:**
- None (verification only)

- [ ] **Step 1: Run full test suite**

Run: `pytest test/ -v`
Expected: All tests PASS

- [ ] **Step 2: Verify imports work end-to-end**

Run:
```bash
python -c "
from pymcts.elo import compute_elo_ratings, run_tournament, RatedPlayer
from pymcts.elo import MatchResult, EloRating, TournamentConfig, TournamentResult
print('All imports OK')
"
```
Expected: `All imports OK`

- [ ] **Step 3: Final commit (if any stragglers)**

Only commit if there are uncommitted changes from minor fixes.
