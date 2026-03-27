# Creating a Game

This guide walks through implementing a complete game for pymcts, using **Tic-tac-toe** as the example. By the end, you'll have a working game that plugs into the training pipeline.

## What you need to implement

1. **`MyGameState(GameState)`** — a class holding your game's state
2. **`MyGame(BaseGame)`** — the game logic, implementing all abstract methods

## Step 1: Define your game state

The game state holds whatever information the neural network needs to evaluate a position. The engine treats it as opaque — only your neural net's `encode()` method will read it.

```python
import numpy as np
from pymcts.core.base_game import GameState

class TicTacToeState(GameState):
    """Holds the board from the current player's perspective."""

    def __init__(self, board: np.ndarray):
        self.board = board  # 3x3 array: 1=current player, -1=opponent, 0=empty
```

!!! tip "Keep states lightweight"
    The state is created on every `get_state()` call (once per MCTS simulation step). Don't store large objects or do expensive computation here.

## Step 2: Define your game

Inherit from `BaseGame` (or `Board2DGame` for grid games) and implement all abstract methods.

```python
import torch
from pymcts.core.base_game import BaseGame

class TicTacToeGame(BaseGame):
    """Tic-tac-toe: 3x3 board, 2 players, 9 possible actions."""

    def __init__(self):
        self._board = np.zeros((3, 3), dtype=np.int8)
        self._current_player = 0  # 0 or 1
        self._winner = None
        self._game_over = False
```

### Properties

```python
    @property
    def current_player(self) -> int:
        return self._current_player

    @property
    def is_over(self) -> bool:
        return self._game_over

    @property
    def winner(self) -> int | None:
        return self._winner

    @property
    def action_space_size(self) -> int:
        return 9  # 3x3 board, one action per cell
```

!!! note "action_space_size is fixed"
    This is the total number of actions that could **ever** be legal, not how many are legal right now. For Tic-tac-toe, there are always 9 cells, even if some are occupied. The mask filters out illegal moves.

### Game state and mask

```python
    def get_state(self) -> TicTacToeState:
        """Return board from current player's perspective."""
        # Player 0 sees the board as-is
        # Player 1 sees the board flipped (their pieces are +1, opponent's are -1)
        if self._current_player == 0:
            canonical = self._board.copy()
        else:
            canonical = -self._board.copy()
        return TicTacToeState(canonical)

    def to_mask(self) -> torch.Tensor:
        """1D boolean tensor: True where the cell is empty."""
        return torch.from_numpy((self._board == 0).flatten())
```

!!! important "Canonicalization"
    `get_state()` must always return the state from the **current player's perspective**. The neural net is trained to always evaluate "as if I'm player 0". How you achieve this is up to you — Tic-tac-toe just negates the board.

    If your game canonicalizes, also override `get_display_state()` to return the **absolute** state (no flipping/transposing). This is used by visualizers. If your game doesn't canonicalize (e.g. symmetric games), you can skip it — the default falls back to `get_state()`.

### Actions

```python
    def make_action(self, action: int) -> None:
        """Place a mark at the given position (0-8)."""
        row, col = divmod(action, 3)

        if self._board[row, col] != 0:
            raise ValueError(f"Cell ({row}, {col}) is already occupied")

        # Current player's mark is always +1 on the internal board
        # Player 0 uses +1, player 1 uses -1
        mark = 1 if self._current_player == 0 else -1
        self._board[row, col] = mark

        # Check win
        if self._check_win(mark):
            self._winner = self._current_player
            self._game_over = True
        elif np.all(self._board != 0):
            self._game_over = True  # draw

        # Switch player
        self._current_player = 1 - self._current_player

    def _check_win(self, mark: int) -> bool:
        """Check if the given mark has three in a row."""
        b = self._board
        # Rows, columns, diagonals
        for i in range(3):
            if np.all(b[i, :] == mark) or np.all(b[:, i] == mark):
                return True
        if b[0, 0] == b[1, 1] == b[2, 2] == mark:
            return True
        if b[0, 2] == b[1, 1] == b[2, 0] == mark:
            return True
        return False
```

### Copy and result

```python
    def copy(self) -> "TicTacToeGame":
        new = TicTacToeGame()
        new._board = self._board.copy()
        new._current_player = self._current_player
        new._winner = self._winner
        new._game_over = self._game_over
        return new

    def get_result(self, player: int) -> float | None:
        if not self._game_over:
            return None
        if self._winner is None:
            return 0.0  # draw
        return 1.0 if self._winner == player else -1.0

    def get_config(self) -> dict:
        return {}  # no configurable parameters
```

## Step 3: Test with RandomPlayer

Before building a neural net, verify your game works with random play:

```python
from pymcts.core.players import RandomPlayer

game = TicTacToeGame()
p1 = RandomPlayer(name="random1")
p2 = RandomPlayer(name="random2")

while not game.is_over:
    player = p1 if game.current_player == 0 else p2
    action = player.get_action(game)
    game.make_action(action)

print(f"Winner: {game.winner}")  # 0, 1, or None (draw)
```

Run this many times to verify:

- Games always terminate
- Both players can win
- Draws happen
- `valid_actions()` shrinks as the game progresses
- `to_mask()` matches `valid_actions()`

## Using Board2DGame

For grid-based games, `Board2DGame` saves you from writing action↔coordinate conversions:

```python
from pymcts.core.base_game import Board2DGame

class TicTacToeGame(Board2DGame):
    def __init__(self):
        super().__init__(board_rows=3, board_cols=3)
        # action_space_size is now 9 automatically
        # self.action_to_row_col(4) → (1, 1)
        # self.row_col_to_action(2, 0) → 6
```

This is optional — `BaseGame` works fine if your game doesn't map cleanly to a grid.

## Complete code

??? example "Full TicTacToeGame implementation (click to expand)"

    ```python
    import numpy as np
    import torch
    from pymcts.core.base_game import BaseGame, GameState


    class TicTacToeState(GameState):
        def __init__(self, board: np.ndarray):
            self.board = board


    class TicTacToeGame(BaseGame):

        def __init__(self):
            self._board = np.zeros((3, 3), dtype=np.int8)
            self._current_player = 0
            self._winner = None
            self._game_over = False

        @property
        def current_player(self) -> int:
            return self._current_player

        @property
        def is_over(self) -> bool:
            return self._game_over

        @property
        def winner(self) -> int | None:
            return self._winner

        @property
        def action_space_size(self) -> int:
            return 9

        def get_state(self) -> TicTacToeState:
            if self._current_player == 0:
                canonical = self._board.copy()
            else:
                canonical = -self._board.copy()
            return TicTacToeState(canonical)

        def to_mask(self) -> torch.Tensor:
            return torch.from_numpy((self._board == 0).flatten())

        def make_action(self, action: int) -> None:
            row, col = divmod(action, 3)
            if self._board[row, col] != 0:
                raise ValueError(f"Cell ({row}, {col}) is occupied")
            mark = 1 if self._current_player == 0 else -1
            self._board[row, col] = mark
            if self._check_win(mark):
                self._winner = self._current_player
                self._game_over = True
            elif np.all(self._board != 0):
                self._game_over = True
            self._current_player = 1 - self._current_player

        def _check_win(self, mark: int) -> bool:
            b = self._board
            for i in range(3):
                if np.all(b[i, :] == mark) or np.all(b[:, i] == mark):
                    return True
            if b[0, 0] == b[1, 1] == b[2, 2] == mark:
                return True
            if b[0, 2] == b[1, 1] == b[2, 0] == mark:
                return True
            return False

        def copy(self) -> "TicTacToeGame":
            new = TicTacToeGame()
            new._board = self._board.copy()
            new._current_player = self._current_player
            new._winner = self._winner
            new._game_over = self._game_over
            return new

        def get_result(self, player: int) -> float | None:
            if not self._game_over:
                return None
            if self._winner is None:
                return 0.0
            return 1.0 if self._winner == player else -1.0

        def get_config(self) -> dict:
            return {}
    ```

## Next step

Now [create a neural network](creating-a-neural-net.md) for your game.
