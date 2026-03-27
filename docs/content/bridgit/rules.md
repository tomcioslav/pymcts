# Bridgit Game Rules

Bridgit is a two-player connection game invented by David Gale in the 1960s. It's the first game implemented in pymcts and serves as the reference implementation.

## Board

The game is played on a grid derived from a board size parameter **n** (default: 5). The actual grid is **(2n+1) x (2n+1)** — for n=5, that's an 11x11 grid.

The grid has three types of cells:

- **Crossings** (interior cells where row+col is even): these are the playable positions
- **Endpoints**: stamped automatically when a bridge is placed
- **Boundary cells**: goal positions for each player

## Players

- **Player 0 (Horizontal)**: connects left edge to right edge
- **Player 1 (Vertical)**: connects top edge to bottom edge

## Gameplay

1. Player 0 goes first with **1 move**
2. After that, players alternate with **2 moves per turn**
3. On each move, the player places a bridge at an empty crossing
4. A bridge claims the crossing and its two endpoints (in the player's direction)
5. The first player to create a connected path from their start edge to their end edge wins

## Winning

Win detection uses Union-Find. When a bridge is placed, its endpoints are unioned with adjacent same-player cells. A player wins when their start sentinel and end sentinel are in the same connected component.

## Canonicalization

Bridgit uses board transposition for canonicalization:

- **Player 0 (Horizontal)**: the board is returned as-is
- **Player 1 (Vertical)**: the board is transposed and negated

This means the neural network always sees the board from the same perspective — the current player is always trying to connect left-to-right. Actions are also transposed: if the current player is Vertical, `make_action()` un-transposes the coordinates before placing.

## Implementation details

The Bridgit implementation lives in `pymcts.games.bridgit`:

- `BridgitGame(Board2DGame)` — the game logic
- `BridgitGameState(GameState)` — holds canonical board, n, and moves_left_in_turn
- `BridgitNet(BaseNeuralNet)` — ResNet with 4 input channels
- `Player` enum — HORIZONTAL (-1) and VERTICAL (1), internal to Bridgit

### Neural net encoding

`BridgitNet.encode()` creates 4 channels from a `BridgitGameState`:

1. Current player's cells (board == -1)
2. Opponent's cells (board == 1)
3. Playability mask (empty crossings)
4. Moves left in turn (constant plane: 1 or 2)

The board is always canonical (transposed for Vertical), so channel semantics are consistent regardless of which player is moving.
