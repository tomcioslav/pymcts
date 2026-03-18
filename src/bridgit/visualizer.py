"""Visualization utilities for Bridgit."""

from __future__ import annotations

import numpy as np
import torch
import plotly.graph_objects as go

from bridgit.config import BoardConfig
from bridgit.game.state import GameState
from bridgit.schema.move import Move
from bridgit.schema.player import Player

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bridgit.ai.mcts import MCTSNode
    from bridgit.schema.game_record import GameRecord


class Visualizer:
    """Plotly-based visualizations for Bridgit."""

    @staticmethod
    def visualize_game_state(state: GameState) -> go.Figure:
        """Visualize a single board state."""
        g = 2 * state.n + 1
        fig = go.Figure()

        # Boundary endpoint dots
        green_x, green_y = [], []
        red_x, red_y = [], []
        for c in range(1, g - 1, 2):
            green_x.extend([c, c])
            green_y.extend([0, g - 1])
        for r in range(1, g - 1, 2):
            red_x.extend([0, g - 1])
            red_y.extend([r, r])

        fig.add_trace(go.Scatter(
            x=green_x, y=green_y, mode="markers",
            marker=dict(size=8, color="green", opacity=0.3),
            showlegend=False, hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=red_x, y=red_y, mode="markers",
            marker=dict(size=8, color="red", opacity=0.3),
            showlegend=False, hoverinfo="skip",
        ))

        # Ghost bridges (potential moves)
        for r in range(1, g - 1):
            for c in range(1, g - 1):
                if (r + c) % 2 != 0 or state.board[r, c] != 0:
                    continue
                for player, color in [(Player.VERTICAL, "green"), (Player.HORIZONTAL, "red")]:
                    eps = state.endpoints(r, c, player)
                    (r0, c0), (r1, c1) = eps
                    fig.add_shape(type="line",
                        x0=c0, x1=c1, y0=r0, y1=r1,
                        line=dict(color=color, width=2),
                        opacity=0.2)

        # Placed bridges and their endpoints
        for r in range(1, g - 1):
            for c in range(1, g - 1):
                if (r + c) % 2 != 0:
                    continue
                val = state.board[r, c]
                if val == 0:
                    continue
                player = Player.VERTICAL if val == 1 else Player.HORIZONTAL
                color = "green" if val == 1 else "red"
                eps = state.endpoints(r, c, player)
                (r0, c0), (r1, c1) = eps
                fig.add_shape(type="line",
                    x0=c0, x1=c1, y0=r0, y1=r1,
                    line=dict(color=color, width=5))
                fig.add_trace(go.Scatter(
                    x=[c0, c1], y=[r0, r1], mode="markers",
                    marker=dict(size=8, color=color),
                    showlegend=False, hoverinfo="skip",
                ))

        fig.update_layout(
            width=120 + g * 50, height=120 + g * 50,
            xaxis=dict(
                range=[-0.5, g - 0.5], scaleanchor="y",
                tickmode="array",
                tickvals=list(range(g)),
                ticktext=[str(i) for i in range(g)],
                title="col",
            ),
            yaxis=dict(
                range=[g - 0.5, -0.5], autorange=False,
                tickmode="array",
                tickvals=list(range(g)),
                ticktext=[str(i) for i in range(g)],
                title="row",
            ),
            plot_bgcolor="white",
            margin=dict(l=50, r=30, t=30, b=50),
        )

        return fig

    @staticmethod
    def visualize_array(
        array: np.ndarray | torch.Tensor,
        title: str = "",
        colorscale: str = "Blues",
    ) -> go.Figure:
        """Visualize a 2D array (mask, policy, visit counts, etc.) as a heatmap."""
        if isinstance(array, torch.Tensor):
            array = array.detach().cpu().numpy()

        fig = go.Figure()
        fig.add_trace(go.Heatmap(
            z=np.flipud(array),
            colorscale=colorscale,
            showscale=True,
        ))
        fig.update_layout(
            width=400, height=400,
            title=title,
        )
        return fig

    @staticmethod
    def visualize_game(record: GameRecord) -> go.Figure:
        """Interactive plotly visualization of a full game with a move slider."""
        from bridgit.game import Bridgit

        board_config = BoardConfig(size=record.board_size)
        g = record.grid_size

        # Replay game to collect states at each step
        game = Bridgit(board_config)
        states: list[GameState] = [game.state]
        for rec in record.moves:
            game.make_move(rec.move)
            states.append(game.state)

        def _make_frame_traces(state: GameState, step: int) -> list[go.Scatter]:
            traces = []

            # Boundary endpoint dots
            green_x, green_y = [], []
            red_x, red_y = [], []
            for c in range(1, g - 1, 2):
                green_x.extend([c, c])
                green_y.extend([0, g - 1])
            for r in range(1, g - 1, 2):
                red_x.extend([0, g - 1])
                red_y.extend([r, r])

            traces.append(go.Scatter(
                x=green_x, y=green_y, mode="markers",
                marker=dict(size=8, color="green", opacity=0.3),
                showlegend=False, hoverinfo="skip",
            ))
            traces.append(go.Scatter(
                x=red_x, y=red_y, mode="markers",
                marker=dict(size=8, color="red", opacity=0.3),
                showlegend=False, hoverinfo="skip",
            ))

            # Played bridges and their endpoints
            for r in range(1, g - 1):
                for c in range(1, g - 1):
                    if (r + c) % 2 != 0:
                        continue
                    val = state.board[r, c]
                    if val == 0:
                        continue
                    player = Player.VERTICAL if val == 1 else Player.HORIZONTAL
                    color = "green" if val == 1 else "red"
                    eps = state.endpoints(r, c, player)
                    (r0, c0), (r1, c1) = eps

                    width = 7 if step > 0 and record.moves[step - 1].move == Move(row=r, col=c) else 4
                    traces.append(go.Scatter(
                        x=[c0, c1], y=[r0, r1], mode="lines+markers",
                        line=dict(color=color, width=width),
                        marker=dict(size=8, color=color),
                        showlegend=False, hoverinfo="skip",
                    ))

            return traces

        # Build raw frame trace lists and find the max trace count
        raw_frames: list[tuple[list[go.Scatter], str]] = []
        for i, state in enumerate(states):
            frame_traces = _make_frame_traces(state, i)
            if i == 0:
                title = "Start"
            else:
                rec = record.moves[i - 1]
                title = f"Move {i}: {rec.player.name} ({rec.move.row},{rec.move.col})"
            raw_frames.append((frame_traces, title))

        max_traces = max(len(traces) for traces, _ in raw_frames)

        # Pad all frames to the same trace count (plotly needs consistent trace count)
        def _pad(traces: list[go.Scatter]) -> list[go.Scatter]:
            while len(traces) < max_traces:
                traces.append(go.Scatter(
                    x=[None], y=[None], mode="markers",
                    marker=dict(size=0, opacity=0),
                    showlegend=False, hoverinfo="skip",
                ))
            return traces

        frames = [
            go.Frame(
                data=_pad(traces),
                name=str(i),
                layout=go.Layout(title_text=title),
            )
            for i, (traces, title) in enumerate(raw_frames)
        ]

        fig = go.Figure(data=frames[0].data, frames=frames)

        # Slider
        sliders = [dict(
            active=0,
            currentvalue=dict(prefix="Step: "),
            steps=[
                dict(args=[[str(i)], dict(frame=dict(duration=0, redraw=True), mode="immediate")],
                     label=str(i), method="animate")
                for i in range(len(frames))
            ],
        )]

        # Play/pause buttons
        updatemenus = [dict(
            type="buttons",
            showactive=False,
            y=1.15, x=0.5, xanchor="center",
            buttons=[
                dict(label="Play", method="animate",
                     args=[None, dict(frame=dict(duration=500, redraw=True),
                                      fromcurrent=True, transition=dict(duration=0))]),
                dict(label="Pause", method="animate",
                     args=[[None], dict(frame=dict(duration=0, redraw=False),
                                        mode="immediate")]),
            ],
        )]

        h_name = record.horizontal_player
        v_name = record.vertical_player
        winner_name = h_name if record.winner == Player.HORIZONTAL else v_name
        fig.update_layout(
            width=150 + g * 50, height=200 + g * 50,
            title=f"{h_name} (H) vs {v_name} (V) — Winner: {winner_name} in {record.num_moves} moves",
            xaxis=dict(
                range=[-0.5, g - 0.5], scaleanchor="y",
                tickmode="array",
                tickvals=list(range(g)),
                title="col",
            ),
            yaxis=dict(
                range=[g - 0.5, -0.5], autorange=False,
                tickmode="array",
                tickvals=list(range(g)),
                title="row",
            ),
            plot_bgcolor="white",
            margin=dict(l=50, r=30, t=80, b=50),
            sliders=sliders,
            updatemenus=updatemenus,
        )

        return fig

    @staticmethod
    def visualize_node(node: MCTSNode) -> go.Figure:
        """Visualize an MCTS node: board state with children info overlaid.

        Shows the current board state and, for each child, displays
        its child_index, move, visit count, Q-value, and prior on the
        board cell where the move would be placed.
        """
        state = node.game.state
        fig = Visualizer.visualize_game_state(state)

        # Node info in title
        path_str = str(node.path) if node.path else "(root)"
        q_str = f"{node.q_value:+.3f}" if node.visit_count > 0 else "N/A"
        solved_str = f" | solved={node.solved_value:+.0f}" if node.solved_value is not None else ""
        title = (
            f"Node {path_str} | player={node.game.current_player.name} | "
            f"visits={node.visit_count} | Q={q_str}{solved_str}"
        )

        if not node.children:
            fig.update_layout(title=title)
            return fig

        # Overlay children info on the board
        annotations = []
        for move, child in node.children.items():
            # The action is in canonical space; decanonicalize for board position
            actual_move = move.decanonicalize(node.game.current_player)
            r, c = actual_move.row, actual_move.col

            child_q = f"{child.q_value:+.3f}" if child.visit_count > 0 else "?"
            solved_tag = f" S={child.solved_value:+.0f}" if child.solved_value is not None else ""
            label = (
                f"idx={child.child_index}<br>"
                f"V={child.visit_count}<br>"
                f"Q={child_q}<br>"
                f"P={child.prior:.2f}{solved_tag}"
            )
            annotations.append(dict(
                x=c, y=r,
                text=label,
                showarrow=False,
                font=dict(size=9, color="black"),
                bgcolor="rgba(255,255,255,0.75)",
                bordercolor="gray",
                borderwidth=1,
                borderpad=2,
            ))

        fig.update_layout(title=title, annotations=annotations)
        return fig
