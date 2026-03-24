"""Visualization utilities for Bridgit."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import plotly.graph_objects as go
import torch

from pymcts.games.bridgit.config import BoardConfig
from pymcts.games.bridgit.game import BridgitGame, BridgitGameState
from pymcts.games.bridgit.player import Player

if TYPE_CHECKING:
    from pymcts.core.game_record import GameRecord
    from pymcts.core.mcts import MCTSNode


class Visualizer:
    """Plotly-based visualizations for Bridgit."""

    @staticmethod
    def visualize_game_state(state: BridgitGameState) -> go.Figure:
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
                    eps = BridgitGame._endpoints(r, c, player)
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
                eps = BridgitGame._endpoints(r, c, player)
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
        config = BoardConfig(**record.game_config)
        g = config.grid_size

        # Replay game to collect board states at each step
        game = BridgitGame(config)
        states: list[BridgitGameState] = [game.get_state()]
        for rec in record.moves:
            game.make_action(rec.action)
            states.append(game.get_state())

        # Map player int ids to names
        player_names = record.player_names

        def _action_to_board_coords(action: int, player_id: int) -> tuple[int, int]:
            """Convert canonical action back to board (row, col)."""
            row, col = divmod(action, g)
            if player_id == BridgitGame._ID_FROM_PLAYER[Player.VERTICAL]:
                row, col = col, row
            return row, col

        def _make_frame_traces(state: BridgitGameState, step: int) -> list[go.Scatter]:
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
                    eps = BridgitGame._endpoints(r, c, player)
                    (r0, c0), (r1, c1) = eps

                    # Highlight the most recent move
                    is_latest = False
                    if step > 0:
                        move_rec = record.moves[step - 1]
                        mr, mc = _action_to_board_coords(move_rec.action, move_rec.player)
                        is_latest = (r == mr and c == mc)

                    width = 7 if is_latest else 4
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
                mr, mc = _action_to_board_coords(rec.action, rec.player)
                pname = player_names[rec.player]
                title = f"Move {i}: {pname} ({mr},{mc})"
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
                dict(args=[[str(i)], dict(frame=dict(duration=0, redraw=True),
                                          transition=dict(duration=0), mode="immediate")],
                     label=str(i), method="animate")
                for i in range(len(frames))
            ],
        )]

        # Play/pause buttons (instant frame changes, no smooth animation)
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
                                        mode="immediate", transition=dict(duration=0))]),
            ],
        )]

        winner_name = record.winner_name() or "draw"
        names_str = " vs ".join(player_names)
        fig.update_layout(
            width=150 + g * 50, height=200 + g * 50,
            title=f"{names_str} — Winner: {winner_name} in {record.num_moves} moves",
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
    def save_game_html(record: GameRecord, path: str) -> None:
        """Save an interactive game visualization as a standalone HTML file."""
        fig = Visualizer.visualize_game(record)
        fig.write_html(path, include_plotlyjs=True, full_html=True,
                       auto_play=False)

    @staticmethod
    def visualize_node(node: MCTSNode) -> go.Figure:
        """Visualize an MCTS node: board state with children info overlaid."""
        game: BridgitGame = node.game
        state = game.get_state()
        fig = Visualizer.visualize_game_state(state)

        # Node info in title
        path_str = str(node.path) if node.path else "(root)"
        q_str = f"{node.q_value:+.3f}" if node.visit_count > 0 else "N/A"
        player_name = game._current_player.name
        title = (
            f"Node {path_str} | player={player_name} | "
            f"visits={node.visit_count} | Q={q_str}"
        )

        if not node.children:
            fig.update_layout(title=title)
            return fig

        g = game._g

        # Overlay children info on the board
        annotations = []
        for action, child in node.children.items():
            # Convert canonical action to board coords
            row, col = divmod(action, g)
            if game._current_player == Player.VERTICAL:
                row, col = col, row

            child_q = f"{child.q_value:+.3f}" if child.visit_count > 0 else "?"
            label = (
                f"idx={child.child_index}<br>"
                f"V={child.visit_count}<br>"
                f"Q={child_q}<br>"
                f"P={child.prior:.2f}"
            )
            annotations.append(dict(
                x=col, y=row,
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
