#!/usr/bin/env python3
"""
GUI interface for playing Bridgit using pygame.

Supports human vs human mode on the (2n+1)×(2n+1) board.
Usage:
    python play.py [board_size]
"""

import argparse

import pygame

from bridgit import Bridgit, Player
from bridgit.game import GameState

# Colors
BACKGROUND = (245, 245, 250)
GREEN_COLOR = (50, 200, 50)
RED_COLOR = (220, 50, 50)
GREEN_LIGHT = (100, 230, 100)
RED_LIGHT = (250, 100, 100)
CROSSING_COLOR = (180, 180, 190)
HOVER_ALPHA = 128
TEXT_COLOR = (40, 40, 60)
PANEL_BG = (255, 255, 255)
PANEL_BORDER = (200, 200, 210)

# Dimensions
WINDOW_WIDTH = 1000
WINDOW_HEIGHT = 700
BOARD_MARGIN = 50
PANEL_WIDTH = 280
LINE_THICKNESS = 6
DOT_RADIUS = 5
CROSSING_RADIUS = 3
GHOST_ALPHA = 50  # ~0.2 opacity (out of 255)
GHOST_THICKNESS = 2


class BridgitGUI:
    """GUI interface for Bridgit game using pygame."""

    def __init__(self, n: int = 5):
        pygame.init()

        self.game = Bridgit(n)
        self.n = n
        self.g = 2 * n + 1  # grid dimension

        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Bridgit")

        self.clock = pygame.time.Clock()
        self.running = True

        # Fonts
        self.title_font = pygame.font.Font(None, 48)
        self.text_font = pygame.font.Font(None, 32)
        self.small_font = pygame.font.Font(None, 24)

        # Calculate board layout
        board_pixel_w = WINDOW_WIDTH - PANEL_WIDTH - BOARD_MARGIN * 2
        board_pixel_h = WINDOW_HEIGHT - BOARD_MARGIN * 2
        self.cell_size = min(board_pixel_w / self.g, board_pixel_h / self.g)

        # Center the board
        self.board_offset_x = BOARD_MARGIN + (board_pixel_w - self.cell_size * self.g) / 2
        self.board_offset_y = BOARD_MARGIN + (board_pixel_h - self.cell_size * self.g) / 2

        self.hover_cell: tuple[int, int] | None = None

    def cell_center(self, row: int, col: int) -> tuple[int, int]:
        """Get pixel center of a grid cell."""
        x = self.board_offset_x + col * self.cell_size + self.cell_size / 2
        y = self.board_offset_y + row * self.cell_size + self.cell_size / 2
        return int(x), int(y)

    def get_cell_from_mouse(self, mouse_pos: tuple[int, int]) -> tuple[int, int] | None:
        """Get grid cell (row, col) from mouse position."""
        mx, my = mouse_pos
        col = int((mx - self.board_offset_x) / self.cell_size)
        row = int((my - self.board_offset_y) / self.cell_size)
        if 0 <= row < self.g and 0 <= col < self.g:
            return row, col
        return None

    def draw_bridge(self, row: int, col: int, player: Player, color: tuple, thickness: int = LINE_THICKNESS):
        """Draw a bridge line from endpoint to endpoint."""
        eps = GameState.endpoints(row, col, player)
        (r0, c0), (r1, c1) = eps
        start = self.cell_center(r0, c0)
        end = self.cell_center(r1, c1)
        pygame.draw.line(self.screen, color, start, end, thickness)

    def draw_bridge_transparent(self, row: int, col: int, player: Player, color: tuple, thickness: int = LINE_THICKNESS):
        """Draw a semi-transparent bridge preview."""
        eps = GameState.endpoints(row, col, player)
        (r0, c0), (r1, c1) = eps
        start = self.cell_center(r0, c0)
        end = self.cell_center(r1, c1)

        # Bounding box for the line
        min_x = min(start[0], end[0]) - thickness
        min_y = min(start[1], end[1]) - thickness
        max_x = max(start[0], end[0]) + thickness
        max_y = max(start[1], end[1]) + thickness
        w = max_x - min_x
        h = max_y - min_y

        surface = pygame.Surface((w, h), pygame.SRCALPHA)
        local_start = (start[0] - min_x, start[1] - min_y)
        local_end = (end[0] - min_x, end[1] - min_y)
        pygame.draw.line(surface, color, local_start, local_end, thickness)
        self.screen.blit(surface, (min_x, min_y))

    def draw_board(self):
        """Draw the game board."""
        board = self.game.grid
        g = self.g

        # 1. Draw boundary dots
        for c in range(1, g, 2):
            # Green boundary dots on top and bottom rows
            pygame.draw.circle(self.screen, GREEN_LIGHT, self.cell_center(0, c), DOT_RADIUS)
            pygame.draw.circle(self.screen, GREEN_LIGHT, self.cell_center(g - 1, c), DOT_RADIUS)
        for r in range(1, g, 2):
            # Red boundary dots on left and right columns
            pygame.draw.circle(self.screen, RED_LIGHT, self.cell_center(r, 0), DOT_RADIUS)
            pygame.draw.circle(self.screen, RED_LIGHT, self.cell_center(r, g - 1), DOT_RADIUS)

        # 2. Draw ghost bridges (potential moves) as very faint lines
        ghost_surface = pygame.Surface((self.screen.get_width(), self.screen.get_height()), pygame.SRCALPHA)
        for r in range(1, g - 1):
            for c in range(1, g - 1):
                if (r + c) % 2 != 0 or board[r, c] != 0:
                    continue
                for player, color in [(Player.VERTICAL, GREEN_COLOR), (Player.HORIZONTAL, RED_COLOR)]:
                    eps = GameState.endpoints(r, c, player)
                    (r0, c0), (r1, c1) = eps
                    start = self.cell_center(r0, c0)
                    end = self.cell_center(r1, c1)
                    pygame.draw.line(ghost_surface, color + (GHOST_ALPHA,), start, end, GHOST_THICKNESS)
        self.screen.blit(ghost_surface, (0, 0))

        # 3. Draw empty crossings as small grey dots
        for r in range(1, g - 1):
            for c in range(1, g - 1):
                if (r + c) % 2 == 0 and board[r, c] == 0:
                    pygame.draw.circle(self.screen, CROSSING_COLOR, self.cell_center(r, c), CROSSING_RADIUS)

        # 3. Draw placed bridges and their endpoint dots
        for r in range(1, g - 1):
            for c in range(1, g - 1):
                if (r + c) % 2 != 0:
                    continue
                val = board[r, c]
                if val == 0:
                    continue
                player = Player.VERTICAL if val == 1 else Player.HORIZONTAL
                color = GREEN_COLOR if val == 1 else RED_COLOR
                self.draw_bridge(r, c, player, color)
                # Draw endpoint dots
                for er, ec in GameState.endpoints(r, c, player):
                    pygame.draw.circle(self.screen, color, self.cell_center(er, ec), DOT_RADIUS)

        # 4. Draw hover preview
        if self.hover_cell and not self.game.game_over:
            row, col = self.hover_cell
            if self.game.is_valid_move(row, col):
                player = self.game.current_player
                base = GREEN_COLOR if player == Player.VERTICAL else RED_COLOR
                color = base + (HOVER_ALPHA,)
                self.draw_bridge_transparent(row, col, player, color)

    def draw_panel(self):
        """Draw the side panel with game info."""
        panel_x = WINDOW_WIDTH - PANEL_WIDTH

        # Panel background
        pygame.draw.rect(self.screen, PANEL_BG, (panel_x, 0, PANEL_WIDTH, WINDOW_HEIGHT))
        pygame.draw.line(self.screen, PANEL_BORDER, (panel_x, 0), (panel_x, WINDOW_HEIGHT), 2)

        y = 40

        # Title
        title = self.title_font.render("BRIDGIT", True, TEXT_COLOR)
        self.screen.blit(title, title.get_rect(centerx=panel_x + PANEL_WIDTH // 2, top=y))
        y += 80

        # Current player
        if not self.game.game_over:
            text = self.small_font.render("Current Turn:", True, TEXT_COLOR)
            self.screen.blit(text, text.get_rect(centerx=panel_x + PANEL_WIDTH // 2, top=y))
            y += 35

            if self.game.current_player == Player.HORIZONTAL:
                name, color = "RED (Horizontal)", RED_COLOR
            else:
                name, color = "GREEN (Vertical)", GREEN_COLOR

            text = self.text_font.render(name, True, color)
            self.screen.blit(text, text.get_rect(centerx=panel_x + PANEL_WIDTH // 2, top=y))
            y += 50

        # Player info boxes
        cx = panel_x + PANEL_WIDTH // 2

        # Green player box
        pygame.draw.rect(self.screen, GREEN_COLOR, (panel_x + 30, y, PANEL_WIDTH - 60, 100), 3)
        text = self.text_font.render("Green", True, GREEN_COLOR)
        self.screen.blit(text, text.get_rect(centerx=cx, top=y + 12))
        line_y = y + 50
        pygame.draw.line(self.screen, GREEN_COLOR, (panel_x + 60, line_y), (panel_x + PANEL_WIDTH - 60, line_y), LINE_THICKNESS)
        text = self.small_font.render("Top \u2194 Bottom", True, TEXT_COLOR)
        self.screen.blit(text, text.get_rect(centerx=cx, top=y + 70))
        y += 120

        # Red player box
        pygame.draw.rect(self.screen, RED_COLOR, (panel_x + 30, y, PANEL_WIDTH - 60, 100), 3)
        text = self.text_font.render("Red", True, RED_COLOR)
        self.screen.blit(text, text.get_rect(centerx=cx, top=y + 12))
        line_y = y + 50
        pygame.draw.line(self.screen, RED_COLOR, (panel_x + 60, line_y), (panel_x + PANEL_WIDTH - 60, line_y), LINE_THICKNESS)
        text = self.small_font.render("Left \u2194 Right", True, TEXT_COLOR)
        self.screen.blit(text, text.get_rect(centerx=cx, top=y + 70))

        # Instructions at bottom
        y = WINDOW_HEIGHT - 120
        for line in ["Click a crossing to place", "Press R to restart", "Press Q to quit"]:
            text = self.small_font.render(line, True, TEXT_COLOR)
            self.screen.blit(text, text.get_rect(centerx=cx, top=y))
            y += 30

    def draw_win_screen(self):
        """Draw the win screen overlay."""
        overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT))
        overlay.set_alpha(180)
        overlay.fill((0, 0, 0))
        self.screen.blit(overlay, (0, 0))

        if self.game.winner == Player.VERTICAL:
            winner_text, color = "GREEN WINS!", GREEN_COLOR
            subtitle = "Top-Bottom Connection Complete"
        else:
            winner_text, color = "RED WINS!", RED_COLOR
            subtitle = "Left-Right Connection Complete"

        text = self.title_font.render(winner_text, True, color)
        self.screen.blit(text, text.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 - 50)))

        text = self.text_font.render(subtitle, True, (255, 255, 255))
        self.screen.blit(text, text.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 + 10)))

        text = self.small_font.render("Press R to play again or Q to quit", True, (200, 200, 200))
        self.screen.blit(text, text.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 + 60)))

    def handle_click(self, pos: tuple[int, int]):
        """Handle mouse click."""
        if self.game.game_over:
            return
        cell = self.get_cell_from_mouse(pos)
        if cell:
            self.game.make_move(*cell)

    def run(self):
        """Main game loop."""
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    self.handle_click(event.pos)
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        self.running = False
                    elif event.key == pygame.K_r:
                        self.game = Bridgit(self.n)

            # Update hover
            self.hover_cell = self.get_cell_from_mouse(pygame.mouse.get_pos())

            # Draw
            self.screen.fill(BACKGROUND)
            self.draw_board()
            self.draw_panel()
            if self.game.game_over:
                self.draw_win_screen()

            pygame.display.flip()
            self.clock.tick(60)

        pygame.quit()


def main():
    parser = argparse.ArgumentParser(description="Play Bridgit")
    parser.add_argument("board_size", nargs="?", type=int, default=5, help="Board size (default: 5)")
    args = parser.parse_args()

    gui = BridgitGUI(n=args.board_size)
    gui.run()


if __name__ == "__main__":
    main()
