import random

import torch

from pymcts.games.bridgit.game import BridgitGame, BridgitGameState
from pymcts.games.bridgit.config import BoardConfig
from pymcts.core.base_game import BaseGame, Board2DGame, GameState


class TestBridgitGame:
    def test_is_base_game(self):
        game = BridgitGame()
        assert isinstance(game, BaseGame)
        assert isinstance(game, Board2DGame)

    def test_action_space_size(self):
        game = BridgitGame(BoardConfig(size=5))
        g = 2 * 5 + 1
        assert game.action_space_size == g * g

    def test_initial_state(self):
        game = BridgitGame()
        assert game.current_player == 0
        assert not game.is_over
        assert game.winner is None

    def test_mask_is_1d(self):
        game = BridgitGame()
        mask = game.to_mask()
        assert mask.dim() == 1
        assert mask.shape[0] == game.action_space_size
        assert mask.dtype == torch.bool

    def test_get_state_returns_bridgit_state(self):
        game = BridgitGame()
        state = game.get_state()
        assert isinstance(state, GameState)
        assert isinstance(state, BridgitGameState)

    def test_make_action_and_player_switch(self):
        game = BridgitGame(BoardConfig(size=3))
        actions = game.valid_actions()
        game.make_action(actions[0])
        assert game.current_player in (0, 1)

    def test_copy_independent(self):
        game = BridgitGame()
        actions = game.valid_actions()
        game.make_action(actions[0])
        copy = game.copy()
        copy_actions = copy.valid_actions()
        copy.make_action(copy_actions[0])
        assert game.to_mask().sum() != copy.to_mask().sum()

    def test_game_completes(self):
        """Random play should eventually end."""
        game = BridgitGame(BoardConfig(size=3))
        for _ in range(200):
            if game.is_over:
                break
            actions = game.valid_actions()
            game.make_action(random.choice(actions))
        assert game.is_over
        assert game.winner is not None

    def test_canonical_mask_matches_valid_actions(self):
        game = BridgitGame(BoardConfig(size=3))
        mask = game.to_mask()
        valid = game.valid_actions()
        for a in valid:
            assert mask[a].item() is True

    def test_get_result(self):
        game = BridgitGame(BoardConfig(size=3))
        assert game.get_result(0) is None  # game not over
        # Play until done
        for _ in range(200):
            if game.is_over:
                break
            actions = game.valid_actions()
            game.make_action(random.choice(actions))
        assert game.get_result(game.winner) == 1.0
        other = 1 - game.winner
        assert game.get_result(other) == -1.0
