"""Batched self-play: run N games concurrently with batched neural net inference."""

import logging
from typing import Callable

import numpy as np
import torch
from tqdm.auto import tqdm

from bridgit.core.base_game import BaseGame
from bridgit.core.base_neural_net import BaseNeuralNet
from bridgit.core.config import MCTSConfig
from bridgit.core.game_record import GameRecord, GameRecordCollection, MoveRecord
from bridgit.core.mcts import MCTS, MCTSNode

logger = logging.getLogger("bridgit.core.self_play")

_VIRTUAL_LOSS = 1.0


class BatchedMCTS:
    """MCTS that batches neural net calls across multiple trees.

    Supports virtual loss and lazy expansion.
    """

    def __init__(self, net: BaseNeuralNet, mcts_config: MCTSConfig):
        self.net = net
        self.mcts_config = mcts_config

    def _predict_batch(self, states: list) -> list[tuple[torch.Tensor, float]]:
        """Run neural net on a batch of game states."""
        log_policies, values = self.net.predict_batch(states)
        policies = torch.exp(log_policies).cpu()
        vals = values.cpu()

        return [
            (policies[i], vals[i].item())
            for i in range(len(states))
        ]

    @staticmethod
    def _set_priors(node: MCTSNode, policy: torch.Tensor, value: float) -> float:
        """Store masked/renormalized priors in node.unexpanded_moves."""
        if node.game.is_over:
            node.is_expanded = True
            return 1.0

        valid_mask = node.game.to_mask().float()

        policy = policy * valid_mask
        policy_sum = policy.sum()
        if policy_sum > 0:
            policy = policy / policy_sum
        else:
            total = valid_mask.sum()
            if total > 0:
                policy = valid_mask / total
            else:
                node.is_expanded = True
                return value

        for action in node.game.valid_actions():
            node.unexpanded_moves[action] = policy[action].item()

        node.is_expanded = True
        return value

    @staticmethod
    def _select_leaf(root: MCTSNode, c_puct: float) -> MCTSNode:
        """Select a leaf node using lazy expansion."""
        node = root
        while node.is_expanded and node.has_candidates:
            node = node.best_child_or_expand(c_puct)
            if not node.is_expanded:
                break
        return node

    @staticmethod
    def _apply_virtual_loss(node: MCTSNode):
        """Apply virtual loss along the path from node to root."""
        n = node
        while n is not None:
            n.visit_count += 1
            n.value_sum -= _VIRTUAL_LOSS
            n = n.parent

    @staticmethod
    def _undo_virtual_loss(node: MCTSNode):
        """Undo virtual loss along the path from node to root."""
        n = node
        while n is not None:
            n.visit_count -= 1
            n.value_sum += _VIRTUAL_LOSS
            n = n.parent

    @staticmethod
    def _backpropagate(node: MCTSNode, value: float):
        """Backpropagate value up the tree."""
        while node is not None:
            node.visit_count += 1
            node.value_sum += value
            if (node.parent is not None
                    and node.game.current_player != node.parent.game.current_player):
                value = -value
            node = node.parent

    @staticmethod
    def _add_dirichlet_noise(node: MCTSNode, alpha: float, epsilon: float):
        """Add Dirichlet noise to root priors."""
        all_actions = list(node.children.keys()) + list(node.unexpanded_moves.keys())
        if not all_actions:
            return
        noise = np.random.dirichlet([alpha] * len(all_actions))
        idx = 0
        for action in list(node.children.keys()):
            node.children[action].prior = (
                (1 - epsilon) * node.children[action].prior + epsilon * noise[idx]
            )
            idx += 1
        for action in list(node.unexpanded_moves.keys()):
            node.unexpanded_moves[action] = (
                (1 - epsilon) * node.unexpanded_moves[action] + epsilon * noise[idx]
            )
            idx += 1

    def search_batch(self, games: list[BaseGame]) -> list[MCTSNode]:
        """Run MCTS for multiple games with batched inference + virtual loss."""
        n = len(games)
        k = self.mcts_config.num_parallel_leaves
        roots = [MCTSNode(g.copy()) for g in games]

        # Initial expansion — batch all roots
        predictions = self._predict_batch([r.game.get_state() for r in roots])
        for i in range(n):
            policy, value = predictions[i]
            self._set_priors(roots[i], policy, value)
            self._add_dirichlet_noise(
                roots[i],
                self.mcts_config.dirichlet_alpha,
                self.mcts_config.dirichlet_epsilon,
            )

        c_puct = self.mcts_config.c_puct
        num_sims = self.mcts_config.num_simulations
        sim = 0

        while sim < num_sims:
            chunk = min(k, num_sims - sim)

            to_predict_leaves: list[MCTSNode] = []
            to_predict_states: list = []

            for i in range(n):
                for _ in range(chunk):
                    leaf = self._select_leaf(roots[i], c_puct)

                    if leaf.game.is_over:
                        self._backpropagate(leaf, 1.0)
                    elif leaf.is_expanded and not leaf.has_candidates:
                        self._backpropagate(
                            leaf, leaf.q_value if leaf.visit_count > 0 else 0.0,
                        )
                    else:
                        self._apply_virtual_loss(leaf)
                        to_predict_leaves.append(leaf)
                        to_predict_states.append(leaf.game.get_state())

            if to_predict_states:
                predictions = self._predict_batch(to_predict_states)

                for j, leaf in enumerate(to_predict_leaves):
                    policy, value = predictions[j]
                    self._undo_virtual_loss(leaf)
                    value = self._set_priors(leaf, policy, value)
                    self._backpropagate(leaf, value)

            sim += chunk

        return roots


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
    """Play self-play games with batched MCTS inference.

    Runs `batch_size` games concurrently with virtual loss
    (mcts_config.num_parallel_leaves) for maximum GPU throughput.
    """
    batched_mcts = BatchedMCTS(net, mcts_config)

    active_size = min(batch_size, num_games)
    games = [game_factory() for _ in range(active_size)]
    move_histories: list[list[MoveRecord]] = [[] for _ in range(active_size)]

    completed: list[GameRecord] = []
    games_started = active_size
    # Track which slots have already been recorded (finished but not replaced)
    recorded: set[int] = set()

    pbar = None
    if verbose:
        pbar = tqdm(total=num_games, desc="Self-play")

    while len(completed) < num_games:
        active_idx = [
            i for i in range(len(games))
            if not games[i].is_over and i not in recorded
        ]
        if not active_idx:
            break

        active_games = [games[i] for i in active_idx]
        roots = batched_mcts.search_batch(active_games)

        for j, i in enumerate(active_idx):
            root = roots[j]
            action_space = games[i].action_space_size
            visit_counts = root.visit_counts(action_space)
            probs = MCTS.visit_counts_to_probs(visit_counts, temperature)

            if probs.sum() == 0:
                probs = games[i].to_mask().float()

            action = torch.multinomial(probs, 1).item()

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
                    player_names=["self-play-0", "self-play-1"],
                )
                completed.append(record)

                if pbar is not None:
                    pbar.update(1)

                if games_started < num_games:
                    games[i] = game_factory()
                    move_histories[i] = []
                    games_started += 1
                else:
                    recorded.add(i)

    if pbar is not None:
        pbar.close()

    logger.info("Batched self-play: %d games completed", len(completed))
    return GameRecordCollection(game_records=completed)
