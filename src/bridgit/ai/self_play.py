"""Batched self-play: run N games concurrently with batched neural net inference."""

import logging

import numpy as np
import torch
from tqdm.auto import tqdm

from bridgit.ai.mcts import MCTS, MCTSNode
from bridgit.ai.neural_net import NetWrapper
from bridgit.config import BoardConfig, MCTSConfig
from bridgit.game import Bridgit
from bridgit.schema import GameRecord, GameRecordCollection, Move, MoveRecord
from bridgit.schema.player import Player

logger = logging.getLogger("bridgit.self_play")

_VIRTUAL_LOSS = 1.0


class BatchedMCTS:
    """MCTS that batches neural net calls across multiple trees.

    Supports virtual loss and lazy expansion.
    """

    def __init__(self, net_wrapper: NetWrapper, mcts_config: MCTSConfig):
        self.net_wrapper = net_wrapper
        self.mcts_config = mcts_config

    def _predict_batch(self, games: list[Bridgit]) -> list[tuple[torch.Tensor, float]]:
        """Run neural net on a batch of game states."""
        model = self.net_wrapper.model
        device = self.net_wrapper.device

        model.eval()
        tensors = torch.stack([g.to_tensor() for g in games]).to(device)

        with torch.no_grad():
            log_policies, values = model(tensors)

        # Keep on CPU for tree operations
        policies = torch.exp(log_policies).cpu()
        vals = values.cpu()

        return [
            (policies[i], vals[i].item())
            for i in range(len(games))
        ]

    @staticmethod
    def _set_priors(node: MCTSNode, policy: torch.Tensor, value: float) -> float:
        """Store masked/renormalized priors in node.unexpanded_moves."""
        if node.game.game_over:
            node.is_expanded = True
            return 1.0

        valid_mask = node.game.to_mask()

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

        for r, c in torch.nonzero(valid_mask, as_tuple=False):
            r, c = r.item(), c.item()
            move = Move(row=r, col=c)
            node.unexpanded_moves[move] = policy[r, c].item()

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
        all_moves = list(node.children.keys()) + list(node.unexpanded_moves.keys())
        if not all_moves:
            return
        noise = np.random.dirichlet([alpha] * len(all_moves))
        idx = 0
        for move in list(node.children.keys()):
            node.children[move].prior = (
                (1 - epsilon) * node.children[move].prior + epsilon * noise[idx]
            )
            idx += 1
        for move in list(node.unexpanded_moves.keys()):
            node.unexpanded_moves[move] = (
                (1 - epsilon) * node.unexpanded_moves[move] + epsilon * noise[idx]
            )
            idx += 1

    def search_batch(self, games: list[Bridgit]) -> list[MCTSNode]:
        """Run MCTS for multiple games with batched inference + virtual loss."""
        n = len(games)
        k = self.mcts_config.num_parallel_leaves
        roots = [MCTSNode(g.copy()) for g in games]

        # Initial expansion — batch all roots
        predictions = self._predict_batch([r.game for r in roots])
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
            to_predict_games: list[Bridgit] = []

            for i in range(n):
                for _ in range(chunk):
                    leaf = self._select_leaf(roots[i], c_puct)

                    if leaf.game.game_over:
                        self._backpropagate(leaf, 1.0)
                    elif leaf.is_expanded and not leaf.has_candidates:
                        self._backpropagate(
                            leaf, leaf.q_value if leaf.visit_count > 0 else 0.0,
                        )
                    else:
                        self._apply_virtual_loss(leaf)
                        to_predict_leaves.append(leaf)
                        to_predict_games.append(leaf.game)

            if to_predict_games:
                predictions = self._predict_batch(to_predict_games)

                for j, leaf in enumerate(to_predict_leaves):
                    policy, value = predictions[j]
                    self._undo_virtual_loss(leaf)
                    value = self._set_priors(leaf, policy, value)
                    self._backpropagate(leaf, value)

            sim += chunk

        return roots


def batched_self_play(
    net_wrapper: NetWrapper,
    board_config: BoardConfig,
    mcts_config: MCTSConfig,
    num_games: int,
    batch_size: int = 8,
    temperature: float = 1.0,
    verbose: bool = True,
) -> GameRecordCollection:
    """Play self-play games with batched MCTS inference.

    Runs `batch_size` games concurrently with virtual loss
    (mcts_config.num_parallel_leaves) for maximum GPU throughput.
    """
    batched_mcts = BatchedMCTS(net_wrapper, mcts_config)
    g = board_config.grid_size

    active_size = min(batch_size, num_games)
    games = [Bridgit(board_config) for _ in range(active_size)]
    move_histories: list[list[MoveRecord]] = [[] for _ in range(active_size)]

    completed: list[GameRecord] = []
    games_started = active_size

    pbar = None
    if verbose:
        pbar = tqdm(total=num_games, desc="Self-play")

    while len(completed) < num_games:
        active_idx = [i for i in range(len(games)) if not games[i].game_over]
        if not active_idx:
            break

        active_games = [games[i] for i in active_idx]
        roots = batched_mcts.search_batch(active_games)

        for j, i in enumerate(active_idx):
            root = roots[j]
            visit_counts = root.visit_counts()
            probs = MCTS.visit_counts_to_probs(visit_counts, temperature)

            if probs.sum() == 0:
                probs = games[i].to_mask()

            flat_idx = torch.multinomial(probs.flatten(), 1).item()
            row, col = divmod(flat_idx, g)
            canonical_move = Move(row=row, col=col)
            actual_move = canonical_move.decanonicalize(games[i].current_player)

            current_player = games[i].current_player
            if not games[i].make_move(actual_move):
                logger.error("Invalid move (%d,%d) in batched self-play", row, col)
                continue

            move_histories[i].append(MoveRecord(
                move=actual_move,
                player=current_player,
                moves_left_after=games[i].moves_left_in_turn,
                policy=probs,
            ))

        for i in range(len(games)):
            if games[i].game_over:
                record = GameRecord(
                    board_size=board_config.size,
                    moves=move_histories[i],
                    winner=games[i].winner,
                    horizontal_player="self-play",
                    vertical_player="self-play",
                )
                completed.append(record)

                if pbar is not None:
                    pbar.update(1)

                if games_started < num_games:
                    games[i] = Bridgit(board_config)
                    move_histories[i] = []
                    games_started += 1

    if pbar is not None:
        pbar.close()

    logger.info("Batched self-play: %d games completed", len(completed))
    return GameRecordCollection(game_records=completed)
