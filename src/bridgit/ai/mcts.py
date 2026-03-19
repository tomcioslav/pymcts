"""Monte Carlo Tree Search with neural network guidance."""

import logging
import math
import uuid

import numpy as np
import torch
from tqdm.auto import tqdm

from bridgit.ai.neural_net import NetWrapper
from bridgit.game import Bridgit
from bridgit.schema import Move
from bridgit.config import MCTSConfig

logger = logging.getLogger("bridgit.mcts")


class MCTSNode:
    """A node in the MCTS tree."""

    __slots__ = ["game", "parent", "action", "children", "visit_count",
                 "value_sum", "prior", "is_expanded",
                 "child_index", "path", "id"]

    def __init__(self, game: Bridgit, parent: "MCTSNode | None" = None,
                 action: Move | None = None, prior: float = 0.0,
                 child_index: int = -1):
        self.game = game
        self.parent = parent
        self.action = action
        self.children: dict[Move, MCTSNode] = {}
        self.visit_count = 0
        self.value_sum = 0.0
        self.prior = prior
        self.is_expanded = False
        self.child_index = child_index
        self.path: tuple[int, ...] = parent.path + (child_index,) if parent else ()
        self.id: str = uuid.uuid4().hex[:8]

    @property
    def q_value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def ucb_score(self, c_puct: float) -> float:
        """PUCT selection score from the parent's perspective.

        Q-value is stored from this node's current_player perspective,
        so we flip it when the parent has a different current_player.
        """
        if self.parent is None:
            return 0.0

        same_player = self.game.current_player == self.parent.game.current_player
        q = self.q_value if same_player else -self.q_value
        parent_visits = self.parent.visit_count
        exploration = c_puct * self.prior * math.sqrt(parent_visits) / (1 + self.visit_count)
        return q + exploration

    def best_child(self, c_puct: float) -> "MCTSNode":
        """Select child with highest UCB score."""
        return max(self.children.values(), key=lambda c: c.ucb_score(c_puct))

    def best_move(self) -> Move:
        """Return the most-visited child's move."""
        return max(self.children, key=lambda m: self.children[m].visit_count)

    def get_node(self, path: tuple[int, ...]) -> "MCTSNode":
        """Retrieve a descendant node by its path of child indices."""
        node = self
        children_list_cache: list[MCTSNode] | None = None
        for idx in path:
            children_list_cache = list(node.children.values())
            node = children_list_cache[idx]
        return node

    def visit_counts(self) -> torch.Tensor:
        """Return visit counts as a (g, g) array."""
        g = self.game.board_config.grid_size
        visits = torch.zeros((g, g), dtype=torch.float32)
        for move, child in self.children.items():
            visits[move.row, move.col] = child.visit_count
        return visits


class MCTS:
    """Monte Carlo Tree Search guided by a neural network."""

    def __init__(self, net_wrapper: NetWrapper, mcts_config: MCTSConfig):
        self.net_wrapper = net_wrapper
        self.mcts_config = mcts_config

    def _predict(self, game: Bridgit) -> tuple[torch.Tensor, float]:
        """Run neural net on game state.

        Returns:
            policy: torch.Tensor of shape (g, g) — probabilities (cpu)
            value: float — position evaluation for current player
        """
        model = self.net_wrapper.model
        device = self.net_wrapper.device

        model.eval()
        tensor = game.to_tensor().unsqueeze(0).to(device)

        with torch.no_grad():
            log_policy, value = model(tensor)

        # log_policy: (1, g, g) → (g, g)
        policy = torch.exp(log_policy[0]).cpu()
        val = value[0].item()

        return policy, val

    def _search(self, game: Bridgit, verbose: bool = False) -> MCTSNode:
        """Run MCTS simulations and return the root node."""
        root = MCTSNode(game.copy())
        self._expand(root)
        self._add_dirichlet_noise(root)
        self.continue_search(root, self.mcts_config.num_simulations, verbose=verbose)
        return root

    def continue_search(self, root: MCTSNode, num_sims: int, verbose: bool = False):
        """Continue running MCTS simulations on an existing root node."""
        iterator = range(num_sims)
        if verbose:
            iterator = tqdm(iterator, desc="MCTS", leave=False)

        for i in iterator:
            node = root

            # SELECT
            while node.is_expanded and node.children:
                node = node.best_child(self.mcts_config.c_puct)
                logger.debug("sim %d: SELECT node %s (action=%s, visits=%d, q=%.3f)",
                             i, node.id, node.action, node.visit_count, node.q_value)

            # TERMINAL
            if node.game.game_over:
                logger.debug("sim %d: node %s is terminal, backprop value=1.0",
                             i, node.id)
                self._backpropagate(node, 1.0)
            else:
                # EXPAND & EVALUATE
                value = self._expand(node)
                logger.debug("sim %d: EXPAND node %s -> %d children, nn_value=%.3f",
                             i, node.id, len(node.children), value)
                # BACKPROPAGATE
                self._backpropagate(node, value)

    def _expand(self, node: MCTSNode) -> float:
        """Expand node using neural network. Returns value estimate."""
        if node.game.game_over:
            node.is_expanded = True
            return 1.0

        policy, value = self._predict(node.game)
        valid_mask = node.game.to_mask()  # (g, g) canonical space

        # Mask and renormalize policy
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

        # Create children — coordinates are canonical, decanonicalize for actual moves
        player = node.game.current_player
        for idx, (r, c) in enumerate(torch.nonzero(valid_mask, as_tuple=False)):
            r, c = r.item(), c.item()
            canonical_move = Move(row=r, col=c)
            actual_move = canonical_move.decanonicalize(player)
            child_game = node.game.copy()
            child_game.make_move(actual_move)
            child = MCTSNode(child_game, parent=node, action=canonical_move,
                             prior=policy[r, c].item(), child_index=idx)
            node.children[canonical_move] = child

        node.is_expanded = True
        return value

    def _backpropagate(self, node: MCTSNode, value: float):
        """Backpropagate value, flipping sign only when the player changes.

        With the 1-2-2 turn structure, consecutive nodes may belong to the
        same player. We only negate the value when crossing a player boundary.
        """
        while node is not None:
            node.visit_count += 1
            node.value_sum += value
            logger.debug("  BACKPROP node %s: value=%.3f, visits=%d, "
                         "q=%.3f", node.id, value, node.visit_count,
                         node.q_value)
            if (node.parent is not None
                    and node.game.current_player != node.parent.game.current_player):
                value = -value
            node = node.parent

    def _add_dirichlet_noise(self, node: MCTSNode):
        """Add Dirichlet noise to root priors for exploration."""
        if not node.children:
            return
        moves = list(node.children.keys())
        noise = np.random.dirichlet([self.mcts_config.dirichlet_alpha] * len(moves))
        eps = self.mcts_config.dirichlet_epsilon
        for i, move in enumerate(moves):
            node.children[move].prior = (
                (1 - eps) * node.children[move].prior + eps * noise[i]
            )

    @staticmethod
    def visit_counts_to_probs(
        visit_counts: torch.Tensor, temperature: float = 1.0
    ) -> torch.Tensor:
        """Convert visit counts to a probability distribution.

        Args:
            visit_counts: tensor of visit counts
            temperature: 1.0 = proportional to visits, 0.0 = greedy
        """
        if temperature == 0:
            best = torch.argmax(visit_counts)
            best_idx = np.unravel_index(best.item(), visit_counts.shape)
            probs = torch.zeros_like(visit_counts)
            probs[best_idx] = torch.tensor(1.0)
            return probs

        counts = visit_counts ** (1.0 / temperature)
        total = counts.sum()
        if total == 0:
            return visit_counts
        return counts / total.item()

    def get_action_probs(
        self, game: Bridgit, temperature: float = 1.0, verbose: bool = False,
    ) -> torch.Tensor:
        """Run MCTS and return action probabilities."""
        root = self._search(game, verbose=verbose)
        logger.info("MCTS search done: root %s, %d visits, best_move=%s, "
                     "q=%.3f", root.id, root.visit_count, root.best_move(),
                     root.best_child(self.mcts_config.c_puct).q_value)
        return self.visit_counts_to_probs(root.visit_counts(), temperature)
