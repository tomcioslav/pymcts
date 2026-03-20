"""Monte Carlo Tree Search with neural network guidance."""

import logging
import math

import numpy as np
import torch
from tqdm.auto import tqdm

from bridgit.ai.neural_net import NetWrapper
from bridgit.game import Bridgit
from bridgit.schema import Move
from bridgit.config import MCTSConfig

logger = logging.getLogger("bridgit.mcts")

_node_counter = 0


class MCTSNode:
    """A node in the MCTS tree.

    Uses lazy expansion: after neural net evaluation, only priors are stored
    (in `unexpanded_moves`). Child nodes are created on demand when selected.
    """

    __slots__ = ["game", "parent", "action", "children", "visit_count",
                 "value_sum", "prior", "is_expanded", "unexpanded_moves",
                 "child_index", "path", "id"]

    def __init__(self, game: Bridgit, parent: "MCTSNode | None" = None,
                 action: Move | None = None, prior: float = 0.0,
                 child_index: int = -1):
        global _node_counter
        self.game = game
        self.parent = parent
        self.action = action
        self.children: dict[Move, MCTSNode] = {}
        self.visit_count = 0
        self.value_sum = 0.0
        self.prior = prior
        self.is_expanded = False
        # Moves waiting to be turned into child nodes: {Move: prior}
        self.unexpanded_moves: dict[Move, float] = {}
        self.child_index = child_index
        self.path: tuple[int, ...] = parent.path + (child_index,) if parent else ()
        self.id = _node_counter
        _node_counter += 1

    @property
    def q_value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    @property
    def fully_expanded(self) -> bool:
        """True when all moves have been turned into child nodes."""
        return self.is_expanded and not self.unexpanded_moves

    def _unexpanded_ucb(self, move: Move, prior: float, c_puct: float) -> float:
        """UCB score for an unexpanded move (visit_count=0, q=0)."""
        return c_puct * prior * math.sqrt(self.visit_count)

    def best_child_or_expand(self, c_puct: float) -> "MCTSNode":
        """Select the best child, possibly creating one from unexpanded moves.

        Compares UCB scores of existing children with potential scores of
        unexpanded moves. If an unexpanded move wins, creates the child node.
        """
        best_score = float("-inf")
        best_child = None
        best_unexpanded: Move | None = None
        best_unexpanded_prior = 0.0

        # Score existing children
        for child in self.children.values():
            score = child.ucb_score(c_puct)
            if score > best_score:
                best_score = score
                best_child = child
                best_unexpanded = None

        # Score unexpanded moves
        for move, prior in self.unexpanded_moves.items():
            score = self._unexpanded_ucb(move, prior, c_puct)
            if score > best_score:
                best_score = score
                best_child = None
                best_unexpanded = move
                best_unexpanded_prior = prior

        if best_unexpanded is not None:
            # Create the child node on demand
            actual_move = best_unexpanded.decanonicalize(self.game.current_player)
            child_game = self.game.copy()
            child_game.make_move(actual_move)
            idx = len(self.children)
            child = MCTSNode(child_game, parent=self, action=best_unexpanded,
                             prior=best_unexpanded_prior, child_index=idx)
            self.children[best_unexpanded] = child
            del self.unexpanded_moves[best_unexpanded]
            return child

        return best_child

    def ucb_score(self, c_puct: float) -> float:
        """PUCT selection score from the parent's perspective."""
        if self.parent is None:
            return 0.0

        same_player = self.game.current_player == self.parent.game.current_player
        q = self.q_value if same_player else -self.q_value
        parent_visits = self.parent.visit_count
        exploration = c_puct * self.prior * math.sqrt(parent_visits) / (1 + self.visit_count)
        return q + exploration

    def best_child(self, c_puct: float) -> "MCTSNode":
        """Select child with highest UCB score (only among existing children)."""
        return max(self.children.values(), key=lambda c: c.ucb_score(c_puct))

    def best_move(self) -> Move:
        """Return the most-visited child's move."""
        return max(self.children, key=lambda m: self.children[m].visit_count)

    def get_node(self, path: tuple[int, ...]) -> "MCTSNode":
        """Retrieve a descendant node by its path of child indices."""
        node = self
        for idx in path:
            children_list = list(node.children.values())
            node = children_list[idx]
        return node

    def visit_counts(self) -> torch.Tensor:
        """Return visit counts as a (g, g) array."""
        g = self.game.board_config.grid_size
        visits = torch.zeros((g, g), dtype=torch.float32)
        for move, child in self.children.items():
            visits[move.row, move.col] = child.visit_count
        return visits

    @property
    def has_candidates(self) -> bool:
        """True if there are children or unexpanded moves to select from."""
        return bool(self.children) or bool(self.unexpanded_moves)


class MCTS:
    """Monte Carlo Tree Search guided by a neural network."""

    def __init__(self, net_wrapper: NetWrapper, mcts_config: MCTSConfig):
        self.net_wrapper = net_wrapper
        self.mcts_config = mcts_config

    def _predict(self, game: Bridgit) -> tuple[torch.Tensor, float]:
        """Run neural net on game state."""
        model = self.net_wrapper.model
        device = self.net_wrapper.device

        model.eval()
        tensor = game.to_tensor().unsqueeze(0).to(device)

        with torch.no_grad():
            log_policy, value = model(tensor)

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

        c_puct = self.mcts_config.c_puct
        for i in iterator:
            node = root

            # SELECT — use best_child_or_expand for lazy creation
            while node.is_expanded and node.has_candidates:
                node = node.best_child_or_expand(c_puct)
                if not node.is_expanded:
                    break  # new unexpanded leaf

            # TERMINAL
            if node.game.game_over:
                self._backpropagate(node, 1.0)
            elif not node.is_expanded:
                # EXPAND & EVALUATE
                value = self._expand(node)
                self._backpropagate(node, value)
            else:
                # Fully expanded leaf with no candidates (shouldn't happen normally)
                self._backpropagate(node, node.q_value if node.visit_count > 0 else 0.0)

    def _expand(self, node: MCTSNode) -> float:
        """Expand node: run neural net and store priors lazily."""
        if node.game.game_over:
            node.is_expanded = True
            return 1.0

        policy, value = self._predict(node.game)
        self._set_priors(node, policy)
        return value

    @staticmethod
    def _set_priors(node: MCTSNode, policy: torch.Tensor):
        """Store masked/renormalized priors in node.unexpanded_moves."""
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
                return

        for r, c in torch.nonzero(valid_mask, as_tuple=False):
            r, c = r.item(), c.item()
            move = Move(row=r, col=c)
            node.unexpanded_moves[move] = policy[r, c].item()

        node.is_expanded = True

    def _backpropagate(self, node: MCTSNode, value: float):
        """Backpropagate value, flipping sign on player boundaries."""
        while node is not None:
            node.visit_count += 1
            node.value_sum += value
            if (node.parent is not None
                    and node.game.current_player != node.parent.game.current_player):
                value = -value
            node = node.parent

    def _add_dirichlet_noise(self, node: MCTSNode):
        """Add Dirichlet noise to root priors for exploration."""
        all_moves = list(node.children.keys()) + list(node.unexpanded_moves.keys())
        if not all_moves:
            return
        noise = np.random.dirichlet([self.mcts_config.dirichlet_alpha] * len(all_moves))
        eps = self.mcts_config.dirichlet_epsilon
        idx = 0
        for move in list(node.children.keys()):
            node.children[move].prior = (
                (1 - eps) * node.children[move].prior + eps * noise[idx]
            )
            idx += 1
        for move in list(node.unexpanded_moves.keys()):
            node.unexpanded_moves[move] = (
                (1 - eps) * node.unexpanded_moves[move] + eps * noise[idx]
            )
            idx += 1

    @staticmethod
    def visit_counts_to_probs(
        visit_counts: torch.Tensor, temperature: float = 1.0
    ) -> torch.Tensor:
        """Convert visit counts to a probability distribution."""
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
        logger.info("MCTS search done: root %d, %d visits, best_move=%s, "
                     "q=%.3f", root.id, root.visit_count, root.best_move(),
                     root.best_child(self.mcts_config.c_puct).q_value)
        return self.visit_counts_to_probs(root.visit_counts(), temperature)
