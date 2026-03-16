"""Reconstruct players from serialized specs (for multiprocess workers)."""

from bridgit.ai.neural_net import BridgitNet, NetWrapper
from bridgit.config import BoardConfig, MCTSConfig, NeuralNetConfig
from bridgit.players.players import BasePlayer, MCTSPlayer, RandomPlayer


def create_player(spec: dict) -> BasePlayer:
    """Create a player from a spec dict produced by BasePlayer.to_spec()."""
    player_type = spec["type"]

    if player_type == "random":
        return RandomPlayer(name=spec["name"])

    if player_type == "mcts":
        board_config = BoardConfig(**spec["board_config"])
        net_config = NeuralNetConfig(**spec["net_config"])
        mcts_config = MCTSConfig(**spec["mcts_config"])

        model = BridgitNet(board_config, net_config)
        model.load_state_dict(spec["model_state_dict"])
        net_wrapper = NetWrapper(model)

        return MCTSPlayer(
            net_wrapper,
            mcts_config,
            temperature=spec["temperature"],
            temp_threshold=spec["temp_threshold"],
            name=spec["name"],
        )

    raise ValueError(f"Unknown player type: {player_type}")
