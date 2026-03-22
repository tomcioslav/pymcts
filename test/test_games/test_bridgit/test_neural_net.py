import torch

from bridgit.games.bridgit.game import BridgitGame, BridgitGameState
from bridgit.games.bridgit.neural_net import BridgitNet
from bridgit.games.bridgit.config import BoardConfig, NeuralNetConfig
from bridgit.core.base_neural_net import BaseNeuralNet


class TestBridgitNet:
    def test_is_base_neural_net(self):
        net = BridgitNet()
        assert isinstance(net, BaseNeuralNet)

    def test_encode_shape(self):
        net = BridgitNet()
        game = BridgitGame()
        state = game.get_state()
        tensor = net.encode(state)
        g = BoardConfig().grid_size
        assert tensor.shape == (4, g, g)

    def test_predict_shapes(self):
        net = BridgitNet()
        game = BridgitGame()
        state = game.get_state()
        policy, value = net.predict(state)
        assert policy.shape == (game.action_space_size,)
        assert isinstance(value, float)

    def test_predict_batch(self):
        net = BridgitNet()
        game = BridgitGame()
        states = [game.get_state(), game.get_state()]
        policies, values = net.predict_batch(states)
        assert policies.shape == (2, game.action_space_size)
        assert values.shape == (2,)

    def test_checkpoint_roundtrip(self, tmp_path):
        net = BridgitNet()
        path = str(tmp_path / "test.pt")
        net.save_checkpoint(path)

        net2 = BridgitNet()
        net2.load_checkpoint(path)

        game = BridgitGame()
        state = game.get_state()
        p1, v1 = net.predict(state)
        p2, v2 = net2.predict(state)
        assert torch.allclose(p1, p2)

    def test_policy_is_1d_flattened(self):
        net = BridgitNet()
        game = BridgitGame()
        state = game.get_state()
        policy, _ = net.predict(state)
        g = BoardConfig().grid_size
        assert policy.shape == (g * g,)

    def test_copy(self):
        net = BridgitNet()
        net_copy = net.copy()
        game = BridgitGame()
        state = game.get_state()
        p1, v1 = net.predict(state)
        p2, v2 = net_copy.predict(state)
        assert torch.allclose(p1, p2)
