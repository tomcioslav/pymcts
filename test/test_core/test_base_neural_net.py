import copy

import torch
import torch.nn as nn

from bridgit.core.base_game import GameState
from bridgit.core.base_neural_net import BaseNeuralNet


class SimpleState(GameState):
    def __init__(self, data: list[float]):
        self.data = data


class SimpleNet(BaseNeuralNet):
    """Trivial net for testing: 4-feature input, 4-action output."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 4)
        self.value_fc = nn.Linear(4, 1)

    def encode(self, state: GameState) -> torch.Tensor:
        return torch.tensor(state.data, dtype=torch.float32)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        policy = torch.log_softmax(self.fc(x), dim=-1)
        value = torch.tanh(self.value_fc(x))
        return policy, value

    def save_checkpoint(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    def load_checkpoint(self, path: str) -> None:
        self.load_state_dict(torch.load(path, weights_only=True))

    def copy(self) -> "SimpleNet":
        return copy.deepcopy(self)


class TestBaseNeuralNet:
    def test_predict_returns_policy_and_value(self):
        net = SimpleNet()
        state = SimpleState([1.0, 0.0, 0.0, 0.0])
        policy, value = net.predict(state)
        assert policy.shape == (4,)
        assert isinstance(value, float)
        assert -1.0 <= value <= 1.0

    def test_predict_batch(self):
        net = SimpleNet()
        states = [SimpleState([1.0, 0, 0, 0]), SimpleState([0, 1.0, 0, 0])]
        policies, values = net.predict_batch(states)
        assert policies.shape == (2, 4)
        assert values.shape == (2,)

    def test_is_nn_module(self):
        net = SimpleNet()
        assert isinstance(net, nn.Module)
        assert len(list(net.parameters())) > 0

    def test_train_on_examples(self):
        net = SimpleNet()
        examples = [
            (SimpleState([1, 0, 0, 0]), torch.tensor([0.7, 0.1, 0.1, 0.1]), 1.0),
            (SimpleState([0, 1, 0, 0]), torch.tensor([0.1, 0.7, 0.1, 0.1]), -1.0),
        ]
        metrics = net.train_on_examples(examples, num_epochs=2, batch_size=2)
        assert "loss" in metrics
        assert "policy_loss" in metrics
        assert "value_loss" in metrics
