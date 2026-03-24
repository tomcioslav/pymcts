"""Bridgit-specific configuration."""

from pydantic import BaseModel


class BoardConfig(BaseModel):
    size: int = 5

    @property
    def grid_size(self) -> int:
        return 2 * self.size + 1


class NeuralNetConfig(BaseModel):
    num_channels: int = 64
    num_res_blocks: int = 4
