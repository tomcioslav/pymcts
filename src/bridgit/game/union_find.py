"""Union-Find for incremental win detection in Bridgit."""


class UnionFind:
    """Weighted Union-Find with path compression.

    Uses two sentinel nodes per player to represent boundaries:
    - HORIZONTAL: LEFT_SENTINEL, RIGHT_SENTINEL
    - VERTICAL: TOP_SENTINEL, BOTTOM_SENTINEL

    Win = the two sentinels are in the same component.
    """

    __slots__ = ["parent", "rank"]

    def __init__(self, size: int):
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, x: int) -> int:
        """Find root with path compression."""
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]  # path halving
            x = self.parent[x]
        return x

    def union(self, x: int, y: int):
        """Union by rank."""
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1

    def connected(self, x: int, y: int) -> bool:
        return self.find(x) == self.find(y)

    def copy(self) -> "UnionFind":
        uf = UnionFind.__new__(UnionFind)
        uf.parent = self.parent.copy()
        uf.rank = self.rank.copy()
        return uf
