import enum


class Position(enum.Enum):
    Empty = enum.auto()
    Black = enum.auto()
    White = enum.auto()


Pos = Position


class Board:
    def __init__(self, n_cols=19, n_rows=19):
        self.n_cols = 19
        self.n_rows = 19
        self.board = [Pos.Empty for _ in range(n_rows * n_cols)]

    def __getitem__(self, item):
        x, y = item
        if x >= self.n_cols or y >= self.n_rows:
            raise ValueError(f"Board is {self.n_cols} x {self.n_rows}: {item} is an invalid coordinate")
        pos_index = x + y * x
        return self.board[pos_index]

    def __setitem__(self, item, value):
        x, y = item
        if x >= self.n_cols or y >= self.n_rows:
            raise ValueError(f"Board is {self.n_cols} x {self.n_rows}: {item} is an invalid coordinate")
        if type(value) is not Position:
            raise ValueError(f"Expected a Position value, but got {value}")
        pos_index = x + y * x
        self.board[pos_index] = value
