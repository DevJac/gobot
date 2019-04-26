from collections import namedtuple
import enum


class Position(enum.Enum):
    Empty = enum.auto()
    Black = enum.auto()
    White = enum.auto()

    @property
    def char(self):
        if self == self.Empty:
            return b' '
        if self == self.Black:
            return b'b'
        if self == self.White:
            return b'w'

    @classmethod
    def from_char(cls, char):
        if char == b' ':
            return cls.Empty
        if char == b'b':
            return cls.Black
        if char == b'w':
            return cls.White


Pos = Position


class Point(namedtuple('Point', 'row col')):
    @property
    def neighbors(self):
        return [
            Point(self.row - 1, self.col),
            Point(self.row + 1, self.col),
            Point(self.row, self.col - 1),
            Point(self.row, self.col + 1),
        ]


P = Point


class Board:
    def __init__(self, n_rows=19, n_cols=19):
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.board = [Pos.Empty for _ in range(n_rows * n_cols)]

    def __getitem__(self, item):
        r, c = item
        if r >= self.n_rows or c >= self.n_cols:
            raise ValueError(f'Board is {self.n_rows} x {self.n_cols}: {item} is an invalid coordinate')
        pos_index = c + r * self.n_cols
        return self.board[pos_index]

    def __setitem__(self, item, value):
        if type(value) is not Position:
            raise ValueError(f"Expected a Position value, but got {value}")
        r, c = item
        if r >= self.n_rows or c >= self.n_cols:
            raise ValueError(f'Board is {self.n_rows} x {self.n_cols}: {item} is an invalid coordinate')
        pos_index = c + r * self.n_cols
        self.board[pos_index] = value

    @property
    def valid_moves(self):
        pass

    @property
    def state(self):
        return b''.join(p.char for p in self.board)
