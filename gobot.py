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

    @property
    def other(self):
        if self == self.Black:
            return self.White
        if self == self.White:
            return self.Black
        if self == self.Empty:
            return self.Empty

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
        return {
            Point(self.row - 1, self.col),
            Point(self.row + 1, self.col),
            Point(self.row, self.col - 1),
            Point(self.row, self.col + 1),
        }


P = Point


class Board:
    def __init__(self, n_rows=19, n_cols=19):
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.board = [Pos.Empty for _ in range(n_rows * n_cols)]
        self.liberties = [0 for _ in range(n_rows * n_cols)]

    def pos_index(self, point):
        r, c = point
        return c + r * self.n_cols

    def __getitem__(self, point):
        if not self.on_board(point):
            raise ValueError(f'Board is {self.n_rows} x {self.n_cols}: {point} is an invalid coordinate')
        return self.board[self.pos_index(point)]

    def __setitem__(self, point, pos):
        if type(pos) is not Position:
            raise ValueError(f"Expected a Position value, but got {pos}")
        if not self.on_board(point):
            raise ValueError(f'Board is {self.n_rows} x {self.n_cols}: {point} is an invalid coordinate')
        self.board[self.pos_index(point)] = pos

    def move(self, point, pos):
        if self[point] != Pos.Empty:
            raise ValueError(f'Invalid move, {pos} is not empty')
        self[point] = pos
        self.update_liberties(point)
        self.remove_stones_without_liberties_of_color(pos.other)
        self.remove_stones_without_liberties_of_color(pos)

    def remove_stones_without_liberties_of_color(self, pos):
        points_removed = set()
        for r in range(self.n_rows):
            for c in range(self.n_cols):
                p = P(r, c)
                if self[p] == pos and self.get_liberties(p) == 0:
                    self[p] = Pos.Empty
                    points_removed.add(p)
        for point_removed in points_removed:
            self.update_liberties(point_removed)

    def get_liberties(self, point):
        return self.liberties[self.pos_index(point)]

    def set_liberties(self, point, liberties):
        self.liberties[self.pos_index(point)] = liberties

    def on_board(self, point):
        r, c = point
        return 0 <= r < self.n_rows and 0 <= c < self.n_cols

    def update_liberties(self, point):
        if self[point] == Pos.Empty:
            self.set_liberties(point, 0)
            return
        group = {point}
        liberties = set()

        def recurse(this_point):
            for neighboring_point in this_point.neighbors:
                if not self.on_board(neighboring_point):
                    continue
                if self[neighboring_point] == Pos.Empty:
                    liberties.add(neighboring_point)
                elif self[this_point] == self[neighboring_point] and neighboring_point not in group:
                    group.add(neighboring_point)
                    recurse(neighboring_point)

        recurse(point)
        for group_point in group:
            self.set_liberties(group_point, len(liberties))

    @property
    def valid_moves(self):
        pass

    @property
    def state(self):
        return b''.join(p.char for p in self.board)

    @property
    def printable_board(self):
        out = b''
        for r in range(self.n_rows):
            out += b'|'
            for c in range(self.n_cols):
               out += self[P(r, c)].char
            out += b'|\n'
        return out.decode('utf8')

    @property
    def printable_liberties(self):
        out = ''
        for r in range(self.n_rows):
            out += '|'
            for c in range(self.n_cols):
                out += '{:2}'.format(self.get_liberties(P(r, c)))
            out += '|\n'
        return out

    def __str__(self):
        return self.printable_board
