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
        if type(char) is str:
            char = char.encode('utf8')
        if char == b' ':
            return cls.Empty
        if char == b'b':
            return cls.Black
        if char == b'w':
            return cls.White
        raise ValueError(f'Unknown Position value: {repr(char)}')


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
        self.board_history = []

    @classmethod
    def from_state_string(cls, s, n_rows=19, n_cols=19):
        b = cls(n_rows, n_cols)
        for i, c in enumerate(s):
            b.board[i] = Pos.from_char(c)
        return b

    def copy(self):
        b = Board(self.n_rows, self.n_cols)
        b.board = self.board[:]
        b.liberties = self.liberties[:]
        b.board_history = self.board_history[:]
        return b

    def pos_index(self, point):
        r, c = point
        return c + r * self.n_cols

    @staticmethod
    def ensure_point(o):
        if type(o) is not Point:
            return Point(*o)
        return o

    @staticmethod
    def ensure_pos(o):
        if type(o) is not Pos:
            return Pos.from_char(o)
        return o

    def __getitem__(self, point):
        point = self.ensure_point(point)
        if not self.on_board(point):
            raise ValueError(f'Board is {self.n_rows} x {self.n_cols}: {repr(point)} is an invalid coordinate')
        return self.board[self.pos_index(point)]

    def __setitem__(self, point, pos):
        point = self.ensure_point(point)
        pos = self.ensure_pos(pos)
        if type(pos) is not Position:
            raise ValueError(f"Expected a Position value, but got {pos}")
        if not self.on_board(point):
            raise ValueError(f'Board is {self.n_rows} x {self.n_cols}: {repr(point)} is an invalid coordinate')
        self.board[self.pos_index(point)] = pos

    def move(self, point, pos):
        point = self.ensure_point(point)
        pos = self.ensure_pos(pos)
        if self[point] != Pos.Empty:
            raise ValueError(f'Invalid move, {repr(pos)} is not empty')
        self[point] = pos
        self.update_liberties({point} | point.neighbors)
        self.remove_stones_without_liberties(pos.other)
        self.remove_stones_without_liberties(pos)
        self.board_history.append(self.state)

    def remove_stones_without_liberties(self, remove_color):
        points_removed = set()
        for r in range(self.n_rows):
            for c in range(self.n_cols):
                p = P(r, c)
                if self[p] == remove_color and self.get_liberties(p) == 0:
                    self[p] = Pos.Empty
                    points_removed.add(p)
        points_removed_plus_neighbors = points_removed | {n for p in points_removed for n in p.neighbors}
        self.update_liberties(points_removed_plus_neighbors)

    def get_liberties(self, point):
        point = self.ensure_point(point)
        return self.liberties[self.pos_index(point)]

    def set_liberties(self, point, liberties):
        point = self.ensure_point(point)
        self.liberties[self.pos_index(point)] = liberties

    def on_board(self, point):
        r, c = point
        return 0 <= r < self.n_rows and 0 <= c < self.n_cols

    def update_liberties(self, points):
        updated_liberties = [-1 for _ in range(len(self.liberties))]
        for point in points:
            if not self.on_board(point):
                continue
            if updated_liberties[self.pos_index(point)] != -1:
                continue
            group = {point}
            group_liberties = set()

            def recurse(this_point):
                for neighboring_point in this_point.neighbors:
                    if not self.on_board(neighboring_point):
                        continue
                    if self[neighboring_point] == Pos.Empty:
                        group_liberties.add(neighboring_point)
                    elif self[this_point] == self[neighboring_point] and neighboring_point not in group:
                        group.add(neighboring_point)
                        recurse(neighboring_point)

            recurse(point)
            for group_point in group:
                updated_liberties[self.pos_index(group_point)] = len(group_liberties)
        for i in range(len(self.liberties)):
            if updated_liberties[i] > -1:
                self.liberties[i] = updated_liberties[i]

    def valid_moves(self, pos):
        valid_moves = set()
        for r in range(self.n_rows):
            for c in range(self.n_cols):
                p = P(r, c)
                if self.can_place_stone_at(p, pos) and self.not_ko(p, pos):
                    valid_moves.add(p)
        return valid_moves

    def can_place_stone_at(self, point, pos):
        point = self.ensure_point(point)
        if self[point] != Pos.Empty:
            return False
        for neighboring_point in point.neighbors:
            if not self.on_board(neighboring_point):
                continue
            if self[neighboring_point] == Pos.Empty:
                return True
            if self[neighboring_point] == pos and self.get_liberties(neighboring_point) > 1:
                return True
            if self[neighboring_point] == pos.other and self.get_liberties(neighboring_point) == 1:
                return True
        return False

    def not_ko(self, point, pos):
        point = self.ensure_point(point)
        if all(not self.on_board(neighboring_point) or self.get_liberties(neighboring_point) != 1
               for neighboring_point in point.neighbors):
            return True
        b = self.copy()
        b.move(point, pos)
        return b.state not in self.board_history

    @property
    def state(self):
        return b''.join(p.char for p in self.board)

    star_points = {
        (19, 19): {(3,  3), (3,  9), (3,  15),
                   (9,  3), (9,  9), (9,  15),
                   (15, 3), (15, 9), (15, 15)}
    }

    @property
    def printable_board(self):
        out = b''
        for r in reversed(range(self.n_rows)):
            out += b'|'
            for c in range(self.n_cols):
                if (r, c) in self.star_points.get((self.n_rows, self.n_cols), set()):
                    out += b'-' + self[r, c].char
                else:
                    out += b' ' + self[r, c].char
            out += b'|\n'
        return out.decode('utf8')

    @property
    def printable_liberties(self):
        out = ''
        for r in reversed(range(self.n_rows)):
            out += '|'
            for c in range(self.n_cols):
                if (r, c) in self.star_points.get((self.n_rows, self.n_cols), set()):
                    out += ' {:02}'.format(self.get_liberties((r, c)))
                else:
                    out += '{:3}'.format(self.get_liberties((r, c)))
            out += '|\n'
        return out

    def __str__(self):
        return self.printable_board
