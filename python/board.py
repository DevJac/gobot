import itertools
from enum import Enum, auto
import numpy as np
import pygoban


P = pygoban.Point


class BoardPosition(Enum):
    EMPTY = auto()
    BLACK = auto()
    WHITE = auto()

    @staticmethod
    def from_char(c):
        if c == ' ':
            return Empty
        if c == 'b':
            return Black
        if c == 'w':
            return White
        raise ValueError(f'Unknown BoardPosition character: {repr(c)}')

    def to_char(self):
        if self == Empty:
            return ' '
        if self == Black:
            return 'b'
        if self == White:
            return 'w'
        raise ValueError(f'Unknown BoardPosition self value: {repr(self)}')


Empty = BoardPosition.EMPTY
Black = BoardPosition.BLACK
White = BoardPosition.WHITE


def ensure_point(p):
    if type(p) is not pygoban.Point:
        return P(*p)
    return p


class Board:
    def __init__(self, size=19):
        self._board = pygoban.Board(size)

    @classmethod
    def from_state_string(cls, s, size=19):
        if len(s) != size**2:
            raise ValueError(
                f'Expected board of size {size} x {size} (= {size**2}), '
                f'but got string of length {len(s)}')
        b = cls(size)
        for i, c in enumerate(s):
            b[P(*b.from_index(i))] = BoardPosition.from_char(c)
        return b

    @property
    def size(self):
        return self._board.size

    def to_index(self, x, y):
        return x * self.size + y

    def from_index(self, i):
        return divmod(i, self.size)

    def __getitem__(self, item):
        return BoardPosition.from_char(self._board[ensure_point(item)])

    def __setitem__(self, item, value):
        self._board[ensure_point(item)] = value.to_char()

    def liberties(self, point):
        return self._board.liberties(ensure_point(point))

    def valid_moves(self, pos):
        return self._board.valid_moves(pos.to_char())

    def play(self, point, pos):
        self._board.play(ensure_point(point), pos.to_char())

    star_points = {
        19: {(3,  3), (3,  9), (3,  15),
             (9,  3), (9,  9), (9,  15),
             (15, 3), (15, 9), (15, 15)}
    }

    def printable_board(self):
        def prettify(pos):
            if pos == Empty:
                return ' '
            if pos == White:
                return '●'
            if pos == Black:
                return '○'
            raise ValueError(f'Unknown position: {repr(pos)}')
        out = ''
        for r in reversed(range(self.size)):
            out += '|'
            for c in range(self.size):
                if (r, c) in self.star_points.get(self.size, set()):
                    out += '-' + prettify(self[r, c])
                else:
                    out += ' ' + prettify(self[r, c])
            out += ' |\n'
        return out

    def __str__(self):
        return self.printable_board()


def encode_board(board, player):
    valid_moves = set(board.valid_moves(player))
    t = np.zeros((11, board.size, board.size))
    for r, c in itertools.product(range(board.size), repeat=2):
        p = P(r, c)
        t[0, r, c] = int(board[p] == Black and board.liberties(p) == 1)
        t[1, r, c] = int(board[p] == Black and board.liberties(p) == 2)
        t[2, r, c] = int(board[p] == Black and board.liberties(p) == 3)
        t[3, r, c] = int(board[p] == Black and board.liberties(p) > 3)
        t[4, r, c] = int(board[p] == White and board.liberties(p) == 1)
        t[5, r, c] = int(board[p] == White and board.liberties(p) == 2)
        t[6, r, c] = int(board[p] == White and board.liberties(p) == 3)
        t[7, r, c] = int(board[p] == White and board.liberties(p) > 3)
        t[8, r, c] = int(player == Black)
        t[9, r, c] = int(player == White)
        t[10, r, c] = int(p in valid_moves)
    return t


if __name__ == '__main__':
    b = Board(9)
    b[P(0, 0)] = Black
    print(encode_board(b, Black))
