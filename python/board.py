import itertools
import numpy as np
import pygoban


P = pygoban.Point


Empty = ' '
Black = 'b'
White = 'w'
VALID_POS = {Empty, Black, White}


def ensure_pos(c):
    if c not in VALID_POS:
        raise ValueError(f'Expected one of {VALID_POS}, but found: {repr(pos)}')
    return c


def ensure_point(p):
    if type(p) != pygoban.Point:
        return P(*p)
    return p


class Board:
    def __init__(self, size=19):
        self._board = pygoban.Board(size)

    @classmethod
    def from_state_string(cls, s, size):
        if len(s) != size**2:
            raise ValueError(
                f'Expected board of size {size} x {size} (= {size**2}), '
                f'but got string of length {len(s)}')
        b = cls(size)
        for i, c in enumerate(s):
            b[P(*b.from_index(i))] = ensure_pos(c)
        return b

    @property
    def size(self):
        return self._board.size

    def to_index(self, x, y):
        return x * self.size + y

    def from_index(self, i):
        return divmod(i, self.size)

    def __getitem__(self, item):
        return self._board[ensure_point(item)]

    def __setitem__(self, item, value):
        self._board[ensure_point(item)] = ensure_pos(value)

    def liberties(self, point):
        return self._board.liberties(ensure_point(point))

    def valid_moves(self, pos):
        return self._board.valid_moves(ensure_pos(pos))

    def play(self, point, pos):
        self._board.play(ensure_point(point), ensure_pos(pos))

    star_points = {
        19: {(3,  3), (3,  9), (3,  15),
             (9,  3), (9,  9), (9,  15),
             (15, 3), (15, 9), (15, 15)}
    }

    def printable_board(self, pretty=False):
        def prettify(c):
            if not pretty:
                return c
            if c == White:
                return '●'
            if c == Black:
                return '○'
            return c

        out = ''
        for r in reversed(range(self.size)):
            out += '|'
            for c in range(self.size):
                if (r, c) in self.star_points.get(self.size, set()):
                    out += '-' + prettify(ensure_pos(self[r, c]))
                else:
                    out += ' ' + prettify(ensure_pos(self[r, c]))
            out += ' |\n'
        return out

    def __str__(self):
        return self.printable_board(True)


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
