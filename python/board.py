from enum import Enum, auto
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

    def copy(self):
        b = Board.__new__(Board)
        b._board = self._board.copy()
        return b

    @classmethod
    def from_string(cls, s, size=19):
        if len(s) != size**2:
            raise ValueError(
                f'Expected board of size {size} x {size} (= {size**2}), '
                f'but got string of length {len(s)}')
        b = cls(size)
        for i, c in enumerate(s):
            b[divmod(i, size)] = BoardPosition.from_char(c)
        return b

    @property
    def size(self):
        return self._board.size

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
