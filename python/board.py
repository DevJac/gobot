import itertools
import numpy as np
import pygoban


P = pygoban.Point


Empty = ' '
Black = 'b'
White = 'w'


class Board:
    def __init__(self, size):
        self._board = pygoban.Board(size)

    @property
    def size(self):
        return self._board.size

    def __getitem__(self, item):
        return self._board[item]

    def __setitem__(self, item, value):
        self._board[item] = value

    def liberties(self, point):
        return self._board.liberties(point)

    def valid_moves(self, pos):
        return self._board.valid_moves(pos)


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
