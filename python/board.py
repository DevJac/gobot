import itertools
import numpy as np
from pygoban import Point, Board


P = Point


Empty = ' '
Black = 'b'
White = 'w'


def encode_board(board, player):
    valid_moves = board.valid_moves(player)
    t = np.zeros((11, board.size, board.size))
    for r, c in itertools.product(range(board.size), repeat=2):
        p = Point(r, c)
        t[0, r, c] = int(board[p] == Black and board.get_liberties(p) == 1)
        t[1, r, c] = int(board[p] == Black and board.get_liberties(p) == 2)
        t[2, r, c] = int(board[p] == Black and board.get_liberties(p) == 3)
        t[3, r, c] = int(board[p] == Black and board.get_liberties(p) > 3)
        t[4, r, c] = int(board[p] == White and board.get_liberties(p) == 1)
        t[5, r, c] = int(board[p] == White and board.get_liberties(p) == 2)
        t[6, r, c] = int(board[p] == White and board.get_liberties(p) == 3)
        t[7, r, c] = int(board[p] == White and board.get_liberties(p) > 3)
        t[8, r, c] = int(player == Black)
        t[9, r, c] = int(player == White)
        t[10, r, c] = int(p in valid_moves)
    return t


if __name__ == '__main__':
    b = Board(9)
    b[P(0, 0)] = Black
    print(encode_board(b, Black))
