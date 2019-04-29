import random
from itertools import count
from gobot import Board, Pos


board = Board()

for move in count(1):
    print('{:,}'.format(move))
    valid_moves = board.valid_moves(Pos.Black)
    if not valid_moves:
        print('White Wins!')
        break
    board.move(random.choice(list(valid_moves)), Pos.Black)

    valid_moves = board.valid_moves(Pos.White)
    if not valid_moves:
        print('Black Wins!')
        break
    board.move(random.choice(list(valid_moves)), Pos.White)
