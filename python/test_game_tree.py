from nnbot import *
import ipdb


def print_moves(moves):
    for move, node in moves[P(0, 1)].moves.items():
        print(move, node)
    print('\n\n')


with ipdb.launch_ipdb_on_exception():


    model = load_model('model.h5')


    t = GameTree(Board(9), Black)
    for _ in range(500):
        t.deepen(model)
        print_moves(t.root.moves)
    print(t.pick_move())



    print('pass')
    ipdb.set_trace()
