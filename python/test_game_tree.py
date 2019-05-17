from nnbot import *
import ipdb
with ipdb.launch_ipdb_on_exception():


    model = load_model('model.h5')


    t = GameTree(Board(9), Black)
    print(t.root.moves, end='\n\n')
    t.deepen(model)
    print(t.root.moves, end='\n\n')
    t.deepen(model)
    print(t.root.moves, end='\n\n')
    t.deepen(model)
    print(t.root.moves, end='\n\n')
    t.deepen(model)
    print(t.root.moves, end='\n\n')
    print(t.pick_move())



    print('pass')
    ipdb.set_trace()
