from gobot import Pos, Board, P


Empty = Pos.Empty
B = Pos.Black
W = Pos.White


def test_move_and_capture():
    b = Board()
    b[0, 0] = W
    b[0, 1] = B
    b[1, 1] = B
    assert b[0, 0] == W
    assert b[0, 1] == B
    assert b[1, 1] == B
    assert b[1, 0] == Empty
    b.move((1, 0), B)
    assert b[0, 0] == Empty
    assert b[0, 1] == B
    assert b[1, 1] == B
    assert b[1, 0] == B
    assert b.get_liberties((0, 0)) == 0
    assert b.get_liberties((1, 0)) == 5
    assert b.get_liberties((1, 1)) == 5
    print(b.printable_liberties)
    print(b)

def test_superko():
    b = Board(9, 9)
    b[1, 0] = B
    b[0, 1] = B
    b[2, 1] = B
    b[0, 2] = W
    b[1, 3] = W
    b[2, 2] = W
    for r in range(9):
        for c in range(9):
            b.update_liberties((r, c))
    print(b)
    b.move((1, 2), B)
    print(b)
    assert P(1, 1) in b.valid_moves(B)
    assert P(1, 1) in b.valid_moves(W)
    b.move((1, 1), W)
    print(b)
    assert b[1, 1] == W
    assert b[1, 2] == Empty
    assert P(1, 2) not in b.valid_moves(B)
    assert P(1, 2) in b.valid_moves(W)
