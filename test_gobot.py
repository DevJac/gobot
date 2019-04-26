from gobot import Pos, Board


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
