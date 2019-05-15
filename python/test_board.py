from board import Board, P, Empty, Black, White


def test_move_and_capture():
    b = Board()
    b[0, 0] = White
    b[0, 1] = Black
    b[1, 1] = Black
    assert b[0, 0] == White
    assert b[0, 1] == Black
    assert b[1, 1] == Black
    assert b[1, 0] == Empty
    b.play((1, 0), Black)
    assert b[0, 0] == Empty
    assert b[0, 1] == Black
    assert b[1, 1] == Black
    assert b[1, 0] == Black
    assert b.liberties((0, 0)) == 0
    assert b.liberties((1, 0)) == 5
    assert b.liberties((1, 1)) == 5


def test_superko():
    b = Board(9)
    b[1, 0] = Black
    b[0, 1] = Black
    b[2, 1] = Black
    b[0, 2] = White
    b[1, 3] = White
    b[2, 2] = White
    print(b)
    b.play((1, 2), Black)
    print(b)
    assert P(1, 1) in b.valid_moves(Black)
    print(b.valid_moves(White))
    assert P(1, 1) in b.valid_moves(White)
    b.play((1, 1), White)
    print(b)
    assert b[1, 1] == White
    assert b[1, 2] == Empty
    assert P(1, 2) not in b.valid_moves(Black)
    assert P(1, 2) in b.valid_moves(White)


def test_cant_play_on_own_stones():
    b = Board(9)
    b.play((0, 0), Black)
    assert P(0, 0) not in b.valid_moves(Black)
    assert P(0, 0) not in b.valid_moves(White)
