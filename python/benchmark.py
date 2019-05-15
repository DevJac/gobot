from gobot import Board, Pos


def bench_valid_moves():
    for board_string in open('benchmark_boards.txt'):
        board_string = board_string.strip()
        board = Board.from_state_string(board_string)
        board.valid_moves(Pos.Black)
        board.valid_moves(Pos.White)


if __name__ == '__main__':
    bench_valid_moves()
