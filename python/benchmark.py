from board import Board, Black, White


def bench_valid_moves():
    for i, board_string in enumerate(open('benchmark_boards.txt')):
        board_string = board_string[:-1]
        board = Board.from_state_string(board_string)
        board.valid_moves(Black)
        board.valid_moves(White)


if __name__ == '__main__':
    bench_valid_moves()
