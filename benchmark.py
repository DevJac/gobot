from gobot import Board, Pos


def bench_valid_moves():
    board = Board()
    for board_string in open('benchmark_board.txt'):
        board_string = board_string.strip()
        board.load_board_state_string(board_string)
        board.valid_moves(Pos.Black)
        board.valid_moves(Pos.White)


if __name__ == '__main__':
    bench_valid_moves()
