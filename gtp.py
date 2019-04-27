import random
import re
import sys
import traceback
from gobot import Board, P, Pos


def letter_to_int(letter):
    return ord(letter) - 64 if letter < 'I' else ord(letter) - 65


def int_to_letter(int):
    return chr(int + 64) if int < (ord('I') - 64) else chr(int + 65)


def gtp_vertex_to_point(v):
    return P(int(v[1:]) - 1, letter_to_int(v[0]) - 1)


def point_to_gtp_vertex(p):
    r, c = p
    return int_to_letter(c + 1) + str(r + 1)


def color_to_pos(c):
    if c in {'B', 'black'}:
        return Pos.Black
    if c in {'W', 'white'}:
        return Pos.White
    raise ValueError(f'Unknown color: {repr(c)}')


def pos_to_color(p):
    return p.char.upper()


class Game:
    def __init__(self):
        self.board = None
        self.board_size = None
        self.komi = None

    @staticmethod
    def command_name(_args):
        return 'Gobot'

    @staticmethod
    def command_version(_args):
        return '1'

    @staticmethod
    def command_protocol_version(_args):
        return '2'

    @staticmethod
    def command_list_commands(_args):
        return '\n'.join([m.partition('_')[2] for m in dir(Game) if m.startswith('command_')])

    @staticmethod
    def command_known_command(command_name):
        return 'true' if command_name in globals() else 'false'

    @staticmethod
    def command_quit(_args):
        pass

    def command_komi(self, new_komi):
        self.komi = float(new_komi)

    def command_boardsize(self, board_size):
        board_size = int(board_size)
        self.board_size = board_size
        self.board = Board(board_size, board_size)

    def command_clear_board(self, _args):
        self.board = Board(self.board_size, self.board_size)

    def command_play(self, args):
        color, _, vertex = args.partition(' ')
        self.board.move(gtp_vertex_to_point(vertex), color_to_pos(color))
        log('Board state:\n' + self.board.printable_board + '\n' + self.board.printable_liberties)

    def command_genmove(self, color):
        valid_moves = self.board.valid_moves(color_to_pos(color))
        if len(valid_moves) == 0:
            return 'resign'
        selected_move = random.choice(list(valid_moves))
        self.board.move(selected_move, color_to_pos(color))
        log('Board state:\n' + self.board.printable_board + '\n' + self.board.printable_liberties)
        return point_to_gtp_vertex(selected_move)


logging = True
log_to = '/tmp/gtpdebug'


def log(s):
    if logging:
        with open(log_to, 'a') as f:
            f.write(s + '\n')


def send_response(s):
    sys.stdout.write(s)
    sys.stdout.flush()
    if logging:
        with open(log_to, 'a') as f:
            f.write(f'Sent response: {repr(s)}\n')


def main():
    log('='*85)
    log('='*40 + ' NEW ' + '='*40)
    log('='*85)
    command_re = re.compile(r'(?P<id>[0-9]+)? ?(?P<command_name>\w+) ?(?P<args>.*)')
    game = Game()
    for line in sys.stdin:
        command = re.match(command_re, line)
        log(f'Parsed {command.groups()} from {repr(line)}')
        try:
            response = getattr(game, 'command_' + command.group('command_name'))(command.group('args'))
            response = response or ''
            response = f"={command.group('id') or ''} {response}\n\n"
            send_response(response)
            if command.group('command_name') == 'quit':
                log('=' * 80 + ' QUIT')
                return
        except Exception:
            log(f'Failed on command: {repr(line)}')
            log(traceback.format_exc())
            raise


if __name__ == '__main__':
    main()
