import enum
import itertools
import functools
import random
import string
import pickle
import numpy as np
from collections import namedtuple
from keras.models import Model
from keras.layers import Conv2D, Dense, Flatten, Input


class Position(enum.Enum):
    Empty = 0
    Black = 1
    White = 2

    @property
    def char(self):
        if self == self.Empty:
            return b' '
        if self == self.Black:
            return b'b'
        if self == self.White:
            return b'w'

    @property
    def other(self):
        if self == self.Black:
            return self.White
        if self == self.White:
            return self.Black
        if self == self.Empty:
            return self.Empty

    @classmethod
    def from_char(cls, char):
        if type(char) is str:
            char = char.encode('utf8')
        if char == b' ':
            return cls.Empty
        if char == b'b':
            return cls.Black
        if char == b'w':
            return cls.White
        raise ValueError(f'Unknown Position value: {repr(char)}')


Pos = Position


@functools.lru_cache(maxsize=1000)
def neighbors(point):
    return frozenset({
        Point(point.row - 1, point.col),
        Point(point.row + 1, point.col),
        Point(point.row, point.col - 1),
        Point(point.row, point.col + 1),
    })


class Point(namedtuple('Point', 'row col')):
    @property
    def neighbors(self):
        return neighbors(self)


P = Point


class Board:
    def __init__(self, n_rows=19, n_cols=19):
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.board = [Pos.Empty for _ in range(n_rows * n_cols)]
        self.liberties = [0 for _ in range(n_rows * n_cols)]
        self.board_history = []

    @classmethod
    def from_state_string(cls, s, n_rows=19, n_cols=19):
        b = cls(n_rows, n_cols)
        for i, c in enumerate(s):
            b.board[i] = Pos.from_char(c)
        return b

    def frozen_copy(self):
        b = Board(self.n_rows, self.n_cols)
        b.board = tuple(self.board)
        b.liberties = tuple(self.liberties)
        b.board_history = tuple()
        return b

    def copy(self):
        b = Board(self.n_rows, self.n_cols)
        b.board = self.board[:]
        b.liberties = self.liberties[:]
        b.board_history = self.board_history[:]
        return b

    def pos_index(self, point):
        r, c = point
        return c + r * self.n_cols

    @staticmethod
    def ensure_point(o):
        if type(o) is not Point:
            return Point(*o)
        return o

    @staticmethod
    def ensure_pos(o):
        if type(o) is not Pos:
            return Pos.from_char(o)
        return o

    def __getitem__(self, point):
        point = self.ensure_point(point)
        if not self.on_board(point):
            raise ValueError(f'Board is {self.n_rows} x {self.n_cols}: {repr(point)} is an invalid coordinate')
        return self.board[self.pos_index(point)]

    def __setitem__(self, point, pos):
        point = self.ensure_point(point)
        pos = self.ensure_pos(pos)
        if type(pos) is not Position:
            raise ValueError(f"Expected a Position value, but got {pos}")
        if not self.on_board(point):
            raise ValueError(f'Board is {self.n_rows} x {self.n_cols}: {repr(point)} is an invalid coordinate')
        self.board[self.pos_index(point)] = pos

    def move(self, point, pos):
        point = self.ensure_point(point)
        pos = self.ensure_pos(pos)
        if self[point] != Pos.Empty:
            raise ValueError(f'Invalid move, {repr(pos)} is not empty')
        self[point] = pos
        self.update_liberties({point} | point.neighbors)
        self.remove_stones_without_liberties(pos.other)
        self.remove_stones_without_liberties(pos)
        self.board_history.append(self.state)

    def remove_stones_without_liberties(self, remove_color):
        points_removed = set()
        for r in range(self.n_rows):
            for c in range(self.n_cols):
                p = P(r, c)
                if self[p] == remove_color and self.get_liberties(p) == 0:
                    self[p] = Pos.Empty
                    points_removed.add(p)
        points_removed_plus_neighbors = points_removed | {n for p in points_removed for n in p.neighbors}
        self.update_liberties(points_removed_plus_neighbors)

    def get_liberties(self, point):
        point = self.ensure_point(point)
        return self.liberties[self.pos_index(point)]

    def set_liberties(self, point, liberties):
        point = self.ensure_point(point)
        self.liberties[self.pos_index(point)] = liberties

    def on_board(self, point):
        r, c = point
        return 0 <= r < self.n_rows and 0 <= c < self.n_cols

    def off_board(self, point):
        return not self.on_board(point)

    def update_all_liberties(self):
        all_points = {P(r, c) for r in range(self.n_rows) for c in range(self.n_cols)}
        self.update_liberties(all_points)

    def update_liberties(self, points):
        updated_liberties = [-1 for _ in range(len(self.liberties))]
        for point in points:
            if self.off_board(point):
                continue
            if updated_liberties[self.pos_index(point)] != -1:
                continue
            group = {point}
            group_liberties = set()

            def recurse(this_point):
                for neighboring_point in this_point.neighbors:
                    if self.off_board(neighboring_point):
                        continue
                    if self[neighboring_point] == Pos.Empty:
                        group_liberties.add(neighboring_point)
                    elif self[this_point] == self[neighboring_point] and neighboring_point not in group:
                        group.add(neighboring_point)
                        recurse(neighboring_point)

            recurse(point)
            for group_point in group:
                updated_liberties[self.pos_index(group_point)] = len(group_liberties)
        for i in range(len(self.liberties)):
            if updated_liberties[i] > -1:
                self.liberties[i] = updated_liberties[i]

    def valid_moves(self, pos):
        valid_moves = set()
        for r in range(self.n_rows):
            for c in range(self.n_cols):
                p = P(r, c)
                if self.can_place_stone_at(p, pos) and self.not_ko(p, pos):
                    valid_moves.add(p)
        return valid_moves

    def can_place_stone_at(self, point, pos):
        if self[point] != Pos.Empty:
            return False
        for neighboring_point in point.neighbors:
            if self.off_board(neighboring_point):
                continue
            neighboring_position = self[neighboring_point]
            if neighboring_position == Pos.Empty:
                return True
            neighboring_liberties = self.get_liberties(neighboring_point)
            if neighboring_position == pos and neighboring_liberties > 1:
                return True
            if neighboring_position == pos.other and neighboring_liberties == 1:
                return True
        return False

    def not_ko(self, point, pos):
        point = self.ensure_point(point)
        if all(self.off_board(neighboring_point) or self.get_liberties(neighboring_point) != 1
               for neighboring_point in point.neighbors):
            return True
        b = self.copy()
        b.move(point, pos)
        return b.state not in self.board_history

    @property
    def state(self):
        return b''.join(p.char for p in self.board)

    @property
    def state_string(self):
        return self.state.decode('utf8')

    star_points = {
        (19, 19): {(3,  3), (3,  9), (3,  15),
                   (9,  3), (9,  9), (9,  15),
                   (15, 3), (15, 9), (15, 15)}
    }

    def printable_board(self, pretty=False):
        def prettify(c):
            if not pretty:
                return c.decode('utf8')
            if c == b'b':
                return '●'
            if c == b'w':
                return '○'
            return c.decode('utf8')

        out = ''
        for r in reversed(range(self.n_rows)):
            out += '|'
            for c in range(self.n_cols):
                if (r, c) in self.star_points.get((self.n_rows, self.n_cols), set()):
                    out += '-' + prettify(self[r, c].char)
                else:
                    out += ' ' + prettify(self[r, c].char)
            out += '|\n'
        return out

    def printable_liberties(self):
        out = ''
        for r in reversed(range(self.n_rows)):
            out += '|'
            for c in range(self.n_cols):
                if (r, c) in self.star_points.get((self.n_rows, self.n_cols), set()):
                    out += ' {:02}'.format(self.get_liberties((r, c)))
                else:
                    out += '{:3}'.format(self.get_liberties((r, c)))
            out += '|\n'
        return out

    def __str__(self):
        return self.printable_board(pretty=True)

    def is_eye(self, point, pos):
        return all(self.off_board(neighboring_point) or
                   (self[neighboring_point] == pos and self.get_liberties(neighboring_point) > 1)
                   for neighboring_point in point.neighbors)

    def reasonable_moves_remain(self, pos):
        for valid_move in self.valid_moves(pos):
            if not self.is_eye(valid_move, pos):
                return True
        return False

    def reasonable_moves(self, pos):
        reasonable_moves = set()
        for valid_move in self.valid_moves(pos):
            if not self.is_eye(valid_move, pos):
                reasonable_moves.add(valid_move)
        return reasonable_moves

    def random_move(self, pos):
        pos = self.ensure_pos(pos)
        valid_moves = self.valid_moves(pos)
        valid_moves_that_keep_eyes = []
        for valid_move in valid_moves:
            if not self.is_eye(valid_move, pos):
                valid_moves_that_keep_eyes.append(valid_move)
        if not valid_moves_that_keep_eyes:
            return None
        return random.choice(valid_moves_that_keep_eyes)


MoveMemory = namedtuple('MoveMemory', 'board player move')
GameMemory = namedtuple('GameMemory', 'move_memories winner')


class NNBot:
    def __init__(self, board_size=(19, 19), explore=True):
        self.board_size = board_size
        self.explore = explore
        self.model = self.create_model()
        self.memory = []

    def genmove(self, board, pos):
        if not board.reasonable_moves_remain(pos):
            return 'resign'
        move_values, odds_win = self.model.predict(np.array([self.encode_board(board, pos)]))
        # TODO: Resign if low odds of winning.
        move_values = move_values.reshape(board.n_rows, board.n_cols)
        if self.explore:
            move_values = np.random.dirichlet(move_values.reshape(1, board.n_rows * board.n_cols)[0])
            move_values = move_values.reshape(board.n_rows, board.n_cols)
        move = None
        reasonable_moves = board.reasonable_moves(pos)
        while move is None or move not in reasonable_moves:
            move = np.unravel_index(np.argmax(move_values), (board.n_rows, board.n_cols))
            move_values[move] = 0
        self.memory.append(MoveMemory(board.state_string, pos, move))
        return move

    def report_winner(self, game_id, winning_player):
        gm = GameMemory(tuple(self.memory), winning_player)
        with open(f'games/{game_id}', 'wb'
                                      '') as f:
            pickle.dump(gm, f, pickle.HIGHEST_PROTOCOL)

    def create_model(self):
        board_input = Input(shape=(11, self.board_size[0], self.board_size[1]), name='board_input')
        conv1 = Conv2D(64, (3, 3), padding='same', activation='relu')(board_input)
        conv2 = Conv2D(64, (3, 3), padding='same', activation='relu')(conv1)
        conv3 = Conv2D(64, (3, 3), padding='same', activation='relu')(conv2)
        flat = Flatten()(conv3)
        processed_board = Dense(512)(flat)
        policy_hidden_layer = Dense(512, activation='relu')(processed_board)
        policy_output = Dense(self.board_size[0] * self.board_size[1], activation='softmax')(policy_hidden_layer)
        value_hidden_layer = Dense(512, activation='relu')(processed_board)
        value_output = Dense(1, activation='tanh')(value_hidden_layer)
        model = Model(inputs=board_input, outputs=[policy_output, value_output])
        return model

    @staticmethod
    def encode_board(board, player):
        valid_moves = board.valid_moves(player)
        t = np.zeros((11, board.n_rows, board.n_cols))
        for r in range(board.n_rows):
            for c in range(board.n_cols):
                p = P(r, c)
                t[0, r, c] = int(board[p] == Pos.Black and board.get_liberties(p) == 1)
                t[1, r, c] = int(board[p] == Pos.Black and board.get_liberties(p) == 2)
                t[2, r, c] = int(board[p] == Pos.Black and board.get_liberties(p) == 3)
                t[3, r, c] = int(board[p] == Pos.Black and board.get_liberties(p) > 3)
                t[4, r, c] = int(board[p] == Pos.White and board.get_liberties(p) == 1)
                t[5, r, c] = int(board[p] == Pos.White and board.get_liberties(p) == 2)
                t[6, r, c] = int(board[p] == Pos.White and board.get_liberties(p) == 3)
                t[7, r, c] = int(board[p] == Pos.White and board.get_liberties(p) > 3)
                t[8, r, c] = int(player == Pos.Black)
                t[9, r, c] = int(player == Pos.White)
                t[10, r, c] = int(p in valid_moves)
        return t


def play_games(n_games, board_size=19, verbose=True):
    for game_number in range(1, n_games+1):
        game_id = ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(6))
        print('Game: {:,} = {}'.format(game_number, game_id))
        black_player = NNBot(board_size=(board_size, board_size))
        white_player = NNBot(board_size=(board_size, board_size))
        board = Board(board_size, board_size)
        while 1:
            # Black's Move
            move = black_player.genmove(board, Pos.Black)
            if move == 'resign':
                if verbose:
                    print('White Wins!')
                black_player.report_winner(game_id, Pos.White)
                white_player.report_winner(game_id, Pos.White)
                break
            board.move(move, Pos.Black)
            if verbose:
                print(board)
            # White's Move
            move = white_player.genmove(board, Pos.White)
            if move == 'resign':
                if verbose:
                    print('Black Wins!')
                black_player.report_winner(game_id, Pos.Black)
                white_player.report_winner(game_id, Pos.Black)
                break
            board.move(move, Pos.White)
            if verbose:
                print(board)
