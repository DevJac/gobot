import enum
import functools
import os
import random
import string
import pickle
import numpy as np
from scipy.stats import binom_test
from collections import namedtuple
from tqdm import tqdm
import keras
from keras.models import Model, load_model
from keras.layers import Conv2D, Dense, Flatten, Input
from keras.optimizers import SGD


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
        if len(s) != n_rows * n_cols:
            raise ValueError(f'Expected board of size {n_rows} x {n_cols}, but got string of length {len(s)}')
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
            if c == b'w':
                return '●'
            if c == b'b':
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
            out += ' |\n'
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
            out += ' |\n'
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


def encode_board(board, player):
    board.update_all_liberties()
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


class NNBot:
    def __init__(self, board_size=(19, 19), explore=True, model_file=None):
        self.model_file = model_file or 'model.h5'
        self.board_size = board_size
        self.explore = explore
        self.model = self.create_model()
        self.memory = []
        self.loaded_files = set()
        self.X = []
        self.Y0 = []
        self.Y1 = []

    def clear_memory(self):
        self.memory = []
        self.loaded_files = set()
        self.X = []
        self.Y0 = []
        self.Y1 = []

    def train(self, batch_size=1000, epochs=1, learning_rate=0.01):
        for game_file in tqdm(os.listdir('games')):
            game_file = game_file.partition('.')[0]
            if game_file in self.loaded_files:
                continue
            x = np.load(os.path.join('games', game_file + '.X.npy'))
            y0 = np.load(os.path.join('games', game_file + '.Y0.npy'))
            y1 = np.load(os.path.join('games', game_file + '.Y1.npy'))
            self.X.append(x)
            self.Y0.append(y0)
            self.Y1.append(y1)
            self.loaded_files.add(game_file)
        self.model.compile(
            optimizer=SGD(lr=learning_rate),
            loss=['categorical_crossentropy', 'mse'],
            loss_weights=[2, 1])
        X = np.concatenate(self.X)
        Y0 = np.concatenate(self.Y0)
        Y1 = np.concatenate(self.Y1)
        print('Training on {:,} games with {:,} moves'.format(len(os.listdir('games')) // 3, X.shape[0]))
        print(f'Shapes: {X.shape} {Y0.shape} {Y1.shape}')
        self.model.fit(X, [Y0, Y1], batch_size=batch_size, epochs=epochs)
        self.model.save(f'{self.model_file}.new')
        os.rename(f'{self.model_file}.new', self.model_file)
        self.model = self.create_model()
        print('Training complete, model saved')

    def save_model(self):
        self.model.save(self.model_file)

    def genmove(self, board, pos):
        valid_moves = board.valid_moves(pos)
        if not valid_moves:
            return 'resign'
        move_values, odds_win = self.model.predict(np.array([encode_board(board, pos)]))
        # TODO: Resign if low odds of winning.
        move_values = move_values.reshape(board.n_rows, board.n_cols)
        if self.explore and random.random() < 0.1:
            move_values = np.random.dirichlet(move_values.reshape(1, board.n_rows * board.n_cols)[0] + 1)
            move_values = move_values.reshape(board.n_rows, board.n_cols)
        move = None
        while move is None or move not in valid_moves:
            move = np.unravel_index(np.argmax(move_values), (board.n_rows, board.n_cols))
            move_values[move] = 0
        self.memory.append(MoveMemory(board.state_string, pos, move))
        return move

    def report_winner(self, game_id, winning_player):
        X = []
        Y = []
        for move_memory in self.memory:
            board = Board.from_state_string(move_memory.board, self.board_size[0], self.board_size[1])
            player = move_memory.player
            move = move_memory.move
            won = winning_player == player
            X.append(encode_board(board, player))
            y = np.zeros(self.board_size)
            y[move] = 1 if won else -1
            Y.append([y, 1 if won else 0])
        X = np.array(X)
        Y0 = np.array([y[0] for y in Y])
        Y0 = Y0.reshape(Y0.shape[0], self.board_size[0] * self.board_size[1])
        Y1 = np.array([y[1] for y in Y])
        np.save(os.path.join('games', game_id + '.X'), X)
        np.save(os.path.join('games', game_id + '.Y0'), Y0)
        np.save(os.path.join('games', game_id + '.Y1'), Y1)
        self.memory = []

    def create_model(self):
        if os.path.exists(self.model_file):
            print('Loading model from file')
            return load_model(self.model_file)
        board_input = Input(shape=(11, self.board_size[0], self.board_size[1]), name='board_input')
        conv1 = Conv2D(100, (3, 3), padding='same', activation='relu')(board_input)
        conv2 = Conv2D(100, (3, 3), padding='same', activation='relu')(conv1)
        conv3 = Conv2D(100, (3, 3), padding='same', activation='relu')(conv2)
        flat = Flatten()(conv3)
        processed_board = Dense(1000)(flat)
        policy_hidden_layer = Dense(1000, activation='relu')(processed_board)
        policy_output = Dense(self.board_size[0] * self.board_size[1], activation='softmax')(policy_hidden_layer)
        value_hidden_layer = Dense(1000, activation='relu')(processed_board)
        value_output = Dense(1, activation='tanh')(value_hidden_layer)
        model = Model(inputs=board_input, outputs=[policy_output, value_output])
        return model


def self_play(n_games, board_size=19, verbose=True):
    player = NNBot(board_size=(board_size, board_size))
    for game_number in range(1, n_games+1):
        game_id = ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(6))
        if verbose:
            print('Game: {:,} = {}'.format(game_number, game_id))
        board = Board(board_size, board_size)
        while 1:
            # Black's Move
            move = player.genmove(board, Pos.Black)
            if move == 'resign':
                if verbose:
                    print('White Wins!')
                player.report_winner(game_id, Pos.White)
                break
            board.move(move, Pos.Black)
            if verbose:
                print(board)
            # White's Move
            move = player.genmove(board, Pos.White)
            if move == 'resign':
                if verbose:
                    print('Black Wins!')
                player.report_winner(game_id, Pos.Black)
                break
            board.move(move, Pos.White)
            if verbose:
                print(board)


def train(board_size=19):
    player = NNBot(board_size=(board_size, board_size))
    player.train()


def compete(p1_model_file, p2_model_file, board_size=19):
    p1_wins = 0
    p2_wins = 0
    model_files = [p1_model_file, p2_model_file]
    for game in range(100):
        keras.backend.clear_session()
        game_id = ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(6))
        random.shuffle(model_files)
        black_player = NNBot(board_size=(board_size, board_size), model_file=model_files[0])
        white_player = NNBot(board_size=(board_size, board_size), model_file=model_files[1])
        board = Board(board_size, board_size)
        while 1:
            # Black's Move
            move = black_player.genmove(board, Pos.Black)
            if move == 'resign':
                print(f'White Wins! ({white_player.model_file})')
                black_player.report_winner(game_id, Pos.White)
                white_player.report_winner(game_id, Pos.White)
                if white_player.model_file == p1_model_file:
                    p1_wins += 1
                else:
                    p2_wins += 1
                break
            board.move(move, Pos.Black)
            print(board)
            print('P1 ({}) wins: {:,}    P2 ({}) wins: {:,}'.format(
                'Black' if black_player.model_file == p1_model_file else 'White',
                p1_wins,
                'Black' if black_player.model_file == p2_model_file else 'White',
                p2_wins))
            # White's Move
            move = white_player.genmove(board, Pos.White)
            if move == 'resign':
                print(f'Black Wins! ({black_player.model_file})')
                black_player.report_winner(game_id, Pos.Black)
                white_player.report_winner(game_id, Pos.Black)
                if black_player.model_file == p1_model_file:
                    p1_wins += 1
                else:
                    p2_wins += 1
                break
            board.move(move, Pos.White)
            print(board)
            print('P1 ({}) wins: {:,}    P2 ({}) wins: {:,}'.format(
                'Black' if black_player.model_file == p1_model_file else 'White',
                p1_wins,
                'Black' if black_player.model_file == p2_model_file else 'White',
                p2_wins))
        print('P1 ({}) wins: {:,}    P2 ({}) wins: {:,}    {:.2}'.format(
            'Black' if black_player.model_file == p1_model_file else 'White',
            p1_wins,
            'Black' if black_player.model_file == p2_model_file else 'White',
            p2_wins,
            binom_test(p2_wins, p1_wins + p2_wins, 0.5)))
