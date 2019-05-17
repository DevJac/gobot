from collections import namedtuple
import itertools
import os
import random
import string
from tqdm import tqdm
import numpy as np
from scipy.stats import binom_test
import keras
from keras.models import Model, load_model
from keras.layers import Conv2D, Dense, Flatten, Input
from keras.optimizers import SGD
from board import P, Black, White, Board


MoveMemory = namedtuple('MoveMemory', 'board player move')


def encode_board(board, player):
    valid_moves = set(board.valid_moves(player))
    t = np.zeros((11, board.size, board.size))
    for r, c in itertools.product(range(board.size), repeat=2):
        p = P(r, c)
        t[0, r, c] = int(board[p] == Black and board.liberties(p) == 1)
        t[1, r, c] = int(board[p] == Black and board.liberties(p) == 2)
        t[2, r, c] = int(board[p] == Black and board.liberties(p) == 3)
        t[3, r, c] = int(board[p] == Black and board.liberties(p) > 3)
        t[4, r, c] = int(board[p] == White and board.liberties(p) == 1)
        t[5, r, c] = int(board[p] == White and board.liberties(p) == 2)
        t[6, r, c] = int(board[p] == White and board.liberties(p) == 3)
        t[7, r, c] = int(board[p] == White and board.liberties(p) > 3)
        t[8, r, c] = int(player == Black)
        t[9, r, c] = int(player == White)
        t[10, r, c] = int(p in valid_moves)
    return t


class Node:
    def __init__(self, board, moves, visits, value):
        self.board = board
        self.moves = moves
        self.visits = visits
        self.value = value

    def __repr__(self):
        return '<Board: {:,} {:,} {:.3}>'.format(
            len(self.moves),
            self.visits,
            self.value)


class GameTree:
    def __init__(self, board, player):
        self.player = player
        self.root = Node(board, {}, 0, -999.9)

    @staticmethod
    def values(model, board, player):
        move_values, position_value = model.predict(np.array([encode_board(board, player)]))
        return move_values[0], position_value[0][0]

    @staticmethod
    def np_to_go(move, board_size):
        return P(*np.unravel_index(move, (board_size, board_size)))

    def init_good_moves(self, model, node, player):
        move_values, position_value = self.values(model, node.board, player)
        node.value = position_value
        move_value_cutoff = np.min(move_values) + ((np.max(move_values) - np.min(move_values)) * (2 / 3))
        sorted_moves = np.argsort(move_values)
        for move in np.flip(sorted_moves):
            if move_values[move] < move_value_cutoff:
                break
            move = self.np_to_go(move, node.board.size)
            b = node.board.copy()
            b.play(move, player)
            value = 99.9 if player == self.player else -99.9
            node.moves[move] = Node(b, {}, 0, value)

    @staticmethod
    def select_weighted_random_move(moves):
        moves = [(m, n.value) for m, n in moves.items()]
        move_values = np.array([m[1] for m in moves])
        move_values += np.random.dirichlet(np.ones(move_values.shape))
        return moves[np.argmax(move_values)][0]

    def deepen(self, model, node=None, player=None):
        node = node or self.root
        player = player or self.player
        if not node.moves:
            self.init_good_moves(model, node, player)
            return
        node.visits += 1
        move = self.select_weighted_random_move(node.moves)
        self.deepen(model, node.moves[move], player.other)
        if player == self.player:
            node.value = min(n.value for n in node.moves.values())
        else:
            node.value = min(n.value for n in node.moves.values())

    def pick_move(self):
        return max(((m, self.root.moves[m].value) for m in self.root.moves), key=lambda i: i[1])[0]


class NNBot:
    def __init__(self, model_file='model.h5', board_size=19, explore=True):
        self.model_file = model_file
        self.board_size = board_size
        self.explore = explore
        self.model = self.create_model()
        self.memory = []

    def genmove(self, board, pos):
        if self.board_size != board.size:
            raise ValueError(f'Expected board size {self.board_size}, got board size {board.size}')
        valid_moves = set(board.valid_moves(pos))
        if not valid_moves:
            return True, None
        move_values, odds_win = self.model.predict(np.array([encode_board(board, pos)]))
        # TODO: Resign if low odds of winning.
        move_values = move_values.reshape(board.size, board.size)
        if self.explore and random.random() < 0.1:
            move_values = np.random.dirichlet(move_values.reshape(1, board.size**2)[0] + 1)
            move_values = move_values.reshape(board.size, board.size)
        move = None
        while move not in valid_moves:
            move = np.unravel_index(np.argmax(move_values), (board.size, board.size))
            move_values[move] -= 999
            move = P(*move)
        self.memory.append(MoveMemory(board.copy(), pos, move))
        return False, move

    def report_winner(self, winning_player, game_id=None):
        game_id = game_id or ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(8))
        X = []
        Y = []
        for move_memory in self.memory:
            board = move_memory.board
            player = move_memory.player
            move = move_memory.move
            won = winning_player == player
            X.append(encode_board(board, player))
            y = np.zeros(board.size**2)
            y[np.ravel_multi_index((move.x, move.y), (self.board_size, self.board_size))] = 1 if won else -1
            Y.append([y, 1 if won else 0])
        X = np.array(X)
        Y0 = np.array([y[0] for y in Y])
        Y0 = Y0.reshape(Y0.shape[0], self.board_size**2)
        Y1 = np.array([y[1] for y in Y])
        np.save(os.path.join('games', game_id + '.X'), X)
        np.save(os.path.join('games', game_id + '.Y0'), Y0)
        np.save(os.path.join('games', game_id + '.Y1'), Y1)
        self.memory = []

    def train(self, batch_size=1000, epochs=1, learning_rate=0.01):
        X = []
        Y0 = []
        Y1 = []
        for game_file in tqdm(os.listdir('games')):
            game_file = game_file.partition('.')[0]
            x = np.load(os.path.join('games', game_file + '.X.npy'))
            y0 = np.load(os.path.join('games', game_file + '.Y0.npy'))
            y1 = np.load(os.path.join('games', game_file + '.Y1.npy'))
            X.append(x)
            Y0.append(y0)
            Y1.append(y1)
        self.model.compile(
            optimizer=SGD(lr=learning_rate),
            loss=['categorical_crossentropy', 'mse'],
            loss_weights=[2, 1])
        X = np.concatenate(X)
        Y0 = np.concatenate(Y0)
        Y1 = np.concatenate(Y1)
        print('Training on {:,} games with {:,} moves'.format(len(os.listdir('games')) // 3, X.shape[0]))
        self.model.fit(X, [Y0, Y1], batch_size=batch_size, epochs=epochs)
        self.save_model()
        self.model = self.create_model()
        print('Training complete, model saved')

    def create_model(self):
        if os.path.exists(self.model_file):
            print('Loading model from file')
            return load_model(self.model_file)
        board_input = Input(shape=(11, self.board_size, self.board_size), name='board_input')
        conv1 = Conv2D(100, (3, 3), padding='same', activation='relu')(board_input)
        conv2 = Conv2D(100, (3, 3), padding='same', activation='relu')(conv1)
        conv3 = Conv2D(100, (3, 3), padding='same', activation='relu')(conv2)
        flat = Flatten()(conv3)
        processed_board = Dense(1000)(flat)
        policy_hidden_layer = Dense(1000, activation='relu')(processed_board)
        policy_output = Dense(self.board_size**2, activation='softmax')(policy_hidden_layer)
        value_hidden_layer = Dense(1000, activation='relu')(processed_board)
        value_output = Dense(1, activation='tanh')(value_hidden_layer)
        model = Model(inputs=board_input, outputs=[policy_output, value_output])
        return model

    def save_model(self):
        self.model.save(f'{self.model_file}.new')
        os.rename(f'{self.model_file}.new', self.model_file)


def gen_games(n_games, board_size=19, verbose=True):
    player = NNBot(board_size=board_size)
    for game_number in range(1, n_games+1):
        if verbose:
            print('Game: {:,}'.format(game_number))
        board = Board(board_size)
        while 1:
            # Black's Move
            resign, move = player.genmove(board, Black)
            if resign:
                if verbose:
                    print('White Wins!')
                player.report_winner(White)
                break
            board.play(move, Black)
            if verbose:
                print(board)
            # White's Move
            resign, move = player.genmove(board, White)
            if resign:
                if verbose:
                    print('Black Wins!')
                player.report_winner(Black)
                break
            board.play(move, White)
            if verbose:
                print(board)


def train(board_size=19):
    player = NNBot(board_size=board_size)
    player.train()


def try_model(p1_model_file, p2_model_file, board_size=19):
    def turn(current_player, other_player, color):
        resign, move = black_player.genmove(board, Black)
        if resign:
            print(f'{color.other} Wins! ({other_player.model_file})')
            black_player.report_winner(color.other)
            white_player.report_winner(color.other)
            wins[current_player.model_file] += 1
            return True
        board.play(move, color)
        print(board)
        print('{} ({}) wins: {:,}    {} ({}) wins: {:,}    {:.2}'.format(
            p1_model_file,
            'Black' if black_player.model_file == p1_model_file else 'White',
            wins[p1_model_file],
            p2_model_file,
            'Black' if black_player.model_file == p2_model_file else 'White',
            wins[p2_model_file],
            binom_test(wins[p1_model_file], sum(wins.values(), 0.5))))
    wins = {p1_model_file: 0, p2_model_file: 0}
    model_files = [p1_model_file, p2_model_file]
    for game in range(100):
        keras.backend.clear_session()
        random.shuffle(model_files)
        black_player = NNBot(board_size=board_size, model_file=model_files[0])
        white_player = NNBot(board_size=board_size, model_file=model_files[1])
        board = Board(board_size)
        while 1:
            if turn(black_player, white_player, Black):
                break
            if turn(white_player, black_player, White):
                break
        print('{} ({}) wins: {:,}    {} ({}) wins: {:,}    {:.2}'.format(
            p1_model_file,
            'Black' if black_player.model_file == p1_model_file else 'White',
            wins[p1_model_file],
            p2_model_file,
            'Black' if black_player.model_file == p2_model_file else 'White',
            wins[p2_model_file],
            binom_test(wins[p1_model_file], sum(wins.values(), 0.5))))


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('command', choices=['gengames', 'train', 'trymodel'])
    ap.add_argument('--verbose', action='store_true')
    ap.add_argument('--models', nargs=2)
    args = ap.parse_args()
    if args.command == 'gengames':
        gen_games(200, 9, args.verbose)
    elif args.command == 'train':
        train(9)
    elif args.command == 'trymodel':
        try_model(args.models[0], args.models[1], 9)
