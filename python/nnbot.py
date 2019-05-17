from collections import namedtuple
import itertools
import os
import random
import string
from time import time
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


def board_point_to_np_index(go_board_point, board_size):
    return np.ravel_multi_index((go_board_point.x, go_board_point.y), (board_size, board_size))


def np_index_to_board_point(np_index, board_size):
    return P(*np.unravel_index(np_index, (board_size, board_size)))


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
        self.root = Node(board.copy(), {}, 0, -99.9)

    @staticmethod
    def values(model, board, player):
        move_values, position_value = model.predict(np.array([encode_board(board, player)]))
        return move_values[0], position_value[0][0]

    def init_good_moves(self, model, node, player):
        valid_moves = set(node.board.valid_moves(player))
        if not valid_moves:
            node.value = 0.0
            return
        if valid_moves != set(node.board.valid_moves(player.other)):
            import ipdb
            ipdb.set_trace()
        move_values, position_value = self.values(model, node.board, player)
        node.value = position_value
        move_values = [(np_index_to_board_point(i, node.board.size), move_values[i]) for i in np.argsort(move_values)]
        move_values = [mv for mv in move_values if mv[0] in valid_moves]
        min_value = move_values[0][1]
        max_value = move_values[-1][1]
        move_value_cutoff = min_value + ((max_value - min_value) * (2 / 3))
        for value, move in np.flip(move_values):
            if value < move_value_cutoff:
                break
            b = node.board.copy()
            b.play(move, player)
            assert move in valid_moves
            node.moves[move] = Node(b, {}, 0, 99.9 if player == self.player else -99.9)

    @staticmethod
    def select_weighted_random_move(moves, max):
        moves = [(move, node.value) for move, node in moves.items()]
        move_values = np.array([m[1] for m in moves])
        move_values += np.random.dirichlet(np.ones(move_values.shape))
        if max:
            return moves[np.argmax(move_values)][0]
        else:
            return moves[np.argmin(move_values)][0]

    def find_new_root(self, needle_board, haystack_tree_node):
        if needle_board == haystack_tree_node.board:
            return haystack_tree_node
        for haystack_next_tree_node in haystack_tree_node.moves.values():
            new_root = self.find_new_root(needle_board, haystack_next_tree_node)
            if new_root:
                return new_root

    def update_board(self, board):
        new_root = self.find_new_root(board.copy(), self.root)
        if new_root is not None:
            self.root = new_root
        else:
            self.root = Node(board.copy(), {}, 0, -99.9)

    def deepen(self, model, node=None, player=None):
        node = node or self.root
        player = player or self.player
        node.visits += 1
        if node.visits == 1:
            self.init_good_moves(model, node, player)
            return
        if not node.moves:
            node.value = 0.0
            return
        move = self.select_weighted_random_move(node.moves, player == self.player)
        self.deepen(model, node.moves[move], player.other)
        if player == self.player:
            node.value = min(n.value for n in node.moves.values())
        else:
            node.value = max(n.value for n in node.moves.values())

    def pick_move(self):
        if not self.root.moves:
            return
        return max(((m, self.root.moves[m].visits) for m in self.root.moves), key=lambda i: i[1])[0]


class NNBot:
    def __init__(self, model_file='model.h5', board_size=19, explore=True):
        self.model_file = model_file
        self.board_size = board_size
        self.explore = explore
        self.model = self.create_model()
        self.memory = []
        self.game_trees = {}

    def intuit_move(self, board, pos):
        if self.board_size != board.size:
            raise ValueError(f'Expected board size {self.board_size}, got board size {board.size}')
        valid_moves = set(board.valid_moves(pos))
        if not valid_moves:
            return True, None
        move_values, odds_win = self.model.predict(np.array([encode_board(board, pos)]))
        # TODO: Resign if low odds of winning.
        move_values = move_values.reshape(board.size, board.size)
        if self.explore and random.random() < 0.1:
            move_values = move_values.reshape(board.size**2) + np.random.dirichlet(np.ones(board.size**2))
            move_values = move_values.reshape(board.size, board.size)
        move = None
        while move not in valid_moves:
            move = np.unravel_index(np.argmax(move_values), (board.size, board.size))
            move_values[move] -= 999
            move = P(*move)
        self.memory.append(MoveMemory(board.copy(), pos, move))
        return False, move

    def genmove(self, board, pos):
        if not board.valid_moves(pos):
            return True, None
        if pos not in self.game_trees:
            self.game_trees[pos] = GameTree(board, pos)
        self.game_trees[pos].update_board(board)
        start_time = time()
        while time() - start_time < 0.1:
            self.game_trees[pos].deepen(self.model)
        move = self.game_trees[pos].pick_move()
        if not move:
            return True, None
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
            y[board_point_to_np_index(move, self.board_size)] = 1 if won else -1
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
            loss_weights=[1, 1])
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


def gen_games(n_games, board_size=19, verbose=True, intuit=False):
    genmove_function = 'intuit_move' if intuit else 'genmove'
    player = NNBot(board_size=board_size)
    for game_number in range(1, n_games+1):
        if verbose:
            print('Game: {:,}'.format(game_number))
        board = Board(board_size)
        while 1:
            if verbose:
                print("Game {:,}: Black's Move".format(game_number))
            resign, move = getattr(player, genmove_function)(board, Black)
            if resign:
                if verbose:
                    print('White Wins!')
                player.report_winner(White)
                break
            assert move in board.valid_moves(Black)
            board.play(move, Black)
            if verbose:
                print(board)
            if verbose:
                print("Game {:,}: White's Move".format(game_number))
            resign, move = getattr(player, genmove_function)(board, White)
            if resign:
                if verbose:
                    print('Black Wins!')
                player.report_winner(Black)
                break
            assert move in board.valid_moves(White)
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
    import ipdb
    ap = argparse.ArgumentParser()
    ap.add_argument('command', choices=['gengames', 'train', 'trymodel'])
    ap.add_argument('--verbose', action='store_true')
    ap.add_argument('--intuit', action='store_true')
    ap.add_argument('--models', nargs=2)
    args = ap.parse_args()
    if args.verbose:
        with ipdb.launch_ipdb_on_exception():
            if args.command == 'gengames':
                gen_games(200, 9, args.verbose, args.intuit)
            elif args.command == 'train':
                train(9)
            elif args.command == 'trymodel':
                try_model(args.models[0], args.models[1], 9)
    else:
        if args.command == 'gengames':
            gen_games(50, 9, args.verbose, args.intuit)
        elif args.command == 'train':
            train(9)
        elif args.command == 'trymodel':
            try_model(args.models[0], args.models[1], 9)
