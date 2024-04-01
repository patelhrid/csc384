##########################################################
# This module contains the main Mancala game which maintains the board, score, and players.  
# Thanks to Daniel Bauer, Columbia University, for a version of Othello that this was based on
# 
# CSC 384 Fall 2023 Assignment 2
# version 1.0
##########################################################

import sys
import subprocess
from threading import Timer

from utils import *

def read_initial_board(init_board):
    if isinstance(init_board, list):
        return read_initial_board_list(init_board)
    elif isinstance(init_board, str):
        return read_initial_board_file(init_board)
    else:
        raise TypeError("init_board should be either a string filename or a list for the board")
    
def read_initial_board_list(init_board):
    """
    Creates a new board from the given list. The list should be 3 elements where

    Element 1: List representing # stones in Top Player's pockets
    Element 2: List representing # stones in Bottom Player's pockets
    Element 3: List representing Mancalas [TOP, BOTTOM]
    """
    return len(init_board[0]), Board([init_board[0], init_board[1]], init_board[2])

def read_initial_board_file(filename):
    """
    Reads the starting state from a file. File should be in this format:

    Line 1: # stones in Top Player's pockets
    Line 2: # stones in Bottom Player's pockets
    Line 3: # stones in Top Player's mancala
    Line 4: # stones in Bottom Player's mancala

    You can also check the example file.
    """
    with open(filename, 'r') as f:
        lines = f.readlines()

    top_stones = [int(x.strip()) for x in lines[0].split(",")]
    bottom_stones = [int(x.strip()) for x in lines[1].split(",")]
    top_mancala = int(lines[2].strip())
    bottom_mancala = int(lines[3].strip())

    assert len(top_stones) == len(bottom_stones)
    dimension = len(top_stones)

    pockets = [top_stones, bottom_stones]
    mancalas = [top_mancala, bottom_mancala]
    return dimension, Board(pockets, mancalas)


class Board(object):

    def __init__(self, pockets, mancalas):
        """
        Create a Mancala game board.
        """
        self.dimension = len(pockets[TOP])
        self.pockets = pockets
        self.mancalas = mancalas

    def __eq__(self, other):
        return self.pockets == other.pockets and self.mancalas == other.mancalas

    def __hash__(self):
        self.pockets = tuple(tuple(sublist) for sublist in self.pockets)
        self.mancalas = tuple(self.mancalas)
        return hash((self.pockets, self.mancalas))
    
    def draw_board(self, return_str=False):
        """
        Print the Mancala game board in a readable format.
        """
        dim = self.dimension
        mancalas = self.mancalas
        pockets = self.pockets

        max_num = max(max(mancalas), max(pockets[TOP]), max(pockets[BOTTOM]))

        grid_size = 3 + len(str(max_num))

        topper_row = " " * (grid_size - 1)
        for i in range(len(pockets[TOP])):
            extra_space = grid_size - 2 - len(str(i+1))
            topper_row += " T" + (str(i + 1)) + " " * extra_space

        top_row = " " * (grid_size - 1)
        for i in pockets[TOP]:
            extra_space = grid_size - 3 - len(str(i))
            top_row += "| " + (" " * extra_space) + str(i) + " "
        top_row += "|"

        middle_row = str(mancalas[TOP])
        extra_space = grid_size - 1 - len(str(mancalas[TOP]))
        middle_row += " " * extra_space + "|"
        middle_row += " " * (len(top_row) - grid_size - 1)
        extra_space = grid_size - 1 - len(str(mancalas[BOTTOM]))
        middle_row += "|" + " " * extra_space + str(mancalas[BOTTOM])

        bottom_row = " " * (grid_size - 1)
        for i in pockets[BOTTOM]:
            extra_space = grid_size - 3 - len(str(i))
            bottom_row += "| " + (" " * extra_space) + str(i) + " "
        bottom_row += "|"

        bottomer_row = " " * (grid_size - 1)
        for i in range(len(pockets[TOP])):
            num = len(pockets[TOP]) - i
            extra_space = grid_size - 2 - len(str(num))
            bottomer_row += " B" + (str(num)) + " " * extra_space

        msg = topper_row
        msg += "\n" + "-" * (dim + 2) * grid_size
        msg += "\n" + top_row
        msg += "\n" + middle_row
        msg += "\n" + bottom_row
        msg += "\n" + "-" * (dim + 2) * grid_size
        msg += "\n" + bottomer_row
        
        if return_str:
            return msg
        else:
            print(msg)
    
    def get_board_list(self):
        """
        Get the board in the format of a list.
        """
        data = [", ".join([str(x) for x in self.pockets[0]]), 
                ", ".join([str(x) for x in self.pockets[1]]), 
                ", ".join([str(x) for x in self.mancalas])]
        return data
    
    def get_possible_moves(self, player):
        """
        Return a list of all possible indices (representing pockets) that the 
        current player can play on the current board.
        """
        moves = []
        for j in range(self.dimension):
            #if the pocket has at least one piece
            if self.pockets[player][j] > 0: 
                moves.append(j)
        return moves
   

class MancalaGameManager(object):

    def __init__(self, dimension=None, initial_board=None, current_player=TOP):
        """
        Create an object to manage the Mancala game.
        Keeps track of the current board and the current player.
        """
        assert dimension is not None or initial_board is not None
        if initial_board is not None:
            if dimension is not None:
                print("Initializing game from", initial_board, ", dimension parameter ignored.")
            self.dimension, self.board = read_initial_board(initial_board)
        elif dimension is not None:
            self.dimension = dimension
            self.board = Board([[4] * self.dimension, [4] * self.dimension], [0, 0])
        
        self.current_player = current_player

    def get_score(self):
        return self.board.mancalas[TOP], self.board.mancalas[BOTTOM]

    def get_possible_moves(self, opponent=False):
        """
        Return a list of all possible indices (representing pockets) that the 
        current player can play on the current board.

        if opponent = True, get possible moves for the opponent of the 
        current player instead. 
        """
        player = self.current_player
        if opponent:
            player = get_opponent(self.current_player)

        return self.board.get_possible_moves(player)
    
    def end_game(self):
        """
        Call this function at the end of the game to move all remaining stones
        on the board.
        Opponent just moved and should have no moves left, only current player
        has stones to move.
        """
        new_board = []
        new_mancalas = self.board.mancalas.copy()
        for row in self.board.pockets: 
            new_board.append(list(row[:]))

        for player in range(len(self.board.pockets)):
            value = 0
            for j in range(len(new_board[player])):
                value +=  new_board[player][j]
                new_board[player][j] = 0
            new_mancalas[player] += value

        final = []
        for row in new_board: 
            final.append(tuple(row))

        self.board.pockets = tuple(final)
        self.board.mancalas = tuple(new_mancalas)

    def play(self, move):
        """
        Play the move.
        :param move: index of the pocket.
        """
        move = int(move)

        if self.board.pockets[self.current_player][move] == 0:
            raise InvalidMoveError("Invalid move: The pocket is empty.")
     
        self.board = play_move(self.board, self.current_player, move)
        self.current_player = get_opponent(self.current_player)

    def draw_board(self):
        self.board.draw_board()

    def get_winner(self):
        """
        Returns the player number of the player with more stones in their mancala.
        """
        return TOP if self.board.mancalas[TOP] > self.board.mancalas[BOTTOM] else BOTTOM


def play_move(board, player, move):
    """
    Play a move on the current board. 
    :param board: the current board
    :param player: the player to move.
    :param move: the move to perform. the index of the pocket.
    """  
    side = player

    new_board = []
    for row in board.pockets: 
        new_board.append(list(row[:]))
    new_mancalas = [board.mancalas[TOP], board.mancalas[BOTTOM]]

    stone_count = board.pockets[side][move] # find the number of stones in the pocket
    new_board[side][move] = 0 # set to 0

    if (player == BOTTOM):
        direction = True
        ind = move + 1
    else:
        direction = False
        ind = move - 1

    while stone_count > 0: #deposit stones around the board
        # we are at the end of the board
        if ind > (len(board.pockets[side])-1) or ind < 0:
            # swap side and change direction
            side = TOP if side == BOTTOM else BOTTOM 
            direction = not direction 

            #if we are at the end of the board, 
            #deposit stone in a mancala before we continue
            if ind > (len(board.pockets[side]) - 1): 
                if player == BOTTOM:
                    stone_count -= 1
                    new_mancalas[player] += 1
                ind = len(board.pockets[side])-1
            else:
                if player == TOP:
                    stone_count -= 1
                    new_mancalas[player] += 1
                ind = 0

        # ran out of stones
        if stone_count == 0: 
            break

        #but if not, put a stone in a pocket and decrement stone count
        new_board[side][ind] = new_board[side][ind] + 1
        stone_count -= 1

        #do we have a capture?
        if stone_count == 0 and new_board[side][ind] == 1 and side == player: 
            captures = new_board[get_opponent(side)][ind] #if yes capture stones in the opposite pit
            new_board[get_opponent(side)][ind] = 0
            new_mancalas[player] += captures

        if direction: 
            ind += 1
        else: 
            ind -= 1

    #return a copy of the board details
    final_pockets = []
    for row in new_board: 
        final_pockets.append(tuple(row))

    return Board(final_pockets, new_mancalas)


class Player(object):
    def __init__(self, player_num, name="Human"):
        """
        Initialize a player of the game.

        player_num: TOP or BOTTOM representing which player they are.
        name: a cool name for this player (give a string) 
        """
        self.name = name
        self.player = player_num

    def get_move(self, manager):
        pass  


class AiPlayerInterface(Player):

    TIMEOUT = 120 # For me to test on bigger boards

    def __init__(self, filename, player, limit, caching=False, heuristic=0):

        self.player = player
        # NOTE: if an error occurs here, changing 'python' to 'python3' may fix it, and vice versa.
        self.process = subprocess.Popen(['python', filename], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        name = self.process.stdout.readline().decode("ASCII").strip()
        print("AI introduced itself as: {}".format(name))
        self.name = name
        self.process.stdin.write((str(player) + "," + str(limit) + "," + str(caching) + "," + str(heuristic) + "\n").encode("ASCII"))
        self.process.stdin.flush()

    def timeout(self): 
        sys.stderr.write("{} timed out.".format(self.name))
        self.process.kill() 
        self.timed_out = True

    def get_move(self, manager):
        white_score, dark_score = manager.board.mancalas[TOP], manager.board.mancalas[BOTTOM]
        self.process.stdin.write("SCORE {} {}\n".format(white_score, dark_score).encode("ASCII"))
        self.process.stdin.flush()
        self.process.stdin.write("{}\n".format(str(manager.board.pockets)).encode("ASCII"))
        self.process.stdin.flush()
        self.process.stdin.write("{}\n".format(str(manager.board.mancalas)).encode("ASCII"))
        self.process.stdin.flush()

        timer = Timer(AiPlayerInterface.TIMEOUT, lambda: self.timeout())
        self.timed_out = False
        timer.start()

        # Wait for the AI call
        move_s = self.process.stdout.readline().decode("ASCII")
        if self.timed_out:  
            raise AiTimeoutError
        timer.cancel()

        return move_s
    
    def kill(self, manager):
        white_score, dark_score = manager.get_score()
        self.process.stdin.write("FINAL {} {}\n".format(white_score, dark_score).encode("ASCII"))
        self.process.kill() 
