###############################################################################
# This is an AI that randomly chooses a legal move. Play against this AI to
# get familiar with the game. Not the best AI opponent, so don't get flattered if you win! 
# You can also have your AI compete against your AI to test its performance. :D
#
# Thanks to Daniel Bauer, Columbia University, for a version of Othello that this was based on
#
# CSC 384 Fall 2023 Assignment 2
# version 1.0
###############################################################################

import random
import time

# You can also use the functions in othello_shared to write your AI 
from mancala_game import Board

def select_move(board, player):
    """
    Given a board and a player color, decide on a move. 
    The return value is a tuple of integers (i,j), where
    i is the column and j is the row on the board.  
    """

    # We just get a list of all permitted moves in this state and select a random one!
    i = None
    moves = board.get_possible_moves(player) # Returns a list of (column, row) tuples.
    if (len(moves) > 0):
        i = random.choice(moves)

    time.sleep(0.5) # Delay, so Randy doesn't look as simple as it really is.  
    return i

def run_ai():
    """
    This function establishes communication with the game manager. 
    It first introduces itself and receives its color. 
    Then it repeatedly receives the current score and current board state
    until the game is over. 
    """
    print("Randy") # First line is the name of this AI  

    arguments = input().split(",")
    color = int(arguments[0]) # We read the color: 1 for dark (goes first), 2 for light. 
    
    while True: # This is the main loop
        # Read in the current game status, for example:
        # "SCORE 2 2" or "FINAL 33 31" if the game is over.
        # The first number is the score for player 1 (dark), the second for player 2 (light)
        try:
            next_input = input() 
        except EOFError:
            continue
        status, dark_score_s, light_score_s = next_input.strip().split()

        if status == "FINAL": # Game is over. 
            print 
        elif status == "SCORE": 
            pockets = eval(input()) # Read in the input and turn it into a Python object
            mancalas = eval(input()) # Read in the input and turn it into a Python object

            # Select the move and send it to the manager
            move = select_move(Board(pockets, mancalas), color)
            print("{}".format(move))


if __name__ == "__main__":
    run_ai()
