###############################################################################
# This file contains helper functions and the heuristic functions
# for our AI agents to play the Mancala game.
#
# CSC 384 Fall 2023 Assignment 2
# version 1.0
###############################################################################

import sys

###############################################################################
### DO NOT MODIFY THE CODE BELOW

### Global Constants ###
TOP = 0
BOTTOM = 1

### Errors ###
class InvalidMoveError(RuntimeError):
    pass

class AiTimeoutError(RuntimeError):
    pass

### Functions ###
def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def get_opponent(player):
    if player == BOTTOM:
        return TOP
    return BOTTOM

### DO NOT MODIFY THE CODE ABOVE
###############################################################################


def heuristic_basic(board, player):
    """
    Compute the heuristic value of the current board for the current player 
    based on the basic heuristic function.

    :param board: the current board.
    :param player: the current player.
    :return: an estimated utility of the current board for the current player.
    """

    player_mancala = board.mancalas[player]
    opponent = get_opponent(player)
    opponent_mancala = board.mancalas[opponent]

    return player_mancala - opponent_mancala


def heuristic_advanced(board, player): 
    """
    Compute the heuristic value of the current board for the current player
    based on the advanced heuristic function.

    :param board: the current board object.
    :param player: the current player.
    :return: an estimated heuristic value of the current board for the current player.
    """

    player_mancala = board.mancalas[player]
    opponent = get_opponent(player)
    opponent_mancala = board.mancalas[opponent]

    empty_pockets_player = sum([1 for stones in board.pockets[player] if stones == 0])
    empty_pockets_opponent = sum([1 for stones in board.pockets[opponent] if stones == 0])

    heuristic_value = player_mancala - opponent_mancala + (empty_pockets_player - empty_pockets_opponent)

    return heuristic_value
