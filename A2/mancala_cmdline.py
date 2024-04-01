###############################################################################
# Play a game of Mancala on the command line.
# Author: Shirley Wang
# 
# CSC 384 Fall 2023 Assignment 2
# version 1.0
###############################################################################

import argparse
import sys
import random
from datetime import datetime

from mancala_game import MancalaGameManager, AiPlayerInterface, Player
from utils import *

class MancalaCommandLine(object):

    def __init__(self, game_manager, player1, player2):

        self.game = game_manager
        self.players = [player1, player2]
        self.height = 2  #2 sides to the board
        self.width = self.game.dimension #pit count
        
        self.offset = 3
        self.cell_size = 100
        self.stone_size = 10

    def user_input_move(self):
        player = "Bottom Player" if self.game.current_player == BOTTOM else "Top Player"
        prompt = "\n" + player + " Please Input Move:"
        text = input(prompt).strip()

        if text[0] == "T" or text[0] == "B":
            # in case students also input the row in the move
            text = text[1:]

        if text == "S":
            # TA hack for easily generating board states
            print("Saved current board state to curr_board.txt")
            self.save_board("curr_board.txt")
            raise InvalidMoveError

        try:
            move_num = int(text)
        except:
            raise InvalidMoveError
        
        if self.game.current_player == TOP:
            move_num = move_num - 1
        else:
            move_num = self.game.dimension - move_num

        if move_num not in self.game.get_possible_moves():
            raise InvalidMoveError

        self.game.play(move_num)
        return
    
    def ai_move(self):
        player_obj = self.players[self.game.current_player]
        move = player_obj.get_move(self.game)
        player = "Bottom Player" if self.game.current_player == BOTTOM else "Top Player"
        player_name = "{} {}".format(player_obj.name, player)

        if move != "None\n": 
            move_view = int(move)
            if self.game.current_player == BOTTOM:
                move_view = self.game.board.dimension - move_view
            else:
                move_view = move_view + 1

            print("{}: {}{} ({})".format(player_name, player[0], move_view, move.strip()))
            print("")
            self.game.play(move)
        else:
            print("Uh this shouldn't happen?")
            raise InvalidMoveError
        return

    def run(self):
        self.game.draw_board()

        # while game is not over
        while len(self.game.get_possible_moves(opponent=True)) > 0 and len(self.game.get_possible_moves()) > 0:
            # run game
            player = "Bottom Player" if self.game.current_player == BOTTOM else "Top Player"
            print("")
            print("Turn:", player)

            success = True
            if isinstance(self.players[self.game.current_player], AiPlayerInterface):
                # call AI to make a move, if timeout then end game
                try:
                    self.ai_move()
                except AiTimeoutError:
                    print("\{} lost due to timeout".format(self.players[self.game.current_player].name))
                    success = False
                    break
            else:
                # get move from user, continue if incorrect
                try:
                    self.user_input_move()
                except InvalidMoveError:
                    print("Invalid Move")
                    success = False

            if success:
                self.game.draw_board()

        print("Ending Game")
        self.game.end_game()
        self.game.draw_board()

        if isinstance(self.players[TOP], AiPlayerInterface): 
            self.players[TOP].kill(self.game)
        if isinstance(self.players[BOTTOM], AiPlayerInterface): 
            self.players[BOTTOM].kill(self.game)

        winner = "Bottom Player" if self.game.get_winner() == BOTTOM else "Top Player"   
        print("GAME OVER: winner is {}".format(winner))
        return
    
    def save_board(self, filename):
        data = [
            ", ".join([str(x) for x in self.game.board.pockets[0]]) + "\n",
            ", ".join([str(x) for x in self.game.board.pockets[1]]) + "\n",
            str(self.game.board.mancalas[0]) + "\n",
            str(self.game.board.mancalas[1]) + "\n"
        ]

        with open(filename, "w") as f:
            f.writelines(data)


def parse_args():
    parser = argparse.ArgumentParser(
        prog="MancalaGUI",
        description="Run this code to start a game of Mancala"
    )

    parser.add_argument("-d", "--dimension", type=int,
                        help="Dimension of mancala board.")
    parser.add_argument("-i", "--initialBoard", 
                        help="File storing the initial state of the board. Overwrites dimension.")

    parser.add_argument("-t", "--agentTop", 
                        help="Agent for the top player, by filename (including extension). If not specified, user inputs moves.")
    parser.add_argument("-b", "--agentBottom", 
                        help="Agent for the bottom player, by filename (including extension). If not specified, user inputs moves.")

    parser.add_argument("-ht", "--heuristicTop", type=int, default=0,
                        help="Heuristic for top player to use. 0 = heuristic_basic, 1 = heuristic_advanced")
    parser.add_argument("-hb", "--heuristicBottom", type=int, default=0,
                        help="Heuristic for bottom player to use. 0 = heuristic_basic, 1 = heuristic_advanced")

    parser.add_argument("-l", "--limit", type=int, default=-1,
                        help="(Optional) Depth limit for agent to use.")
    
    parser.add_argument("-c", "--caching", action="store_true",
                        help="Use flag if agent should use caching.")


    args = parser.parse_args()    
    return args

def main():
    random.seed(datetime.now().timestamp())
    args = parse_args()

    if args.dimension is not None and args.dimension <= 0 and args.initialBoard is None: #if no dimension provided
        print('Please provide a valid board size (at least 1).')
        sys.exit(2)
    
    assert args.heuristicTop in [0, 1]
    assert args.heuristicBottom in [0, 1]

    if args.agentTop != None:
        p1 = AiPlayerInterface(args.agentTop, TOP, args.limit, 1 if args.caching else 0, args.heuristicTop)
    else:
        p1 = Player(TOP)

    if args.agentBottom != None:
        p2 = AiPlayerInterface(args.agentBottom, BOTTOM, args.limit, 1 if args.caching else 0, args.heuristicBottom)
    else:
        p2 = Player(BOTTOM)
        
    game = MancalaGameManager(args.dimension, args.initialBoard)
    gui = MancalaCommandLine(game, p1, p2) 
    gui.run()

if __name__ == "__main__":
    main()
