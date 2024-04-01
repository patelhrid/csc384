###############################################################################
# This module contains a simple graphical user interface for Mancala. 
#
# Thanks to Daniel Bauer, Columbia University, for a version of Othello that this was based on.
# Modified by Shirley Wang.
#
# CSC 384 Fall 2023 Assignment 2
# version 1.0
###############################################################################

import argparse
import sys
import random
from datetime import datetime

from tkinter import *
from tkinter import scrolledtext

from mancala_game import MancalaGameManager, AiPlayerInterface, Player
from utils import *

class MancalaGui(object):

    def __init__(self, game_manager, player1, player2):

        self.game = game_manager
        self.players = [player1, player2]
        self.height = 2  #2 sides to the board
        self.width = self.game.dimension #pit count
        
        self.offset = 3
        self.cell_size = 100
        self.stone_size = 10

        root = Tk()
        root.wm_title("Mancala")
        root.lift()
        root.attributes("-topmost", True)
        self.root = root
        self.canvas = Canvas(root,height = self.cell_size * (self.height+1) + self.offset,width = self.cell_size * (self.width+2) + self.offset)
        self.move_label = Label(root)
        self.score_label = Label(root)
        self.text = scrolledtext.ScrolledText(root, width=70, height=10)
        self.move_label.pack(side="top")
        self.score_label.pack(side="top")
        self.canvas.pack()
        self.text.pack()
        self.draw_board()

    def get_position(self, x, y):
        i = (x -self.offset) // self.cell_size
        j = (y -self.offset) // self.cell_size
        return i,j

    def mouse_pressed(self, event):
        # get the human move
        i, j = self.get_position(event.x, event.y)
        try:
            player = "Bottom Player" if self.game.current_player == BOTTOM else "Top Player"
            self.log("{}: {},{}".format(player, i-1, j))
            
            if j != self.game.current_player:
                raise InvalidMoveError("Invalid move: Not the current player.")

            self.game.play(i-1)
            self.draw_board()

            possible_moves = self.game.get_possible_moves()
            #a = get_possible_moves(self.game.board, self.game.current_player)

            if not possible_moves:
                self.game.end_game()
                self.draw_board()

                winner = "Bottom Player" if (self.game.get_winner() == BOTTOM) else "Top Player" 
                print('{} {} {}\n'.format(winner, self.game.board.mancalas[TOP], self.game.board.mancalas[BOTTOM]))
                
                ## changes over
                self.log("GAME OVER: winner is {}".format(winner))
                self.shutdown("Game Over")

            elif isinstance(self.players[self.game.current_player], AiPlayerInterface):
                self.root.unbind("<Button-1>")
                self.root.after(100, lambda: self.ai_move())
        
        except InvalidMoveError:
            self.log("Invalid move. {},{}".format(i,j))

    def shutdown(self, text):
        self.move_label["text"] = text 
        self.root.unbind("<Button-1>")
        if isinstance(self.players[TOP], AiPlayerInterface): 
            self.players[TOP].kill(self.game)
        if isinstance(self.players[BOTTOM], AiPlayerInterface): 
            self.players[BOTTOM].kill(self.game)

        # NOTE: starter code doesn't have self.root.destroy()
        #self.root.destroy()
 
    def ai_move(self):
        player_obj = self.players[self.game.current_player]
        try:
            #get the AI move
            flag = False
            move = player_obj.get_move(self.game)
            player = "Bottom Player" if self.game.current_player == BOTTOM else "Top Player"
            player = "{} {}".format(player_obj.name, player)
            if move != "None\n": #
                flag = True
                self.log("{}: {}".format(player, move))
                self.game.play(move)
            
            self.draw_board()
            
            if not flag or not self.game.get_possible_moves():
                self.game.end_game()
                self.draw_board()

                winner = "Bottom Player" if (self.game.get_winner() == BOTTOM) else "Top Player" 
                print('{} {} {}\n'.format(winner, self.game.board.mancalas[TOP], self.game.board.mancalas[BOTTOM]))     

                self.log("GAME OVER: winner is {}".format(winner))
                self.shutdown("Game Over")
            elif isinstance(self.players[self.game.current_player], AiPlayerInterface):
                self.root.after(1, lambda: self.ai_move())
            else: 
                self.root.bind("<Button-1>",lambda e: self.mouse_pressed(e))        
        except AiTimeoutError:
            self.shutdown("Game Over, {} lost (timeout)".format(player_obj.name))

    def run(self):
        if isinstance(self.players[TOP], AiPlayerInterface):
            self.root.after(10, lambda: self.ai_move())
        else: 
            self.root.bind("<Button-1>",lambda e: self.mouse_pressed(e))        
        self.draw_board()
        self.canvas.mainloop()

    def draw_board(self):
        self.draw_pits()
        self.draw_stones()
        player = "Bottom Player" if self.game.current_player == BOTTOM else "Top Player"
        self.move_label["text"]= player
        self.score_label["text"]= "Top Player {} : {} Bottom Player".format(*self.game.board.mancalas) 
   
    def log(self, msg, newline = True): 
        self.text.insert("end","{}{}".format(msg, "\n" if newline else ""))
        self.text.see("end")
 
    def draw_pits(self):
        colors = ("light green", "light blue") if self.game.current_player == BOTTOM else ("light blue", "light green")
        for i in range(1, self.width+1):
            self.canvas.create_oval(i*self.cell_size + self.offset, self.offset, 
                                    (i+1)*self.cell_size + self.offset, self.cell_size + self.offset, 
                                    fill=colors[0])
            self.canvas.create_oval(i*self.cell_size + self.offset, self.cell_size + self.offset, 
                                    (i+1)*self.cell_size + self.offset, 2*self.cell_size + self.offset, 
                                    fill=colors[1])
        
        #pits for players
        self.canvas.create_oval(self.offset, self.offset, 
                                self.cell_size + self.offset, 2*self.cell_size + self.offset, 
                                fill="white")
        self.canvas.create_oval((self.width+1)*self.cell_size + self.offset, self.offset, 
                                (self.width+2)*self.cell_size + self.offset, 2*self.cell_size + self.offset, 
                                fill="white")

    def draw_stone(self, i, j):
        x = (i + 0.5) * self.cell_size - self.stone_size/2 + random.randint(0,20) - 10
        y = (j + 0.5) * self.cell_size - self.stone_size/2 + random.randint(0,20) - 10
        
        self.canvas.create_oval(x, y, x+self.stone_size, y+self.stone_size, fill="green")
        
    def draw_stones(self):       
        for i in range(2):
            for j in range(1, len(self.game.board.pockets[i])+1):
                x = (j + 0.5) * self.cell_size + self.offset
                y = (i+1)*self.cell_size - 2*self.offset
                for k in range(self.game.board.pockets[i][j-1]):
                    self.draw_stone(j, i)
                self.canvas.create_text(x, y, font="Arial", text=str(self.game.board.pockets[i][j-1]))

        #draw disks on the top
        for i in range(self.game.board.mancalas[TOP]):
            x = self.cell_size/2 + random.randint(0,20) - 10
            y = self.cell_size + random.randint(0,20) - 10
            self.canvas.create_oval(x, y, x + self.stone_size, y + self.stone_size, fill="blue")
        x = self.cell_size/2
        y = 2*self.cell_size - 2*self.offset
        self.canvas.create_text(x, y,font="Arial", text=str(self.game.board.mancalas[TOP]))

        #draw disks on the bottom
        for i in range(self.game.board.mancalas[BOTTOM]):
            x = (self.width+1.5)*self.cell_size + random.randint(0,20) - 10
            y = self.cell_size + random.randint(0,20) - 10 
            self.canvas.create_oval(x, y, x + self.stone_size, y + self.stone_size, fill="red")
        x = (self.width+1.5)*self.cell_size
        y = 2*self.cell_size - 2*self.offset
        self.canvas.create_text(x, y,font="Arial", text=str(self.game.board.mancalas[BOTTOM]))

def parse_args():
    parser = argparse.ArgumentParser(
        prog="MancalaGUI",
        description="Run this code to start a game of mancala"
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
    gui = MancalaGui(game, p1, p2) 
    gui.run()

if __name__ == "__main__":
    main()
