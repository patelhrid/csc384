############################################################
## CSC 384, Intro to AI, University of Toronto.
## Assignment 1 Starter Code
## v1.1
##
## Changes: 
## v1.1: Changed comments in Board class to create boxes, storage, and obstacles as lists. 
##       Removed the commented out __eq__ function in State class.
############################################################


from typing import List

# Define characters for the elements in the puzzle
CHAR_WALL = '#'
CHAR_STORAGE = '.'
CHAR_BOX = '?'
CHAR_BOX_IN_STORAGE = '*' # a box is at a storage point.
CHAR_ROBOT = 'a'
CHAR_ROBOT_IN_STORAGE = 'A' # a robot is at a storage point.

class Board:
    """
    Represents the puzzle board.
    """

    def __init__(self, name: str, width: int, height: int, robots: object, boxes: object, storage: object,
                 obstacles: object) -> object:
        """
        Creates a Sokoban board.

        :param name: the name of the Sokoban board
        :type name: str
        :param width: the width of the Sokoban board
        :type width: int
        :param height: the height of the Sokoban board
        :type height: int
        :param robots: positions for each robot that is on the board. Each robot position is a tuple (x, y), 
                       that denotes the robot’s x and y position.
        :type robots: List[tuple]
        
        :param boxes: positions for each box in a list. Each position is an (x, y) tuple.
        :type boxes: List[tuple]
        
        :param storage: positions for all the storage points in a list.
        :type storage: List[tuple]

        :param obstacles: locations of all of the obstacles (i.e. walls) in a list.
        :type obstacles: List[tuple]
        :rtype: Board
        """
        self.name = name
        self.width = width
        self.height = height
        self.boxes = boxes
        self.robots = robots
        self.storage = storage
        self.obstacles = obstacles

    def __hash__(self):
        '''
        Return a data item that can be used as a dictionary key to UNIQUELY represent a board.
        '''
        return hash(self.__str__())

    def display(self):
        print(self.__str__())

    def __str__(self):
        '''
        Returns a string representation of a state that can be printed to stdout.
        '''
        map = []
        for y in range(0, self.height):
            row = []
            for x in range(0, self.width):
                row += [' ']
            map += [row]

        # storage points are represented by dots.
        for storage_point in self.storage:
            map[storage_point[1]][storage_point[0]] = CHAR_STORAGE

        # walls are represented by #.
        for obstacle in self.obstacles:
            map[obstacle[1]][obstacle[0]] = CHAR_WALL

        # robots are represented by A
        for i, robot in enumerate(self.robots):
            if robot in self.storage:
                map[robot[1]][robot[0]] = chr(ord(CHAR_ROBOT_IN_STORAGE) + i)
            else:
                map[robot[1]][robot[0]] = chr(ord(CHAR_ROBOT) + i)

        # boxes are represented by ? or * if they are at storage points.
        for box in self.boxes:
            if box in self.storage:
                map[box[1]][box[0]] = CHAR_BOX_IN_STORAGE
            else:
                map[box[1]][box[0]] = CHAR_BOX

        s = ''
        for row in map:
            for char in row:
                s += char
            s += '\n'

        return s

    # customized eq for object comparison.
    def __eq__(self, other):
        if isinstance(other, Board):
            return self.__str__() == other.__str__()
        return False


class State:
    """
    State class wrapping a Board with some extra current state information.
    Note that State and Board are different. Board has the locations of the cars.
    State has a Board and some extra information that is relevant to the search: 
    heuristic function, f value, current depth and parent.
    """

    def __init__(self, board: Board, hfn, f: int, depth: int, parent=None):
        """
        :param board: The board of the state.
        :type board: Board
        :param parent: The parent of current state.
        :type parent: Optional[State]
        :param hfn: The heuristic function.
        :type hfn: Optional[Heuristic] (a Heuristic is a function that consumes a Board and 
                   produces a numeric heuristic value)
        :param f: The f value of current state.
        :type f: int
        :param depth: The depth of current state in the search tree. Depth of the root node is 0.
        :type depth: int
        """
        self.board = board
        self.parent = parent
        self.hfn = hfn
        self.f = f
        self.depth = depth

        self.id = hash(board)  # The id for breaking ties.

    # customized lt for object comparison.
    def __lt__(self, other):
        return self.f < other.f

    def __str__(self):
        return str(self.board)


def heuristic_zero(board: Board):
    """
    Simply return zero for any state.
    """

    return 0


def read_from_file(filename: str) -> Board:
    """
    Reads in the puzzle in the given file 
    and returns a Board
    
    :param filename: The name of the given file.
    :type filename: str
    :return: the loaded Board
    :rtype: Board
    """

    puzzle_file = open(filename, "r")
    counter = 0
    width = -1
    height = -1
    name = ""

    row = 0
    board = Board("", -1, -1, [], [], [], [])

    for line in puzzle_file:

        if counter == 0: # first line has name of puzzle
            board.name = line.strip()
        elif counter == 1: # second line has width
            board.width = int(line)
        elif counter == 2: # third line has height
            board.height = int(line)
        else: # the following lines describe cars
            for col in range(len(line)):
                char = line[col]
                if char == CHAR_WALL:
                    board.obstacles.append((col, row))
                elif char == CHAR_BOX_IN_STORAGE:
                    board.boxes.append((col, row))
                    board.storage.append((col, row))
                elif char == CHAR_BOX:
                    board.boxes.append((col, row))
                elif char == CHAR_STORAGE:
                    board.storage.append((col, row))
                elif char.isalpha() and char.isupper():
                    board.robots.append((col, row))
                    board.storage.append((col, row))
                elif char.isalpha() and char.islower():
                    board.robots.append((col, row))
            row += 1

        counter += 1

    puzzle_file.close()
    return board