############################################################
## CSC 384, Intro to AI, University of Toronto.
## Assignment 1 Starter Code
## v1.1
##
## Changes:
## v1.1: removed the hfn paramete from dfs. Updated solve_puzzle() accordingly.
############################################################

from typing import List
import heapq
from heapq import heappush, heappop
import time
import argparse
import math # for infinity

from board import *

def is_goal(state):
    """
    Returns True if the state is the goal state and False otherwise.

    :param state: the current state.
    :type state: State
    :return: True or False
    :rtype: bool
    """

    # Check if all boxes are on storage points
    return all(box in state.board.storage for box in state.board.boxes)


def get_path(state):
    """
    Return a list of states containing the nodes on the path
    from the initial state to the given state in order.

    :param state: The current state.
    :type state: State
    :return: The path.
    :rtype: List[State]
    """

    curr = state
    retrieved_path = [curr]
    while curr.parent:
        retrieved_path.append(curr.parent)
        curr = curr.parent
    return list(reversed(retrieved_path))


def get_successors(state: State) -> List[State]:
    """
    Return a list containing the successor states of the given state.
    The states in the list may be in any arbitrary order.

    :param state: The current state.
    :type state: State
    :return: The list of successor states.
    :rtype: List[State]
    """

    # Get the current board and its attributes
    current_board = state.board
    width = current_board.width
    height = current_board.height
    robots = current_board.robots
    boxes = current_board.boxes
    obstacles = current_board.obstacles

    successors = []

    for robot in robots:
        x, y = robot

        # Try moving the robot in four directions: up, down, left, right
        moves = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]

        for move_x, move_y in moves:
            # Check if the move is within the bounds of the board
            if 0 <= move_x < width and 0 <= move_y < height:
                # Check if the move is not into an obstacle
                if (move_x, move_y) not in obstacles:
                    # Check if the move results in pushing a box
                    if (move_x, move_y) in boxes:
                        # Calculate the new box position after the push
                        new_box_x, new_box_y = move_x + (move_x - x), move_y + (move_y - y)
                        # Check if the new box position is within bounds and not an obstacle
                        if (0 <= new_box_x < width and 0 <= new_box_y < height and
                                (new_box_x, new_box_y) not in obstacles and
                                (new_box_x, new_box_y) not in boxes):
                            # Create a new board with the updated positions
                            new_robots = [(move_x, move_y) if (rx, ry) == robot else (rx, ry) for rx, ry in robots]
                            new_boxes = [(new_box_x, new_box_y) if (bx, by) == (move_x, move_y) else (bx, by) for bx, by in boxes]
                            if callable(state.hfn):
                                # Use state.hfn as the heuristic function
                                new_state = State(Board(current_board.name, width, height, new_robots, new_boxes,
                                                        current_board.storage, obstacles), state.hfn, state.f,
                                                  state.depth + 1, state)
                            else:
                                # Use 0 as the heuristic value when state.hfn is not callable
                                new_state = State(Board(current_board.name, width, height, new_robots, new_boxes,
                                                        current_board.storage, obstacles), 0, state.f, state.depth + 1,
                                                  state)

                            successors.append(new_state)
                    else:
                        # Create a new board with the updated robot position
                        new_robots = [(move_x, move_y) if (rx, ry) == robot else (rx, ry) for rx, ry in robots]
                        new_state = State(Board(current_board.name, width, height, new_robots, boxes, current_board.storage, obstacles), state.hfn, state.f, state.depth + 1, state)
                        successors.append(new_state)

    return successors


def dfs(init_board):
    """
    Run the DFS algorithm given an initial board.

    If the function finds a goal state, it returns a list of states representing
    the path from the initial state to the goal state in order and the cost of
    the solution found.
    Otherwise, it returns am empty list and -1.

    :param init_board: The initial board.
    :type init_board: Board
    :return: (the path to goal state, solution cost)
    :rtype: List[State], int
    """

    stack = []
    curr = State(init_board, hfn=heuristic_zero, f=0, depth=0, parent=None)
    stack.append(curr)
    visited = set()

    while stack:
        st = stack.pop()
        if is_goal(st):
            return get_path(st), st.depth
        else:
            visited.add(st.board)
            for successor in get_successors(st):
                if successor.board not in visited:
                    stack.append(successor)

    return [], -1


def a_star(init_board, hfn) -> (List[State], int):
    """
    Run the A_star search algorithm given an initial board and a heuristic function.

    If the function finds a goal state, it returns a list of states representing
    the path from the initial state to the goal state in order and the cost of
    the solution found.
    Otherwise, it returns am empty list and -1.

    :param init_board: The initial starting board.
    :type init_board: Board
    :param hfn: The heuristic function.
    :type hfn: Heuristic (a function that consumes a Board and produces a numeric heuristic value)
    :return: (the path to goal state, solution cost)
    :rtype: List[State], int
    """

    # Priority queue (heap) to store states based on their f values
    open_set = []
    heapq.heappush(open_set, (0, State(init_board, hfn, 0, 0, None)))
    visited = set()

    while open_set:
        # Pop the state with the lowest f value
        _, st = heapq.heappop(open_set)

        if is_goal(st):
            return get_path(st), st.depth
        else:
            visited.add(st.board)
            for successor in get_successors(st):
                if successor.board not in visited:
                    # Calculate f = g + h
                    g = st.depth + 1
                    h = successor.hfn(successor.board)
                    f = g + h
                    heapq.heappush(open_set, (f, successor))

    return [], -1


def heuristic_basic(board):
    """
    Returns the heuristic value for the given board
    based on the Manhattan Distance Heuristic function.

    Returns the sum of the Manhattan distances between each box
    and its closest storage point.

    :param board: The current board.
    :type board: Board
    :return: The heuristic value.
    :rtype: int
    """

    total_distance = 0

    for box in board.boxes:
        min_distance = float('inf')  # Initialize to positive infinity
        for storage in board.storage:
            # Calculate Manhattan distance between box and storage
            distance = abs(box[0] - storage[0]) + abs(box[1] - storage[1])
            if distance < min_distance:
                min_distance = distance
        total_distance += min_distance

    return total_distance


def heuristic_advanced(board):
    """
    An advanced heuristic of your own choosing and invention.

    :param board: The current board.
    :type board: Board
    :return: The heuristic value.
    :rtype: int
    """

    # Missing tiles heuristic
    missing_tiles = 0

    for box in board.boxes:
        if box not in board.storage:
            missing_tiles += 1

    return missing_tiles


def solve_puzzle(board: Board, algorithm: str, hfn):
    """
    Solve the given puzzle using the given type of algorithm.

    :param algorithm: the search algorithm
    :type algorithm: str
    :param hfn: The heuristic function
    :type hfn: Optional[Heuristic]

    :return: the path from the initial state to the goal state
    :rtype: List[State]
    """

    print("Initial board")
    board.display()

    time_start = time.time()

    if algorithm == 'a_star':
        print("Executing A* search")
        path, step = a_star(board, hfn)
    elif algorithm == 'dfs':
        print("Executing DFS")
        path, step = dfs(board)
    else:
        raise NotImplementedError

    time_end = time.time()
    time_elapsed = time_end - time_start

    if not path:

        print('No solution for this puzzle')
        return []

    else:

        print('Goal state found: ')
        path[-1].board.display()

        print('Solution is: ')

        counter = 0
        while counter < len(path):
            print(counter + 1)
            path[counter].board.display()
            print()
            counter += 1

        print('Solution cost: {}'.format(step))
        print('Time taken: {:.2f}s'.format(time_elapsed))

        return path


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inputfile",
        type=str,
        required=True,
        help="The file that contains the puzzle."
    )
    parser.add_argument(
        "--outputfile",
        type=str,
        required=True,
        help="The file that contains the solution to the puzzle."
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        required=True,
        choices=['a_star', 'dfs'],
        help="The searching algorithm."
    )
    parser.add_argument(
        "--heuristic",
        type=str,
        required=False,
        default=None,
        choices=['zero', 'basic', 'advanced'],
        help="The heuristic used for any heuristic search."
    )
    args = parser.parse_args()

    # set the heuristic function
    heuristic = heuristic_zero
    if args.heuristic == 'basic':
        heuristic = heuristic_basic
    elif args.heuristic == 'advanced':
        heuristic = heuristic_advanced

    # read the boards from the file
    board = read_from_file(args.inputfile)

    # solve the puzzles
    path = solve_puzzle(board, args.algorithm, heuristic)

    # save solution in output file
    outputfile = open(args.outputfile, "w")
    counter = 1
    for state in path:
        print(counter, file=outputfile)
        print(state.board, file=outputfile)
        counter += 1
    outputfile.close()