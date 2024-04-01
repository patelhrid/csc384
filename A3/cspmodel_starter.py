############################################################
## CSC 384, Intro to AI, University of Toronto.
## Assignment 3 Starter Code
## v1.1
## Changes:
##   v1.1: updated the comments in kropki_model. 
##         the second return value should be a 2d list of variables.
############################################################

from board import *
from cspbase import *

def kropki_model(board):
    """
    Create a CSP for a Kropki Sudoku Puzzle given a board of dimension.

    If a variable has an initial value, its domain should only contain the initial value.
    Otherwise, the variable's domain should contain all possible values (1 to dimension).

    We will encode all the constraints as binary constraints.
    Each constraint is represented by a list of tuples, representing the values that
    satisfy this constraint. (This is the table representation taught in lecture.)

    Remember that a Kropki sudoku has the following constraints.
    - Row constraint: every two cells in a row must have different values.
    - Column constraint: every two cells in a column must have different values.
    - Cage constraint: every two cells in a 2x3 cage (for 6x6 puzzle) 
            or 3x3 cage (for 9x9 puzzle) must have different values.
    - Black dot constraints: one value is twice the other value.
    - White dot constraints: the two values are consecutive (differ by 1).

    Make sure that you return a 2D list of variables separately. 
    Once the CSP is solved, we will use this list of variables to populate the solved board.
    Take a look at csprun.py for the expected format of this 2D list.

    :returns: A CSP object and a list of variables.
    :rtype: CSP, List[List[Variable]]

    """

    raise NotImplementedError
    
    
    
def create_initial_domain(dim):
    """
    Return a list of values for the initial domain of any unassigned variable.
    [1, 2, ..., dimension]

    :param dim: board dimension
    :type dim: int

    :returns: A list of values for the initial domain of any unassigned variable.
    :rtype: List[int]
    """

    raise NotImplementedError



def create_variables(dim):
    """
    Return a list of variables for the board.

    We recommend that your name each variable Var(row, col).

    :param dim: Size of the board
    :type dim: int

    :returns: A list of variables, one for each cell on the board
    :rtype: List[Variables]
    """

    raise NotImplementedError

    
def satisfying_tuples_difference_constraints(dim):
    """
    Return a list of satifying tuples for binary difference constraints.

    :param dim: Size of the board
    :type dim: int

    :returns: A list of satifying tuples
    :rtype: List[(int,int)]
    """

    raise NotImplementedError
  
  
def satisfying_tuples_white_dots(dim):
    """
    Return a list of satifying tuples for white dot constraints.

    :param dim: Size of the board
    :type dim: int

    :returns: A list of satifying tuples
    :rtype: List[(int,int)]
    """

    raise NotImplementedError
  
def satisfying_tuples_black_dots(dim):
    """
    Return a list of satifying tuples for black dot constraints.

    :param dim: Size of the board
    :type dim: int

    :returns: A list of satifying tuples
    :rtype: List[(int,int)]
    """

    raise NotImplementedError
    
def create_row_and_col_constraints(dim, sat_tuples, variables):
    """
    Create and return a list of binary all-different row/column constraints.

    :param dim: Size of the board
    :type dim: int

    :param sat_tuples: A list of domain value pairs (value1, value2) such that 
        the two values in each tuple are different.
    :type sat_tuples: List[(int, int)]

    :param variables: A list of all the variables in the CSP
    :type variables: List[Variable]
        
    :returns: A list of binary all-different constraints
    :rtype: List[Constraint]
    """
    
    raise NotImplementedError
    
    
def create_cage_constraints(dim, sat_tuples, variables):
    """
    Create and return a list of binary all-different constraints for all cages.

    :param dim: Size of the board
    :type dim: int

    :param sat_tuples: A list of domain value pairs (value1, value2) such that 
        the two values in each tuple are different.
    :type sat_tuples: List[(int, int)]

    :param variables: A list of all the variables in the CSP
    :type variables: List[Variable]
        
    :returns: A list of binary all-different constraints
    :rtype: List[Constraint]
    """

    raise NotImplementedError
    
def create_dot_constraints(dim, dots, white_tuples, black_tuples, variables):
    """
    Create and return a list of binary constraints, one for each dot.

    :param dim: Size of the board
    :type dim: int
    
    :param dots: A list of dots, each dot is a Dot object.
    :type dots: List[Dot]

    :param white_tuples: A list of domain value pairs (value1, value2) such that 
        the two values in each tuple satisfy the white dot constraint.
    :type white_tuples: List[(int, int)]
    
    :param black_tuples: A list of domain value pairs (value1, value2) such that 
        the two values in each tuple satisfy the black dot constraint.
    :type black_tuples: List[(int, int)]

    :param variables: A list of all the variables in the CSP
    :type variables: List[Variable]
        
    :returns: A list of binary dot constraints
    :rtype: List[Constraint]
    """

    raise NotImplementedError

