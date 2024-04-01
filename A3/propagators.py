############################################################
## CSC 384, Intro to AI, University of Toronto.
## Assignment 3 Starter Code
## v1.0
##
############################################################


def prop_FC(csp, last_assigned_var=None):
    """
    This is a propagator to perform forward checking. 

    First, collect all the relevant constraints.
    If the last assigned variable is None, then no variable has been assigned 
    and we are performing propagation before search starts.
    In this case, we will check all the constraints.
    Otherwise, we will only check constraints involving the last assigned variable.

    Among all the relevant constraints, focus on the constraints with one unassigned variable. 
    Consider every value in the unassigned variable's domain, if the value violates 
    any constraint, prune the value. 

    :param csp: The CSP problem
    :type csp: CSP

    :param last_assigned_var: The last variable assigned before propagation.
        None if no variable has been assigned yet (that is, we are performing
        propagation before search starts).
    :type last_assigned_var: Variable

    :returns: The boolean indicates whether forward checking is successful.
        The boolean is False if at least one domain becomes empty after forward checking.
        The boolean is True otherwise.
        Also returns a list of variable and value pairs pruned.
    :rtype: boolean, List[(Variable, Value)]
    """

    if not last_assigned_var:
        return True, []

    pruned_values = []

    relevant_constraints = csp.get_cons_with_var(last_assigned_var)

    for constraint in relevant_constraints:
        scope = constraint.get_scope()

        for neighbor in scope:
            if neighbor == last_assigned_var or neighbor.is_assigned():
                continue

            for val in neighbor.cur_domain():
                consistent_assignment = constraint.check([val, last_assigned_var.get_assigned_value()])

                if not consistent_assignment:
                    neighbor.prune_value(val)
                    pruned_values.append((neighbor, val))

                    if not neighbor.cur_domain():
                        return False, pruned_values

    return True, pruned_values


def prop_AC3(csp, last_assigned_var=None):
    """
    This is a propagator to perform the AC-3 algorithm.

    Keep track of all the constraints in a queue (list).
    If the last_assigned_var is not None, then we only need to
    consider constraints that involve the last assigned variable.

    For each constraint, consider every variable in the constraint and
    every value in the variable's domain.
    For each variable and value pair, prune it if it is not part of
    a satisfying assignment for the constraint.
    Finally, if we have pruned any value for a variable,
    add other constraints involving the variable back into the queue.

    :param csp: The CSP problem
    :type csp: CSP

    :param last_assigned_var: The last variable assigned before propagation.
        None if no variable has been assigned yet (that is, we are performing
        propagation before search starts).
    :type last_assigned_var: Variable

    :returns: a boolean indicating if the current assignment satisfies
        all the constraints and a list of variable and value pairs pruned.
    :rtype: boolean, List[(Variable, Value)]
    """

    queue = []

    # Initialize the queue with all arcs
    for constraint in csp.get_all_cons():
        for arc in get_all_arcs(constraint.get_scope()):
            queue.append(arc)

    while queue:
        (var_i, var_j) = queue.pop(0)

        # Apply revise operation to prune values
        revised, pruned_values = revise(csp, var_i, var_j)

        if revised:
            if not var_i.cur_domain():
                return False, pruned_values

            # Add neighbors of var_i (excluding var_j) to the queue
            neighbors = [neighbor for neighbor in get_all_neighbors(var_i, csp) if neighbor != var_j]
            queue.extend([(neighbor, var_i) for neighbor in neighbors])

    return True, []


def revise(csp, var_i, var_j):
    revised = False
    pruned_values = []

    constraint_between_ij = get_constraint_between_vars(var_i, var_j, csp)

    for val_i in var_i.cur_domain():
        if not any(constraint_between_ij.check((val_i, val_j)) for val_j in var_j.cur_domain()):
            var_i.prune_value(val_i)
            pruned_values.append((var_i, val_i))
            revised = True

    return revised, pruned_values


def get_constraint_between_vars(var_i, var_j, csp):
    constraints = csp.get_cons_with_var(var_i)
    for constraint in constraints:
        if var_j in constraint.get_scope():
            return constraint

    return None


def get_all_arcs(scope):
    return [(var_i, var_j) for var_i in scope for var_j in scope if var_i != var_j]


def get_all_neighbors(var, csp):
    return [neighbor for neighbor in csp.get_all_vars() if neighbor != var and neighbor in var.cur_domain()]


def ord_mrv(csp):
    """
    Implement the Minimum Remaining Values (MRV) heuristic.
    Choose the next variable to assign based on MRV.

    If there is a tie, we will choose the first variable. 

    :param csp: A CSP problem
    :type csp: CSP

    :returns: the next variable to assign based on MRV

    """

    return min(csp.vars, key=lambda var: len(var.cur_domain()))


###############################################################################
# Do not modify the prop_BT function below
###############################################################################


def prop_BT(csp, last_assigned_var=None):
    """
    This is a basic propagator for plain backtracking search.

    Check if the current assignment satisfies all the constraints.
    Note that we only need to check all the fully instantiated constraints 
    that contain the last assigned variable.
    
    :param csp: The CSP problem
    :type csp: CSP

    :param last_assigned_var: The last variable assigned before propagation.
        None if no variable has been assigned yet (that is, we are performing 
        propagation before search starts).
    :type last_assigned_var: Variable

    :returns: a boolean indicating if the current assignment satisifes all the constraints 
        and a list of variable and value pairs pruned. 
    :rtype: boolean, List[(Variable, Value)]

    """
    
    # If we haven't assigned any variable yet, return true.
    if not last_assigned_var:
        return True, []
        
    # Check all the constraints that contain the last assigned variable.
    for c in csp.get_cons_with_var(last_assigned_var):

        # All the variables in the constraint have been assigned.
        if c.get_num_unassigned_vars() == 0:

            # get the variables
            vars = c.get_scope() 

            # get the list of values
            vals = []
            for var in vars: #
                vals.append(var.get_assigned_value())

            # check if the constraint is satisfied
            if not c.check(vals): 
                return False, []

    return True, []
