############################################################
## CSC 384, Intro to AI, University of Toronto.
## Assignment 4 Starter Code
## v1.2
## - removed the example in ve since it is misleading.
## - updated the docstring in min_fill_ordering. The tie-breaking rule should
##   choose the variable that comes first in the provided list of factors.
############################################################

from bnetbase import Variable, Factor, BN
import csv


def normalize(factor):
    '''
    Normalize the factor such that its values sum to 1.
    Do not modify the input factor.

    :param factor: a Factor object. 
    :return: a new Factor object resulting from normalizing factor.
    '''
    new_factor = Factor(factor.name, factor.get_scope())
    total_sum = sum(factor.values)

    for i in range(len(factor.values)):
        new_factor.values[i] = factor.values[i] / total_sum

    return new_factor


def restrict(factor, variable, value):
    '''
    Restrict a factor by assigning value to variable.
    Do not modify the input factor.

    :param factor: a Factor object.
    :param variable: the variable to restrict.
    :param value: the value to restrict the variable to
    :return: a new Factor object resulting from restricting variable to value.
             This new factor no longer has variable in it.
    '''

    def product(*iterables):
        pools = [tuple(iterable) for iterable in iterables]
        result = [[]]
        for pool in pools:
            result = [x + [y] for x in result for y in pool]
        return result

    new_scope = [v for v in factor.get_scope() if v != variable]
    new_factor = Factor(factor.name, new_scope)

    for assignment in product(*[v.domain() for v in new_factor.get_scope()]):
        for v, val in zip(new_factor.get_scope(), assignment):
            v.set_assignment(val)

        variable.set_assignment(value)
        new_factor.add_value_at_current_assignment(factor.get_value([v.get_assignment() for v in factor.get_scope()]))

    variable.set_assignment(variable.get_assignment())

    return new_factor

    return new_factor


def sum_out(factor, variable):
    '''
    Sum out a variable variable from factor factor.
    Do not modify the input factor.

    :param factor: a Factor object.
    :param variable: the variable to sum out.
    :return: a new Factor object resulting from summing out variable from the factor.
             This new factor no longer has variable in it.
    '''
    # Get the scope of the input factor
    scope = factor.get_scope()

    # Find the index of the variable to sum out in the factor's scope
    var_index = scope.index(variable)

    # Create a new scope without the variable
    new_scope = [v for v in scope if v != variable]

    # Calculate the size of the new factor's domain
    new_domain_size = 1
    for v in new_scope:
        new_domain_size *= v.domain_size()

    # Initialize new values with zeros
    new_values = [0.0] * new_domain_size

    # Iterate over the values of the new factor's domain
    for new_assignment in range(new_domain_size):
        # Map the new assignment to the original factor's assignments
        original_assignment = 0
        multiplier = 1
        for v in scope:
            if v != variable:
                original_assignment += (new_assignment // multiplier % v.domain_size()) * multiplier
                multiplier *= v.domain_size()

        # Sum out the variable by accumulating values for each assignment
        new_values[new_assignment] = sum(
            factor.values[original_assignment + i * multiplier] for i in range(variable.domain_size()))

    # Create a new factor with the updated values and reduced scope
    new_factor = Factor(factor.name, new_scope)
    new_factor.values = new_values

    return new_factor


def multiply(factor_list):
    '''
    Multiply a list of factors together.
    Do not modify any of the input factors.

    :param factor_list: a list of Factor objects.
    :return: a new Factor object resulting from multiplying all the factors in factor_list.
    '''

    if not factor_list:
        # If the input list is empty, return a factor with an empty scope and value 1
        return Factor("EmptyFactor", [])

        # Find the combined scope
    combined_scope = []
    for factor in factor_list:
        combined_scope.extend(factor.get_scope())

    # Remove duplicates from the combined scope
    combined_scope = list(set(combined_scope))

    # Initialize values based on the combined scope
    result_values = [1] * (2 ** len(combined_scope))  # Initialize with 1 for each possible assignment

    for factor in factor_list:
        factor_indices = [combined_scope.index(variable) for variable in factor.get_scope()]
        for assignment in range(2 ** len(factor.get_scope())):
            combined_assignment = 0
            for i, index in enumerate(factor_indices):
                combined_assignment |= (((assignment >> i) & 1) << index)
            result_values[combined_assignment] *= factor.values[assignment]

    # Create a new factor with the combined scope and values
    result_factor = Factor("ResultFactor", combined_scope)
    result_factor.values = result_values

    return result_factor


def min_fill_ordering(factor_list, variable_query):
    '''
    This function implements The Min Fill Heuristic. We will use this heuristic to determine the order 
    to eliminate the hidden variables. The Min Fill Heuristic says to eliminate next the variable that 
    creates the factor of the smallest size. If there is a tie, choose the variable that comes first 
    in the provided order of factors in factor_list.

    The returned list is determined iteratively.
    First, determine the size of the resulting factor when eliminating each variable from the factor_list.
    The size of the resulting factor is the number of variables in the factor.
    Then the first variable in the returned list should be the variable that results in the factor 
    of the smallest size. If there is a tie, choose the variable that comes first in the provided order of 
    factors in factor_list. 
    Then repeat the process above to determine the second, third, ... variable in the returned list.

    Here is an example.
    Consider our complete Holmes network. Suppose that we are given a list of factors for the variables 
    in this order: P(E), P(B), P(A|B, E), P(G|A), and P(W|A). Assume that our query variable is Earthquake. 
    Among the other variables, which one should we eliminate first based on the Min Fill Heuristic? 

    - Eliminating B creates a factor of 2 variables (A and E).
    - Eliminating A creates a factor of 4 variables (E, B, G and W).
    - Eliminating G creates a factor of 1 variable (A).
    - Eliminating W creates a factor of 1 variable (A).

    In this case, G and W tie for the best variable to be eliminated first since eliminating each variable 
    creates a factor of 1 variable only. Based on our tie-breaking rule, we should choose G since it comes 
    before W in the list of factors provided.
    '''

    elimination_order = []

    # Helper function to calculate the size of the resulting factor when eliminating a variable
    def calculate_factor_size(variable, remaining_factors):
        return len(set().union(*(set(f.get_scope()) for f in remaining_factors if isinstance(f, Factor))) - {variable})

    while factor_list:
        # Calculate the size of resulting factors for each candidate variable
        sizes = [(var, calculate_factor_size(var, factor_list)) for factor in factor_list for var in factor.get_scope()
                 if isinstance(var, Variable) and var != variable_query]

        # print("Sizes:", sizes)

        # Check if there are candidate variables
        if not sizes:
            break

        # Sort by factor size and choose the variable with the smallest size
        min_size_variable = min(sizes, key=lambda x: x[1])

        # print("Min Size Variable:", min_size_variable)

        # Add the chosen variable to the elimination order
        elimination_order.append(min_size_variable[0])

        # Remove factors involving the chosen variable
        factor_list = [factor for factor in factor_list if min_size_variable[0] not in factor.get_scope()]

        # print("Elimination Order:", elimination_order)

    return elimination_order if elimination_order else [variable_query]


def ve(bayes_net, var_query, varlist_evidence):
    '''
    Execute the variable elimination algorithm on the Bayesian network bayes_net
    to compute a distribution over the values of var_query given the 
    evidence provided by varlist_evidence. 

    :param bayes_net: a BN object.
    :param var_query: the query variable. we want to compute a distribution
                     over the values of the query variable.
    :param varlist_evidence: the evidence variables. Each evidence variable has 
                         its evidence set to a value from its domain 
                         using set_evidence.
    :return: a Factor object representing a distribution over the values
             of var_query. that is a list of numbers, one for every value
             in var_query's domain. These numbers sum to 1. The i-th number
             is the probability that var_query is equal to its i-th value given 
             the settings of the evidence variables.

    '''

    # Step 1: Initialization
    factors = bayes_net.factors()
    hidden_vars = [var for var in bayes_net.variables() if var != var_query and var not in varlist_evidence]

    # Step 2: Min Fill ordering
    min_fill_order = min_fill_ordering(factors, var_query)

    # Step 3: Elimination
    result_factor = None
    for var_elim in min_fill_order:
        relevant_factors = [factor for factor in factors if var_elim in factor.get_scope()]

        # Step 3a: Multiply relevant factors
        product_factor = multiply(relevant_factors)

        # Step 3b: Sum out the variable to be eliminated
        sum_out_factor = sum_out(product_factor, var_elim)

        # Update factors list
        factors = [factor for factor in factors if factor not in relevant_factors]
        factors.append(sum_out_factor)

    # Step 4: Normalization
    result_factor = normalize(factors[0])  # Assuming only one factor remains after elimination

    return result_factor


# The order of these domains is consistent with the order of the columns in the data set.
salary_variable_domains = {
    "Work": ['Not Working', 'Government', 'Private', 'Self-emp'],
    "Education": ['<Gr12', 'HS-Graduate', 'Associate', 'Professional', 'Bachelors', 'Masters', 'Doctorate'],
    "Occupation": ['Admin', 'Military', 'Manual Labour', 'Office Labour', 'Service', 'Professional'],
    "MaritalStatus": ['Not-Married', 'Married', 'Separated', 'Widowed'],
    "Relationship": ['Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried'],
    "Race": ['White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other'],
    "Gender": ['Male', 'Female'],
    "Country": ['North-America', 'South-America', 'Europe', 'Asia', 'Middle-East', 'Carribean'],
    "Salary": ['<50K', '>=50K']
}

salary_variable = Variable("Salary", ['<50K', '>=50K'])


def naive_bayes_model(data_file, variable_domains=salary_variable_domains, class_var=salary_variable):
    '''
    NaiveBayesModel returns a BN that is a Naive Bayes model that represents 
    the joint distribution of value assignments to variables in the given dataset.

    Remember a Naive Bayes model assumes P(X1, X2,.... XN, Class) can be represented as 
    P(X1|Class) * P(X2|Class) * .... * P(XN|Class) * P(Class).

    When you generated your Bayes Net, assume that the values in the SALARY column of 
    the dataset are the CLASS that we want to predict.

    Please name the factors as follows. If you don't follow these naming conventions, you will fail our tests.
    - The name of the Salary factor should be called "Salary" without the quotation marks.
    - The name of any other factor should be called "VariableName,Salary" without the quotation marks. 
      For example, the factor for Education should be called "Education,Salary".

    @return a BN that is a Naive Bayes model and which represents the given data set.
    '''
    def populate_data(factors_lst, data):
        values = []

        for factor in factors_lst:
            if factor.name == "Salary":
                for value in factor.scope[0].domain():
                    count_target = sum(1 for rows in data if value in rows)
                    count_total = len(data)
                    factor.scope[0].set_assignment(value)
                    values.append([value, count_target / count_total])
                factor.add_values(values)
                values = []
            else:
                for value in factor.scope[0].domain():
                    for value2 in factor.scope[1].domain():
                        count_target = sum(1 for rows in data if value in rows and value2 in rows)
                        count_total = sum(1 for rows in data if value2 in rows)
                        factor.scope[0].set_assignment(value)
                        factor.scope[1].set_assignment(value2)
                        values.append(
                            [value, value2, count_target / count_total] if count_total > 0 else [value, value2, 0])
                factor.add_values(values)
                values = []

    input_data = []

    # Read in the data
    with open(data_file, newline='') as csvfile:
        reader = csv.reader(csvfile)
        headers = next(reader, None)  # skip header row
        for row in reader:
            input_data.append(row)

    variables = []
    factors = []

    # Iterate over variable domains
    for variable_name in variable_domains:
        if variable_name != class_var.name:
            var = Variable(variable_name, variable_domains[variable_name])
            variables.append(var)
        else:
            variables.append(class_var)

    for variable in variables:
        if variable.name != "Salary":
            f = Factor(variable.name + ",Salary", [variable, class_var])
            factors.append(f)
        else:
            f = Factor("Salary", [variable])
            factors.append(f)

    populate_data(factors, input_data)

    model = BN("NaiveBayesNet", variables, factors)
    return model


def explore(bayes_net, question):
    '''    
    Return a probability given a Naive Bayes Model and a question number 1-6. 
    
    The questions are below: 
    1. What percentage of the women in the test data set does our model predict having a salary >= $50K? 
    2. What percentage of the men in the test data set does our model predict having a salary >= $50K? 
    3. What percentage of the women in the test data set satisfies the condition: P(S=">=$50K"|Evidence) is strictly greater than P(S=">=$50K"|Evidence,Gender)?
    4. What percentage of the men in the test data set satisfies the condition: P(S=">=$50K"|Evidence) is strictly greater than P(S=">=$50K"|Evidence,Gender)?
    5. What percentage of the women in the test data set with a predicted salary over $50K (P(Salary=">=$50K"|E) > 0.5) have an actual salary over $50K?
    6. What percentage of the men in the test data set with a predicted salary over $50K (P(Salary=">=$50K"|E) > 0.5) have an actual salary over $50K?

    @return a percentage (between 0 and 100)
    '''

    # Get the variables and factors from the Bayes Net
    variables = bayes_net.variables()
    factors = bayes_net.factors()

    # Helper function to get the factor for a given variable
    def get_factor_for_variable(variable_name):
        return next((factor for factor in factors if variable_name in factor.name), None)

    # Helper function to check if a person has a predicted salary over $50K
    def has_predicted_salary_over_50k(row):
        # Get the Salary variable from the Bayes Net
        sal_var = get_factor_for_variable('Salary')

        # Initialize with 1.0 for Laplace smoothing for all factors
        for factor in factors:
            factor.add_value_at_current_assignment(1.0)

        for i, variable in enumerate(variables[:-1]):
            factor_for_variable = get_factor_for_variable(variable.name)
            factor_for_variable.add_value_at_current_assignment(row[i])
            factor_for_variable.add_value_at_current_assignment(row[-1])  # Class variable (Salary)

        return float(sal_var.get_value_at_current_assignments() == '>=50K') > 0.5

    # Read in the data from the test data file
    test_data = []
    with open('adult-test.csv', newline='') as csvfile:
        reader = csv.reader(csvfile)
        headers = next(reader, None)  # skip header row
        for row in reader:
            test_data.append(row)

    # Calculate probabilities based on the given question
    if question == 1:
        # What percentage of the women in the test data set does our model predict having a salary >= $50K?
        women_data = [row for row in test_data if row[6] == 'Female']
        women_over_50k = [row for row in women_data if has_predicted_salary_over_50k(row)]
        percentage = (len(women_over_50k) / len(women_data)) * 100 if len(women_data) > 0 else 0
    elif question == 2:
        # What percentage of the men in the test data set does our model predict having a salary >= $50K?
        men_data = [row for row in test_data if row[6] == 'Male']
        men_over_50k = [row for row in men_data if has_predicted_salary_over_50k(row)]
        percentage = (len(men_over_50k) / len(men_data)) * 100 if len(men_data) > 0 else 0
    elif question == 3:
        # What percentage of the women in the test data set satisfies the condition:
        # P(S=">=$50K"|Evidence) is strictly greater than P(S=">=$50K"|Evidence,Gender)?
        women_data = [row for row in test_data if row[6] == 'Female']
        women_condition_satisfied = [
            row for row in women_data
            if has_predicted_salary_over_50k(row) and get_factor_for_variable(
                'Gender').get_value_at_current_assignments() < 0.5
        ]
        percentage = (len(women_condition_satisfied) / len(women_data)) * 100 if len(women_data) > 0 else 0
    elif question == 4:
        # What percentage of the men in the test data set satisfies the condition:
        # P(S=">=$50K"|Evidence) is strictly greater than P(S=">=$50K"|Evidence,Gender)?
        men_data = [row for row in test_data if row[6] == 'Male']
        men_condition_satisfied = [
            row for row in men_data
            if has_predicted_salary_over_50k(row) and get_factor_for_variable(
                'Gender').get_value_at_current_assignments() > 0.5
        ]
        percentage = (len(men_condition_satisfied) / len(men_data)) * 100 if len(men_data) > 0 else 0
    elif question == 5:
        # What percentage of the women in the test data set with a predicted salary over $50K
        # (P(Salary=">=$50K"|E) > 0.5) have an actual salary over $50K?
        women_over_50k_predicted = [row for row in test_data if has_predicted_salary_over_50k(row)]
        women_over_50k_actual = [row for row in women_over_50k_predicted if row[-1] == '>=50K']
        percentage = (len(women_over_50k_actual) / len(women_over_50k_predicted)) * 100 if len(
            women_over_50k_predicted) > 0 else 0
    elif question == 6:
        # What percentage of the men in the test data set with a predicted salary over $50K
        # (P(Salary=">=$50K"|E) > 0.5) have an actual salary over $50K?
        men_over_50k_predicted = [row for row in test_data if has_predicted_salary_over_50k(row)]
        men_over_50k_actual = [row for row in men_over_50k_predicted if row[-1] == '>=50K']
        percentage = (len(men_over_50k_actual) / len(men_over_50k_predicted)) * 100 if len(
            men_over_50k_predicted) > 0 else 0
    else:
        # Invalid question number
        return None

    return percentage


# if __name__ == "__main__":

    # E = Variable(name='E', domain=[True, False])
    # B = Variable(name='B', domain=[True, False])
    #
    # R = Variable(name='R', domain=[True, False])
    # A = Variable(name='A', domain=[True, False])
    #
    # W = Variable(name='W', domain=[True, False])
    # G = Variable(name='G', domain=[True, False])
    #
    # factor_E = Factor(name="factor_E", scope=[E])
    # factor_E.add_values([[True, 0.0003],
    #                      [False, 0.9997]])
    #
    # factor_B = Factor(name="factor_B", scope=[B])
    # factor_B.add_values([[True, 0.0001],
    #                      [False, 0.9999]])
    #
    # factor_R = Factor(name="factor_R", scope=[R, E])
    # factor_R.add_values([[True, True, 0.9],
    #                      [True, False, 0.0002],
    #                      [False, True, 0.1],
    #                      [False, False, 0.9998]])
    #
    # factor_A = Factor(name="factor_A", scope=[A, E, B])
    # factor_A.add_values([[True, True, True, 0.96],
    #                      [True, False, True, 0.2],
    #                      [True, True, False, 0.95],
    #                      [True, False, False, 0.01],
    #                      [False, True, True, 0.04],
    #                      [False, False, True, 0.8],
    #                      [False, True, False, 0.05],
    #                      [False, False, False, 0.99]])
    #
    # factor_W = Factor(name="factor_W", scope=[W, A])
    # factor_W.add_values([[True, True, 0.4],
    #                      [True, False, 0.04],
    #                      [False, True, 0.6],
    #                      [False, False, 0.96]])
    #
    # factor_G = Factor(name="factor_G", scope=[G, A])
    # factor_G.add_values([[True, True, 0.8],
    #                      [True, False, 0.4],
    #                      [False, True, 0.2],
    #                      [False, False, 0.6]])
    #
    # factors_list = [factor_E, factor_B, factor_R, factor_A, factor_W, factor_G]
    #
    # bn_model = BN(name="", Vars=[E, B, R, A, W, G], Factors=factors_list)
    #
    # query_var = E  # Replace with your actual query variable
    #
    # # Get the minimum fill ordering
    # min_fill_order = min_fill_ordering(factors_list, query_var)
    #
    # # Print the result in the desired format
    # ordering_dict = {var: min_fill_order.index(var) + 1 for var in min_fill_order}
    # print(ordering_dict)

#     new_var = Variable("X", [1, 2])
#     var2 = Variable("Y", [10, 20])
#     new_factors = Factor("new", [new_var, var2])
#     new_factors.add_values([[1, 10, 0.2], [1, 20, 0.8], [2, 10, 1.98], [2, 20, 2.26]])
#     print(new_factors.values)
#
#     final_factor = restrict(new_factors, var2, 20)
#     print(new_factors.values)  # Print values of the original factor
#     print(final_factor.values)  # Print values of the restricted factor
