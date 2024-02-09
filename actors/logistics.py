
from deap import base
from deap import creator
from deap import tools,algorithms
from Toolkit.queue_decision_support import *
import copy
import random
import numpy as np
from scipy.optimize import linear_sum_assignment
import gurobipy as gp


def hungarian_algorithm(individual,weight):

    """
    Quickly apply the Hungarian Algorithm for fitness evaluation.

    Args:
        individual (list): The currently evaluated quota
        weight (np.array): The matrix to be transformed in NxN where assignment should happen.

    Returns:
        score : The utility from this assignment
    """

    # Delete windows with zero representation in the quota
    cols_delete = []
    for ind_x,x in enumerate(args.windows):
        if individual[ind_x] == 0:
            cols_delete.append(ind_x)

    hung_weight = np.empty((weight.shape[0],0)) # Helper array to store original/modified values

    # Add extra windows based on occurence in the quota to prepare for hungarian algorithm
    for ind_x,x in enumerate(args.windows):

        # Number of times to copy the column
        k = individual[ind_x]

        if k <= 0:
            continue
        else:

            # Extract the specified column
            column_to_copy = weight[:, ind_x]

            # Use np.repeat to replicate the column 'k' times
            copied_column = np.repeat(column_to_copy, k)

            # Reshape the copied column to match the number of rows in the original array
            copied_column = copied_column.reshape(-1, k)

            # Create a new array by horizontally stacking the copied column with the original array
            hung_weight = np.hstack([hung_weight, copied_column])

    weight = np.empty((0,hung_weight.shape[1]))
    for ind_x,x in enumerate(args.companies):

        # Number of times to copy the row
        k = len(args.company_trucks[x])
        if k <= 0:
            continue
        else:

            # Extract the specified row
            row_to_copy = hung_weight[ind_x, :]

            # Use np.tile to replicate the row 'k' times
            copied_row = np.tile(row_to_copy, (k, 1))

            # Create a new array by vertically stacking the copied row with the original array
            weight = np.vstack([weight, copied_row])

    row_ind, col_ind = linear_sum_assignment(weight,maximize=True)    
    score = -weight[row_ind, col_ind].sum()

    return score


def linear_assignment_constr(weight,quotas,individual,save=False):

    """
    Optimal assignment that can also assert guaranteed slots for specific actors

    Args:
        weight (np.array): The matrix  where assignment should happen.
        quotas (dict) : Dictionary relating to spots acquired from company for vessel-trucks/ train/trucks
        individual (list): The currently evaluated quota.

    Returns:
        mdl.ObjVal : The utility from this assignment
        opt_assingment (dict) : The optimal assignment of trucks per company
        comp_train (dict) : The assignment of trucks per train
        comp_vessel (dict) : The optimal assignment of trucks per vessel
    """

    mdl = gp.Model("linear_assingment_constr")

    x = {}
    for t in args.companies:
        for w in args.windows:
            x[t, w] = mdl.addVar(vtype=gp.GRB.INTEGER, name=f'x[{t}, {w}]')

    o = {}
    for t in args.companies:
        for w in args.windows:
            o[t, w] = mdl.addVar(vtype=gp.GRB.INTEGER, name=f'o[{t}, {w}]')

    y = {}
    for t in args.companies:
        for w in args.windows:
            for v in args.vessels:
                y[t, w, v] = mdl.addVar(vtype=gp.GRB.INTEGER, name=f'y[{t}, {w}, {v}]')

    z = {}
    for t in args.companies:
        for w in args.windows:
            for v in args.trains:
                z[t, w, v] = mdl.addVar(vtype=gp.GRB.INTEGER, name=f'z[{t}, {w}, {v}]')


    # Constraint: each window gets assigned exactly the trucks of the quota
    for w in args.windows:
        mdl.addConstr(gp.quicksum(x[t, w] for t in args.companies) == individual[w-1])

    # Constraint: each company gets assigned exactly the trucks it has
    for t in args.companies:
        mdl.addConstr(gp.quicksum(x[t, w] for w in args.windows) == len(args.company_trucks[t]))

    for t in args.companies:
        mdl.addConstr(gp.quicksum(o[t, w] for w in args.windows) == len(args.company_cav_trucks[t]))

    for t in args.companies:
        for w in args.windows:
            mdl.addConstr(o[t, w] <= x[t,w])

    if args.cav_req!=[]:
        for w in args.windows:
            mdl.addConstr(gp.quicksum(o[t, w] for t in args.companies) <= args.cav_req[w-1])


    if quotas['vessels'] != {}:
        for v in args.vessels:
            for w in args.windows:
                mdl.addConstr(gp.quicksum(y[t, w, v] for t in args.companies) == quotas['vessels'][v][w-1])

        for t in args.companies:
            for v in args.vessels:
                mdl.addConstr(gp.quicksum(y[t, w, v] for w in args.windows) == len(args.trucks_per_company_vessel[(t,v)]))

        for t in args.companies:
            for w in args.windows:
                mdl.addConstr(gp.quicksum(y[t, w, v] for v in args.vessels) <= x[t,w])

    if quotas['trains'] != {}:

        for v in args.trains:
            for w in args.windows:
                mdl.addConstr(gp.quicksum(z[t, w, v] for t in args.companies) == quotas['trains'][v][w-1])

        for t in args.companies:
            for v in args.trains:
                mdl.addConstr(gp.quicksum(z[t, w, v] for w in args.windows) == len(args.trucks_per_company_train[(t,v)]))

        for t in args.companies:
            for w in args.windows:
                mdl.addConstr(gp.quicksum(z[t, w, v] for v in args.trains) <= x[t,w])

    if quotas['trains'] != {} and quotas['vessels'] != {}:
        for t in args.companies:
            for w in args.windows:
                mdl.addConstr(gp.quicksum(z[t, w, v] for v in args.trains) + gp.quicksum(y[t, w, v] for v in args.vessels)  <= x[t,w])    


    # Return optimal schedule for quota and vessel quota
    mdl.Params.LogToConsole = 0
    mdl.setObjective(gp.quicksum(x[t, w]*weight[t-1,w-1] for t in args.companies for w in args.windows),gp.GRB.MAXIMIZE)
    mdl.optimize()


    if mdl.Status == gp.GRB.INFEASIBLE or mdl.Status == gp.GRB.INF_OR_UNBD:
        return 1e5,{},{}
    

    opt_assingment = {}
    for t in args.companies:
        opt_assingment[t] = np.zeros(args.N_WINDOWS)
        for w in args.windows:
            if x[t, w].x >=0.5:
                opt_assingment[t][w-1]= x[t, w].x
    
    comp = {}
    
    comp_train = {}
    for t in args.companies:
        comp_train[t] = np.zeros(args.N_WINDOWS)
        for v in args.trains:
            for w in args.windows:
                if z[t, w, v].x >=0.5:
                    comp_train[t][w-1] += z[t, w, v].x

    comp['trains'] = comp_train

    comp_vessel = {}
    for t in args.companies:
        comp_vessel[t] = np.zeros(args.N_WINDOWS)
        for v in args.vessels:
            for w in args.windows:
                if y[t, w, v].x >=0.5:
                    comp_vessel[t][w-1] += y[t, w, v].x
    
    comp['vessels'] = comp_vessel
    comp['cavs'] = [sum(o[t,w].x for t in args.companies) for w in args.windows]

    if save == True:
        return mdl.ObjVal,opt_assingment,comp
    else:
        return mdl.ObjVal,opt_assingment

def evaluate_logistics(individual, save=False,quotas = {},extra=False):
    """
    Evaluate the logistics assignment and return a score.

    Args:
        individual (list): A list representing the truck assignment.
        save (bool): If True, return additional information along with the score.
        quotas (dict): Dictionary communicating guaranteed spots for specific actors

    Returns:
        float or tuple: The logistics assignment score, and optionally, additional information.
    """

    # Initializer returns
    score = 0
    if quotas  == {}:
        quotas['vessels'],quotas['trains'],quotas['reduced'] = {},{},{}

    result = [abs(x - y) for x, y in zip(individual, args.orig_quota)]
    result =  [0 if x < 12 else (x-12)*1000 for x in result]
    score +=sum(result)      

    # Assert all demand is assigned by assigning a big value if not covered
    if sum(individual) - args.N_TRUCKS !=0 :
        score +=1e4*(abs(sum(individual) - args.N_TRUCKS))

    else:
        ### Queuing approximation ###

        l =  np.array([sum(individual[n::args.N_WINDOWS]) for n in range(args.N_WINDOWS)])
        b = [0 for _ in range(args.N_WINDOWS)]
        l_star = [0 for _ in range(args.N_WINDOWS)]
        P = [0 for _ in range(args.N_WINDOWS)]
        E = [0 for _ in range(args.N_WINDOWS)]
        l_mar = [0 for _ in range(args.N_WINDOWS)]
        l_star[0] = l[0]
        L = [0 for _ in range(args.N_WINDOWS)]

        for w in range(args.N_WINDOWS):
            P[w],E[w],l_mar[w] = get_approx_measures(l_star[w],args.mu,args.c)
            b[w] = P[w] * l_star[w]

            if w != args.N_WINDOWS-1:
                l_star[w+1] = l[w+1] + b[w]

            Lq_mmc,W_mmc = get_mmc_measures(l_mar[w],args.mu,args.c) 
            Lq_mdc,W_mdc = get_mdc_measures(l_mar[w],args.mu,args.c,W_mmc)
            W = (0.1*W_mmc + (1-0.1)*W_mdc)
            L[w] = W*l_mar[w]

        # Queues per time-windows
        L = [x if not np.isnan(x) else 0 for x in L]

        # Create weighted costs from queues and willingess to pay
        weight = np.zeros((args.N_COMPANIES,args.N_WINDOWS))
 
        # Original weights per time-window and company
        for ind_x,x in enumerate(args.companies):
            for ind_y,_ in enumerate(args.windows):
                weight[ind_x][ind_y] = args.WTP_C[x][ind_y] - L[ind_y]*args.CC[x]

        orig_weight = copy.deepcopy(weight)

        if quotas["vessels"] == {} and quotas["vessels"] == {}:
            score += hungarian_algorithm(individual,weight)
        else:
            if save == True:
                score1,_,_ = linear_assignment_constr(orig_weight,quotas,individual,save=True)  
            else:
                score1,_ = linear_assignment_constr(orig_weight,quotas,individual)
            score+=score1  

    if save:
        return linear_assignment_constr(orig_weight,quotas,individual,extra)      
    else:
        return score,


def logistics_ga(global_vals):

    """
    The GA algorithm for logistics companies 

    Args:
        globals (args): Argument passed form the main file

    Returns:
        float or tuple: The logistics assignment score, and optionally, additional information.
    """

    # Initializing args
    global args
    args = copy.deepcopy(global_vals)
    
    # GA MODEL 
    creator.create("FitnessLog", base.Fitness, weights=(-1.0,))
    creator.create("IndividualLog", list, fitness=creator.FitnessLog)
    logistics_toolbox = base.Toolbox()
    logistics_toolbox.register("attr_int", random.randint, args.INT_MIN, args.INT_MAX)
    logistics_toolbox.register("individual", tools.initRepeat, creator.IndividualLog,
                    logistics_toolbox.attr_int, n=args.N_WINDOWS)

    logistics_toolbox.register("population", tools.initRepeat, list, logistics_toolbox.individual)
    logistics_toolbox.register("evaluate", evaluate_logistics)
    logistics_toolbox.register("mate", tools.cxTwoPoint)
    logistics_toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    logistics_toolbox.register("select", tools.selTournament, tournsize=3)

    pop = logistics_toolbox.population(n=args.N_POP)
    hof = tools.HallOfFame(args.max_hofs)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    pop, log = algorithms.eaSimple(pop, logistics_toolbox, cxpb=args.CXPB, mutpb=args.MUTPB, ngen=30, 
                                   stats=stats, halloffame=hof, verbose=True)
    return pop, log, hof

def evaluate_once_logistics(global_vals,hof,quotas):
    """

    """
    # Initializing args
    global args
    args = copy.deepcopy(global_vals)
    
    best_score,term,comp = evaluate_logistics(hof[0],save=True,quotas = quotas,extra = True)

    return best_score,term,comp