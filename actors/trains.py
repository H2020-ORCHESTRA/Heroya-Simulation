
from deap import base
from deap import creator
from deap import tools,algorithms
from Toolkit.queue_decision_support import *
import copy
import random
import numpy as np

def transform_to_proc(data, m):
    """
    Transform a NumPy array by comparing each cell's value to a threshold 'm'.
    If a cell's value is greater than 'm', the difference between the cell's
    value and 'm' is added to the next column in the same row, and the cell is
    set to 'm'.

    Parameters:
        data (numpy.ndarray): The input NumPy array to be transformed.
        m (float): The threshold value.

    Returns:
        numpy.ndarray: The transformed NumPy array.
    """
    # Iterate through each row
    for row in data:
        for i in range(len(row) - 1):
            if row[i] > m:
                diff = row[i] - m
                row[i + 1] += diff
                row[i] = m
    return data


def evaluate_trains(individual,save=False,quotas = {}):
    """
    Evaluate the trains assignment and return a score.

    Args:
        individual (list): A list representing the truck assignment.
        save (bool): If True, return additional information along with the score.
        quotas (dict): Dictionary communicating guaranteed spots for specific actors.

    Returns:
        float or tuple: The logistics assignment score, and optionally, additional information.
    """
   
    # Initialize score and returns
    score = 0

    if quotas  == {}:
        quotas['vessels'],quotas['trains'],quotas['reduced'] = {},{},{} 

    # Check if assignment does not match to the number of required trucks
    n_trucks_assigned = abs(sum(individual)- args.N_TRUCKS) 

    result = [abs(x - y) for x, y in zip(individual, args.orig_quota)]
    result =  [0 if x < 12 else (x-12)*1000 for x in result]
    score +=sum(result)     

    if n_trucks_assigned != 0:
        score = 1e4*n_trucks_assigned
    
    else:
   
        ### Find optimal order of vessel arrivals based on quota ###
        # First is the one with the earliest LB_vessels and biggest associated traffic # 
        
        # Define order of vessels to be served
        order = []
        quotas_v = {}

        for t in args.trains:
            order.append([args.LB_trains[t],len(args.train_trucks[t]),t])
        order = sorted(order, key=lambda x: (x[0], -x[1]))
        
        order_id = [x[-1] for x in order]


        # Initialize arrival rates
        l = np.zeros((args.N_TRAINS,args.N_WINDOWS))

        if 'proc_t' not in args:
            args.proc_t = {(v,key): args.proc_trucks for key in args.windows for v in args.trains}

        # Defined order of arrivals when considering only the amount of trucks that can be processed per time-window
        if quotas['trains'] == {} and quotas['vessels'] == {}: # Enters when first
            for i,ind in enumerate(individual):
                sum_ind = ind
                for j,v in enumerate(args.trains):
                    if order[j][0]<= i+1:
                        l[j][i] = min(sum_ind,order[j][1],args.proc_t[(order_id[j],i+1)])
                        order[j][1] -=  l[j][i]
                        sum_ind -=  l[j][i]

                        if sum_ind == 0:
                            break
        

        else : # Enters when after vessels
            for i,ind in enumerate(quotas['reduced']):
                sum_ind = ind
                for j,v in enumerate(args.trains):
                    if order[j][0]<= i+1:
                        l[j][i] = min(sum_ind,order[j][1],args.proc_t[(order_id[j],i+1)])
                        order[j][1] -=  l[j][i]
                        sum_ind -=  l[j][i]

                        if sum_ind == 0:
                            break 

        for i,_ in enumerate(individual):
            for j,v in enumerate(args.trains):
                score += l[j][i] * abs((i+1-order[j][0]))
        # Penalty for bad logistics assignment
        score += 10000*(args.N_TRUCKS_TRAINS-sum(sum(l)))


        or_l = copy.deepcopy(l)       
        # Save quotas for optimal solution
        for j,v in enumerate(args.trains):
            if save == True:
                quotas_v[v] = or_l[order_id[j]-1]

    if save == True:       
        return score, quotas_v
    
    return score, 


def trains_ga(global_vals):
    """
    The GA algorithm for trains

    Args:
        global_vals (args): Argument passed form the main file

    Returns:
        float or tuple: The logistics assignment score, and optionally, additional information.
    """

    # Initializing args
    global args
    args = copy.deepcopy(global_vals)

    # GA MODEL 
    creator.create("FitnessMax", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    train_toolbox = base.Toolbox()
    train_toolbox.register("attr_int", random.randint, args.INT_MIN, args.INT_MAX)
    train_toolbox.register("individual", tools.initRepeat, creator.Individual,
                    train_toolbox.attr_int, n=args.N_WINDOWS)

    train_toolbox.register("population", tools.initRepeat, list, train_toolbox.individual)
    train_toolbox.register("evaluate", evaluate_trains)
    train_toolbox.register("mate", tools.cxTwoPoint)
    train_toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    train_toolbox.register("select", tools.selTournament, tournsize=3)
    
    pop = train_toolbox.population(n=args.N_POP)
    hof = tools.HallOfFame(args.max_hofs)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    pop, log = algorithms.eaSimple(pop, train_toolbox, cxpb=args.CXPB, mutpb=args.MUTPB, ngen=args.N_GEN, 
                                   stats=stats, halloffame=hof, verbose=True)
    return pop, log, hof


def evaluate_once_trains(global_vals,hof,quotas):
    """

    """
    # Initializing args
    global args
    args = copy.deepcopy(global_vals)
    
    best_score,term = evaluate_trains(hof[0],save=True,quotas = quotas)

    return best_score,term