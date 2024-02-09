
from deap import base
from deap import creator
from deap import tools,algorithms
from Toolkit.queue_decision_support import *
import copy
import random
import numpy as np
import gurobipy as gp


def evaluate_terminal(individual, save=False,quotas = {},extra=False):
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
        violated_quota = [1e4 for x in individual if x >= 1.75 * (args.N_TRUCKS/args.N_WINDOWS)]
        score +=sum(violated_quota)

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
        score+=sum(L)

        if args.cav_req == []:
            args.cav_req = [min(x,args.max_cavs*5) for x in individual]
        else:
            score += sum([1e2 for x in args.cav_req if x > args.max_cavs*5])
    
    if save == True:
        return score,args
    else:
        return score,


def terminal_ga(global_vals):

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
    creator.create("FitnessTer", base.Fitness, weights=(-1.0,))
    creator.create("IndividualTer", list, fitness=creator.FitnessTer)
    terminal_toolbox = base.Toolbox()
    terminal_toolbox.register("attr_int", random.randint, args.INT_MIN, args.INT_MAX)
    terminal_toolbox.register("individual", tools.initRepeat, creator.IndividualTer,
                    terminal_toolbox.attr_int, n=args.N_WINDOWS)

    terminal_toolbox.register("population", tools.initRepeat, list, terminal_toolbox.individual)
    terminal_toolbox.register("evaluate", evaluate_terminal)
    terminal_toolbox.register("mate", tools.cxTwoPoint)
    terminal_toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    terminal_toolbox.register("select", tools.selTournament, tournsize=3)

    pop = terminal_toolbox.population(n=args.N_POP)
    hof = tools.HallOfFame(args.max_hofs)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    pop, log = algorithms.eaSimple(pop, terminal_toolbox, cxpb=args.CXPB, mutpb=args.MUTPB, ngen=args.N_GEN, 
                                   stats=stats, halloffame=hof, verbose=True)
    return pop, log, hof

def evaluate_once_terminal(global_vals,hof,quotas):
    """

    """
    # Initializing args
    global args
    args = copy.deepcopy(global_vals)
    
    best_score = evaluate_terminal(hof[0],save=True,quotas = quotas,extra = True)
    return best_score