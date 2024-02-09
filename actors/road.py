
from deap import base
from deap import creator
from deap import tools,algorithms
from Toolkit.queue_decision_support import *
import copy
import random
import numpy as np
import gurobipy as gp
import ast

def evaluate_road(individual, save=False,quotas = {},extra=False):
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

        try:
            lst = ast.literal_eval(args.avoid)
        except:
            lst = args.avoid

        for av in lst:
            for x,q in enumerate(individual):
                if x+1 == int(av):
                    score += (q)

    
    if save == True:
        return score,args
    else:
        return score,


def road_ga(global_vals):

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
    creator.create("FitnessRoad", base.Fitness, weights=(-1.0,))
    creator.create("IndividualRoad", list, fitness=creator.FitnessRoad)
    road_toolbox = base.Toolbox()
    road_toolbox.register("attr_int", random.randint, args.INT_MIN, args.INT_MAX)
    road_toolbox.register("individual", tools.initRepeat, creator.IndividualRoad,
                    road_toolbox.attr_int, n=args.N_WINDOWS)

    road_toolbox.register("population", tools.initRepeat, list, road_toolbox.individual)
    road_toolbox.register("evaluate", evaluate_road)
    road_toolbox.register("mate", tools.cxTwoPoint)
    road_toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    road_toolbox.register("select", tools.selTournament, tournsize=3)

    pop = road_toolbox.population(n=args.N_POP)
    hof = tools.HallOfFame(args.max_hofs)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    pop, log = algorithms.eaSimple(pop, road_toolbox, cxpb=args.CXPB, mutpb=args.MUTPB, ngen=args.N_GEN, 
                                   stats=stats, halloffame=hof, verbose=True)
    return pop, log, hof

def evaluate_once_road(global_vals,hof,quotas):
    """

    """
    # Initializing args
    global args
    args = copy.deepcopy(global_vals)
    
    best_score = evaluate_road(hof[0],save=True,quotas = quotas,extra = True)

    return best_score