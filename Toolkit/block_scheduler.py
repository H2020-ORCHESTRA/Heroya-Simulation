import copy
from actors.logistics import *
from actors.trains import *
from actors.vessels import *  
from actors.terminal import *  
from actors.road import * 

from visualization import map_initialization, map_running
import pygame

def tactical(permutations,model):
    sols = {}
    visualization = {}
    quotas_save = {}
    quotas_save_vessel = {}
    quotas_save_train = {}
    truck_assignment = {}
    save = {}
    trucks_save = {}
    for counter,p in enumerate(permutations):
        print(p)
        if not(are_consecutive_values(p, 'vessels', 'trains')):
            continue

        trucks = {}


        print("ENTER : ----->",p)
        sol_c = 0
        try:
            if p[0]!= current_p:
                solve_first = False
        except:
            solve_first = False

        while True:
            model.args.cav_req = [] 
            quotas = {"trains" : {}, "vessels" : {}, "reduced" : {}}
            for idx,actor in enumerate(p):

                if actor == "terminal" and idx == 0:
                    break
                

                if idx == 0 and solve_first == False:
                    
                    # Call GA  
                    func1 = globals()[f"{actor}_ga"]
                    pop, log, hof = func1(model.args)
                    solve_first =True
                    current_p = actor
                
                if idx == 0:

                    # Store Best Solution
                    func2 = globals()[f"evaluate_{actor}"]
                    best_score,opt_assignment= func2(hof[sol_c], save=True)
                    edit_opt = copy.deepcopy(opt_assignment)
                    
                    if actor == 'logistics':
                        _,_,comp = evaluate_once_logistics(model.args,[hof[sol_c]],quotas)
                        truck_assignment[(p,sol_c)] = opt_assignment
                        if p.index('vessels') < p.index('trains'):
                            if p.index('vessels') < p.index('trains'):
                                trucks = {}
                                model.args.proc_v,edit_opt,trucks = get_proc_rates(model,edit_opt,trucks,flag = 'vessels')
                                model.args.proc_t,edit_opt,trucks = get_proc_rates(model,edit_opt,trucks,flag = 'trains')
                            else:
                                trucks = {}
                                model.args.proc_t,edit_opt,trucks = get_proc_rates(model,edit_opt,trucks,flag = 'trains')
                                model.args.proc_v,edit_opt,trucks = get_proc_rates(model,edit_opt,trucks,flag = 'vessels')
                        model.args.cav_req = comp['cavs']
                        
                    if actor == "terminal" or actor == "road":
                        model.args = copy.deepcopy(opt_assignment)

                    active = copy.deepcopy(opt_assignment)
                    sols[(p,sol_c)] = []
                    quotas_save[(p,sol_c)] = []

                    if actor == "vessels" or actor == "trains":
                        quotas[actor] = opt_assignment
                        if actor == "vessels":
                            quotas_save_vessel[(p,sol_c)] = quotas['vessels']
                        else:
                            quotas_save_train[(p,sol_c)] = quotas['trains']

                    if actor != "terminal" and actor != 'road':
                        quotas = reduced_quotas(quotas,opt_assignment,hof)

                else:
                    if actor != "logistics":
                        func3= globals()[f"evaluate_once_{actor}"]
                        if actor == "vessels" or actor == "trains":
                            best_score,term = func3(model.args,[hof[sol_c]],quotas)
                            quotas[actor] = term
                            quotas = reduced_quotas(quotas,term,[hof[sol_c]])

                            if actor == "vessels":
                                quotas_save_vessel[(p,sol_c)] = quotas['vessels']
                            else:
                                quotas_save_train[(p,sol_c)] = quotas['trains']
                            
                        else:
                            best_score,_ = func3(model.args,[hof[sol_c]],quotas)
                            
                    else:
                        # Store Best Solution
                        func3 = globals()[f"evaluate_once_{actor}"]
                        best_score,term,comp = func3(model.args,[hof[sol_c]],quotas)
                        edit_opt = copy.deepcopy(term)
                        
                        if term == {}:
                            break

                        truck_assignment[(p,sol_c)] = term
                        edit_quotas  = copy.deepcopy(quotas)
                        edit_comp = copy.deepcopy(comp)
                        if p.index('vessels') < p.index('logistics') and p.index('vessels') < p.index('trains'):
                            trucks = {}
                            trucks,edit_quotas,edit_comp = create_trucks(model,edit_quotas,edit_comp,trucks,'vessels')
                            trucks,edit_quotas,edit_comp = create_trucks(model,edit_quotas,edit_comp,trucks,'trains')
                        elif p.index('vessels') < p.index('logistics') and p.index('vessels') > p.index('trains'):
                            trucks = {}
                            trucks,edit_quotas,edit_comp = create_trucks(model,edit_quotas,edit_comp,trucks,'trains')
                            trucks,edit_quotas,edit_comp = create_trucks(model,edit_quotas,edit_comp,trucks,'vessels') 
                        else:
                            if p.index('vessels') < p.index('trains'):
                                trucks = {}
                                model.args.proc_v,edit_opt,trucks = get_proc_rates(model,edit_opt,trucks,flag = 'vessels')
                                model.args.proc_t,edit_opt,trucks = get_proc_rates(model,edit_opt,trucks,flag = 'trains')
                            else:
                                trucks = {}
                                model.args.proc_t,edit_opt,trucks = get_proc_rates(model,edit_opt,trucks,flag = 'trains')
                                model.args.proc_v,edit_opt,trucks = get_proc_rates(model,edit_opt,trucks,flag = 'vessels')


                        active = {}

                        for key in term.keys():
                            try:
                                if key in comp[p[idx-1]].keys():
                                    # Perform element-wise subtraction of the NumPy arrays
                                    result_array = term[key] - comp[p[idx-1]][key]
                                    active[key] = result_array
                            except:
                                pass
                        if p[idx-1]!= "terminal" and p[idx-1]!= 'road':
                            cur_actor = [x for x in ['vessels','trains'] if x != p[0]]
                            quotas,quotas_x = adjust_quotas(model.args,cur_actor[0],active,[hof[sol_c]],quotas)
                        model.args.cav_req = comp['cavs']

                sols[(p,sol_c)].append(best_score)
            try:
                quotas_save[(p,sol_c)].append(hof[sol_c])
                trucks_save[(p,sol_c)] = trucks  
                print((p,sol_c),'----->', sols[(p,sol_c)])
                sol_c +=1
            except:
                break
            if sol_c >= int(model.args.max_hofs):
                break
        if not(hasattr(model, 'map_properties')):
            model.map_properties = map_initialization(reso=(900,700))
        pygame.display.flip()
        visualization['tactical'] = {}
        visualization['disruption'] = {}
        visualization['graph'] = {}
        visualization['operational'] = {}
        visualization['tactical']['progress'] = f" Solve status : {round(100*counter/len(permutations),2)} %"
        visualization['tactical']['priority'] = f" Actor focus : {p[0].capitalize()}"
        save[counter] = visualization
        escape_pressed = map_running(model.map_properties, visualization, counter,model.args.instance_name)
    model.save = save
    model.counter = counter
    return sols,quotas_save,truck_assignment,visualization, trucks_save

def get_best_sol(sols,actors,model):

    copy_sols = copy.deepcopy(sols)

    max_actor = {}
    min_actor = {}
    for x in copy_sols.items():
        if len(x[1]) != len(actors):
            del sols[x[0]]
        for idx,y in enumerate(list(x[0][0])):

            if idx == len(x[1]):
                break

            if y not in max_actor.keys():
                if x[1][idx] <10000 :
                    max_actor[y] = x[1][idx]
            else:
                if max_actor[y]< x[1][idx] and max_actor[y] <10000 and x[1][idx] < 10000:
                    max_actor[y] = x[1][idx]
            if y not in min_actor.keys() :
                if x[1][idx] >=0:
                    min_actor[y] = x[1][idx]
            else:
                if min_actor[y]>= x[1][idx] and min_actor[y] >= 0:
                    min_actor[y] = x[1][idx]

            if idx == len(x[1]):
                break
    
    b_st = copy.deepcopy(min_actor['logistics'])
    min_actor['logistics'] = max_actor['logistics']
    max_actor['logistics'] = b_st

    mod_sols = {}
    min_stats = {}

    for x in sols.items():
        flag = True
        mod_sols[x[0]] = []
        min_stats[x[0]] = {}
        for idx,y in enumerate(list(x[0][0])):
            if x[1][idx] < 10000:
                val = round((x[1][idx]-min_actor[y])/(max_actor[y]-min_actor[y]+1e-4),2)
                if val > 1.0:
                    del mod_sols[x[0]]
                    del min_stats[x[0]]
                    flag = False
                    break
                else:                    
                    mod_sols[x[0]].append(val)
            else:
                del mod_sols[x[0]]
                del min_stats[x[0]]
                flag = False
                break
        if flag:
            min_stats[x[0]]['std'] =  np.std(mod_sols[x[0]])
            min_stats[x[0]]['mu'] =  np.mean(mod_sols[x[0]])
            min_stats[x[0]]['var'] =  np.var(mod_sols[x[0]])
            min_stats[x[0]]['cv'] = np.std(mod_sols[x[0]]) / np.mean(mod_sols[x[0]])

    min_key_m = min(min_stats, key=lambda k: min_stats[k]['mu'])
    mod_sols = dict(sorted(mod_sols.items(), key=lambda x: min_stats[x[0]]['mu']))

    model.visualization['tactical'] = {}
    
    transform = '-'.join(word.capitalize() for word in min_key_m[0])
    model.visualization['tactical']['sol'] =f"Priority : {transform}"
    model.visualization['tactical']['sol_rank'] = f"Solution Rank : {min_key_m[1]}"
    model.visualization['tactical']['sol_quality'] = f"Solution Compatibility (Mean) : {round(100*(1-min_stats[min_key_m]['mu']),2)} %"
    model.visualization['tactical']['sol_quality_var'] = f"Solution Compatibility (Variance) : {round(100*(1-min_stats[min_key_m]['std']),2)} %"
    return mod_sols,min_stats,min_key_m

def adjust_quotas(global_vals,actor,active,hof,quotas):
    """
    Randomly assign values from sample_list to the values of my_dict.

    Args:
        my_dict (dict): The dictionary with lists as values to be filled with values from sample_list.
        sample_list (list): The list of values to be assigned randomly to the dictionary values.

    Returns:
        dict: The modified dictionary with randomized values in its lists.
    """

    # Initializing args

    if actor == "logistics":
        pass
    elif actor == "trains":
        lb = [v for v in global_vals.LB_trains.values()]
        ordered = np.lexsort((global_vals.train_trucks_traffic, lb))
        comp_trucks = global_vals.trucks_per_company_train
    elif actor == "vessels":
        lb = [v for v in global_vals.LB_vessels.values()]
        ordered = np.lexsort((global_vals.vessel_trucks_traffic, lb))
        comp_trucks = global_vals.trucks_per_company_vessel

    quotas_x = {}
    for v in ordered:
        quotas_x[v+1] = [0 for _ in range(global_vals.N_WINDOWS)]
        for idx,value in comp_trucks.items():
            if idx[1] == v+1:
                for _ in range(len(value)):
                    try:
                        nonzero_indices = np.nonzero(active[int(idx[0])])
                        quotas_x[v+1][nonzero_indices[0][0]]+=1
                        active[int(idx[0])][nonzero_indices[0][0]]-=1
                    except:
                        pass
        quotas_x[v+1] = np.array(quotas_x[v+1])

    quotas[actor] = quotas_x

    return quotas,quotas_x


def reduced_quotas(quotas,quotas_x,hof):

    # Reduced capacity for guaranteed spots 
    quotas_r = np.array(copy.deepcopy(hof[0]))
    for q in quotas_x.values():
        quotas_r = np.subtract(quotas_r, q)
    quotas['reduced'] = quotas_r
    
    return quotas

def get_proc_rates(model,edit_opt,trucks,flag = 'vessels'):

    if flag == "vessels":
        trucks_ex = model.args.trucks_per_company_vessel
        modes = model.args.vessels
    else:
        trucks_ex = model.args.trucks_per_company_train
        modes = model.args.trains

    proc = {}
    save = copy.deepcopy(trucks_ex)
    
    for w in model.args.windows:
        for v in modes:
            proc[(v,w)] = 0 
            for c in edit_opt.keys():
                val = min(edit_opt[c][w-1],len(save[(c,v)]))
                edit_opt[c][w-1]-= val
                proc[v,w]+= val
                for t in range(int(val)):
                    if type(save[(c,v)]) != list:
                        save[(c,v)] = list(save[(c,v)])
                    trucks[(save[(c,v)][t],c)] = w
                save[(c,v)] = list(save[(c,v)])[int(val):]

    return proc,edit_opt,trucks

def are_consecutive_values(tup, value1, value2):
    for i in range(len(tup) - 1):
        if (tup[i] == value1 and tup[i + 1] == value2) or (tup[i] == value2 and tup[i + 1] == value1):
            return True
    return False

def create_trucks(model,edit_quotas,edit_comp,trucks,flag):

    if flag == "vessels":
        trucks_ex = model.args.trucks_per_company_vessel
        modes = model.args.vessels
        LBs = model.args.LB_vessels
    else:
        trucks_ex = model.args.trucks_per_company_train
        modes = model.args.trains
        LBs = model.args.LB_trains

    for c in model.args.companies:
        for v in modes:
            cur_trucks = list(copy.deepcopy(trucks_ex[(c,v)])) 
            send = [min(x,y,len(cur_trucks)) for x,y in zip(edit_comp[flag][c],edit_quotas[flag][v])]
            send = [x if a+1 >= LBs[v] else 0 for a,x in enumerate(send)]
            edit_quotas[flag][v] = [x - y for x, y in zip(edit_quotas[flag][v], send)]
            edit_comp[flag][c] = [x - y for x, y in zip(edit_comp[flag][c], send)]
            for win,vols in enumerate(send):
                cc = -1
                for cc in range(int(vols)):
                    try:
                        trucks[(cur_trucks[cc],c)] = win+1
                    except:
                        break
                cur_trucks = cur_trucks[cc+1:]
                if cur_trucks == []:
                    break

    return trucks,edit_quotas,edit_comp

def tactical_from_quota(permutations,model,load_quota):
    sols = {}
    visualization = {}
    quotas_save = {}
    quotas_save_vessel = {}
    quotas_save_train = {}
    truck_assignment = {}
    save = {}
    trucks_save = {}
    for counter,p in enumerate(permutations):
        
        if not(are_consecutive_values(p, 'vessels', 'trains')):
            continue
        print(p)

        trucks = {}


        print("ENTER : ----->",p)
        sol_c = 0

        while True:
            model.args.cav_req = []

            quotas = {"trains" : {}, "vessels" : {}, "reduced" : {}}
            for idx,actor in enumerate(p):
                               
                if idx == 0:

                    # Store Best Solution
                    if actor!="logistics":
                        func2 = globals()[f"evaluate_once_{actor}"]
                        best_score,opt_assignment= func2(model.args,[load_quota],quotas)
                    edit_opt = copy.deepcopy(opt_assignment)
                    
                    if actor == 'logistics':
                        best_score,opt_assignment,comp = evaluate_once_logistics(model.args,[load_quota],quotas)
                        edit_opt = copy.deepcopy(opt_assignment)
                        truck_assignment[(p,sol_c)] = opt_assignment
                        if p.index('vessels') < p.index('trains'):
                            if p.index('vessels') < p.index('trains'):
                                trucks = {}
                                model.args.proc_v,edit_opt,trucks = get_proc_rates(model,edit_opt,trucks,flag = 'vessels')
                                model.args.proc_t,edit_opt,trucks = get_proc_rates(model,edit_opt,trucks,flag = 'trains')
                            else:
                                trucks = {}
                                model.args.proc_t,edit_opt,trucks = get_proc_rates(model,edit_opt,trucks,flag = 'trains')
                                model.args.proc_v,edit_opt,trucks = get_proc_rates(model,edit_opt,trucks,flag = 'vessels')
                        model.args.cav_req = comp['cavs']
                        
                    if actor == "terminal" or actor == "road":
                        model.args = copy.deepcopy(opt_assignment)

                    active = copy.deepcopy(opt_assignment)
                    sols[(p,sol_c)] = []
                    quotas_save[(p,sol_c)] = []

                    if actor == "vessels" or actor == "trains":
                        quotas[actor] = opt_assignment
                        if actor == "vessels":
                            quotas_save_vessel[(p,sol_c)] = quotas['vessels']
                        else:
                            quotas_save_train[(p,sol_c)] = quotas['trains']

                    if actor != "terminal" and actor != 'road':
                        quotas = reduced_quotas(quotas,opt_assignment,[load_quota])

                else:
                    if actor != "logistics":
                        func3= globals()[f"evaluate_once_{actor}"]
                        if actor == "vessels" or actor == "trains":
                            best_score,term = func3(model.args,[load_quota],quotas)
                            quotas[actor] = term
                            quotas = reduced_quotas(quotas,term,[load_quota])

                            if actor == "vessels":
                                quotas_save_vessel[(p,sol_c)] = quotas['vessels']
                            else:
                                quotas_save_train[(p,sol_c)] = quotas['trains']
                            
                        else:
                            best_score,_ = func3(model.args,[load_quota],quotas)
                            
                    else:
                        # Store Best Solution
                        func3 = globals()[f"evaluate_once_{actor}"]
                        best_score,term,comp = func3(model.args,[load_quota],quotas)
                        edit_opt = copy.deepcopy(term)
                        
                        if term == {}:
                            break

                        truck_assignment[(p,sol_c)] = term
                        edit_quotas  = copy.deepcopy(quotas)
                        edit_comp = copy.deepcopy(comp)
                        if p.index('vessels') < p.index('logistics') and p.index('vessels') < p.index('trains'):
                            trucks = {}
                            trucks,edit_quotas,edit_comp = create_trucks(model,edit_quotas,edit_comp,trucks,'vessels')
                            trucks,edit_quotas,edit_comp = create_trucks(model,edit_quotas,edit_comp,trucks,'trains')
                        elif p.index('vessels') < p.index('logistics') and p.index('vessels') > p.index('trains'):
                            trucks = {}
                            trucks,edit_quotas,edit_comp = create_trucks(model,edit_quotas,edit_comp,trucks,'trains')
                            trucks,edit_quotas,edit_comp = create_trucks(model,edit_quotas,edit_comp,trucks,'vessels') 
                        else:
                            if p.index('vessels') < p.index('trains'):
                                trucks = {}
                                model.args.proc_v,edit_opt,trucks = get_proc_rates(model,edit_opt,trucks,flag = 'vessels')
                                model.args.proc_t,edit_opt,trucks = get_proc_rates(model,edit_opt,trucks,flag = 'trains')
                            else:
                                trucks = {}
                                model.args.proc_t,edit_opt,trucks = get_proc_rates(model,edit_opt,trucks,flag = 'trains')
                                model.args.proc_v,edit_opt,trucks = get_proc_rates(model,edit_opt,trucks,flag = 'vessels')


                        active = {}

                        for key in term.keys():
                            try:
                                if key in comp[p[idx-1]].keys():
                                    # Perform element-wise subtraction of the NumPy arrays
                                    result_array = term[key] - comp[p[idx-1]][key]
                                    active[key] = result_array
                            except:
                                pass
                        if p[idx-1]!= "terminal" and p[idx-1]!= 'road':
                            cur_actor = [x for x in ['vessels','trains'] if x != p[0]]
                            quotas,quotas_x = adjust_quotas(model.args,cur_actor[0],active,[load_quota],quotas)
                        model.args.cav_req = comp['cavs']

                sols[(p,sol_c)].append(best_score)
            quotas_save[(p,sol_c)].append(load_quota)
            trucks_save[(p,sol_c)] = trucks  
            print((p,sol_c),'----->', sols[(p,sol_c)])
            sol_c +=1
            if sol_c >= int(1):
                break
        if not(hasattr(model, 'map_properties')):
            model.map_properties = map_initialization(reso=(900,700))
        pygame.display.flip()
        visualization['tactical'] = {}
        visualization['disruption'] = {}
        visualization['graph'] = {}
        visualization['operational'] = {}
        visualization['tactical']['progress'] = f" Solve status : {round(100*counter/len(permutations),2)} %"
        visualization['tactical']['priority'] = f" Actor focus : {p[0].capitalize()}"
        save[counter] = visualization
        escape_pressed = map_running(model.map_properties, visualization, counter,model.args.instance_name)
    model.save = save
    model.counter = counter
    return sols,quotas_save,truck_assignment,visualization, trucks_save


