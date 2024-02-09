import pandas as pd
from itertools import combinations,permutations
import gurobipy as gp
from gurobipy import GRB
import pygame
from visualization import map_initialization,map_running,create_folder_path

def convert_to_time_string(input_str):
    # Ensure the input string is a valid integer
    try:
        minutes = int(input_str)
    except ValueError:
        return "Invalid input"

    # Calculate hours and minutes
    hours, remainder = divmod(minutes, 60)

    # Format the time string
    formatted_time = f"{hours % 12 or 12}:{remainder:02d} {'AM' if hours < 12 else 'PM'}"
    
    return formatted_time


def bundling(agents,step,size):

    # Inner loop for job generation for bundles of length up to d
    jobs = pd.DataFrame()
    destinations = (set(agents['TERMINAL'].tolist()))

    for d in range(1,size):
        truck_comb = list(combinations(agents.index,d))
        # Storing performance of examined combinations
        comb_performance = {}
        
        for comb in truck_comb:
            #print(comb)
        
            # Loading Input specific to combination 
            trucks = list(comb)

            # Isolating job's information for specific bundle
            trucks_input = {}
            for i,row in agents.iterrows():
                if i in trucks:
                    trucks_input[i] = row[1],row[-1]-step,row[2]
            trucks,target_dest,arrival,chaperon= gp.multidict(trucks_input)
            destinations = list(set(target_dest.values()))

            # Exit Loop if a truck in the combination does not need chaperoning 
            ind = False
            for t in trucks:
                if chaperon[t] == 0 and len(trucks)!=1:
                    ind= True
                    continue

            # Possible States based on number of destinations
            states = list(range(1,d+1))
            destinations.append("Gate")
            
            # Possible Combinations of ordering 
            indices = list(permutations(comb, len(states)))

            # Initialize earliest job start
            idle_gate_time = 0
            earliest_start = max(arrival.values())
            for t in trucks: 
                idle_gate_time += earliest_start-arrival[t]
            
            # Determined Traversal inside the port
            min_cost = 1e+4 
            for index in indices:
                travel_cost = []
                entry_time = []
                finish_time = []
                start = "Gate"
                for m,t in enumerate(index):
                    if m == 0:
                        cur_t = target_dest[t]
                        travel_cost.append(3)
                        finish_time.append(sum(travel_cost))
                        entry_time.append(earliest_start+travel_cost[-1])
                    else:
                        if target_dest[t] == cur_t:
                            travel_cost.append(travel_cost[-1])
                            finish_time.append(finish_time[-1])
                            entry_time.append(entry_time[-1])
                        else:
                            cur_t = target_dest[t]
                            travel_cost.append(3)
                            finish_time.append(travel_cost[-1]+3)
                            entry_time.append(entry_time[-1]+travel_cost[-1])                        
                
                # Travel Cost Equal to extra time and travel cost
                turnaround = max(finish_time)+3
                cost = turnaround + idle_gate_time
                if cost < min_cost:
                    best_ind = index
                    min_cost = cost
                    best_turnaround =  turnaround
                    destinations = [target_dest[x] for x in best_ind]
                    entry_time = [x for x in entry_time]
                    best_ind  = tuple([str(x) for x in list(best_ind)])
                    comb_performance[best_ind] = [min_cost,best_turnaround,destinations,entry_time]

            # Store results in dictionary
            results = {}
            results["W"] = [step//15]
            for c in comb_performance.items():
                results["T"] = [c[1][0]]
                results["Sq"] = [','.join(c[0])]
                results["ET"] = [max(arrival.values())]
                results["En"] =[",".join(f"{x-results['ET'][0]}" for x in c[1][3])]
                results["LT"] = [sum(results["ET"]) + c[1][0]]
                results["F"] =[",".join(str(x) for x in c[1][2])]
                results["I"] =[idle_gate_time]
                    
            jobs = pd.concat([jobs, pd.DataFrame.from_dict(results)], ignore_index=True)
    
    max_j = int(2*len(agents))
    try:
        jobs = jobs.set_index("Sq")
        if len(jobs) >= max_j:
            jobs = jobs.sort_values(by='I',).iloc[0: max_j]
            jobs2 = jobs[jobs['I'] == 0]
            if len(jobs2) >= len(jobs):
                jobs = jobs2 
        jobs.to_csv(f"data\schedules\jobs_{step}.csv")
    except:
        pass
    #print(f"BUNDLING DONE {step}, {len(agents)}")
    return jobs

def scheduling(self,agents,bundles,steps,cavs,destinations,del_thres = 60,time_window_len=15,w_pen = 3,occupied = {}):

    if agents.empty:
        return {}
        
    multidict_input = {}
    for i,row in agents.iterrows():
        multidict_input[i] =row[1],row[-1]-steps[0],row[2]
    trucks,target_dest,arrival,escort = gp.multidict(multidict_input)


    multidict_input = {}
    idx = 0
    for i,row in bundles.iterrows():
        name_index = f"J_{idx}"
        multidict_input[name_index] = i,row[0],row[1],row[2],row[3],row[4],row[5]
        idx+=1
    jobs,Sq,W,TC,ET,En,LT,F = gp.multidict(multidict_input)
    window = row[0]


    N = {}
    for j in jobs:
        N[j] = (len(Sq[j].split(",")))

   
    model = gp.Model('PJS')

    # Decision Variables
    z = model.addVars(jobs,steps,cavs,vtype=GRB.BINARY,name="z")
    x = model.addVars(jobs,steps,cavs,vtype=GRB.BINARY,name="x")
    w = model.addVars(jobs,trucks,vtype=GRB.BINARY,name="w")

    # Penalty DVs
    dv = model.addVars(trucks,vtype=GRB.INTEGER,name="dv")
    wp = model.addVars(jobs,steps,cavs,vtype=GRB.INTEGER,name="wp")

    # Constraint (1) 
    for t in trucks:
        flag = False
        for j in jobs:
            if str(t) in Sq[j]:
                flag = True
                break
        if flag:
            model.addConstr(sum(w[j,t] for j in jobs if str(t) in Sq[j]) == 1)
        model.addConstr(sum(w[j,t] for j in jobs if str(t) not in Sq[j]) == 0)

    # Constraint (6) 
    for t in trucks:
        for j in jobs:
            model.addConstr(dv[t]+arrival[t]>= sum((s-steps[0])*z[j,s,c] - steps[-1]*(1-w[j,t]) for s in steps for c in cavs))
            model.addConstr(sum(z[j,s,c] for s in steps for c in cavs)>=w[j,t])

    # Constraint (5)
    for j in jobs:
        for s in steps:
            if s < steps[0] + ET[j]:
                model.addConstr(sum(z[j,s,c] for c in cavs) == 0) # Fast

   # Constraint (6)
    for j in jobs:
        for s in steps:
            for c in cavs:
                model.addConstr((z[j,s,c] == 1) >> (sum(w[j,t] for t in trucks) == N[j])) # Fast
                if s!= steps[-1]:
                    model.addConstr((z[j,s,c] == 1) >> (sum(z[j,s+1,c] for j in jobs for c in cavs) <=0)) # Fast

    # Constraint (4)
    for s in steps:
        model.addConstr(sum(z[j,s,c] for j in jobs for c in cavs) <=1)


    # # Constraint (6)
    # for j in jobs:
    #     for s in range(steps[0],steps[-3]):
    #         for c in cavs:
    #             model.addConstr((z[j,s,c] == 1) >> (sum(z[j,s+1,c]+z[j,s+2,c] for j in jobs for c in cavs) == 0)) # Fast

    # Constraint (8)
    for j in jobs:
        for s in steps:
            cur_steps = range(s,s+TC[j])
            if cur_steps[-1] <= steps[-1]:
                for c in cavs:
                    model.addConstr((z[j,s,c] == 1) >> (sum(x[j,s_prime,c] for s_prime in cur_steps) == TC[j])) # Fast
            else:
                for c in cavs:
                    model.addConstr((z[j,s,c] == 0))   

    for j in jobs:
        model.addConstr(sum(x[j,s,c] for s in steps for c in cavs) <= TC[j]*(sum(w[j,t] for t in trucks)/N[j]))

    for s in steps:
        for c in cavs:
            if c >= 1:
                model.addConstr(sum(x[j,s,c] for j in jobs) <=1)


    # Constraint (15) - (17)    
    for j in jobs:
        seq_j = list(Sq[j].split(","))

        if len(seq_j) >= 2:
            model.addConstr(sum(z[j,s,0] for s in steps) == 0)
        
        if len(seq_j) == 1:
            if escort[int(seq_j[0])] == 1:
                model.addConstr(sum(z[j,s,0] for s in steps) == 0)
            else:
                model.addConstr(sum(z[j,s,0] for s in steps) == 1)


    # Constraint (18)  
    for i in occupied.keys():
        model.addConstr(sum(x[j,i[1],i[2]] for j in jobs) == 0)
        
    model.setParam("LogToConsole",0)
    model.setObjective(sum(dv[t] for t in trucks)
                        ,GRB.MINIMIZE)
    model.optimize()
    occupied = {}
    if model.status == GRB.OPTIMAL:
        df = pd.DataFrame(columns = ["Truck","Arrival","Start","End","Deviation","CAV"])
        for j in jobs:
            for s in steps:
                for c in cavs:
                    if s> steps[0] + time_window_len:
                        if x[j,s,c].x >=0.9999 and c>=1:
                            occupied[j,s,c]=1
                            #print(s,steps,x[j,s,c].x,j,s,c)

        for j in jobs:
            for t in trucks:
                if w[j,t].x >= 0.99:
                    for s in steps:
                        for c in cavs:
                            if z[j,s,c].x >= 0.99:
                                df2 = pd.DataFrame({"Truck" : [t],
                                                            "Arrival" : [steps[0]+arrival[t]],
                                                            "Start" : [s],
                                                            "End" : [s+LT[j]-ET[j]],
                                                            "Deviation" : [dv[t].x],
                                                            "CAV" : [c],
                                                            "Terminal" : [F[j]]
                                                            })
                                df = pd.concat([df, df2], ignore_index = True, axis = 0)
                                for agent in self.schedule.agents:
                                    if agent.unique_id == t:
                                        if not(agent.enter):
                                            self.deviation += dv[t].x
                                            agent.w_time = dv[t].x
                                            agent.platoon = len(F[j])
                                            self.tot_trucks+=1
                                            agent.enter = True
                                        else:
                                            self.deviation += dv[t].x
                                            self.deviation -= agent.w_time
                                            agent.w_time = dv[t].x
                                            agent.platoon = len(F[j])
                                          
                                            
        df = df[(df['Arrival'] >= self.schedule.steps) & (df['Arrival'] <= self.schedule.steps+14)]
        df['window'] = (df['Start'] / 60.1).astype(int)
        dummy = pd.Series(0, index=self.sum_df1.index)
        dummy += df['window'].value_counts()
        dummy = dummy.fillna(0)
        self.sum_df2 += dummy
        create_folder_path(f"data/instances/{self.args.instance_name}/schedules")
        df.to_excel(f"data/instances/{self.args.instance_name}/schedules/individual_schedule_{steps[0]}.xlsx")
        obj =model.objVal
        self.platoons += df[df.duplicated('Start', keep=False)].shape[0]
        self.visualization['operational']['time'] = f" Time : {convert_to_time_string(f'{self.schedule.steps}')}"
        self.visualization['operational']['deviation'] = f" Total Waiting Time : {round(self.deviation)} min"
        try:
            self.visualization['operational']['avg_dev'] = f" Average Gate Waiting time (time-window) : {round(df['Deviation'].mean())} min"
            self.visualization['operational']['max_dev'] = f" Maximum Gate Waiting time (time-window) : {round(df['Deviation'].max())} min"
        except:
            self.visualization['operational']['avg_dev'] = f" Average Gate Waiting time (time-window) : {0} min"
            self.visualization['operational']['max_dev'] = f" Maximum Gate Waiting time (time-window) : {0} min"            
        self.visualization['operational']['cav1'] = f" CAV 1 Active : {df['CAV'].isin([1]).any()}"
        if self.args.max_cavs == 2:
            self.visualization['operational']['cav2'] = f" CAV 2 Active : {df['CAV'].isin([2]).any()}"
        self.visualization['operational']['plat'] = f" Total Platoons : {self.platoons}"
        self.visualization['operational']['tot'] = f" Total Truck Traffic : {self.tot_trucks}"
        self.visualization['operational']['tot_arr'] = f" Total Trucks En-Route : {max(0,self.tot_trucks-self.arrived)}"            
        if not(hasattr(self, 'map_properties')):
            self.map_properties = map_initialization(reso=(900,700))
            self.visualization['disruption'] = {}

        pygame.display.flip()
        escape_pressed = map_running(self.map_properties, self.visualization, self.counter,self.args.instance_name)
        return occupied
    else:
        print("we have an issue")
        # model.write('model.lp')
        # model.computeIIS()
        # model.write(f'model_{steps[0]}.ilp')
        return {}
    