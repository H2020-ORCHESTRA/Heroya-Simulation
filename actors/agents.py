
########################################################## agents.py ########################################################## 

# Import Libraries
import mesa
import pandas as pd
import calendar
import pickle
import itertools
import random
from datetime import datetime,timedelta
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from memory_profiler import profile

# Custom Initializations - Orchestra Toolkit
from Toolkit.demand_generation import * 
from Toolkit.network_represenentations import *
from Toolkit.dynamic_guidance import * 
from Toolkit.block_scheduler import * 
from Toolkit.individual_scheduler import *

# Initialize Truck Class 
class Truck(mesa.Agent):

    def __init__(self,unique_id,model,company,CAV,time,activation,terminal,lonlat,c_index):

        """
        :params unique_id, model : inherited from mesa.Model environment
        :param company (str): Name of Company
        :param CAV (int) : 0 or 1 indicating need for guidance
        :params time (int) : time to reach the terminal
        :params activation (int) : time (minutes from midnight) that the trip starts
        :param lonlat (tuple) : Coordinates of starting position 
        """

        # Defined initials passed by input
        super().__init__(unique_id, model)
        self.company = company
        self.CAV = CAV
        self.time = time
        self.activation = activation
        self.terminal = terminal
        self.lonlat =lonlat
        self.c_index = c_index

        self.detour = False
        self.w_time = 0
        self.platoon = 0
        self.enter = False

    def assign(self):

        '''
        Determine assignemnt of Truck and store KPIs when Passenger is activated

        '''     
        # Current step as timestamp
        self.D_TIME = pd.Timestamp(year=self.model.date.year,month=self.model.date.month,day=self.model.date.day,
                    hour =self.model.schedule.steps//60,
                    minute=self.model.schedule.steps%60)
         
        # Initialize route handling based on disruption occurence
        if self.model.disruption == {} or self.model.schedule.steps not in self.model.dis_range:
            coords = (self.lonlat,self.model.heroya_lonlat)
            CAR_ROUTE = return_car_route(self.model.client,coords,radius=250)
        else:
            coords = (self.lonlat,self.model.heroya_lonlat)
            CAR_ROUTE = get_road_route(self.model.client,self.lonlat,self.model.heroya_lonlat,self.model.disruption)   
        
        # Extract KPIS
        extract_KPIS(self.model,self,CAR_ROUTE)


class Heroya(mesa.Model):
    
    '''
        A model representation of the environment surrounding HIP
    '''

    def __init__(self,args,client):

        """
        :param args (argparse.args): passed from config file
        :param client (openrouteservice.client): ORS local clients
        """

        # Define case study imposed parameters - Variable
        
        self.args = args
        self.client = client
        random.seed(args.seed)

        # Empty model initials
        self.deviation = 0
        self.platoons = 0
        self.tot_trucks = 0
        self.delay_dis = 0
        self.affected_trucks = 0
        self.arrived = 0
        self.occupied = {}
        self.visualization = {'operational':{},'graph':{},"tactical" : {}, 'disruptions' : {}}
        self.disruption = {}
        self.dis_range = range(1440)
        self.args.weight = 1

        # Time Parameters from args
        self.window_len = self.args.window_len
        self.date = pd.Timestamp(self.args.date).date()
        self.date_str = self.date.strftime('X%d/X%m/%Y').replace('X0','X').replace('X','').replace('/','-')
        self.month = self.date.month
        self.month_name = calendar.month_name[self.month]
        self.start_stamp = pd.Timestamp(self.args.date) + pd.Timedelta(minutes=self.args.start_stamp)

        # Area related Parameters
        self.area = "heroya"
        self.heroya_lonlat = (9.6301312,59.1317741)
        self.important_terminals = json.load(open('data\hip\important_terminals.txt'))
        if os.path.exists("data/hip/industrial_areas.pkl"):
            with open("data/hip/industrial_areas.pkl", 'rb') as file:
                self.industrial_areas = pickle.load(file)
        else:
            self.industrial_areas = find_industrial_areas('ORS/heroya-oslo.osm.pbf', output=False)
        self.prob_terminal = assign_terminals_to_origins(self.important_terminals, self.industrial_areas, self.client,weight=float(self.args.weight),save=True)
        self.prob_terminal = self.prob_terminal.drop_duplicates(subset='name', keep="first").reset_index(drop=True)
        self.daily_arrivals = process_hip_data(self.date,self.prob_terminal,self.args)#.set_index('Name', inplace=True)
        self.ODs = pd.read_csv(f"data/hip/OD.csv",encoding= 'iso-8859-1',index_col=2)
        self.windows = self.daily_arrivals.columns.tolist()
  
        # Align demand generation with inputs 
        self.args.N_TRUCKS = int(self.daily_arrivals.sum().sum())
        self.args.N_WINDOWS = int(len(self.windows))
        self.args.N_COMPANIES = int(self.daily_arrivals.shape[0])
        args = get_parameters(args)
        self.args.company_trucks =  {key: [] for key in self.args.companies}
        

        # with open('data/trucks/company_trucks.json', 'r') as json_file:
        #     self.args.company_trucks = json.load(json_file)

        # with open('data/trucks/WTP_C.json', 'r') as json_file:
        #     self.args.WTP_C = json.load(json_file)

        # # Convert lists back to NumPy arrays
        # for key, value in self.args.WTP_C.items():
        #     self.args.WTP_C[key] = np.array(value)
        
        # print(self.args.N_COMPANIES,len(self.args.WTP_C.keys()))

        # with open('data/trucks/CC.json', 'r') as json_file:
        #     self.args.CC = json.load(json_file)

        # self.args.WTP_C = {int(key): value for key, value in self.args.WTP_C.items()}
        # self.args.CC = {int(key): value for key, value in self.args.CC.items()}

        # Fix traffic based on input 
        c = 1
        red_trucks = copy.deepcopy(self.args.trucks)
        for _, r in self.daily_arrivals.iterrows():
            self.args.company_trucks[c] = random.sample(red_trucks, int(sum(r)))

            for element in self.args.company_trucks[c]:
                red_trucks.remove(element)
            weights = [round(random.uniform(10, 15)*x/sum(r.tolist())) for x in r.tolist()]

            # if self.args.flexibility:
            #     # Find the maximum value
            #     max_value = max(weights)

            #     # Find indices of all occurrences of the maximum value
            #     max_indices = [i for i, v in enumerate(weights) if v == max_value]

            #     # Check if adjacent values are zero and add half of the maximum value
            #     for max_index in max_indices:
            #         if max_index > 0 and weights[max_index - 1] == 0:
            #             weights[max_index - 1] += 0.5 * max_value

            #         if max_index < len(weights) - 1 and weights[max_index + 1] == 0:
            #             weights[max_index + 1] += 0.5 * max_value
            self.args.WTP_C[c] = np.array(weights)
            self.args.WTP_C[c] = np.ceil(self.args.WTP_C[c]).astype(int)
            c+=1
        

        # Update args and jobs
        args = company_demand(args)
        self.jobs = demand_to_jobs(self.daily_arrivals,self.ODs,self.heroya_lonlat,client,args)

        # Keep track of vessel/train trucks
        self.dict_v = {key: len(values) for key, values in self.args.vessel_trucks.items()}
        self.dict_t = {key: len(values) for key, values in self.args.train_trucks.items()}


        # Data Collector
        self.schedule = mesa.time.RandomActivationByType(self)
        self.datacollector = mesa.DataCollector(
            model_reporters={"TRIP_STARTS": lambda m: sum([1 for a in m.schedule.agents_by_type[Truck].values() if a.TRAVEL_TIME_CAR!= None and a.activation==m.schedule.steps]),
                            "TRIP_ENDS": lambda m: sum([1 for a in m.schedule.agents_by_type[Truck].values() if a.E_TIME.hour*60+a.E_TIME.minute==m.schedule.steps])
                             },
            agent_reporters={"END_TIME": lambda a: self.get_KPI(a,"E_TIME"),"TERMINAL": lambda a: self.get_KPI(a,"terminal"),"CAV": lambda a: self.get_KPI(a,"CAV"),
                             "ACCESS_TIME": lambda a: self.get_KPI(a,"TRAVEL_TIME_CAR"),"ACCESS_DISTANCE": lambda a: self.get_KPI(a,"DISTANCE_CAR"),
                             "DETOUR": lambda a: self.get_KPI(a,"detour"),
                             "COMPANY": lambda a: self.get_KPI(a,"company"),"ACTIVATION": lambda a: self.get_KPI(a,"D_TIME"),
                             "WTP": lambda a: self.get_KPI(a,"wtp"),"WAITING_TIME": lambda a: self.get_KPI(a,"w_time"),"PLATOON": lambda a: self.get_KPI(a,"platoon")
                             }  
        )    
    def place_truck_agent(self):

        '''
        Place agent

        :
        '''           
        for idx,val in self.jobs.items():
            if val[4] == self.schedule.steps:             
                truck = Truck(val[1],self,val[0],val[2],val[3],val[4],val[5],val[6],val[7])
                self.schedule.add(truck)

    def step(self):
        '''
        Step Function that updates simulation by the minute
        '''   
        self.schedule.steps += 1 #important for data collector to track number of steps
    
    def get_KPI(self,agent,KPI):

        '''
        For agent reporters in data collector

        :param agent (mesa.Agent) : The specific Passenger Agent that we request the KPI for
        :param KPI (str): The name of requested KPI ; consult documentation for proper usage

        return attribute value of requested KPI may be numeric, datetime or string
        '''

        if type(agent) == Truck:
            attr = getattr(agent, KPI)
            try:
                if KPI != "MULTIMODAL_EFFECT" and KPI!= "IR_RATIO":
                    attr = round(attr)
            except:
                pass
            return attr

    def get_fig(self):
        """
        Generate a bar graph representing truck arrivals per time window.

        """
        # Combine the sums into a new DataFrame
        sum_df = pd.DataFrame({self.flag1: self.sum_df1, 'Simulated': self.sum_df2})
        start_time = datetime.strptime("06:00", "%H:%M")
        end_time = datetime.strptime("07:00", "%H:%M")
        time_periods = [start_time + timedelta(hours=i) for i in range(len(sum_df))]

        # Create a Figure and Axes
        fig, ax = plt.subplots(figsize=(9, 3))

        # Plotting the bar graph with custom x-axis ticks
        sum_df.plot(kind='bar', ax=ax)
        ax.set_xticks(range(len(sum_df)))
        ax.set_xticklabels([f"{time.strftime('%H:%M')}" for time in time_periods], rotation=20, ha='right')  # Rotate ticks by 45 degrees
        ax.set_title('Arrivals per Time-Window')
        ax.set_xlabel('Time Window')
        ax.set_ylabel('Truck Arrivals')

        # Create a FigureCanvasAgg object, which will render the figure to a Pygame surface
        canvas = FigureCanvasAgg(fig)
        canvas.draw()

        # Convert the rendered figure to a Pygame surface
        renderer = canvas.get_renderer()
        raw_data = renderer.tostring_rgb()
        size = canvas.get_width_height()
        pygame_surface = pygame.image.fromstring(raw_data, size, "RGB")
        self.visualization['graph'] = pygame_surface
        plt.close(fig)  # Close the Matplotlib figure
        del renderer    # Delete the renderer object
        del canvas 
    @profile
    def run_model(self, step_count):
        '''
        Runs model based on the current step

        :step_count (int): total steps in the simulation
        :start (int): flag to ignore assignment on steps before that 
        '''
        if self.args.tactical:
            self.args.orig_quota = self.daily_arrivals.sum().tolist()
            

            if self.args.seed <100:
                

                actors = ["terminal","vessels",'trains','road','logistics']
                permutations = list(itertools.permutations(actors, len(actors)))
                sols,quotas_save,truck_assignment,self.visualization,trucks_store = tactical(permutations,self)
                
                # shared_keys = set(sols.keys()) & set(trucks_store.keys())
                # sols = {key: sols[key] for key in shared_keys}

                for key,val in trucks_store.items():
                    if len(val.keys()) != self.args.N_TRUCKS_TRAINS + self.args.N_TRUCKS_VESSELS:
                        del sols[key]

                mod_sols,min_stats,min_key_m = get_best_sol(sols,actors,self)
                trucks_store = trucks_store[min_key_m]

            else:
                with open(f'data/trucks/quota.json', 'r') as json_file:
                    quota = json.load(json_file)
                quota = [int(x) for x in quota]
                actors = ["vessels",'trains',"terminal",'road','logistics']
                permutations = list(itertools.permutations(actors, len(actors)))
                sols,quotas_save,truck_assignment,self.visualization,trucks_store = tactical_from_quota(permutations,self,quota)
                for key,val in trucks_store.items():
                    if len(val.keys()) != self.args.N_TRUCKS_TRAINS + self.args.N_TRUCKS_VESSELS:
                        del sols[key]

                mod_sols,min_stats,min_key_m = get_best_sol(sols,actors,self)
                trucks_store = trucks_store[min_key_m]

            df = pd.DataFrame(truck_assignment[min_key_m]).transpose()
            df.set_index(self.daily_arrivals.index, inplace=True)
            df.columns = self.daily_arrivals.columns

            # Calculate the sum of columns for each DataFrame
            self.sum_df1 = df.sum()
            self.sum_df2 = pd.Series(0, index=self.sum_df1.index)
            self.flag1 = "Tactical"
            self.daily_arrivals = df

            if self.args.seed == 42:
                magic_quota = self.daily_arrivals.sum().tolist()
                with open(f'data/trucks/quota.json', 'w') as json_file:
                    json.dump(magic_quota, json_file)
            self.jobs = demand_to_jobs(self.daily_arrivals,self.ODs,self.heroya_lonlat,self.client,self.args,trucks_store)
            
        else:
            self.counter = 0
            self.save = {}
            self.flag1 = "Uncoordinated"
            self.sum_df1 = self.daily_arrivals.sum()
            self.sum_df2 = pd.Series(0, index=self.sum_df1.index)   
            print(self.sum_df1)   


        for st in range(step_count):
            self.place_truck_agent()
            pass_shuffle = list(self.schedule.agents_by_type[Truck].values())

            if self.schedule.steps > self.dis_range[-1]:
                self.dis_range = range(1440)
                self.visualization['disruption'] = {}
            
            for key in self.disruptions.keys():
                if self.disruptions[key]['active'] and st in self.disruptions[key]['period']:
                    self.disruption = self.disruptions[key]
                    if self.disruptions[key]['period'][0] > self.dis_range[0] and self.disruptions[key]['period'][-1] < self.dis_range[-1]:
                        self.dis_range = self.disruptions[key]['period']

            for agent in pass_shuffle:
                if self.schedule.steps == agent.activation:
                    agent.assign()

                if agent.ARR_STEP > self.schedule.steps and self.schedule.steps == self.dis_range[0]:
                    d1 = agent.ARR_STEP
                    point = int((1 - (agent.ARR_STEP - self.schedule.steps) / (agent.ARR_STEP - agent.activation)) * len(agent.pl)/1.5)
                    agent.lonlat = (agent.pl[point][1], agent.pl[point][0]) if agent.pl[point][1] < agent.pl[point][0] else agent.pl[point]
                    agent.detour = True
                    agent.assign()
                    self.delay_dis += max(0, agent.ARR_STEP - d1)
                    self.affected_trucks += 1
                    self.visualization['disruption']['active'] = f"Delays on {self.disruption['name']}"
                    self.visualization['disruption']['affected'] = f"Affected Trucks en-route : {self.affected_trucks}"
                    self.visualization['disruption']['delay'] = f"Average delay : {round(self.delay_dis/self.affected_trucks)}"

                if agent.ARR_STEP == self.schedule.steps:
                    self.arrived += 1

                    val = None
                    for truck_type, type_trucks in self.args.vessel_trucks.items():
                        if agent.unique_id in type_trucks:
                            val = truck_type
                            key_v = f"v{val}"
                            self.visualization['operational'][key_v] = f" Vessel {val} : {self.dict_v[val] - 1} trucks remaining"
                            self.dict_v[val] -= 1
                            break

                    for truck_type, type_trucks in self.args.train_trucks.items():
                        if agent.unique_id in type_trucks:
                            val = truck_type
                            key_t = f"t{val}"
                            self.visualization['operational'][key_t] = f" Train {val} : {self.dict_t[val] - 1} trucks remaining"
                            self.dict_t[val] -= 1
                            break

            if self.schedule.steps not in self.dis_range:
                self.visualization['disruption'] = {}

            if self.schedule.steps % 15 == 0:
                examined_agents = self.datacollector.get_agent_vars_dataframe()
                check_for_nan = examined_agents.isnull().values.any()

                if not(examined_agents.empty) and not(check_for_nan):
                    examined_agents = examined_agents.reset_index(level=['Step', 'AgentID'])
                    examined_agents = examined_agents[examined_agents["Step"] == self.schedule.steps - 1].drop(columns=['Step']).set_index("AgentID")
                    examined_agents["time"] = 60 * examined_agents['END_TIME'].dt.hour + examined_agents['END_TIME'].dt.minute
                    examined_agents = examined_agents[examined_agents["time"] >= self.schedule.steps]
                    examined_agents = examined_agents[examined_agents["time"] <= self.schedule.steps + 15]
                    bundles = bundling(examined_agents, self.schedule.steps,self.args.max_platoon_size+1)
                    self.occupied = scheduling(self, examined_agents, bundles, steps=range(self.schedule.steps, self.schedule.steps + 90), cavs=range(0, self.args.max_cavs + 1), destinations=range(1, 6), occupied=self.occupied)
                    try:
                        self.get_fig()
                    except:
                        pass
            self.counter +=1
            self.save[self.counter] = self.visualization
            self.datacollector.collect(self)
            self.step()