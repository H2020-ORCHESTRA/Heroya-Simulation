import pandas as pd
import numpy as np
import random
import osmium as o
from sklearn.neighbors import NearestNeighbors
import shapely.wkb as wkblib
import random
import copy
import numpy as np

def find_industrial_areas(osm_file_path, output=False):
    """
    Find and return industrial areas from an OSM file.

    Args:
        osm_file_path (str): The path to the OSM file in PBF format.
        output (bool): If True, print industrial areas; otherwise, return them.

    Returns:
        list: A list of tuples containing industrial area information (lon, lat, landuse, name).
            If output is True, returns an empty list.
    """
    # Create a WKBFactory for handling geometries
    wkbfab = o.geom.WKBFactory()

    class AmenityListHandler(o.SimpleHandler):
        def __init__(self,key,factor) -> None:
            super().__init__()
            self.key = key
            self.factor = factor
            self.save_industrial = []

        def save_amenity(self, tags, lon, lat):
            name = tags.get('name', '')
            if tags[self.key] == self.factor:
                self.save_industrial.append((lon, lat,tags[self.key],name))

        def node(self, n):
            if self.key in n.tags:
                self.save_amenity(n.tags, n.location.lon, n.location.lat)

        def area(self, a):
            if self.key in a.tags:
                try:
                    wkb = wkbfab.create_multipolygon(a)
                    poly = wkblib.loads(wkb, hex=True)
                    centroid = poly.representative_point()
                    self.save_amenity(a.tags, centroid.x, centroid.y)
                except: 
                    pass

    key = 'landuse'
    factor = 'industrial'
    handler = AmenityListHandler(key,factor)

    handler.apply_file(osm_file_path)

    return handler.save_industrial


def assign_terminals_to_origins(important_terminals, industrial_data, client,hip_point = (9.6301312, 59.1317741),weight=3,save=False):
    """
    Assigns industrial terminals to the nearest origins based on latitude and longitude.

    Parameters:
    important_terminals (dict): A dictionary containing terminal names as keys and their coordinates as values.
    industrial_data (list): A list of industrial data containing tuples with latitude, longitude, and other information.
    client: A client object for the mapping service
    hip_point (tuple) : The coordinates for HIP

    Returns:
    pd.DataFrame: A DataFrame with origins assigned to their nearest terminal, including terminal names and additional information.
    """

    list_of_origins = industrial_data.copy()
    points = [(x[1],x[0]) for x in list_of_origins]
    points_df = pd.DataFrame.from_records(points, columns =['lat', 'long'])

    nn = NearestNeighbors(metric="haversine")
    nn.fit(points_df)

    for key,value in important_terminals.items():
        point = pd.DataFrame({"lat": [value[0]], "long": [value[1]]})
        nearest = nn.kneighbors(point, n_neighbors=1, return_distance=True)
        list_of_origins[nearest[1][0][0]] = (value[0],value[1],'industrial',key)
        
    # Assign name to Terminal
    points_df['name'] = [x[-1] for x in list_of_origins]
    p = points_df[points_df['name'] != '']
    p[p['name'] != 'HIP']
    p = p.reset_index(drop=True)

    c = 0
    p['DISTANCE'] = 0
    p['LIKELIHOOD'] = 0
    for idx, r in p.iterrows():
        coords = ((r[1],r[0]),hip_point)
        try:
            CAR_ROUTE = client.directions(coords,radiuses=500)
            p.at[idx, "DISTANCE"] = CAR_ROUTE['routes'][0]['summary']['distance']//1000
            p.at[idx, "LIKELIHOOD"] = 1/(p.at[idx, "DISTANCE"])
            if r[2] in important_terminals.keys():
                p.at[idx, "LIKELIHOOD"] = weight
        except:
            p.at[idx, "DISTANCE"] = 0   

    p = p[p['DISTANCE'] != 0]
    p = p[p['DISTANCE'] >= 10]
    p['LIKELIHOOD']=  p['LIKELIHOOD']/p['LIKELIHOOD'].sum()
    p = p.reset_index(drop=True)

    p = p.sort_values(by='DISTANCE')

    p = p.head(40)


    if save == True:
        p.to_csv('data/hip/OD.csv',encoding='iso-8859-1',index=False)
    return p

def process_hip_data(date,p,args):
    """
    Process HIP (Herøya Industripark) data for a specific date.

    Args:

    Returns:
        None
    """
    str_date = date.strftime('%d.%m.%Y')
    hip_data = pd.read_csv('data/hip/hip_data.csv',encoding= 'iso-8859-1')
    hip_data =  hip_data[hip_data["Dato"]==str_date]
    IN = np.zeros([p.values.shape[0],24])
    OUT = np.zeros([p.values.shape[0],24])

    for window in range(24):
        time =  pd.Timedelta(hours = window)
        str_time = f'{int(time.seconds/3600)}:00'
        row_data = hip_data[hip_data["Til tidspunkt"] == str_time]
        row_data_inc = row_data[row_data["Felt"] == "Totalt i retning Herøya industripark"]['>=7.6m'].values
        row_data_out=  row_data[row_data["Felt"] == "Totalt i retning Kulltangen"]['>=7.6m'].values
        if row_data_inc.size > 0 :

            c = random.choices(p.index.to_list(),  weights  = p['LIKELIHOOD'].to_list(),k = int(row_data_inc[0]))
            for s in c:
                IN[s,window] +=1

        if row_data_out.size > 0:
            c = random.choices(p.index.to_list(),  weights  = p['LIKELIHOOD'].to_list(),k = int(row_data_out[0]))
            for s in c:
                OUT[s,window] +=1

    df = pd.DataFrame(IN, columns = list(range(24)), index = p.name.to_list() )
    #df2 = pd.DataFrame(OUT, columns = list(range(24)), index = p.name.to_list() )
    
    h_range = list(range(args.start_stamp//60,(args.start_stamp+ int(args.simulated_time))//60+1))

    a = df  
    a = df[h_range]
    a = a.sort_values(by=h_range,ascending=False)
    a = a[(a.T != 0).any()]
    return a

def demand_to_jobs(arrivals,ODs,heroya_lonlat,client,args,trucks_store = {}):
    
    '''
    Generates exact trip characteristics per cluster based on the generate_demand provided information:

    Args:
        date (str) : date in the (%d-%m-%YYYY) format
        cluster (int) : Examined time-window
        demand (csv file): Dataframe that is direct output of the generate_demand function.;
                            File should be located in  data/milano/<examined month (Name)>/<examined date (%d-%m-%YYYY)>/demand_<examined cluster>.csv 
        passenger_data (csv file): Dataframe read that should be stored in path data/mxp/<examined month (Name)>/<examined date (%d-%m-%YYYY)>/trip_matrix_milano.csv;
                                    derived from flights_to_demand.ipynb\
        polygon (GeoJSON) : Set of polygons describing the spatial characteristics of the NILS; derived from the link below :
                                    https://dati.comune.milano.it/dataset/ds964-nil-vigenti-pgt-2030/resource/9c4e0776-56fc-4f3d-8a90-f4992a3be426 
        no_persons (list) : The possible group size of all trips
        save (bool) : Optional parameter to save all resulting Dataframe

    Returns:
        user_df (pd.Dataframe): Dataframe containing exact information per initiated trip for that time-window.

    '''

    jobs = {}
    c = 0
    c1 = 1 

    def sort_dict_demand(data,c,w):
        result = {}
        for (x, y), z in data.items():
            if y == c and z == w:
                key = (y, z)
                if key in result:
                    result[key].append(x)
                else:
                    result[key] = [x]

        return result

    for idx,r in arrivals.iterrows():
        

        #Assign Job Company and Type 
        jobs_c = [idx for x in args.company_trucks[c1]]
        jobs_c_index = [c1 for x in args.company_trucks[c1]]

        jobs_w = {}
        for w in range(1,13):
            jobs_w[w] = sort_dict_demand(trucks_store,c1,w).values()
            if len(jobs_w[w]) == 1:
                jobs_w[w] =  list(jobs_w[w])[0]
        flattened_list = [element for sublist in jobs_w.values() for element in sublist]

        jobs_t = []#args.company_trucks[c1] 
        jobs_cav = [1 if x in args.company_cav_trucks[c1] else 0 for x in args.company_trucks[c1]]


        # Determine Time to Heroya
        coords =  ((ODs[ODs.index == idx]['long'].values[0],ODs[ODs.index == idx]['lat'].values[0]),heroya_lonlat)
        CAR_ROUTE = client.directions(coords,radiuses=500)
        time =  CAR_ROUTE['routes'][0]['summary']['duration']//60
        jobs_time =  [time] * int(sum(r))

        jobs_activation = []
        jobs_t = []
        for w, val in enumerate(r.tolist()):
            for x in range(int(val)):
                jobs_activation.append(arrivals.columns[w]*60 + random.randrange(1,60)-time)
    
            if len(jobs_w[w+1]) < val:    
                sampled = random.sample([elem for elem in args.company_trucks[c1]  if elem not in flattened_list+jobs_t], k=int(val)-len(jobs_w[w+1]))
                merged_list = list(jobs_w[w+1]) + sampled
                jobs_t = jobs_t + merged_list
            else:
                jobs_t = jobs_t + list(jobs_w[w+1])
        
        TERMINALS = 6 
        for j,_ in enumerate(jobs_c):

            jobs[c+j] = [jobs_c[j],jobs_t[j],jobs_cav[j],jobs_time[j],jobs_activation[j], random.randrange(1,TERMINALS),coords[0],jobs_c_index[j]]

        c += int(sum(r))
        c1+=1

    return jobs


def randomize_dict(my_dict, sample_list):
    """
    Randomly assign values from sample_list to the values of my_dict.

    Args:
        my_dict (dict): The dictionary with lists as values to be filled with values from sample_list.
        sample_list (list): The list of values to be assigned randomly to the dictionary values.

    Returns:
        dict: The modified dictionary with randomized values in its lists.
    """

    # Create a deep copy of sample_list to avoid modifying the original list
    sample = copy.deepcopy(sample_list)

    # Iterate over the dictionary and assign values randomly from the shuffled list
    for key in my_dict:
        if sample:
            random_element = random.choice(sample)
            my_dict[key].append(random_element)
            sample.remove(random_element)

    # If there are remaining elements in sample, assign them randomly to dictionary values
    while sample:
        random_element = random.choice(sample)
        random_index = random.choice(list(my_dict.keys()))
        my_dict[random_index].append(random_element)
        sample.remove(random_element)

    return my_dict

def get_parameters(args,thres=3,max_wtp = 15,max_cc = 1):

    """
    Randomly determines parameters relating to vessels, trucks and trains such as Upper Bounds, Willingness-to-pay and Congestion costs.

    Args:
        args (args): Arguments passed from parser.
        thres (int): Minimum threshold between upper and lower bound
        max_wtp (int) : Maximum willingness to pay for a slot
        max_cc(int) : Maximum possible congestion cost for a slot slot

    Returns:
        args : The updated args with all modified parameters
    """

    args.wtps = [random.randint(0, max_wtp) for x in range(int(args.N_WINDOWS)*int(args.N_COMPANIES))] # Willingness to pay
    args.thres = thres

    # Initialize sets
    args.windows = list(range(1,int(args.N_WINDOWS)+1))
    args.companies = list(range(1,int(args.N_COMPANIES)+1))
    args.trucks = list(range(1,int(args.N_TRUCKS)+1))
    args.vessels = list(range(1,int(args.N_VESSELS)+1))
    args.trains = list(list(range(1,int(args.N_TRAINS)+1)))

    # Vessel Parameters
    items = copy.deepcopy(args.trucks)
    random.shuffle(items)
    args.trucks_v_list = items[:int(args.N_TRUCKS_VESSELS)]
    args.vessel_trucks =  {key: [] for key in args.vessels}
    args.vessel_trucks = randomize_dict(args.vessel_trucks,args.trucks_v_list)
    args.vessel_trucks_traffic = [-len(x) for x in args.vessel_trucks.values()]
    
    # CAV dependent trucks
    random.shuffle(items)
    args.trucks_cav_list = items[:int(args.N_TRUCKS_CAVS)]

    # Initialize UB_vessels and LB_vessels
    args.LB_vessels = {key: args.windows[0] for key in args.vessels}
    args.UB_vessels = {key: args.windows[-1] for key in args.vessels}
    for key, value in args.LB_vessels.items():
        args.LB_vessels[key] = random.randint(1,3)
        args.UB_vessels[key] = random.randint(10,12)

    args.LB_vessels[1] = 1
    args.UB_vessels[1] = 7

    args.LB_vessels[2] = 3
    args.UB_vessels[2] = 12

    # Train Parameters
    reduced_trucks = list(set(args.trucks) - set(args.trucks_v_list))
    random.shuffle(reduced_trucks)
    args.trucks_t_list = reduced_trucks[:int(args.N_TRUCKS_TRAINS)]
    args.train_trucks =  {key: [] for key in args.trains}
    args.train_trucks = randomize_dict(args.train_trucks,args.trucks_t_list)
    args.train_trucks_traffic = [-len(x) for x in args.train_trucks.values()]   

    # Initialize UB_vessels and LB_vessels
    args.LB_trains = {key: args.windows[0] for key in args.trains}
    for key, value in args.LB_trains.items():
        lb = random.randint(args.windows[0],args.windows[int(int(args.N_WINDOWS)/3)])
        args.LB_trains[key] = lb
        args.LB_trains[key] = 2 # Custom

    # Initialize WTPs
    args.WTP_C = {key: [] for key in args.companies}
    args.CC = {key: [] for key in args.companies}
    for key, value in args.WTP_C.items():
        for w in range(int(args.N_WINDOWS)):
            if args.wtps:
                value.append(args.wtps.pop())
        args.CC[key] = random.random() # Cost of congestion

    # MAX selected value by a population
    args.INT_MIN,args.INT_MAX = 0,33
    return args

def company_demand(args):

    args.company_cav_trucks =  {key: [] for key in args.companies}
    for t in args.trucks_cav_list:
        for c in args.company_cav_trucks.keys():
            if t in args.company_trucks[c]:
                args.company_cav_trucks[c].append(t)
                break

    args.trucks_per_company_vessel = {}
    args.trucks_per_company_train = {}
    
    # Determine trucks each company has on a vessel
    for t in args.company_trucks.items():
        for v in args.vessel_trucks.items():
            intersection = set(t[1]).intersection(v[1])
            args.trucks_per_company_vessel[(t[0],v[0])] = intersection

    # Determine trucks each company has on a train
    for t in args.company_trucks.items():
        for v in args.train_trucks.items():
            intersection = set(t[1]).intersection(v[1])
            args.trucks_per_company_train[(t[0],v[0])] = intersection

    return args

