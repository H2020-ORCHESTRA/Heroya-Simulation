##################################################################### Dynamic Guidance for incoming agents ##################################################################
import polyline
import pandas as pd

def get_road_route(client,origin,destination,disruption):
    '''
    Returns optimal route based on existing network status

    Args:
        client (openrouteservice.client) : The loaded local instance of the ORS router
        origin (list): Starting position for the route in the (long,lat) format
        destination (list) : Finishing position for the route in the (long,lat) format
        disruption (dict) : Disruption affecting the examined route

    Returns:
        route (dict) : openrouteservice request
    '''

    request_params = {'coordinates': [origin,destination],
                    'format_out': 'geojson',
                    'profile': 'driving-car',
                    'preference': 'recommended',
                    'attributes' : ['avgspeed'],
                    'instructions': True}
    
    base_route = client.directions(**request_params)
    request_params['options'] = {'avoid_polygons': disruption['avoid_polygon']}
    while True:
        try:
            avoid_route = client.directions(**request_params)
            break
        except:
            request_params['coordinates'][0][0] +=0.001
            request_params['coordinates'][0][1] +=0.001

    if  disruption['perc_road'] == 0 :
        return base_route
    
    if disruption['slow_down'] == False:
        request_params['options'] = {'avoid_polygons': disruption['avoid_polygon']}
        avoid_route = client.directions(**request_params)   
        return avoid_route
    
    if disruption['slow_down'] == True:
        slow_route = base_route
        segments = base_route['features'][0]['properties']['segments'][0]['steps']
        for edge in segments:
            if edge['name'] in disruption['edges_speed'].keys():
                edge['duration'] = disruption['edges_speed'][edge['name']]
        
        slow_route['features'][0]['properties']['segments'][0]['steps'] = segments
        slow_route['features'][0]['properties']['summary']['duration'] = sum([x['duration'] for x in segments])

        if slow_route['features'][0]['properties']['summary']['duration'] < avoid_route['features'][0]['properties']['summary']['duration']:
            return slow_route
        else:
            return avoid_route

def get_route_to_list(route):
    '''
    Returns route in a list coordinates format for folium visualization 

    Args:
        route (dict) : openrouteservice request

    Returns:
        loc (list) : list of coordinates
    '''

    loc = route['features'][0]['geometry']['coordinates']
    loc = [(x[1],x[0]) for x in loc]
    return loc

def extract_KPIS(model,agent,CAR_ROUTE,TRANS_ROUTE=None,TRANSFERS=None):

    # Compute Travel Time and Distance by Car
    
    if model.disruption != {} and model.schedule.steps in model.dis_range:
        CAR_ROUTE['routes'] = [{}]
        CAR_ROUTE['routes'][0]['summary'] = CAR_ROUTE['features'][0]['properties']['summary']
        pl = CAR_ROUTE['features'][0]['geometry']['coordinates']
    else:
        pl = polyline.decode(CAR_ROUTE['routes'][0]['geometry'])
    agent.pl = pl
    agent.TRAVEL_TIME_CAR = CAR_ROUTE['routes'][0]['summary']['duration']/60
    agent.DISTANCE_CAR = CAR_ROUTE['routes'][0]['summary']['distance']//1000

    # Define Approach of Car route
    pl = [x[0] for x in pl]
    if max(pl) == pl[-1]:
        agent.APPROACH = "SOUTH"
    else:
        agent.APPROACH = "NORTH"

    # Compute Start of Trip and 
    agent.E_TIME = agent.D_TIME + pd.Timedelta(minutes=agent.TRAVEL_TIME_CAR).ceil("T")
    agent.ARR_STEP = agent.E_TIME.hour * 60 + agent.E_TIME.minute
    win = int((agent.ARR_STEP-model.args.start_step)//60)
    if win >= 12:
        win = 11
    agent.wtp = model.args.WTP_C[agent.c_index][win]
    
def return_car_route(client,coords,radius):
    while True:
        try:
            CAR_ROUTE = client.directions(coords,radiuses=radius)
            break
        except:
            radius +=250
    return CAR_ROUTE