##################################################################### Network Representations for Use-Cases ##################################################################

import subprocess
import requests
from shapely.geometry import LineString,MultiLineString
from shapely.ops import unary_union,linemerge
import json
import geopandas as gpd
import os
import pandas as pd
from pyproj import Geod
import folium
from time import sleep
import webbrowser
from datetime import datetime

def open_ORS():
    '''
    Process to turn on the OpenRouteService API
    '''
    print("Started loading ORS")
    subprocess.Popen('wsl ~ docker compose -f ~/heroya/docker-compose.yml up',stdout=subprocess.DEVNULL,stderr=subprocess.STDOUT, shell=True)

    # Making a GET request to ensure that ORS is up
    while True:
        try:
            r = requests.get('http://localhost:8080/ors')
            break
        except:
            pass

    print("ORS is up")

def close_ORS():
    '''
    Process to turn off the OpenRouteService API
    '''
    subprocess.Popen('wsl ~ docker compose -f ~/heroya/docker-compose.yml down',stdout=subprocess.DEVNULL,stderr=subprocess.STDOUT, shell=True)
    print("ORS is closed")

def get_road_geometry(client,road_name,start,finish,save=True):
    '''
    Process to get the polyline for an examined route. 

    Args:
        client (openrouteservice.client) : The loaded local instance of the ORS router
        road_name (str) : Examined road name as listed in the CSV on the data/disruptions/roads.csv file
        start (list): Starting position for the route in the (long,lat) format
        finish (list) : Finishing position for the route in the (long,lat) format

    Returns:
        polyline (shapely.LineString) : Polyline form in the examined route
    '''

    # Call Router
    request_params = {'coordinates': [start,finish],
                    'format_out': 'geojson',
                    'profile': 'driving-car',
                    'preference': 'recommended',
                    'attributes' : ['avgspeed'],
                    'instructions': True}
    
    route_directions = client.directions(**request_params)
    loc = route_directions['features'][0]['geometry']['coordinates']

    # Reverse x,y because of ORS
    loc = [(x[1],x[0]) for x in loc]
    polyline = LineString(loc)

    # Convert the geometry to a GeoJSON-like dictionary
    geometry_dict = polyline.__geo_interface__

    # Save the dictionary to a GeoJSON file
    if save:
        output_file = f"data/disruptions/{road_name}.geojson"
        with open(output_file, "w") as f:
            json.dump(geometry_dict, f)

        output_file = f"data/disruptions/{road_name}_speed.json"
        speed = route_directions['features'][0]['properties']['segments'][0]['steps']
        with open(output_file, "w") as f:
            json.dump(speed, f)        
    return polyline,speed

def get_disruption_from_file(df,road_name):
    '''
    Process to read disruption from file. 

    Args:
        df (pd.DataFrame) : Dataframe stored in the data/disruptions/roads.csv file
        road_name (str) : Examined road name as listed in the CSV on the data/disruptions/roads.csv file

    Returns:
        start : List of coordinates in the long,lat format (starting position)
        finish : List of coordinates in the long,lat format (starting position)
    '''
    ex_road = df[df['name'] == road_name]
    start,finish = ex_road.iloc[0][1],ex_road.iloc[0][2]
    start,finish = list(start.split(",")),list(finish.split(","))
    start,finish = [float(x) for x in start],[float(x) for x in finish]

    return start,finish

def segments(curve):
    '''
    Splits LineString to multiple linestrings

    Args:
        curve (shapely.LineString) : Polyline of a route

    Returns:
        List of linestrings
    '''
    return list(map(LineString, zip(curve.coords[:-1], curve.coords[1:])))


def insert_disruption(client,road_name,perc = 1,slow_down = False,slow_perc = 2):
    '''
    Process to get the polyline for an examined route. 

    Args:
        client (openrouteservice.client) : The loaded local instance of the ORS router
        road_name (str) : Examined road name as listed in the CSV on the data/disruptions/roads.csv file
        perc (float) : Percentage of the road that is affected

    Returns:
        polygon_avoid : List of coordinates forming a polygon in the long,lat format
        polygon_plot : List of coordinates forming a polygon in the lat,long format
    '''

    if os.path.exists(f"data/disruptions/{road_name}.geojson"):
        polyline = gpd.read_file(f"data/disruptions/{road_name}.geojson")
        polyline = polyline.iloc[0][0]
        speed = json.load(open(f'data/disruptions/{road_name}_speed.json'))
        print("Loaded disruption from GeoJSON")
    else:
        roads = pd.read_csv("data/disruptions/roads.csv")
        ex_road = roads[roads['name'] == road_name]
        if ex_road.empty:
            print('No such road on file - No Disruption applied')
            return None
        else:
            start,finish = get_disruption_from_file(roads,road_name)        
            polyline,speed = get_road_geometry(client,road_name,start,finish,save=True)
    
    if polyline != None:
        
        # Reduce from start up to examined percentage
        seg = segments(polyline)
        seg = MultiLineString(seg[0:int(perc*len(seg))])
        reduced_polyline = linemerge(seg)

        # Create a buffer around the polyline
        buffer_distance = 1e-5 # Adjust this value to control the buffer size
        buffered_polyline = reduced_polyline.buffer(buffer_distance)

        # If you want to handle multiple buffers as a single geometry (e.g., for overlapping buffers)
        buffered_union = unary_union(buffered_polyline)
        xx, yy = buffered_union.exterior.coords.xy      
        polygon_avoid = [[y,x] for x,y in zip(xx,yy)]
        polygon_plot = [[[x,y] for x,y in zip(xx,yy)]]

        # Print Geodesic area covered 
        geod = Geod(ellps="WGS84")
        area = abs(geod.geometry_area_perimeter(buffered_union)[0])
        #print('# Geodesic area: {:.3f} m^2'.format(area))
        
        if slow_down == False:
            return polygon_avoid,polygon_plot,{}
        else:
            edge_speed_keys = [x['name'] for x in speed]
            edge_speed_values = [slow_perc*perc*x['duration']+(1-perc)*x['duration'] for x in speed]
            edge_speed = {index: element for index, element in zip(edge_speed_keys,edge_speed_values)}
            return polygon_avoid,polygon_plot,edge_speed    

def plot_map(heroya_point,disruption,lines=[]):

    '''
    Produces route visualizations. 

    Args:
        heroya_point (list) : Coordinates in the long,lat format
        disruption (list) : List of coordinates forming the polygon under disruption
        lines ([]) : List of routes to be visualized

    Returns:
    '''
    map = folium.Map(location=heroya_point,zoom_start=10)

    if disruption !=[]:
        folium.PolyLine(disruption,color='red',weight=10,opacity=0.8).add_to(map)

    for line in lines:
        folium.PolyLine(line,color='yellow',weight=5,opacity=0.8).add_to(map)

    map.save("map.html")
    webbrowser.get('windows-default').open("map.html")
    sleep(10)
    os.remove("map.html") 

def time_range_to_minutes(start_time, end_time):
    # Convert the times to datetime objects
    start_datetime = datetime.strptime(start_time, "%H:%M")
    end_datetime = datetime.strptime(end_time, "%H:%M")

    # Calculate the minutes from midnight
    start_minutes = start_datetime.hour * 60 + start_datetime.minute
    end_minutes = end_datetime.hour * 60 + end_datetime.minute

    return start_minutes, end_minutes

def read_disruptions(args,client):

    names = ["E18",'Rv32','Rv40']
    reds = [1/float(args.speed_reductionE18),1/float(args.speed_reductionRV32),1/float(args.speed_reductionRV40)]
    dis_ranges = [args.disruption_time_roadE18,args.disruption_time_roadRV32,args.disruption_time_roadRV40]
    dis_ranges = [x.split("-") for x in dis_ranges]
    percs = [0.5 if x > 1 else 0 for x in reds]
    active = [True if x > 1 else False for x in reds]
    slow = [True if x != 0 else False for x in reds]
    dis_ranges = [(0,1440) if x == [''] else time_range_to_minutes(x[0],x[1]) for x in dis_ranges]
    disruptions = {}
    idx = 1
    for x,y,z,a,b,c in zip(names,percs,slow,reds,dis_ranges,active):
        disruptions[idx] = {'name': x,'perc_road':y,"slow_down" : z, "slow_down_perc" : a,'period' :range(*b),"active": c}
        idx+=1

    for key in disruptions.keys():
        if disruptions[key]['perc_road'] !=0:
            dis_poly,dis_plot,disruptions[key]['edges_speed'] = insert_disruption(client,disruptions[key]['name'],disruptions[key]['perc_road'],disruptions[key]['slow_down'],disruptions[key]['slow_down_perc'])
            disruptions[key]['avoid_polygon'] = {"coordinates": [dis_poly],"type": "Polygon"}
    
    return disruptions
