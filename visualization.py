import numpy as np
import pygame as pg
from math import *
import os
import ctypes
import pandas as pd
import time as timer

# Initialize values
# User inputs (may be changed)
disp_truck_id = True  # display truck id number
disp_cav_id = True
disp_waypoint_id = True  # display waypoint id number
disp_radar_truck = False  # display which other truck a truck sees
disp_time = True  # display current time
disp_waypoint = True  # display dot of the waypoint
disp_vehicles = True

#Lines below are used in visualization, dont change them (line 26-46)
piclist = []  # list to store all pics
rectlist = []  # list to store the rectangular areas of the surface
carpiclist = []  # list to store all pics
carrectlist = []  # list to store the rectangular areas of the surface
squared_display = True  # create squared display
boundary_margin = 0.01  # add boundary margin to be sure that points are within boundary
screen_percentage = 0.94  # percentage of pixels used in the critical axis
horizontal_sep = 0.01  # percentage of pixels used in between dual screen

red = (255, 0, 0)  # define colors
lightgreen = (0, 255, 147)
green = (0, 255, 0)
darkgreen = (0, 135, 0)
yellow = (255, 251, 45)
orange = (244, 149, 66)
pink = (255, 192, 203)
purple = (255, 0, 255)

blue = (0, 0, 255)
black = (10, 10, 10)
white = (255, 255, 255)
lightblue = (173, 216, 230)

# Functions
def c2m_x(x_coord, min_x, reso_x, x_range, shift=0):  # function to convert x-coordinate location to pixel in screen
    return int((float(x_coord - min_x) / x_range) * reso_x + shift)

def c2m_y(y_coord, max_y, reso_y, y_range, shift=0):  # function to convert y-coordinate location to pixel in screen
    return int(((float(
        y_coord - max_y) / y_range) * reso_y) * -1 - shift)  # -1 is used since pixels are measured from top of screen (y = 0 at top)

def plot_rectangle(scr, color_code, reso, radius, coord_1, x0, y0, x_range, y_range, x_shift=0, y_shift=0):
    
    # Calculate the dimensions of the rectangle
    width = 40  # Adjust the width as needed
    height = 2 * radius  # Adjust the height as needed
    
    # Determine the top-left corner of the rectangle
    rect_x = coord_1[0]   # Adjust the x-coordinate as needed
    rect_y = coord_1[1]-height  # Adjust the y-coordinate as needed
    
    # Draw the rectangle on the screen
    pg.draw.rect(scr, color_code, (rect_x, rect_y, width, height))

def plot_text(scr, text_str, color_code, fontsize, reso, x, y, x0, y0, x_range, y_range, x_shift=0, y_shift=0):
    font_path = 'misc/Century Gothic.ttf'
    font = pg.font.Font(font_path, fontsize)  # set the font + font size
    text = font.render(text_str, 1,
                       color_code)  # draw text on surface, using found density string, 1 = smooth edges, 10-10-10 is black color
    textpos = text.get_rect()  # get rectangle of text string
    wp_map_x = c2m_x(x, x0, reso[0], x_range, x_shift)  # convert x-coordinates to map coordinates
    wp_map_y = c2m_y(y, y0, reso[1], y_range, y_shift)  # convert y-coordinates to map coordinates
    textpos.left = wp_map_x  # determine x-location of text string
    textpos.centery = wp_map_y  # determine y-location of text string
    scr.blit(text, textpos)


def map_initialization(reso = (600,600)):  # function to initialise mapf
    map_properties = dict()  # create dict to return all properties
    map_get_range(map_properties,reso)  # get info about the screen range properties

    ctypes.windll.user32.SetProcessDPIAware()  # line of code required to get correct resolution of screen
    true_res = (ctypes.windll.user32.GetSystemMetrics(0),
                ctypes.windll.user32.GetSystemMetrics(1))  # line of code required to get correct resolution of screen

    outer_reso = reso  # set resolution full interface screen
    inner_reso = reso # set resolution simulation screen within full interface

    os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (int(0.85 * true_res[0] - 0.85 * outer_reso[0]), int(
        0.5 * true_res[1] - 0.5 * outer_reso[1]))  # make sure maps comes in middle screen

    pg.init()  # initialise pygame
    pg.display.set_caption('ORCHESTRA Outputs')
    scr = pg.display.set_mode(outer_reso)
    scrrect = scr.get_rect()  # get rectangular area of the surface
    scr.fill(white)  # set background color

    map_properties['outer_reso'] = outer_reso  # store created information (resolution)
    map_properties['inner_reso'] = inner_reso  # resolution port layout
    map_properties['scr'] = scr  # background
    map_properties['scrrect'] = scrrect  # surface of background
    map_properties['horizontal_sep'] = horizontal_sep  # margin around screen

    return map_properties  # return all information


def map_get_range(map_properties,reso):

    min_x = reso[0]
    max_x = 0
    min_y = reso[1]
    max_y = 0

    x_range = max_x - min_x  # get difference between max and min x-coordinate
    y_range = max_y - min_y  # get difference between max and min y-coordinate

    if squared_display:  # in case a squared display has to be created, x + y have to be adapted
        if x_range > y_range:
            min_x -= int(boundary_margin * x_range)
            max_x += int(boundary_margin * x_range)
            diff_range = (max_x - min_x) - y_range
            min_y -= int(0.5 * diff_range)
            max_y += int(0.5 * diff_range)

        elif y_range > x_range:
            min_y -= int(boundary_margin * y_range)
            max_y += int(boundary_margin * y_range)
            diff_range = (max_y - min_y) - x_range
            min_x -= int(0.5 * diff_range)
            max_x += int(0.5 * diff_range)
        else:
            min_x -= int(boundary_margin * x_range)
            max_x += int(boundary_margin * x_range)
            min_y -= int(boundary_margin * y_range)
            max_y += int(boundary_margin * y_range)
    else:
        min_x -= int(boundary_margin * x_range)
        max_x += int(boundary_margin * x_range)
        min_y -= int(boundary_margin * y_range)
        max_y += int(boundary_margin * y_range)

    x_range = max_x - min_x  # get difference between max and min x-coordinate
    y_range = max_y - min_y  # get difference between max and min y-coordinate

    map_properties['min_x'] = min_x  # store all parameters
    map_properties['max_x'] = max_x
    map_properties['x_range'] = x_range
    map_properties['min_y'] = min_y
    map_properties['max_y'] = max_y
    map_properties['y_range'] = y_range


def map_running(map_properties, visualization, t,name):  # function to update the map
    """
    Function updates Pygame map based on the map_properties, current state of the vehicles and the time.
    Collissions are detected if two truck are at the same xy_position. HINT: Is a collision the only conflict?    
    If escape key is pressed, pygame closes.
    If "p" key is pressed, pygame pauses. If enter is pressed pygame continues.
    INPUT:
        - map_properties = dict with properties as created in map_intialization.
        - current_states = dict with id, heading and xy_pos of all active truck.
        - t = time.
    RETURNS:
        - Function updates pygame.
        - escape_pressed = boolean (True/False) = Used to end simulation loop if escape is pressed.
    """
    reso = map_properties['inner_reso']  # get resolution
    scr = map_properties['scr']  # get screen
    scrrect = map_properties['scrrect']  # get screen surface

    min_x = map_properties['min_x']  # get x0
    x_range = map_properties['x_range']  # get horizontal range
    max_y = map_properties['max_y']  # get y0 (measured from above)
    y_range = map_properties['y_range']  # get vertical range
    empty_surface = pg.Surface(scrrect.size)

    # You can fill the surface with a color if needed
    empty_surface.fill((255, 255, 255))
    scr.blit(empty_surface,scrrect)  # print layout on screen

    if visualization['graph']!= {}:
        scr.blit(visualization['graph'], (0, 350))     
            
    if disp_time:

      y_block = 0.05
      plot_text(scr,f"Instance {name}".zfill(2), black, 12, reso, min_x +0.37*x_range, max_y - y_block * y_range, min_x, max_y, x_range, y_range)
      # Terminal Performance Block
      y_ex = 0
      y_block = 0.1
      plot_text(scr,"Operational Planning".zfill(2), black, 30, reso, min_x +0.025*x_range, max_y - y_block * y_range, min_x, max_y, x_range, y_range)
      for key in visualization['operational'].keys():
        plot_text(scr, str(visualization['operational'][key]).zfill(2), black, 15, reso, min_x+0.025*x_range, max_y - (y_block+0.05+y_ex) * y_range, min_x, max_y, x_range, y_range)
        y_ex +=0.03


    # Terminal Announcements Block
    y_ex = 0
    y_block = 0.1
    plot_text(scr,"Tactical Planning".zfill(2), black, 30, reso, min_x +0.6*x_range, max_y - y_block * y_range, min_x, max_y, x_range, y_range)
    for key in visualization['tactical'].keys():
        plot_text(scr, str(visualization['tactical'][key]).zfill(2), black, 15, reso, min_x+0.6*x_range, max_y - (y_block+0.05+y_ex) * y_range, min_x, max_y, x_range, y_range)
        y_ex +=0.03

    # Terminal Announcements Block
    y_ex = 0
    y_block = 0.3
    plot_text(scr,"Disruptions".zfill(2), black, 30, reso, min_x +0.6*x_range, max_y - y_block * y_range, min_x, max_y, x_range, y_range)
    for key in visualization['disruption'].keys():
        plot_text(scr, str(visualization['disruption'][key]).zfill(2), black, 15, reso, min_x+0.6*x_range, max_y - (y_block+0.05+y_ex) * y_range, min_x, max_y, x_range, y_range)
        y_ex +=0.03

    pg.display.flip()  # Update the full display Surface to the screen
    try:
        file_path = f"data/instances/{name}/pics"
        create_folder_path(file_path)
        file_path = f"data/instances/{name}/pics/screen_image_{t}.png"
        pg.image.save(scr, file_path)
        pg.event.pump()  # internally process pygame event handlers
        keys = pg.key.get_pressed()  # get the state of all keyboard buttons

    except:
        pass
    
    if keys[pg.K_ESCAPE]:  # if the escape key is being pressed
        escape_pressed = True  # stop running
        print('Visualization aborted by escape key')
        pg.quit()
    else:
        escape_pressed = False

    if keys[pg.K_p]:
        input('Paused, press enter to continue')
    return escape_pressed

def get_instance_name(args,model):
    d = args.date.split('-')[-1]
    t = "U"

    if args.tactical:
        t = "T"

    f = "NF"
    if args.flexibility:
        f = "F"

    dis = "D"
    if model.disruptions[1]['perc_road'] == 0:
        dis  = "ND"

    instance_name = f'{d}_{t}_{f}_{dis}_{args.N_TRUCKS_TRAINS}_{args.N_TRUCKS_VESSELS}_{args.N_TRUCKS_CAVS}_{args.max_cavs}_{args.max_platoon_size}'
    model.args.instance_name = instance_name

import os

def create_folder_path(folder_path):
    try:
        os.makedirs(folder_path, exist_ok=True)
    except OSError as e:
        print(f"Error creating folder path '{folder_path}': {e}")