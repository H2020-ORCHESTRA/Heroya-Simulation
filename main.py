########################################################## Macrosimulation main file for Heroya Simulation ##########################################################

# Import Libraries
import pandas as pd
import openrouteservice
import time

# Custom Initializations - Matplotlib plotting style
import matplotlib.pyplot as plt
plt.style.use(['science',"no-latex"])
plt.rc('font',**{'family':'sans-serif','sans-serif':['Century Gothic']})

# Custom Initializations - Dont show warnings  
import warnings
warnings.filterwarnings("ignore")

# Custom Initializations - Hide Pygame message
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'

# Custom Initializations - Load agents
from actors.agents import *
from visualization import *

# Custom Initializations - Load Parameters from config.py
from config import get_config
import gc 

if __name__ == "__main__":

    # Clarify user inputs from GUI

    args = get_config()
    args.seed = 42

    # Open Router
    open_ORS()
    client = openrouteservice.Client(base_url='http://localhost:8080/ors')

    # Define simulation start time and total steps
    hours, minutes = map(int, args.start.split(':'))
    start = (hours * 60) + minutes
    steps = start + int(int(args.simulated_time)+180)
    args.start_stamp = start
    args.start_step = start
    
    # Read model
    model = Heroya(args,client)

    # Insert Disruptions
    disruptions = read_disruptions(args,client)
    model.disruptions = disruptions

    get_instance_name(args,model)
    model.args.instance_name = f"{model.args.instance_name}_{args.seed}"

    # Run Model
    st = time.time()
    model.run_model(step_count=steps)
    end = time.time()

    agent_results = model.datacollector.get_agent_vars_dataframe()
    agent_results = agent_results.reset_index(level=['Step','AgentID'])
    agent_results = agent_results[agent_results["Step"] == steps-1].drop(columns=['Step']).set_index("AgentID")
    agent_results.to_excel(f'data/instances/{model.args.instance_name}/report.xlsx')
    gc.collect()