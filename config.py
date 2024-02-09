# Loading configuration of optimization

# Importing Libraries
from gooey import Gooey, GooeyParser


@Gooey(program_name='ORCHESTRA Simulation',image_dir="misc",tabbed_groups=True,default_size=(900, 900),optional_cols=1,required_cols=1)
def get_config():

    # Creating configuration file
    parser = GooeyParser(description="Heroya Living Lab")

    group1 =  parser.add_argument_group('Simulation Parameters')

    group1.add_argument(
        "--start",
        metavar="Start Time",
        help="Select custom starting time of simulation in the HH:MM format.",
        default = "6:00"
    )

    group1.add_argument(
            "--date",
            metavar="Examined date",
            help="Select date to initiate the simulation from the dropdown menu.",
            choices=["2023-3-17" ,"2023-3-18" ,"2023-3-19" , "2023-3-20","2023-3-21" ,"2023-3-22" ,"2023-3-23","2023-3-24","2023-3-25",
                     "2023-3-27"],
            widget="Dropdown",
            default = "2023-3-22"
        )

    group2 = parser.add_argument_group('Optional Parameters', gooey_options={'columns':2})

    group2.add_argument(
            "--simulated_time",
            metavar="Simulated Time",
            help="Select simulated time for examined instance (minutes).",
            widget='Slider', gooey_options={
                'min': 30, 
                'max': 1410, 
                'increment': 30},
            default=660,
            type = int
        )

    group2.add_argument(
            "--window_len",
            metavar="Time Window",
            help="Select length of time-window for examined instance (minutes).",
            widget='Slider', gooey_options={
                'min': 15, 
                'max': 240, 
                'increment': 15},
            default=15,
            type = int
        )
    
    group2.add_argument(
        "--tactical",
        metavar="Deploy Block Scheduler",
        help = "Use Tactical Planning to shape arrivals prior to trip start.",
        action="store_true",
        default = False
        )

    group2.add_argument(
        "--flexibility",
        metavar="Simulate TSP flexibility",
        help = "Allow for Logistic Companies to have more flexibility in time-window selection.",
        action="store_true",
        default = False
        )


    group3 =  parser.add_argument_group('Mode Specific Parameters', gooey_options={'columns':3})

    group3.add_argument(
        "--N_VESSELS",
        metavar="Examined vessels. (NV)",
        help="Select number of vessels.",
        widget='Slider', gooey_options={
            'min': 0, 
            'max': 5, 
            'increment': 1},
        default=2,
        type = int
    )

    group3.add_argument(
        "--N_TRAINS",
        metavar="Examined trains. (NT)",
        help="Select number of trains",
        widget='Slider', gooey_options={
            'min': 0, 
            'max': 5, 
            'increment': 1},
        default=1,
        type = int
        
    )

    group3.add_argument(
        "--max_cavs",
        metavar="Available CAVs. (NC)",
        help="Select number of CAVs.",
        widget='Slider', gooey_options={
            'min': 0, 
            'max': 10, 
            'increment': 1},
        default=1,
        type = int
    )

    group3.add_argument(
        "--N_TRUCKS_VESSELS",
        metavar="Examined trucks related to vessels. (NT_v)",
        help="Select number of trucks.",
        widget='Slider', gooey_options={
            'min': 0, 
            'max': 300, 
            'increment': 1},
        default=60,
        type = int
    )


    group3.add_argument(
        "--N_TRUCKS_TRAINS",
        metavar="Examined trucks related to trains. (NT_t)",
        help="Select number of trucks.",
        widget='Slider', gooey_options={
            'min': 0, 
            'max': 300, 
            'increment': 1},
        default=40,
        type = int
    )


    group3.add_argument(
        "--N_TRUCKS_CAVS",
        metavar="Examined trucks related to CAVs. (NT_c)",
        help="Select number of trucks.",
        widget='Slider', gooey_options={
            'min': 0, 
            'max': 300, 
            'increment': 1},
        default=30,
        type = int
    )

    group4 =  parser.add_argument_group('Service Parameters', gooey_options={'columns':2})

    group4.add_argument(
        "--mu",
        metavar="Gate Processing rate. (P_g)",
        help="Select Average Processing rate of trucks by the gate per time-window.",
        widget='Slider', gooey_options={
            'min': 0, 
            'max': 50, 
            'increment': 1},
        default=30,
        type = int
    )


    group4.add_argument(
        "--max_platoon_size",
        metavar="Platoon Length. (PL)",
        help="Select the maximum number of trucks in a platoon.",
        widget='Slider', gooey_options={
            'min': 0, 
            'max': 10, 
            'increment': 1},
        default=1,
        type = int
    )


    group4.add_argument(
        "--c",
        metavar="Gates in the port.",
        help="Select number of gates to process trucks.",
        widget='Slider', gooey_options={
            'min': 0, 
            'max': 5, 
            'increment': 1},
        default=1,
        type = int
    )

    group4.add_argument(
        "--proc_trucks",
        metavar="Truck Processing rate. (P_t) ",
        help="Select Average Processing rate of trucks by terminal per time-window.",
        widget='Slider', gooey_options={
            'min': 0, 
            'max': 50, 
            'increment': 1},
        default=15,
        type = int
    )

    group3 =  parser.add_argument_group('Genetic Algorithm Parameters')

    group3.add_argument(
        "--max_hofs",
        metavar="Maximum suboptimal solutions per actor",
        help="Select the maximum number of suboptimal solutions per actor",
        widget='Slider', gooey_options={
            'min': 0, 
            'max': 100, 
            'increment': 1},
        default=10,
        type = int
    )

    group3.add_argument(
        "--N_GEN",
        metavar="N_GEN",
        help="N_GEN",
        widget='Slider', gooey_options={
            'min': 0, 
            'max': 10000, 
            'increment': 100},
        default=100,
        type = int
    )

    group3.add_argument(
        "--N_POP",
        metavar="N_POP",
        help="N_POP",
        widget='Slider', gooey_options={
            'min': 0, 
            'max': 10000, 
            'increment': 50},
        default=2000,
        type = int
    )

    group3.add_argument(
        "--CXPB", 
        widget='DecimalField',
        help='Enter a float value:',
        default=0.5,
        gooey_options={
            'min': 0.0,
            'max': 10.0,
            'increment': 0.1,
            'precision': 1  # You can set precision to control the number of decimal places
        },
        type = float)
    
    group3.add_argument(
        "--MUTPB", 
        widget='DecimalField',
        help='Enter a float value:',
        default=0.2,
        gooey_options={
            'min': 0.0,
            'max': 10.0,
            'increment': 0.1,
            'precision': 1  # You can set precision to control the number of decimal places
        },
        type = float)

    group8 = parser.add_argument_group('Disruptions - Road Network', gooey_options={'columns':1})

    group8.add_argument(
        "--avoid",
        metavar="Time-Windows to avoid ",
        default= [1,2,3]
    )
    group8.add_argument('--query-string8', help='the search string',gooey_options= {'visible': False})
    child_three = group8.add_argument_group('Delays on SS336', gooey_options={'show_border': True})        

    child_three.add_argument(
            "--speed_reductionE18",
            metavar="Speed reduction on E18 (0 blocked - 1 No Reduction)",
            help="",
            widget='DecimalField', gooey_options={
                'min': 0.0, 
                'max': 1.0, 
                'increment': 0.01},
            default=1.0,
        type = float)
        

    child_three.add_argument(
        "--disruption_time_roadE18",
        metavar="Disruption time period on E18",
        help="Insert time(s) in the HH:MM format\n(Followed by semicolons start and end)",
        default="8:00-10:00"
    )

    child_three.add_argument(
            "--speed_reductionRV32",
            metavar="Speed reduction on Rv32 (0 blocked - 1 No Reduction)",
            help="",
            widget='DecimalField', gooey_options={
                'min': 0.0, 
                'max': 1.0, 
                'increment': 0.01},
            default=1.0,
        type = float)
        

    child_three.add_argument(
        "--disruption_time_roadRV32",
        metavar="Disruption time period on Rv32",
        help="Insert time(s) in the HH:MM format\n(Followed by semicolons start and end)",
        default="10:00-11:00"
    )

    child_three.add_argument(
            "--speed_reductionRV40",
            metavar="Speed reduction on Rv40 (0 blocked - 1 No Reduction)",
            help="",
            widget='DecimalField', gooey_options={
                'min': 0.0, 
                'max': 1.0, 
                'increment': 0.01},
            default=1.0,
        type = float)
        

    child_three.add_argument(
        "--disruption_time_roadRV40",
        metavar="Disruption time period on Rv40",
        help="Insert time(s) in the HH:MM format\n(Followed by semicolons start and end)",
        default=""
    )

    return parser.parse_args()


